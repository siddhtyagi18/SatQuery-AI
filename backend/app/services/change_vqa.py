"""
backend/app/services/change_vqa.py
===================================
Real Vision-Language Change VQA & Interpretation Service.

This service connects the output of the bi-temporal change detection model
(SiameseUNet binary change mask + quantitative statistics) to the domain-adapted
Vision-Language Model (SmolVLM-500M-Instruct + LoRA checkpoint, or Cloud AI Gateway).

Pipeline Architecture:
----------------------
1. Image A (T1: Before) + Image B (T2: After)
2. Quantitative pixel stats from SiameseUNet (changed_pixel_pct, severity, threshold)
3. Tri-panel composite generation:
     [ Panel 1: T1 (Before) | Panel 2: T2 (After) | Panel 3: T2 + Red Change Overlay ]
   Dimensions: 3W x H, RGB format.
4. Change-conditioned prompt generation with anti-hallucination instructions.
5. Real VLM inference via VQAService (SmolVLM+LoRA / Cloud Gateway).
6. Evidence & provenance assembly separating detector facts from VLM interpretation.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from ..config import get_settings
from ..logging_setup import logger
from .change_detection import _ensure_results_dir

settings = get_settings()


@dataclass
class ChangeVQAResult:
    """Structured result returned by run_change_vqa."""
    answer: str
    confidence: Optional[float] = None  # Strictly None — never fabricate confidence
    evidence: List[str] = field(default_factory=list)
    tool_id: str = "change_vqa"
    is_mock: bool = False
    composite_url: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)


def validate_change_vqa_vlm_output(raw_text: Optional[str]) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate raw Vision-Language Model output for Change VQA.

    Enforces that the VLM produced genuine qualitative natural-language text
    and rejects degenerate outputs (e.g., lone coordinates, floats like "1.000000",
    empty strings, or ungrounded numeric tokens).

    Returns:
        (is_valid, cleaned_answer, rejection_reason)
    """
    if raw_text is None:
        return False, None, "VLM returned null/empty response"

    cleaned = raw_text.strip()
    if not cleaned:
        return False, cleaned, "Empty or whitespace-only output"

    # 1. Pure floating point or integer number (e.g. "1.000000", "0.500000", "1", "42")
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", cleaned):
        return False, cleaned, f"Degenerate isolated numeric token ({cleaned})"

    # 2. Coordinate-like strings (e.g. "[0.0 0.0, 1.0 1.0]", "<point>(0.5, 0.5)</point>", "(0.1, 0.2)")
    if (
        re.fullmatch(r"\[\s*[-+]?\d+(?:\.\d+)?(?:\s+[-+]?\d+(?:\.\d+)?)*(?:,\s*[-+]?\d+(?:\.\d+)?(?:\s+[-+]?\d+(?:\.\d+)?)*)*\s*\]", cleaned)
        or "<point>" in cleaned
        or re.fullmatch(r"\(\s*[-+]?\d+(?:\.\d+)?\s*,\s*[-+]?\d+(?:\.\d+)?\s*\)", cleaned)
    ):
        return False, cleaned, f"Degenerate coordinate-like token string ({cleaned})"

    # 3. Single-character or too short
    if len(cleaned) < 2:
        return False, cleaned, "Single-character output"

    # 4. Output containing no alphabetic words of at least 2 letters
    words = re.findall(r"[A-Za-z]{2,}", cleaned)
    if not words:
        return False, cleaned, f"No natural-language words found in output ({cleaned})"

    return True, cleaned, None


# ---------------------------------------------------------------------------
# 1. Subtle Contour Overlay & Composite Generator
# ---------------------------------------------------------------------------

def _to_pil_rgb(image_input: Union[Image.Image, Path, str, np.ndarray]) -> Image.Image:
    """Ensure input is loaded as a PIL RGB Image."""
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.exists():
            raise FileNotFoundError(f"Image file not found: {p}")
        return Image.open(p).convert("RGB")
    if isinstance(image_input, np.ndarray):
        if image_input.dtype != np.uint8:
            arr = np.clip(image_input * 255 if image_input.max() <= 1.0 else image_input, 0, 255).astype(np.uint8)
        else:
            arr = image_input
        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L").convert("RGB")
        return Image.fromarray(arr).convert("RGB")
    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def _build_subtle_contour_overlay(
    mask_binary: np.ndarray,
    target_size: Tuple[int, int],
    contour_color: Tuple[int, int, int, int] = (255, 60, 60, 220),
    fill_color: Tuple[int, int, int, int] = (255, 80, 80, 40),
    thickness: int = 1,
) -> Image.Image:
    """
    Construct a subtle contour overlay that highlights detected change regions
    as spatial guides without heavily tinting underlying image pixels.

    - Starts with transparent RGBA canvas.
    - Low-opacity warm fill (~15% opacity) to subtly mark region without obliterating terrain.
    - Sharp 1px contour around connected change components.
    """
    w, h = target_size
    if mask_binary.shape != (h, w):
        mask_pil = Image.fromarray(mask_binary.astype(np.uint8)).resize((w, h), Image.NEAREST)
        mask_u8 = (np.asarray(mask_pil) > 0).astype(np.uint8)
    else:
        mask_u8 = (mask_binary > 0).astype(np.uint8)

    overlay_np = np.zeros((h, w, 4), dtype=np.uint8)

    # 1. Very low-opacity subtle fill (~15% alpha)
    if fill_color[3] > 0:
        overlay_np[mask_u8 == 1] = fill_color

    # 2. Thin contour extraction
    try:
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(overlay_np, contours, -1, contour_color, thickness=thickness)
    except Exception as err:
        logger.debug(f"[ChangeVQA] cv2 contour extraction fallback: {err}")
        kernel = np.ones((3, 3), dtype=np.uint8)
        dilated = cv2.dilate(mask_u8, kernel, iterations=1)
        edge = (dilated - mask_u8) > 0
        overlay_np[edge] = contour_color

    return Image.fromarray(overlay_np, mode="RGBA")


def create_change_composite(
    img_a: Union[Image.Image, Path, str, np.ndarray],
    img_b: Union[Image.Image, Path, str, np.ndarray],
    change_mask: Union[np.ndarray, Image.Image, Path, str, None] = None,
    two_panel: bool = False,
) -> Image.Image:
    """
    Synthesize a bi-temporal comparison strip for visual reasoning.

    Panel Layout (two_panel=False, default):
    ----------------------------------------
    [ Panel 1: Before (T1) | Panel 2: After (T2) | Panel 3: T2 + Subtle Change Contour ]
    Dimensions: (3 * W) x H, RGB format.

    Panel Layout (two_panel=True):
    ------------------------------
    [ Panel 1: Before (T1) | Panel 2: After (T2) ]
    Dimensions: (2 * W) x H, RGB format.

    Parameters
    ----------
    img_a       : T1 (Before) image.
    img_b       : T2 (After) image.
    change_mask : Optional binary change mask (H x W uint8 {0, 1}) or RGBA overlay.
    two_panel   : If True, omit panel 3 and generate a direct 2-panel T1|T2 strip.

    Returns
    -------
    PIL.Image.Image: (3W x H) or (2W x H) RGB image.
    """
    pil_a = _to_pil_rgb(img_a)
    pil_b = _to_pil_rgb(img_b)

    # Align dimensions to img_a if sizes differ
    if pil_a.size != pil_b.size:
        logger.warning(
            f"[ChangeVQA] Resizing img_b {pil_b.size} to match img_a {pil_a.size}"
        )
        pil_b = pil_b.resize(pil_a.size, Image.LANCZOS)

    w, h = pil_a.size

    if two_panel:
        composite = Image.new("RGB", (2 * w, h), color=(0, 0, 0))
        composite.paste(pil_a, (0, 0))
        composite.paste(pil_b, (w, 0))
        return composite

    # Build Panel 3: After (T2) with subtle contour change overlay
    panel3_rgb: Image.Image
    if change_mask is not None:
        if isinstance(change_mask, (str, Path)):
            mask_p = Path(change_mask)
            if mask_p.exists():
                loaded_img = Image.open(mask_p)
                if loaded_img.size != (w, h):
                    loaded_img = loaded_img.resize((w, h), Image.NEAREST)
                if loaded_img.mode == "RGBA":
                    mask_np = np.asarray(loaded_img)
                    binary_mask = (mask_np[:, :, 3] > 20).astype(np.uint8)
                else:
                    gray = loaded_img.convert("L")
                    mask_np = np.asarray(gray)
                    binary_mask = (mask_np > 0).astype(np.uint8)
                overlay = _build_subtle_contour_overlay(binary_mask, (w, h))
                panel3 = Image.alpha_composite(pil_b.convert("RGBA"), overlay)
                panel3_rgb = panel3.convert("RGB")
            else:
                panel3_rgb = pil_b.copy()
        elif isinstance(change_mask, Image.Image):
            loaded_img = change_mask
            if loaded_img.size != (w, h):
                loaded_img = loaded_img.resize((w, h), Image.NEAREST)
            if loaded_img.mode == "RGBA":
                mask_np = np.asarray(loaded_img)
                binary_mask = (mask_np[:, :, 3] > 20).astype(np.uint8)
            else:
                gray = loaded_img.convert("L")
                mask_np = np.asarray(gray)
                binary_mask = (mask_np > 0).astype(np.uint8)
            overlay = _build_subtle_contour_overlay(binary_mask, (w, h))
            panel3 = Image.alpha_composite(pil_b.convert("RGBA"), overlay)
            panel3_rgb = panel3.convert("RGB")
        elif isinstance(change_mask, np.ndarray):
            mask_arr = change_mask
            if mask_arr.ndim != 2:
                raise ValueError(
                    f"Change mask must be 2-dimensional (H, W); got shape {mask_arr.shape}"
                )
            binary_mask = (mask_arr > 0).astype(np.uint8)
            overlay = _build_subtle_contour_overlay(binary_mask, (w, h))
            panel3 = Image.alpha_composite(pil_b.convert("RGBA"), overlay)
            panel3_rgb = panel3.convert("RGB")
        else:
            raise TypeError(f"Unsupported change_mask type: {type(change_mask)}")
    else:
        panel3_rgb = pil_b.copy()

    # Assemble 3-panel horizontal composite
    composite = Image.new("RGB", (3 * w, h), color=(0, 0, 0))
    composite.paste(pil_a, (0, 0))
    composite.paste(pil_b, (w, 0))
    composite.paste(panel3_rgb, (2 * w, 0))

    return composite


# ---------------------------------------------------------------------------
# 2. Change-Aware Prompt Builder (Telemetry Excluded from VLM Context)
# ---------------------------------------------------------------------------

def build_change_vqa_prompt(
    query: str,
    changed_pixel_pct: Optional[float] = None,
    severity: Optional[str] = None,
    threshold: Optional[float] = None,
    date_a: Optional[str] = None,
    date_b: Optional[str] = None,
    use_three_panel: bool = False,
    two_panel: bool = True,
) -> str:
    """
    Construct a concise, domain-grounded prompt instructing the VLM to analyze
    the satellite comparison strip.

    Step 9B/16C Design:
    -------------------
    - Default reasoning image is 2-PANEL (T1: Before | T2: After).
    - Explicitly identifies LEFT as Before/T1 and RIGHT as After/T2.
    - Instructs VLM to compare images directly and describe visually supported changes.
    - Requires 1-2 complete natural-language sentences using remote-sensing terminology.
    - Strictly prohibits isolated coordinates, numbers, JSON, fabricated dates, or percentages.
    - Quantitative detector telemetry is strictly EXCLUDED from the VLM prompt to enforce visual reasoning.
    """
    is_three_panel = use_three_panel or (not two_panel and use_three_panel)

    if is_three_panel:
        prompt_parts = [
            "You are analyzing a bi-temporal satellite image comparison strip containing three horizontal panels.",
            "- Left panel = BEFORE (T1) earlier acquisition.",
            "- Center panel = AFTER (T2) later acquisition.",
            "- Right panel = T2 with a subtle outline showing pixels detected as changed.",
            "",
            "Instructions:",
            "- Compare the Left (Before / T1) and Center (After / T2) panels directly to identify physical changes, using the Right panel only to locate candidate difference regions.",
            "- Use the RIGHT panel only to locate where the change detector identified differences.",
            "- Produce 1 to 2 complete natural-language sentences describing only visually supported qualitative changes using remote-sensing terminology (e.g. new structures, vegetation alteration, road development).",
            "- Do not describe image annotations, colors, masks, overlays, or graphics as physical objects.",
            "- Do not invent exact counts, measurements, percentages, coordinates, dates, or confidence scores.",
            "- Do not output coordinate tokens, JSON, or isolated numeric values.",
        ]
        if date_a or date_b:
            prompt_parts.append(f"Temporal baseline: {date_a or 'T1'} to {date_b or 'T2'}.")
        prompt_parts.extend([
            "",
            f"User Question: {query.strip()}",
            "",
            "Qualitative Scene Change Description:",
        ])
        return "\n".join(prompt_parts)

    # DEFAULT: 2-Panel Concise Prompt
    prompt_parts = [
        "You are analyzing a bi-temporal satellite image comparison strip containing two temporal acquisitions of the same location.",
        "The left satellite image (Before / T1) is the earlier acquisition. The right satellite image (After / T2) is the later acquisition.",
        "Compare the left satellite image (Before / T1) with the right satellite image (After / T2). "
        "Identify only physical changes visible between the two images.",
        "",
        "Instructions:",
        "- Produce 1 to 2 complete natural-language sentences describing only visually supported qualitative changes using remote-sensing terminology (e.g. new building construction, surface clearing, vegetation loss, road expansion).",
        "- Do not describe image annotations, colors, masks, overlays, or graphics as physical objects.",
        "- Do not invent exact counts, measurements, or object identities unless clearly visible.",
        "- Do not invent exact counts, measurements, percentages, coordinates, dates, or confidence scores.",
        "- Do not output coordinate tokens, JSON, or isolated numeric values.",
    ]
    if date_a or date_b:
        prompt_parts.append(f"Temporal baseline: {date_a or 'T1'} to {date_b or 'T2'}.")

    prompt_parts.extend([
        "",
        f"User Question: {query.strip()}",
        "",
        "Qualitative Scene Change Description:",
    ])
    return "\n".join(prompt_parts)


# ---------------------------------------------------------------------------
# 3. Change VQA Execution Engine (Dual-Artifact Architecture)
# ---------------------------------------------------------------------------

def run_change_vqa(
    img_a_path: Path,
    img_b_path: Path,
    query: str,
    change_mask: Union[np.ndarray, Image.Image, Path, str, None] = None,
    change_stats: Optional[Dict[str, Any]] = None,
    analysis_id: str = "change_vqa",
    preferred_provider: Optional[str] = None,
    use_three_panel_reasoning: bool = False,
    two_panel: bool = True,
) -> ChangeVQAResult:
    """
    Execute real Change VQA over bi-temporal satellite imagery.

    Step 9B Dual-Artifact Architecture:
    ----------------------------------
    1. REASONING IMAGE (passed to VLM):
       2-panel composite [ T1 (Before) | T2 (After) ], dimensions: 2W x H.
       Provides clean temporal contrast without visual clutter or mask artifacts.
    2. EVIDENCE IMAGE (saved for UI / analyst evidence):
       3-panel composite [ T1 (Before) | T2 (After) | T2 + Subtle Change Contour ], dimensions: 3W x H.
       Provides spatial grounding and visual change localization.
    3. PROMPT GENERATION:
       Concise, domain-grounded prompt without detector telemetry leakage.
    4. VLM INFERENCE:
       Executes real SmolVLM-500M (+ LoRA) on the 2-panel reasoning image.
    5. RESPONSE FORMATTING:
       Preserves exact raw VLM answer and appends quantitative detector telemetry post-generation.
    """
    t0 = time.perf_counter()
    stats = change_stats or {}
    changed_pct = stats.get("changed_pixel_pct")
    severity = stats.get("severity", "unknown")
    thresh_used = stats.get("threshold_used", 0.70)
    date_a = stats.get("date_a")
    date_b = stats.get("date_b")

    # If change_mask is not directly provided, check stats for mask_path or overlay_url
    effective_mask = change_mask
    if effective_mask is None:
        if "mask_path" in stats and Path(stats["mask_path"]).exists():
            effective_mask = Path(stats["mask_path"])
        elif "overlay_url" in stats:
            url_name = Path(stats["overlay_url"]).name
            candidate_p = _ensure_results_dir() / url_name
            if candidate_p.exists():
                effective_mask = candidate_p

    # 1. Synthesize both reasoning and evidence composites
    try:
        # Reasoning image: 2-panel direct comparison (T1 | T2)
        reasoning_img = create_change_composite(
            img_a=img_a_path,
            img_b=img_b_path,
            two_panel=True,
        )
        # Evidence image: 3-panel subtle contour comparison (T1 | T2 | T2+Contour)
        evidence_img = create_change_composite(
            img_a=img_a_path,
            img_b=img_b_path,
            change_mask=effective_mask,
            two_panel=False,
        )
    except Exception as comp_err:
        logger.error(f"[ChangeVQA] Failed to create change composite: {comp_err}")
        return ChangeVQAResult(
            answer=(
                f"[Change-VQA — Composite generation failed] {comp_err}. "
                f"Quantitative detector reported {changed_pct:.1f}% change (severity: {severity})."
            ) if changed_pct is not None else f"[Change-VQA Error] Composite generation failed: {comp_err}",
            confidence=None,
            evidence=[f"Change composite creation failed: {comp_err}"],
            tool_id="change_vqa",
            is_mock=True,
            stats={**stats, "execution_mode": "mock", "raw_vlm_answer": None},
        )

    # 2. Save both image artifacts to disk
    results_dir = _ensure_results_dir()

    # 2a. Evidence image (3-panel) -> UI / visual evidence display
    comp_filename = f"change_vqa_composite_{analysis_id}.png"
    comp_path = results_dir / comp_filename
    evidence_url = None
    try:
        evidence_img.save(comp_path, format="PNG", optimize=False)
        evidence_url = f"/api/results/{comp_filename}"
    except Exception as save_err:
        logger.warning(f"[ChangeVQA] Failed to save evidence composite PNG: {save_err}")

    # 2b. Reasoning image (2-panel) -> sent to VLM
    reasoning_filename = f"change_vqa_reasoning_{analysis_id}.png"
    reasoning_path = results_dir / reasoning_filename
    reasoning_url = None
    try:
        reasoning_img.save(reasoning_path, format="PNG", optimize=False)
        reasoning_url = f"/api/results/{reasoning_filename}"
    except Exception as save_err:
        logger.warning(f"[ChangeVQA] Failed to save reasoning composite PNG: {save_err}")

    # Determine which image is passed to the VLM
    if use_three_panel_reasoning:
        vlm_image_path = comp_path
        vlm_image_name = "3-panel (T1 | T2 | T2+Contour)"
        active_reasoning_dims = [evidence_img.width, evidence_img.height]
    else:
        vlm_image_path = reasoning_path
        vlm_image_name = "2-panel (T1 | T2)"
        active_reasoning_dims = [reasoning_img.width, reasoning_img.height]

    # 3. Build concise prompt (strictly WITHOUT telemetry)
    vlm_prompt = build_change_vqa_prompt(
        query=query,
        date_a=date_a,
        date_b=date_b,
        use_three_panel=use_three_panel_reasoning,
    )

    # 4. Delegate to VQAService
    from .vqa_service import get_vqa_service
    vqa_service = get_vqa_service()

    try:
        vqa_res = vqa_service.run_real_or_fallback(
            query=vlm_prompt,
            mode="single_image",
            image_file_paths=[vlm_image_path],
            tasks=["vqa"],
            tool_id="change_vqa",
            preferred_provider=preferred_provider,
        )

        if vqa_res.is_mock:
            # VLM failed or fell back to mock inside vqa_service
            logger.warning("[ChangeVQA] VQAService returned mock fallback result.")
            fallback_ans = (
                f"[Change-VQA — VLM reasoning unavailable; showing quantitative detector statistics]\n\n"
                f"The Siamese U-Net change detector measured **{changed_pct:.2f}%** of the scene area as changed "
                f"(severity: **{severity}**, threshold: `{thresh_used:.2f}`).\n\n"
                f"Natural-language visual reasoning was unavailable for this request."
            ) if changed_pct is not None else vqa_res.answer

            return ChangeVQAResult(
                answer=fallback_ans,
                confidence=None,
                evidence=vqa_res.evidence + ["[ChangeVQA] Execution fell back to quantitative mock summary."],
                tool_id="change_vqa",
                is_mock=True,
                composite_url=evidence_url,
                stats={
                    **stats,
                    "execution_mode": "mock",
                    "composite_url": evidence_url,
                    "reasoning_url": reasoning_url,
                    "reasoning_image_dimensions": active_reasoning_dims,
                    "evidence_image_dimensions": [evidence_img.width, evidence_img.height],
                    "reasoning_image_passed_to_vlm": vlm_image_name,
                    "raw_vlm_answer": None,
                },
            )

        # 5. Format successful real VLM output, preserving raw VLM generation and validating quality
        elapsed_sec = time.perf_counter() - t0
        raw_vlm_answer = vqa_res.answer.strip()
        is_valid_vlm, cleaned_vlm, rejection_reason = validate_change_vqa_vlm_output(raw_vlm_answer)

        if is_valid_vlm:
            vlm_section = (
                f"**Qualitative Visual Interpretation (VLM):**\n"
                f"{cleaned_vlm}\n\n"
                f"**VLM Interpretation Validation:** `ACCEPTED` (Natural-language visual reasoning validated)"
            )
        else:
            vlm_section = (
                f"**Qualitative Visual Interpretation (VLM):**\n"
                f"[VLM interpretation unavailable: Output was rejected by validation as degenerate ({rejection_reason}). Raw output: `{raw_vlm_answer}`]\n\n"
                f"**VLM Interpretation Validation:** `REJECTED` ({rejection_reason})"
            )

        formatted_answer = (
            f"### Bi-Temporal Scene Change Interpretation\n\n"
            f"{vlm_section}\n\n"
            f"**Quantitative Detection Telemetry:**\n"
            f"- **Detected Changed Area:** `{changed_pct:.2f}%` (Severity: **{severity}**)\n"
            f"- **Change Detection Threshold:** `{thresh_used:.2f}` (Siamese U-Net)\n"
            f"- **Detector Model:** SiameseUNet (~490K parameters, LEVIR-CD trained checkpoint)\n"
            f"- **Inference Provenance:** Vision-Language Model ({vqa_res.run_context.model_id if vqa_res.run_context else 'local:SmolVLM'})"
        ) if changed_pct is not None else f"### Bi-Temporal Scene Change Interpretation\n\n{vlm_section}"

        # Separate Change Detector evidence from VLM evidence
        evidence: List[str] = [
            f"[Change Detector] Model: SiameseUNet (~490K params, LEVIR-CD trained checkpoint).",
            f"[Change Detector] Measured change area: {changed_pct:.2f}% (severity: {severity}, threshold: {thresh_used:.2f})."
            if changed_pct is not None else "[Change Detector] Quantitative stats provided.",
            f"[Change Visualizer] Reasoning image passed to VLM: {vlm_image_name} ({active_reasoning_dims[0]}x{active_reasoning_dims[1]}px).",
            f"[Change Visualizer] Evidence visualization synthesized: 3-panel contour strip ({evidence_img.width}x{evidence_img.height}px).",
        ]
        if vqa_res.evidence:
            for ev in vqa_res.evidence:
                if not any(k in ev.lower() for k in ("output shape", "pillow backend")):
                    evidence.append(f"[VLM Reasoning] {ev}")

        if is_valid_vlm:
            evidence.append("[VLM Validation] Output passed natural-language quality validation.")
        else:
            evidence.append(f"[VLM Validation] Output REJECTED as degenerate: {rejection_reason} (raw: {raw_vlm_answer!r}).")

        evidence.append(
            "[Integrity] The change mask was generated by SiameseUNet; qualitative interpretation was processed by domain-adapted VLM."
        )
        evidence.append("Model does not emit a calibrated confidence score; confidence=null.")

        return ChangeVQAResult(
            answer=formatted_answer,
            confidence=None,  # Never fabricate confidence
            evidence=evidence,
            tool_id="change_vqa",
            is_mock=False,
            composite_url=evidence_url,
            stats={
                **stats,
                "execution_mode": "real",
                "composite_url": evidence_url,
                "reasoning_url": reasoning_url,
                "reasoning_image_dimensions": active_reasoning_dims,
                "evidence_image_dimensions": [evidence_img.width, evidence_img.height],
                "reasoning_image_passed_to_vlm": vlm_image_name,
                "raw_vlm_answer": raw_vlm_answer,
                "vlm_validation": "accepted" if is_valid_vlm else "rejected",
                "vlm_rejection_reason": rejection_reason,
                "processing_time_sec": round(elapsed_sec, 2),
            },
        )

    except Exception as exc:
        logger.exception(f"[ChangeVQA] Real Change VQA execution failed: {exc}")
        fallback_ans = (
            f"[Change-VQA — VLM inference error; showing quantitative detector statistics]\n\n"
            f"The Siamese U-Net change detector measured **{changed_pct:.2f}%** of the scene area as changed "
            f"(severity: **{severity}**, threshold: `{thresh_used:.2f}`).\n\n"
            f"Natural-language visual reasoning encountered an error: {type(exc).__name__}: {exc}."
        ) if changed_pct is not None else f"[Change-VQA Error] VLM reasoning failed: {exc}"

        return ChangeVQAResult(
            answer=fallback_ans,
            confidence=None,
            evidence=[
                f"[ChangeVQA Error] {type(exc).__name__}: {exc}",
                "[ChangeVQA Fallback] Quantitative change detector statistics preserved.",
                "Model does not emit a calibrated confidence score; confidence=null.",
            ],
            tool_id="change_vqa",
            is_mock=True,
            composite_url=evidence_url,
            stats={
                **stats,
                "execution_mode": "mock",
                "composite_url": evidence_url,
                "reasoning_url": reasoning_url,
                "raw_vlm_answer": None,
            },
        )
