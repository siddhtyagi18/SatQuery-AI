"""
backend/app/services/change_detection.py
-----------------------------------------
CPU-only bi-temporal change detection using Pillow + NumPy.
No GPU, no ML model downloads, no training.

Algorithm
---------
1. Open before and after images with Pillow; convert both to RGB.
2. Resize the after image to match before's dimensions if sizes differ.
3. Convert both to grayscale NumPy arrays (float32, [0, 1]).
4. Compute absolute per-pixel difference.
5. Apply a configurable threshold to produce a binary change mask.
6. Clean noise via simple 3x3 erosion -> dilation (NumPy only, no scipy).
7. Save a false-colour RGBA PNG overlay (changed = semi-transparent red).
8. Return structured result compatible with the AnalysisResult schema.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..logging_setup import logger

# ---------------------------------------------------------------------------
# Output dir — lives under backend/data/results/
# ---------------------------------------------------------------------------
_RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "results"


def _ensure_results_dir() -> Path:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return _RESULTS_DIR


# ---------------------------------------------------------------------------
# Dataclass for structured return value
# ---------------------------------------------------------------------------
@dataclass
class ChangeDetectionResult:
    answer: str
    confidence: None  # Always None — not a calibrated model score
    change_map: Dict[str, Any]
    evidence: List[str]
    stats: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _open_rgb(path: Path) -> Image.Image:
    """Open any Pillow-readable image as RGB."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _align_dimensions(
    before: Image.Image, after: Image.Image
) -> Tuple[Image.Image, Image.Image, bool]:
    """
    If the images have different sizes, resize 'after' to match 'before'.
    Returns (before, after, size_mismatch_occurred).
    """
    if before.size == after.size:
        return before, after, False
    logger.warning(
        "Image size mismatch: before=%s after=%s. "
        "Resizing 'after' to match 'before'.",
        before.size,
        after.size,
    )
    after = after.resize(before.size, Image.LANCZOS)
    return before, after, True


def _to_gray_array(img: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to a float32 grayscale array in [0, 1]."""
    gray = img.convert("L")
    return np.asarray(gray, dtype=np.float32) / 255.0


def _binary_erode(mask: np.ndarray, k: int = 3) -> np.ndarray:
    """Simple 2-D binary erosion with a k x k flat structuring element (NumPy only)."""
    pad = k // 2
    padded = np.pad(mask, pad, mode="constant", constant_values=0)
    h, w = mask.shape
    result = np.ones((h, w), dtype=np.uint8)
    for dy in range(k):
        for dx in range(k):
            result &= padded[dy : dy + h, dx : dx + w].astype(np.uint8)
    return result


def _binary_dilate(mask: np.ndarray, k: int = 3) -> np.ndarray:
    """Simple 2-D binary dilation with a k x k flat structuring element (NumPy only)."""
    pad = k // 2
    padded = np.pad(mask, pad, mode="constant", constant_values=0)
    h, w = mask.shape
    result = np.zeros((h, w), dtype=np.uint8)
    for dy in range(k):
        for dx in range(k):
            result |= padded[dy : dy + h, dx : dx + w].astype(np.uint8)
    return result


def _clean_noise(mask: np.ndarray) -> np.ndarray:
    """
    Morphological opening (erode then dilate) to remove isolated noise pixels.
    Uses a 3x3 structuring element — preserves blobs of 3+ connected pixels.
    """
    eroded = _binary_erode(mask, k=3)
    return _binary_dilate(eroded, k=3)


def _severity_label(changed_pct: float) -> str:
    """
    Return an interpretive severity label based on the changed area percentage.

    Thresholds are heuristic and intended for human interpretation only.
    They do NOT correspond to any validated remote sensing standard.

    - low      < 5 %   — minor local change, could be noise/illumination
    - moderate 5–25 %  — significant localised change visible in image
    - high     > 25 %  — large-scale scene change across most of image area
    """
    if changed_pct < 5.0:
        return "low"
    elif changed_pct <= 25.0:
        return "moderate"
    else:
        return "high"


def _build_overlay(change_mask: np.ndarray, size_wh: Tuple[int, int]) -> Image.Image:
    """
    Build an RGBA false-colour overlay:
      Changed pixels   -> semi-transparent red (255, 60, 60, 180)
      Unchanged pixels -> fully transparent    (0, 0, 0, 0)
    """
    w, h = size_wh
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    changed = change_mask == 1
    rgba[changed, 0] = 255   # R
    rgba[changed, 1] = 60    # G
    rgba[changed, 2] = 60    # B
    rgba[changed, 3] = 180   # A — semi-transparent
    return Image.fromarray(rgba, mode="RGBA")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_cpu_change_detection(
    before_path: Path,
    after_path: Path,
    analysis_id: str,
    threshold: float = 35.0 / 255.0,  # ~0.137; configurable per call
) -> ChangeDetectionResult:
    """
    Perform classical pixel-difference change detection between two images.

    Parameters
    ----------
    before_path  : Path to the 'before' (T1) image.
    after_path   : Path to the 'after' (T2) image.
    analysis_id  : Used to name the output PNG uniquely.
    threshold    : Normalised [0, 1] absolute-difference threshold.

    Returns
    -------
    ChangeDetectionResult with answer, null confidence, change_map dict,
    evidence list, and pixel statistics.
    """
    t0 = time.perf_counter()
    logger.info(
        "[change_detection] Starting CPU change detection analysis=%s threshold=%.4f",
        analysis_id,
        threshold,
    )

    # 1. Load ----------------------------------------------------------------
    before_img = _open_rgb(before_path)
    after_img = _open_rgb(after_path)
    orig_before_size = before_img.size  # (W, H)
    orig_after_size = after_img.size

    # 2. Align ---------------------------------------------------------------
    before_img, after_img, size_mismatch = _align_dimensions(before_img, after_img)

    # 3. Grayscale arrays ----------------------------------------------------
    before_gray = _to_gray_array(before_img)
    after_gray = _to_gray_array(after_img)
    h, w = before_gray.shape

    # 4. Difference + threshold ----------------------------------------------
    diff = np.abs(after_gray - before_gray)
    raw_mask = (diff > threshold).astype(np.uint8)

    # 5. Noise cleanup -------------------------------------------------------
    clean_mask = _clean_noise(raw_mask)

    # 6. Statistics ----------------------------------------------------------
    total_pixels = h * w
    changed_pixels = int(clean_mask.sum())
    changed_pct = round(changed_pixels / total_pixels * 100, 2) if total_pixels > 0 else 0.0
    unchanged_pct = round(100.0 - changed_pct, 2)
    severity = _severity_label(changed_pct)

    # 7. Save overlay PNG ----------------------------------------------------
    out_dir = _ensure_results_dir()
    mask_filename = f"{analysis_id}_changemap.png"
    mask_path = out_dir / mask_filename
    overlay = _build_overlay(clean_mask, (w, h))
    overlay.save(mask_path, format="PNG", optimize=False)
    overlay_url = f"/api/results/{mask_filename}"

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        "[change_detection] Done in %dms — changed=%.2f%% unchanged=%.2f%% severity=%s mask=%s",
        elapsed_ms,
        changed_pct,
        unchanged_pct,
        severity,
        mask_path,
    )

    # 8. Build result --------------------------------------------------------
    threshold_raw = int(threshold * 255)

    size_note = (
        f" The 'after' image ({orig_after_size[0]}×{orig_after_size[1]} px) was "
        f"resampled to match the 'before' image ({orig_before_size[0]}×{orig_before_size[1]} px) "
        f"using LANCZOS interpolation before comparison."
        if size_mismatch
        else ""
    )

    severity_words = {"low": "minor", "moderate": "significant", "high": "extensive"}
    severity_word = severity_words[severity]

    answer = (
        f"## Change Detection Report\n\n"
        f"**Pixel-level visual change was detected across {changed_pct:.1f}% of the comparable "
        f"image area** ({changed_pixels:,} of {total_pixels:,} pixels), indicating {severity_word} "
        f"scene change at the pixel level (severity: **{severity}**).\n\n"
        f"The remaining {unchanged_pct:.1f}% of pixels showed no detectable visual difference "
        f"above the configured sensitivity threshold ({threshold_raw}/255 in grayscale intensity).\n\n"
        f"**Interpretation guidance:** This result reflects pixel-level visual intensity differences "
        f"between the two image acquisitions. It does NOT identify or classify land-use change, "
        f"vegetation loss, flood extent, or any semantic category. Illumination variation, "
        f"atmospheric conditions, seasonal effects, sensor noise, or image compression can all "
        f"produce pixel-level differences that are not related to actual ground change.\n\n"
        f"**Recommended next action:** For official reporting or decision support, validate this "
        f"result using georeferenced image pairs with known acquisition metadata, or apply a "
        f"trained semantic change-detection model calibrated to the specific sensor and region.\n\n"
        f"*Analysis performed by CPU-only classical image processing (absolute grayscale pixel "
        f"difference + 3×3 morphological opening). No GPU, no trained model, no internet access "
        f"was required.{size_note}*"
    )

    evidence = [
        f"Algorithm: absolute grayscale pixel difference (threshold={threshold:.4f}, i.e. {threshold_raw}/255).",
        f"Noise cleanup: 3×3 morphological opening (erosion then dilation) using NumPy — "
        f"removes isolated changed pixels smaller than ~3×3 px.",
        f"Changed pixels (post noise-cleanup): {changed_pixels:,} / {total_pixels:,} ({changed_pct:.2f}%).",
        f"Unchanged pixels: {total_pixels - changed_pixels:,} / {total_pixels:,} ({unchanged_pct:.2f}%).",
        f"Severity label: {severity} (heuristic: low <5%, moderate 5–25%, high >25%).",
        f"Reference image dimensions: {w}×{h} px (before-image used as spatial reference).",
        f"Change mask overlay saved to: {overlay_url}",
        "CPU-only processing — no GPU, no trained ML model, no internet access required.",
        (
            "Confidence is null: classical pixel-difference algorithms produce no calibrated "
            "probability score. This is an honest representation of the algorithm's output, "
            "not an indicator of low quality."
        ),
    ]
    if size_mismatch:
        evidence.append(
            f"Dimension mismatch corrected: 'after' image resampled from "
            f"{orig_after_size[0]}×{orig_after_size[1]} px to "
            f"{orig_before_size[0]}×{orig_before_size[1]} px (LANCZOS). "
            "For higher accuracy, use geometrically co-registered image pairs."
        )

    change_map = {
        "overlayUrl": overlay_url,
        "legend": [
            {"label": f"Changed region ({changed_pct:.1f}% of area)", "color": "#FF3C3C"},
            {"label": f"No change detected ({unchanged_pct:.1f}% of area)", "color": "transparent"},
        ],
    }

    stats = {
        "changed_pixel_pct": changed_pct,
        "unchanged_pixel_pct": unchanged_pct,
        "changed_pixel_count": changed_pixels,
        "unchanged_pixel_count": total_pixels - changed_pixels,
        "total_pixel_count": total_pixels,
        "image_size_wh": [w, h],
        "image_size_str": f"{w}x{h}",
        "threshold_used": round(threshold, 5),
        "threshold_raw_255": threshold_raw,
        "processing_time_ms": elapsed_ms,
        "size_mismatch_corrected": size_mismatch,
        "severity": severity,
        "overlay_url": overlay_url,
    }

    return ChangeDetectionResult(
        answer=answer,
        confidence=None,
        change_map=change_map,
        evidence=evidence,
        stats=stats,
    )
