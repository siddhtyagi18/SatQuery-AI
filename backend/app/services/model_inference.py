"""
backend/app/services/model_inference.py
========================================
Change detection inference adapter.

This module is the single entry point for bi-temporal change detection inference.
It implements a dispatcher that selects between:

  1. Trained model checkpoint (SiameseUNet)
     — used when CHANGE_DETECTION_CHECKPOINT is configured and the file exists.
     — tiles the input images into 256×256 patches, runs the model, reassembles.
     — explicitly labels outputs as "model_checkpoint" mode.

  2. CPU classical baseline (existing change_detection.py)
     — used when no checkpoint is configured or found.
     — grayscale absolute pixel difference + morphological noise cleanup.
     — explicitly labels outputs as "cpu_classical" mode.

The output of both paths is a ChangeDetectionResult from change_detection.py
(same schema, same frontend API contract), augmented with an ``execution_mode``
field in the evidence and stats dicts.

NO FABRICATED PREDICTIONS:
  - Classical mode always produces real pixel statistics from the actual images.
  - Model mode always runs the actual checkpoint. If the checkpoint fails to load
    or inference fails, the dispatcher falls back to classical and reports so.
  - The ``execution_mode`` string in every result clearly identifies which path ran.

IMAGE TILING FOR MODEL INFERENCE:
  LEVIR-CD images are 1024×1024. The model is trained on 256×256 crops.
  For inference we tile with 256×256 patches and 32px overlap, then stitch
  probability maps back to full resolution before thresholding.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Literal, Optional

from PIL import Image
import numpy as np

from .change_detection import run_cpu_change_detection, ChangeDetectionResult, _build_overlay
from ..config import get_settings
from ..logging_setup import logger


# Execution mode type
InferenceMode = Literal["model_checkpoint", "cpu_classical"]

# Module-level checkpoint cache (avoids reloading weights on every request)
_loaded_checkpoint_path: Optional[str] = None
_loaded_model = None  # SiameseUNet or None


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _try_load_checkpoint(checkpoint_path: Path):
    """
    Attempt to load a SiameseUNet from a .pt checkpoint file.

    Returns the model on success, raises on failure (caller handles fallback).
    Caches the loaded model by path to avoid repeated disk I/O.
    """
    global _loaded_checkpoint_path, _loaded_model

    ckpt_str = str(checkpoint_path)
    if _loaded_checkpoint_path == ckpt_str and _loaded_model is not None:
        logger.debug("[model_inference] Using cached checkpoint: %s", ckpt_str)
        return _loaded_model

    try:
        import torch
        from .models.siamese_unet import SiameseUNet
    except ImportError as e:
        raise RuntimeError(f"torch is required for checkpoint inference: {e}") from e

    logger.info("[model_inference] Loading checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Support both raw state_dict and checkpoint dict
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model = SiameseUNet(in_channels=6, base_filters=16)
    model.load_state_dict(state_dict)
    model.eval()

    _loaded_checkpoint_path = ckpt_str
    _loaded_model = model
    logger.info("[model_inference] Checkpoint loaded successfully.")
    return model


# ---------------------------------------------------------------------------
# Model-based inference (with tiling)
# ---------------------------------------------------------------------------

_TILE_SIZE = 256
_TILE_OVERLAP = 32


def _tile_inference(
    model,
    before_path: Path,
    after_path: Path,
    tile_size: int = _TILE_SIZE,
    overlap: int = _TILE_OVERLAP,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Run model inference on a full-resolution image pair using overlapping tiling.

    Returns a binary change mask (H × W uint8) with values {0, 1}.

    Steps:
    1. Open and align images.
    2. Slide a tile_size × tile_size window with given overlap.
    3. For each tile, run forward pass → probability.
    4. Accumulate probabilities in a full-size float32 accumulator (average overlapping tiles).
    5. Threshold accumulated probabilities at 0.5.
    """
    try:
        import torch
        import torchvision.transforms.functional as TF
    except ImportError as e:
        raise RuntimeError(f"torch + torchvision required for tiled inference: {e}") from e

    before_img = Image.open(before_path).convert("RGB")
    after_img = Image.open(after_path).convert("RGB")
    if before_img.size != after_img.size:
        after_img = after_img.resize(before_img.size, Image.LANCZOS)

    W, H = before_img.size
    stride = tile_size - overlap

    # Accumulators: sum of probabilities + count of contributions per pixel
    prob_sum = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        y = 0
        while y < H:
            y_end = min(y + tile_size, H)
            y_start = max(0, y_end - tile_size)

            x = 0
            while x < W:
                x_end = min(x + tile_size, W)
                x_start = max(0, x_end - tile_size)

                tile_a = before_img.crop((x_start, y_start, x_end, y_end))
                tile_b = after_img.crop((x_start, y_start, x_end, y_end))

                # Pad to tile_size if near boundary
                if tile_a.size != (tile_size, tile_size):
                    pad_w = tile_size - tile_a.size[0]
                    pad_h = tile_size - tile_a.size[1]
                    tile_a = _pad_image(tile_a, tile_size, tile_size)
                    tile_b = _pad_image(tile_b, tile_size, tile_size)
                else:
                    pad_w = pad_h = 0

                t_a = TF.to_tensor(tile_a).unsqueeze(0)   # (1, 3, tile_size, tile_size)
                t_b = TF.to_tensor(tile_b).unsqueeze(0)
                inp = torch.cat([t_a, t_b], dim=1)        # (1, 6, tile_size, tile_size)

                logits = model(inp)                        # (1, 1, tile_size, tile_size)
                prob = torch.sigmoid(logits).squeeze().cpu().numpy()  # (tile_size, tile_size)

                # Trim padding
                actual_h = y_end - y_start
                actual_w = x_end - x_start
                prob_crop = prob[:actual_h, :actual_w]

                prob_sum[y_start:y_end, x_start:x_end] += prob_crop
                count_map[y_start:y_end, x_start:x_end] += 1.0

                if x_end == W:
                    break
                x += stride

            if y_end == H:
                break
            y += stride

    # Average and threshold
    count_map = np.maximum(count_map, 1.0)  # avoid division by zero
    avg_prob = prob_sum / count_map
    binary_mask = (avg_prob >= threshold).astype(np.uint8)
    return binary_mask


def _pad_image(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Pad an image with black on the right/bottom to reach (target_w, target_h)."""
    new_img = Image.new(img.mode, (target_w, target_h), 0)
    new_img.paste(img, (0, 0))
    return new_img


def _run_model_inference(
    model,
    before_path: Path,
    after_path: Path,
    analysis_id: str,
    checkpoint_path_str: str,
) -> ChangeDetectionResult:
    """Run model-based change detection and return a ChangeDetectionResult."""
    from pathlib import Path as _Path
    import time as _time

    # Ensure results dir exists
    from .change_detection import _RESULTS_DIR, _ensure_results_dir
    _ensure_results_dir()

    t0 = time.perf_counter()
    logger.info(
        "[model_inference] Running SiameseUNet inference: analysis=%s checkpoint=%s",
        analysis_id, checkpoint_path_str,
    )

    binary_mask = _tile_inference(
        model, before_path, after_path,
        tile_size=_TILE_SIZE, overlap=_TILE_OVERLAP, threshold=0.5,
    )

    # Load before image for size reference
    before_img = Image.open(before_path).convert("RGB")
    W, H = before_img.size

    # Statistics
    total_pixels = H * W
    changed_pixels = int(binary_mask.sum())
    changed_pct = round(changed_pixels / total_pixels * 100, 2) if total_pixels > 0 else 0.0
    unchanged_pct = round(100.0 - changed_pct, 2)

    # Severity label (shared with classical path)
    from .change_detection import _severity_label
    severity = _severity_label(changed_pct)

    # Save overlay PNG
    results_dir = _ensure_results_dir()
    mask_filename = f"{analysis_id}_changemap.png"
    mask_path = results_dir / mask_filename
    overlay = _build_overlay(binary_mask, (W, H))
    overlay.save(mask_path, format="PNG", optimize=False)
    overlay_url = f"/api/results/{mask_filename}"

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        "[model_inference] Done in %dms — changed=%.2f%% severity=%s",
        elapsed_ms, changed_pct, severity,
    )

    answer = (
        f"## Change Detection Report (Siamese U-Net Model)\n\n"
        f"**Trained SiameseUNet model detected change across {changed_pct:.1f}% of the image area** "
        f"({changed_pixels:,} of {total_pixels:,} pixels), indicating {severity} scene change "
        f"(severity: **{severity}**).\n\n"
        f"The remaining {unchanged_pct:.1f}% of pixels were classified as unchanged.\n\n"
        f"**Model details:** Lightweight Siamese U-Net (~490K parameters) trained on the LEVIR-CD "
        f"building change detection dataset (1024×1024 RGB image pairs, binary change masks). "
        f"Inference uses overlapping 256×256 tile sliding with {_TILE_OVERLAP}px overlap.\n\n"
        f"**Interpretation:** This result reflects the model's learned change representation. "
        f"The model was trained on building-related changes in bi-temporal satellite imagery. "
        f"Results outside that domain may be less reliable.\n\n"
        f"*Analysis performed by trained SiameseUNet checkpoint: {checkpoint_path_str}*"
    )

    evidence = [
        f"Inference mode: trained SiameseUNet model checkpoint ({checkpoint_path_str}).",
        f"Tile size: {_TILE_SIZE}×{_TILE_SIZE}px with {_TILE_OVERLAP}px overlap (probability averaging).",
        f"Changed pixels (model prediction): {changed_pixels:,} / {total_pixels:,} ({changed_pct:.2f}%).",
        f"Unchanged pixels: {total_pixels - changed_pixels:,} / {total_pixels:,} ({unchanged_pct:.2f}%).",
        f"Severity label: {severity} (heuristic: low <5%, moderate 5–25%, high >25%).",
        f"Reference image dimensions: {W}×{H} px.",
        f"Change mask overlay saved to: {overlay_url}",
        "Output is from a trained model checkpoint, not fabricated or from a template.",
    ]

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
        "image_size_wh": [W, H],
        "image_size_str": f"{W}x{H}",
        "threshold_used": 0.5,
        "threshold_raw_255": 128,  # 0.5 × 255
        "processing_time_ms": elapsed_ms,
        "size_mismatch_corrected": False,
        "severity": severity,
        "overlay_url": overlay_url,
        "execution_mode": "model_checkpoint",
        "checkpoint_path": checkpoint_path_str,
    }

    return ChangeDetectionResult(
        answer=answer,
        confidence=None,  # Model outputs a mask, not a single calibrated score
        change_map=change_map,
        evidence=evidence,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def get_inference_mode() -> InferenceMode:
    """
    Return the current inference mode without running inference.

    Returns "model_checkpoint" if CHANGE_DETECTION_CHECKPOINT is configured
    and the file exists. Otherwise returns "cpu_classical".
    """
    settings = get_settings()
    ckpt = settings.CHANGE_DETECTION_CHECKPOINT
    if ckpt and Path(ckpt).exists():
        return "model_checkpoint"
    return "cpu_classical"


def run_change_detection(
    before_path: Path,
    after_path: Path,
    analysis_id: str,
    threshold: float = 35.0 / 255.0,
) -> ChangeDetectionResult:
    """
    Dispatcher: run bi-temporal change detection using the best available method.

    Decision tree:
    1. If CHANGE_DETECTION_CHECKPOINT is set and the file exists:
       → Load checkpoint and run SiameseUNet tile inference.
       → On any failure, log the error and fall back to CPU classical.
    2. Otherwise:
       → Run CPU classical pixel-difference (existing change_detection.py).

    The result's ``stats["execution_mode"]`` always indicates which path ran:
    "model_checkpoint" or "cpu_classical".

    Parameters
    ----------
    before_path  : Path to the "before" (T1) image.
    after_path   : Path to the "after" (T2) image.
    analysis_id  : Unique ID for naming output files.
    threshold    : Only used for classical fallback path.

    Returns
    -------
    ChangeDetectionResult — same schema regardless of which path ran.
    """
    settings = get_settings()
    ckpt_cfg = settings.CHANGE_DETECTION_CHECKPOINT

    if ckpt_cfg:
        ckpt_path = Path(ckpt_cfg)
        if ckpt_path.exists():
            try:
                model = _try_load_checkpoint(ckpt_path)
                result = _run_model_inference(
                    model, before_path, after_path, analysis_id, str(ckpt_path)
                )
                logger.info(
                    "[model_inference] Dispatcher: used model_checkpoint path for analysis=%s",
                    analysis_id,
                )
                return result
            except Exception as exc:
                logger.warning(
                    "[model_inference] Checkpoint inference failed (%s: %s). "
                    "Falling back to CPU classical baseline.",
                    type(exc).__name__, exc,
                )
                # Fall through to classical
        else:
            logger.info(
                "[model_inference] Checkpoint configured (%s) but file does not exist. "
                "Falling back to CPU classical baseline.",
                ckpt_cfg,
            )
    else:
        logger.debug(
            "[model_inference] No checkpoint configured. Using CPU classical baseline."
        )

    # --- CPU classical fallback ---
    result = run_cpu_change_detection(
        before_path=before_path,
        after_path=after_path,
        analysis_id=analysis_id,
        threshold=threshold,
    )
    # Tag execution mode in stats
    result.stats["execution_mode"] = "cpu_classical"
    result.stats["checkpoint_path"] = None
    return result
