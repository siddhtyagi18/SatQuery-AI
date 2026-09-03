#!/usr/bin/env python3
"""
backend/scripts/visualize_change_predictions.py
================================================
Qualitative evaluation utility for the Siamese U-Net change detector on LEVIR-CD.

This script:
1. Loads the trained Siamese U-Net checkpoint (e.g. backend/checkpoints/best_model.pt).
2. Loads the genuine LEVIR-CD test split.
3. Runs genuine model inference on selected test pairs.
4. For each sample, generates and saves a 6-panel qualitative comparison:
   - Panel 1: Before Image A (T1)
   - Panel 2: After Image B (T2)
   - Panel 3: Ground-truth change mask
   - Panel 4: Predicted probability heatmap [0, 1]
   - Panel 5: Predicted binary change mask (thresholded)
   - Panel 6: Change overlay on After Image B
5. Computes and prints exact pixel counts (GT changed, Predicted changed, FP, FN, TP, TN)
   and standard change detection metrics (IoU, F1/Dice, Precision, Recall, Accuracy).
6. Outputs clean summary tables and saves visual figures to the target output directory.

Usage examples:
--------------
# Default: 10 random test samples on CPU/CUDA
python scripts/visualize_change_predictions.py

# Custom sample count and threshold:
python scripts/visualize_change_predictions.py --num-samples 10 --threshold 0.5

# Specify custom checkpoint, dataset root, output directory:
python scripts/visualize_change_predictions.py \\
    --checkpoint ./checkpoints/best_model.pt \\
    --data-root C:/Users/Lenovo/Downloads/LEVIR-CD \\
    --output-dir outputs/change_detection/test_predictions \\
    --num-samples 10 \\
    --threshold 0.5
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Resolve backend root to allow importing internal services
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


# ---------------------------------------------------------------------------
# Torch requirement check
# ---------------------------------------------------------------------------

def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torchvision.transforms.functional as TF
        return torch, nn, TF
    except ImportError as e:
        print(f"ERROR: torch / torchvision is required for qualitative evaluation: {e}")
        print("Install via: pip install torch torchvision")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Metrics Data Structure & Computation
# ---------------------------------------------------------------------------

@dataclass
class SampleMetrics:
    sample_id: str
    filename: str
    gt_changed: int
    pred_changed: int
    tp: int
    fp: int
    fn: int
    tn: int
    total_pixels: int
    threshold: float
    iou: float
    f1: float
    precision: float
    recall: float
    accuracy: float


def compute_sample_metrics(
    prob_map: np.ndarray,
    gt_mask: np.ndarray,
    threshold: float = 0.5,
    sample_id: str = "",
    filename: str = "",
    smooth: float = 1e-6,
) -> Tuple[SampleMetrics, np.ndarray]:
    """
    Compute pixel-level change detection metrics for a single sample.

    prob_map : (H, W) float array with values in [0.0, 1.0]
    gt_mask  : (H, W) binary array with values {0, 1}
    """
    pred_bin = (prob_map >= threshold).astype(np.uint8)
    gt_bin = (gt_mask > 0.5).astype(np.uint8)

    pred_flat = pred_bin.reshape(-1)
    gt_flat = gt_bin.reshape(-1)

    tp = int(((pred_flat == 1) & (gt_flat == 1)).sum())
    fp = int(((pred_flat == 1) & (gt_flat == 0)).sum())
    fn = int(((pred_flat == 0) & (gt_flat == 1)).sum())
    tn = int(((pred_flat == 0) & (gt_flat == 0)).sum())

    gt_changed = tp + fn
    pred_changed = tp + fp
    total_pixels = len(pred_flat)

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1 = 2 * precision * recall / (precision + recall + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    accuracy = (tp + tn + smooth) / (tp + fp + fn + tn + smooth)

    metrics = SampleMetrics(
        sample_id=sample_id,
        filename=filename,
        gt_changed=gt_changed,
        pred_changed=pred_changed,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        total_pixels=total_pixels,
        threshold=threshold,
        iou=round(float(iou), 4),
        f1=round(float(f1), 4),
        precision=round(float(precision), 4),
        recall=round(float(recall), 4),
        accuracy=round(float(accuracy), 4),
    )
    return metrics, pred_bin


# ---------------------------------------------------------------------------
# Colormap & Visual Overlay Helpers (Pure NumPy + Pillow)
# ---------------------------------------------------------------------------

def apply_turbo_colormap(prob: np.ndarray) -> Image.Image:
    """
    Convert a 2D float array in [0.0, 1.0] to a smooth Turbo/Jet colormap RGB image.
    """
    x = np.clip(prob, 0.0, 1.0)
    # Fast polynomial approximation of Turbo / Jet colormap
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def create_change_overlay(
    img_b: Image.Image,
    pred_bin: np.ndarray,
    color: Tuple[int, int, int] = (255, 50, 50),
    alpha: float = 0.50,
) -> Image.Image:
    """
    Overlay predicted binary change mask on top of Image B with translucent color.
    """
    base = img_b.convert("RGBA")
    H, W = pred_bin.shape
    overlay_arr = np.zeros((H, W, 4), dtype=np.uint8)
    mask = pred_bin > 0
    overlay_arr[mask] = [color[0], color[1], color[2], int(255 * alpha)]
    overlay_img = Image.fromarray(overlay_arr, mode="RGBA")
    blended = Image.alpha_composite(base, overlay_img)
    return blended.convert("RGB")


def create_composite_figure(
    img_a: Image.Image,
    img_b: Image.Image,
    gt_mask: np.ndarray,
    prob_map: np.ndarray,
    pred_bin: np.ndarray,
    overlay_b: Image.Image,
    metrics: SampleMetrics,
) -> Image.Image:
    """
    Assemble a 2x3 grid comparison figure with a header banner.

    Layout:
    Header: Filename, Threshold, Metrics (IoU, F1, Prec, Rec), Pixel Statistics (GT, Pred, FP, FN)
    Row 1: [ (A) Before Image (T1) | (B) After Image (T2) | (C) Ground Truth Mask ]
    Row 2: [ (D) Predicted Probability Heatmap | (E) Predicted Binary Mask | (F) Change Overlay on Image B ]
    """
    tile_w, tile_h = img_a.size

    # Ground truth as RGB (0=black, 255=white)
    gt_rgb = Image.fromarray((gt_mask * 255).astype(np.uint8), mode="L").convert("RGB")

    # Prob map colormap
    prob_rgb = apply_turbo_colormap(prob_map)
    if prob_rgb.size != (tile_w, tile_h):
        prob_rgb = prob_rgb.resize((tile_w, tile_h), Image.BILINEAR)

    # Pred binary mask as RGB (0=black, 255=white / cyan highlight)
    pred_rgb_arr = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    pred_rgb_arr[pred_bin == 1] = [0, 220, 255]  # Crisp cyan for predicted change
    pred_rgb = Image.fromarray(pred_rgb_arr, mode="RGB")

    # Dimensions
    pad = 12
    label_h = 24
    header_h = 70
    grid_cols, grid_rows = 3, 2

    canvas_w = grid_cols * tile_w + (grid_cols + 1) * pad
    canvas_h = header_h + grid_rows * (tile_h + label_h) + (grid_rows + 1) * pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(18, 22, 28))
    draw = ImageDraw.Draw(canvas)

    # Load default font
    font = ImageFont.load_default()

    # Draw Header Banner
    draw.rectangle([(0, 0), (canvas_w, header_h)], fill=(28, 34, 44))
    title_line1 = f"SatQuery-AI Change Detection Evaluation | Sample: {metrics.filename} ({metrics.sample_id})"
    title_line2 = (
        f"Metrics: IoU={metrics.iou:.4f}  |  F1/Dice={metrics.f1:.4f}  |  "
        f"Precision={metrics.precision:.4f}  |  Recall={metrics.recall:.4f}  |  Accuracy={metrics.accuracy:.4f}"
    )
    title_line3 = (
        f"Pixel Counts: GT Changed={metrics.gt_changed:,}  |  Pred Changed={metrics.pred_changed:,}  |  "
        f"TP={metrics.tp:,}  |  FP={metrics.fp:,}  |  FN={metrics.fn:,}  |  Threshold={metrics.threshold}"
    )

    draw.text((pad + 4, 8), title_line1, fill=(240, 245, 255), font=font)
    draw.text((pad + 4, 28), title_line2, fill=(80, 220, 160), font=font)
    draw.text((pad + 4, 48), title_line3, fill=(180, 200, 220), font=font)

    # Panels configuration
    panels = [
        ("(A) Before Image (T1)", img_a),
        ("(B) After Image (T2)", img_b),
        (f"(C) Ground-Truth Mask ({metrics.gt_changed:,} px)", gt_rgb),
        (f"(D) Predicted Probability Map [0.0 - 1.0]", prob_rgb),
        (f"(E) Predicted Binary Mask (thr={metrics.threshold}) ({metrics.pred_changed:,} px)", pred_rgb),
        ("(F) Predicted Change Overlay on (B)", overlay_b),
    ]

    for idx, (title, p_img) in enumerate(panels):
        row = idx // grid_cols
        col = idx % grid_cols

        x = pad + col * (tile_w + pad)
        y = header_h + pad + row * (tile_h + label_h + pad)

        # Draw panel title
        draw.text((x + 2, y), title, fill=(200, 215, 230), font=font)

        # Paste panel image
        canvas.paste(p_img, (x, y + label_h))

        # Thin border around panel
        draw.rectangle(
            [(x - 1, y + label_h - 1), (x + tile_w, y + label_h + tile_h)],
            outline=(60, 75, 95),
            width=1,
        )

    return canvas


# ---------------------------------------------------------------------------
# Model Loader
# ---------------------------------------------------------------------------

def load_siamese_model(checkpoint_path: Path, device: "torch.device"):
    """
    Load the SiameseUNet model from the specified checkpoint.
    """
    import torch
    from app.services.models.siamese_unet import SiameseUNet

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            epoch = ckpt.get("epoch", "unknown")
            metrics = ckpt.get("metrics", {})
            print(f"  Checkpoint details: Epoch={epoch}, Saved metrics={metrics}")
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    model = SiameseUNet(in_channels=6, base_filters=16)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model loaded successfully: SiameseUNet ({param_count:,} parameters) on {device}")
    return model


# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SatQuery-AI Qualitative Change Detection Visualizer on LEVIR-CD Test Split"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to trained SiameseUNet checkpoint (default: checks .env or backend/checkpoints/best_model.pt)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Path to LEVIR-CD root directory (default: checks LEVIR_CD_DATASET_PATH or LEVIR_CD_ROOT in .env)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/change_detection/test_predictions"),
        help="Directory to save visual comparison figures (default: outputs/change_detection/test_predictions)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of test samples to visualize (default: 10)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binary change detection threshold in [0.0, 1.0] (default: 0.5)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=256,
        help="Image size for center cropping test images (default: 256, matching test evaluation)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sample selection (default: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on: 'cuda', 'cpu', or auto-detect (default: auto)",
    )
    args = parser.parse_args()

    torch, nn, TF = _require_torch()

    # 1. Resolve Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = (device.type == "cuda")

    # 2. Resolve Checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        env_ckpt = os.getenv("CHANGE_DETECTION_CHECKPOINT")
        if env_ckpt and Path(env_ckpt).exists():
            checkpoint_path = Path(env_ckpt)
        else:
            default_candidates = [
                _BACKEND_ROOT / "checkpoints" / "best_model.pt",
                Path("checkpoints/best_model.pt"),
                Path("./backend/checkpoints/best_model.pt"),
            ]
            for c in default_candidates:
                if c.exists():
                    checkpoint_path = c
                    break

    if checkpoint_path is None or not checkpoint_path.exists():
        print(f"ERROR: Trained checkpoint not found at: {checkpoint_path}")
        print("Please provide --checkpoint path/to/best_model.pt or set CHANGE_DETECTION_CHECKPOINT in .env")
        sys.exit(1)

    # 3. Resolve Dataset Root
    data_root = args.data_root
    if data_root is None:
        env_root = os.getenv("LEVIR_CD_DATASET_PATH") or os.getenv("LEVIR_CD_ROOT")
        if env_root and Path(env_root).exists():
            data_root = Path(env_root)
        else:
            default_candidates = [
                Path("C:/Users/Lenovo/Downloads/LEVIR-CD"),
                Path("./data/LEVIR-CD"),
                Path("../data/LEVIR-CD"),
            ]
            for d in default_candidates:
                if d.exists():
                    data_root = d
                    break

    if data_root is None or not data_root.exists():
        print(f"ERROR: LEVIR-CD root directory not found at: {data_root}")
        print("Please provide --data-root path/to/LEVIR-CD or set LEVIR_CD_DATASET_PATH in .env")
        sys.exit(1)

    test_dir = data_root / "test"
    if not test_dir.exists():
        print(f"ERROR: Test split directory not found: {test_dir}")
        sys.exit(1)

    dir_a = test_dir / "A"
    dir_b = test_dir / "B"
    dir_label = test_dir / "label"

    for d in (dir_a, dir_b, dir_label):
        if not d.exists():
            print(f"ERROR: Subdirectory missing in test split: {d}")
            sys.exit(1)

    # 4. Find matched test triplets
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    files_a = {f.name: f for f in dir_a.iterdir() if f.is_file() and f.suffix.lower() in exts}
    files_b = {f.name: f for f in dir_b.iterdir() if f.is_file() and f.suffix.lower() in exts}
    files_l = {f.name: f for f in dir_label.iterdir() if f.is_file() and f.suffix.lower() in exts}

    matched_names = sorted(set(files_a) & set(files_b) & set(files_l))
    if not matched_names:
        print(f"ERROR: No matching A/B/label image triplets found in {test_dir}")
        sys.exit(1)

    print("\n" + "=" * 75)
    print("SatQuery-AI Qualitative Evaluation Utility (Siamese U-Net)")
    print("=" * 75)
    print(f"  Device:           {device} (CUDA: {is_cuda})")
    print(f"  Checkpoint:       {checkpoint_path}")
    print(f"  Dataset Root:     {data_root}")
    print(f"  Test Triplets:    {len(matched_names)} total pairs available")
    print(f"  Samples to Eval:  {args.num_samples}")
    print(f"  Crop Size:        {args.img_size}×{args.img_size}")
    print(f"  Binary Threshold: {args.threshold}")
    print(f"  Output Directory: {args.output_dir}")
    print("=" * 75 + "\n")

    # 5. Load Model
    model = load_siamese_model(checkpoint_path, device)

    # 6. Sample Selection
    random.seed(args.seed)
    n_samples = min(args.num_samples, len(matched_names))
    selected_names = random.sample(matched_names, n_samples)
    selected_names.sort()

    # 7. Prepare output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample_metrics_list: List[SampleMetrics] = []
    generated_files: List[Path] = []

    print(f"Running genuine model inference on {len(selected_names)} test samples...\n")
    print(
        f"{'Sample ID':<12} | {'Filename':<14} | {'GT (px)':<9} | {'Pred (px)':<9} | "
        f"{'FP (px)':<8} | {'FN (px)':<8} | {'TP (px)':<8} | {'IoU':<6} | {'F1':<6} | {'Prec':<6} | {'Recall':<6}"
    )
    print("-" * 115)

    crop_size = args.img_size

    for idx, name in enumerate(selected_names, 1):
        path_a = files_a[name]
        path_b = files_b[name]
        path_l = files_l[name]

        # Open raw images
        raw_a = Image.open(path_a).convert("RGB")
        raw_b = Image.open(path_b).convert("RGB")
        raw_l = Image.open(path_l).convert("L")

        # Centre crop (consistent with test evaluation)
        W, H = raw_a.size
        i = (H - crop_size) // 2
        j = (W - crop_size) // 2

        crop_a = TF.crop(raw_a, i, j, crop_size, crop_size)
        crop_b = TF.crop(raw_b, i, j, crop_size, crop_size)
        crop_l = TF.crop(raw_l, i, j, crop_size, crop_size)

        # Convert to tensor [0.0, 1.0]
        t_a = TF.to_tensor(crop_a).unsqueeze(0).to(device)  # (1, 3, H, W)
        t_b = TF.to_tensor(crop_b).unsqueeze(0).to(device)  # (1, 3, H, W)

        # Ground truth binary mask {0, 1}
        gt_arr = (np.array(crop_l) > 127).astype(np.uint8)

        # Run genuine Siamese U-Net forward pass
        inp = torch.cat([t_a, t_b], dim=1)  # (1, 6, H, W)
        with torch.no_grad():
            logits = model(inp)  # (1, 1, H, W)
            prob_tensor = torch.sigmoid(logits)  # (1, 1, H, W)
            prob_map = prob_tensor.squeeze().cpu().numpy()  # (H, W) float32 in [0, 1]

        # Compute metrics and binary prediction
        sample_id = f"sample_{idx:02d}"
        metrics, pred_bin = compute_sample_metrics(
            prob_map=prob_map,
            gt_mask=gt_arr,
            threshold=args.threshold,
            sample_id=sample_id,
            filename=name,
        )
        sample_metrics_list.append(metrics)

        # Create overlay on Image B
        overlay_b = create_change_overlay(crop_b, pred_bin, color=(255, 40, 40), alpha=0.50)

        # Create composite figure
        fig_img = create_composite_figure(
            img_a=crop_a,
            img_b=crop_b,
            gt_mask=gt_arr,
            prob_map=prob_map,
            pred_bin=pred_bin,
            overlay_b=overlay_b,
            metrics=metrics,
        )

        # Save output figure with original filename stem
        stem = Path(name).stem
        out_filename = f"{sample_id}_{stem}_prediction.png"
        out_path = args.output_dir / out_filename
        fig_img.save(out_path, quality=95)
        generated_files.append(out_path)

        print(
            f"{metrics.sample_id:<12} | {metrics.filename:<14} | {metrics.gt_changed:<9,d} | "
            f"{metrics.pred_changed:<9,d} | {metrics.fp:<8,d} | {metrics.fn:<8,d} | {metrics.tp:<8,d} | "
            f"{metrics.iou:<6.4f} | {metrics.f1:<6.4f} | {metrics.precision:<6.4f} | {metrics.recall:<6.4f}"
        )

    print("-" * 115)

    # 8. Overall Summary Statistics
    total_gt = sum(m.gt_changed for m in sample_metrics_list)
    total_pred = sum(m.pred_changed for m in sample_metrics_list)
    total_tp = sum(m.tp for m in sample_metrics_list)
    total_fp = sum(m.fp for m in sample_metrics_list)
    total_fn = sum(m.fn for m in sample_metrics_list)
    total_tn = sum(m.tn for m in sample_metrics_list)

    mean_iou = float(np.mean([m.iou for m in sample_metrics_list]))
    mean_f1 = float(np.mean([m.f1 for m in sample_metrics_list]))
    mean_prec = float(np.mean([m.precision for m in sample_metrics_list]))
    mean_rec = float(np.mean([m.recall for m in sample_metrics_list]))
    mean_acc = float(np.mean([m.accuracy for m in sample_metrics_list]))

    # Global aggregate metrics (micro-averaged across all evaluated pixels)
    smooth = 1e-6
    global_prec = (total_tp + smooth) / (total_tp + total_fp + smooth)
    global_rec = (total_tp + smooth) / (total_tp + total_fn + smooth)
    global_f1 = 2 * global_prec * global_rec / (global_prec + global_rec + smooth)
    global_iou = (total_tp + smooth) / (total_tp + total_fp + total_fn + smooth)

    print("\n" + "=" * 75)
    print("EVALUATION SUMMARY OVER EVALUATED TEST SAMPLES")
    print("=" * 75)
    print(f"  Evaluated Samples:      {len(sample_metrics_list)}")
    print(f"  Prediction Threshold:   {args.threshold}")
    print(f"  Device Used:            {device} (CUDA: {is_cuda})")
    print(f"  Loaded Checkpoint:      {checkpoint_path}")
    print(f"  Output Directory:       {args.output_dir}")
    print("  -------------------------------------------------------------")
    print(f"  Mean Sample IoU:        {mean_iou:.4f}")
    print(f"  Mean Sample F1/Dice:    {mean_f1:.4f}")
    print(f"  Mean Sample Precision:  {mean_prec:.4f}")
    print(f"  Mean Sample Recall:     {mean_rec:.4f}")
    print(f"  Mean Sample Accuracy:   {mean_acc:.4f}")
    print("  -------------------------------------------------------------")
    print(f"  Global Micro IoU:       {global_iou:.4f}")
    print(f"  Global Micro F1/Dice:   {global_f1:.4f}")
    print(f"  Global Micro Precision: {global_prec:.4f}")
    print(f"  Global Micro Recall:    {global_rec:.4f}")
    print("  -------------------------------------------------------------")
    print(f"  Total GT Changed Px:    {total_gt:,}")
    print(f"  Total Pred Changed Px:  {total_pred:,}")
    print(f"  Total True Positives:   {total_tp:,}")
    print(f"  Total False Positives:  {total_fp:,}")
    print(f"  Total False Negatives:  {total_fn:,}")
    print("=" * 75)
    print(f"\n[SUCCESS] Generated {len(generated_files)} qualitative comparison figures:")
    for f in generated_files:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
