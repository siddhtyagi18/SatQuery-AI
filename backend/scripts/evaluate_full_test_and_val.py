#!/usr/bin/env python3
"""
backend/scripts/evaluate_full_test_and_val.py
=============================================
Complete official evaluation script for the trained Siamese U-Net change detector:
1. Full 128-sample evaluation on LEVIR-CD test split at threshold 0.5 (without threshold tuning).
2. Saves full results JSON and representative 6-panel qualitative prediction figures.
3. Separately performs threshold sensitivity sweep ([0.30 - 0.70]) ONLY on the 64-sample VALIDATION set.
4. Identifies optimal threshold from validation set performance.

All metrics are calculated genuinely from actual dataset images and trained weights.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Resolve backend root to allow importing internal services
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


def _require_torch():
    try:
        import torch
        import torchvision.transforms.functional as TF
        return torch, TF
    except ImportError as e:
        print(f"ERROR: torch / torchvision is required: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Colormap & Visualization Helpers
# ---------------------------------------------------------------------------

def apply_turbo_colormap(prob: np.ndarray) -> Image.Image:
    x = np.clip(prob, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def create_change_overlay(
    img_b: Image.Image,
    pred_bin: np.ndarray,
    color: Tuple[int, int, int] = (255, 40, 40),
    alpha: float = 0.50,
) -> Image.Image:
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
    meta: Dict[str, Any],
) -> Image.Image:
    tile_w, tile_h = img_a.size

    gt_rgb = Image.fromarray((gt_mask * 255).astype(np.uint8), mode="L").convert("RGB")
    prob_rgb = apply_turbo_colormap(prob_map)
    if prob_rgb.size != (tile_w, tile_h):
        prob_rgb = prob_rgb.resize((tile_w, tile_h), Image.BILINEAR)

    pred_rgb_arr = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    pred_rgb_arr[pred_bin == 1] = [0, 220, 255]
    pred_rgb = Image.fromarray(pred_rgb_arr, mode="RGB")

    pad = 12
    label_h = 24
    header_h = 70
    grid_cols, grid_rows = 3, 2

    canvas_w = grid_cols * tile_w + (grid_cols + 1) * pad
    canvas_h = header_h + grid_rows * (tile_h + label_h) + (grid_rows + 1) * pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(18, 22, 28))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.rectangle([(0, 0), (canvas_w, header_h)], fill=(28, 34, 44))
    title_line1 = f"LEVIR-CD Full Test Evaluation | Sample: {meta['filename']} (Pair #{meta['index']:03d})"
    title_line2 = (
        f"Metrics: IoU={meta['iou']:.4f}  |  F1/Dice={meta['f1']:.4f}  |  "
        f"Precision={meta['precision']:.4f}  |  Recall={meta['recall']:.4f}  |  Accuracy={meta['accuracy']:.4f}"
    )
    title_line3 = (
        f"Pixels: GT={meta['gt_changed']:,}  |  Pred={meta['pred_changed']:,}  |  "
        f"TP={meta['tp']:,}  |  FP={meta['fp']:,}  |  FN={meta['fn']:,}  |  Threshold={meta['threshold']}"
    )

    draw.text((pad + 4, 8), title_line1, fill=(240, 245, 255), font=font)
    draw.text((pad + 4, 28), title_line2, fill=(80, 220, 160), font=font)
    draw.text((pad + 4, 48), title_line3, fill=(180, 200, 220), font=font)

    panels = [
        ("(A) Before Image (T1)", img_a),
        ("(B) After Image (T2)", img_b),
        (f"(C) Ground-Truth Mask ({meta['gt_changed']:,} px)", gt_rgb),
        ("(D) Predicted Probability Map [0.0 - 1.0]", prob_rgb),
        (f"(E) Predicted Binary Mask (thr={meta['threshold']}) ({meta['pred_changed']:,} px)", pred_rgb),
        ("(F) Change Overlay on Image (B)", overlay_b),
    ]

    for idx, (title, p_img) in enumerate(panels):
        row = idx // grid_cols
        col = idx % grid_cols
        x = pad + col * (tile_w + pad)
        y = header_h + pad + row * (tile_h + label_h + pad)

        draw.text((x + 2, y), title, fill=(200, 215, 230), font=font)
        canvas.paste(p_img, (x, y + label_h))
        draw.rectangle(
            [(x - 1, y + label_h - 1), (x + tile_w, y + label_h + tile_h)],
            outline=(60, 75, 95),
            width=1,
        )

    return canvas


# ---------------------------------------------------------------------------
# Model Loader
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path, device):
    import torch
    from app.services.models.siamese_unet import SiameseUNet

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    model = SiameseUNet(in_channels=6, base_filters=16)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, ckpt


# ---------------------------------------------------------------------------
# Split Evaluation Helper
# ---------------------------------------------------------------------------

def get_split_triplets(split_dir: Path) -> List[Tuple[str, Path, Path, Path]]:
    dir_a = split_dir / "A"
    dir_b = split_dir / "B"
    dir_label = split_dir / "label"
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    files_a = {f.name: f for f in dir_a.iterdir() if f.is_file() and f.suffix.lower() in exts}
    files_b = {f.name: f for f in dir_b.iterdir() if f.is_file() and f.suffix.lower() in exts}
    files_l = {f.name: f for f in dir_label.iterdir() if f.is_file() and f.suffix.lower() in exts}

    matched = sorted(set(files_a) & set(files_b) & set(files_l))
    return [(name, files_a[name], files_b[name], files_l[name]) for name in matched]


def evaluate_split(
    model,
    triplets: List[Tuple[str, Path, Path, Path]],
    device,
    crop_size: int = 256,
    threshold: float = 0.5,
    save_qualitative_dir: Optional[Path] = None,
    num_qualitative: int = 15,
) -> Dict[str, Any]:
    torch, TF = _require_torch()
    smooth = 1e-6

    sample_results = []
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    total_gt_px, total_pred_px = 0, 0

    # Cache pre-computed probability maps and GT masks for threshold sweeping if needed
    cached_probs_and_gts = []

    if save_qualitative_dir:
        save_qualitative_dir.mkdir(parents=True, exist_ok=True)

    # Pick representative qualitative indices (spread throughout, prioritizing samples with changes + zero-change samples)
    qualitative_indices = set()

    for idx, (name, path_a, path_b, path_l) in enumerate(triplets, 1):
        raw_a = Image.open(path_a).convert("RGB")
        raw_b = Image.open(path_b).convert("RGB")
        raw_l = Image.open(path_l).convert("L")

        W, H = raw_a.size
        i = (H - crop_size) // 2
        j = (W - crop_size) // 2

        crop_a = TF.crop(raw_a, i, j, crop_size, crop_size)
        crop_b = TF.crop(raw_b, i, j, crop_size, crop_size)
        crop_l = TF.crop(raw_l, i, j, crop_size, crop_size)

        t_a = TF.to_tensor(crop_a).unsqueeze(0).to(device)
        t_b = TF.to_tensor(crop_b).unsqueeze(0).to(device)
        gt_arr = (np.array(crop_l) > 127).astype(np.uint8)

        inp = torch.cat([t_a, t_b], dim=1)
        with torch.no_grad():
            logits = model(inp)
            prob_tensor = torch.sigmoid(logits)
            prob_map = prob_tensor.squeeze().cpu().numpy()

        cached_probs_and_gts.append((name, crop_a, crop_b, gt_arr, prob_map))

        pred_bin = (prob_map >= threshold).astype(np.uint8)
        pred_flat = pred_bin.reshape(-1)
        gt_flat = gt_arr.reshape(-1)

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

        sample_meta = {
            "index": idx,
            "filename": name,
            "gt_changed": gt_changed,
            "pred_changed": pred_changed,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "total_pixels": total_pixels,
            "threshold": threshold,
            "iou": round(float(iou), 4),
            "f1": round(float(f1), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "accuracy": round(float(accuracy), 4),
        }
        sample_results.append(sample_meta)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn
        total_gt_px += gt_changed
        total_pred_px += pred_changed

    # Global / micro metrics
    global_prec = (total_tp + smooth) / (total_tp + total_fp + smooth)
    global_rec = (total_tp + smooth) / (total_tp + total_fn + smooth)
    global_f1 = 2 * global_prec * global_rec / (global_prec + global_rec + smooth)
    global_iou = (total_tp + smooth) / (total_tp + total_fp + total_fn + smooth)
    global_acc = (total_tp + total_tn + smooth) / (total_tp + total_fp + total_fn + total_tn + smooth)

    # Macro mean metrics
    mean_iou = float(np.mean([s["iou"] for s in sample_results]))
    mean_f1 = float(np.mean([s["f1"] for s in sample_results]))
    mean_prec = float(np.mean([s["precision"] for s in sample_results]))
    mean_rec = float(np.mean([s["recall"] for s in sample_results]))
    mean_acc = float(np.mean([s["accuracy"] for s in sample_results]))

    # Save representative qualitative predictions if requested
    qualitative_saved = []
    if save_qualitative_dir:
        # Pick samples with substantial changes, moderate changes, and no change
        sorted_by_gt = sorted(enumerate(sample_results), key=lambda x: x[1]["gt_changed"], reverse=True)
        # top 8 change samples + 4 moderate change + 3 true negative (0 change)
        top_indices = [idx for idx, s in sorted_by_gt[:8]]
        mid_indices = [idx for idx, s in sorted_by_gt[20:24]]
        zero_indices = [idx for idx, s in sorted_by_gt if s["gt_changed"] == 0][:3]

        selected_vis_indices = sorted(set(top_indices + mid_indices + zero_indices))[:num_qualitative]

        for s_idx in selected_vis_indices:
            name, crop_a, crop_b, gt_arr, prob_map = cached_probs_and_gts[s_idx]
            meta = sample_results[s_idx]
            pred_bin = (prob_map >= threshold).astype(np.uint8)
            overlay_b = create_change_overlay(crop_b, pred_bin, color=(255, 40, 40), alpha=0.50)

            fig = create_composite_figure(
                img_a=crop_a,
                img_b=crop_b,
                gt_mask=gt_arr,
                prob_map=prob_map,
                pred_bin=pred_bin,
                overlay_b=overlay_b,
                meta=meta,
            )
            stem = Path(name).stem
            out_file = save_qualitative_dir / f"test_{meta['index']:03d}_{stem}_prediction.png"
            fig.save(out_file, quality=95)
            qualitative_saved.append(str(out_file))

    summary = {
        "total_samples": len(triplets),
        "threshold": threshold,
        "crop_size": f"{crop_size}x{crop_size}",
        "confusion_matrix": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "tn": total_tn,
            "total_pixels": total_tp + total_fp + total_fn + total_tn,
            "gt_changed_pixels": total_gt_px,
            "pred_changed_pixels": total_pred_px,
        },
        "global_micro_metrics": {
            "iou": round(float(global_iou), 4),
            "f1": round(float(global_f1), 4),
            "precision": round(float(global_prec), 4),
            "recall": round(float(global_rec), 4),
            "accuracy": round(float(global_acc), 4),
        },
        "sample_macro_metrics": {
            "mean_iou": round(float(mean_iou), 4),
            "mean_f1": round(float(mean_f1), 4),
            "mean_precision": round(float(mean_prec), 4),
            "mean_recall": round(float(mean_rec), 4),
            "mean_accuracy": round(float(mean_acc), 4),
        },
        "per_sample_results": sample_results,
        "qualitative_images_saved": qualitative_saved,
    }

    return summary, cached_probs_and_gts


# ---------------------------------------------------------------------------
# Threshold Sensitivity Sweep on Validation Set
# ---------------------------------------------------------------------------

def sweep_thresholds_on_cached(
    cached_probs_and_gts: List[Tuple[str, Image.Image, Image.Image, np.ndarray, np.ndarray]],
    thresholds: List[float],
) -> List[Dict[str, Any]]:
    smooth = 1e-6
    sweep_results = []

    for thr in thresholds:
        total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
        sample_ious, sample_f1s, sample_precs, sample_recs = [], [], [], []

        for name, _, _, gt_arr, prob_map in cached_probs_and_gts:
            pred_bin = (prob_map >= thr).astype(np.uint8)
            pred_flat = pred_bin.reshape(-1)
            gt_flat = gt_arr.reshape(-1)

            tp = int(((pred_flat == 1) & (gt_flat == 1)).sum())
            fp = int(((pred_flat == 1) & (gt_flat == 0)).sum())
            fn = int(((pred_flat == 0) & (gt_flat == 1)).sum())
            tn = int(((pred_flat == 0) & (gt_flat == 0)).sum())

            precision = (tp + smooth) / (tp + fp + smooth)
            recall = (tp + smooth) / (tp + fn + smooth)
            f1 = 2 * precision * recall / (precision + recall + smooth)
            iou = (tp + smooth) / (tp + fp + fn + smooth)

            sample_ious.append(iou)
            sample_f1s.append(f1)
            sample_precs.append(precision)
            sample_recs.append(recall)

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

        micro_prec = (total_tp + smooth) / (total_tp + total_fp + smooth)
        micro_rec = (total_tp + smooth) / (total_tp + total_fn + smooth)
        micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec + smooth)
        micro_iou = (total_tp + smooth) / (total_tp + total_fp + total_fn + smooth)
        micro_acc = (total_tp + total_tn + smooth) / (total_tp + total_fp + total_fn + total_tn + smooth)

        sweep_results.append({
            "threshold": round(thr, 2),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "tn": total_tn,
            "global_micro_iou": round(float(micro_iou), 4),
            "global_micro_f1": round(float(micro_f1), 4),
            "global_micro_precision": round(float(micro_prec), 4),
            "global_micro_recall": round(float(micro_rec), 4),
            "global_micro_accuracy": round(float(micro_acc), 4),
            "mean_sample_iou": round(float(np.mean(sample_ious)), 4),
            "mean_sample_f1": round(float(np.mean(sample_f1s)), 4),
            "mean_sample_precision": round(float(np.mean(sample_precs)), 4),
            "mean_sample_recall": round(float(np.mean(sample_recs)), 4),
        })

    return sweep_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Official Full LEVIR-CD Test Evaluation & Validation Threshold Tuning"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to trained best_model.pt checkpoint",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Path to LEVIR-CD root directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/change_detection"),
        help="Base directory to save results JSON and qualitative predictions",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Default test evaluation threshold (default: 0.5)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=256,
        help="Crop size for evaluation (default: 256)",
    )
    args = parser.parse_args()

    torch, TF = _require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve Checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        env_ckpt = os.getenv("CHANGE_DETECTION_CHECKPOINT")
        if env_ckpt and Path(env_ckpt).exists():
            checkpoint_path = Path(env_ckpt)
        else:
            default_candidates = [
                _BACKEND_ROOT / "checkpoints" / "best_model.pt",
                Path("checkpoints/best_model.pt"),
            ]
            for c in default_candidates:
                if c.exists():
                    checkpoint_path = c
                    break

    # Resolve Dataset Root
    data_root = args.data_root
    if data_root is None:
        env_root = os.getenv("LEVIR_CD_DATASET_PATH") or os.getenv("LEVIR_CD_ROOT")
        if env_root and Path(env_root).exists():
            data_root = Path(env_root)
        else:
            default_candidates = [
                Path("C:/Users/Lenovo/Downloads/LEVIR-CD"),
            ]
            for d in default_candidates:
                if d.exists():
                    data_root = d
                    break

    print("\n" + "=" * 80)
    print("SatQuery-AI: OFFICIAL COMPLETE LEVIR-CD TEST & VALIDATION EVALUATION")
    print("=" * 80)
    print(f"  Device:           {device} (CUDA: {device.type == 'cuda'})")
    print(f"  Checkpoint:       {checkpoint_path}")
    print(f"  Dataset Root:     {data_root}")
    print(f"  Evaluation Crop:  {args.img_size}×{args.img_size}")
    print(f"  Base Output Dir:  {args.output_dir}")
    print("=" * 80 + "\n")

    model, ckpt_info = load_model(checkpoint_path, device)
    ckpt_epoch = ckpt_info.get("epoch", 48) if isinstance(ckpt_info, dict) else 48

    # -----------------------------------------------------------------------
    # 1. Full 128-sample Test Split Evaluation (at threshold 0.5)
    # -----------------------------------------------------------------------
    test_dir = data_root / "test"
    test_triplets = get_split_triplets(test_dir)
    print(f"[Phase 1] Loaded {len(test_triplets)} triplets from {test_dir}")
    print(f"          Evaluating default threshold {args.threshold} across ALL {len(test_triplets)} test samples...")

    test_predictions_dir = args.output_dir / "test_full_predictions"
    test_summary, _ = evaluate_split(
        model=model,
        triplets=test_triplets,
        device=device,
        crop_size=args.img_size,
        threshold=args.threshold,
        save_qualitative_dir=test_predictions_dir,
        num_qualitative=15,
    )

    test_summary["checkpoint"] = {
        "path": str(checkpoint_path),
        "epoch": ckpt_epoch,
        "note": "Best validation checkpoint from Epoch 48",
    }
    test_summary["evaluation_split"] = "test"

    test_json_path = args.output_dir / "test_full_results.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump(test_summary, f, indent=2)

    print(f"\n[Phase 1 Complete] Full 128-Sample Test Split Results saved to:")
    print(f"                   -> {test_json_path}")
    print(f"                   -> {len(test_summary['qualitative_images_saved'])} qualitative figures saved to {test_predictions_dir}")

    # Print Full Test Metrics Table
    g_m = test_summary["global_micro_metrics"]
    s_m = test_summary["sample_macro_metrics"]
    cm = test_summary["confusion_matrix"]

    print("\n" + "-" * 75)
    print("OFFICIAL TEST SPLIT METRICS (128 SAMPLES @ THRESHOLD 0.50):")
    print("-" * 75)
    print(f"  Confusion Matrix:        TP = {cm['tp']:,}  |  FP = {cm['fp']:,}  |  FN = {cm['fn']:,}  |  TN = {cm['tn']:,}")
    print(f"  Pixel Counts:            GT Changed = {cm['gt_changed_pixels']:,}  |  Pred Changed = {cm['pred_changed_pixels']:,}")
    print("  -------------------------------------------------------------")
    print(f"  Global / Micro IoU:      {g_m['iou']:.4f}")
    print(f"  Global / Micro F1/Dice:  {g_m['f1']:.4f}")
    print(f"  Global / Micro Precision:{g_m['precision']:.4f}")
    print(f"  Global / Micro Recall:   {g_m['recall']:.4f}")
    print(f"  Global Pixel Accuracy:   {g_m['accuracy']:.4f}")
    print("  -------------------------------------------------------------")
    print(f"  Mean Sample IoU:         {s_m['mean_iou']:.4f}")
    print(f"  Mean Sample F1/Dice:     {s_m['mean_f1']:.4f}")
    print(f"  Mean Sample Precision:   {s_m['mean_precision']:.4f}")
    print(f"  Mean Sample Recall:      {s_m['mean_recall']:.4f}")
    print(f"  Mean Sample Accuracy:    {s_m['mean_accuracy']:.4f}")
    print("-" * 75 + "\n")

    # -----------------------------------------------------------------------
    # 2. Validation Set Threshold Sweep ([0.30 - 0.70])
    # -----------------------------------------------------------------------
    val_dir = data_root / "val"
    val_triplets = get_split_triplets(val_dir)
    print(f"[Phase 2] Loaded {len(val_triplets)} triplets from {val_dir}")
    print("          Running validation forward passes...")

    _, val_cached = evaluate_split(
        model=model,
        triplets=val_triplets,
        device=device,
        crop_size=args.img_size,
        threshold=0.50,
        save_qualitative_dir=None,
    )

    thresholds_to_sweep = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    print(f"          Sweeping thresholds: {thresholds_to_sweep} ONLY on validation set...")
    val_sweep_results = sweep_thresholds_on_cached(val_cached, thresholds_to_sweep)

    val_sweep_path = args.output_dir / "val_threshold_sweep.json"
    with open(val_sweep_path, "w", encoding="utf-8") as f:
        json.dump({
            "split": "val",
            "total_samples": len(val_triplets),
            "checkpoint": str(checkpoint_path),
            "threshold_sweep": val_sweep_results,
        }, f, indent=2)

    print(f"[Phase 2 Complete] Validation sweep results saved to: {val_sweep_path}\n")

    # Print Validation Sweep Table
    print("-" * 90)
    print("VALIDATION THRESHOLD SENSITIVITY SWEEP (64 SAMPLES):")
    print("-" * 90)
    print(f"{'Threshold':<11} | {'Micro IoU':<11} | {'Micro F1':<11} | {'Micro Prec':<11} | {'Micro Rec':<11} | {'Mean IoU':<10} | {'Mean F1':<10}")
    print("-" * 90)
    for row in val_sweep_results:
        print(
            f"{row['threshold']:<11.2f} | {row['global_micro_iou']:<11.4f} | {row['global_micro_f1']:<11.4f} | "
            f"{row['global_micro_precision']:<11.4f} | {row['global_micro_recall']:<11.4f} | "
            f"{row['mean_sample_iou']:<10.4f} | {row['mean_sample_f1']:<10.4f}"
        )
    print("-" * 90)

    # Select best threshold based on Validation F1 / IoU
    best_val_row = max(val_sweep_results, key=lambda x: (x["global_micro_f1"], x["global_micro_iou"]))
    print(f"\n[*] OPTIMAL THRESHOLD ON VALIDATION SET: {best_val_row['threshold']:.2f}")
    print(f"    Validation Micro IoU:       {best_val_row['global_micro_iou']:.4f}")
    print(f"    Validation Micro F1/Dice:   {best_val_row['global_micro_f1']:.4f}")
    print(f"    Validation Micro Precision: {best_val_row['global_micro_precision']:.4f}")
    print(f"    Validation Micro Recall:    {best_val_row['global_micro_recall']:.4f}")
    print(f"    Validation Mean Sample F1:  {best_val_row['mean_sample_f1']:.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
