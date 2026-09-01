#!/usr/bin/env python3
"""
backend/scripts/train_change_detector.py
=========================================
Training, evaluation, and checkpointing script for the SiameseUNet change detector.

Usage examples
--------------
# CPU smoke test (verifies code runs, 2 mini-batches, tiny 64×64 images):
python scripts/train_change_detector.py --smoke-test --data-root C:/path/to/LEVIR-CD

# Full training (do NOT run locally without GPU — set this up on cloud/Colab):
python scripts/train_change_detector.py \\
    --data-root C:/path/to/LEVIR-CD \\
    --epochs 50 \\
    --batch-size 4 \\
    --img-size 256 \\
    --checkpoint-dir ./checkpoints

# Resume training from a checkpoint:
python scripts/train_change_detector.py \\
    --data-root C:/path/to/LEVIR-CD \\
    --resume ./checkpoints/last_model.pt

# Evaluate only (no training):
python scripts/train_change_detector.py \\
    --data-root C:/path/to/LEVIR-CD \\
    --resume ./checkpoints/best_model.pt \\
    --eval-only

Important Notes
---------------
- Without a GPU, full training on LEVIR-CD (445 train + 64 val × 1024×1024) is
  extremely slow. Use the --smoke-test flag for local CPU verification only.
- This script is designed to be run on a GPU machine (Colab, Kaggle, cloud VM).
- The trained checkpoint can then be copied to the local machine and configured via
  CHANGE_DETECTION_CHECKPOINT in .env for real inference.

Hardware requirements for real training
----------------------------------------
- GPU: NVIDIA with ≥6GB VRAM recommended (T4/V100/A100 on Colab/cloud)
- RAM: ≥8GB
- Disk: LEVIR-CD is ~8GB extracted

Metrics computed
----------------
- IoU (Intersection over Union) — primary metric for change detection
- F1 / Dice coefficient
- Precision and Recall
- Pixel-level accuracy
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Resolve backend package (allow running as top-level script)
# ---------------------------------------------------------------------------
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_ROOT))


def _require_torch():
    """Check torch is available and import it."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        return torch, nn, DataLoader
    except ImportError as e:
        print(f"ERROR: torch is not installed: {e}")
        print("Install via: pip install torch torchvision")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class ChangeDetectionMetrics:
    """Pixel-level metrics for binary change detection."""
    iou: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    accuracy: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0


def compute_metrics(
    pred_masks: "torch.Tensor",
    true_masks: "torch.Tensor",
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> ChangeDetectionMetrics:
    """
    Compute pixel-level change detection metrics.

    Parameters
    ----------
    pred_masks : (B, 1, H, W) probability map (after sigmoid), values in [0, 1].
    true_masks : (B, 1, H, W) binary ground truth {0, 1}.
    threshold  : Probability threshold for binary prediction.
    smooth     : Numerical stability constant.

    Returns
    -------
    ChangeDetectionMetrics with IoU, F1, Precision, Recall, Accuracy.
    """
    import torch

    pred_bin = (pred_masks >= threshold).float()
    pred_flat = pred_bin.view(-1)
    true_flat = true_masks.view(-1)

    tp = int((pred_flat * true_flat).sum().item())
    fp = int((pred_flat * (1 - true_flat)).sum().item())
    fn = int(((1 - pred_flat) * true_flat).sum().item())
    tn = int(((1 - pred_flat) * (1 - true_flat)).sum().item())

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1 = 2 * precision * recall / (precision + recall + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    accuracy = (tp + tn + smooth) / (tp + fp + fn + tn + smooth)

    return ChangeDetectionMetrics(
        iou=round(iou, 4),
        f1=round(f1, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        accuracy=round(accuracy, 4),
        tp=tp, fp=fp, fn=fn, tn=tn,
    )


def aggregate_metrics(metrics_list: List[ChangeDetectionMetrics]) -> ChangeDetectionMetrics:
    """Average a list of metrics over batches."""
    if not metrics_list:
        return ChangeDetectionMetrics()
    n = len(metrics_list)
    return ChangeDetectionMetrics(
        iou=round(sum(m.iou for m in metrics_list) / n, 4),
        f1=round(sum(m.f1 for m in metrics_list) / n, 4),
        precision=round(sum(m.precision for m in metrics_list) / n, 4),
        recall=round(sum(m.recall for m in metrics_list) / n, 4),
        accuracy=round(sum(m.accuracy for m in metrics_list) / n, 4),
        tp=sum(m.tp for m in metrics_list),
        fp=sum(m.fp for m in metrics_list),
        fn=sum(m.fn for m in metrics_list),
        tn=sum(m.tn for m in metrics_list),
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    metrics: ChangeDetectionMetrics,
    checkpoint_dir: Path,
    name: str = "last_model.pt",
    scheduler=None,
) -> Path:
    """Save a training checkpoint with model weights, optimizer state, epoch, metrics, and scheduler state."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / name
    import torch
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": asdict(metrics),
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, path)
    return path


def load_checkpoint_for_training(
    model, optimizer, path: Path, scheduler=None
) -> Tuple[int, ChangeDetectionMetrics]:
    """
    Load a checkpoint for resuming training.

    Returns (start_epoch, best_metrics).
    """
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception as e:
            print(f"[checkpoint] Notice: could not restore scheduler state: {e}")
    epoch = ckpt.get("epoch", 0)
    metrics_dict = ckpt.get("metrics", {})
    best_metrics = ChangeDetectionMetrics(**{
        k: v for k, v in metrics_dict.items()
        if k in ChangeDetectionMetrics.__dataclass_fields__
    })
    print(f"[checkpoint] Resumed from epoch {epoch}. Metrics: {asdict(best_metrics)}")
    return epoch + 1, best_metrics


# ---------------------------------------------------------------------------
# Training / evaluation loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    loss_fn,
    epoch: int,
    max_batches: Optional[int] = None,
) -> Tuple[float, ChangeDetectionMetrics]:
    """
    Train for one epoch.

    Returns (average_loss, average_metrics_over_epoch).
    """
    import torch

    model.train()
    total_loss = 0.0
    batch_metrics: List[ChangeDetectionMetrics] = []
    n_batches = 0

    for batch_idx, (img_a, img_b, label) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        img_a = img_a.to(device)
        img_b = img_b.to(device)
        label = label.to(device)

        # Concatenate images along channel dim → 6-channel input
        inp = torch.cat([img_a, img_b], dim=1)  # (B, 6, H, W)

        optimizer.zero_grad()
        logits = model(inp)
        loss = loss_fn(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        with torch.no_grad():
            prob = torch.sigmoid(logits)
            m = compute_metrics(prob, label)
            batch_metrics.append(m)

        if batch_idx % 20 == 0:
            print(
                f"  [Epoch {epoch}] batch {batch_idx}/{len(loader)} "
                f"loss={loss.item():.4f} iou={m.iou:.4f} f1={m.f1:.4f}"
            )

    avg_loss = total_loss / max(n_batches, 1)
    avg_metrics = aggregate_metrics(batch_metrics)
    return avg_loss, avg_metrics


def evaluate(
    model,
    loader,
    device,
    loss_fn,
    max_batches: Optional[int] = None,
) -> Tuple[float, ChangeDetectionMetrics]:
    """
    Evaluate model on a DataLoader.

    Returns (average_loss, average_metrics).
    """
    import torch

    model.eval()
    total_loss = 0.0
    batch_metrics: List[ChangeDetectionMetrics] = []
    n_batches = 0

    with torch.no_grad():
        for batch_idx, (img_a, img_b, label) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            img_a = img_a.to(device)
            img_b = img_b.to(device)
            label = label.to(device)

            inp = torch.cat([img_a, img_b], dim=1)
            logits = model(inp)
            loss = loss_fn(logits, label)

            total_loss += loss.item()
            n_batches += 1

            prob = torch.sigmoid(logits)
            m = compute_metrics(prob, label)
            batch_metrics.append(m)

    avg_loss = total_loss / max(n_batches, 1)
    avg_metrics = aggregate_metrics(batch_metrics)
    return avg_loss, avg_metrics


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def run_smoke_test(data_root: Path, img_size: int = 64) -> bool:
    """
    Run a tiny CPU smoke test to verify the full training pipeline works.

    Creates a minimal synthetic dataset (4 random images, 64×64) and runs
    2 training batches + 1 validation batch. Does NOT use real LEVIR-CD images.

    Returns True on success, False on failure.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    print("\n" + "=" * 60)
    print("SMOKE TEST: Verifying training pipeline on CPU with synthetic data")
    print("=" * 60)

    try:
        from app.services.models.siamese_unet import SiameseUNet, combined_loss

        # Create 4 synthetic samples (64×64)
        B, C, H, W = 4, 3, img_size, img_size
        img_a = torch.rand(B, C, H, W)
        img_b = torch.rand(B, C, H, W)
        labels = (torch.rand(B, 1, H, W) > 0.5).float()

        dataset = TensorDataset(img_a, img_b, labels)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)

        # Model + optimiser
        model = SiameseUNet(in_channels=6, base_filters=16)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel: SiameseUNet | trainable parameters: {total_params:,}")

        device = torch.device("cpu")
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # 2 training batches
        model.train()
        for i, (a, b, lbl) in enumerate(loader):
            a, b, lbl = a.to(device), b.to(device), lbl.to(device)
            inp = torch.cat([a, b], dim=1)
            logits = model(inp)
            loss = combined_loss(logits, lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prob = torch.sigmoid(logits)
            m = compute_metrics(prob, lbl)
            print(f"  Train batch {i+1}/2 | loss={loss.item():.4f} | iou={m.iou:.4f} f1={m.f1:.4f}")

        # 1 validation batch
        model.eval()
        with torch.no_grad():
            a, b, lbl = next(iter(loader))
            inp = torch.cat([a, b], dim=1)
            logits = model(inp)
            prob = torch.sigmoid(logits)
            m = compute_metrics(prob, lbl)
            print(f"  Val   batch 1/1 | iou={m.iou:.4f} f1={m.f1:.4f} precision={m.precision:.4f} recall={m.recall:.4f}")

        # Verify output shape
        assert logits.shape == (2, 1, img_size, img_size), f"Unexpected output shape: {logits.shape}"
        assert prob.min() >= 0.0 and prob.max() <= 1.0, "Probability out of [0, 1] range"

        print("\n[PASS] SMOKE TEST PASSED - training pipeline is functional on CPU.")
        print("       Next step: run full training on GPU/cloud with real LEVIR-CD data.")
        return True

    except Exception as e:
        import traceback
        print(f"\n[FAIL] SMOKE TEST FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SatQuery-AI SiameseUNet Change Detector Training Script"
    )
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Path to LEVIR-CD root directory")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("./checkpoints"),
                        help="Directory to save checkpoints (default: ./checkpoints)")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size (default: 4; use 2 if low VRAM)")
    parser.add_argument("--img-size", type=int, default=256,
                        help="Random crop size for training images (default: 256)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3; Adam optimiser)")
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader workers (default: 0 = main thread, safe on Windows)")
    parser.add_argument("--base-filters", type=int, default=16,
                        help="Base filter count for SiameseUNet (default: 16 → ~490K params)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a tiny CPU smoke test instead of full training")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; only evaluate the resumed checkpoint on val/test")
    parser.add_argument("--log-file", type=Path, default=None,
                        help="Optional path to write training log as JSON")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve LEVIR-CD dataset root
    # ------------------------------------------------------------------
    data_root = args.data_root
    if data_root is None and not args.smoke_test:
        env_val = os.getenv("LEVIR_CD_DATASET_PATH") or os.getenv("LEVIR_CD_ROOT")
        if env_val:
            data_root = Path(env_val)

    # ------------------------------------------------------------------
    # Smoke test mode (CPU-safe, no real dataset needed)
    # ------------------------------------------------------------------
    if args.smoke_test:
        success = run_smoke_test(data_root or Path("."), img_size=args.img_size)
        sys.exit(0 if success else 1)

    # ------------------------------------------------------------------
    # Full training / eval mode
    # ------------------------------------------------------------------
    if data_root is None:
        print("ERROR: LEVIR-CD data path not provided.")
        print("Please provide --data-root /path/to/LEVIR-CD or set LEVIR_CD_DATASET_PATH (or LEVIR_CD_ROOT) in your .env.")
        parser.print_help()
        sys.exit(1)

    torch, nn, DataLoader = _require_torch()
    from torch.utils.data import DataLoader as _DataLoader

    try:
        from app.services.datasets.levir_cd import LEVIRDataset
        from app.services.models.siamese_unet import SiameseUNet, combined_loss
    except ImportError as e:
        print(f"ERROR: Cannot import required modules: {e}")
        print("Run from the backend/ directory: python scripts/train_change_detector.py ...")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 60}")
    print(f"SatQuery-AI SiameseUNet Change Detector Training")
    print(f"{'=' * 60}")
    print(f"  Device:        {device}")
    print(f"  Data root:     {data_root}")
    print(f"  Checkpoint dir:{args.checkpoint_dir}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Image size:    {args.img_size}×{args.img_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Base filters:  {args.base_filters}")
    print()

    # Datasets
    print("Loading datasets...")
    train_ds = LEVIRDataset(data_root, split="train", img_size=args.img_size, augment=True)
    val_ds = LEVIRDataset(data_root, split="val", img_size=args.img_size, augment=False)
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val:   {len(val_ds)} samples")

    train_loader = _DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = _DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )

    # Model
    model = SiameseUNet(in_channels=6, base_filters=args.base_filters).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: SiameseUNet | trainable parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    loss_fn = lambda logits, target: combined_loss(logits, target, bce_weight=0.5, dice_weight=0.5)

    # Checkpoint resume
    start_epoch = 1
    best_iou = 0.0
    history: List[Dict] = []

    log_path = args.log_file or (args.checkpoint_dir / "training_log.json")
    if log_path.exists():
        try:
            with open(log_path, "r") as f:
                history = json.load(f)
            print(f"Loaded existing training log history ({len(history)} epochs) from {log_path}")
        except Exception:
            history = []

    if args.resume and args.resume.exists():
        start_epoch, best_metrics_ckpt = load_checkpoint_for_training(
            model, optimizer, args.resume, scheduler=scheduler
        )
        best_iou = best_metrics_ckpt.iou
        print(f"Resumed from {args.resume} at next epoch {start_epoch}, best_iou={best_iou:.4f}")

    # Eval only mode
    if args.eval_only:
        print("\nEval-only mode: running validation...")
        val_loss, val_metrics = evaluate(model, val_loader, device, loss_fn)
        print(f"Val   | loss={val_loss:.4f} | iou={val_metrics.iou:.4f} f1={val_metrics.f1:.4f} "
              f"precision={val_metrics.precision:.4f} recall={val_metrics.recall:.4f}")

        try:
            test_ds = LEVIRDataset(data_root, split="test", img_size=args.img_size, augment=False)
            test_loader = _DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
            test_loss, test_metrics = evaluate(model, test_loader, device, loss_fn)
            print(f"Test  | loss={test_loss:.4f} | iou={test_metrics.iou:.4f} f1={test_metrics.f1:.4f} "
                  f"precision={test_metrics.precision:.4f} recall={test_metrics.recall:.4f}")
        except Exception as e:
            print(f"Test eval skipped: {e}")
        return

    # Check if target epochs already reached
    if start_epoch > args.epochs:
        print(f"\n[INFO] Checkpoint is already at epoch {start_epoch - 1}, which has reached or exceeded the requested target --epochs {args.epochs}.")
        print(f"       To continue training further, specify a higher --epochs value (e.g. --epochs {start_epoch + 10} or --epochs 100).")
        return

    # Training loop
    print(f"\nStarting training from epoch {start_epoch} to {args.epochs}...")
    print("NOTE: Without GPU, each epoch on full LEVIR-CD (445 samples) will be slow.")
    print("      Recommend training on GPU/cloud (Colab, Kaggle, etc.).\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t_epoch_start = time.time()

        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, device, loss_fn, epoch)
        val_loss, val_metrics = evaluate(model, val_loader, device, loss_fn)

        t_epoch = time.time() - t_epoch_start
        scheduler.step(val_metrics.iou)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | {t_epoch:.1f}s | "
            f"Train loss={train_loss:.4f} iou={train_metrics.iou:.4f} f1={train_metrics.f1:.4f} | "
            f"Val   loss={val_loss:.4f}  iou={val_metrics.iou:.4f}  f1={val_metrics.f1:.4f}"
        )

        # Save last checkpoint every epoch
        save_checkpoint(
            model, optimizer, epoch, val_metrics, args.checkpoint_dir, "last_model.pt", scheduler=scheduler
        )

        # Save best checkpoint when val IoU improves
        if val_metrics.iou > best_iou:
            best_iou = val_metrics.iou
            best_path = save_checkpoint(
                model, optimizer, epoch, val_metrics, args.checkpoint_dir, "best_model.pt", scheduler=scheduler
            )
            print(f"  [*] New best IoU: {best_iou:.4f} -- checkpoint saved to {best_path}")

        # Record history
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_iou": train_metrics.iou,
            "train_f1": train_metrics.f1,
            "val_loss": round(val_loss, 4),
            "val_iou": val_metrics.iou,
            "val_f1": val_metrics.f1,
            "val_precision": val_metrics.precision,
            "val_recall": val_metrics.recall,
            "elapsed_s": round(t_epoch, 1),
        })

    print(f"\nTraining complete. Best Val IoU: {best_iou:.4f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print(f"\nTo use this checkpoint for inference, set in .env:")
    print(f"  CHANGE_DETECTION_CHECKPOINT={args.checkpoint_dir / 'best_model.pt'}")

    # Save training log
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(args.log_file, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Training log saved to: {args.log_file}")


if __name__ == "__main__":
    main()
