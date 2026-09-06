#!/usr/bin/env python3
"""Timing benchmark: train for 2 epochs with 32 samples to estimate per-epoch time."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
_BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND))
os.chdir(_BACKEND)

from dotenv import load_dotenv
load_dotenv(".env")

import torch
import json
from app.services.datasets.levir_cd import make_levir_loaders
from app.services.models.siamese_unet import SiameseUNet, build_loss_function, train_one_epoch, validate
from app.services.utils.checkpoints import save_checkpoint, load_checkpoint_for_training
from app.services.utils.seed import set_seed

set_seed(42)
device = torch.device("cpu")

DATA_ROOT = Path("./data/results/LEVIR-CD")
loaders = make_levir_loaders(
    root=DATA_ROOT, split="train",
    image_size=256, batch_size=4, num_workers=0,
    train_limit=32, val_limit=20, seed=42,
)
train_loader, val_loader = loaders

model = SiameseUNet(in_channels=6, base_filters=16).to(device)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

# Pretend to start at epoch 49 baseline (no actual resume needed for timing)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=2e-4)
loss_fn = build_loss_function(loss_type="combo", pos_weight=2.5)

TOTAL_START = time.time()
for ep in range(1, 3):
    t0 = time.time()
    train_metrics = train_one_epoch(model, train_loader, loss_fn, opt, device, epoch=ep, silent=True)
    t1 = time.time()
    val_metrics = validate(model, val_loader, device, epoch=ep, silent=True)
    t2 = time.time()
    print(f"[Epoch {ep}] Train {t1-t0:.1f}s loss={train_metrics['loss']:.4f} | Val {t2-t1:.1f}s IoU={val_metrics['iou']:.4f} F1={val_metrics['f1']:.4f}")

total = time.time() - TOTAL_START
print(f"\nTotal 2 epochs (train+val): {total:.1f}s => per-epoch ~ {total/2:.1f}s")
print(f"Estimated 27 epochs: {27 * (total/2) / 60:.1f} minutes")
