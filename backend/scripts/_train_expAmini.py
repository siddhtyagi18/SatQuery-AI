#!/usr/bin/env python3
"""
Experiment A-mini: FAST HYPERPARAMETER PROBE
Resume from Epoch 48 baseline, train ~250 mini-batches (~3.4 epochs equivalent).
Use improved settings:
  - combo/enhanced_hybrid loss (BCE 0.25 + Dice 0.30 + Tversky α=0.3/β=0.7 0.30 + Focal γ=2 0.15), pos_weight=3.0
  - AdamW(lr=8e-5, weight_decay=2.5e-4) — lower LR for fine-tuning
  - ReduceLROnPlateau scheduler (factor=0.5, patience=2)
  - Eval val every 50 batches; save best via composite score (F1 + IoU)/2
"""
from __future__ import annotations

import json, os, sys, time, random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

def flush(*args, **kwargs):
    print(*args, **kwargs, flush=True)

import builtins
builtins.print = lambda *a, **k: flush(*a, **k)

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

# Load env for DATA_ROOT
from dotenv import load_dotenv
load_dotenv(".env")

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tud
import torchvision.transforms.functional as TF
from PIL import Image

# ---------- Import project modules ----------
from app.services.models.siamese_unet import (
    SiameseUNet, enhanced_hybrid_loss,
)
from app.services.datasets.levir_cd import LEVIRDataset

# ---------- Inline helpers (originally from train_change_detector.py) ----------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(model, optimizer, epoch, metrics, checkpoint_dir, name="best_model.pt", scheduler=None):
    checkpoint_dir = Path(checkpoint_dir); checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "epoch": epoch, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "metrics": metrics,
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, checkpoint_dir / name)

# ---------- Config ----------
SEED = 1337
set_seed(SEED)
device = torch.device("cpu")
print(f"[config] torch threads: {torch.get_num_threads()}")

DATA_ROOT = Path(os.getenv("LEVIR_CD_DATASET_PATH", "./data/results/LEVIR-CD")).resolve()
CHECKPOINT_DIR = BACKEND / "checkpoints" / "experiment_A_mini"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[config] DATA_ROOT={DATA_ROOT} exists={DATA_ROOT.exists()}")
print(f"[config] CHECKPOINT_DIR={CHECKPOINT_DIR}")

# Model / hyperparams
BASELINE_CKPT = BACKEND / "checkpoints" / "best_model.pt"
N_UPDATES = 250
BATCH_SIZE = 6
LR = 8e-5
WD = 2.5e-4
POS_WEIGHT = 3.0

# Load baseline model
ckpt0 = torch.load(BASELINE_CKPT, map_location=device, weights_only=True)
sd = ckpt0["model_state_dict"] if "model_state_dict" in ckpt0 else ckpt0
base_filters = sd["enc1.block.0.block.0.weight"].shape[0]
model = SiameseUNet(in_channels=6, base_filters=base_filters).to(device)
miss, unexp = model.load_state_dict(sd, strict=False)
print(f"[model] base_filters={base_filters} params={sum(p.numel() for p in model.parameters()):,} baseline_epoch={ckpt0.get('epoch','?')}")
print(f"  load: missing={len(miss)} unexpected={len(unexp)}")
print(f"  baseline val metrics: {ckpt0.get('metrics')}")

# Loss
def loss_fn(logits, target):
    return enhanced_hybrid_loss(
        logits, target,
        bce_weight=0.25, dice_weight=0.30, tversky_weight=0.30, focal_weight=0.15,
        pos_weight=torch.as_tensor(POS_WEIGHT, dtype=torch.float32, device=logits.device),
        tversky_alpha=0.3, tversky_beta=0.7,
    )

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode="max", factor=0.5, patience=2, min_lr=1e-7)

# Datasets
train_ds = LEVIRDataset(root=DATA_ROOT, split="train", img_size=256, augment=True)
val_ds   = LEVIRDataset(root=DATA_ROOT, split="val",   img_size=256, augment=False)
g = torch.Generator(); g.manual_seed(SEED)
train_loader = tud.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, generator=g)
val_loader   = tud.DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"[data] train ds={len(train_ds)} batches_per_epoch={len(train_loader)} val ds={len(val_ds)}")

# ---------- Val eval function (threshold sweep on val at start/end + default thr every 50 batches) ----------
SMOOTH = 1e-6

def batch_to_tensor(batch):
    # LEVIRDataset.__getitem__ returns Tuple[Tensor(3,H,W), Tensor(3,H,W), Tensor(1,H,W)]
    # Model expects 6-channel input
    if isinstance(batch, dict):
        return batch["image"].to(device), batch["label"].to(device)
    t_a, t_b, t_l = batch
    inp = torch.cat([t_a.to(device), t_b.to(device)], dim=1)
    return inp, t_l.to(device)

@torch.no_grad()
def eval_val(tag: str, thresholds=(0.3, 0.35, 0.4, 0.45, 0.5, 0.6)):
    model.eval()
    # cache (prob, gt) for each val sample
    cached = []
    val_losses = []
    n_batches = len(val_loader)
    for bi, b in enumerate(val_loader, 1):
        inp, gt = batch_to_tensor(b)
        logits = model(inp)
        val_losses.append(loss_fn(logits, gt).item())
        prob = torch.sigmoid(logits).squeeze(1).cpu().numpy()  # (B, H, W)
        gt_np = (gt.squeeze(1).cpu().numpy() > 0.5).astype(np.uint8)
        for i in range(prob.shape[0]):
            cached.append((prob[i], gt_np[i]))
        if bi % 2 == 0 or bi == n_batches:
            flush(f"    val eval [{bi}/{n_batches}] batches done", end="\r")
    if val_losses:
        pass
    # Threshold sweep
    best = None
    rows = []
    for thr in thresholds:
        tpt=fpt=fnt=tnt=0
        for p, g in cached:
            pb = (p >= thr).astype(np.uint8).ravel(); gf = g.ravel()
            tpt += int(((pb==1)&(gf==1)).sum())
            fpt += int(((pb==1)&(gf==0)).sum())
            fnt += int(((pb==0)&(gf==1)).sum())
            tnt += int(((pb==0)&(gf==0)).sum())
        prec = (tpt+SMOOTH)/(tpt+fpt+SMOOTH); rec = (tpt+SMOOTH)/(tpt+fnt+SMOOTH)
        f1 = 2*prec*rec/(prec+rec+SMOOTH); iou=(tpt+SMOOTH)/(tpt+fpt+fnt+SMOOTH)
        acc=(tpt+tnt+SMOOTH)/(tpt+fpt+fnt+tnt+SMOOTH)
        m = dict(thr=thr, iou=round(float(iou),4), f1=round(float(f1),4),
                 precision=round(float(prec),4), recall=round(float(rec),4),
                 accuracy=round(float(acc),4), tp=tpt, fp=fpt, fn=fnt, tn=tnt,
                 val_loss=round(float(np.mean(val_losses)),4))
        rows.append(m)
        if best is None or (m["f1"], m["iou"]) > (best["f1"], best["iou"]):
            best = m
    model.train()
    print(f"  [VAL {tag}] best_thr={best['thr']:.2f}  loss={best['val_loss']:.4f}  "
          f"IoU={best['iou']:.4f}  F1={best['f1']:.4f}  P={best['precision']:.4f}  R={best['recall']:.4f}")
    return best, rows

# ---------- Initial baseline val metrics (pre-training) ----------
print("\n=== VALIDATION BEFORE START (baseline frozen checkpoint metrics) ===")
init_best, init_rows = eval_val("initial")
for r in init_rows:
    print(f"    thr={r['thr']:.2f}  IoU={r['iou']:.4f} F1={r['f1']:.4f}")

# ---------- Training loop ----------
history = {
    "config": {
        "baseline_epoch": ckpt0.get("epoch"),
        "baseline_val_metrics": ckpt0.get("metrics"),
        "n_updates": N_UPDATES, "batch_size": BATCH_SIZE,
        "lr_start": LR, "weight_decay": WD, "pos_weight": POS_WEIGHT,
        "loss": "enhanced_hybrid 0.25BCE+0.30Dice+0.30Tversky(0.3,0.7)+0.15Focal(γ=2)",
        "scheduler": "ReduceLROnPlateau factor=0.5 patience=2",
        "device": str(device), "seed": SEED,
    },
    "initial_val_best": init_best, "initial_val_rows": init_rows,
    "val_snapshots": [], "batch_loss_log": [],
}

t0 = time.time()
best_score = 0.0  # composite (F1+IoU)/2
train_iter = iter(train_loader)
model.train()
last_20_losses = []
for step in range(1, N_UPDATES + 1):
    try:
        batch = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        batch = next(train_iter)
    inp, gt = batch_to_tensor(batch)

    opt.zero_grad()
    logits = model(inp)
    loss = loss_fn(logits, gt)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    lv = float(loss.item())
    last_20_losses.append(lv)
    history["batch_loss_log"].append(lv)

    if step == 1 or step % 20 == 0:
        lr_now = opt.param_groups[0]["lr"]
        if step == 1:
            avgl = lv
        else:
            avgl = float(np.mean(last_20_losses[-20:]))
        eta_sec = (N_UPDATES - step) * (time.time() - t0) / step
        print(f"[step {step:4d}/{N_UPDATES}] loss(avg20)={avgl:.4f}  lr={lr_now:.2e}  "
              f"t_elapsed={(time.time()-t0)/60:.1f}min  ETA={eta_sec/60:.1f}min")

    # Val snapshot every 50 batches / start / end
    if step == 1 or step % 50 == 0 or step == N_UPDATES:
        vb, vrows = eval_val(f"@step={step}")
        vb["step"] = step
        vb["snapshot_rows"] = vrows
        vb["lr_at_eval"] = opt.param_groups[0]["lr"]
        vb["train_loss_avg20"] = round(float(np.mean(last_20_losses[-20:])), 4)
        history["val_snapshots"].append(vb)
        score = (vb["f1"] + vb["iou"]) / 2
        print(f"    composite(F1+IoU)/2 = {score:.4f}")
        # scheduler step based on composite
        sched.step(score)

        if score > best_score:
            best_score = score
            print(f"    *** NEW BEST checkpoint saved (score={score:.4f}) -> best_model.pt")
            save_checkpoint(model=model, optimizer=opt, epoch=48 + step // max(1, len(train_loader)),
                            metrics={k: v for k, v in vb.items() if k not in ("snapshot_rows",)},
                            checkpoint_dir=CHECKPOINT_DIR, name="best_model.pt", scheduler=sched)
        # Always save last
        save_checkpoint(model=model, optimizer=opt, epoch=48 + step // max(1, len(train_loader)),
                        metrics={k: v for k, v in vb.items() if k not in ("snapshot_rows",)},
                        checkpoint_dir=CHECKPOINT_DIR, name="last_model.pt", scheduler=sched)

total_t = time.time() - t0
history["total_time_s"] = round(total_t, 1)
history["final_val_best"] = history["val_snapshots"][-1]
# Find overall best snapshot
best_snap = max(history["val_snapshots"], key=lambda s: (s["f1"], s["iou"]))
history["overall_best_snapshot"] = {k: v for k, v in best_snap.items() if k != "snapshot_rows"}

with open(CHECKPOINT_DIR / "training_history.json", "w") as f:
    json.dump(history, f, indent=2)

print("\n================== Experiment A-mini FINISHED ==================")
print(f"  Total time: {total_t/60:.2f} min ({total_t:.0f} s)  |  Updates: {N_UPDATES}")
print(f"  Baseline VAL before: F1={init_best['f1']:.4f}  IoU={init_best['iou']:.4f}")
print(f"  Best   VAL achieved: F1={best_snap['f1']:.4f}  IoU={best_snap['iou']:.4f} @ step={best_snap['step']}")
print(f"  Final  VAL (last)  : F1={history['val_snapshots'][-1]['f1']:.4f}  IoU={history['val_snapshots'][-1]['iou']:.4f}")
print(f"  Delta F1  = {(best_snap['f1'] - init_best['f1'])*100:+.2f} pp")
print(f"  Delta IoU = {(best_snap['iou'] - init_best['iou'])*100:+.2f} pp")
print(f"  Best checkpoint: {CHECKPOINT_DIR / 'best_model.pt'}")
print("==================================================================")
