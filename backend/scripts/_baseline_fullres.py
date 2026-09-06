#!/usr/bin/env python3
"""Full-resolution baseline evaluation.
Evaluates baseline with:
  - Official 256x256 center crop (training protocol)
  - Larger 512x512 center crop
  - Full 1024x1024 (direct forward; memory safe given SiameseUNet is small)

Sweeps thresholds on VAL only, then applies best val threshold for each crop-size,
runs TEST once.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))
os.chdir(BACKEND_ROOT)

from dotenv import load_dotenv
load_dotenv(".env")

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

from app.services.models.siamese_unet import SiameseUNet

DATA_ROOT = Path(os.getenv("LEVIR_CD_DATASET_PATH", "./data/results/LEVIR-CD")).resolve()
CKPT_PATH = BACKEND_ROOT / "checkpoints" / "best_model.pt"
OUT_DIR = BACKEND_ROOT / "evaluation_results" / "baseline_fullres_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SMOOTH = 1e-6

device = torch.device("cpu")
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
base_filters = sd["enc1.block.0.block.0.weight"].shape[0]
model = SiameseUNet(in_channels=6, base_filters=base_filters)
model.load_state_dict(sd)
model.to(device)
model.eval()
print(f"Loaded Epoch={ckpt.get('epoch','?')} base_filters={base_filters} params={sum(p.numel() for p in model.parameters()):,}")


def matched_triplets(split):
    d = DATA_ROOT / split
    a = {f.name: f for f in (d / "A").glob("*.png")}
    b = {f.name: f for f in (d / "B").glob("*.png")}
    l = {f.name: f for f in (d / "label").glob("*.png")}
    return sorted([(n, a[n], b[n], l[n]) for n in set(a) & set(b) & set(l)])


val_trips = matched_triplets("val")
test_trips = matched_triplets("test")
print(f"Val matched: {len(val_trips)}")
print(f"Test matched: {len(test_trips)}")


def forward_and_bin(name_pa_pb_pl, crop=None):
    """Returns (prob_map, gt_mask, tp_fp_fn_tn @ 0.5 DEFAULT only).

    prob_map, gt_mask are at the evaluated resolution (full or crop).
    """
    name, pa, pb, pl = name_pa_pb_pl
    imgA = np.array(Image.open(pa).convert("RGB")).astype(np.uint8)
    imgB = np.array(Image.open(pb).convert("RGB")).astype(np.uint8)
    gt_mask = (np.array(Image.open(pl).convert("L")) > 127).astype(np.uint8)
    if crop:
        sz = crop
        H, W = gt_mask.shape
        i, j = (H - sz) // 2, (W - sz) // 2
        imgA = imgA[i:i+sz, j:j+sz]
        imgB = imgB[i:i+sz, j:j+sz]
        gt_mask = gt_mask[i:i+sz, j:j+sz]
    with torch.no_grad():
        ta = TF.to_tensor(Image.fromarray(imgA)).unsqueeze(0)
        tb = TF.to_tensor(Image.fromarray(imgB)).unsqueeze(0)
        inp = torch.cat([ta, tb], dim=1)
        H0, W0 = inp.shape[2:]
        pad_h = (16 - H0 % 16) % 16
        pad_w = (16 - W0 % 16) % 16
        if pad_h or pad_w:
            inp = F.pad(inp, (0, pad_w, 0, pad_h), mode="reflect")
        logits = model(inp)
        prob = torch.sigmoid(logits)[0, 0, :H0, :W0].cpu().numpy()
    pf = (prob >= 0.5).astype(np.uint8).ravel()
    gf = gt_mask.ravel()
    tp = int(((pf == 1) & (gf == 1)).sum())
    fp = int(((pf == 1) & (gf == 0)).sum())
    fn = int(((pf == 0) & (gf == 1)).sum())
    tn = int(((pf == 0) & (gf == 0)).sum())
    return prob.astype(np.float32), gt_mask, (tp, fp, fn, tn)


def metrics_from(tp, fp, fn, tn):
    p = (tp + SMOOTH) / (tp + fp + SMOOTH)
    r = (tp + SMOOTH) / (tp + fn + SMOOTH)
    f1 = 2 * p * r / (p + r + SMOOTH)
    iou = (tp + SMOOTH) / (tp + fp + fn + SMOOTH)
    acc = (tp + tn + SMOOTH) / (tp + fp + fn + tn + SMOOTH)
    return {
        "iou": round(float(iou), 4),
        "f1": round(float(f1), 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "accuracy": round(float(acc), 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

RESULTS = {"epoch": ckpt.get("epoch"), "base_filters": base_filters}

for crop_size in [256, 512, None]:
    label = "full_1024" if crop_size is None else f"crop_{crop_size}"
    print(f"\n=== {label} ===")

    # --- RUN VAL ---
    print(f"  Val forward passes...")
    val_cached = []
    for t in val_trips:
        prob, gt, _ = forward_and_bin(t, crop=crop_size)
        val_cached.append((prob, gt))

    # --- VAL sweep ---
    val_sweep = {}
    for thr in THRESHOLDS:
        tpt = fpt = fnt = tnt = 0
        for prob, gt in val_cached:
            pb = (prob >= thr).astype(np.uint8).ravel()
            gf = gt.ravel()
            tpt += int(((pb == 1) & (gf == 1)).sum())
            fpt += int(((pb == 1) & (gf == 0)).sum())
            fnt += int(((pb == 0) & (gf == 1)).sum())
            tnt += int(((pb == 0) & (gf == 0)).sum())
        m = metrics_from(tpt, fpt, fnt, tnt)
        val_sweep[f"{thr:.2f}"] = m
    best_thr = max(THRESHOLDS, key=lambda t: (val_sweep[f"{t:.2f}"]["f1"], val_sweep[f"{t:.2f}"]["iou"]))
    best_val_m = val_sweep[f"{best_thr:.2f}"]
    print(f"  Val best threshold: {best_thr}  ->  IoU={best_val_m['iou']:.4f} F1={best_val_m['f1']:.4f} P={best_val_m['precision']:.4f} R={best_val_m['recall']:.4f}")
    for thr in THRESHOLDS:
        m = val_sweep[f"{thr:.2f}"]
        marker = "  <--BEST" if thr == best_thr else ""
        print(f"    thr={thr:.2f}  IoU={m['iou']:.4f} F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f}{marker}")

    # --- RUN TEST once at best_val_thr ---
    print(f"  Test forward passes @ thr={best_thr}...")
    tpt = fpt = fnt = tnt = 0
    for t in test_trips:
        prob, gt, _ = forward_and_bin(t, crop=crop_size)
        pb = (prob >= best_thr).astype(np.uint8).ravel()
        gf = gt.ravel()
        tpt += int(((pb == 1) & (gf == 1)).sum())
        fpt += int(((pb == 1) & (gf == 0)).sum())
        fnt += int(((pb == 0) & (gf == 1)).sum())
        tnt += int(((pb == 0) & (gf == 0)).sum())
    test_m = metrics_from(tpt, fpt, fnt, tnt)
    print(f"  TEST @ thr={best_thr}  IoU={test_m['iou']:.4f} F1={test_m['f1']:.4f} P={test_m['precision']:.4f} R={test_m['recall']:.4f} Acc={test_m['accuracy']:.4f}")

    RESULTS[label] = {
        "val_threshold_sweep": val_sweep,
        "best_val_threshold": best_thr,
        "val_at_best_thr": best_val_m,
        "test_at_best_val_thr": test_m,
    }


# Save JSON
with open(OUT_DIR / "baseline_fullres_metrics.json", "w", encoding="utf-8") as f:
    json.dump(RESULTS, f, indent=2)
print(f"\nSaved: {OUT_DIR / 'baseline_fullres_metrics.json'}")

# Final summary
print("\n====== FINAL BASELINE SUMMARY (ONCE-EVALUATED TEST SET NUMBERS) ======")
for crop, label_name in [(256, "256-center-crop (training/val/test protocol)"),
                          (512, "512-center-crop"),
                          (None, "FULL 1024x1024 image")]:
    lab = "full_1024" if crop is None else f"crop_{crop}"
    r = RESULTS[lab]
    print(f"\n[{label_name}]")
    v = r["val_at_best_thr"]; t = r["test_at_best_val_thr"]
    print(f"  Best val thr : {r['best_val_threshold']}")
    print(f"  Val  IoU={v['iou']:.4f}  F1={v['f1']:.4f}  Prec={v['precision']:.4f}  Rec={v['recall']:.4f}  Acc={v['accuracy']:.4f}")
    print(f"  Test IoU={t['iou']:.4f}  F1={t['f1']:.4f}  Prec={t['precision']:.4f}  Rec={t['recall']:.4f}  Acc={t['accuracy']:.4f}")
