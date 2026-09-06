#!/usr/bin/env python3
"""
Baseline evaluation: run existing best_model.pt on val + test matched triplets.
Save: validation threshold sweep, test metrics @ best val threshold, qualitative 6-panel figs.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_ROOT))
os.chdir(_BACKEND_ROOT)

from dotenv import load_dotenv
load_dotenv(".env")

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from app.services.models.siamese_unet import SiameseUNet

DATA_ROOT = Path(os.getenv("LEVIR_CD_DATASET_PATH", "./data/results/LEVIR-CD")).resolve()
print(f"DATA_ROOT={DATA_ROOT} exists={DATA_ROOT.exists()}")

CKPT_PATH = _BACKEND_ROOT / "checkpoints" / "best_model.pt"
print(f"CKPT_PATH={CKPT_PATH} exists={CKPT_PATH.exists()}")

OUT_DIR = _BACKEND_ROOT / "evaluation_results" / "baseline_official_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# 1. Load model
# -----------------------------
device = torch.device("cpu")
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
base_filters = sd["enc1.block.0.block.0.weight"].shape[0]
print(f"Detected base_filters={base_filters}")
model = SiameseUNet(in_channels=6, base_filters=base_filters)
model.load_state_dict(sd)
model.to(device)
model.eval()
print(f"Loaded model from epoch={ckpt.get('epoch', '?')} total_params={sum(p.numel() for p in model.parameters()):,}")


# -----------------------------
# 2. Get matched triplets per split
# -----------------------------
def matched_triplets(split: str):
    d = DATA_ROOT / split
    a = {f.name: f for f in (d / "A").glob("*.png")}
    b = {f.name: f for f in (d / "B").glob("*.png")}
    l = {f.name: f for f in (d / "label").glob("*.png")}
    names = sorted(set(a) & set(b) & set(l))
    print(f"  split={split:5s} matched={len(names):3d}  A={len(a)} B={len(b)} label={len(l)}")
    return [(n, a[n], b[n], l[n]) for n in names]

print("\nMatched triplets per split:")
splits = {s: matched_triplets(s) for s in ("train", "val", "test")}


# -----------------------------
# 3. Evaluate single split with center crop
# -----------------------------
SMOOTH = 1e-6
CROP = 256

def run_split_eval(triplets, threshold_default=0.50):
    cached = []  # (name, crop_a_PIL, crop_b_PIL, gt_arr, prob_map_arr)
    per_sample = []
    tp_tot = fp_tot = fn_tot = tn_tot = 0

    for idx, (name, pa, pb, pl) in enumerate(triplets, 1):
        ra = Image.open(pa).convert("RGB")
        rb = Image.open(pb).convert("RGB")
        rl = Image.open(pl).convert("L")
        W, H = ra.size
        i, j = (H - CROP) // 2, (W - CROP) // 2
        ca = TF.crop(ra, i, j, CROP, CROP)
        cb = TF.crop(rb, i, j, CROP, CROP)
        cl = TF.crop(rl, i, j, CROP, CROP)

        ta = TF.to_tensor(ca).unsqueeze(0)
        tb = TF.to_tensor(cb).unsqueeze(0)
        gt_arr = (np.array(cl) > 127).astype(np.uint8)

        inp = torch.cat([ta, tb], dim=1)
        with torch.no_grad():
            logits = model(inp)
            prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()

        cached.append((name, ca, cb, gt_arr, prob_map))

        pred_bin = (prob_map >= threshold_default).astype(np.uint8)
        pf, gf = pred_bin.reshape(-1), gt_arr.reshape(-1)
        tp = int(((pf == 1) & (gf == 1)).sum())
        fp = int(((pf == 1) & (gf == 0)).sum())
        fn = int(((pf == 0) & (gf == 1)).sum())
        tn = int(((pf == 0) & (gf == 0)).sum())

        prec = (tp + SMOOTH) / (tp + fp + SMOOTH)
        rec = (tp + SMOOTH) / (tp + fn + SMOOTH)
        f1 = 2 * prec * rec / (prec + rec + SMOOTH)
        iou = (tp + SMOOTH) / (tp + fp + fn + SMOOTH)
        acc = (tp + tn + SMOOTH) / (tp + fp + fn + tn + SMOOTH)

        per_sample.append({
            "idx": idx, "name": name,
            "gt_changed": int(gf.sum()), "pred_changed": int(pf.sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "threshold_default": threshold_default,
            "iou": round(float(iou), 4), "f1": round(float(f1), 4),
            "precision": round(float(prec), 4), "recall": round(float(rec), 4),
            "accuracy": round(float(acc), 4),
        })
        tp_tot += tp; fp_tot += fp; fn_tot += fn; tn_tot += tn

    g_prec = (tp_tot + SMOOTH) / (tp_tot + fp_tot + SMOOTH)
    g_rec  = (tp_tot + SMOOTH) / (tp_tot + fn_tot + SMOOTH)
    g_f1   = 2 * g_prec * g_rec / (g_prec + g_rec + SMOOTH)
    g_iou  = (tp_tot + SMOOTH) / (tp_tot + fp_tot + fn_tot + SMOOTH)
    g_acc  = (tp_tot + tn_tot + SMOOTH) / (tp_tot + fp_tot + fn_tot + tn_tot + SMOOTH)
    return {
        "global_micro": {
            "iou": round(float(g_iou), 4), "f1": round(float(g_f1), 4),
            "precision": round(float(g_prec), 4), "recall": round(float(g_rec), 4),
            "accuracy": round(float(g_acc), 4),
            "tp": tp_tot, "fp": fp_tot, "fn": fn_tot, "tn": tn_tot,
        },
        "sample_macro": {
            "mean_iou": round(float(np.mean([s["iou"] for s in per_sample])), 4),
            "mean_f1":  round(float(np.mean([s["f1"]  for s in per_sample])), 4),
            "mean_precision": round(float(np.mean([s["precision"] for s in per_sample])), 4),
            "mean_recall":    round(float(np.mean([s["recall"]    for s in per_sample])), 4),
            "mean_accuracy":  round(float(np.mean([s["accuracy"]  for s in per_sample])), 4),
        },
        "per_sample": per_sample,
        "cached": cached,
        "threshold_default": threshold_default,
    }


# -----------------------------
# 4. Val: run sweep, pick best threshold by F1+IoU
# -----------------------------
print("\n[1/3] Running VAL forward passes (for threshold sweep)...")
val_res = run_split_eval(splits["val"], threshold_default=0.50)
thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

def sweep_thresholds(cached, thresholds):
    rows = []
    for thr in thresholds:
        tpt = fpt = fnt = tnt = 0
        ious, f1s = [], []
        for (_, _, _, gt_arr, prob_map) in cached:
            pb = (prob_map >= thr).astype(np.uint8).reshape(-1)
            gf = gt_arr.reshape(-1)
            tp = int(((pb == 1) & (gf == 1)).sum())
            fp = int(((pb == 1) & (gf == 0)).sum())
            fn = int(((pb == 0) & (gf == 1)).sum())
            tn = int(((pb == 0) & (gf == 0)).sum())
            pr = (tp + SMOOTH) / (tp + fp + SMOOTH)
            rc = (tp + SMOOTH) / (tp + fn + SMOOTH)
            f1 = 2 * pr * rc / (pr + rc + SMOOTH)
            iou = (tp + SMOOTH) / (tp + fp + fn + SMOOTH)
            ious.append(iou); f1s.append(f1)
            tpt += tp; fpt += fp; fnt += fn; tnt += tn
        gp = (tpt + SMOOTH) / (tpt + fpt + SMOOTH)
        gr = (tpt + SMOOTH) / (tpt + fnt + SMOOTH)
        gf = 2 * gp * gr / (gp + gr + SMOOTH)
        gi = (tpt + SMOOTH) / (tpt + fpt + fnt + SMOOTH)
        rows.append({
            "threshold": round(thr, 2),
            "micro_iou": round(float(gi), 4), "micro_f1": round(float(gf), 4),
            "micro_precision": round(float(gp), 4), "micro_recall": round(float(gr), 4),
            "mean_sample_iou": round(float(np.mean(ious)), 4),
            "mean_sample_f1":  round(float(np.mean(f1s)),  4),
        })
    return rows

val_sweep = sweep_thresholds(val_res["cached"], thresholds)
best = max(val_sweep, key=lambda r: (r["micro_f1"], r["micro_iou"]))
print(f"\n=== VALIDATION THRESHOLD SWEEP ({len(splits['val'])} samples) ===")
print(f"{'thr':<6} {'mIoU':<8} {'mF1':<8} {'mP':<8} {'mR':<8}")
for r in val_sweep:
    marker = "  <-- BEST" if r["threshold"] == best["threshold"] else ""
    print(f"{r['threshold']:<6.2f} {r['micro_iou']:<8.4f} {r['micro_f1']:<8.4f} {r['micro_precision']:<8.4f} {r['micro_recall']:<8.4f}{marker}")

BEST_THR = best["threshold"]
print(f"\nOPTIMAL VALIDATION THRESHOLD = {BEST_THR}")
print(f"  At best: Val Micro IoU={best['micro_iou']:.4f}  F1={best['micro_f1']:.4f}  Prec={best['micro_precision']:.4f}  Rec={best['micro_recall']:.4f}")


# -----------------------------
# 5. Recompute VAL final metrics at best threshold
# -----------------------------
val_final_res = run_split_eval(splits["val"], threshold_default=BEST_THR)


# -----------------------------
# 6. TEST: run ONCE at BEST_THR only (no test-data tuning!)
# -----------------------------
print(f"\n[2/3] Running TEST forward passes @ threshold {BEST_THR} (ONCE - no tuning on test)...")
test_res = run_split_eval(splits["test"], threshold_default=BEST_THR)

gm = test_res["global_micro"]
sm = test_res["sample_macro"]
print(f"\n{'=' * 70}")
print(f"BASELINE TEST METRICS @ BEST VAL THRESHOLD={BEST_THR}  (N={len(splits['test'])} matched test triplets)")
print(f"{'=' * 70}")
print(f"  Global Micro IoU:       {gm['iou']:.4f}")
print(f"  Global Micro F1 / Dice: {gm['f1']:.4f}")
print(f"  Global Micro Precision: {gm['precision']:.4f}")
print(f"  Global Micro Recall:    {gm['recall']:.4f}")
print(f"  Global Pixel Accuracy:  {gm['accuracy']:.4f}")
print(f"  Confusion (pix): TP={gm['tp']:,} FP={gm['fp']:,} FN={gm['fn']:,} TN={gm['tn']:,}")
print(f"  --- per-sample means ---")
print(f"  Mean IoU:      {sm['mean_iou']:.4f}")
print(f"  Mean F1:       {sm['mean_f1']:.4f}")
print(f"  Mean Prec:     {sm['mean_precision']:.4f}")
print(f"  Mean Recall:   {sm['mean_recall']:.4f}")
print(f"  Mean Accuracy: {sm['mean_accuracy']:.4f}")
print(f"{'=' * 70}\n")


# -----------------------------
# 7. Qualitative 6-panel predictions on 10 representative TEST samples
# -----------------------------
print("[3/3] Generating 6-panel qualitative prediction figures on TEST samples...")

def apply_turbo(prob):
    x = np.clip(prob, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return Image.fromarray((np.stack([r, g, b], axis=-1) * 255).astype(np.uint8), mode="RGB")

def six_panel(name, ca, cb, gt, prob, pred_bin, meta):
    tw, th = ca.size
    gt_rgb = Image.fromarray((gt * 255).astype(np.uint8), "L").convert("RGB")
    prob_rgb = apply_turbo(prob)
    if prob_rgb.size != (tw, th):
        prob_rgb = prob_rgb.resize((tw, th), Image.BILINEAR)
    pred_rgb = np.zeros((th, tw, 3), dtype=np.uint8)
    pred_rgb[pred_bin == 1] = [0, 220, 255]
    pred_rgb = Image.fromarray(pred_rgb, "RGB")
    # overlay: red blend on img B
    base = cb.convert("RGBA")
    ov = np.zeros((th, tw, 4), dtype=np.uint8)
    ov[pred_bin == 1] = [255, 40, 40, 140]
    blended = Image.alpha_composite(base, Image.fromarray(ov, "RGBA")).convert("RGB")

    pad, lh, hh = 12, 24, 70
    cols, rows = 3, 2
    cw = cols * tw + (cols + 1) * pad
    ch = hh + rows * (th + lh) + (rows + 1) * pad
    canvas = Image.new("RGB", (cw, ch), color=(18, 22, 28))
    dr = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    dr.rectangle([(0, 0), (cw, hh)], fill=(28, 34, 44))
    dr.text((pad + 4, 8),
            f"Baseline (Epoch {ckpt.get('epoch','?')}) | Sample: {name}  Thr={meta['threshold_default']}",
            fill=(240, 245, 255), font=font)
    dr.text((pad + 4, 28),
            f"IoU={meta['iou']:.4f}  F1={meta['f1']:.4f}  P={meta['precision']:.4f}  R={meta['recall']:.4f}  Acc={meta['accuracy']:.4f}",
            fill=(80, 220, 160), font=font)
    dr.text((pad + 4, 48),
            f"GT={meta['gt_changed']:,}  Pred={meta['pred_changed']:,}  "
            f"TP={meta['tp']:,} FP={meta['fp']:,} FN={meta['fn']:,}",
            fill=(180, 200, 220), font=font)
    panels = [
        ("(A) Before (T1)", ca),
        ("(B) After (T2)", cb),
        (f"(C) GT Mask ({meta['gt_changed']:,} px)", gt_rgb),
        ("(D) Probability Map [0-1]", prob_rgb),
        (f"(E) Pred Mask (thr={meta['threshold_default']})", pred_rgb),
        ("(F) Overlay on Image B", blended),
    ]
    for idx, (title, pimg) in enumerate(panels):
        r, c = divmod(idx, cols)
        x = pad + c * (tw + pad)
        y = hh + pad + r * (th + lh + pad)
        dr.text((x + 2, y), title, fill=(200, 215, 230), font=font)
        canvas.paste(pimg, (x, y + lh))
        dr.rectangle([(x - 1, y + lh - 1), (x + tw, y + lh + th)],
                     outline=(60, 75, 95), width=1)
    return canvas

# Pick 10 representative: top-5 most-change + 3 medium + 2 zero/least
samples = sorted(test_res["per_sample"], key=lambda s: s["gt_changed"], reverse=True)
pick_idx = sorted(set(
    [s["idx"] - 1 for s in samples[:5]] +
    [s["idx"] - 1 for s in samples[len(samples)//3:len(samples)//3 + 3]] +
    [s["idx"] - 1 for s in samples[-2:]]
))[:10]

qual_dir = OUT_DIR / "qualitative_test"
qual_dir.mkdir(exist_ok=True)
for si in pick_idx:
    meta = test_res["per_sample"][si]
    name, ca, cb, gt, prob = test_res["cached"][si]
    pred_bin = (prob >= meta["threshold_default"]).astype(np.uint8)
    fig = six_panel(name, ca, cb, gt, prob, pred_bin, meta)
    stem = Path(name).stem
    fig.save(qual_dir / f"test_{meta['idx']:03d}_{stem}_prediction.png", quality=92)
print(f"  saved {len(pick_idx)} qualitative figures to {qual_dir}")


# -----------------------------
# 8. Save JSON results
# -----------------------------
def dump_json(obj, path):
    def drop_cached(o):
        if isinstance(o, dict):
            return {k: (drop_cached(v) if k != "cached" else f"[{len(v)} samples cached, dropped]") for k, v in o.items()}
        if isinstance(o, list):
            return [drop_cached(x) for x in o]
        return o

    with open(path, "w", encoding="utf-8") as f:
        json.dump(drop_cached(obj), f, indent=2)

baseline_summary = {
    "checkpoint": str(CKPT_PATH),
    "checkpoint_epoch": ckpt.get("epoch", "?"),
    "checkpoint_val_metrics": ckpt.get("metrics", {}),
    "base_filters": base_filters,
    "device": str(device),
    "num_cpu_threads": torch.get_num_threads(),
    "splits_summary": {
        s: {"matched": len(trips),
            "A": len(set(p.name for p in (DATA_ROOT / s / "A").glob("*.png"))),
            "B": len(set(p.name for p in (DATA_ROOT / s / "B").glob("*.png"))),
            "label": len(set(p.name for p in (DATA_ROOT / s / "label").glob("*.png")))}
        for s, trips in splits.items()
    },
    "validation_threshold_sweep": val_sweep,
    "best_validation_threshold": BEST_THR,
    "validation_at_best_threshold": {
        "global_micro": val_final_res["global_micro"],
        "sample_macro": val_final_res["sample_macro"],
    },
    "test_at_best_threshold": {
        "global_micro": test_res["global_micro"],
        "sample_macro": test_res["sample_macro"],
        "per_sample_count": len(test_res["per_sample"]),
    },
    "test_per_sample": test_res["per_sample"],
}
dump_json(baseline_summary, OUT_DIR / "baseline_metrics.json")
print(f"\nAll baseline metrics saved to:\n  {OUT_DIR / 'baseline_metrics.json'}")
print(f"\nDone.")
