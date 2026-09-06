import sys
from pathlib import Path
import torch

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.models.siamese_unet import SiameseUNet

for ckpt_name in ["best_model.pt", "experiment_01/best_model.pt", "experiment_02/best_model.pt"]:
    p = BACKEND_ROOT / "checkpoints" / ckpt_name
    if not p.exists():
        print(f"\n[SKIP] {ckpt_name}: missing")
        continue
    print(f"\n=== {ckpt_name} ({p.stat().st_size / 1024:.1f} KB ===")
    ckpt = torch.load(p, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict):
        top_keys = [k for k in ckpt.keys()]
        print(f"  Top-level keys: {top_keys}")
        if "epoch" in ckpt:
            print(f"  epoch={ckpt['epoch']}")
        if "metrics" in ckpt:
            print(f"  metrics={ckpt['metrics']}")
        sd_key = "model_state_dict" if "model_state_dict" in ckpt else "state_dict" if "state_dict" in ckpt else None
        if sd_key is None:
            sd = ckpt
            print(f"  (raw state_dict keys: {len(sd)}")
        else:
            sd = ckpt[sd_key]
            print(f"  {sd_key} keys: {len(sd)}")
        total = sum(p.numel() for p in sd.values())
        print(f"  total params: {total:,}")
        first_conv_key = next(iter(sd.keys()))
        print(f"  first key: {first_conv_key} shape={sd[first_conv_key].shape}")
        if 'enc1.block.0.block.0.weight' in sd:
            f = sd['enc1.block.0.block.0.weight'].shape[0]
            print(f"  DEDUCED base_filters: {f} (out_ch of first conv)")
    else:
        print(f"  type = {type(ckpt).__name__}, not dict")
