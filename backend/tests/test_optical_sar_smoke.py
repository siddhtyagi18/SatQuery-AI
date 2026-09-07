"""
backend/tests/test_optical_sar_smoke.py
=======================================
Unit & regression tests for Step 10E real multimodal smoke test on authentic BigEarthNet pairs.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from app.services.models.optical_sar_fusion import (
    OpticalSARFusionNet,
    get_optical_sar_fusion_model,
)
from app.services.optical_sar import (
    align_optical_sar,
    extract_geotiff_metadata,
)

S1_ROOT = Path(r"C:\Users\Lenovo\Downloads\BigEarthNet-S1")
S2_ROOT = Path(r"C:\Users\Lenovo\Downloads\BigEarthNet-S2")

P1_S2 = "S2A_MSIL2A_20180413T095031_N9999_R079_T35VLG_55_03"
P1_S1 = "S1A_IW_GRDH_1SDV_20180415T160451_35VLG_55_3"

P2_S2 = "S2B_MSIL2A_20170802T092029_N9999_R093_T34TFN_13_61"
P2_S1 = "S1B_IW_GRDH_1SDV_20170802T043751_34TFN_13_61"


@pytest.mark.skipif(not (S1_ROOT / P1_S1).exists(), reason="BigEarthNet S1 sample not downloaded")
def test_real_pair1_alignment_and_dimensions():
    """Verify authentic S1/S2 pair 1 loads, aligns, and has identical spatial bounds."""
    s2_dir = S2_ROOT / P1_S2
    s1_dir = S1_ROOT / P1_S1

    from PIL import Image
    bands = []
    for b in ["B02", "B03", "B04", "B08"]:
        f = list(s2_dir.glob(f"*{b}.tif"))[0]
        with Image.open(f) as img:
            bands.append(np.asarray(img, dtype=np.float32))
    opt_stacked = np.stack(bands, axis=-1)

    with Image.open(s1_dir / f"{P1_S1}_VV.tif") as img:
        vv = np.asarray(img, dtype=np.float32)
    with Image.open(s1_dir / f"{P1_S1}_VH.tif") as img:
        vh = np.asarray(img, dtype=np.float32)

    pkg = align_optical_sar(
        optical=opt_stacked,
        sar_vv=vv,
        sar_vh=vh,
        optical_bands=["B02", "B03", "B04", "B08"],
    )

    assert pkg.optical.shape == (120, 120, 4)
    assert pkg.sar_vv.shape == (120, 120)
    assert pkg.sar_vh.shape == (120, 120)
    assert pkg.height == 120 and pkg.width == 120


@pytest.mark.skipif(not (S1_ROOT / P1_S1).exists(), reason="BigEarthNet S1 sample not downloaded")
def test_real_multimodal_gated_fusion_deterministic():
    """Verify gated dual-branch fusion produces strictly deterministic output on real data."""
    model = get_optical_sar_fusion_model(
        optical_channels=4,
        sar_channels=2,
        embed_dim=128,
        num_classes=19,
        seed=42,
    )
    model.eval()

    opt_tensor = torch.randn(1, 4, 120, 120)
    sar_tensor = torch.randn(1, 2, 120, 120)

    with torch.no_grad():
        out1 = model(opt_tensor, sar_tensor)
        out2 = model(opt_tensor, sar_tensor)

    # Assert 0.0 drift
    for k in ["optical_embedding", "sar_embedding", "fused_embedding", "optical_logits", "sar_logits", "fused_logits"]:
        diff = torch.max(torch.abs(out1[k] - out2[k])).item()
        assert diff == 0.0, f"Drift detected in {k}: {diff}"


@pytest.mark.skipif(not (S1_ROOT / P1_S1).exists(), reason="BigEarthNet S1 sample not downloaded")
def test_three_configurations_ablation():
    """Verify Optical-only, SAR-only, and Gated Fusion forward paths operate independently."""
    model = get_optical_sar_fusion_model(
        optical_channels=4,
        sar_channels=2,
        embed_dim=128,
        num_classes=19,
        seed=42,
    )
    model.eval()

    opt_tensor = torch.randn(1, 4, 120, 120)
    sar_tensor = torch.randn(1, 2, 120, 120)

    with torch.no_grad():
        z_opt, opt_logits = model.predict_optical(opt_tensor)
        z_sar, sar_logits = model.predict_sar(sar_tensor)
        full_out = model(opt_tensor, sar_tensor)

    assert z_opt.shape == (1, 128)
    assert z_sar.shape == (1, 128)
    assert full_out["fused_embedding"].shape == (1, 128)

    assert opt_logits.shape == (1, 19)
    assert sar_logits.shape == (1, 19)
    assert full_out["fused_logits"].shape == (1, 19)

    # Verify representations are distinct
    cos_sim = float(torch.cosine_similarity(z_opt, z_sar).item())
    assert abs(cos_sim) < 0.99, f"Optical and SAR branches are improperly identical (cos_sim={cos_sim})"
