"""
backend/tests/test_optical_sar_fusion.py
========================================
Unit and regression tests for Optical + SAR dual-branch neural fusion architecture (OpticalSARFusionNet).
Verifies:
- Optical-only, SAR-only, and Fused paths
- Correct channel handling (RGB 3-ch, multispectral 4-ch, single-pol 1-ch, dual-pol 2-ch)
- Numerical distinction between optical and radar backscatter
- Deterministic inference reproducibility
- Strict avoidance of mock outputs and fabricated metrics
- Provenance propagation
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch

from app.services.models.optical_sar_fusion import (
    OpticalSARFusionNet,
    OpticalSARInferenceResult,
    get_optical_sar_fusion_model,
    run_optical_sar_fusion_inference,
    MODEL_STATUS_BASELINE,
)
from app.services.optical_sar import (
    align_optical_sar,
    run_optical_sar_analysis,
)


def test_optical_branch_channel_counts():
    """Verify optical encoder accepts both 3-channel (RGB) and 4-channel (B02, B03, B04, B08) inputs."""
    model3 = OpticalSARFusionNet(optical_channels=3, sar_channels=2, embed_dim=128)
    x3 = torch.randn(2, 3, 64, 64)
    z3 = model3.encode_optical(x3)
    assert z3.shape == (2, 128)

    model4 = OpticalSARFusionNet(optical_channels=4, sar_channels=2, embed_dim=128)
    x4 = torch.randn(2, 4, 64, 64)
    z4 = model4.encode_optical(x4)
    assert z4.shape == (2, 128)


def test_sar_branch_channel_counts():
    """Verify SAR encoder accepts both 1-channel (single-pol VV) and 2-channel (dual-pol VV, VH)."""
    model_single = OpticalSARFusionNet(optical_channels=3, sar_channels=1, embed_dim=128)
    s1 = torch.randn(2, 1, 64, 64)
    zs1 = model_single.encode_sar(s1)
    assert zs1.shape == (2, 128)

    model_dual = OpticalSARFusionNet(optical_channels=3, sar_channels=2, embed_dim=128)
    s2 = torch.randn(2, 2, 64, 64)
    zs2 = model_dual.encode_sar(s2)
    assert zs2.shape == (2, 128)


def test_optical_only_path():
    """Verify optical-only representation can be extracted without SAR."""
    model = OpticalSARFusionNet(optical_channels=3, sar_channels=2, embed_dim=128)
    opt_in = torch.ones(1, 3, 32, 32)
    z_opt = model.encode_optical(opt_in)
    assert z_opt.shape == (1, 128)
    assert torch.isfinite(z_opt).all()


def test_sar_only_path():
    """Verify SAR-only representation can be extracted without optical."""
    model = OpticalSARFusionNet(optical_channels=3, sar_channels=2, embed_dim=128)
    sar_in = torch.ones(1, 2, 32, 32) * -15.0  # -15 dB backscatter
    z_sar = model.encode_sar(sar_in)
    assert z_sar.shape == (1, 128)
    assert torch.isfinite(z_sar).all()


def test_fused_path_and_gate_weights():
    """Verify fused path computes both fused embedding and gate weights."""
    model = OpticalSARFusionNet(optical_channels=3, sar_channels=2, embed_dim=128)
    opt_in = torch.randn(1, 3, 32, 32)
    sar_in = torch.randn(1, 2, 32, 32)

    out = model(opt_in, sar_in)
    assert "optical_embedding" in out
    assert "sar_embedding" in out
    assert "fused_embedding" in out
    assert "gate_weights" in out

    assert out["fused_embedding"].shape == (1, 128)
    assert out["gate_weights"].shape == (1, 128)
    # Sigmoid gate weights must lie in [0, 1]
    assert (out["gate_weights"] >= 0.0).all() and (out["gate_weights"] <= 1.0).all()


def test_no_rgb_conversion_of_sar():
    """Verify SAR backscatter in decibels is fed numerically without conversion to 3-channel RGB."""
    # Create float32 dB array
    sar_vv = np.full((64, 64), -18.5, dtype=np.float32)
    sar_vh = np.full((64, 64), -24.2, dtype=np.float32)
    opt_rgb = np.ones((64, 64, 3), dtype=np.float32) * 0.4

    res = run_optical_sar_fusion_inference(opt_rgb, sar_vv, sar_vh)
    # The SAR tensor shape fed to the network must have C=2 channels, NOT 3
    assert res.sar_shape[1] == 2
    assert res.optical_shape[1] == 3


def test_deterministic_inference():
    """Verify running inference twice on identical inputs produces identical embeddings."""
    opt_arr = np.random.RandomState(42).rand(64, 64, 3).astype(np.float32)
    sar_vv = (np.random.RandomState(43).rand(64, 64) * -25.0).astype(np.float32)
    sar_vh = (np.random.RandomState(44).rand(64, 64) * -30.0).astype(np.float32)

    res1 = run_optical_sar_fusion_inference(opt_arr, sar_vv, sar_vh)
    res2 = run_optical_sar_fusion_inference(opt_arr, sar_vv, sar_vh)

    assert np.allclose(res1.optical_embedding, res2.optical_embedding, atol=1e-6)
    assert np.allclose(res1.sar_embedding, res2.sar_embedding, atol=1e-6)
    assert np.allclose(res1.fused_embedding, res2.fused_embedding, atol=1e-6)


def test_model_parameter_count():
    """Verify model parameter count is lightweight (~300k parameters)."""
    model = OpticalSARFusionNet(optical_channels=3, sar_channels=2, embed_dim=128)
    param_count = model.parameter_count
    assert 200_000 <= param_count <= 500_000
    assert param_count == 302_880


def test_provenance_and_model_status_propagation():
    """Verify model status and provenance are explicitly declared without fabricated accuracy."""
    opt_arr = np.zeros((32, 32, 3), dtype=np.float32)
    sar_vv = np.zeros((32, 32), dtype=np.float32) - 10.0

    res = run_optical_sar_fusion_inference(opt_arr, sar_vv, provenance_tag="real_local_geotiff_scene")
    assert res.model_status == MODEL_STATUS_BASELINE
    assert "supervised benchmark trained" not in res.model_status.lower() or "no supervised benchmark trained" in res.model_status.lower()
    assert res.provenance == "real_local_geotiff_scene"


def test_no_mock_random_inference():
    """Verify end-to-end analysis returns real neural fusion statistics and confidence is None."""
    tmp_opt = Path("test_opt.tif")
    tmp_vv = Path("test_vv.tif")
    # Use synthetic test pair fixture or real file if available
    from PIL import Image
    Image.new("RGB", (64, 64), color=(100, 150, 200)).save(tmp_opt)
    Image.fromarray(np.full((64, 64), -14.0, dtype=np.float32)).save(tmp_vv)

    try:
        result = run_optical_sar_analysis(
            optical_path=tmp_opt,
            sar_path=tmp_vv,
            query="Test fusion integration",
        )
        assert result.confidence is None  # Strictly no fabricated confidence
        assert result.is_mock is False
        assert "fusion_model" in result.stats
        assert result.stats["fusion_model"] == "OpticalSARFusionNet-DualBranch-v1"
        assert result.stats["fused_embedding_dim"] == 128
        assert "ablation" in result.stats
        assert "cosine_sim_opt_vs_fused" in result.stats["ablation"]
    finally:
        if tmp_opt.exists():
            tmp_opt.unlink()
        if tmp_vv.exists():
            tmp_vv.unlink()
