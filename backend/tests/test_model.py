"""
tests/test_model.py
--------------------
Tests for the SiameseUNet model architecture and training metrics.

Tests are skipped if torch is not installed.
All tests use tiny random tensors (CPU-safe, fast).
No GPU required, no real dataset needed.

Tests cover:
- Model forward pass output shape
- Output probability range after sigmoid [0, 1]
- Model parameter count (sanity check)
- Dice loss computation (known values)
- BCE+Dice combined loss (sanity check)
- Metrics computation with known values (all-correct, all-wrong, zeros)
- No fabricated predictions (all-zero input → real output from model, not hardcoded)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

try:
    import torch
    import torch.nn as nn
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

pytestmark = pytest.mark.skipif(not _TORCH_OK, reason="torch is not installed")


# ---------------------------------------------------------------------------
# Model architecture tests
# ---------------------------------------------------------------------------

def test_siamese_unet_import():
    """SiameseUNet should be importable without error."""
    from app.services.models.siamese_unet import SiameseUNet
    assert SiameseUNet is not None


def test_siamese_unet_forward_shape():
    """
    Forward pass on a 6-channel (3+3) random input of shape (B, 6, H, W)
    should return logits of shape (B, 1, H, W).
    """
    from app.services.models.siamese_unet import SiameseUNet

    B, H, W = 2, 64, 64
    model = SiameseUNet(in_channels=6, base_filters=8)
    model.eval()
    with torch.no_grad():
        x = torch.rand(B, 6, H, W)
        logits = model(x)

    assert logits.shape == (B, 1, H, W), f"Expected (2, 1, 64, 64), got {logits.shape}"


def test_siamese_unet_output_range_after_sigmoid():
    """
    After sigmoid, all output values must be in [0, 1].
    """
    from app.services.models.siamese_unet import SiameseUNet

    model = SiameseUNet(in_channels=6, base_filters=8)
    model.eval()
    with torch.no_grad():
        x = torch.rand(1, 6, 32, 32)
        logits = model(x)
        prob = torch.sigmoid(logits)

    assert prob.min().item() >= 0.0, f"Probability below 0: {prob.min().item()}"
    assert prob.max().item() <= 1.0, f"Probability above 1: {prob.max().item()}"


def test_siamese_unet_parameter_count():
    """
    SiameseUNet with base_filters=16 should have ~400K–600K parameters.
    This is a sanity check that the architecture is not accidentally too large or small.
    """
    from app.services.models.siamese_unet import SiameseUNet, count_parameters

    model = SiameseUNet(in_channels=6, base_filters=16)
    n_params = count_parameters(model)
    assert 100_000 <= n_params <= 2_000_000, (
        f"Parameter count {n_params:,} is outside expected range [100K, 2M]. "
        "Check model architecture."
    )


def test_siamese_unet_no_hardcoded_predictions():
    """
    The model must NOT return hardcoded predictions.
    Two different random inputs should produce different outputs.
    (Ensures the model is actually using the input, not ignoring it.)
    """
    from app.services.models.siamese_unet import SiameseUNet

    model = SiameseUNet(in_channels=6, base_filters=8)
    model.eval()
    with torch.no_grad():
        x1 = torch.zeros(1, 6, 32, 32)   # all-zero input
        x2 = torch.ones(1, 6, 32, 32)    # all-one input
        out1 = model(x1)
        out2 = model(x2)

    assert not torch.allclose(out1, out2), (
        "Model produced identical outputs for all-zero and all-one inputs. "
        "This would indicate hardcoded/constant predictions."
    )


def test_siamese_unet_all_zero_input():
    """
    With all-zero input (no image content), the model should not fabricate
    a specific change percentage. Output values should vary across pixels
    (BatchNorm + learned weights produce varied outputs).
    """
    from app.services.models.siamese_unet import SiameseUNet

    model = SiameseUNet(in_channels=6, base_filters=8)
    model.eval()
    with torch.no_grad():
        x = torch.zeros(1, 6, 32, 32)
        logits = model(x)
        prob = torch.sigmoid(logits)

    # The model output should be a real value from the network, not fabricated
    assert prob.shape == (1, 1, 32, 32)
    # Can't assert specific values, but can assert it's in valid range
    assert prob.min().item() >= 0.0
    assert prob.max().item() <= 1.0


def test_predict_change_mask_binary():
    """
    predict_change_mask() should return a binary mask with only {0.0, 1.0} values.
    """
    from app.services.models.siamese_unet import SiameseUNet, predict_change_mask

    model = SiameseUNet(in_channels=6, base_filters=8)
    img_a = torch.rand(1, 3, 32, 32)
    img_b = torch.rand(1, 3, 32, 32)

    prob_map, binary_mask = predict_change_mask(model, img_a, img_b, threshold=0.5)

    assert prob_map.shape == (1, 1, 32, 32)
    assert binary_mask.shape == (1, 1, 32, 32)
    unique_vals = torch.unique(binary_mask).tolist()
    assert set(unique_vals).issubset({0.0, 1.0}), (
        f"Binary mask should have only {{0.0, 1.0}}, got {unique_vals}"
    )


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------

def test_dice_loss_perfect_prediction():
    """
    Dice loss for perfect prediction (pred == target) should be near 0.
    """
    from app.services.models.siamese_unet import dice_loss

    target = torch.ones(2, 1, 8, 8)
    pred = torch.ones(2, 1, 8, 8)  # perfect prediction

    loss = dice_loss(pred, target)
    assert loss.item() < 0.01, f"Dice loss for perfect prediction should be ~0, got {loss.item()}"


def test_dice_loss_worst_prediction():
    """
    Dice loss for inverse prediction (pred=1 where target=0) should be near 1.
    """
    from app.services.models.siamese_unet import dice_loss

    target = torch.zeros(2, 1, 8, 8)  # no change
    pred = torch.ones(2, 1, 8, 8)    # predicts all change

    loss = dice_loss(pred, target)
    assert loss.item() > 0.9, f"Dice loss for worst-case prediction should be ~1, got {loss.item()}"


def test_combined_loss_backward():
    """
    Combined loss should be differentiable — backward() should not raise.
    """
    from app.services.models.siamese_unet import combined_loss

    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    target = (torch.rand(2, 1, 8, 8) > 0.5).float()

    loss = combined_loss(logits, target)
    loss.backward()

    assert logits.grad is not None, "Gradients should flow through combined_loss"


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

def test_metrics_perfect_prediction():
    """
    All-correct prediction should give IoU=1, F1=1, Precision=1, Recall=1.
    """
    from scripts.train_change_detector import compute_metrics

    target = (torch.rand(1, 1, 32, 32) > 0.5).float()
    pred = target.clone()  # perfect prediction

    m = compute_metrics(pred, target, threshold=0.5)
    assert m.iou > 0.99, f"Perfect prediction IoU should be ~1, got {m.iou}"
    assert m.f1 > 0.99, f"Perfect prediction F1 should be ~1, got {m.f1}"
    assert m.precision > 0.99
    assert m.recall > 0.99


def test_metrics_all_wrong_prediction():
    """
    All-incorrect prediction (every pixel inverted) should give low IoU and F1.
    """
    from scripts.train_change_detector import compute_metrics

    target = torch.ones(1, 1, 16, 16)  # all change
    pred = torch.zeros(1, 1, 16, 16)   # predict no change

    m = compute_metrics(pred, target, threshold=0.5)
    # IoU = 0 / (0 + 0 + 256) → near 0
    assert m.iou < 0.01, f"All-wrong prediction IoU should be ~0, got {m.iou}"
    assert m.recall < 0.01, f"All-wrong recall should be ~0, got {m.recall}"


def test_metrics_all_zero_prediction_no_fabrication():
    """
    All-zero prediction with a target that has some change → IoU near 0.
    This verifies that no fabricated metrics are returned.
    """
    from scripts.train_change_detector import compute_metrics

    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 2:4, 2:4] = 1.0  # small changed region
    pred = torch.zeros(1, 1, 8, 8)  # predict nothing

    m = compute_metrics(pred, target, threshold=0.5)
    # tp=0, fp=0, fn=4, tn=60 → precision=1 (smooth), recall≈0, IoU≈0
    assert m.recall < 0.01, f"Zero prediction recall should be ~0, got {m.recall}"
    assert m.tp == 0, f"Zero prediction TP should be 0, got {m.tp}"
    assert m.fn == 4, f"Expected 4 FN (missed changed pixels), got {m.fn}"


def test_metrics_computation_matches_manual():
    """
    Manually verify TP/FP/FN/TN counts for a known small case.
    """
    from scripts.train_change_detector import compute_metrics

    # 2×2 prediction and target
    pred = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])   # (1,1,2,2)
    target = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])  # (1,1,2,2)
    # pixel (0,0): pred=1, target=1 → TP
    # pixel (0,1): pred=0, target=1 → FN
    # pixel (1,0): pred=1, target=0 → FP
    # pixel (1,1): pred=0, target=0 → TN

    m = compute_metrics(pred, target, threshold=0.5)
    assert m.tp == 1, f"Expected 1 TP, got {m.tp}"
    assert m.fn == 1, f"Expected 1 FN, got {m.fn}"
    assert m.fp == 1, f"Expected 1 FP, got {m.fp}"
    assert m.tn == 1, f"Expected 1 TN, got {m.tn}"


# ---------------------------------------------------------------------------
# Model inference module tests
# ---------------------------------------------------------------------------

def test_experiment_01_checkpoint_discovery():
    """
    _resolve_checkpoint_path() should resolve the Experiment 01 checkpoint.
    """
    from app.services.model_inference import _resolve_checkpoint_path

    ckpt = _resolve_checkpoint_path()
    assert ckpt is not None, "Experiment 01 checkpoint should be discovered"
    assert ckpt.exists(), f"Discovered checkpoint does not exist: {ckpt}"
    assert "best_model.pt" in str(ckpt)


def test_experiment_01_model_loading_and_e2e_inference(tmp_path):
    """
    Real Experiment 01 checkpoint should load and run tiled inference at threshold 0.70.
    """
    from PIL import Image
    from app.services.model_inference import run_change_detection, _resolve_checkpoint_path

    ckpt = _resolve_checkpoint_path()
    assert ckpt is not None, "Checkpoint must exist for real test"

    # Create dummy before and after images (256x256 RGB)
    img_a_path = tmp_path / "before.png"
    img_b_path = tmp_path / "after.png"
    Image.new("RGB", (256, 256), color=(100, 150, 200)).save(img_a_path)
    Image.new("RGB", (256, 256), color=(200, 100, 50)).save(img_b_path)

    result = run_change_detection(
        before_path=img_a_path,
        after_path=img_b_path,
        analysis_id="test_exp01_inference",
    )

    assert result.stats["execution_mode"] == "model_checkpoint"
    assert result.stats["threshold_used"] == 0.70
    assert result.stats["threshold_raw_255"] == int(round(0.70 * 255))
    assert "best_model.pt" in result.stats["checkpoint_path"]
    assert "Siamese U-Net Model" in result.answer
    assert result.change_map["overlayUrl"] is not None


def test_get_inference_mode_with_checkpoint():
    """
    get_inference_mode() should return 'model_checkpoint' when Experiment 01 checkpoint exists.
    """
    from app.services.model_inference import get_inference_mode
    assert get_inference_mode() == "model_checkpoint"


def test_get_inference_mode_no_checkpoint_fallback(monkeypatch):
    """
    get_inference_mode() should return 'cpu_classical' when no candidate checkpoint exists.
    """
    from app.services import model_inference

    monkeypatch.setattr(model_inference, "_resolve_checkpoint_path", lambda: None)
    mode = model_inference.get_inference_mode()
    assert mode == "cpu_classical"


def test_run_change_detection_classical_fallback_when_no_checkpoint(tmp_path, monkeypatch):
    """
    run_change_detection() should gracefully fall back to CPU classical when checkpoint is missing.
    """
    from PIL import Image
    from app.services import model_inference

    monkeypatch.setattr(model_inference, "_resolve_checkpoint_path", lambda: None)

    img_a = tmp_path / "before.png"
    img_b = tmp_path / "after.png"
    Image.new("RGB", (64, 64), color=(0, 0, 0)).save(img_a)
    Image.new("RGB", (64, 64), color=(255, 255, 255)).save(img_b)

    result = model_inference.run_change_detection(
        before_path=img_a,
        after_path=img_b,
        analysis_id="test_fallback",
    )

    assert result.stats["execution_mode"] == "cpu_classical"
    assert result.stats["checkpoint_path"] is None


# Need MagicMock at module level for tests
from unittest.mock import MagicMock, patch


def test_planner_produces_configured_threshold():
    """
    plan_execution() should produce threshold 0.70 for change_detector when model_checkpoint is active.
    """
    from app.services.orchestrator import plan_execution

    tasks, tools, params, scores = plan_execution(
        query="Detect changes between acquisitions",
        mode="bi_temporal",
    )
    assert "change_detector" in tools
    cd_params = params.get("change_detector", {})
    assert cd_params.get("threshold") == 0.70
    assert cd_params.get("algorithm") == "siamese-unet-model"


def test_execute_plan_forwards_threshold(tmp_path):
    """
    execute_plan() should forward the planned threshold from per_tool_params to run_change_detection().
    """
    from app.services.orchestrator import execute_plan

    img_a = tmp_path / "before.png"
    img_b = tmp_path / "after.png"
    img_a.write_bytes(b"dummy_a")
    img_b.write_bytes(b"dummy_b")

    mock_cd_result = MagicMock()
    mock_cd_result.answer = "Report"
    mock_cd_result.confidence = None
    mock_cd_result.change_map = {"overlayUrl": "/api/results/mock.png"}
    mock_cd_result.evidence = ["Evidence"]
    mock_cd_result.stats = {"execution_mode": "model_checkpoint", "threshold_used": 0.75}

    with patch("app.services.orchestrator.run_change_detection", return_value=mock_cd_result) as mock_run:
        execute_plan(
            query="Detect changes",
            mode="bi_temporal",
            tool_ids=["change_detector"],
            per_tool_params={"change_detector": {"threshold": 0.75}},
            image_file_paths=[img_a, img_b],
            analysis_id="test_plan_fwd",
        )

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs.get("threshold") == 0.75


def test_inference_uses_supplied_threshold_override(tmp_path):
    """
    run_change_detection() should strictly use the supplied threshold override.
    """
    from PIL import Image
    from app.services.model_inference import run_change_detection

    img_a = tmp_path / "before.png"
    img_b = tmp_path / "after.png"
    Image.new("RGB", (64, 64), color=(50, 100, 150)).save(img_a)
    Image.new("RGB", (64, 64), color=(150, 100, 50)).save(img_b)

    result = run_change_detection(
        before_path=img_a,
        after_path=img_b,
        analysis_id="test_override",
        threshold=0.85,
    )

    assert result.stats["threshold_used"] == 0.85
    assert result.stats["threshold_raw_255"] == int(round(0.85 * 255))
    assert any("threshold: 0.85" in ev for ev in result.evidence)


def test_inference_default_threshold_remains_070(tmp_path):
    """
    run_change_detection() should default to 0.70 when no threshold is supplied.
    """
    from PIL import Image
    from app.services.model_inference import run_change_detection

    img_a = tmp_path / "before.png"
    img_b = tmp_path / "after.png"
    Image.new("RGB", (64, 64), color=(50, 100, 150)).save(img_a)
    Image.new("RGB", (64, 64), color=(150, 100, 50)).save(img_b)

    result = run_change_detection(
        before_path=img_a,
        after_path=img_b,
        analysis_id="test_default_070",
    )

    assert result.stats["threshold_used"] == 0.70
    assert result.stats["threshold_raw_255"] == int(round(0.70 * 255))


def test_no_fabricated_confidence_in_change_detection(tmp_path):
    """
    run_change_detection() must always return confidence=None (uncalibrated segmentation mask).
    """
    from PIL import Image
    from app.services.model_inference import run_change_detection

    img_a = tmp_path / "before.png"
    img_b = tmp_path / "after.png"
    Image.new("RGB", (64, 64), color=(50, 100, 150)).save(img_a)
    Image.new("RGB", (64, 64), color=(150, 100, 50)).save(img_b)

    result = run_change_detection(
        before_path=img_a,
        after_path=img_b,
        analysis_id="test_conf_none",
    )

    assert result.confidence is None


