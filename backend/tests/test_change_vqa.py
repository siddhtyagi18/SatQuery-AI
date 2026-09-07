"""
backend/tests/test_change_vqa.py
================================
Unit and integration tests for the real Change VQA service pipeline.

Verifies:
A. Tri-panel composite generation (3W x H, RGB, mask alignment, dimension handling).
B. Domain-conditioned change prompt formatting with quantitative stats and anti-hallucination rules.
C. Orchestrator dispatching to real Change VQA with genuine image paths and detection stats.
D. Strict preservation of confidence=None (anti-fabrication).
E. Graceful fallback on VLM runtime errors without fabricated answers.
F. Real local VLM+LoRA integration verification.
"""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from app.services.change_vqa import (
    ChangeVQAResult,
    build_change_vqa_prompt,
    create_change_composite,
    run_change_vqa,
)
from app.services.orchestrator import execute_plan, plan_execution


@pytest.fixture
def sample_pair(tmp_path: Path):
    """Generate two distinct 64x48 RGB test images."""
    p_a = tmp_path / "img_a.png"
    p_b = tmp_path / "img_b.png"
    Image.new("RGB", (64, 48), color=(40, 60, 80)).save(p_a)
    Image.new("RGB", (64, 48), color=(180, 160, 140)).save(p_b)
    return p_a, p_b


# ---------------------------------------------------------------------------
# A. Composite Generation Tests
# ---------------------------------------------------------------------------

def test_create_change_composite_dimensions_and_mode(sample_pair):
    """Composite must be exactly 3W x H and in RGB mode."""
    p_a, p_b = sample_pair
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[10:25, 15:35] = 1

    comp = create_change_composite(p_a, p_b, change_mask=mask)

    assert isinstance(comp, Image.Image)
    assert comp.mode == "RGB"
    assert comp.width == 64 * 3  # 192
    assert comp.height == 48
    assert comp.tobytes() != b""  # non-empty output


def test_create_change_composite_size_mismatch_handling(tmp_path):
    """Composite must safely resize mismatched T2 to match T1 dimensions."""
    p_a = tmp_path / "img_a.png"
    p_b = tmp_path / "img_b_diff.png"
    Image.new("RGB", (100, 80), color=(10, 20, 30)).save(p_a)
    Image.new("RGB", (200, 160), color=(200, 100, 50)).save(p_b)

    comp = create_change_composite(p_a, p_b)

    assert comp.width == 100 * 3
    assert comp.height == 80


def test_create_change_composite_with_none_mask(sample_pair):
    """Composite handles change_mask=None gracefully."""
    p_a, p_b = sample_pair
    comp = create_change_composite(p_a, p_b, change_mask=None)
    assert comp.width == 64 * 3
    assert comp.height == 48


def test_create_change_composite_invalid_mask_dimensions_raises(sample_pair):
    """3D or completely invalid mask dimension raises ValueError."""
    p_a, p_b = sample_pair
    invalid_mask = np.zeros((48, 64, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="must be 2-dimensional"):
        create_change_composite(p_a, p_b, change_mask=invalid_mask)


def test_create_change_composite_two_panel(sample_pair):
    """two_panel=True generates a 2W x H RGB composite."""
    p_a, p_b = sample_pair
    comp = create_change_composite(p_a, p_b, two_panel=True)
    assert isinstance(comp, Image.Image)
    assert comp.mode == "RGB"
    assert comp.width == 64 * 2  # 128
    assert comp.height == 48


def test_create_change_composite_subtle_overlay_preserves_rgb(sample_pair):
    """Panel 3 subtle contour must not obliterate underlying image into solid red."""
    p_a, p_b = sample_pair
    mask = np.ones((48, 64), dtype=np.uint8)  # all changed
    comp = create_change_composite(p_a, p_b, change_mask=mask)

    # Panel 2 is T2, Panel 3 is T2 + subtle contour
    comp_arr = np.asarray(comp)
    panel2 = comp_arr[:, 64:128, :]
    panel3 = comp_arr[:, 128:192, :]

    # In solid red overlay, G and B channels drop toward 0 or red is 255
    # With subtle low-opacity fill (~15-18%), G and B should remain close to T2
    assert panel3.shape == panel2.shape
    # Check that panel 3 is not solid red [255, 0, 0]
    assert not np.all(panel3[:, :, 0] == 255)
    # Check green and blue channels still have significant non-zero underlying content
    assert panel3[:, :, 1].mean() > 50


# ---------------------------------------------------------------------------
# B. Prompt Building Tests (Step 9B: Concise 2-Panel Default)
# ---------------------------------------------------------------------------

def test_build_change_vqa_prompt_default_is_concise_two_panel():
    """Default prompt must be concise 2-panel comparison and strictly EXCLUDE telemetry."""
    query = "Were new residential buildings constructed in the north quadrant?"
    prompt = build_change_vqa_prompt(
        query=query,
        changed_pixel_pct=14.25,
        severity="moderate",
        threshold=0.70,
        date_a="2021-06-15",
        date_b="2023-08-20",
    )

    # 1. Concise 2-panel instructions
    assert query in prompt
    assert "left satellite image (Before / T1)" in prompt
    assert "right satellite image (After / T2)" in prompt
    assert "Identify only physical changes visible between the two images" in prompt
    assert "Do not describe image annotations, colors, masks, overlays, or graphics as physical objects" in prompt
    assert "Do not invent exact counts, measurements, or object identities" in prompt
    assert "2021-06-15 to 2023-08-20" in prompt

    # 2. P0/Step 9B: Telemetry is strictly EXCLUDED from prompt sent to VLM
    assert "14.25" not in prompt
    assert "moderate" not in prompt
    assert "0.70" not in prompt
    assert "Siamese" not in prompt
    assert "Quantitative change detection telemetry" not in prompt
    assert "Detailed Change Assessment:" not in prompt


def test_build_change_vqa_prompt_three_panel_optional():
    """Three-panel prompt retains subtle outline guidance when use_three_panel=True."""
    query = "What structures changed?"
    prompt = build_change_vqa_prompt(query=query, use_three_panel=True)
    assert query in prompt
    assert "Left panel = BEFORE (T1)" in prompt
    assert "Center panel = AFTER (T2)" in prompt
    assert "Right panel = T2 with a subtle outline" in prompt
    assert "Use the RIGHT panel only to locate where the change detector identified differences" in prompt


# ---------------------------------------------------------------------------
# C. Orchestrator & Dispatch Tests
# ---------------------------------------------------------------------------

def test_plan_execution_allocates_change_vqa_parameters():
    """plan_execution attaches default temperature and provider to change_vqa."""
    tasks, tools, params, scores = plan_execution(
        query="What changed between the two dates?",
        mode="bi_temporal",
    )
    assert "change_vqa" in tools
    cvqa_params = params.get("change_vqa", {})
    assert cvqa_params.get("temperature") == 0.2
    assert cvqa_params.get("max_tokens") == 768
    assert "provider" in cvqa_params


def test_execute_plan_routes_real_change_vqa(sample_pair):
    """execute_plan invokes run_change_vqa and reports executionMode=real upon VLM success."""
    p_a, p_b = sample_pair

    mock_cvqa_res = ChangeVQAResult(
        answer="Several new structures appeared in the central clearing.",
        confidence=None,
        evidence=["[VLM Reasoning] Real SmolVLM+LoRA inference completed."],
        tool_id="change_vqa",
        is_mock=False,
    )

    with patch("app.services.change_vqa.run_change_vqa", return_value=mock_cvqa_res) as mock_run:
        merged_ans, conf, invocations, boxes, evidence, cmap, modes, stats = execute_plan(
            query="Explain the changes",
            mode="bi_temporal",
            tool_ids=["change_vqa"],
            per_tool_params={"change_vqa": {"provider": "local"}},
            image_file_paths=[p_a, p_b],
            analysis_id="test_cvqa_orch",
        )

        mock_run.assert_called_once()
        assert modes.get("change_vqa") == "real"
        assert conf is None
        cvqa_inv = next(i for i in invocations if i.toolId == "change_vqa")
        assert cvqa_inv.executionMode == "real"


# ---------------------------------------------------------------------------
# D. Confidence Non-Fabrication Tests & Raw VLM Preservation
# ---------------------------------------------------------------------------

def test_change_vqa_confidence_is_strictly_none_and_separates_images(sample_pair):
    """
    run_change_vqa must:
    - Return confidence=None.
    - Preserve raw_vlm_answer without alteration.
    - Track reasoning image (2-panel) and evidence image (3-panel) dimensions.
    """
    p_a, p_b = sample_pair

    mock_vqa_svc_res = MagicMock()
    mock_vqa_svc_res.answer = "Vegetation removal is visible."
    mock_vqa_svc_res.confidence = None
    mock_vqa_svc_res.is_mock = False
    mock_vqa_svc_res.evidence = ["Provider: local"]
    mock_vqa_svc_res.run_context.model_id = "local:SmolVLM"

    with patch("app.services.vqa_service.VQAService.run_real_or_fallback", return_value=mock_vqa_svc_res):
        res = run_change_vqa(
            img_a_path=p_a,
            img_b_path=p_b,
            query="What changed?",
            change_stats={"changed_pixel_pct": 8.5, "severity": "moderate", "threshold_used": 0.70},
            analysis_id="test_conf",
        )

        assert res.confidence is None
        assert not res.is_mock
        assert any("confidence=null" in ev for ev in res.evidence)
        # Raw VLM output preserved without tampering
        assert res.stats["raw_vlm_answer"] == "Vegetation removal is visible."
        # Telemetry is appended to formatted answer post-generation
        assert "Vegetation removal is visible." in res.answer
        assert "8.50%" in res.answer
        assert "moderate" in res.answer
        # Separate reasoning and evidence image tracking
        assert res.stats["reasoning_image_dimensions"] == [64 * 2, 48]  # 2-panel
        assert res.stats["evidence_image_dimensions"] == [64 * 3, 48]   # 3-panel
        assert res.stats["reasoning_image_passed_to_vlm"] == "2-panel (T1 | T2)"


# ---------------------------------------------------------------------------
# E. Failure & Fallback Tests
# ---------------------------------------------------------------------------

def test_change_vqa_graceful_fallback_on_vlm_exception(sample_pair):
    """VLM runtime failure sets is_mock=True and provides quantitative summary without crashing."""
    p_a, p_b = sample_pair

    with patch("app.services.vqa_service.VQAService.run_real_or_fallback", side_effect=RuntimeError("GPU OOM")):
        res = run_change_vqa(
            img_a_path=p_a,
            img_b_path=p_b,
            query="Describe changes",
            change_stats={"changed_pixel_pct": 12.0, "severity": "moderate", "threshold_used": 0.70},
            analysis_id="test_err_fallback",
        )

        assert res.is_mock is True
        assert res.confidence is None
        assert "RuntimeError: GPU OOM" in res.evidence[0]
        assert "12.00%" in res.answer
        assert "moderate" in res.answer
        assert res.stats["raw_vlm_answer"] is None


# ---------------------------------------------------------------------------
# F. End-to-End Local VLM LoRA Path Invocation Test (Step 9B Dual-Artifact)
# ---------------------------------------------------------------------------

def test_change_vqa_passes_2panel_reasoning_image_to_vlm(sample_pair):
    """
    Step 9B Verification:
    Proves run_change_vqa passes the 2-PANEL reasoning image to the VLM by default,
    retains the 3-panel strip as the evidence artifact, and prompt does NOT leak detector telemetry.
    """
    p_a, p_b = sample_pair

    captured_inputs = []

    dummy_loaded = MagicMock()
    dummy_loaded.metadata = {"lora_adapted": True, "num_params_millions": 507.5, "device_actual": "cpu"}
    dummy_loaded.age_sec = 2.0
    dummy_loaded.load_duration_sec = 0.5

    dummy_adapter = MagicMock()
    def mock_preprocess(inf_input, loaded):
        captured_inputs.append(inf_input)
        return {"input_ids": [1, 2, 3]}
    dummy_adapter.preprocess_input = mock_preprocess
    from app.services.vqa_adapter import VQAInferenceOutput
    dummy_adapter.infer.return_value = VQAInferenceOutput(
        answer_text="The visual comparison confirms industrial expansion with new warehouses in T2.",
        confidence=None,
        model_id="local:SmolVLM",
    )

    with patch("app.services.model_manager.ModelManager.load", return_value=dummy_loaded), \
         patch("app.services.vqa_service.get_adapter_for_model", return_value=dummy_adapter), \
         patch("app.services.vqa_service.settings.VQA_MODE", "real"):

        res = run_change_vqa(
            img_a_path=p_a,
            img_b_path=p_b,
            query="What buildings appeared?",
            change_stats={"changed_pixel_pct": 18.2, "severity": "moderate", "threshold_used": 0.70},
            analysis_id="test_step9b_dual_artifact",
            preferred_provider="local",
        )

        assert res.is_mock is False
        assert res.confidence is None
        assert "industrial expansion" in res.answer
        # Proves telemetry is appended AFTER generation in formatted answer
        assert "18.20%" in res.answer
        assert res.stats["raw_vlm_answer"] == "The visual comparison confirms industrial expansion with new warehouses in T2."
        assert len(captured_inputs) == 1

        # Proves the 2-PANEL reasoning strip (64*2 = 128px) was passed to VLM
        assert captured_inputs[0].rgb_image.width == 64 * 2
        assert captured_inputs[0].rgb_image.height == 48

        # Proves the 3-PANEL evidence strip (64*3 = 192px) was synthesized as evidence artifact
        assert res.stats["reasoning_image_dimensions"] == [128, 48]
        assert res.stats["evidence_image_dimensions"] == [192, 48]
        assert res.stats["reasoning_image_passed_to_vlm"] == "2-panel (T1 | T2)"
        assert res.composite_url == "/api/results/change_vqa_composite_test_step9b_dual_artifact.png"

        # P0: Prompt sent to VLM must NOT contain telemetry
        assert "18.20%" not in captured_inputs[0].query_text
        assert "moderate" not in captured_inputs[0].query_text


# ---------------------------------------------------------------------------
# G. Step 16C.2 Tests: Tool Registry, Degenerate Protection & Provenance
# ---------------------------------------------------------------------------

def test_change_vqa_tool_registry_metadata():
    """Tool registry for change_vqa must report available and 0.3.0-p0 without 'mock'."""
    from app.services.tool_registry import get_tool
    tool = get_tool("change_vqa")
    assert tool["status"] == "available"
    assert tool["version"] == "0.3.0-p0"
    assert "mock" not in tool["status"].lower()
    assert "mock" not in tool["version"].lower()
    assert "mock" not in tool["description"].lower()


def test_validate_change_vqa_vlm_output_accepts_natural_language():
    """Natural-language remote-sensing answers must be accepted."""
    from app.services.change_vqa import validate_change_vqa_vlm_output

    sample_1 = "New residential buildings were constructed on former agricultural parcels."
    valid, cleaned, reason = validate_change_vqa_vlm_output(sample_1)
    assert valid is True
    assert cleaned == sample_1
    assert reason is None

    sample_2 = "Vegetation removal and earthworks are visible along the central road corridor."
    valid, cleaned, reason = validate_change_vqa_vlm_output(sample_2)
    assert valid is True
    assert cleaned == sample_2
    assert reason is None


def test_validate_change_vqa_vlm_output_rejects_degenerate_cases():
    """Degenerate outputs like '1.000000', empty text, or coordinates must be rejected."""
    from app.services.change_vqa import validate_change_vqa_vlm_output

    # 1. Lone numeric tokens
    for num_str in ("1.000000", "0.500000", "0.0", "1", "42", "0.70"):
        valid, _, reason = validate_change_vqa_vlm_output(num_str)
        assert valid is False
        assert "isolated numeric" in reason.lower()

    # 2. Empty / whitespace
    for empty_str in ("", "   ", "\n\t", None):
        valid, _, reason = validate_change_vqa_vlm_output(empty_str)
        assert valid is False

    # 3. Coordinate brackets & points
    for coord_str in ("[0.0 0.0, 1.0 1.0]", "<point>(0.2, 0.4)</point>", "(0.5, 0.5)"):
        valid, _, reason = validate_change_vqa_vlm_output(coord_str)
        assert valid is False
        assert "coordinate" in reason.lower() or "natural-language" in reason.lower()


def test_change_vqa_rejects_degenerate_1_000000_in_pipeline(sample_pair):
    """
    When the VLM outputs degenerate '1.000000':
    - Model execution was real (is_mock=False).
    - Confidence remains null (never fabricated).
    - Output is marked as REJECTED with rejection reason.
    - Detector telemetry is preserved separately.
    - mock_specialists is NOT called.
    """
    p_a, p_b = sample_pair

    mock_vqa_svc_res = MagicMock()
    mock_vqa_svc_res.answer = "1.000000"
    mock_vqa_svc_res.confidence = None
    mock_vqa_svc_res.is_mock = False
    mock_vqa_svc_res.evidence = ["Provider: local (SmolVLM with domain-adapted LoRA)."]
    mock_vqa_svc_res.run_context = MagicMock()
    mock_vqa_svc_res.run_context.model_id = "local:HuggingFaceTB/SmolVLM-500M-Instruct"

    with patch("app.services.vqa_service.VQAService.run_real_or_fallback", return_value=mock_vqa_svc_res), \
         patch("app.services.mock_specialists.run_tool") as mock_fallback:

        res = run_change_vqa(
            img_a_path=p_a,
            img_b_path=p_b,
            query="What land-use changes occurred between these two dates?",
            change_stats={"changed_pixel_pct": 26.61, "severity": "high", "threshold_used": 0.70},
            analysis_id="test_reject_1_000000",
        )

        # 1. Real execution preserved
        assert res.is_mock is False
        assert res.confidence is None

        # 2. Validation marked as rejected
        assert res.stats["vlm_validation"] == "rejected"
        assert "isolated numeric" in res.stats["vlm_rejection_reason"].lower()
        assert res.stats["raw_vlm_answer"] == "1.000000"

        # 3. Answer clearly identifies rejection and preserves raw VLM text
        assert "REJECTED" in res.answer
        assert "1.000000" in res.answer
        assert "VLM interpretation unavailable" in res.answer

        # 4. Detector telemetry remains strictly separate
        assert "Detected Changed Area:" in res.answer
        assert "26.61%" in res.answer
        assert "high" in res.answer
        assert "0.70" in res.answer
        assert "SiameseUNet" in res.answer

        # 5. Evidence includes validation failure
        assert any("REJECTED as degenerate" in ev for ev in res.evidence)

        # 6. mock_specialists was NOT invoked
        mock_fallback.assert_not_called()


def test_change_vqa_accepts_valid_natural_language_in_pipeline(sample_pair):
    """
    When the VLM outputs valid natural language:
    - Marked as ACCEPTED.
    - Real execution preserved (is_mock=False, confidence=None).
    - Qualitative interpretation and detector telemetry displayed separately.
    """
    p_a, p_b = sample_pair

    mock_vqa_svc_res = MagicMock()
    mock_vqa_svc_res.answer = "Significant new building construction is visible across the eastern quadrant."
    mock_vqa_svc_res.confidence = None
    mock_vqa_svc_res.is_mock = False
    mock_vqa_svc_res.evidence = ["Provider: local"]
    mock_vqa_svc_res.run_context = MagicMock()
    mock_vqa_svc_res.run_context.model_id = "local:HuggingFaceTB/SmolVLM-500M-Instruct"

    with patch("app.services.vqa_service.VQAService.run_real_or_fallback", return_value=mock_vqa_svc_res):
        res = run_change_vqa(
            img_a_path=p_a,
            img_b_path=p_b,
            query="What land-use changes occurred between these two dates?",
            change_stats={"changed_pixel_pct": 26.61, "severity": "high", "threshold_used": 0.70},
            analysis_id="test_accept_nl",
        )

        assert res.is_mock is False
        assert res.confidence is None
        assert res.stats["vlm_validation"] == "accepted"
        assert res.stats["vlm_rejection_reason"] is None
        assert "ACCEPTED" in res.answer
        assert "Significant new building construction is visible across the eastern quadrant." in res.answer
        assert "26.61%" in res.answer

