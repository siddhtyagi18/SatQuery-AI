"""
backend/tests/test_p0_compliance.py
===================================
Automated verification suite for SatQuery-AI P0 Compliance Requirements:
1. Real Single-Image Captioning
2. Real Optical + SAR Cross-Modal Specialist Analysis
3. Remote-Sensing VLM Domain Adaptation (LoRA)
"""
import shutil
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np
import pytest

from app.schemas import AnalysisMode
from app.services.task_classifier import classify_task
from app.services.tool_registry import get_tool, list_tools
from app.services.optical_sar import (
    _extract_sar_physics,
    _extract_optical_features,
    _create_cross_modal_composite,
    run_optical_sar_analysis,
)
from app.services.orchestrator import plan_execution, execute_plan
from app.services.vqa_service import VQAService


@pytest.fixture
def sample_optical_and_sar_images(tmp_path: Path):
    """Create synthetic optical (RGB) and SAR (grayscale backscatter) images."""
    opt_path = tmp_path / "sample_optical.png"
    sar_path = tmp_path / "sample_sar.png"

    # Optical RGB image
    opt_arr = np.zeros((128, 128, 3), dtype=np.uint8)
    opt_arr[:64, :, 0] = 200  # Red upper
    opt_arr[64:, :, 1] = 220  # Green lower (vegetation)
    Image.fromarray(opt_arr).save(opt_path)

    # SAR backscatter image (high return in center, low return on edges)
    sar_arr = np.full((128, 128), 30, dtype=np.uint8)
    sar_arr[40:88, 40:88] = 240  # Built-up / high backscatter
    sar_arr[:20, :20] = 5        # Specular water / low backscatter
    Image.fromarray(sar_arr, mode="L").save(sar_path)

    return opt_path, sar_path


# ---------------------------------------------------------------------------
# P0-1: Task Classifier & Captioning Routing
# ---------------------------------------------------------------------------

def test_task_classifier_expanded_grounding_keywords():
    tasks, scores = classify_task("Highlight the water body in this image", "single_image")
    assert "grounding" in tasks or scores.get("grounding", 0) > 0


def test_task_classifier_captioning():
    tasks, scores = classify_task("Describe the land-cover and major objects visible in this image", "single_image")
    assert "captioning" in tasks


def test_tool_registry_p0_status():
    caption_tool = get_tool("rs_caption")
    assert caption_tool["status"] == "available"

    sar_tool = get_tool("optical_sar_analyzer")
    assert sar_tool["status"] == "available"

    cd_tool = get_tool("change_detector")
    assert cd_tool["status"] == "available"


# ---------------------------------------------------------------------------
# P0-2: Optical + SAR Cross-Modal Analytics Engine
# ---------------------------------------------------------------------------

def test_sar_physics_extraction():
    # Create test SAR image with known high/low zones
    arr = np.full((100, 100), 10, dtype=np.uint8)
    arr[20:40, 20:40] = 250
    img = Image.fromarray(arr, mode="L")

    stats = _extract_sar_physics(img)
    assert "mean_backscatter_db" in stats
    assert "double_bounce_high_backscatter_pct" in stats
    assert "specular_low_backscatter_pct" in stats
    assert stats["double_bounce_high_backscatter_pct"] > 0
    assert stats["specular_low_backscatter_pct"] > 0


def test_optical_feature_extraction():
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:, :, 1] = 200  # High green channel
    img = Image.fromarray(arr)

    stats = _extract_optical_features(img)
    assert "mean_optical_brightness" in stats
    assert "estimated_vegetation_cover_pct" in stats
    assert stats["estimated_vegetation_cover_pct"] > 50.0


def test_cross_modal_composite_synthesis(tmp_path: Path):
    opt = Image.new("RGB", (64, 64), color=(255, 0, 0))
    sar = Image.new("L", (64, 64), color=128)

    comp_img, comp_url = _create_cross_modal_composite(opt, sar, analysis_id="test_comp")
    assert comp_img.size == (64, 64)
    assert comp_img.mode == "RGB"
    # Channel 0 should be Optical Red (255), Channel 2 should be SAR backscatter (128)
    comp_arr = np.asarray(comp_img)
    assert comp_arr[0, 0, 0] == 255
    assert comp_arr[0, 0, 2] == 128


def test_run_optical_sar_analysis_pipeline(sample_optical_and_sar_images):
    opt_path, sar_path = sample_optical_and_sar_images
    result = run_optical_sar_analysis(
        optical_path=opt_path,
        sar_path=sar_path,
        query="Use optical and SAR to identify structures and water",
        analysis_id="test_opt_sar_pipeline",
    )
    assert result.answer is not None
    assert len(result.answer) > 50
    assert result.confidence is None  # No fabricated confidence
    assert len(result.evidence) >= 4
    assert result.is_mock is False
    assert "mean_backscatter_db" in result.stats


def test_orchestrator_routes_real_optical_sar(sample_optical_and_sar_images):
    opt_path, sar_path = sample_optical_and_sar_images
    query = "Analyze combined optical and SAR sensors"
    mode: AnalysisMode = "optical_sar"

    tasks, tool_ids, params, _ = plan_execution(query, mode)
    assert "optical_sar_analyzer" in tool_ids

    (
        merged_answer,
        agg_conf,
        invocations,
        all_boxes,
        all_evidence,
        change_map_out,
        tool_execution_modes,
        change_stats_out,
    ) = execute_plan(
        query=query,
        mode=mode,
        tool_ids=tool_ids,
        per_tool_params=params,
        tasks=tasks,
        image_file_paths=[opt_path, sar_path],
        analysis_id="test_orch_opt_sar",
    )

    assert "optical_sar_analyzer" in tool_execution_modes
    assert tool_execution_modes["optical_sar_analyzer"] == "real"
    assert "Optical + SAR" in merged_answer or "SAR Radar" in merged_answer


# ---------------------------------------------------------------------------
# P0-3: VLM Domain Adaptation & Training Script
# ---------------------------------------------------------------------------

def test_train_vqa_lora_sampling():
    from scripts.train_vqa_lora import load_bigearthnet_samples
    samples = load_bigearthnet_samples(parquet_path=None, num_samples=20)
    assert len(samples) == 20
    assert "query" in samples[0]
    assert "answer" in samples[0]


def test_vqa_service_supports_captioning_prompt(monkeypatch):
    from app.config import get_settings
    settings = get_settings()
    monkeypatch.setattr(settings, "VQA_MODE", "auto")
    vqa = VQAService()
    assert vqa.should_use_real_vqa("single_image", tasks=["captioning"]) is True

    monkeypatch.setattr(settings, "VQA_MODE", "mock")
    assert vqa.should_use_real_vqa("single_image", tasks=["captioning"]) is False


# ---------------------------------------------------------------------------
# Scientific Integrity & Anti-Fabrication Invariant Tests
# ---------------------------------------------------------------------------

def test_scientific_integrity_mock_confidence_is_strictly_none():
    """Rule 1: If any_real == False (mock mode), confidence must strictly be None."""
    from app.services import mock_specialists
    for tid in ["rs_vqa", "rs_caption", "rs_grounding", "change_detector", "optical_sar_analyzer", "spatial_analyzer"]:
        res = mock_specialists.run_tool(tid, "query", "single_image")
        assert res.get("confidence") is None, f"Tool {tid} fabricated confidence in mock mode: {res.get('confidence')}"

    # Verify aggregation in orchestrator:
    merged_ans, agg_conf, invocations, _, _, _, tool_modes, _ = execute_plan(
        query="Mock query",
        mode="single_image",
        tool_ids=["rs_vqa", "rs_grounding"],
        per_tool_params={},
        tasks=["vqa", "grounding"],
        image_file_paths=[],
    )
    assert agg_conf is None, f"Orchestrator aggregated non-null confidence in mock mode: {agg_conf}"
    assert all(m == "mock" for m in tool_modes.values())


def test_input_validation_modality_unknown_needs_review(tmp_path: Path):
    """Rule 3: If modality cannot be determined, compatibility must be needs_review, never compatible."""
    from app.services.satellite_compatibility import (
        ImageInspectionReport,
        SatelliteCompatibilityService,
    )
    # Synthetic image report with unknown modality
    rep = ImageInspectionReport(
        file_name="unidentified_sensor_raster.bin",
        file_path=str(tmp_path / "unidentified.bin"),
        format="RAW",
        width=256,
        height=256,
        band_count=3,
        dtype="uint8",
        modality_hint="unknown",
        sensor_hint="unknown",
    )
    res = SatelliteCompatibilityService.check_single_compatibility(rep)
    assert res.status == "needs_review", f"Expected needs_review, got {res.status}"
    assert any("modality" in w.lower() for w in res.warnings)

    # In pair mode, unknown modality must halt execution
    rep2 = ImageInspectionReport(
        file_name="unidentified_sensor_raster_t2.bin",
        file_path=str(tmp_path / "unidentified_t2.bin"),
        format="RAW",
        width=256,
        height=256,
        band_count=3,
        dtype="uint8",
        modality_hint="unknown",
        sensor_hint="unknown",
    )
    pair_res = SatelliteCompatibilityService.check_pair_compatibility(rep, rep2, mode="bi_temporal")
    assert pair_res.status == "needs_review", f"Expected needs_review, got {pair_res.status}"

    # Verify orchestrator guardrail halts execution
    ans, conf, invs, _, ev, _, modes, _ = execute_plan(
        query="Detect changes",
        mode="bi_temporal",
        tool_ids=["change_detector"],
        per_tool_params={},
        tasks=["change_detection"],
        image_file_paths=[],
        compatibility_context=pair_res.to_dict(),
    )
    assert conf is None
    assert "NEEDS_REVIEW" in ans
    assert invs[0].toolId == "satellite_compatibility_guardrail"
    assert "change_detector" not in modes
