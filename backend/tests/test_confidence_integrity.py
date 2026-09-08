"""
backend/tests/test_confidence_integrity.py
=========================================
Strict regression suite enforcing confidence integrity across SatQuery-AI.
Verifies that:
1. Real VQA returns confidence is None.
2. Real Change Detection returns confidence is None.
3. Real Change VQA returns confidence is None.
4. Real Optical+SAR returns confidence is None.
5. Orchestrator overall aggregated confidence remains None across all modes.
6. Generated reports contain no fabricated percentage confidence (e.g. 93%, 82%).
7. Operating threshold 0.70 is not exposed as a confidence score.
8. Heuristic image quality scores cannot leak into model confidence.
9. Bi-temporal analysis isolates change tools and prevents single-image reports.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image
import pytest

from app.schemas import AnalysisMode
from app.services.image_analysis import analyze_satellite_image
from app.services.model_inference import run_change_detection
from app.services.orchestrator import plan_execution, execute_plan
from app.services.vqa_service import get_vqa_service


@pytest.fixture
def sample_optical_pair(tmp_path: Path):
    """Create a temporary pair of synthetic optical images."""
    p_before = tmp_path / "before.png"
    p_after = tmp_path / "after.png"
    arr1 = np.full((128, 128, 3), 100, dtype=np.uint8)
    arr2 = np.full((128, 128, 3), 150, dtype=np.uint8)
    Image.fromarray(arr1).save(p_before)
    Image.fromarray(arr2).save(p_after)
    return p_before, p_after


@pytest.fixture
def sample_optical_and_sar(tmp_path: Path):
    """Create synthetic optical and SAR image pair."""
    opt_path = tmp_path / "opt.png"
    sar_path = tmp_path / "sar.png"
    opt_arr = np.full((128, 128, 3), 120, dtype=np.uint8)
    sar_arr = np.full((128, 128), 80, dtype=np.uint8)
    Image.fromarray(opt_arr).save(opt_path)
    Image.fromarray(sar_arr, mode="L").save(sar_path)
    return opt_path, sar_path


def test_image_analysis_confidence_is_strictly_none(sample_optical_pair):
    """Heuristic image_analysis engine must return confidence is None and no fabricated %."""
    p_before, _ = sample_optical_pair
    res = analyze_satellite_image(p_before, query="What land cover is visible?")
    
    assert res["confidence"] is None, "Heuristic image_analysis must set confidence=None"
    assert "image_quality_score" in res["stats"], "Internal quality score must be named image_quality_score"
    assert "confidence" not in res["stats"] or res["stats"].get("confidence") is None
    
    forbidden = ["93%", "82%", "calibrated at", "calibrated confidence", "observation certainty: 93%"]
    for fb in forbidden:
        assert fb not in res["answer"], f"Found forbidden '{fb}' in image_analysis answer"
    assert "Confidence: Not calibrated for this analysis." in res["answer"]


def test_vqa_service_confidence_is_strictly_none(sample_optical_pair):
    """Real VQA service execution must always return confidence is None."""
    p_before, _ = sample_optical_pair
    vqa = get_vqa_service()
    res = vqa.run_real_or_fallback(
        query="What land cover is visible in this scene?",
        mode="single_image",
        image_file_paths=[p_before],
        tasks=["vqa"],
        tool_id="rs_vqa",
    )
    assert res.confidence is None, "VQA service must return confidence=None"
    assert "93%" not in res.answer, "VQA answer must not contain fabricated 93% confidence"
    assert "calibrated at" not in res.answer.lower()


def test_change_detection_confidence_is_strictly_none(sample_optical_pair):
    """Siamese U-Net / change detection must return confidence is None, operating threshold 0.70 is not confidence."""
    p_before, p_after = sample_optical_pair
    res = run_change_detection(p_before, p_after, analysis_id="test_integrity_cd")
    
    assert res.confidence is None, "Change detection must return confidence=None"
    assert res.stats["threshold_used"] == 0.70
    # The threshold must not be described as a confidence score
    assert "confidence = 70%" not in res.answer.lower()
    assert "70% confidence" not in res.answer.lower()
    assert "operating threshold: 0.70" in res.answer.lower()


def test_optical_sar_confidence_is_strictly_none(sample_optical_and_sar):
    """Optical+SAR fusion must return confidence is None."""
    opt_path, sar_path = sample_optical_and_sar
    mode: AnalysisMode = "optical_sar"
    tasks, tool_ids, params, _ = plan_execution("Analyze optical and SAR cross-modal features", mode)
    
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
        query="Analyze optical and SAR cross-modal features",
        mode=mode,
        tool_ids=tool_ids,
        per_tool_params=params,
        tasks=tasks,
        image_file_paths=[opt_path, sar_path],
    )
    
    assert agg_conf is None, "Optical+SAR overall confidence must be None"
    assert "91%" not in merged_answer, "Forbidden 91% confidence claim in Optical+SAR answer"


def test_orchestrator_overall_confidence_is_strictly_none_across_modes(sample_optical_pair):
    """Orchestrator aggregated confidence MUST be None for all modes."""
    p_before, p_after = sample_optical_pair
    
    for mode in ("single_image", "bi_temporal"):
        img_paths = [p_before] if mode == "single_image" else [p_before, p_after]
        tasks, tool_ids, params, _ = plan_execution("Detect features", mode)
        
        _, agg_conf, _, _, _, _, _, _ = execute_plan(
            query="Detect features",
            mode=mode,
            tool_ids=tool_ids,
            per_tool_params=params,
            tasks=tasks,
            image_file_paths=img_paths,
        )
        assert agg_conf is None, f"Orchestrator overall confidence must be None for mode '{mode}'"


def test_bi_temporal_tool_isolation():
    """Bi-temporal planning must NOT select single-image VQA or captioning tools."""
    tasks, tools, params, _ = plan_execution(
        "Identify and report any land cover changes or new building structures between the baseline and follow-up acquisitions.",
        "bi_temporal",
    )
    assert "change_detector" in tools, "change_detector must be selected for bi_temporal"
    assert "change_vqa" in tools, "change_vqa must be selected for bi_temporal"
    assert "rs_vqa" not in tools, "rs_vqa must not be selected in bi_temporal mode"
    assert "rs_caption" not in tools, "rs_caption must not be selected in bi_temporal mode"
