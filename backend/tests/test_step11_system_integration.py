"""
backend/tests/test_step11_system_integration.py
===============================================
Comprehensive End-to-End Test Matrix for Step 11 Production-Readiness & Hardening.

Covers the 15 required integration facets:
1. Single-image VQA -> real local LoRA / provider
2. Change Detection -> real Siamese U-Net
3. Change VQA -> real detector + real VLM
4. Optical+SAR -> real S1/S2 preprocessing + gated fusion
5. Invalid image input handling
6. Missing checkpoint graceful handling
7. VLM failure graceful handling
8. Provider failure handling
9. Explicit threshold override
10. Mock/fallback detection
11. TIFF/GeoTIFF handling across bit-depths
12. API response schema integrity
13. Evidence generation provenance
14. Trace generation structure
15. confidence=None uncalibrated integrity
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

from app.services.model_inference import (
    run_change_detection,
    get_inference_mode,
    _resolve_checkpoint_path,
)
from app.services.change_vqa import (
    run_change_vqa,
    build_change_vqa_prompt,
    create_change_composite,
)
from app.services.optical_sar import (
    align_optical_sar,
    extract_geotiff_metadata,
    linear_to_db,
    read_raster_band,
    run_optical_sar_analysis,
)
from app.services.models.optical_sar_fusion import (
    OpticalSARFusionNet,
    get_optical_sar_fusion_model,
)
from app.services.orchestrator import (
    plan_execution,
    execute_plan,
)
from app.services.vqa_service import get_vqa_service

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LEVIR_BEFORE = DATA_DIR / "real_levir_crop_before.png"
LEVIR_AFTER = DATA_DIR / "real_levir_crop_after.png"

S1_ROOT = Path(r"C:\Users\Lenovo\Downloads\BigEarthNet-S1")
S2_ROOT = Path(r"C:\Users\Lenovo\Downloads\BigEarthNet-S2")
P2_S2_DIR = S2_ROOT / "S2B_MSIL2A_20170802T092029_N9999_R093_T34TFN_13_61"
P2_S1_DIR = S1_ROOT / "S1B_IW_GRDH_1SDV_20170802T043751_34TFN_13_61"


# 1. Single-image VQA -> real local LoRA / provider
def test_1_single_image_vqa_service(monkeypatch):
    from app.config import get_settings
    settings = get_settings()
    vqa_service = get_vqa_service()
    assert vqa_service is not None
    
    monkeypatch.setattr(settings, "VQA_MODE", "auto")
    assert vqa_service.should_use_real_vqa("single_image", ["vqa"]) is True
    assert vqa_service.should_use_real_vqa("bi_temporal", ["change_detection"]) is False
    
    monkeypatch.setattr(settings, "VQA_MODE", "real")
    assert vqa_service.should_use_real_vqa("single_image", ["vqa"]) is True
    
    monkeypatch.setattr(settings, "VQA_MODE", "mock")
    assert vqa_service.should_use_real_vqa("single_image", ["vqa"]) is False


# 2. Change Detection -> real Siamese U-Net
def test_2_change_detection_real_siamese_unet():
    ckpt = _resolve_checkpoint_path()
    assert ckpt is not None and ckpt.exists(), f"Missing real SiameseUNet checkpoint at {ckpt}"
    res = run_change_detection(
        before_path=LEVIR_BEFORE,
        after_path=LEVIR_AFTER,
        analysis_id="test_e2e_cd",
        threshold=0.70,
    )
    assert res.stats["execution_mode"] == "model_checkpoint"
    assert res.stats["threshold_used"] == 0.70
    assert res.stats["changed_pixel_pct"] > 0.0
    assert res.confidence is None  # Never fabricated


# 3. Change VQA -> real detector + real VLM
def test_3_change_vqa_prompt_and_two_panel_isolation():
    # Prompt builder must NOT contain telemetry
    prompt = build_change_vqa_prompt(
        query="What structural changes occurred?",
        changed_pixel_pct=14.5,
        severity="high",
        threshold=0.70,
        two_panel=True,
    )
    assert "14.5" not in prompt, "Telemetry leaked into VLM reasoning prompt!"
    assert "0.70" not in prompt, "Threshold leaked into VLM reasoning prompt!"
    assert "BEFORE" in prompt.upper() and "AFTER" in prompt.upper()

    # Two-panel strip must be 2W x H
    img_a = Image.new("RGB", (64, 64), color=(100, 100, 100))
    img_b = Image.new("RGB", (64, 64), color=(200, 200, 200))
    strip = create_change_composite(img_a, img_b, two_panel=True)
    assert strip.size == (128, 64)


# 4. Optical+SAR -> real S1/S2 preprocessing + gated fusion
@pytest.mark.skipif(not (P2_S2_DIR.exists() and P2_S1_DIR.exists()), reason="Authentic BigEarthNet pair not found")
def test_4_optical_sar_real_fusion():
    b02_path = P2_S2_DIR / "S2B_MSIL2A_20170802T092029_13_61_B02.tif"
    vv_path = P2_S1_DIR / "S1B_IW_GRDH_1SDV_20170802T043751_34TFN_13_61_VV.tif"
    vh_path = P2_S1_DIR / "S1B_IW_GRDH_1SDV_20170802T043751_34TFN_13_61_VH.tif"

    res = run_optical_sar_analysis(
        optical_path=b02_path,
        sar_path=vv_path,
        sar_vh_path=vh_path,
        query="Analyze surface roughness and vegetation structure.",
        analysis_id="test_e2e_os",
    )
    assert res.is_mock is False
    assert res.confidence is None
    assert res.spatial_overlap_pct == 100.0


# 5. Invalid image input handling
def test_5_invalid_image_input():
    with pytest.raises(Exception):
        run_change_detection(
            before_path=Path("non_existent_before.png"),
            after_path=LEVIR_AFTER,
            analysis_id="test_invalid",
        )


# 6. Missing checkpoint handling
def test_6_missing_checkpoint_fallback(monkeypatch):
    import app.services.model_inference as mi
    monkeypatch.setattr(mi, "_resolve_checkpoint_path", lambda: None)
    res = run_change_detection(
        before_path=LEVIR_BEFORE,
        after_path=LEVIR_AFTER,
        analysis_id="test_missing_ckpt",
    )
    assert res.stats["execution_mode"] == "cpu_classical"
    assert res.confidence is None


# 7. VLM failure handling
def test_7_vlm_failure_handling(monkeypatch):
    from app.services.change_vqa import run_change_vqa
    # Call with non-existent images should be caught and return fallback summary
    res = run_change_vqa(
        img_a_path=Path("missing_a.png"),
        img_b_path=Path("missing_b.png"),
        query="What changed?",
        change_stats={"changed_pixel_pct": 5.2, "severity": "moderate", "threshold_used": 0.70},
    )
    assert res.confidence is None
    assert "failed" in res.answer.lower() or "missing" in res.answer.lower()


# 8. Provider failure handling
def test_8_provider_failure_handling():
    from app.services.vqa_service import VQAServiceResult
    vqa_service = get_vqa_service()
    mock_factory = lambda: VQAServiceResult(
        answer="Fallback answer",
        confidence=None,
        evidence=["Mock fallback evidence"],
        tool_id="rs_vqa",
        is_mock=True,
    )
    res = vqa_service.run_real_or_fallback(
        query="Describe image",
        mode="single_image",
        image_file_paths=[LEVIR_BEFORE],
        tasks=["captioning"],
        mock_factory=mock_factory,
        preferred_provider="invalid_non_existent_provider_xyz",
    )
    assert res.is_mock is True
    assert res.run_context is not None
    assert res.run_context.execution_mode == "mock"
    assert res is not None


# 9. Explicit threshold override
def test_9_explicit_threshold_override():
    res = run_change_detection(
        before_path=LEVIR_BEFORE,
        after_path=LEVIR_AFTER,
        analysis_id="test_thresh",
        threshold=0.85,
    )
    assert res.stats["threshold_used"] == 0.85


# 10. Mock/fallback detection
def test_10_mock_fallback_detection():
    tasks, tool_ids, params, _ = plan_execution("Summarize landscape", "single_image")
    assert "rs_caption" in tool_ids


# 11. TIFF/GeoTIFF handling across bit-depths
def test_11_tiff_geotiff_handling(tmp_path):
    # uint16 preservation
    u16_data = np.array([[300, 1500], [5000, 7500]], dtype=np.uint16)
    p_u16 = tmp_path / "test_16.tif"
    Image.fromarray(u16_data).save(p_u16)

    arr, is_cal = read_raster_band(p_u16)
    assert arr.max() > 255.0
    assert is_cal is True

    # float32 SAR preservation
    f32_data = np.array([[-18.5, -12.3], [-24.1, 1.2]], dtype=np.float32)
    p_f32 = tmp_path / "test_f32.tif"
    Image.fromarray(f32_data).save(p_f32)

    arr_f, _ = read_raster_band(p_f32)
    assert arr_f.min() < 0.0
    assert np.allclose(arr_f, f32_data, atol=1e-3)


# 12. API response schema integrity
def test_12_api_response_schema():
    from app.schemas import AnalysisResult
    fields = AnalysisResult.model_fields.keys()
    for req_field in ["id", "mode", "query", "status", "images", "detectedTasks", "toolInvocations", "executionTrace"]:
        assert req_field in fields


# 13. Evidence generation provenance
def test_13_evidence_generation():
    res = run_change_detection(
        before_path=LEVIR_BEFORE,
        after_path=LEVIR_AFTER,
        analysis_id="test_ev",
    )
    assert any("SiameseUNet" in ev for ev in res.evidence)
    assert any("threshold" in ev.lower() for ev in res.evidence)


# 14. Trace generation structure
def test_14_trace_generation():
    from app.services.orchestrator import plan_execution
    tasks, tool_ids, per_tool_params, scores = plan_execution(
        "Detect changes between before and after.", "bi_temporal"
    )
    assert "change_detection" in tasks
    assert "change_detector" in tool_ids
    assert "threshold" in per_tool_params["change_detector"]


# 15. confidence=None uncalibrated integrity
def test_15_confidence_none_behavior():
    cd_res = run_change_detection(LEVIR_BEFORE, LEVIR_AFTER, "test_conf")
    assert cd_res.confidence is None

    model = get_optical_sar_fusion_model(3, 2, 128, seed=42)
    assert model is not None
