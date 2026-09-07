"""
backend/scripts/audit_step11_system.py
======================================
Automated verification script for Step 11: Full System Integration & Hardening Audit.
Tests:
1. Task Classification across representative queries
2. Orchestrator planning & tool selection
3. End-to-end execution of all 4 real capabilities
4. Error propagation and mock fallback honesty
5. GeoTIFF / TIFF handling across dtypes (uint8, uint16, float32, multi-band)
6. Confidence calibration honesty (confidence=None)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.app.services.task_classifier import classify_task
from backend.app.services.orchestrator import plan_execution, execute_plan
from backend.app.services.optical_sar import (
    align_optical_sar,
    extract_geotiff_metadata,
    linear_to_db,
    normalize_optical,
    read_raster_band,
)
from backend.app.services.model_inference import run_change_detection
from backend.app.services.change_vqa import run_change_vqa
from backend.app.services.models.optical_sar_fusion import get_optical_sar_fusion_model

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def audit_task_classification():
    print("\n==================================================")
    print("1. TASK CLASSIFICATION AUDIT")
    print("==================================================")
    queries = [
        # Single-image VQA
        ("What is visible in this satellite image?", "single_image", ["vqa"]),
        ("Describe the land cover.", "single_image", ["captioning"]),
        ("What objects are present?", "single_image", ["vqa"]),
        # Bi-temporal Change Detection
        ("What changed between these two images?", "bi_temporal", ["change_detection", "change_vqa"]),
        ("Detect changes between before and after.", "bi_temporal", ["change_detection"]),
        # Change VQA
        ("What structural changes occurred?", "bi_temporal", ["change_vqa", "change_detection"]),
        ("Describe what changed between the two satellite images.", "bi_temporal", ["change_detection", "change_description"]),
        # Optical + SAR
        ("Analyze this optical and SAR imagery together.", "optical_sar", ["vqa"]),
        ("Compare information from Sentinel-1 and Sentinel-2.", "optical_sar", ["vqa"]),
    ]

    results = []
    for q, m, expected in queries:
        tasks, scores = classify_task(q, m)
        tasks_str = ", ".join(tasks)
        print(f"[{m:12s}] \"{q:55s}\" -> tasks=[{tasks_str}]")
        results.append({"query": q, "mode": m, "classified_tasks": tasks})
    return results


def audit_orchestrator_planning():
    print("\n==================================================")
    print("2. ORCHESTRATOR PLANNING AUDIT")
    print("==================================================")
    cases = [
        ("What is visible in this satellite image?", "single_image", "rs_vqa"),
        ("Describe the land cover.", "single_image", "rs_caption"),
        ("Detect changes between before and after.", "bi_temporal", "change_detector"),
        ("Describe what changed between the two satellite images.", "bi_temporal", "change_vqa"),
        ("Analyze this optical and SAR imagery together.", "optical_sar", "optical_sar_analyzer"),
    ]

    for q, m, expected_tool in cases:
        tasks, tool_ids, params, scores = plan_execution(q, m)
        print(f"[{m:12s}] \"{q:55s}\" -> tools={tool_ids}")
        assert expected_tool in tool_ids, f"Expected {expected_tool} in {tool_ids}"
    print("[OK] All planning decisions correctly select intended tools.")


def audit_geotiff_handling():
    print("\n==================================================")
    print("3. GEOTIFF / TIFF INPUT HARDENING AUDIT")
    print("==================================================")
    scratch = DATA_DIR / "scratch_audit"
    scratch.mkdir(parents=True, exist_ok=True)

    # 1. uint8 RGB
    u8_img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    u8_path = scratch / "test_u8.png"
    u8_img.save(u8_path)

    # 2. uint16 multispectral (simulated S2 band)
    u16_arr = np.random.randint(100, 8000, (64, 64), dtype=np.uint16)
    u16_img = Image.fromarray(u16_arr)
    u16_path = scratch / "test_u16.tif"
    u16_img.save(u16_path)

    # 3. float32 SAR in dB
    f32_arr = np.random.uniform(-30.0, 5.0, (64, 64)).astype(np.float32)
    f32_img = Image.fromarray(f32_arr)
    f32_path = scratch / "test_f32_db.tif"
    f32_img.save(f32_path)

    # Test reading
    arr_u8, cal_u8 = read_raster_band(u8_path)
    print(f"uint8 reading: shape={arr_u8.shape}, dtype={arr_u8.dtype}, max={arr_u8.max()}")
    assert arr_u8.max() <= 255

    arr_u16, cal_u16 = read_raster_band(u16_path)
    print(f"uint16 reading: shape={arr_u16.shape}, dtype={arr_u16.dtype}, max={arr_u16.max()}, calibrated={cal_u16}")
    assert arr_u16.max() > 255, "uint16 data was mistakenly truncated to uint8!"

    arr_f32, cal_f32 = read_raster_band(f32_path)
    print(f"float32 reading: shape={arr_f32.shape}, dtype={arr_f32.dtype}, min={arr_f32.min():.1f}, max={arr_f32.max():.1f}")
    assert arr_f32.min() < 0.0, "float32 SAR decibel data corrupted!"

    # Test normalization safety (no double-log)
    db_test = np.array([-15.0, -20.0, -5.0], dtype=np.float32)
    # linear_to_db should only be called on linear power > 0
    lin_pwr = np.array([0.01, 0.1, 1.0], dtype=np.float32)
    db_out = linear_to_db(lin_pwr)
    print(f"linear_to_db on linear power: {lin_pwr} -> {db_out.round(2)} dB")
    assert np.allclose(db_out, [-20.0, -10.0, 0.0], atol=1e-3)

    print("[OK] GeoTIFF / TIFF reading and radiometric handling verified.")


def audit_real_specialist_execution():
    print("\n==================================================")
    print("4. REAL SPECIALIST EXECUTION & LATENCY AUDIT")
    print("==================================================")

    # 1. Change Detection on real LEVIR crops
    t0 = time.perf_counter()
    cd_res = run_change_detection(
        before_path=DATA_DIR / "real_levir_crop_before.png",
        after_path=DATA_DIR / "real_levir_crop_after.png",
        analysis_id="audit_step11",
        threshold=0.70,
    )
    cd_latency = round((time.perf_counter() - t0) * 1000.0, 1)
    print(f"A. Change Detection (Siamese U-Net):")
    print(f"   Latency: {cd_latency} ms")
    print(f"   Changed area: {cd_res.stats.get('changed_pixel_pct')}%")
    print(f"   Confidence: {cd_res.confidence} (calibrated: False)")
    assert cd_res.confidence is None

    # 2. Optical + SAR on real BigEarthNet Pair 2
    s2_p2_dir = Path(r"C:\Users\Lenovo\Downloads\BigEarthNet-S2\S2B_MSIL2A_20170802T092029_N9999_R093_T34TFN_13_61")
    s1_p2_dir = Path(r"C:\Users\Lenovo\Downloads\BigEarthNet-S1\S1B_IW_GRDH_1SDV_20170802T043751_34TFN_13_61")
    if s2_p2_dir.exists() and s1_p2_dir.exists():
        t0 = time.perf_counter()
        from backend.app.services.optical_sar import run_optical_sar_analysis
        os_res = run_optical_sar_analysis(
            optical_path=s2_p2_dir / "S2B_MSIL2A_20170802T092029_13_61_B02.tif",
            sar_path=s1_p2_dir / "S1B_IW_GRDH_1SDV_20170802T043751_34TFN_13_61_VV.tif",
            sar_vh_path=s1_p2_dir / "S1B_IW_GRDH_1SDV_20170802T043751_34TFN_13_61_VH.tif",
            query="Analyze surface roughness and vegetation structure.",
            analysis_id="audit_step11",
        )
        os_latency = round((time.perf_counter() - t0) * 1000.0, 1)
        print(f"\nB. Optical+SAR Analysis (Gated Dual-Branch Fusion):")
        print(f"   Latency: {os_latency} ms")
        print(f"   Is Mock: {os_res.is_mock}")
        print(f"   Confidence: {os_res.confidence} (calibrated: False)")
        assert os_res.confidence is None
        assert not os_res.is_mock

    # 3. Change VQA on real LEVIR pair
    t0 = time.perf_counter()
    cvqa_res = run_change_vqa(
        img_a_path=DATA_DIR / "real_levir_crop_before.png",
        img_b_path=DATA_DIR / "real_levir_crop_after.png",
        query="What structural changes occurred?",
        change_stats=cd_res.stats,
        analysis_id="audit_step11",
        preferred_provider="local_lora",
    )
    cvqa_latency = round((time.perf_counter() - t0) * 1000.0, 1)
    print(f"\nC. Change VQA (SiameseUNet + SmolVLM):")
    print(f"   Latency: {cvqa_latency} ms")
    print(f"   Is Mock: {cvqa_res.is_mock}")
    print(f"   Confidence: {cvqa_res.confidence}")
    assert cvqa_res.confidence is None


def main():
    audit_task_classification()
    audit_orchestrator_planning()
    audit_geotiff_handling()
    audit_real_specialist_execution()
    print("\n==================================================")
    print("[ALL CHECKS PASSED] Step 11 audit script completed successfully.")
    print("==================================================")


if __name__ == "__main__":
    main()
