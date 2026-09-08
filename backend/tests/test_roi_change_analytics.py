"""
backend/tests/test_roi_change_analytics.py
-----------------------------------------
Unit & regression tests for Interactive Region of Interest (ROI) Change Analytics.

Covers prompt-specified test cases A through O:
A. full-image ROI
B. small ROI
C. zero-change ROI
D. all-change ROI
E. ROI partially containing a change region
F. multiple ROI hotspots
G. invalid coordinates
H. reversed coordinates
I. out-of-bounds coordinates
J. display-to-original coordinate scaling
K. ROI physical area with valid resolution
L. ROI physical area unavailable without resolution
M. original global mask remains unchanged
N. ROI calculation does not invoke Change Detection
O. ROI statistics are deterministic
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import numpy as np
import pytest

from app.services.roi_change_analytics import (
    compute_roi_change_analytics,
    validate_and_clamp_roi,
)


# ---------------------------------------------------------------------------
# Test A: Full-Image ROI
# ---------------------------------------------------------------------------
def test_a_full_image_roi():
    """Full-image ROI should match full-image statistics exactly."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:40, 30:50] = 1  # 20 * 20 = 400 pixels

    bounds = {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}
    res = compute_roi_change_analytics(
        mask, (100, 100), bounds, is_normalized=True, global_changed_pct=4.0
    )

    stats = res["statistics"]
    assert stats["total_pixels"] == 10000
    assert stats["changed_pixels"] == 400
    assert stats["unchanged_pixels"] == 9600
    assert stats["changed_percentage"] == 4.0
    assert stats["unchanged_percentage"] == 96.0

    comp = res["global_comparison"]
    assert comp["global_changed_percentage"] == 4.0
    assert comp["roi_changed_percentage"] == 4.0
    assert comp["difference_percentage"] == 0.0


# ---------------------------------------------------------------------------
# Test B: Small ROI
# ---------------------------------------------------------------------------
def test_b_small_roi():
    """Small ROI captures local change accurately."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:15, 10:15] = 1  # 25 pixels

    # 10x10 ROI around the change
    bounds = {"x1": 10, "y1": 10, "x2": 20, "y2": 20}
    res = compute_roi_change_analytics(
        mask, (100, 100), bounds, is_normalized=False
    )

    stats = res["statistics"]
    assert stats["total_pixels"] == 100
    assert stats["changed_pixels"] == 25
    assert stats["changed_percentage"] == 25.0
    assert stats["unchanged_percentage"] == 75.0


# ---------------------------------------------------------------------------
# Test C: Zero-Change ROI
# ---------------------------------------------------------------------------
def test_c_zero_change_roi():
    """ROI with zero changed pixels returns 0.0% change cleanly."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[50:80, 50:80] = 1  # change elsewhere

    bounds = {"x1": 0.0, "y1": 0.0, "x2": 0.3, "y2": 0.3}
    res = compute_roi_change_analytics(
        mask, (100, 100), bounds, is_normalized=True
    )

    stats = res["statistics"]
    assert stats["changed_pixels"] == 0
    assert stats["changed_percentage"] == 0.0
    assert stats["unchanged_percentage"] == 100.0
    assert res["hotspots"]["hotspots_count_total"] == 0
    assert res["hotspots"]["largest_hotspot"] is None


# ---------------------------------------------------------------------------
# Test D: All-Change ROI
# ---------------------------------------------------------------------------
def test_d_all_change_roi():
    """ROI fully covered in change returns 100.0% change."""
    mask = np.ones((50, 50), dtype=np.uint8)

    bounds = {"x1": 10, "y1": 10, "x2": 30, "y2": 30}
    res = compute_roi_change_analytics(
        mask, (50, 50), bounds, is_normalized=False
    )

    stats = res["statistics"]
    assert stats["total_pixels"] == 400
    assert stats["changed_pixels"] == 400
    assert stats["unchanged_pixels"] == 0
    assert stats["changed_percentage"] == 100.0
    assert stats["unchanged_percentage"] == 0.0


# ---------------------------------------------------------------------------
# Test E: ROI Partially Containing a Change Region
# ---------------------------------------------------------------------------
def test_e_partial_change_roi():
    """ROI partially intersecting a 20x20 change block only counts inside pixels."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:30, 10:30] = 1  # 400 px total change in image

    # ROI covers x: [0, 20], y: [0, 20] -> overlap is [10:20, 10:20] = 100 px
    bounds = {"x1": 0, "y1": 0, "x2": 20, "y2": 20}
    res = compute_roi_change_analytics(
        mask, (100, 100), bounds, is_normalized=False
    )

    stats = res["statistics"]
    assert stats["total_pixels"] == 400
    assert stats["changed_pixels"] == 100
    assert stats["changed_percentage"] == 25.0


# ---------------------------------------------------------------------------
# Test F: Multiple ROI Hotspots
# ---------------------------------------------------------------------------
def test_f_multiple_roi_hotspots():
    """ROI with multiple distinct change components isolates them correctly."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Component 1: 50 px
    mask[10:15, 10:20] = 1
    # Component 2: 100 px
    mask[30:40, 30:40] = 1

    bounds = {"x1": 0, "y1": 0, "x2": 50, "y2": 50}
    res = compute_roi_change_analytics(
        mask, (100, 100), bounds, is_normalized=False, min_hotspot_size_px=10
    )

    hotspots = res["hotspots"]
    assert hotspots["hotspots_count_total"] == 2
    assert hotspots["hotspots_count_significant"] == 2
    assert hotspots["largest_hotspot"] is not None
    assert hotspots["largest_hotspot"]["pixel_area"] == 100
    assert hotspots["largest_hotspot"]["pct_of_roi_change"] == round((100 / 150) * 100.0, 2)


# ---------------------------------------------------------------------------
# Test G: Invalid Coordinates
# ---------------------------------------------------------------------------
def test_g_invalid_coordinates():
    """Non-numeric or degenerate coordinates raise clear ValueError."""
    with pytest.raises(ValueError):
        validate_and_clamp_roi({"x1": "invalid", "y1": 0, "x2": 10, "y2": 10}, 100, 100)

    # Point selection (zero width and height)
    with pytest.raises(ValueError, match="zero width or height"):
        validate_and_clamp_roi({"x1": 10, "y1": 10, "x2": 10, "y2": 10}, 100, 100, is_normalized=False)


# ---------------------------------------------------------------------------
# Test H: Reversed Coordinates
# ---------------------------------------------------------------------------
def test_h_reversed_coordinates():
    """Reversed coordinates (e.g. user dragged right-to-left) are safely normalized."""
    x1, y1, x2, y2 = validate_and_clamp_roi(
        {"x1": 80, "y1": 90, "x2": 20, "y2": 30}, 100, 100, is_normalized=False
    )
    assert x1 == 20
    assert y1 == 30
    assert x2 == 80
    assert y2 == 90


# ---------------------------------------------------------------------------
# Test I: Out-of-Bounds Coordinates
# ---------------------------------------------------------------------------
def test_i_out_of_bounds_coordinates():
    """Coordinates outside the image boundary are clamped safely to image bounds."""
    x1, y1, x2, y2 = validate_and_clamp_roi(
        {"x1": -50, "y1": -20, "x2": 150, "y2": 120}, 100, 100, is_normalized=False
    )
    assert x1 == 0
    assert y1 == 0
    assert x2 == 100
    assert y2 == 100


# ---------------------------------------------------------------------------
# Test J: Display-to-Original Coordinate Scaling
# ---------------------------------------------------------------------------
def test_j_display_to_original_coordinate_scaling():
    """Normalized [0, 1] coordinates map accurately to image dimensions."""
    # Given image of 1024 x 768
    x1, y1, x2, y2 = validate_and_clamp_roi(
        {"x1": 0.25, "y1": 0.50, "x2": 0.75, "y2": 1.00}, 1024, 768, is_normalized=True
    )
    assert x1 == 256
    assert y1 == 384
    assert x2 == 768
    assert y2 == 768


# ---------------------------------------------------------------------------
# Test K: ROI Physical Area with Valid Resolution
# ---------------------------------------------------------------------------
def test_k_roi_physical_area_valid_resolution():
    """When GSD metadata is present, computes accurate physical area."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[0:10, 0:10] = 1  # 100 px change

    # ROI of 20 x 20 = 400 px, GSD = 10.0m (Sentinel-2 style)
    # 1 pixel = 100 m^2
    # Total ROI area = 400 * 100 = 40,000 m^2 = 4.0 hectares
    # Changed ROI area = 100 * 100 = 10,000 m^2 = 1.0 hectare
    meta = {"gsdMeters": 10.0}
    bounds = {"x1": 0, "y1": 0, "x2": 20, "y2": 20}
    res = compute_roi_change_analytics(
        mask, (100, 100), bounds, is_normalized=False, metadata=meta
    )

    pa = res["physical_area"]
    assert pa["available"] is True
    assert pa["gsd_meters"] == 10.0
    assert pa["roi_total_area_m2"] == 40000.0
    assert pa["roi_total_area_hectares"] == 4.0
    assert pa["roi_changed_area_m2"] == 10000.0
    assert pa["roi_changed_area_hectares"] == 1.0


# ---------------------------------------------------------------------------
# Test L: ROI Physical Area Unavailable Without Resolution
# ---------------------------------------------------------------------------
def test_l_roi_physical_area_unavailable_without_resolution():
    """When no GSD is provided, physical area is explicitly unavailable without fabricating."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    bounds = {"x1": 0, "y1": 0, "x2": 20, "y2": 20}
    res = compute_roi_change_analytics(
        mask, (100, 100), bounds, is_normalized=False, metadata=None
    )

    pa = res["physical_area"]
    assert pa["available"] is False
    assert pa["gsd_meters"] is None
    assert pa["roi_total_area_m2"] is None
    assert "reliable spatial resolution metadata was not provided" in pa["reason"]


# ---------------------------------------------------------------------------
# Test M: Original Global Mask Remains Unchanged
# ---------------------------------------------------------------------------
def test_m_mask_immutability():
    """ROI extraction must never mutate the original mask array."""
    mask = np.random.randint(0, 2, size=(80, 80), dtype=np.uint8)
    mask_copy = copy.deepcopy(mask)
    initial_hash = hashlib.sha256(mask.tobytes()).hexdigest()

    bounds = {"x1": 0.1, "y1": 0.1, "x2": 0.8, "y2": 0.8}
    _ = compute_roi_change_analytics(mask, (80, 80), bounds, is_normalized=True)

    post_hash = hashlib.sha256(mask.tobytes()).hexdigest()
    assert initial_hash == post_hash
    np.testing.assert_array_equal(mask, mask_copy)


# ---------------------------------------------------------------------------
# Test N: ROI Calculation Does Not Invoke Change Detection
# ---------------------------------------------------------------------------
def test_n_roi_does_not_invoke_change_detection(monkeypatch):
    """ROI calculation must consume existing mask and NEVER rerun model inference."""
    import app.services.model_inference as mi

    def _forbidden_call(*args, **kwargs):
        raise AssertionError("Change Detection model inference was illegally invoked during ROI calculation!")

    monkeypatch.setattr(mi, "run_change_detection", _forbidden_call)

    mask = np.ones((50, 50), dtype=np.uint8)
    bounds = {"x1": 0.2, "y1": 0.2, "x2": 0.6, "y2": 0.6}

    # Should run smoothly without triggering the monkeypatched forbidden function
    res = compute_roi_change_analytics(mask, (50, 50), bounds, is_normalized=True)
    assert res["statistics"]["changed_pixels"] > 0


# ---------------------------------------------------------------------------
# Test O: ROI Statistics Are Deterministic
# ---------------------------------------------------------------------------
def test_o_roi_statistics_are_deterministic():
    """Identical ROI coordinates over the same mask return bit-for-bit identical stats."""
    np.random.seed(42)
    mask = np.random.randint(0, 2, size=(64, 64), dtype=np.uint8)
    bounds = {"x1": 16, "y1": 16, "x2": 48, "y2": 48}

    res1 = compute_roi_change_analytics(mask, (64, 64), bounds, is_normalized=False)
    res2 = compute_roi_change_analytics(mask, (64, 64), bounds, is_normalized=False)

    assert res1["statistics"] == res2["statistics"]
    assert res1["hotspots"]["hotspots_count_total"] == res2["hotspots"]["hotspots_count_total"]
    assert res1["roi"] == res2["roi"]


# ---------------------------------------------------------------------------
# Test P: API Endpoint Integration
# ---------------------------------------------------------------------------
def test_p_api_roi_endpoint(monkeypatch):
    """Test the POST /api/analysis/{analysis_id}/roi-analysis endpoint."""
    from fastapi.testclient import TestClient
    from app.main import app
    from app.database import get_db
    from app.models import Analysis

    client = TestClient(app)

    # 1. Invalid coordinates return 400
    # First query DB for an existing analysis or create a dummy session
    db_gen = get_db()
    db = next(db_gen)
    analysis = db.query(Analysis).filter(Analysis.mode == "bi_temporal").first()
    if analysis is None:
        from app.services.change_detection import _ensure_results_dir
        from PIL import Image
        r_dir = _ensure_results_dir()
        dummy_mask_path = r_dir / "test_roi_endpoint_changemap.png"
        Image.new("RGBA", (100, 100), (255, 0, 0, 128)).save(dummy_mask_path)

        analysis = Analysis(
            id="test_roi_endpoint_id",
            mode="bi_temporal",
            query="Detect change",
            status="completed",
            change_map={
                "overlayUrl": "/api/results/test_roi_endpoint_changemap.png",
                "changed_pixel_pct": 50.0,
            },
        )
        db.add(analysis)
        db.commit()

    aid = analysis.id

    # Test valid ROI
    resp = client.post(
        f"/api/analysis/{aid}/roi-analysis",
        json={"x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.5, "is_normalized": True, "run_vqa": False},
    )
    assert resp.status_code == 200, f"Failed: {resp.text}"
    data = resp.json()
    assert "roi" in data
    assert "statistics" in data
    assert "physical_area" in data
    assert "global_comparison" in data
    assert "hotspots" in data
    assert data["vqa"] is None

    # Test invalid coordinates (point selection)
    bad_resp = client.post(
        f"/api/analysis/{aid}/roi-analysis",
        json={"x1": 0.2, "y1": 0.2, "x2": 0.2, "y2": 0.2, "is_normalized": True},
    )
    assert bad_resp.status_code == 400
    assert "zero width or height" in bad_resp.json()["detail"]

    # Test non-existent analysis
    nf_resp = client.post(
        "/api/analysis/non_existent_id_999/roi-analysis",
        json={"x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.5},
    )
    assert nf_resp.status_code == 404

