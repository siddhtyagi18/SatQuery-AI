"""
backend/tests/test_geospatial_change_analytics.py
-------------------------------------------------
Comprehensive test suite for the Geo-Spatial Change Analytics service.

Verifies:
A. All-zero mask
B. All-one mask
C. Simple known changed region
D. Multiple connected components
E. Largest hotspot identification
F. Hotspot percentage calculations
G. Empty-change case without errors
H. Physical area with valid resolution
I. Physical area unavailable without resolution
J. Missing metadata handling
K. Mask dimensions & input validation
L. Golden baseline regression check (Siamese U-Net output matches exactly)
"""
from pathlib import Path
import numpy as np
import pytest

from app.services.geospatial_change_analytics import (
    compute_geospatial_change_analytics,
    _extract_epsg_code,
)


def test_a_all_zero_mask():
    """All-zero mask: no changes, 0 hotspots, largest region is None, valid structure."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    analytics = compute_geospatial_change_analytics(mask, (100, 100))

    stats = analytics["global_statistics"]
    assert stats["width"] == 100
    assert stats["height"] == 100
    assert stats["total_pixel_count"] == 10000
    assert stats["changed_pixel_count"] == 0
    assert stats["unchanged_pixel_count"] == 10000
    assert stats["changed_percentage"] == 0.0
    assert stats["unchanged_percentage"] == 100.0

    assert analytics["hotspots_count_total"] == 0
    assert analytics["hotspots_count_filtered"] == 0
    assert analytics["largest_change_region"] is None
    assert analytics["hotspots"] == []
    assert analytics["change_density"]["mean_change_density_pct"] == 0.0


def test_b_all_one_mask():
    """All-one mask: 100% changed, exactly 1 hotspot covering the entire area."""
    mask = np.ones((50, 80), dtype=np.uint8)
    analytics = compute_geospatial_change_analytics(mask, (80, 50))

    stats = analytics["global_statistics"]
    assert stats["width"] == 80
    assert stats["height"] == 50
    assert stats["total_pixel_count"] == 4000
    assert stats["changed_pixel_count"] == 4000
    assert stats["unchanged_pixel_count"] == 0
    assert stats["changed_percentage"] == 100.0

    assert analytics["hotspots_count_total"] == 1
    largest = analytics["largest_change_region"]
    assert largest is not None
    assert largest["pixel_area"] == 4000
    assert largest["percentage_of_total_changed"] == 100.0
    assert largest["bounding_box"]["width"] == 80
    assert largest["bounding_box"]["height"] == 50


def test_c_simple_known_changed_region():
    """A single known 10x20 rectangular patch at (x=10..30, y=5..15)."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[5:15, 10:30] = 1  # area = 10 * 20 = 200 px
    analytics = compute_geospatial_change_analytics(mask, (100, 100))

    stats = analytics["global_statistics"]
    assert stats["changed_pixel_count"] == 200
    assert stats["total_pixel_count"] == 10000
    assert stats["changed_percentage"] == 2.0

    assert analytics["hotspots_count_total"] == 1
    largest = analytics["largest_change_region"]
    assert largest["pixel_area"] == 200
    assert largest["bounding_box"]["x_min"] == 10
    assert largest["bounding_box"]["y_min"] == 5
    assert largest["bounding_box"]["width"] == 20
    assert largest["bounding_box"]["height"] == 10
    # Centroid: x_center = 10 + (20-1)/2 = 19.5, y_center = 5 + (10-1)/2 = 9.5
    assert pytest.approx(largest["centroid_px"]["x"], 0.1) == 19.5
    assert pytest.approx(largest["centroid_px"]["y"], 0.1) == 9.5


def test_d_multiple_connected_components():
    """Three separate non-touching changed regions."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[5:10, 5:10] = 1    # 5x5 = 25 px
    mask[20:30, 20:30] = 1  # 10x10 = 100 px
    mask[50:80, 50:70] = 1  # 30x20 = 600 px

    analytics = compute_geospatial_change_analytics(mask, (100, 100), min_hotspot_size_px=0)

    assert analytics["global_statistics"]["changed_pixel_count"] == 725
    assert analytics["hotspots_count_total"] == 3
    assert len(analytics["hotspots"]) == 3

    # Ensure sorted descending by area
    areas = [h["pixel_area"] for h in analytics["hotspots"]]
    assert areas == [600, 100, 25]


def test_e_largest_hotspot_identification():
    """Verify largest hotspot accurately identified among multiple components."""
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[2:6, 2:6] = 1      # 16 px
    mask[15:35, 15:35] = 1  # 400 px (largest)
    mask[45:50, 45:50] = 1  # 25 px

    analytics = compute_geospatial_change_analytics(mask, (60, 60))
    largest = analytics["largest_change_region"]

    assert largest is not None
    assert largest["pixel_area"] == 400
    assert largest["bounding_box"]["width"] == 20
    assert largest["bounding_box"]["height"] == 20


def test_f_hotspot_percentages_sum_to_100():
    """The percentage contributions of all individual hotspots sum to 100% of total change."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[0:10, 0:10] = 1    # 100 px
    mask[30:50, 30:50] = 1  # 400 px
    mask[70:80, 70:80] = 1  # 100 px
    total_changed = 600

    analytics = compute_geospatial_change_analytics(mask, (100, 100), min_hotspot_size_px=0)
    pct_sum = sum(h["percentage_of_total_changed"] for h in analytics["hotspots"])
    assert pytest.approx(pct_sum, 0.001) == 100.0


def test_g_empty_change_case_returns_valid_structure():
    """Empty mask must never raise an exception, returning clean empty structures."""
    mask = np.zeros((1, 1), dtype=np.uint8)
    analytics = compute_geospatial_change_analytics(mask, (1, 1))

    assert analytics["largest_change_region"] is None
    assert analytics["hotspots"] == []
    assert analytics["global_statistics"]["changed_pixel_count"] == 0
    assert analytics["global_statistics"]["changed_percentage"] == 0.0


def test_h_physical_area_with_valid_resolution():
    """Physical area calculated correctly when spatial metadata contains gsd_meters."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[0:20, 0:50] = 1  # 1000 px changed

    # 10m spatial resolution (e.g. Sentinel-2) -> 1 px = 100 m²
    metadata = {
        "crs": "EPSG:32643",
        "gsd_meters": 10.0,
    }
    analytics = compute_geospatial_change_analytics(mask, (100, 100), metadata=metadata)
    pa = analytics["physical_area"]

    assert pa["physical_area_available"] is True
    assert pa["area_unavailable_reason"] is None
    assert pa["pixel_area_m2"] == 100.0
    # 1000 pixels * 100 m² = 100,000 m² = 10.0 ha = 0.1 km²
    assert pa["changed_area_m2"] == 100000.0
    assert pa["changed_area_ha"] == 10.0
    assert pa["changed_area_km2"] == 0.1
    assert "ha" in pa["formatted_summary"]


def test_i_physical_area_unavailable_without_resolution():
    """Physical area explicitly unavailable when spatial resolution is omitted."""
    mask = np.ones((50, 50), dtype=np.uint8)
    analytics = compute_geospatial_change_analytics(mask, (50, 50), metadata={})
    pa = analytics["physical_area"]

    assert pa["physical_area_available"] is False
    assert pa["changed_area_m2"] is None
    assert "reliable spatial resolution metadata was not provided" in pa["area_unavailable_reason"]
    assert "N/A" in pa["formatted_summary"]


def test_j_metadata_missing_handling():
    """When metadata is None, spatial fields are explicitly None without fabricating values."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    analytics = compute_geospatial_change_analytics(mask, (10, 10), metadata=None)

    geo_meta = analytics["geospatial_metadata"]
    assert geo_meta["crs"] is None
    assert geo_meta["epsg_code"] is None
    assert geo_meta["resolution"] is None
    assert geo_meta["transform"] is None
    assert geo_meta["image_bounds"] is None


def test_k_mask_dimensions_and_filtering():
    """Hotspot filtering respects min_hotspot_size_px without mutating total change."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[0:2, 0:2] = 1      # 4 px (tiny noise)
    mask[10:13, 10:13] = 1  # 9 px (tiny noise)
    mask[50:60, 50:60] = 1  # 100 px (significant)

    analytics = compute_geospatial_change_analytics(mask, (100, 100), min_hotspot_size_px=10)

    # Total hotspots before filtering: 3
    assert analytics["hotspots_count_total"] == 3
    # Filtered hotspots: only the 100px one
    assert analytics["hotspots_count_filtered"] == 1
    assert len(analytics["hotspots"]) == 1
    assert analytics["hotspots"][0]["pixel_area"] == 100
    # Original total changed pixel count remains complete
    assert analytics["global_statistics"]["changed_pixel_count"] == 113


def test_l_epsg_code_extraction():
    """Verify EPSG integer parsing from various standard CRS strings."""
    assert _extract_epsg_code("EPSG:32643") == 32643
    assert _extract_epsg_code("EPSG:4326") == 4326
    assert _extract_epsg_code("urn:ogc:def:crs:EPSG::3857") == 3857
    assert _extract_epsg_code("WGS 84") is None
    assert _extract_epsg_code(None) is None


def test_m_golden_baseline_regression():
    """
    Regression test: verify Siamese U-Net change detection on baseline optical pair
    produces identical outputs when consumed by Geo-Spatial Change Analytics.
    """
    p1 = Path("public/demo/optical_before.jpg")
    p2 = Path("public/demo/optical_after.jpg")
    if not p1.exists():
        p1 = Path("../public/demo/optical_before.jpg")
        p2 = Path("../public/demo/optical_after.jpg")

    if not (p1.exists() and p2.exists()):
        pytest.skip("Baseline demo assets not found")

    from app.services.model_inference import run_change_detection
    cd_result = run_change_detection(p1, p2, analysis_id="regression_analytics_test")

    # Verify existing Golden Baseline outputs remain 100% identical
    assert cd_result.stats["changed_pixel_count"] == 71495
    assert cd_result.stats["total_pixel_count"] == 786432
    assert cd_result.stats["changed_pixel_pct"] == 9.09
    assert cd_result.stats["unchanged_pixel_pct"] == 90.91
    assert cd_result.stats["threshold_used"] == 0.70
    assert cd_result.stats["execution_mode"] == "model_checkpoint"
    assert cd_result.confidence is None
