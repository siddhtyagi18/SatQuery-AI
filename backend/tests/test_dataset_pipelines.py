"""
backend/tests/test_dataset_pipelines.py
======================================
Unit and smoke tests for BigEarthNet multimodal loader, Optical + SAR scientific pipeline,
and dataset validation scripts.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.services.datasets.bigearthnet_multimodal import (
    BigEarthNetMultimodalDataset,
    BigEarthNetSample,
    assemble_sentinel2_rgb,
    _percentile_stretch_uint8,
)
from app.services.optical_sar import (
    linear_to_db,
    extract_sar_polarimetric_physics,
    extract_geotiff_metadata,
    compute_spatial_overlap,
    create_cross_modal_composite,
    GeoSpatialMetadata,
    PolarimetricSARFeatures,
    AlignedMultimodalPackage,
    normalize_optical,
    align_optical_sar,
    create_diagnostic_visualization,
    run_optical_sar_analysis,
)
from scripts.validate_bigearthnet_multimodal import (
    check_parquet_metadata,
    validate_patch_directory,
    run_validation as run_ben_validation,
)
from scripts.validate_optical_sar_pairs import (
    validate_pair,
    run_pairs_validation,
)


@pytest.fixture
def synthetic_s2_patch_dir(tmp_path: Path) -> Path:
    """Create a temporary synthetic Sentinel-2 patch folder with B02, B03, B04, B08."""
    patch_dir = tmp_path / "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57"
    patch_dir.mkdir(parents=True)

    rng = np.random.RandomState(42)
    for band in ["B02", "B03", "B04", "B08"]:
        # 120x120 uint16 surface reflectance [0, 10000]
        data = (rng.uniform(100, 3000, size=(120, 120))).astype(np.uint16)
        img = Image.fromarray(data)
        img.save(patch_dir / f"{band}.tif")

    return patch_dir


@pytest.fixture
def synthetic_optical_sar_pair(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create synthetic optical RGB and SAR VV/VH float32 TIFFs."""
    rng = np.random.RandomState(123)

    # 1. Optical RGB (256x256 uint8)
    opt_path = tmp_path / "scene_001_opt.tif"
    opt_data = rng.randint(0, 255, size=(256, 256, 3), dtype=np.uint8)
    Image.fromarray(opt_data, mode="RGB").save(opt_path)

    # 2. SAR VV float32 linear intensity
    sar_vv_path = tmp_path / "scene_001_sar_vv.tif"
    vv_data = rng.uniform(0.001, 0.5, size=(256, 256)).astype(np.float32)
    Image.fromarray(vv_data).save(sar_vv_path)

    # 3. SAR VH float32 linear intensity
    sar_vh_path = tmp_path / "scene_001_sar_vh.tif"
    vh_data = rng.uniform(0.0001, 0.1, size=(256, 256)).astype(np.float32)
    Image.fromarray(vh_data).save(sar_vh_path)

    return opt_path, sar_vv_path, sar_vh_path


# =========================================================================
# 1. BigEarthNet Multimodal Loader Tests
# =========================================================================

def test_percentile_stretch_uint8():
    """Verify robust contrast stretching handles arbitrary float ranges."""
    arr = np.linspace(0.0, 10000.0, 1000, dtype=np.float32)
    stretched = _percentile_stretch_uint8(arr, pmin=2.0, pmax=98.0)
    assert stretched.dtype == np.uint8
    assert stretched.min() == 0
    assert stretched.max() == 255


def test_assemble_sentinel2_rgb(synthetic_s2_patch_dir: Path):
    """Verify assembly of RGB composite from band GeoTIFFs."""
    rgb_img, stats = assemble_sentinel2_rgb(synthetic_s2_patch_dir)
    assert isinstance(rgb_img, Image.Image)
    assert rgb_img.size == (120, 120)
    assert rgb_img.mode == "RGB"
    assert "B02_mean" in stats
    assert "B03_mean" in stats
    assert "B04_mean" in stats


def test_bigearthnet_dataset_missing_root(tmp_path: Path):
    """Verify loader initializes gracefully when image folder does not exist."""
    # Point to a dummy parquet or real parquet if available
    real_parquet = Path("C:/Users/Lenovo/Downloads/BigEarthNet.txt.parquet")
    if not real_parquet.exists():
        pytest.skip("BigEarthNet.txt.parquet not found on host machine.")

    ds = BigEarthNetMultimodalDataset(
        parquet_path=real_parquet,
        images_root=tmp_path / "non_existent_images",
        max_samples=5,
    )
    # When images_root has no matching directories, dataset indexing succeeds without error
    assert len(ds) >= 0


# =========================================================================
# 2. Optical + SAR Physics Pipeline Tests
# =========================================================================

def test_linear_to_db():
    """Verify mathematical fidelity of decibel backscatter conversion."""
    intensity = np.array([1.0, 0.1, 0.01, 0.001], dtype=np.float32)
    db = linear_to_db(intensity)
    assert np.isclose(db[0], 0.0, atol=1e-3)
    assert np.isclose(db[1], -10.0, atol=1e-3)
    assert np.isclose(db[2], -20.0, atol=1e-3)
    assert np.isclose(db[3], -30.0, atol=1e-3)


def test_extract_sar_polarimetric_physics_dual_pol():
    """Verify dual-polarization cross-ratio and physical thresholding."""
    vv_arr = np.full((50, 50), 0.1, dtype=np.float32)  # -10 dB
    vh_arr = np.full((50, 50), 0.02, dtype=np.float32) # -16.99 dB

    features = extract_sar_polarimetric_physics(vv_arr, vh_arr, is_raw_amplitude=False)
    assert features.is_calibrated is True
    assert features.is_dual_pol is True
    assert np.isclose(features.mean_vv_db, -10.0, atol=0.1)
    assert features.vh_vv_ratio_mean is not None
    assert np.isclose(features.vh_vv_ratio_mean, 0.2, atol=0.01)
    assert features.vv_vh_diff_db is not None
    assert np.isclose(features.vv_vh_diff_db, 6.99, atol=0.2)


def test_compute_spatial_overlap():
    """Verify bounding box intersection calculations."""
    meta1 = GeoSpatialMetadata(crs="EPSG:32633", bounds=(0.0, 0.0, 100.0, 100.0))
    meta2 = GeoSpatialMetadata(crs="EPSG:32633", bounds=(50.0, 0.0, 150.0, 100.0))
    overlap = compute_spatial_overlap(meta1, meta2)
    assert overlap is not None
    assert np.isclose(overlap, 50.0, atol=0.1)


def test_create_cross_modal_composite():
    """Verify synthesis of 3-channel false-color composite."""
    opt_img = Image.new("RGB", (64, 64), color=(200, 100, 50))
    sar_vv = np.random.uniform(0.01, 0.8, size=(64, 64)).astype(np.float32)
    comp_img, _ = create_cross_modal_composite(opt_img, sar_vv, analysis_id="test_smoke")
    assert isinstance(comp_img, Image.Image)
    assert comp_img.size == (64, 64)
    assert comp_img.mode == "RGB"


def test_run_optical_sar_analysis_calibrated(synthetic_optical_sar_pair: tuple[Path, Path, Path]):
    """Smoke test executing end-to-end optical SAR analysis on calibrated float TIFFs."""
    opt_p, vv_p, vh_p = synthetic_optical_sar_pair
    result = run_optical_sar_analysis(
        optical_path=opt_p,
        sar_path=vv_p,
        query="Analyze surface scattering and water presence",
        sar_vh_path=vh_p,
        analysis_id="test_exec",
    )
    assert result.is_mock is False
    assert result.is_calibrated_sar is True
    assert "mean_backscatter_db" in result.stats
    assert "specular_low_backscatter_pct" in result.stats
    assert len(result.evidence) > 0


# =========================================================================
# 3. Comprehensive Step 10 Tests A through P
# =========================================================================

def test_a_valid_optical_sar_pair(synthetic_optical_sar_pair: tuple[Path, Path, Path]):
    """Test A: Verify validation of a complete and valid Optical-SAR pair."""
    opt_p, vv_p, vh_p = synthetic_optical_sar_pair
    res = validate_pair(opt_p, vv_p, vh_p)
    assert res["is_valid"] is True
    assert len(res["errors"]) == 0
    assert res["sar_physics"]["is_dual_pol"] is True
    assert res["alignment_status"] == "ALIGNED_COMMON_GRID"


def test_b_missing_optical_file(tmp_path: Path, synthetic_optical_sar_pair: tuple[Path, Path, Path]):
    """Test B: Verify clear error reporting when optical file is missing."""
    _, vv_p, vh_p = synthetic_optical_sar_pair
    missing_opt = tmp_path / "non_existent_optical.tif"
    res = validate_pair(missing_opt, vv_p, vh_p)
    assert res["is_valid"] is False
    assert any("Optical file missing" in err for err in res["errors"])


def test_c_missing_vv_file(tmp_path: Path, synthetic_optical_sar_pair: tuple[Path, Path, Path]):
    """Test C: Verify clear error reporting when SAR VV file is missing."""
    opt_p, _, vh_p = synthetic_optical_sar_pair
    missing_vv = tmp_path / "non_existent_vv.tif"
    res = validate_pair(opt_p, missing_vv, vh_p)
    assert res["is_valid"] is False
    assert any("SAR VV file missing" in err for err in res["errors"])


def test_d_missing_vh_file(synthetic_optical_sar_pair: tuple[Path, Path, Path]):
    """Test D: Single-polarization (VV only, missing/omitted VH) is gracefully supported."""
    opt_p, vv_p, _ = synthetic_optical_sar_pair
    res = validate_pair(opt_p, vv_p, None)
    assert res["is_valid"] is True
    assert res["sar_physics"]["is_dual_pol"] is False
    assert res["sar_physics"]["vh_vv_cross_ratio"] is None


def test_e_invalid_crs():
    """Test E: Mismatched or invalid CRS returns 0.0 spatial overlap."""
    meta1 = GeoSpatialMetadata(crs="EPSG:32635", bounds=(100.0, 100.0, 200.0, 200.0))
    meta2 = GeoSpatialMetadata(crs="EPSG:32633", bounds=(100.0, 100.0, 200.0, 200.0))
    overlap = compute_spatial_overlap(meta1, meta2)
    assert overlap == 0.0


def test_f_non_overlapping_extents():
    """Test F: Spatially disjoint bounding boxes in the same CRS return 0.0 overlap."""
    meta1 = GeoSpatialMetadata(crs="EPSG:32635", bounds=(0.0, 0.0, 50.0, 50.0))
    meta2 = GeoSpatialMetadata(crs="EPSG:32635", bounds=(100.0, 100.0, 150.0, 150.0))
    overlap = compute_spatial_overlap(meta1, meta2)
    assert overlap == 0.0


def test_g_mismatched_dimensions():
    """Test G: Resampling aligns mismatched optical and SAR grids deterministically."""
    opt_arr = np.ones((256, 256, 3), dtype=np.float32) * 128.0
    sar_vv = np.zeros((120, 120), dtype=np.float32) - 10.0
    pkg = align_optical_sar(optical=opt_arr, sar_vv=sar_vv, target_grid="sar")
    assert pkg.height == 120
    assert pkg.width == 120
    assert pkg.optical.shape == (120, 120, 3)
    assert pkg.sar_vv.shape == (120, 120)
    assert pkg.preprocessing["optical_resampled"] is True


def test_h_linear_sar_conversion():
    """Test H: Linear power SAR input is correctly converted to decibels."""
    linear_arr = np.full((32, 32), 0.1, dtype=np.float32)  # 0.1 linear power = -10 dB
    features = extract_sar_polarimetric_physics(linear_arr, sar_representation="linear_power")
    assert features.sar_representation == "linear_power"
    assert np.isclose(features.mean_vv_db, -10.0, atol=0.01)


def test_i_already_db_sar_input_no_double_log():
    """Test I: Input already in decibels does NOT get log-converted again."""
    db_arr = np.full((32, 32), -15.5, dtype=np.float32)
    features = extract_sar_polarimetric_physics(db_arr, sar_representation="auto")
    assert features.sar_representation == "sigma0_db"
    assert np.isclose(features.mean_vv_db, -15.5, atol=0.01)
    assert np.isclose(features.min_vv_db, -15.5, atol=0.01)


def test_j_deterministic_optical_normalization():
    """Test J: Deterministic optical normalization records scale, offset, and normalization applied."""
    s2_l2a_dn = np.array([0, 1000, 5000, 10000], dtype=np.uint16)
    norm, meta = normalize_optical(s2_l2a_dn, method="sentinel2_l2a")
    assert np.allclose(norm, [0.0, 0.1, 0.5, 1.0])
    assert meta["scale"] == 0.0001
    assert meta["normalization_applied"] == "sentinel2_l2a_scale_10000"

    uint8_img = np.array([0, 128, 255], dtype=np.uint8)
    norm_u8, meta_u8 = normalize_optical(uint8_img, method="uint8_scale")
    assert np.isclose(norm_u8[0], 0.0)
    assert np.isclose(norm_u8[-1], 1.0)
    assert meta_u8["normalization_applied"] == "uint8_scale_255"


def test_k_alignment_preserves_crs():
    """Test K: Alignment preserves the coordinate reference system of the target grid."""
    sar_meta = GeoSpatialMetadata(crs="EPSG:32635", transform=(505980.0, 10.0, 0.0, 7079640.0, 0.0, -10.0))
    opt_arr = np.ones((64, 64, 3), dtype=np.float32)
    vv_arr = np.ones((64, 64), dtype=np.float32) * -12.0
    pkg = align_optical_sar(optical=opt_arr, sar_vv=vv_arr, sar_meta=sar_meta)
    assert pkg.crs == "EPSG:32635"


def test_l_alignment_preserves_spatial_extent_semantics():
    """Test L: Alignment preserves spatial transform and bounds."""
    expected_bounds = (505980.0, 7078440.0, 507180.0, 7079640.0)
    expected_transform = (505980.0, 10.0, 0.0, 7079640.0, 0.0, -10.0)
    sar_meta = GeoSpatialMetadata(crs="EPSG:32635", bounds=expected_bounds, transform=expected_transform)
    opt_arr = np.ones((64, 64, 3), dtype=np.float32)
    vv_arr = np.ones((64, 64), dtype=np.float32)
    pkg = align_optical_sar(optical=opt_arr, sar_vv=vv_arr, sar_meta=sar_meta)
    assert pkg.bounds == expected_bounds
    assert pkg.transform == expected_transform


def test_m_bigearthnet_band_loading(synthetic_s2_patch_dir: Path, tmp_path: Path):
    """Test M: BigEarthNet band loading reads B02, B03, B04, B08 and fails on missing band."""
    rgb_img, stats = assemble_sentinel2_rgb(synthetic_s2_patch_dir)
    assert rgb_img.size == (120, 120)
    assert "B02_mean" in stats and "B04_mean" in stats

    corrupt_dir = tmp_path / "corrupt_s2_patch"
    corrupt_dir.mkdir()
    # Missing required bands
    with pytest.raises(FileNotFoundError):
        assemble_sentinel2_rgb(corrupt_dir)


def test_n_multimodal_output_contains_all_three_modalities():
    """Test N: Multimodal output package contains optical, SAR VV, and SAR VH modalities."""
    opt_arr = np.zeros((32, 32, 3), dtype=np.float32)
    vv_arr = np.zeros((32, 32), dtype=np.float32) - 10.0
    vh_arr = np.zeros((32, 32), dtype=np.float32) - 16.0
    pkg = align_optical_sar(optical=opt_arr, sar_vv=vv_arr, sar_vh=vh_arr)
    assert "sentinel-2-optical" in pkg.modalities
    assert "sentinel-1-vv" in pkg.modalities
    assert "sentinel-1-vh" in pkg.modalities
    assert pkg.optical is not None
    assert pkg.sar_vv is not None
    assert pkg.sar_vh is not None


def test_o_metadata_is_preserved():
    """Test O: Complete source and preprocessing metadata are preserved in output package."""
    opt_arr = np.zeros((32, 32, 3), dtype=np.float32)
    vv_arr = np.zeros((32, 32), dtype=np.float32) - 12.0
    pkg = align_optical_sar(optical=opt_arr, sar_vv=vv_arr)
    d = pkg.to_dict()
    assert "source_metadata" in d
    assert "preprocessing" in d
    assert "optical_normalization" in d["preprocessing"]
    assert "sar_physics" in d["preprocessing"]


def test_p_no_fake_mock_multimodal_data(synthetic_optical_sar_pair: tuple[Path, Path, Path]):
    """Test P: Analysis result has is_mock=False and confidence=None (no fabricated metrics)."""
    opt_p, vv_p, vh_p = synthetic_optical_sar_pair
    result = run_optical_sar_analysis(
        optical_path=opt_p,
        sar_path=vv_p,
        query="Assess surface roughness and radar scattering",
        sar_vh_path=vh_p,
        analysis_id="test_integrity",
    )
    assert result.is_mock is False
    assert result.confidence is None  # Strictly no fabricated confidence
    assert "mean_backscatter_db" in result.stats
    assert result.stats.get("sar_representation") in ("linear_power", "sigma0_db", "uncalibrated_8bit_proxy")
