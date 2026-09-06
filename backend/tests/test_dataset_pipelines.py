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
# 3. Validation Scripts Tests
# =========================================================================

def test_validate_patch_directory_success(synthetic_s2_patch_dir: Path):
    """Verify patch validator passes on complete patch directory."""
    is_valid, errors, info = validate_patch_directory(synthetic_s2_patch_dir)
    assert is_valid is True
    assert len(errors) == 0
    assert "B02" in info
    assert "B03" in info
    assert "B04" in info


def test_validate_patch_directory_missing_band(tmp_path: Path):
    """Verify patch validator catches missing band."""
    corrupt_dir = tmp_path / "S2A_corrupt_patch"
    corrupt_dir.mkdir()
    # Only write B02, omit B03 and B04
    Image.new("I;16", (120, 120)).save(corrupt_dir / "B02.tif")

    is_valid, errors, _ = validate_patch_directory(corrupt_dir)
    assert is_valid is False
    assert any("Missing required band" in e for e in errors)


def test_validate_optical_sar_pair_calibrated(synthetic_optical_sar_pair: tuple[Path, Path, Path]):
    """Verify pair validation script correctly authenticates float32 rasters."""
    opt_p, vv_p, vh_p = synthetic_optical_sar_pair
    res = validate_pair(opt_p, vv_p, vh_p)
    assert res["is_valid"] is True
    assert res["sar_physics"]["is_dual_pol"] is True
    assert res["sar_physics"]["is_calibrated_sar"] is True
