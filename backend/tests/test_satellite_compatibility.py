"""
backend/tests/test_satellite_compatibility.py
=============================================
Rigorous verification of the Satellite Compatibility & Adaptation Layer.
Tests all 22 required failure, edge, and standard cases:

 1. PNG RGB
 2. JPEG RGB
 3. uint16 TIFF
 4. float32 TIFF
 5. GeoTIFF with CRS
 6. GeoTIFF without CRS
 7. Grayscale
 8. Multispectral
 9. Corrupt image
10. Unsupported extension
11. Mismatched image dimensions
12. Mismatched CRS
13. Non-overlapping GeoTIFFs
14. Same-time pair
15. Missing temporal metadata
16. Invalid temporal ordering
17. Excessive nodata
18. SAR already in dB
19. SAR linear power
20. Missing VV
21. Missing VH
22. Unsupported sensor/modality / LEVIR domain guardrail
"""
import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.services.satellite_compatibility import (
    SatelliteImageInspector,
    SatelliteCompatibilityService,
    SatelliteInputAdapter,
    TemporalValidator,
    SpatialValidator,
    ImageInspectionReport,
)


@pytest.fixture
def test_dir(tmp_path: Path) -> Path:
    d = tmp_path / "sat_compat_test"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Helpers to generate test files
# ---------------------------------------------------------------------------

def create_png_rgb(path: Path, size=(256, 256)):
    arr = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    im = Image.fromarray(arr, mode="RGB")
    im.save(path, format="PNG")
    return path


def create_jpeg_rgb(path: Path, size=(256, 256)):
    arr = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    im = Image.fromarray(arr, mode="RGB")
    im.save(path, format="JPEG")
    return path


def create_uint16_tiff(path: Path, size=(256, 256)):
    arr = np.random.randint(0, 4000, (size[1], size[0]), dtype=np.uint16)
    im = Image.fromarray(arr)
    im.save(path, format="TIFF")
    return path


def create_float32_tiff(path: Path, size=(256, 256)):
    arr = np.random.uniform(0.0, 1.0, (size[1], size[0])).astype(np.float32)
    im = Image.fromarray(arr, mode="F")
    im.save(path, format="TIFF")
    return path


def create_geotiff_mock(path: Path, with_crs: bool = True, bounds=(100.0, 100.0, 200.0, 200.0), epsg=32630):
    arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    im = Image.fromarray(arr, mode="RGB")
    tiffinfo = Image.Exif()
    # ModelPixelScaleTag (33550): (scale_x, scale_y, 0)
    # ModelTiepointTag (33922): (0, 0, 0, ox, oy, 0)
    sx = (bounds[2] - bounds[0]) / 128.0
    sy = (bounds[3] - bounds[1]) / 128.0
    ox, oy = bounds[0], bounds[3]
    tiffinfo[33550] = (sx, sy, 0.0)
    tiffinfo[33922] = (0.0, 0.0, 0.0, ox, oy, 0.0)
    if with_crs:
        # GeoKeyDirectoryTag (34735) with EPSG
        tiffinfo[34735] = (1, 1, 0, 1, 3072, 0, 1, epsg)
    im.save(path, format="TIFF", tiffinfo=tiffinfo)
    return path


# ---------------------------------------------------------------------------
# Test Cases 1 - 22
# ---------------------------------------------------------------------------

def test_case_1_png_rgb(test_dir: Path):
    """Case 1: Standard PNG RGB image inspection & adaptation."""
    p = create_png_rgb(test_dir / "scene_rgb.png")
    rep = SatelliteImageInspector.inspect(p)

    assert not rep.is_corrupted
    assert rep.format == "PNG"
    assert rep.width == 256 and rep.height == 256
    assert rep.band_count == 3
    assert rep.modality_hint == "rgb_optical"

    comp = SatelliteCompatibilityService.check_single_compatibility(rep)
    assert comp.status == "compatible"
    assert "rs_vqa" in comp.specialist_candidates

    arr, tele = SatelliteInputAdapter.adapt_optical_to_numpy(rep)
    assert arr.shape == (256, 256, 3)
    assert arr.dtype == np.float32
    assert 0.0 <= arr.min() and arr.max() <= 1.0


def test_case_2_jpeg_rgb(test_dir: Path):
    """Case 2: Standard JPEG RGB image inspection & adaptation."""
    p = create_jpeg_rgb(test_dir / "scene_rgb.jpg")
    rep = SatelliteImageInspector.inspect(p)

    assert not rep.is_corrupted
    assert rep.format in ("JPG", "JPEG")
    assert rep.width == 256 and rep.height == 256
    assert rep.band_count == 3

    comp = SatelliteCompatibilityService.check_single_compatibility(rep)
    assert comp.status == "compatible"


def test_case_3_uint16_tiff(test_dir: Path):
    """Case 3: uint16 TIFF (e.g. raw Sentinel-2 DN)."""
    p = create_uint16_tiff(test_dir / "s2_band_uint16.tif")
    rep = SatelliteImageInspector.inspect(p)

    assert not rep.is_corrupted
    assert rep.bit_depth == 16
    assert rep.max_val > 255.0

    comp = SatelliteCompatibilityService.check_single_compatibility(rep)
    assert comp.status == "adaptable"
    assert any("16" in a or "float32" in a for a in comp.required_adaptations)

    arr, tele = SatelliteInputAdapter.adapt_optical_to_numpy(rep)
    assert arr.shape == (256, 256, 3)
    assert arr.dtype == np.float32
    assert arr.max() <= 1.0
    assert tele.adapted_dtype == "float32"


def test_case_4_float32_tiff(test_dir: Path):
    """Case 4: float32 TIFF inspection & adaptation."""
    p = create_float32_tiff(test_dir / "surface_reflectance.tif")
    rep = SatelliteImageInspector.inspect(p)

    assert not rep.is_corrupted
    assert rep.bit_depth == 32 or "F" in rep.dtype

    comp = SatelliteCompatibilityService.check_single_compatibility(rep)
    assert comp.status in ("compatible", "adaptable")

    arr, tele = SatelliteInputAdapter.adapt_optical_to_numpy(rep)
    assert arr.shape == (256, 256, 3)
    assert arr.dtype == np.float32


def test_case_5_geotiff_with_crs(test_dir: Path):
    """Case 5: GeoTIFF with embedded CRS & georeferencing."""
    p = create_geotiff_mock(test_dir / "geo_with_crs.tif", with_crs=True, epsg=32630)
    rep = SatelliteImageInspector.inspect(p)

    assert not rep.is_corrupted
    assert rep.is_geotiff
    assert rep.crs is not None
    assert "32630" in rep.crs or "GeoTIFF" in rep.crs
    assert rep.bounds is not None
    assert rep.resolution is not None


def test_case_6_geotiff_without_crs(test_dir: Path):
    """Case 6: GeoTIFF with scale/tiepoint but missing CRS."""
    p = create_geotiff_mock(test_dir / "geo_no_crs.tif", with_crs=False)
    rep = SatelliteImageInspector.inspect(p)

    assert not rep.is_corrupted
    assert rep.bounds is not None
    assert rep.crs is None


def test_case_7_grayscale(test_dir: Path):
    """Case 7: Single-channel grayscale image."""
    arr = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
    p = test_dir / "panchromatic.png"
    Image.fromarray(arr, mode="L").save(p)

    rep = SatelliteImageInspector.inspect(p)
    assert rep.band_count == 1
    assert rep.modality_hint == "grayscale"

    comp = SatelliteCompatibilityService.check_single_compatibility(rep)
    assert comp.status == "adaptable"
    assert "single_channel_to_pseudo_rgb" in comp.required_adaptations

    adapted, tele = SatelliteInputAdapter.adapt_optical_to_numpy(rep)
    assert adapted.shape == (128, 128, 3)
    assert np.allclose(adapted[..., 0], adapted[..., 1])


def test_case_8_multispectral(test_dir: Path):
    """Case 8: Multispectral image (>3 bands)."""
    # Create 4-channel image
    arr = np.random.randint(0, 255, (128, 128, 4), dtype=np.uint8)
    p = test_dir / "sentinel2_b02_b03_b04_b08.png"
    Image.fromarray(arr, mode="RGBA").save(p)

    rep = SatelliteImageInspector.inspect(p)
    assert rep.band_count == 4
    assert rep.modality_hint in ("multispectral_optical", "rgb_optical")

    adapted, tele = SatelliteInputAdapter.adapt_optical_to_numpy(rep)
    assert adapted.shape == (128, 128, 3)
    assert tele.selected_bands == [1, 2, 3]


def test_case_9_corrupt_image(test_dir: Path):
    """Case 9: Corrupt / truncated image file."""
    p = test_dir / "corrupted_satellite.png"
    with open(p, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRcorrupted_garbage_bytes")

    rep = SatelliteImageInspector.inspect(p)
    assert rep.is_corrupted
    assert rep.error_message is not None

    comp = SatelliteCompatibilityService.check_single_compatibility(rep)
    assert comp.status == "invalid"
    assert any("corrupted" in r.lower() for r in comp.reasons)


def test_case_10_unsupported_extension(test_dir: Path):
    """Case 10: Unsupported file extension."""
    p = test_dir / "satellite_model.xyz"
    p.write_text("dummy 3D point cloud")

    rep = SatelliteImageInspector.inspect(p)
    assert rep.is_corrupted
    assert "unsupported file extension" in rep.error_message.lower()


def test_case_11_mismatched_image_dimensions(test_dir: Path):
    """Case 11: Bi-temporal pair with mismatched dimensions."""
    p1 = create_png_rgb(test_dir / "before_256.png", size=(256, 256))
    p2 = create_png_rgb(test_dir / "after_512.png", size=(512, 512))

    rep1 = SatelliteImageInspector.inspect(p1)
    rep2 = SatelliteImageInspector.inspect(p2)

    pair_comp = SatelliteCompatibilityService.check_pair_compatibility(rep1, rep2, mode="bi_temporal")
    assert pair_comp.dimension_status == "mismatch"
    assert "spatial_resampling_to_match_dimensions" in pair_comp.required_adaptations
    assert pair_comp.status == "adaptable"

    # Adapt pair
    arr1, arr2, tele1, tele2 = SatelliteInputAdapter.adapt_bitemporal_pair(rep1, rep2)
    assert arr1.shape == arr2.shape == (256, 256, 3)
    assert tele2.resampling is not None


def test_case_12_mismatched_crs(test_dir: Path):
    """Case 12: Mismatched CRS between image A and image B."""
    p1 = create_geotiff_mock(test_dir / "t1_utm.tif", with_crs=True, epsg=32630)
    p2 = create_geotiff_mock(test_dir / "t2_wgs84.tif", with_crs=True, epsg=4326)

    rep1 = SatelliteImageInspector.inspect(p1)
    rep2 = SatelliteImageInspector.inspect(p2)

    overlap, crs_status = SpatialValidator.compute_overlap(rep1, rep2)
    assert crs_status.startswith("mismatched_crs")
    assert overlap == 0.0


def test_case_13_non_overlapping_geotiffs(test_dir: Path):
    """Case 13: Non-overlapping geographic bounds (0% overlap)."""
    p1 = create_geotiff_mock(test_dir / "scene_london.tif", bounds=(0.0, 51.0, 0.5, 51.5))
    p2 = create_geotiff_mock(test_dir / "scene_paris.tif", bounds=(2.0, 48.0, 2.5, 48.5))

    rep1 = SatelliteImageInspector.inspect(p1)
    rep2 = SatelliteImageInspector.inspect(p2)

    pair_comp = SatelliteCompatibilityService.check_pair_compatibility(rep1, rep2, mode="bi_temporal")
    assert pair_comp.spatial_overlap_pct == 0.0
    assert any("non-overlap" in l.lower() for l in pair_comp.limitations)


def test_case_14_same_time_pair(test_dir: Path):
    """Case 14: Identical acquisition dates."""
    p1 = create_png_rgb(test_dir / "scene_20230501_A.png")
    p2 = create_png_rgb(test_dir / "scene_20230501_B.png")

    rep1 = SatelliteImageInspector.inspect(p1)
    rep2 = SatelliteImageInspector.inspect(p2)

    pair_comp = SatelliteCompatibilityService.check_pair_compatibility(rep1, rep2, mode="bi_temporal")
    assert pair_comp.temporal_status == "same_time"
    assert any("identical acquisition dates" in w.lower() for w in pair_comp.warnings)


def test_case_15_missing_temporal_metadata(test_dir: Path):
    """Case 15: Missing dates in metadata -> 'unknown' temporal status without guessing."""
    p1 = create_png_rgb(test_dir / "patch_alpha.png")
    p2 = create_png_rgb(test_dir / "patch_beta.png")

    rep1 = SatelliteImageInspector.inspect(p1)
    rep2 = SatelliteImageInspector.inspect(p2)

    pair_comp = SatelliteCompatibilityService.check_pair_compatibility(rep1, rep2, mode="bi_temporal")
    assert pair_comp.temporal_status == "unknown"


def test_case_16_invalid_temporal_ordering(test_dir: Path):
    """Case 16: Inverted temporal ordering (T1 later than T2)."""
    p1 = create_png_rgb(test_dir / "scene_20240101_after.png")
    p2 = create_png_rgb(test_dir / "scene_20200101_before.png")

    rep1 = SatelliteImageInspector.inspect(p1)
    rep2 = SatelliteImageInspector.inspect(p2)

    pair_comp = SatelliteCompatibilityService.check_pair_compatibility(rep1, rep2, mode="bi_temporal")
    assert pair_comp.temporal_status == "invalid"
    assert any("inverted temporal" in w.lower() for w in pair_comp.warnings)


def test_case_17_excessive_nodata(test_dir: Path):
    """Case 17: Image with >95% nodata / unreadable pixels."""
    arr = np.full((128, 128), np.nan, dtype=np.float32)
    # Set just 10 valid pixels
    arr[0, :10] = 0.5
    p = test_dir / "corrupt_nodata.tif"
    Image.fromarray(arr, mode="F").save(p)

    rep = SatelliteImageInspector.inspect(p)
    assert rep.nodata_pct > 95.0

    comp = SatelliteCompatibilityService.check_single_compatibility(rep)
    assert comp.status == "invalid"
    assert any("nodata" in r.lower() for r in comp.reasons)


def test_case_18_sar_already_in_db(test_dir: Path):
    """Case 18: SAR backscatter already in decibels (dB) -> preserves dB, NEVER takes log10 on negative values."""
    # Values typical of Sentinel-1 calibrated sigma0 in dB: [-25.0, -5.0]
    arr = np.random.uniform(-25.0, -5.0, (128, 128)).astype(np.float32)
    p = test_dir / "s1a_iw_grdh_vv_db.tif"
    Image.fromarray(arr, mode="F").save(p)

    rep = SatelliteImageInspector.inspect(p)
    assert rep.modality_hint == "sar"
    assert rep.min_val < -1.0

    sar_db, tele = SatelliteInputAdapter.adapt_sar_to_numpy(rep)
    assert tele.normalization == "sar_db_preservation"
    assert np.allclose(sar_db, arr)
    assert any("already in calibrated decibel" in n.lower() for n in tele.notes)


def test_case_19_sar_linear_power(test_dir: Path):
    """Case 19: SAR backscatter in linear power -> converted to dB via 10*log10."""
    # Positive linear power values: [0.001, 2.0]
    arr = np.random.uniform(0.001, 2.0, (128, 128)).astype(np.float32)
    p = test_dir / "sentinel1_linear_intensity_vv.tif"
    Image.fromarray(arr, mode="F").save(p)

    rep = SatelliteImageInspector.inspect(p)
    assert rep.min_val >= 0.0

    sar_db, tele = SatelliteInputAdapter.adapt_sar_to_numpy(rep)
    assert tele.normalization == "linear_power_to_db_10log10"
    # Should be negative in dB for intensity < 1
    expected = 10.0 * np.log10(arr)
    assert np.allclose(sar_db, expected, atol=1e-4)


def test_case_20_missing_vv_for_optical_sar(test_dir: Path):
    """Case 20: Missing SAR for optical_sar mode -> flagged as unsupported."""
    p1 = create_png_rgb(test_dir / "opt1.png")
    p2 = create_png_rgb(test_dir / "opt2.png")

    rep1 = SatelliteImageInspector.inspect(p1)
    rep2 = SatelliteImageInspector.inspect(p2)

    pair_comp = SatelliteCompatibilityService.check_pair_compatibility(rep1, rep2, mode="optical_sar")
    assert pair_comp.status == "unsupported"
    assert any("missing sar modality" in r.lower() for r in pair_comp.reasons)


def test_case_21_missing_vh_handled_gracefully(test_dir: Path):
    """Case 21: SAR single-pol VV only (missing VH) -> handled safely."""
    opt_p = create_png_rgb(test_dir / "opt_scene.png")
    sar_p = test_dir / "sentinel1_vv.png"
    Image.fromarray(np.random.randint(0, 255, (256, 256), dtype=np.uint8), mode="L").save(sar_p)

    rep_opt = SatelliteImageInspector.inspect(opt_p)
    rep_sar = SatelliteImageInspector.inspect(sar_p)

    pair_comp = SatelliteCompatibilityService.check_pair_compatibility(rep_opt, rep_sar, mode="optical_sar")
    assert pair_comp.status in ("adaptable", "compatible")
    assert "optical_sar_analyzer" in pair_comp.specialist_candidates


def test_case_22_unsupported_sensor_modality_guardrail(test_dir: Path):
    """Case 22: SAR imagery passed to optical change detection OR query asks for unsupported domain (flood on LEVIR)."""
    # 22a. SAR imagery in bi_temporal mode
    opt_p = create_png_rgb(test_dir / "opt_t1.png")
    sar_p = test_dir / "sentinel1_radar_t2.png"
    Image.fromarray(np.random.randint(0, 255, (256, 256), dtype=np.uint8), mode="L").save(sar_p)

    rep_opt = SatelliteImageInspector.inspect(opt_p)
    rep_sar = SatelliteImageInspector.inspect(sar_p)

    pair_comp = SatelliteCompatibilityService.check_pair_compatibility(rep_opt, rep_sar, mode="bi_temporal")
    assert pair_comp.status in ("unsupported", "unsupported_for_reliable_inference")
    assert any("sar" in r.lower() for r in pair_comp.reasons)
    assert any("sar change detection is currently unsupported" in l.lower() for l in pair_comp.limitations)

    # 22b. Query asking for flood change on optical pair -> domain guardrail records limitation
    opt_p2 = create_png_rgb(test_dir / "opt_t2.png")
    rep_opt2 = SatelliteImageInspector.inspect(opt_p2)
    pair_flood = SatelliteCompatibilityService.check_pair_compatibility(
        rep_opt, rep_opt2, mode="bi_temporal", query="Detect flood inundation and water boundary change"
    )
    assert any("flood/water inundation change" in l.lower() for l in pair_flood.limitations)
    assert any("levir-cd" in l.lower() for l in pair_flood.limitations)
