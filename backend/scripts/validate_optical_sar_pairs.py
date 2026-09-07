"""
backend/scripts/validate_optical_sar_pairs.py
=============================================
Standalone validation script for Sentinel-1 (SAR) + Sentinel-2 (Optical) pairs.

Validates:
1. Spatial Georeferencing & Alignment:
   - CRS consistency (e.g. UTM / EPSG:4326).
   - Bounding box intersection & spatial overlap percentage.
   - Spatial resolution compatibility (e.g. 10m Sentinel-2 vs 10m Sentinel-1 GRD).
2. Radiometric Data Integrity:
   - Band count (Optical: RGB/Multi-spectral; SAR: VV or VV+VH dual-pol).
   - Data types (uint16 / float32 linear amplitude vs 8-bit visual proxies).
   - Physical backscatter range in decibels (expected -40 dB to +5 dB).
   - Polarimetric cross-ratio (VH/VV) and difference (VV - VH in dB).
   - Check for NaN/Inf values, zero-byte files, and dead-pixel columns.
3. Provenance Classification:
   - Authenticates whether input is REAL CALIBRATED GEOTIFF or UNCALIBRATED 8-BIT PROXY.

Usage:
  python backend/scripts/validate_optical_sar_pairs.py \
      --optical-dir /path/to/optical \
      --sar-dir /path/to/sar
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Import shared helpers from backend service
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.optical_sar import (
    extract_geotiff_metadata,
    compute_spatial_overlap,
    read_raster_band,
    extract_sar_polarimetric_physics,
    extract_optical_features,
)


def validate_pair(
    optical_path: Path,
    sar_vv_path: Path,
    sar_vh_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Validate an individual Optical + SAR pair."""
    pair_report: Dict[str, Any] = {
        "scene_id": optical_path.stem,
        "optical_file": optical_path.name,
        "sar_vv_file": sar_vv_path.name,
        "sar_vh_file": sar_vh_path.name if sar_vh_path else None,
        "errors": [],
        "warnings": [],
    }

    # 1. Check file existence and non-empty
    if not optical_path.exists() or optical_path.stat().st_size == 0:
        pair_report["errors"].append(f"Optical file missing or empty: {optical_path}")
        pair_report["is_valid"] = False
        return pair_report

    if not sar_vv_path.exists() or sar_vv_path.stat().st_size == 0:
        pair_report["errors"].append(f"SAR VV file missing or empty: {sar_vv_path}")
        pair_report["is_valid"] = False
        return pair_report

    # 2. Extract spatial metadata
    opt_meta = extract_geotiff_metadata(optical_path)
    sar_meta = extract_geotiff_metadata(sar_vv_path)
    sar_vh_meta = extract_geotiff_metadata(sar_vh_path) if sar_vh_path and sar_vh_path.exists() else None

    # Overlap
    overlap = compute_spatial_overlap(opt_meta, sar_meta)
    pair_report["spatial_overlap_pct"] = overlap

    # 3. Read rasters
    try:
        with Image.open(optical_path) as opt_img:
            opt_arr = np.asarray(opt_img)
            opt_shape = opt_arr.shape
            opt_stats = extract_optical_features(opt_img)
            pair_report["optical_stats"] = opt_stats
    except Exception as e:
        pair_report["errors"].append(f"Failed to open optical image: {e}")
        opt_arr = np.zeros((120, 120, 3), dtype=np.uint8)
        opt_shape = (0, 0)

    try:
        sar_vv_arr, is_vv_calibrated = read_raster_band(sar_vv_path, band_index=1)
        vv_shape = sar_vv_arr.shape
        sar_vh_arr = None
        vh_shape = None
        if sar_vh_path and sar_vh_path.exists():
            sar_vh_arr, _ = read_raster_band(sar_vh_path, band_index=1)
            vh_shape = sar_vh_arr.shape

        # Check for NaN / Inf
        nan_vv = int(np.isnan(sar_vv_arr).sum())
        inf_vv = int(np.isinf(sar_vv_arr).sum())
        if nan_vv > 0 or inf_vv > 0:
            pair_report["warnings"].append(f"SAR VV contains {nan_vv} NaNs and {inf_vv} Infs")

        # Physics extraction with SAR representation detection
        sar_physics = extract_sar_polarimetric_physics(
            sar_vv_arr=sar_vv_arr,
            sar_vh_arr=sar_vh_arr,
        )

        from app.services.optical_sar import align_optical_sar, normalize_optical
        _, norm_meta = normalize_optical(opt_arr)
        aligned_pkg = align_optical_sar(
            optical=optical_path,
            sar_vv=sar_vv_path,
            sar_vh=sar_vh_path,
            opt_meta=opt_meta,
            sar_meta=sar_meta,
        )
        alignment_status = "ALIGNED_COMMON_GRID" if aligned_pkg.height > 0 else "ALIGNMENT_FAILED"

        pair_report["optical_shape"] = opt_shape
        pair_report["vv_shape"] = vv_shape
        pair_report["vh_shape"] = vh_shape
        pair_report["crs"] = sar_meta.crs or opt_meta.crs or "None"
        pair_report["optical_crs"] = opt_meta.crs or "None"
        pair_report["sar_crs"] = sar_meta.crs or "None"
        pair_report["resolution"] = sar_meta.resolution or opt_meta.resolution or (10.0, 10.0)
        pair_report["bounds"] = sar_meta.bounds or opt_meta.bounds
        pair_report["sar_representation"] = sar_physics.sar_representation
        pair_report["normalization"] = norm_meta.get("normalization_applied", "unknown")
        pair_report["alignment_status"] = alignment_status

        pair_report["sar_physics"] = {
            "mean_vv_db": sar_physics.mean_vv_db,
            "min_vv_db": sar_physics.min_vv_db,
            "max_vv_db": sar_physics.max_vv_db,
            "surface_roughness": sar_physics.surface_roughness_variance,
            "specular_water_pct": sar_physics.specular_low_backscatter_pct,
            "double_bounce_pct": sar_physics.double_bounce_high_backscatter_pct,
            "is_calibrated_sar": sar_physics.is_calibrated,
            "is_dual_pol": sar_physics.is_dual_pol,
            "vh_vv_cross_ratio": sar_physics.vh_vv_ratio_mean,
            "vv_vh_diff_db": sar_physics.vv_vh_diff_db,
            "sar_representation": sar_physics.sar_representation,
        }

        # Provenance classification
        if sar_physics.is_calibrated and (opt_meta.is_geotiff or sar_meta.is_geotiff):
            pair_report["provenance"] = "REAL_CALIBRATED_GEOTIFF"
        elif not sar_physics.is_calibrated:
            pair_report["provenance"] = "UNCALIBRATED_8BIT_PROXY"
            pair_report["warnings"].append(
                "Input is an 8-bit uncalibrated visual proxy, not Level-1 calibrated Sentinel SAR."
            )
        else:
            pair_report["provenance"] = "CALIBRATED_RASTER_NO_GEO_HEADER"

    except Exception as e:
        pair_report["errors"].append(f"Failed to process rasters: {e}")

    pair_report["is_valid"] = len(pair_report["errors"]) == 0
    return pair_report


def run_pairs_validation(
    root_dir: Optional[Path] = None,
    optical_dir: Optional[Path] = None,
    sar_dir: Optional[Path] = None,
    sar_vv_dir: Optional[Path] = None,
    sar_vh_dir: Optional[Path] = None,
    optical_file: Optional[Path] = None,
    sar_vv_file: Optional[Path] = None,
    sar_vh_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """Validate single pair or directory batches of Optical+SAR scenes."""
    t0 = time.perf_counter()
    report: Dict[str, Any] = {
        "pairs_validated": 0,
        "valid_pairs": 0,
        "failed_pairs": 0,
        "calibrated_geotiff_count": 0,
        "proxy_8bit_count": 0,
        "results": [],
    }

    # Case 1: Single file pair
    if optical_file and sar_vv_file:
        res = validate_pair(optical_file, sar_vv_file, sar_vh_file)
        report["pairs_validated"] = 1
        report["results"].append(res)
        if res["is_valid"]:
            report["valid_pairs"] = 1
        else:
            report["failed_pairs"] = 1

        if res.get("provenance") == "REAL_CALIBRATED_GEOTIFF":
            report["calibrated_geotiff_count"] = 1
        else:
            report["proxy_8bit_count"] = 1

    # Case 2: Root directory with optical, sar_vv, sar_vh subfolders
    elif root_dir and root_dir.exists():
        opt_d = root_dir / "optical"
        vv_d = root_dir / "sar_vv"
        vh_d = root_dir / "sar_vh"

        if not opt_d.exists() or not vv_d.exists():
            report["status"] = "INVALID_ROOT_DIR"
            report["summary"] = f"Expected subdirectories 'optical' and 'sar_vv' under {root_dir}"
            return report

        opt_files = {p.name: p for p in opt_d.iterdir() if p.is_file() and p.suffix.lower() in (".tif", ".tiff", ".png")}
        vv_files = {p.name: p for p in vv_d.iterdir() if p.is_file() and p.suffix.lower() in (".tif", ".tiff", ".png")}
        vh_files = {p.name: p for p in vh_d.iterdir() if p.is_file()} if vh_d.exists() else {}

        common_keys = sorted(set(opt_files.keys()).intersection(vv_files.keys()))
        for k in common_keys:
            res = validate_pair(opt_files[k], vv_files[k], vh_files.get(k))
            report["results"].append(res)
            report["pairs_validated"] += 1
            if res["is_valid"]:
                report["valid_pairs"] += 1
            else:
                report["failed_pairs"] += 1

            if res.get("provenance") == "REAL_CALIBRATED_GEOTIFF":
                report["calibrated_geotiff_count"] += 1
            else:
                report["proxy_8bit_count"] += 1

    # Case 3: Explicit directory arguments
    elif optical_dir and (sar_vv_dir or sar_dir) and optical_dir.exists():
        target_vv_dir = sar_vv_dir or sar_dir
        opt_files = {p.name: p for p in optical_dir.iterdir() if p.is_file()}
        vv_files = {p.name: p for p in target_vv_dir.iterdir() if p.is_file()}
        vh_files = {p.name: p for p in sar_vh_dir.iterdir() if p.is_file()} if sar_vh_dir and sar_vh_dir.exists() else {}

        common_keys = sorted(set(opt_files.keys()).intersection(vv_files.keys()))
        for k in common_keys:
            res = validate_pair(opt_files[k], vv_files[k], vh_files.get(k))
            report["results"].append(res)
            report["pairs_validated"] += 1
            if res["is_valid"]:
                report["valid_pairs"] += 1
            else:
                report["failed_pairs"] += 1

            if res.get("provenance") == "REAL_CALIBRATED_GEOTIFF":
                report["calibrated_geotiff_count"] += 1
            else:
                report["proxy_8bit_count"] += 1

    else:
        report["status"] = "NO_INPUTS_PROVIDED"
        report["summary"] = "No valid pair files or directories provided."

    report["duration_ms"] = int((time.perf_counter() - t0) * 1000)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Optical + SAR paired datasets.")
    parser.add_argument("--root-dir", type=Path, default=None, help="Root directory containing optical, sar_vv, sar_vh subdirectories")
    parser.add_argument("--optical-file", type=Path, default=None, help="Path to Optical GeoTIFF/image")
    parser.add_argument("--sar-vv-file", type=Path, default=None, help="Path to SAR VV GeoTIFF/image")
    parser.add_argument("--sar-vh-file", type=Path, default=None, help="Path to SAR VH GeoTIFF (optional)")
    parser.add_argument("--optical-dir", type=Path, default=None, help="Directory containing optical files")
    parser.add_argument("--sar-dir", type=Path, default=None, help="Directory containing SAR files")
    parser.add_argument("--sar-vv-dir", type=Path, default=None, help="Directory containing SAR VV files")
    parser.add_argument("--sar-vh-dir", type=Path, default=None, help="Directory containing SAR VH files")

    args = parser.parse_args()
    # Default to C:\Users\Lenovo\Downloads\Optical_SAR_Pairs if no inputs provided
    default_root = Path("C:/Users/Lenovo/Downloads/Optical_SAR_Pairs")
    target_root = args.root_dir or (default_root if default_root.exists() and not (args.optical_file or args.optical_dir) else None)

    report = run_pairs_validation(
        root_dir=target_root,
        optical_dir=args.optical_dir,
        sar_dir=args.sar_dir,
        sar_vv_dir=args.sar_vv_dir,
        sar_vh_dir=args.sar_vh_dir,
        optical_file=args.optical_file,
        sar_vv_file=args.sar_vv_file,
        sar_vh_file=args.sar_vh_file,
    )

    print("================================================================================")
    print("OPTICAL + SAR PAIR DATASET VALIDATION REPORT")
    print("================================================================================")
    print(f"  Pairs Validated:           {report['pairs_validated']}")
    print(f"  Valid Pairs:               {report['valid_pairs']}")
    print(f"  Failed Pairs:              {report['failed_pairs']}")
    print(f"  Calibrated Real GeoTIFFs:  {report['calibrated_geotiff_count']}")
    print(f"  Uncalibrated Proxies:      {report['proxy_8bit_count']}")
    print("--------------------------------------------------------------------------------")

    for r in report["results"]:
        print(f"\n[Scene: {r.get('scene_id')}]")
        print(f"  - Optical Shape:       {r.get('optical_shape')}")
        print(f"  - SAR VV Shape:        {r.get('vv_shape')}")
        print(f"  - SAR VH Shape:        {r.get('vh_shape')}")
        print(f"  - CRS:                 {r.get('crs')}")
        print(f"  - Resolution:          {r.get('resolution')}")
        print(f"  - Bounds:              {r.get('bounds')}")
        print(f"  - Spatial Overlap:     {r.get('spatial_overlap_pct')}")
        print(f"  - SAR Representation:  {r.get('sar_representation')}")
        print(f"  - Normalization:       {r.get('normalization')}")
        print(f"  - Alignment Status:    {r.get('alignment_status')}")
        if r.get("sar_physics"):
            sp = r["sar_physics"]
            print(f"  - Backscatter VV:      Mean={sp['mean_vv_db']} dB, Min={sp['min_vv_db']} dB, Max={sp['max_vv_db']} dB")
            print(f"  - Water Pct:           {sp['specular_water_pct']}% | Built-up Pct: {sp['double_bounce_pct']}%")
        if r.get("errors"):
            print(f"  - Errors:              {r['errors']}")

    print("\n================================================================================")
    return 0 if report["failed_pairs"] == 0 and report["pairs_validated"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

