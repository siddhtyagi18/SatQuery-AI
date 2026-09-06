"""
backend/scripts/download_stage1_data.py
======================================
Streamed acquisition of Stage 1 Real Data:
1. Exactly 15 real Sentinel-2 BigEarthNet patches (B02, B03, B04, B08 GeoTIFFs).
2. Exactly 5 real Sentinel-1 SAR dual-pol patches (VV + VH GeoTIFFs).
3. Zero multi-GB waste: streams and extracts via HTTP in real time (< 3 MB transfer).
"""
from __future__ import annotations

import io
import shutil
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image
import pyarrow.parquet as pq

ZENODO_S2_URL = "https://zenodo.org/records/12687186/files/BigEarthNet-S2-v1.0.tar.gz"
ZENODO_S1_URL = "https://zenodo.org/records/12687186/files/BigEarthNet-S1-v1.0.tar.gz"

METADATA_PARQUET = Path("C:/Users/Lenovo/Downloads/metadata.parquet")
S2_DEST_ROOT = Path("C:/Users/Lenovo/Downloads/BigEarthNet-S2")
OPTICAL_SAR_ROOT = Path("C:/Users/Lenovo/Downloads/Optical_SAR_Pairs")


def build_name_mapping() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build lookup tables from s2v1_name -> patch_id and s1_name -> patch_id."""
    if not METADATA_PARQUET.exists():
        raise FileNotFoundError(f"metadata.parquet not found at {METADATA_PARQUET}")

    df = pq.read_table(METADATA_PARQUET, columns=["patch_id", "s2v1_name", "s1_name"]).to_pandas()
    s2_map = dict(zip(df["s2v1_name"], df["patch_id"]))
    s1_map = dict(zip(df["s1_name"], df["patch_id"]))
    return s2_map, s1_map


def download_15_s2_patches(s2_map: Dict[str, str], target_count: int = 15) -> List[str]:
    """Stream and extract exactly target_count Sentinel-2 patches."""
    print(f"\n[1/2] Streaming {target_count} Sentinel-2 patch folders from official Zenodo archive...")
    t0 = time.perf_counter()
    S2_DEST_ROOT.mkdir(parents=True, exist_ok=True)

    downloaded_patches: List[str] = []
    current_patch_v1: str = ""
    current_patch_dir: Optional[Path] = None
    target_bands = {"B02", "B03", "B04", "B08"}

    req = urllib.request.Request(ZENODO_S2_URL, headers={"User-Agent": "SatQuery-AI/1.0"})
    response = urllib.request.urlopen(req)
    tar_stream = tarfile.open(fileobj=response, mode="r|gz")

    for member in tar_stream:
        if len(downloaded_patches) >= target_count and not member.name.startswith(f"BigEarthNet-v1.0/{current_patch_v1}"):
            break

        parts = member.name.split("/")
        if len(parts) < 2 or not parts[1]:
            continue

        patch_v1 = parts[1]

        # New patch encountered
        if patch_v1 != current_patch_v1:
            if len(downloaded_patches) >= target_count:
                break

            current_patch_v1 = patch_v1
            patch_id = s2_map.get(patch_v1, patch_v1)
            current_patch_dir = S2_DEST_ROOT / patch_id
            current_patch_dir.mkdir(parents=True, exist_ok=True)
            downloaded_patches.append(patch_id)
            print(f"  -> Extracting S2 Patch {len(downloaded_patches)}/{target_count}: {patch_id}")

        # Extract band file if matches target 10m bands
        if member.isfile() and current_patch_dir:
            filename = Path(member.name).name
            for b in target_bands:
                if f"_{b}.tif" in filename or filename.endswith(f"{b}.tif"):
                    dest_file = current_patch_dir / filename
                    extracted = tar_stream.extractfile(member)
                    if extracted:
                        dest_file.write_bytes(extracted.read())
                    break

    tar_stream.close()
    response.close()
    elapsed = round(time.perf_counter() - t0, 2)
    print(f"  [OK] Successfully extracted {len(downloaded_patches)} Sentinel-2 patches in {elapsed}s.")
    return downloaded_patches


def download_5_matching_s1_patches(s2_patch_ids: List[str], target_count: int = 5) -> List[str]:
    """Stream and extract Sentinel-1 dual-pol patches corresponding to the downloaded S2 patches."""
    print(f"\n[2/2] Streaming {target_count} Sentinel-1 SAR dual-pol patches...")
    t0 = time.perf_counter()

    opt_dir = OPTICAL_SAR_ROOT / "optical"
    sar_vv_dir = OPTICAL_SAR_ROOT / "sar_vv"
    sar_vh_dir = OPTICAL_SAR_ROOT / "sar_vh"
    opt_dir.mkdir(parents=True, exist_ok=True)
    sar_vv_dir.mkdir(parents=True, exist_ok=True)
    sar_vh_dir.mkdir(parents=True, exist_ok=True)

    # Find matching s1_names for the first 5 S2 patches
    df = pq.read_table(METADATA_PARQUET, columns=["patch_id", "s1_name"]).to_pandas()
    subset_df = df[df["patch_id"].isin(s2_patch_ids[:target_count])]
    s1_targets = dict(zip(subset_df["s1_name"], subset_df["patch_id"]))
    print(f"  Targeting S1 scenes for S2 patches: {list(s1_targets.keys())}")

    extracted_s1: List[str] = []
    current_s1_name: str = ""
    current_target_id: Optional[str] = None

    req = urllib.request.Request(ZENODO_S1_URL, headers={"User-Agent": "SatQuery-AI/1.0"})
    response = urllib.request.urlopen(req)
    tar_stream = tarfile.open(fileobj=response, mode="r|gz")

    for member in tar_stream:
        if len(extracted_s1) >= target_count:
            break

        parts = member.name.split("/")
        if len(parts) < 2 or not parts[1]:
            continue

        s1_folder = parts[1]
        if s1_folder in s1_targets:
            if s1_folder != current_s1_name:
                current_s1_name = s1_folder
                current_target_id = s1_targets[s1_folder]
                extracted_s1.append(s1_folder)
                print(f"  -> Extracting SAR Pair {len(extracted_s1)}/{target_count}: {s1_folder}")

            if member.isfile():
                filename = Path(member.name).name
                extracted = tar_stream.extractfile(member)
                if extracted:
                    file_bytes = extracted.read()
                    if "_VV.tif" in filename or filename.endswith("VV.tif"):
                        (sar_vv_dir / f"{current_target_id}_sar_vv.tif").write_bytes(file_bytes)
                    elif "_VH.tif" in filename or filename.endswith("VH.tif"):
                        (sar_vh_dir / f"{current_target_id}_sar_vh.tif").write_bytes(file_bytes)

    tar_stream.close()
    response.close()

    # Assemble and copy corresponding Optical RGB images into optical/
    for pid in s2_patch_ids[:len(extracted_s1)]:
        s2_patch_dir = S2_DEST_ROOT / pid
        if s2_patch_dir.exists():
            from app.services.datasets.bigearthnet_multimodal import assemble_sentinel2_rgb
            try:
                rgb_img, _ = assemble_sentinel2_rgb(s2_patch_dir)
                rgb_img.save(opt_dir / f"{pid}_opt.tif", format="TIFF")
            except Exception as e:
                print(f"  Warning: failed to assemble optical composite for {pid}: {e}")

    elapsed = round(time.perf_counter() - t0, 2)
    print(f"  [OK] Successfully extracted {len(extracted_s1)} Sentinel-1 dual-pol patches in {elapsed}s.")
    return extracted_s1


def main() -> int:
    s2_map, s1_map = build_name_mapping()
    s2_patches = download_15_s2_patches(s2_map, target_count=15)
    s1_patches = download_5_matching_s1_patches(s2_patches, target_count=5)
    print("\nStage 1 Acquisition Complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
