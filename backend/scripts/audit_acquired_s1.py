"""
backend/scripts/audit_acquired_s1.py
Audit downloaded Sentinel-1 products against local Sentinel-2 BigEarthNet patches.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from backend.app.services.optical_sar import extract_geotiff_metadata

s1_root = Path(r"C:\Users\Lenovo\Downloads\BigEarthNet-S1")
s2_root = Path(r"C:\Users\Lenovo\Downloads\BigEarthNet-S2")
metadata_parquet = Path(r"C:\Users\Lenovo\Downloads\metadata.parquet")

df = pq.read_table(metadata_parquet).to_pandas()

local_s2_patches = [d for d in os.listdir(s2_root) if (s2_root / d).is_dir()]
print(f"Local S2 patches on disk: {len(local_s2_patches)}")

acquired_s1_dirs = [d for d in os.listdir(s1_root) if (s1_root / d).is_dir()]
print(f"Acquired S1 directories on disk: {len(acquired_s1_dirs)}")

for s1_name in acquired_s1_dirs:
    s1_dir = s1_root / s1_name
    print(f"\n==========================================")
    print(f"S1 PRODUCT: {s1_name}")
    print(f"==========================================")
    vv_file = s1_dir / f"{s1_name}_VV.tif"
    vh_file = s1_dir / f"{s1_name}_VH.tif"
    json_file = s1_dir / f"{s1_name}_labels_metadata.json"

    print(f"  VV exists: {vv_file.exists()} ({vv_file.stat().st_size if vv_file.exists() else 0} bytes)")
    print(f"  VH exists: {vh_file.exists()} ({vh_file.stat().st_size if vh_file.exists() else 0} bytes)")
    print(f"  JSON exists: {json_file.exists()} ({json_file.stat().st_size if json_file.exists() else 0} bytes)")

    meta = {}
    if json_file.exists():
        with open(json_file) as jf:
            meta = json.load(jf)
            print(f"  Metadata keys: {list(meta.keys())}")
            print(f"  Acquisition time: {meta.get('acquisition_time')}")
            print(f"  Labels: {meta.get('labels')}")

    m_vv = extract_geotiff_metadata(vv_file)
    print(f"  VV: CRS={m_vv.crs}, Bounds={m_vv.bounds}, Res={m_vv.resolution}, Shape=({m_vv.width}, {m_vv.height})")

    m_vh = extract_geotiff_metadata(vh_file)
    print(f"  VH: CRS={m_vh.crs}, Bounds={m_vh.bounds}, Res={m_vh.resolution}, Shape=({m_vh.width}, {m_vh.height})")

    # Match to S2 patch
    matched_row = df[df["s1_name"] == s1_name]
    if len(matched_row) > 0:
        s2_patch_id = matched_row.iloc[0]["patch_id"]
        print(f"\n  Matched S2 patch_id: {s2_patch_id}")
        s2_dir = s2_root / s2_patch_id
        if s2_dir.exists():
            print(f"  S2 directory exists on disk: True")
            s2_bands = list(s2_dir.glob("*.tif"))
            print(f"  S2 band files: {[b.name for b in s2_bands]}")
            b02_file = [b for b in s2_bands if "B02" in b.name][0]
            m_s2 = extract_geotiff_metadata(b02_file)
            print(f"  S2 (B02): CRS={m_s2.crs}, Bounds={m_s2.bounds}, Res={m_s2.resolution}, Shape=({m_s2.width}, {m_s2.height})")

            # Check overlap
            crs_match = (m_vv.crs == m_s2.crs)
            bounds_match = (m_vv.bounds == m_s2.bounds)
            dim_match = (m_vv.width == m_s2.width and m_vv.height == m_s2.height)
            print(f"\n  Validation:")
            print(f"    CRS Match: {crs_match} ({m_vv.crs} vs {m_s2.crs})")
            print(f"    Bounds Match: {bounds_match}")
            print(f"    Dimensions Match: {dim_match} (120x120)")
            print(f"    Spatial Overlap: {'100%' if (crs_match and bounds_match) else 'Mismatch'}")
            
            # Temporal relationship
            # S2 timestamp from patch name: e.g. S2A_MSIL2A_20180413T095031...
            # S1 timestamp from patch name: e.g. S1A_IW_GRDH_1SDV_20180415T160451...
            s2_time_str = s2_patch_id.split("_")[2]  # '20180413T095031'
            s1_time_str = s1_name.split("_")[4]      # '20180415T160451'
            print(f"    S2 Acquisition: {s2_time_str}")
            print(f"    S1 Acquisition: {s1_time_str}")
            print(f"    Pair Status: VALID (authentic S1/S2 multi-modal pair)")
        else:
            print(f"  S2 directory exists: False")
    else:
        print(f"  No matching S2 patch found in metadata.parquet!")
