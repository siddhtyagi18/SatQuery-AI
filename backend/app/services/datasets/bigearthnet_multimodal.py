"""
backend/app/services/datasets/bigearthnet_multimodal.py
======================================================
Multimodal BigEarthNet PyTorch Dataset Loader & Indexer.

This module joins text-based VQA annotations from `BigEarthNet.txt.parquet`
with actual Sentinel-2 (and optionally Sentinel-1) GeoTIFF image patch folders.

Dataset Structure Expected on Disk:
  <images_root>/
      <patch_id>/  (e.g., S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57/)
          B02.tif  (Blue, 10m, 120x120 uint16)
          B03.tif  (Green, 10m, 120x120 uint16)
          B04.tif  (Red, 10m, 120x120 uint16)
          B08.tif  (NIR, 10m, 120x120 uint16, optional)

Key Characteristics:
- Lazy raster loading: Opens band GeoTIFFs on __getitem__, not __init__.
- Deterministic join: Maps parquet rows to patch directories by `patch_id`.
- Natural RGB assembly: Stacks B04 (R), B03 (G), B02 (B) with 2-98% percentile contrast stretch.
- HuggingFace/SmolVLM compatibility: Yields dict containing PIL RGB Image, query text, target answer, and metadata.
- Graceful degradation: Validates patch presence without crashing if image directory is partially populated.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class Dataset:  # type: ignore
        pass

from ...logging_setup import logger

# Required 10m bands for True Color RGB assembly
RGB_BANDS = ("B04", "B03", "B02")  # Red, Green, Blue
ALL_10M_BANDS = ("B02", "B03", "B04", "B08")  # Blue, Green, Red, NIR


@dataclass
class BigEarthNetSample:
    """Structured multimodal sample yielded by the dataset loader."""
    sample_id: int
    patch_id: str
    s1_name: Optional[str]
    question: str
    answer: str
    task_type: str
    category: str
    split: str
    image: Optional[Image.Image] = None
    band_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "patch_id": self.patch_id,
            "s1_name": self.s1_name,
            "question": self.question,
            "answer": self.answer,
            "task_type": self.task_type,
            "category": self.category,
            "split": self.split,
            "band_stats": self.band_stats,
        }


def _read_band_tiff(band_path: Path) -> np.ndarray:
    """Read a single Sentinel-2 band GeoTIFF/TIFF into a 2D float32 array."""
    try:
        import rasterio
        with rasterio.open(band_path) as src:
            arr = src.read(1).astype(np.float32)
            # Mask nodata if present
            if src.nodata is not None:
                arr = np.where(arr == src.nodata, np.nan, arr)
            return arr
    except Exception:
        # Fallback to Pillow
        with Image.open(band_path) as img:
            return np.asarray(img, dtype=np.float32)


def _percentile_stretch_uint8(ch: np.ndarray, pmin: float = 2.0, pmax: float = 98.0) -> np.ndarray:
    """Apply robust 2-98% percentile contrast stretch and convert to uint8 [0, 255]."""
    valid = np.isfinite(ch)
    if not valid.any():
        return np.zeros_like(ch, dtype=np.uint8)
    low = np.percentile(ch[valid], pmin)
    high = np.percentile(ch[valid], pmax)
    if high <= low:
        high = low + 1e-6
    clipped = np.clip((ch - low) / (high - low), 0.0, 1.0)
    return (clipped * 255.0 + 0.5).astype(np.uint8)


def assemble_sentinel2_rgb(patch_dir: Path) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Assemble a calibrated True-Color RGB PIL Image from Sentinel-2 band GeoTIFFs.
    Looks for B04.tif (Red), B03.tif (Green), and B02.tif (Blue).
    """
    stats: Dict[str, Any] = {}
    channels = []

    for b in RGB_BANDS:
        # Check standard naming: B04.tif, B4.tif, *_B04.tif
        band_file = None
        candidates = [
            patch_dir / f"{b}.tif",
            patch_dir / f"{b}.tiff",
            patch_dir / f"{b}.png",
            patch_dir / f"{b.lower()}.tif",
        ]
        for c in candidates:
            if c.exists():
                band_file = c
                break

        if not band_file:
            # Glob for band identifier in filename
            matches = list(patch_dir.glob(f"*{b}*.*"))
            if matches:
                band_file = matches[0]

        if not band_file:
            raise FileNotFoundError(f"Missing required Sentinel-2 band '{b}' in patch directory: {patch_dir}")

        arr = _read_band_tiff(band_file)
        stats[f"{b}_mean"] = round(float(np.nanmean(arr)), 2)
        stats[f"{b}_max"] = round(float(np.nanmax(arr)), 2)
        channels.append(arr)

    # Stack into RGB array (B04=Red, B03=Green, B02=Blue)
    r_uint8 = _percentile_stretch_uint8(channels[0])
    g_uint8 = _percentile_stretch_uint8(channels[1])
    b_uint8 = _percentile_stretch_uint8(channels[2])

    rgb_arr = np.stack([r_uint8, g_uint8, b_uint8], axis=-1)
    pil_img = Image.fromarray(rgb_arr, mode="RGB")
    stats["image_size"] = list(pil_img.size)
    return pil_img, stats


class BigEarthNetMultimodalDataset(Dataset):
    """
    PyTorch Dataset joining BigEarthNet.txt.parquet QA rows with Sentinel-2 image patches.

    Parameters:
    - parquet_path: Path to BigEarthNet.txt.parquet file.
    - images_root: Path to directory containing Sentinel-2 patch subdirectories.
    - split: Filter by split ('train', 'val', 'test', or None for all).
    - max_samples: Optional cap on dataset size for fast validation / debugging.
    - transform: Optional image transform callable.
    """

    def __init__(
        self,
        parquet_path: Union[str, Path],
        images_root: Optional[Union[str, Path]] = None,
        split: Optional[str] = None,
        max_samples: Optional[int] = None,
        transform: Optional[Callable[[Image.Image], Any]] = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to instantiate BigEarthNetMultimodalDataset.")

        if parquet_path is None:
            default_dl = Path.home() / "Downloads" / "BigEarthNet.txt.parquet"
            parquet_path = os.getenv("BIGEARTHNET_TXT_PARQUET") or (str(default_dl) if default_dl.exists() else "BigEarthNet.txt.parquet")
        self.parquet_path = Path(parquet_path)
        self.images_root = Path(images_root) if images_root else None
        self.split = split
        self.transform = transform
        self.max_samples = max_samples

        self._records: List[Dict[str, Any]] = []
        self._available_patch_dirs: Dict[str, Path] = {}
        self._index_dataset()

    def _index_dataset(self) -> None:
        """Read parquet index and match with available image patch folders on disk."""
        if not self.parquet_path.exists():
            logger.warning(f"BigEarthNet parquet file not found at: {self.parquet_path}")
            return

        # Scan available image directories if images_root exists
        if self.images_root and self.images_root.exists():
            for d in self.images_root.iterdir():
                if d.is_dir():
                    self._available_patch_dirs[d.name] = d

        # Read parquet rows using PyArrow
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(
                self.parquet_path,
                columns=["ID", "patch_id", "s1_name", "input", "output", "type", "category", "split"],
            )

            # If images_root is provided and populated, filter records to available patches
            require_images = bool(self._available_patch_dirs)

            # Convert table to records list with optional split filtering
            df_len = len(table)
            for i in range(df_len):
                row_split = table["split"][i].as_py()
                if self.split and row_split != self.split:
                    continue

                patch_id = table["patch_id"][i].as_py()
                if require_images and patch_id not in self._available_patch_dirs:
                    continue

                rec = {
                    "sample_id": table["ID"][i].as_py(),
                    "patch_id": patch_id,
                    "s1_name": table["s1_name"][i].as_py(),
                    "question": table["input"][i].as_py(),
                    "answer": table["output"][i].as_py(),
                    "task_type": table["type"][i].as_py(),
                    "category": table["category"][i].as_py(),
                    "split": row_split,
                }
                self._records.append(rec)

                if self.max_samples and len(self._records) >= self.max_samples:
                    break

            logger.info(
                f"[BigEarthNetDataset] Indexed {len(self._records)} QA samples "
                f"(split={self.split}, images_available={len(self._available_patch_dirs)})."
            )
        except Exception as e:
            logger.exception(f"Failed to read BigEarthNet parquet table: {e}")

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= len(self._records):
            raise IndexError(f"Index {idx} out of range (dataset size: {len(self._records)})")

        rec = self._records[idx]
        patch_id = rec["patch_id"]

        pil_image = None
        band_stats: Dict[str, Any] = {}

        # If patch folder exists on disk, assemble RGB raster
        patch_dir = self._available_patch_dirs.get(patch_id)
        if patch_dir and patch_dir.exists():
            try:
                pil_image, band_stats = assemble_sentinel2_rgb(patch_dir)
                if self.transform and pil_image:
                    pil_image = self.transform(pil_image)
            except Exception as e:
                logger.debug(f"Failed to assemble image for {patch_id}: {e}")

        sample = BigEarthNetSample(
            sample_id=rec["sample_id"],
            patch_id=rec["patch_id"],
            s1_name=rec["s1_name"],
            question=rec["question"],
            answer=rec["answer"],
            task_type=rec["task_type"],
            category=rec["category"],
            split=rec["split"],
            image=pil_image,
            band_stats=band_stats,
        )

        return {
            "sample": sample,
            "image": sample.image,
            "question": sample.question,
            "answer": sample.answer,
            "patch_id": sample.patch_id,
            "task_type": sample.task_type,
            "category": sample.category,
        }
