from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from ..logging_setup import logger

try:
    import rasterio
    from rasterio.enums import Resampling
    RASTERIO_AVAILABLE = True
except Exception:
    RASTERIO_AVAILABLE = False
    logger.warning("rasterio not available — GeoTIFF preprocessing will use Pillow fallback only")

try:
    import pyproj
    PYPROJ_AVAILABLE = True
except Exception:
    PYPROJ_AVAILABLE = False
    logger.warning("pyproj not available — CRS transforms will be skipped")


RASTERIO_EXTS = {".tif", ".tiff", ".gtiff"}
PIL_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class ImageryPreprocessingError(Exception):
    """Raised when image preprocessing fails (invalid imagery, corrupt file, etc.)."""
    pass


class PreprocessingResult:
    """Container for the output of the preprocessing pipeline."""

    def __init__(
        self,
        rgb_image: Image.Image,
        source_path: Path,
        preprocessing_meta: Dict[str, Any],
    ):
        self.rgb_image = rgb_image
        self.source_path = source_path
        self.preprocessing_meta = preprocessing_meta


def _percent_stretch(arr: np.ndarray, pmin: float = 2.0, pmax: float = 98.0) -> np.ndarray:
    """Apply a robust percent-based contrast stretch to a single-channel float/int array."""
    if arr.size == 0:
        return arr
    low = np.percentile(arr, pmin)
    high = np.percentile(arr, pmax)
    if high <= low:
        high = low + 1e-6
    stretched = np.clip((arr.astype(np.float32) - low) / (high - low), 0.0, 1.0)
    return stretched


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert a float [0,1] array to uint8 [0,255]."""
    return (arr * 255.0 + 0.5).astype(np.uint8)


def _select_rgb_bands(arr: np.ndarray, band_count: int) -> np.ndarray:
    """Pick an RGB triplet from a multi-band raster.

    Heuristic: for >=3 bands, use bands 1,2,3 (often true RGB for optical).
    For single-band / panchromatic, replicate to 3 channels (pseudo-RGB).
    For exactly 2 bands (rare), replicate band 1 as B channel.
    """
    if band_count >= 3:
        return arr[:3]
    if band_count == 1:
        return np.stack([arr[0], arr[0], arr[0]], axis=0)
    if band_count == 2:
        return np.stack([arr[0], arr[1], arr[0]], axis=0)
    raise ImageryPreprocessingError(f"Unexpected band count for RGB assembly: {band_count}")


def _preprocess_rasterio(file_path: Path) -> Tuple[Image.Image, Dict[str, Any]]:
    """Preprocess a GeoTIFF/TIFF via rasterio into a PIL RGB Image for VLM input."""
    meta: Dict[str, Any] = {
        "backend": "rasterio",
        "crs": None,
        "width_px": 0,
        "height_px": 0,
        "band_count": 0,
        "stretch": "2-98 percentile",
        "nodata_handling": "masked",
    }
    try:
        with rasterio.open(file_path) as src:
            meta["width_px"] = src.width
            meta["height_px"] = src.height
            meta["band_count"] = src.count
            if src.crs:
                meta["crs"] = str(src.crs)

            if src.count == 0:
                raise ImageryPreprocessingError(f"Raster has 0 bands: {file_path.name}")

            max_dim = 2048
            scale_factor = 1.0
            w, h = src.width, src.height
            if max(w, h) > max_dim:
                scale_factor = max_dim / max(w, h)
                new_w = max(1, int(w * scale_factor))
                new_h = max(1, int(h * scale_factor))
                meta["resized_from"] = [w, h]
                meta["resized_to"] = [new_w, new_h]
                data = src.read(
                    out_shape=(src.count, new_h, new_w),
                    resampling=Resampling.bilinear,
                    masked=True,
                )
            else:
                data = src.read(masked=True)

            if src.nodata is not None:
                meta["nodata_value"] = src.nodata
    except ImageryPreprocessingError:
        raise
    except Exception as e:
        raise ImageryPreprocessingError(f"rasterio read failed for {file_path.name}: {e}") from e

    filled_data = np.ma.filled(data, fill_value=np.nan)
    band_count = filled_data.shape[0]
    meta["input_band_count"] = band_count

    rgb_bands = _select_rgb_bands(filled_data, band_count)
    meta["rgb_band_source"] = "natural" if band_count >= 3 else "pseudo"

    stretched_channels = []
    for i in range(3):
        ch = rgb_bands[i]
        mask = np.isfinite(ch)
        if not mask.any():
            stretched_channels.append(np.zeros_like(ch, dtype=np.float32))
            continue
        ch_clean = np.where(mask, ch, np.nanmedian(ch[mask]))
        stretched_channels.append(_percent_stretch(ch_clean))

    rgb_stretched = np.stack(stretched_channels, axis=-1)
    rgb_uint8 = _normalize_to_uint8(rgb_stretched)

    pil_img = Image.fromarray(rgb_uint8, mode="RGB")
    meta["output_shape"] = list(pil_img.size) + [3]
    return pil_img, meta


def _preprocess_pillow(file_path: Path) -> Tuple[Image.Image, Dict[str, Any]]:
    """Preprocess standard imagery (PNG/JPEG) via Pillow into a consistent RGB PIL Image."""
    meta: Dict[str, Any] = {
        "backend": "pillow",
        "crs": None,
    }
    try:
        with Image.open(file_path) as im:
            meta["width_px"] = im.width
            meta["height_px"] = im.height
            meta["native_mode"] = im.mode
            max_dim = 2048
            w, h = im.width, im.height
            if max(w, h) > max_dim:
                ratio = max_dim / max(w, h)
                new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
                meta["resized_from"] = [w, h]
                meta["resized_to"] = list(new_size)
                im = im.resize(new_size, Image.Resampling.LANCZOS)
            if im.mode == "RGBA":
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.split()[-1])
                im = bg
                meta["alpha_handling"] = "composited onto white"
            elif im.mode != "RGB":
                im = im.convert("RGB")
                meta["mode_conversion"] = f"{meta['native_mode']}->RGB"
            rgb_img = im.copy()
    except Exception as e:
        raise ImageryPreprocessingError(f"Pillow read failed for {file_path.name}: {e}") from e

    arr = np.asarray(rgb_img).astype(np.float32) / 255.0
    stretched = np.stack([_percent_stretch(arr[..., i]) for i in range(3)], axis=-1)
    rgb_stretched_uint8 = _normalize_to_uint8(stretched)
    pil_img = Image.fromarray(rgb_stretched_uint8, mode="RGB")
    meta["band_count"] = 3
    meta["stretch"] = "2-98 percentile (per-channel)"
    meta["output_shape"] = list(pil_img.size) + [3]
    return pil_img, meta


def preprocess_imagery_for_vqa(file_path: Path) -> PreprocessingResult:
    """Convert any supported imagery (GeoTIFF, TIFF, PNG, JPEG) into a VLM-ready PIL RGB image.

    This is the **preprocessing service entry point**. It is intentionally decoupled from
    any ML model code: the output is a plain PIL Image + a metadata dict describing what
    transformations were applied. A different downstream model adapter could reuse this
    preprocessing pipeline as-is or apply additional transforms.
    """
    if not file_path.exists():
        raise ImageryPreprocessingError(f"Input file not found: {file_path}")
    if file_path.stat().st_size == 0:
        raise ImageryPreprocessingError(f"Input file is empty (0 bytes): {file_path.name}")

    ext = file_path.suffix.lower()

    if ext in RASTERIO_EXTS and RASTERIO_AVAILABLE:
        try:
            rgb_img, meta = _preprocess_rasterio(file_path)
            return PreprocessingResult(rgb_image=rgb_img, source_path=file_path, preprocessing_meta=meta)
        except ImageryPreprocessingError:
            raise
        except Exception as e:
            logger.warning(f"rasterio preprocessing failed for {file_path.name}, falling back to Pillow: {e}")

    if ext in RASTERIO_EXTS | PIL_EXTS:
        rgb_img, meta = _preprocess_pillow(file_path)
        return PreprocessingResult(rgb_image=rgb_img, source_path=file_path, preprocessing_meta=meta)

    raise ImageryPreprocessingError(
        f"Unsupported file format for imagery preprocessing: {ext}. "
        f"Supported: {', '.join(sorted(RASTERIO_EXTS | PIL_EXTS))}"
    )
