import re
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image

from ..config import get_settings
from ..logging_setup import logger

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except Exception:
    RASTERIO_AVAILABLE = False
    logger.warning("rasterio not available — GeoTIFF metadata extraction will be limited")


settings = get_settings()


SAR_KEYWORDS = ["sar", "risat", "sentinel-1", "sentinel1", "radar", "vv", "vh", "hh", "hv", "c-band", "x-band"]
MULTISPECTRAL_KEYWORDS = ["liss", "msi", "multispectral", "s2", "sentinel-2", "sentinel2", "awifs", "ndvi", "4band", "8band"]
OPTICAL_KEYWORDS = ["pan", "optical", "cartosat", "resourcesat", "rgb", "panchromatic", "landsat"]


def detect_modality(file_name: str) -> Tuple[str, Optional[float]]:
    name = file_name.lower()
    if any(k in name for k in SAR_KEYWORDS):
        return "sar", 0.8
    if any(k in name for k in MULTISPECTRAL_KEYWORDS):
        return "multispectral", 0.78
    if any(k in name for k in OPTICAL_KEYWORDS):
        return "optical", 0.75
    ext = Path(file_name).suffix.lower()
    if ext in (".tif", ".tiff"):
        return "optical", 0.55
    return "unknown", None


def detect_format(file_name: str, content_type: Optional[str] = None) -> str:
    ext = Path(file_name).suffix.lower()
    if ext in (".tif", ".tiff"):
        return "GeoTIFF"
    if ext == ".png":
        return "PNG"
    if ext in (".jpg", ".jpeg"):
        return "JPEG"
    if content_type:
        ct = content_type.lower()
        if "tiff" in ct:
            return "TIFF"
        if "png" in ct:
            return "PNG"
        if "jpeg" in ct or "jpg" in ct:
            return "JPEG"
    return "TIFF"


def extract_acquisition_date(file_name: str) -> Optional[str]:
    patterns = [
        r"(\d{4})[-_](\d{2})[-_](\d{2})",
        r"(\d{4})(\d{2})(\d{2})",
    ]
    for pat in patterns:
        m = re.search(pat, file_name)
        if m:
            y, mo, d = m.group(1), m.group(2), m.group(3)
            try:
                if 1900 < int(y) < 2100 and 1 <= int(mo) <= 12 and 1 <= int(d) <= 31:
                    return f"{y}-{mo}-{d}"
            except ValueError:
                continue
    return None


def extract_pillow_metadata(file_path: Path) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    try:
        with Image.open(file_path) as im:
            meta["width_px"] = im.width
            meta["height_px"] = im.height
            bands = len(im.getbands())
            meta["band_count"] = bands
    except Exception as e:
        logger.debug(f"Pillow metadata extraction failed for {file_path.name}: {e}")
    return meta


def extract_rasterio_metadata(file_path: Path) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    if not RASTERIO_AVAILABLE:
        return meta
    try:
        with rasterio.open(file_path) as src:
            meta["width_px"] = src.width
            meta["height_px"] = src.height
            meta["band_count"] = src.count
            if src.crs:
                meta["crs"] = str(src.crs)
            if src.transform:
                try:
                    res_x, res_y = src.transform[0], abs(src.transform[4])
                    if res_x > 0 and res_y > 0:
                        meta["gsd_meters"] = round((res_x + res_y) / 2, 4)
                except Exception:
                    pass
            try:
                bounds = src.bounds
                meta["bounds"] = {
                    "left": bounds.left,
                    "bottom": bounds.bottom,
                    "right": bounds.right,
                    "top": bounds.top,
                }
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"rasterio metadata extraction failed for {file_path.name}: {e}")
    return meta


def validate_and_extract_metadata(
    file_name: str,
    file_path: Path,
    file_size_bytes: int,
    mime_type: Optional[str] = None,
) -> Dict[str, Any]:
    ext = Path(file_name).suffix.lower()
    allowed = set(settings.allowed_extensions_list)
    if ext not in allowed:
        raise ValueError(
            f"Unsupported file extension {ext}. Allowed: {', '.join(sorted(allowed))}"
        )

    if file_size_bytes > settings.max_upload_bytes:
        raise ValueError(
            f"File too large: {file_size_bytes} bytes. Max: {settings.max_upload_bytes} bytes "
            f"({settings.MAX_UPLOAD_SIZE_MB} MB)"
        )

    file_format = detect_format(file_name, mime_type)
    modality, modality_conf = detect_modality(file_name)
    acquisition_date = extract_acquisition_date(file_name)

    meta: Dict[str, Any] = {
        "file_format": file_format,
        "file_size_bytes": file_size_bytes,
        "mime_type": mime_type,
        "modality": modality,
        "modality_confidence": modality_conf,
        "acquisition_date": acquisition_date,
    }

    if file_format in ("GeoTIFF", "TIFF"):
        ri_meta = extract_rasterio_metadata(file_path)
        meta.update(ri_meta)
    else:
        p_meta = extract_pillow_metadata(file_path)
        meta.update(p_meta)

    return meta


def unique_stored_name(original_name: str) -> str:
    suf = Path(original_name).suffix or ".bin"
    return f"{uuid.uuid4().hex}{suf}"
