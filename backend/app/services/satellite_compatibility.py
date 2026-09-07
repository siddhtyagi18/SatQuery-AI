"""
backend/app/services/satellite_compatibility.py
================================================
General Satellite Image Compatibility + Adaptation Layer for SatQuery-AI.

Architecture:
USER UPLOAD
    ↓
FILE VALIDATION (FastAPI)
    ↓
SATELLITE IMAGE INSPECTOR (SatelliteImageInspector)
    ↓
COMPATIBILITY + QUALITY CHECK (SatelliteCompatibilityService)
    ↓
ADAPTATION / NORMALIZATION (SatelliteInputAdapter)
    ↓
TASK-SPECIFIC SPECIALIST (with Domain Guardrails)
    ↓
RESULT + EVIDENCE + TRACE + LIMITATIONS

Core Invariants:
1. No fabricated metadata: Sensor name is "unknown" unless metadata/filename genuinely confirms it.
2. No fabricated confidence scores.
3. No silent model misuse: Unsupported domains (SAR change, flood, wildfire, agriculture on LEVIR-CD model)
   yield structured limitations and transparent guidance.
4. Scientific SAR integrity: SAR in dB is preserved; linear_to_db is NEVER applied to negative dB data.
5. Temporal integrity: If dates are missing, temporal status is "unknown", never invented.
"""
from __future__ import annotations

import math
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from PIL import Image

from ..logging_setup import logger

try:
    import rasterio
    from rasterio.enums import Resampling
    RASTERIO_AVAILABLE = True
except Exception:
    RASTERIO_AVAILABLE = False
    logger.warning("rasterio not available — satellite inspection will use Pillow fallback")

try:
    import pyproj
    PYPROJ_AVAILABLE = True
except Exception:
    PYPROJ_AVAILABLE = False


# Type aliases
ModalityHint = Literal["rgb_optical", "multispectral_optical", "sar", "grayscale", "unknown"]
SensorHint = Literal["sentinel-2", "sentinel-1", "landsat", "planet", "unknown"]
CompatibilityStatus = Literal["compatible", "adaptable", "unsupported", "insufficient_metadata", "invalid", "unsupported_for_reliable_inference", "needs_review"]
TemporalStatus = Literal["ordered", "same_time", "unknown", "invalid"]

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gtiff", ".webp", ".bmp"}
RASTERIO_EXTS = {".tif", ".tiff", ".gtiff"}

# Known SAR keywords in filenames
SAR_FILENAME_KEYWORDS = ["sar", "sentinel-1", "sentinel1", "s1a", "s1b", "radar", "_vv", "_vh", "grd", "slc"]
# Known Multispectral keywords
MS_FILENAME_KEYWORDS = ["s2a", "s2b", "sentinel-2", "sentinel2", "msil2a", "b02", "b03", "b04", "b08", "multispectral", "landsat"]


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ImageInspectionReport:
    """Detailed technical inspection of an uploaded satellite/aerial image."""
    file_path: str
    file_name: str
    format: str
    width: int = 0
    height: int = 0
    band_count: int = 0
    dtype: str = "unknown"
    bit_depth: int = 8
    min_val: float = 0.0
    max_val: float = 0.0
    mean_val: float = 0.0
    nodata_value: Optional[float] = None
    nodata_pct: float = 0.0
    crs: Optional[str] = None
    bounds: Optional[Dict[str, float]] = None  # {left, bottom, right, top}
    resolution: Optional[Tuple[float, float]] = None  # (res_x, res_y)
    geotransform: Optional[Tuple[float, ...]] = None
    modality_hint: ModalityHint = "unknown"
    sensor_hint: str = "unknown"
    is_geotiff: bool = False
    is_corrupted: bool = False
    error_message: Optional[str] = None
    band_metadata: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SingleCompatibilityReport:
    """Compatibility assessment for a single image."""
    status: CompatibilityStatus
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    required_adaptations: List[str] = field(default_factory=list)
    specialist_candidates: List[str] = field(default_factory=list)
    modality: ModalityHint = "unknown"
    sensor: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PairCompatibilityReport:
    """Compatibility assessment for two images (bi-temporal change or optical+SAR)."""
    status: CompatibilityStatus
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    required_adaptations: List[str] = field(default_factory=list)
    specialist_candidates: List[str] = field(default_factory=list)
    temporal_status: TemporalStatus = "unknown"
    temporal_notes: Optional[str] = None
    spatial_overlap_pct: Optional[float] = None
    crs_status: str = "unknown"
    dimension_status: str = "match"
    limitations: List[str] = field(default_factory=list)
    modality_pair: Tuple[str, str] = ("unknown", "unknown")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AdaptationTelemetry:
    """Scientific trace of transformations applied during adaptation."""
    original_dtype: str
    adapted_dtype: str
    original_shape: Tuple[int, ...]
    adapted_shape: Tuple[int, ...]
    normalization: str
    resampling: Optional[str] = None
    reprojection: Optional[str] = None
    selected_bands: Optional[List[int]] = None
    modality: str = "unknown"
    crs: Optional[str] = None
    resolution: Optional[Tuple[float, float]] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Satellite Image Inspector
# ---------------------------------------------------------------------------

class SatelliteImageInspector:
    """
    Reusable inspection service for PNG, JPEG, TIFF, and GeoTIFF images.
    Extracts deep metadata, radiometric statistics, and sensor hints without guessing.
    """

    @staticmethod
    def inspect(file_path: Union[str, Path]) -> ImageInspectionReport:
        p = Path(file_path)
        fname = p.name
        ext = p.suffix.lower()

        report = ImageInspectionReport(
            file_path=str(p.resolve()) if p.exists() else str(p),
            file_name=fname,
            format=ext.lstrip(".").upper() if ext else "UNKNOWN",
        )

        if not p.exists():
            report.is_corrupted = True
            report.error_message = f"File does not exist: {p}"
            return report

        try:
            fsize = p.stat().st_size
            if fsize == 0:
                report.is_corrupted = True
                report.error_message = f"File is empty (0 bytes): {fname}"
                return report
        except Exception as e:
            report.is_corrupted = True
            report.error_message = f"Stat failed: {e}"
            return report

        if ext not in SUPPORTED_IMAGE_EXTS:
            report.is_corrupted = True
            report.error_message = f"Unsupported file extension: {ext}. Supported: {sorted(SUPPORTED_IMAGE_EXTS)}"
            return report

        # 1. Try Rasterio if GeoTIFF/TIFF
        if ext in RASTERIO_EXTS and RASTERIO_AVAILABLE:
            try:
                return SatelliteImageInspector._inspect_rasterio(p, report)
            except Exception as rexc:
                logger.warning(f"rasterio inspection failed for {fname}, falling back to Pillow: {rexc}")

        # 2. Fallback / Standard Pillow Inspection
        try:
            return SatelliteImageInspector._inspect_pillow(p, report)
        except Exception as pexc:
            report.is_corrupted = True
            report.error_message = f"Failed to read image with Pillow and rasterio: {pexc}"
            return report

    @staticmethod
    def _inspect_rasterio(p: Path, report: ImageInspectionReport) -> ImageInspectionReport:
        with rasterio.open(p) as src:
            report.width = src.width
            report.height = src.height
            report.band_count = src.count
            report.dtype = str(src.dtypes[0]) if src.count > 0 else "unknown"
            report.is_geotiff = bool(src.crs is not None or src.transform is not None)

            # Bit depth
            if "8" in report.dtype:
                report.bit_depth = 8
            elif "16" in report.dtype:
                report.bit_depth = 16
            elif "32" in report.dtype:
                report.bit_depth = 32
            elif "64" in report.dtype:
                report.bit_depth = 64

            # CRS & GeoTransform
            if src.crs:
                report.crs = str(src.crs)
            if src.transform:
                report.geotransform = tuple(src.transform)[:6]
                try:
                    res_x, res_y = abs(src.transform[0]), abs(src.transform[4])
                    if res_x > 0 and res_y > 0:
                        report.resolution = (round(res_x, 6), round(res_y, 6))
                except Exception:
                    pass

            if src.bounds:
                report.bounds = {
                    "left": float(src.bounds.left),
                    "bottom": float(src.bounds.bottom),
                    "right": float(src.bounds.right),
                    "top": float(src.bounds.top),
                }

            if src.nodata is not None:
                report.nodata_value = float(src.nodata)

            # Sample data stats safely
            try:
                sample_h = min(src.height, 1024)
                sample_w = min(src.width, 1024)
                arr = src.read(
                    1,
                    out_shape=(sample_h, sample_w),
                    resampling=Resampling.nearest,
                    masked=True,
                )
                nodata_mask = np.ma.getmaskarray(arr)
                if report.nodata_value is not None:
                    nodata_mask = nodata_mask | (arr == report.nodata_value)

                data_unmasked = np.ma.filled(arr, fill_value=np.nan)
                invalid_mask = nodata_mask | ~np.isfinite(data_unmasked)
                valid_count = np.count_nonzero(~invalid_mask)
                total_count = arr.size

                if total_count > 0:
                    report.nodata_pct = round(float((total_count - valid_count) / total_count * 100.0), 2)

                if valid_count > 0:
                    valid_pixels = data_unmasked[~invalid_mask]
                    report.min_val = float(np.min(valid_pixels))
                    report.max_val = float(np.max(valid_pixels))
                    report.mean_val = float(np.mean(valid_pixels))
                else:
                    report.min_val = 0.0
                    report.max_val = 0.0
                    report.mean_val = 0.0
            except Exception as e:
                logger.debug(f"Rasterio stats sampling error for {p.name}: {e}")

            report.modality_hint, report.sensor_hint = SatelliteImageInspector._infer_modality_and_sensor(
                file_name=p.name,
                band_count=src.count,
                dtype=report.dtype,
                min_val=report.min_val,
                max_val=report.max_val,
                tags=src.tags(),
            )
            return report

    @staticmethod
    def _inspect_pillow(p: Path, report: ImageInspectionReport) -> ImageInspectionReport:
        with Image.open(p) as img:
            report.width, report.height = img.size
            bands = img.getbands()
            report.band_count = len(bands)
            report.dtype = getattr(img, "mode", "unknown")

            # Determine bit depth
            if img.mode in ("1",):
                report.bit_depth = 1
            elif img.mode in ("L", "P", "RGB", "RGBA"):
                report.bit_depth = 8
            elif img.mode in ("I;16", "I;16B", "I;16L", "I"):
                report.bit_depth = 16
            elif img.mode in ("F",):
                report.bit_depth = 32

            # Check for TIFF Geo tags if Pillow opened a TIFF
            if hasattr(img, "tag_v2"):
                tags = img.tag_v2
                scale = tags.get(33550)  # ModelPixelScaleTag
                tiepoint = tags.get(33922)  # ModelTiepointTag
                geokeys = tags.get(34735)  # GeoKeyDirectoryTag

                if scale and tiepoint:
                    report.is_geotiff = True
                    ox, oy = float(tiepoint[3]), float(tiepoint[4])
                    sx, sy = float(scale[0]), float(scale[1])
                    report.resolution = (round(sx, 6), round(sy, 6))
                    report.bounds = {
                        "left": ox,
                        "bottom": oy - report.height * sy,
                        "right": ox + report.width * sx,
                        "top": oy,
                    }
                    report.geotransform = (ox, sx, 0.0, oy, 0.0, -sy)

                if geokeys:
                    report.is_geotiff = True
                    try:
                        num_keys = geokeys[3]
                        for idx in range(num_keys):
                            k_idx = 4 + idx * 4
                            if k_idx + 3 < len(geokeys):
                                if geokeys[k_idx] in (3072, 2048) and geokeys[k_idx + 3] > 0:
                                    report.crs = f"EPSG:{geokeys[k_idx + 3]}"
                                    break
                    except Exception:
                        pass
                    if not report.crs:
                        report.crs = "GeoTIFF Embedded CRS"

            # Sample radiometric stats using thumbnail/small array
            try:
                thumb = img.copy()
                if max(thumb.size) > 1024:
                    thumb.thumbnail((1024, 1024), Image.Resampling.NEAREST)
                arr = np.asarray(thumb)
                finite_mask = np.isfinite(arr)
                if finite_mask.any():
                    valid_px = arr[finite_mask]
                    report.min_val = float(np.min(valid_px))
                    report.max_val = float(np.max(valid_px))
                    report.mean_val = float(np.mean(valid_px))
                report.nodata_pct = round(float((arr.size - np.count_nonzero(finite_mask)) / arr.size * 100.0), 2)
            except Exception as e:
                logger.debug(f"Pillow stats error: {e}")

            report.modality_hint, report.sensor_hint = SatelliteImageInspector._infer_modality_and_sensor(
                file_name=p.name,
                band_count=report.band_count,
                dtype=report.dtype,
                min_val=report.min_val,
                max_val=report.max_val,
                tags={},
            )
            return report

    @staticmethod
    def _infer_modality_and_sensor(
        file_name: str,
        band_count: int,
        dtype: str,
        min_val: float,
        max_val: float,
        tags: Dict[str, Any],
    ) -> Tuple[ModalityHint, str]:
        """
        Infers modality and sensor strictly without fabrication.
        If metadata is insufficient, sensor is 'unknown'.
        """
        name = file_name.lower()

        # 1. Sensor identification (Strict: only if authentic metadata or standard product naming pattern)
        sensor_hint = "unknown"
        if any(k in name for k in ["s2a_", "s2b_", "sentinel-2", "sentinel2", "msil2a"]):
            sensor_hint = "sentinel-2"
        elif any(k in name for k in ["s1a_", "s1b_", "sentinel-1", "sentinel1"]):
            sensor_hint = "sentinel-1"
        elif "landsat" in name:
            sensor_hint = "landsat"

        for k, v in tags.items():
            val_str = str(v).lower()
            if "sentinel-2" in val_str or "sentinel 2" in val_str:
                sensor_hint = "sentinel-2"
            elif "sentinel-1" in val_str or "sentinel 1" in val_str:
                sensor_hint = "sentinel-1"

        # 2. Modality identification
        is_sar_named = any(k in name for k in SAR_FILENAME_KEYWORDS)
        if is_sar_named or (min_val < -3.0 and max_val <= 10.0 and band_count in (1, 2)):
            return "sar", sensor_hint

        if band_count > 3 or any(name.startswith(b) or f"_{b}." in name or f"_{b}_" in name for b in ["b02", "b03", "b04", "b08", "b11", "b12"]):
            return "multispectral_optical", sensor_hint

        if band_count == 1 or dtype in ("L", "I;16", "I"):
            return "grayscale", sensor_hint

        if band_count in (3, 4) or dtype in ("RGB", "RGBA"):
            return "rgb_optical", sensor_hint

        return "unknown", sensor_hint


# ---------------------------------------------------------------------------
# Temporal Pair Validator
# ---------------------------------------------------------------------------

class TemporalValidator:
    """Validates temporal metadata without inventing ordering."""

    @staticmethod
    def extract_acquisition_date(file_name: str) -> Optional[str]:
        """Extract YYYY-MM-DD or YYYYMMDD from filename patterns."""
        patterns = [
            r"(\d{4})[-_](\d{2})[-_](\d{2})",
            r"(\d{4})(\d{2})(\d{2})T\d{6}",
            r"(\d{4})(\d{2})(\d{2})",
        ]
        for pat in patterns:
            m = re.search(pat, file_name)
            if m:
                y, mo, d = m.group(1), m.group(2), m.group(3)
                try:
                    if 1950 <= int(y) <= 2050 and 1 <= int(mo) <= 12 and 1 <= int(d) <= 31:
                        return f"{y}-{mo}-{d}"
                except ValueError:
                    continue
        return None

    @classmethod
    def validate_temporal(
        cls,
        report_a: ImageInspectionReport,
        report_b: ImageInspectionReport,
    ) -> Tuple[TemporalStatus, Optional[str]]:
        date_a = cls.extract_acquisition_date(report_a.file_name)
        date_b = cls.extract_acquisition_date(report_b.file_name)

        if not date_a or not date_b:
            return "unknown", f"Acquisition dates could not be verified from metadata (A={date_a or 'unknown'}, B={date_b or 'unknown'}). Temporal ordering is assumed per user designation."

        if date_a < date_b:
            return "ordered", f"Temporal ordering verified: T1={date_a} < T2={date_b}."
        elif date_a == date_b:
            return "same_time", f"Both images appear to have the same acquisition date: {date_a}. Genuine temporal change detection may be invalid."
        else:
            return "invalid", f"Inverted temporal ordering detected: T1 ({date_a}) is later than T2 ({date_b}). Before and after images may be swapped."


# ---------------------------------------------------------------------------
# Spatial Overlap Validator
# ---------------------------------------------------------------------------

class SpatialValidator:
    """Calculates spatial overlap and CRS alignment."""

    @staticmethod
    def compute_overlap(
        report_a: ImageInspectionReport,
        report_b: ImageInspectionReport,
    ) -> Tuple[Optional[float], str]:
        """
        Returns (overlap_percentage, crs_status).
        If bounds are missing, returns (None, 'unreferenced_raster').
        """
        if not report_a.bounds or not report_b.bounds:
            return None, "unreferenced_raster"

        if report_a.crs and report_b.crs:
            if report_a.crs != report_b.crs:
                return 0.0, f"mismatched_crs (A={report_a.crs}, B={report_b.crs})"
            crs_status = "matching_crs"
        else:
            crs_status = "crs_unverified"

        ba = report_a.bounds
        bb = report_b.bounds

        ix_min = max(ba["left"], bb["left"])
        iy_min = max(ba["bottom"], bb["bottom"])
        ix_max = min(ba["right"], bb["right"])
        iy_max = min(ba["top"], bb["top"])

        if ix_min >= ix_max or iy_min >= iy_max:
            return 0.0, crs_status

        inter_area = (ix_max - ix_min) * (iy_max - iy_min)
        area_a = (ba["right"] - ba["left"]) * (ba["top"] - ba["bottom"])
        if area_a <= 0:
            return 0.0, crs_status

        pct = round(float(inter_area / area_a * 100.0), 2)
        return pct, crs_status


# ---------------------------------------------------------------------------
# Satellite Compatibility Service
# ---------------------------------------------------------------------------

class SatelliteCompatibilityService:
    """
    Evaluates compatibility of single images or image pairs against SatQuery specialists.
    Distinguishes: COMPATIBLE, ADAPTABLE, UNSUPPORTED, INSUFFICIENT_METADATA, INVALID.
    """

    SUPPORTED_SINGLE_SPECIALISTS = ["rs_vqa", "rs_caption", "rs_grounding"]
    OPTICAL_CHANGE_SPECIALISTS = ["change_detector", "change_vqa"]
    OPTICAL_SAR_SPECIALISTS = ["optical_sar_analyzer"]

    @classmethod
    def check_single_compatibility(
        cls,
        report: ImageInspectionReport,
        task: Optional[str] = None,
    ) -> SingleCompatibilityReport:
        reasons: List[str] = []
        warnings: List[str] = []
        adaptations: List[str] = []
        specialists: List[str] = []

        if report.is_corrupted:
            return SingleCompatibilityReport(
                status="invalid",
                reasons=[f"Image is corrupted or unreadable: {report.error_message}"],
                modality=report.modality_hint,
                sensor=report.sensor_hint,
            )

        if report.width <= 0 or report.height <= 0:
            return SingleCompatibilityReport(
                status="invalid",
                reasons=["Image has invalid zero dimensions."],
                modality=report.modality_hint,
                sensor=report.sensor_hint,
            )

        if report.nodata_pct > 95.0:
            return SingleCompatibilityReport(
                status="invalid",
                reasons=[f"Image consists almost entirely of nodata / unreadable values ({report.nodata_pct}%)."],
                modality=report.modality_hint,
                sensor=report.sensor_hint,
            )

        # Modality check
        if report.modality_hint == "sar":
            adaptations.append("sar_backscatter_normalization")
            specialists.append("optical_sar_analyzer")
            warnings.append("SAR imagery requires specialized radar interpretation; optical VLM features may not directly transfer.")
            status = "adaptable"
        elif report.modality_hint == "multispectral_optical":
            adaptations.append("multispectral_to_rgb_selection")
            specialists.extend(cls.SUPPORTED_SINGLE_SPECIALISTS)
            status = "adaptable"
        elif report.modality_hint == "grayscale":
            adaptations.append("single_channel_to_pseudo_rgb")
            specialists.extend(cls.SUPPORTED_SINGLE_SPECIALISTS)
            status = "adaptable"
        elif report.modality_hint == "rgb_optical":
            specialists.extend(cls.SUPPORTED_SINGLE_SPECIALISTS)
            status = "compatible"
        else:
            status = "needs_review"
            warnings.append("Sensor modality could not be determined; results may be unreliable.")
            reasons.append("Sensor modality could not be determined from metadata.")

        # Dtype adaptations
        if report.bit_depth > 8 or "16" in report.dtype or "float" in report.dtype:
            adaptations.append(f"{report.dtype}_to_normalized_float32")
            if status == "compatible":
                status = "adaptable"

        if report.width > 2048 or report.height > 2048:
            adaptations.append("dimension_downsampling_max_2048")
            if status == "compatible":
                status = "adaptable"

        reasons.append(f"Image inspected: {report.width}x{report.height}px, {report.band_count} band(s), {report.dtype}.")
        return SingleCompatibilityReport(
            status=status,
            reasons=reasons,
            warnings=warnings,
            required_adaptations=adaptations,
            specialist_candidates=specialists,
            modality=report.modality_hint,
            sensor=report.sensor_hint,
        )

    @classmethod
    def check_pair_compatibility(
        cls,
        report_a: ImageInspectionReport,
        report_b: ImageInspectionReport,
        mode: str,
        query: Optional[str] = None,
    ) -> PairCompatibilityReport:
        reasons: List[str] = []
        warnings: List[str] = []
        adaptations: List[str] = []
        specialists: List[str] = []
        limitations: List[str] = []

        # Single image sanity checks
        comp_a = cls.check_single_compatibility(report_a)
        comp_b = cls.check_single_compatibility(report_b)

        if comp_a.status == "invalid" or comp_b.status == "invalid":
            errs = comp_a.reasons + comp_b.reasons
            return PairCompatibilityReport(
                status="invalid",
                reasons=[f"Pair contains invalid imagery: {'; '.join(errs)}"],
                modality_pair=(report_a.modality_hint, report_b.modality_hint),
            )

        # Temporal analysis
        temp_status, temp_notes = TemporalValidator.validate_temporal(report_a, report_b)
        if temp_status == "same_time":
            warnings.append("Both images have identical acquisition dates. Change detection on identical timestamps may detect noise or registration artifacts rather than genuine temporal change.")
        elif temp_status == "invalid":
            warnings.append("Inverted temporal timestamps detected (Image A is chronologically later than Image B).")

        # Spatial overlap analysis
        overlap_pct, crs_status = SpatialValidator.compute_overlap(report_a, report_b)
        if overlap_pct is not None:
            if overlap_pct == 0.0:
                warnings.append("Spatial overlap between Image A and Image B is 0.0% (images cover non-overlapping geographic regions).")
                limitations.append("Spatial non-overlap: change detection will yield meaningless results because the images do not cover the same area.")
            elif overlap_pct < 50.0:
                warnings.append(f"Low spatial overlap ({overlap_pct}%). Peripheral changes may reflect boundary misalignment.")

        # Dimension alignment
        dim_status = "match"
        if (report_a.width, report_a.height) != (report_b.width, report_b.height):
            dim_status = "mismatch"
            adaptations.append("spatial_resampling_to_match_dimensions")
            warnings.append(f"Dimension mismatch: A={report_a.width}x{report_a.height}, B={report_b.width}x{report_b.height}. Image B will be deterministically resampled.")

        # Mode-specific evaluations
        # Modality check: If modality cannot be determined, flag needs_review
        if report_a.modality_hint == "unknown" or report_b.modality_hint == "unknown":
            return PairCompatibilityReport(
                status="needs_review",
                reasons=["Sensor modality could not be determined from input metadata for one or both images."],
                warnings=["Sensor modality could not be determined; results may be unreliable."],
                specialist_candidates=[],
                temporal_status=temp_status,
                temporal_notes=temp_notes,
                spatial_overlap_pct=overlap_pct,
                crs_status=crs_status,
                dimension_status=dim_status,
                limitations=[
                    "Sensor modality could not be determined; results may be unreliable.",
                    "The system will not silently execute specialist models on unverified or unidentified imagery.",
                ],
                modality_pair=(report_a.modality_hint, report_b.modality_hint),
            )

        if mode == "bi_temporal":
            # 1. Modality check: Both images must be optical
            if report_a.modality_hint == "sar" or report_b.modality_hint == "sar":
                return PairCompatibilityReport(
                    status="unsupported_for_reliable_inference",
                    reasons=["Bi-temporal change detection specialist currently does not support SAR-only or mixed SAR-Optical change detection."],
                    warnings=["Input contains SAR imagery for bi_temporal mode."],
                    specialist_candidates=[],
                    temporal_status=temp_status,
                    temporal_notes=temp_notes,
                    spatial_overlap_pct=overlap_pct,
                    crs_status=crs_status,
                    dimension_status=dim_status,
                    limitations=[
                        "SAR change detection is currently unsupported. The change detection model is specialized for optical structural/building change."
                    ],
                    modality_pair=(report_a.modality_hint, report_b.modality_hint),
                )

            # 2. Spatial non-overlap check
            if overlap_pct is not None and overlap_pct == 0.0:
                return PairCompatibilityReport(
                    status="unsupported_for_reliable_inference",
                    reasons=["Spatial overlap between Image A and Image B is 0.0% (images cover non-overlapping geographic regions)."],
                    warnings=["Spatial non-overlap detected (0.0%)."],
                    specialist_candidates=[],
                    temporal_status=temp_status,
                    temporal_notes=temp_notes,
                    spatial_overlap_pct=overlap_pct,
                    crs_status=crs_status,
                    dimension_status=dim_status,
                    limitations=[
                        "Spatial non-overlap: change detection cannot produce meaningful comparative results between two completely different geographic locations."
                    ],
                    modality_pair=(report_a.modality_hint, report_b.modality_hint),
                )

            # 3. Query domain check: flood, wildfire, agriculture, etc.
            unsupported_domains = {
                "flood": "flood/water inundation change",
                "wildfire": "wildfire burn severity change",
                "fire": "wildfire burn scar change",
                "forest": "deforestation/forestry change",
                "deforestation": "deforestation change",
                "crop": "agricultural crop cycle change",
                "agriculture": "agricultural change",
                "coastline": "coastal erosion change",
            }
            q_lower = (query or "").lower()
            detected_unsupported_domains = [desc for word, desc in unsupported_domains.items() if word in q_lower]
            if detected_unsupported_domains:
                return PairCompatibilityReport(
                    status="unsupported_for_reliable_inference",
                    reasons=[
                        f"Queried change domain ({', '.join(detected_unsupported_domains)}) is outside the validated domain of the current change detector."
                    ],
                    warnings=[f"Query targets {', '.join(detected_unsupported_domains)}."],
                    specialist_candidates=[],
                    temporal_status=temp_status,
                    temporal_notes=temp_notes,
                    spatial_overlap_pct=overlap_pct,
                    crs_status=crs_status,
                    dimension_status=dim_status,
                    limitations=[
                        "The current change detector is specialized for LEVIR-CD-style optical building/structural change.",
                        f"Domain '{', '.join(detected_unsupported_domains)}' is outside the validated training distribution and cannot be reliably segmented.",
                    ],
                    modality_pair=(report_a.modality_hint, report_b.modality_hint),
                )

            # 4. Sensor & Resolution / Scale Check (LEVIR-CD VHR ~0.5m/px vs Sentinel-2 10m-60m)
            is_coarse_sensor = (report_a.sensor_hint in ("sentinel-2", "landsat") or report_b.sensor_hint in ("sentinel-2", "landsat"))
            has_coarse_resolution = False
            for rep in (report_a, report_b):
                if rep.resolution and (rep.resolution[0] > 3.0 or rep.resolution[1] > 3.0):
                    has_coarse_resolution = True
                    break

            if is_coarse_sensor or has_coarse_resolution:
                sensor_name = report_a.sensor_hint if report_a.sensor_hint != "unknown" else report_b.sensor_hint
                res_str = f"~{max(report_a.resolution or (10.0, 10.0))}m/pixel" if (report_a.resolution or report_b.resolution) else "moderate-resolution satellite scale"
                return PairCompatibilityReport(
                    status="unsupported_for_reliable_inference",
                    reasons=[
                        f"Spatial resolution scale mismatch: Provided imagery is {sensor_name} ({res_str}). The current change detector is trained on Very High Resolution (VHR) imagery (~0.5m/pixel).",
                        "At moderate resolution, individual building structural contours cannot be reliably segmented by this specialist.",
                    ],
                    warnings=[f"Detected moderate-resolution satellite imagery ({sensor_name})."],
                    specialist_candidates=[],
                    temporal_status=temp_status,
                    temporal_notes=temp_notes,
                    spatial_overlap_pct=overlap_pct,
                    crs_status=crs_status,
                    dimension_status=dim_status,
                    limitations=[
                        "The current change detector is validated primarily on LEVIR-CD-style high-resolution optical building/structural change (~0.5m/pixel).",
                        f"Medium-resolution satellite imagery ({sensor_name}) is unsupported for reliable structural change inference.",
                    ],
                    modality_pair=(report_a.modality_hint, report_b.modality_hint),
                )

            specialists.extend(cls.OPTICAL_CHANGE_SPECIALISTS)
            status: CompatibilityStatus = "compatible" if not adaptations and not warnings else "adaptable"

            # Always record the honest domain boundary
            limitations.append("Current change detector is validated primarily on LEVIR-CD-style optical building/structural change.")

            return PairCompatibilityReport(
                status=status,
                reasons=reasons or ["Optical bi-temporal imagery verified."],
                warnings=warnings,
                required_adaptations=adaptations,
                specialist_candidates=specialists,
                temporal_status=temp_status,
                temporal_notes=temp_notes,
                spatial_overlap_pct=overlap_pct,
                crs_status=crs_status,
                dimension_status=dim_status,
                limitations=limitations,
                modality_pair=(report_a.modality_hint, report_b.modality_hint),
            )

        elif mode == "optical_sar":
            modalities = {report_a.modality_hint, report_b.modality_hint}
            has_sar = "sar" in modalities
            has_optical = bool(modalities & {"rgb_optical", "multispectral_optical", "grayscale"})

            if not has_sar:
                warnings.append("Neither image was identified as SAR backscatter. Optical+SAR cross-modal fusion requires at least one SAR raster.")
                limitations.append("Optical+SAR analyzer requires authentic or proxy SAR backscatter.")
                return PairCompatibilityReport(
                    status="unsupported",
                    reasons=["Missing SAR modality for optical_sar analysis."],
                    warnings=warnings,
                    specialist_candidates=[],
                    limitations=limitations,
                    modality_pair=(report_a.modality_hint, report_b.modality_hint),
                )

            if not has_optical:
                warnings.append("Neither image was identified as Optical imagery.")
                limitations.append("Optical+SAR analyzer requires an optical raster to fuse with SAR.")
                return PairCompatibilityReport(
                    status="unsupported",
                    reasons=["Missing Optical modality for optical_sar analysis."],
                    warnings=warnings,
                    specialist_candidates=[],
                    limitations=limitations,
                    modality_pair=(report_a.modality_hint, report_b.modality_hint),
                )

            # SAR checks: verify dB vs linear
            sar_rep = report_b if report_b.modality_hint == "sar" else report_a
            if sar_rep.min_val < -1.0:
                adaptations.append("preserve_sar_db_representation")
            else:
                adaptations.append("convert_sar_linear_power_to_db")

            specialists.extend(cls.OPTICAL_SAR_SPECIALISTS)
            limitations.append("Optical+SAR fusion baseline is demonstrated on Sentinel-1 VV/VH and Sentinel-2 optical bands; general arbitrary satellite fusion is not guaranteed.")

            return PairCompatibilityReport(
                status="adaptable",
                reasons=["Compatible Optical and SAR modalities identified."],
                warnings=warnings,
                required_adaptations=adaptations,
                specialist_candidates=specialists,
                temporal_status=temp_status,
                temporal_notes=temp_notes,
                spatial_overlap_pct=overlap_pct,
                crs_status=crs_status,
                dimension_status=dim_status,
                limitations=limitations,
                modality_pair=(report_a.modality_hint, report_b.modality_hint),
            )

        else:
            return PairCompatibilityReport(
                status="compatible",
                reasons=["General pair inspected."],
                warnings=warnings,
                required_adaptations=adaptations,
                specialist_candidates=specialists,
                temporal_status=temp_status,
                temporal_notes=temp_notes,
                spatial_overlap_pct=overlap_pct,
                crs_status=crs_status,
                dimension_status=dim_status,
                limitations=limitations,
                modality_pair=(report_a.modality_hint, report_b.modality_hint),
            )


# ---------------------------------------------------------------------------
# Satellite Input Adapter
# ---------------------------------------------------------------------------

class SatelliteInputAdapter:
    """
    Deterministic adaptation layer for satellite imagery.
    Produces float32 normalized representations and rich telemetry.
    """

    @staticmethod
    def adapt_optical_to_numpy(
        report: ImageInspectionReport,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, AdaptationTelemetry]:
        """
        Loads and adapts an optical image into float32 (H, W, 3) in [0, 1].
        """
        p = Path(report.file_path)
        notes: List[str] = []
        resampling = None

        # 1. Read raw raster
        if report.format in ("TIF", "TIFF", "GTIFF") and RASTERIO_AVAILABLE:
            with rasterio.open(p) as src:
                orig_shape = (src.count, src.height, src.width)
                orig_dtype = str(src.dtypes[0])
                data = src.read(masked=True)
                raw = np.ma.filled(data, fill_value=np.nan)
        else:
            with Image.open(p) as img:
                orig_shape = (len(img.getbands()), img.height, img.width)
                orig_dtype = str(getattr(img, "mode", "unknown"))
                arr = np.asarray(img)
                if arr.ndim == 2:
                    raw = arr[np.newaxis, ...]
                else:
                    raw = np.moveaxis(arr, -1, 0)  # (C, H, W)

        # 2. Band Selection to 3 channels
        c, h, w = raw.shape
        selected_bands = [1, 2, 3]
        if c >= 3:
            rgb_bands = raw[:3].astype(np.float32)
            selected_bands = [1, 2, 3]
        elif c == 1:
            rgb_bands = np.repeat(raw[:1].astype(np.float32), 3, axis=0)
            selected_bands = [1, 1, 1]
            notes.append("Replicated single-band grayscale to 3 channels.")
        elif c == 2:
            rgb_bands = np.stack([raw[0], raw[1], raw[0]], axis=0).astype(np.float32)
            selected_bands = [1, 2, 1]
            notes.append("Dual-channel image: synthesized 3rd channel.")
        else:
            raise ValueError(f"Invalid channel count: {c}")

        # Move to (H, W, 3)
        hwc = np.moveaxis(rgb_bands, 0, -1)

        # 3. Normalization to [0, 1]
        raw_max = float(np.nanmax(hwc)) if np.isfinite(hwc).any() else 1.0
        raw_min = float(np.nanmin(hwc)) if np.isfinite(hwc).any() else 0.0

        if raw_max > 255.0:
            scale = 1.0 / 10000.0 if raw_max <= 20000.0 else 1.0 / 65535.0
            norm_name = f"uint16_scale_{int(1/scale)}"
            adapted = np.clip(hwc * scale, 0.0, 1.0)
            notes.append(f"Normalized high dynamic range data (max={raw_max:.1f}) using {norm_name}.")
        elif raw_max > 1.0 or "uint8" in orig_dtype.lower() or orig_dtype in ("RGB", "RGBA", "L"):
            adapted = np.clip(hwc / 255.0, 0.0, 1.0)
            norm_name = "uint8_div_255"
        else:
            adapted = np.clip(hwc, 0.0, 1.0)
            norm_name = "unit_float_passthrough"

        # Handle NaNs or infinities
        nan_mask = ~np.isfinite(adapted)
        if nan_mask.any():
            adapted[nan_mask] = 0.0
            notes.append(f"Replaced {np.count_nonzero(nan_mask)} non-finite pixels with 0.0.")

        # 4. Resizing if target_size specified
        if target_size and (adapted.shape[1], adapted.shape[0]) != target_size:
            target_w, target_h = target_size
            pil_temp = Image.fromarray((adapted * 255.0).astype(np.uint8))
            pil_resized = pil_temp.resize((target_w, target_h), Image.Resampling.BILINEAR)
            adapted = np.asarray(pil_resized).astype(np.float32) / 255.0
            resampling = f"bilinear_{adapted.shape[1]}x{adapted.shape[0]}->{target_w}x{target_h}"
            notes.append(f"Resampled image to {target_w}x{target_h}.")

        telemetry = AdaptationTelemetry(
            original_dtype=orig_dtype,
            adapted_dtype="float32",
            original_shape=orig_shape,
            adapted_shape=adapted.shape,
            normalization=norm_name,
            resampling=resampling,
            selected_bands=selected_bands,
            modality="optical",
            crs=report.crs,
            resolution=report.resolution,
            notes=notes,
        )
        return adapted, telemetry

    @staticmethod
    def adapt_sar_to_numpy(
        report: ImageInspectionReport,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, AdaptationTelemetry]:
        """
        Loads and adapts SAR imagery into float32 (H, W) dB backscatter.
        Strictly avoids applying 10*log10 to already-negative dB data.
        """
        p = Path(report.file_path)
        notes: List[str] = []
        resampling = None

        if report.format in ("TIF", "TIFF", "GTIFF") and RASTERIO_AVAILABLE:
            with rasterio.open(p) as src:
                orig_shape = (src.height, src.width)
                orig_dtype = str(src.dtypes[0])
                arr = src.read(1).astype(np.float32)
                if src.nodata is not None:
                    arr = np.where(arr == src.nodata, np.nan, arr)
        else:
            with Image.open(p) as img:
                orig_shape = (img.height, img.width)
                orig_dtype = getattr(img, "mode", "unknown")
                arr = np.asarray(img, dtype=np.float32)
                if arr.ndim == 3:
                    arr = arr[..., 0]

        raw_min = float(np.nanmin(arr)) if np.isfinite(arr).any() else 0.0
        raw_max = float(np.nanmax(arr)) if np.isfinite(arr).any() else 1.0

        # Scientific SAR scale detection
        if raw_min < -1.0 and raw_max <= 20.0:
            sar_db = arr
            norm_name = "sar_db_preservation"
            notes.append(f"Input is already in calibrated decibel (dB) scale (min={raw_min:.2f}dB, max={raw_max:.2f}dB). Logarithmic transform skipped.")
        else:
            valid_intensity = np.where(arr > 0, arr, 1e-6)
            sar_db = 10.0 * np.log10(valid_intensity)
            norm_name = "linear_power_to_db_10log10"
            notes.append(f"Converted linear backscatter power to decibel scale using 10*log10(I + 1e-6).")

        sar_db = np.clip(sar_db, -45.0, 15.0)

        nan_mask = ~np.isfinite(sar_db)
        if nan_mask.any():
            sar_db[nan_mask] = -25.0
            notes.append(f"Imputed {np.count_nonzero(nan_mask)} radar nodata/NaN pixels with -25.0 dB.")

        if target_size and (sar_db.shape[1], sar_db.shape[0]) != target_size:
            target_w, target_h = target_size
            db_norm = (sar_db - (-45.0)) / (15.0 - (-45.0))
            pil_temp = Image.fromarray((db_norm * 255.0).astype(np.uint8), mode="L")
            pil_resized = pil_temp.resize((target_w, target_h), Image.Resampling.BILINEAR)
            resampled_norm = np.asarray(pil_resized, dtype=np.float32) / 255.0
            sar_db = resampled_norm * 60.0 - 45.0
            resampling = f"bilinear_{sar_db.shape[1]}x{sar_db.shape[0]}->{target_w}x{target_h}"
            notes.append(f"Resampled SAR grid to {target_w}x{target_h}.")

        telemetry = AdaptationTelemetry(
            original_dtype=orig_dtype,
            adapted_dtype="float32",
            original_shape=orig_shape,
            adapted_shape=sar_db.shape,
            normalization=norm_name,
            resampling=resampling,
            modality="sar",
            crs=report.crs,
            resolution=report.resolution,
            notes=notes,
        )
        return sar_db, telemetry

    @staticmethod
    def adapt_bitemporal_pair(
        report_a: ImageInspectionReport,
        report_b: ImageInspectionReport,
    ) -> Tuple[np.ndarray, np.ndarray, AdaptationTelemetry, AdaptationTelemetry]:
        """
        Adapts two optical images for bi-temporal change detection.
        Ensures identical spatial dimensions and normalized float32 format.
        """
        arr_a, tele_a = SatelliteInputAdapter.adapt_optical_to_numpy(report_a)
        target_size = (arr_a.shape[1], arr_a.shape[0])
        arr_b, tele_b = SatelliteInputAdapter.adapt_optical_to_numpy(report_b, target_size=target_size)
        return arr_a, arr_b, tele_a, tele_b
