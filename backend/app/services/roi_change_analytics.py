"""
backend/app/services/roi_change_analytics.py
--------------------------------------------
Interactive Region of Interest (ROI) Change Analytics service for SatQuery-AI.

Consumes the EXISTING binary change mask produced by Change Detection without
rerunning the Siamese U-Net or classical change detector.

Rules & Guarantees:
1. NEVER reruns or alters the Change Detection model.
2. NEVER regenerates or mutates the original change mask.
3. NEVER fabricates or assumes spatial metadata (GSD, CRS, etc.).
4. Physical area is calculated ONLY when reliable spatial resolution metadata is present.
5. Connected components / hotspots are strictly geometric and contain NO semantic labels.
6. Confidence strictly remains None / uncalibrated.
7. Change VQA is invoked ONLY when user explicitly triggers it, never on ROI selection.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from scipy import ndimage

from ..logging_setup import logger


def validate_and_clamp_roi(
    roi_bounds: Dict[str, float],
    image_width: int,
    image_height: int,
    is_normalized: bool = True,
) -> Tuple[int, int, int, int]:
    """
    Validate, normalize, and clamp ROI coordinates to integer pixel coordinates.

    Parameters
    ----------
    roi_bounds : dict
        Dict with keys 'x1', 'y1', 'x2', 'y2'.
    image_width : int
        Width of the reference image in pixels.
    image_height : int
        Height of the reference image in pixels.
    is_normalized : bool
        If True, coordinates are expected in [0.0, 1.0] range.

    Returns
    -------
    Tuple[int, int, int, int]
        Clamped pixel bounds (x1, y1, x2, y2) where x1 < x2 and y1 < y2.

    Raises
    ------
    ValueError
        If ROI coordinates are non-numeric, degenerate (zero width/height),
        or completely outside image bounds.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Invalid image dimensions: ({image_width}x{image_height})")

    try:
        x1_raw = float(roi_bounds["x1"])
        y1_raw = float(roi_bounds["y1"])
        x2_raw = float(roi_bounds["x2"])
        y2_raw = float(roi_bounds["y2"])
    except (KeyError, TypeError, ValueError) as err:
        raise ValueError(f"Malformed ROI coordinates; must provide numeric x1, y1, x2, y2: {err}")

    # Handle reversed coordinates gracefully by sorting
    x_min_raw = min(x1_raw, x2_raw)
    x_max_raw = max(x1_raw, x2_raw)
    y_min_raw = min(y1_raw, y2_raw)
    y_max_raw = max(y1_raw, y2_raw)

    if is_normalized:
        # Scale [0, 1] normalized coordinates to pixel dimensions
        x1_px = int(round(x_min_raw * image_width))
        y1_px = int(round(y_min_raw * image_height))
        x2_px = int(round(x_max_raw * image_width))
        y2_px = int(round(y_max_raw * image_height))
    else:
        x1_px = int(round(x_min_raw))
        y1_px = int(round(y_min_raw))
        x2_px = int(round(x_max_raw))
        y2_px = int(round(y_max_raw))

    # Clamp safely to image boundaries
    x1_clamped = max(0, min(image_width, x1_px))
    x2_clamped = max(0, min(image_width, x2_px))
    y1_clamped = max(0, min(image_height, y1_px))
    y2_clamped = max(0, min(image_height, y2_px))

    # Validate non-zero dimensions
    roi_w = x2_clamped - x1_clamped
    roi_h = y2_clamped - y1_clamped

    if roi_w <= 0 or roi_h <= 0:
        raise ValueError(
            f"Invalid ROI bounds: selection yields zero width or height "
            f"({roi_w}x{roi_h} px) within image boundaries (0,0, {image_width},{image_height})."
        )

    return (x1_clamped, y1_clamped, x2_clamped, y2_clamped)


def _compute_roi_hotspots(
    roi_mask: np.ndarray,
    offset_x: int,
    offset_y: int,
    min_hotspot_size_px: int = 5,
    max_hotspots_return: int = 10,
) -> Dict[str, Any]:
    """
    Identify connected components inside the ROI mask using 8-connectivity.
    Hotspot centroids are returned in original global image coordinates.
    """
    total_changed = int(np.sum(roi_mask > 0))
    if total_changed == 0:
        return {
            "hotspots_count_total": 0,
            "hotspots_count_significant": 0,
            "largest_hotspot": None,
            "hotspots": [],
        }

    # 8-connectivity structure
    structure = ndimage.generate_binary_structure(2, 2)
    labeled_array, num_features = ndimage.label(roi_mask > 0, structure=structure)

    if num_features == 0:
        return {
            "hotspots_count_total": 0,
            "hotspots_count_significant": 0,
            "largest_hotspot": None,
            "hotspots": [],
        }

    component_sizes = ndimage.sum(roi_mask > 0, labeled_array, range(1, num_features + 1))
    if isinstance(component_sizes, (int, float, np.floating, np.integer)):
        component_sizes = [int(component_sizes)]
    else:
        component_sizes = [int(s) for s in component_sizes]

    # Find objects (bounding boxes) and center of mass
    objects = ndimage.find_objects(labeled_array)
    centers = ndimage.center_of_mass(roi_mask > 0, labeled_array, range(1, num_features + 1))
    if not isinstance(centers, list):
        centers = [centers]

    hotspots: List[Dict[str, Any]] = []
    largest_hotspot: Optional[Dict[str, Any]] = None
    max_size = 0

    for i, (size, obj_slice, center) in enumerate(zip(component_sizes, objects, centers)):
        if size <= 0:
            continue
        pct_of_roi_change = round((size / total_changed) * 100.0, 2)
        local_cy, local_cx = center
        global_cx = round(offset_x + float(local_cx), 1)
        global_cy = round(offset_y + float(local_cy), 1)

        # Slice gives [slice(y_min, y_max), slice(x_min, x_max)]
        y_slice, x_slice = obj_slice
        bbox_global = [
            offset_x + x_slice.start,
            offset_y + y_slice.start,
            offset_x + x_slice.stop,
            offset_y + y_slice.stop,
        ]

        item = {
            "id": i + 1,
            "pixel_area": size,
            "pct_of_roi_change": pct_of_roi_change,
            "centroid_px": [global_cx, global_cy],
            "bbox_px": bbox_global,
        }

        if size > max_size:
            max_size = size
            largest_hotspot = item

        if size >= min_hotspot_size_px:
            hotspots.append(item)

    # Sort descending by pixel area
    hotspots.sort(key=lambda h: h["pixel_area"], reverse=True)
    significant_count = len(hotspots)

    return {
        "hotspots_count_total": num_features,
        "hotspots_count_significant": significant_count,
        "largest_hotspot": largest_hotspot,
        "hotspots": hotspots[:max_hotspots_return],
    }


def compute_roi_change_analytics(
    binary_mask: np.ndarray,
    dimensions: Tuple[int, int],
    roi_bounds: Dict[str, float],
    is_normalized: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
    global_changed_pct: Optional[float] = None,
    min_hotspot_size_px: int = 5,
) -> Dict[str, Any]:
    """
    Compute comprehensive ROI analytics strictly from the existing binary change mask.

    Parameters
    ----------
    binary_mask : np.ndarray
        2D binary change mask (0 = unchanged, >0 = changed).
    dimensions : Tuple[int, int]
        (width, height) of the full image.
    roi_bounds : Dict[str, float]
        Bounding box dict: {'x1': ..., 'y1': ..., 'x2': ..., 'y2': ...}.
    is_normalized : bool
        Whether bounds are in [0, 1] relative units.
    metadata : Optional[Dict[str, Any]]
        Satellite metadata dictionary from image inspection.
    global_changed_pct : Optional[float]
        Global change percentage from the full scene.
    min_hotspot_size_px : int
        Minimum connected-component size to be reported in significant hotspots list.

    Returns
    -------
    Dict[str, Any]
        Structured ROI change analytics payload.
    """
    width, height = dimensions
    if binary_mask.ndim != 2:
        raise ValueError(f"Change mask must be 2-dimensional (H, W); got {binary_mask.shape}")

    # 1. Validate & clamp bounds
    x1, y1, x2, y2 = validate_and_clamp_roi(
        roi_bounds, width, height, is_normalized=is_normalized
    )

    roi_width = x2 - x1
    roi_height = y2 - y1
    roi_total_pixels = roi_width * roi_height

    # 2. Extract slice from existing mask (NO model rerun, instant <2ms)
    roi_slice = binary_mask[y1:y2, x1:x2]
    roi_changed_pixels = int(np.sum(roi_slice > 0))
    roi_unchanged_pixels = roi_total_pixels - roi_changed_pixels

    roi_changed_pct = round((roi_changed_pixels / roi_total_pixels) * 100.0, 2) if roi_total_pixels > 0 else 0.0
    roi_unchanged_pct = round(100.0 - roi_changed_pct, 2) if roi_total_pixels > 0 else 100.0

    # 3. Global comparison
    comparison: Dict[str, Any] = {
        "global_changed_percentage": global_changed_pct,
        "roi_changed_percentage": roi_changed_pct,
        "difference_percentage": None,
        "relative_density_factor": None,
        "summary": None,
    }
    if global_changed_pct is not None:
        diff = round(roi_changed_pct - global_changed_pct, 2)
        comparison["difference_percentage"] = diff
        if global_changed_pct > 0.0:
            factor = round(roi_changed_pct / global_changed_pct, 2)
            comparison["relative_density_factor"] = factor
            if factor > 1.05:
                comparison["summary"] = f"Selected ROI has {factor}x higher change density than the overall scene."
            elif factor < 0.95:
                comparison["summary"] = f"Selected ROI has lower change density ({factor}x) than the overall scene."
            else:
                comparison["summary"] = "Selected ROI change density is approximately equal to the scene average."
        else:
            comparison["relative_density_factor"] = None
            comparison["summary"] = "Scene baseline change is 0.00%."

    # 4. Physical Area (Rules: GSD metadata MUST be present and reliable; NO assumptions)
    physical_area: Dict[str, Any] = {
        "available": False,
        "gsd_meters": None,
        "roi_total_area_m2": None,
        "roi_total_area_hectares": None,
        "roi_total_area_sqkm": None,
        "roi_changed_area_m2": None,
        "roi_changed_area_hectares": None,
        "roi_changed_area_sqkm": None,
        "reason": "Physical area unavailable — reliable spatial resolution metadata was not provided.",
    }

    if metadata and isinstance(metadata, dict):
        gsd_val = metadata.get("gsdMeters") or metadata.get("resolution") or metadata.get("pixel_size_m")
        if gsd_val is not None:
            try:
                gsd = float(gsd_val)
                if gsd > 0:
                    px_area_m2 = gsd * gsd
                    total_m2 = roi_total_pixels * px_area_m2
                    changed_m2 = roi_changed_pixels * px_area_m2

                    physical_area["available"] = True
                    physical_area["gsd_meters"] = gsd
                    physical_area["roi_total_area_m2"] = round(total_m2, 2)
                    physical_area["roi_total_area_hectares"] = round(total_m2 / 10000.0, 4)
                    physical_area["roi_total_area_sqkm"] = round(total_m2 / 1_000_000.0, 6)
                    physical_area["roi_changed_area_m2"] = round(changed_m2, 2)
                    physical_area["roi_changed_area_hectares"] = round(changed_m2 / 10000.0, 4)
                    physical_area["roi_changed_area_sqkm"] = round(changed_m2 / 1_000_000.0, 6)
                    physical_area["reason"] = None
                    physical_area["metadata_source"] = "verified_image_metadata"
            except (ValueError, TypeError):
                pass

    # 5. Connected Component Hotspots
    hotspot_results = _compute_roi_hotspots(
        roi_mask=roi_slice,
        offset_x=x1,
        offset_y=y1,
        min_hotspot_size_px=min_hotspot_size_px,
    )

    return {
        "roi": {
            "pixel_coordinates": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": roi_width,
                "height": roi_height,
            },
            "normalized_coordinates": {
                "x1": round(x1 / width, 4),
                "y1": round(y1 / height, 4),
                "x2": round(x2 / width, 4),
                "y2": round(y2 / height, 4),
            },
            "image_dimensions": {
                "width": width,
                "height": height,
            },
        },
        "statistics": {
            "total_pixels": roi_total_pixels,
            "changed_pixels": roi_changed_pixels,
            "unchanged_pixels": roi_unchanged_pixels,
            "changed_percentage": roi_changed_pct,
            "unchanged_percentage": roi_unchanged_pct,
        },
        "physical_area": physical_area,
        "global_comparison": comparison,
        "hotspots": hotspot_results,
        "vqa": None,
    }


def execute_roi_change_vqa(
    analysis_id: str,
    t1_image_path: Union[str, Path],
    t2_image_path: Union[str, Path],
    roi_bounds_px: Tuple[int, int, int, int],
    change_mask_path: Optional[Union[str, Path]] = None,
    roi_stats: Optional[Dict[str, Any]] = None,
    query: Optional[str] = None,
    preferred_provider: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute Change VQA on the selected Region of Interest.

    Crops T1 and T2 images to ROI bounding box and passes them to the EXISTING
    run_change_vqa pipeline without modifying the global VLM architecture or prompt.

    Confidence strictly remains None (uncalibrated).
    """
    from .change_detection import _ensure_results_dir
    from .change_vqa import run_change_vqa

    t1_p = Path(t1_image_path)
    t2_p = Path(t2_image_path)
    if not t1_p.exists() or not t2_p.exists():
        raise FileNotFoundError(f"T1 or T2 image missing: {t1_p}, {t2_p}")

    x1, y1, x2, y2 = roi_bounds_px
    results_dir = _ensure_results_dir()

    # Crop T1 and T2
    with Image.open(t1_p) as img_a:
        roi_t1 = img_a.crop((x1, y1, x2, y2))
    with Image.open(t2_p) as img_b:
        roi_t2 = img_b.crop((x1, y1, x2, y2))

    # Save cropped images
    roi_t1_path = results_dir / f"roi_t1_{analysis_id}.png"
    roi_t2_path = results_dir / f"roi_t2_{analysis_id}.png"
    roi_t1.save(roi_t1_path, format="PNG")
    roi_t2.save(roi_t2_path, format="PNG")

    # Crop mask if available
    roi_mask_img = None
    if change_mask_path and Path(change_mask_path).exists():
        with Image.open(change_mask_path) as mask_img:
            roi_mask_img = mask_img.crop((x1, y1, x2, y2))

    # Prepare query
    vqa_query = query or "Describe the physical changes observed within this selected region of interest."

    # Invoke existing change VQA pipeline
    res = run_change_vqa(
        img_a_path=roi_t1_path,
        img_b_path=roi_t2_path,
        query=vqa_query,
        change_mask=roi_mask_img,
        change_stats=roi_stats,
        analysis_id=f"roi_{analysis_id}",
        preferred_provider=preferred_provider,
    )

    return {
        "answer": res.answer,
        "confidence": None,  # Strictly None — confidence is uncalibrated per rules
        "confidence_label": "N/A — Uncalibrated",
        "evidence": res.evidence,
        "composite_url": res.composite_url,
        "is_mock": res.is_mock,
        "stats": res.stats,
    }
