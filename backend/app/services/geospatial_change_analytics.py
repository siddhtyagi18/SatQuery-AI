"""
backend/app/services/geospatial_change_analytics.py
---------------------------------------------------
Additive Geo-Spatial Change Analytics service for SatQuery-AI.

Consumes the EXISTING binary change mask produced by the Change Detection
pipeline (Siamese U-Net or classical CPU fallback).

Rules & Guarantees:
1. NEVER runs or alters the Change Detection model.
2. NEVER regenerates or mutates the existing change mask.
3. NEVER fabricates or assumes spatial metadata (e.g. no 10m Sentinel assumptions).
4. Physical area is calculated ONLY when reliable spatial resolution metadata is present.
5. Connected components / hotspots are strictly geometric and contain NO semantic labels.
6. Handles empty / all-zero change masks without throwing errors.
7. NEVER outputs or generates model confidence.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import ndimage

from ..logging_setup import logger


def _extract_epsg_code(crs_str: Optional[str]) -> Optional[int]:
    """Extract numeric EPSG code from a CRS string like 'EPSG:32643' or 'urn:ogc:def:crs:EPSG::32643'."""
    if not crs_str:
        return None
    match = re.search(r"EPSG[:\s]*(\d+)", crs_str, re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def _calculate_quadrant_density(
    binary_mask: np.ndarray, H: int, W: int
) -> Dict[str, float]:
    """
    Calculate spatial change density across 4 quadrants: NW, NE, SW, SE.
    Returns density percentage (0.0 - 100.0) for each quadrant.
    """
    if H == 0 or W == 0:
        return {
            "northwest": 0.0,
            "northeast": 0.0,
            "southwest": 0.0,
            "southeast": 0.0,
        }

    mid_y = H // 2
    mid_x = W // 2

    q_nw = binary_mask[:mid_y, :mid_x]
    q_ne = binary_mask[:mid_y, mid_x:]
    q_sw = binary_mask[mid_y:, :mid_x]
    q_se = binary_mask[mid_y:, mid_x:]

    def _pct(arr: np.ndarray) -> float:
        total = arr.size
        return float(arr.sum() / total * 100.0) if total > 0 else 0.0

    return {
        "northwest": round(_pct(q_nw), 2),
        "northeast": round(_pct(q_ne), 2),
        "southwest": round(_pct(q_sw), 2),
        "southeast": round(_pct(q_se), 2),
    }


def _calculate_grid_density(
    binary_mask: np.ndarray, H: int, W: int, grid_rows: int = 4, grid_cols: int = 4
) -> List[List[float]]:
    """Calculate a coarse spatial change density grid for spatial heatmap representation."""
    if H == 0 or W == 0:
        return [[0.0] * grid_cols for _ in range(grid_rows)]

    y_edges = np.linspace(0, H, grid_rows + 1, dtype=int)
    x_edges = np.linspace(0, W, grid_cols + 1, dtype=int)

    grid: List[List[float]] = []
    for r in range(grid_rows):
        row_vals: List[float] = []
        y0, y1 = y_edges[r], y_edges[r + 1]
        for c in range(grid_cols):
            x0, x1 = x_edges[c], x_edges[c + 1]
            cell = binary_mask[y0:y1, x0:x1]
            cell_pct = float(cell.sum() / cell.size * 100.0) if cell.size > 0 else 0.0
            row_vals.append(round(cell_pct, 2))
        grid.append(row_vals)
    return grid


def compute_geospatial_change_analytics(
    binary_mask: np.ndarray,
    dimensions: Tuple[int, int],
    metadata: Optional[Dict[str, Any]] = None,
    min_hotspot_size_px: int = 10,
) -> Dict[str, Any]:
    """
    Compute additive Geo-Spatial Change Analytics from an existing binary change mask.

    Parameters
    ----------
    binary_mask : np.ndarray
        2D array of shape (H, W) where 1 indicates changed pixels, 0 unchanged.
    dimensions : Tuple[int, int]
        (width, height) in pixels.
    metadata : Optional[Dict[str, Any]]
        Reliable geospatial metadata extracted from source image (e.g. CRS, GSD in meters).
        None if imagery lacks spatial metadata.
    min_hotspot_size_px : int
        Minimum pixel count threshold to filter tiny noise hotspots in reporting.
        Does NOT alter the original mask.

    Returns
    -------
    Dict[str, Any]
        Structured analytics containing global statistics, physical area (if available),
        hotspots list, largest change region, and change density.
    """
    metadata = metadata or {}
    W, H = dimensions

    # Guard against dimension mismatch or non-2D input
    mask_2d = np.asarray(binary_mask)
    if mask_2d.ndim != 2:
        raise ValueError(f"binary_mask must be 2D, got shape {mask_2d.shape}")

    mask_h, mask_w = mask_2d.shape
    if (mask_w, mask_h) != (W, H):
        # Use actual mask dimensions if mismatch occurs
        W, H = mask_w, mask_h

    # Ensure binary uint8 [0, 1] without mutating the original caller's array
    clean_mask = (mask_2d > 0).astype(np.uint8)

    # -------------------------------------------------------------------------
    # 1. Global Change Statistics
    # -------------------------------------------------------------------------
    total_pixel_count = int(H * W)
    changed_pixel_count = int(clean_mask.sum())
    unchanged_pixel_count = total_pixel_count - changed_pixel_count

    changed_percentage = (
        (changed_pixel_count / total_pixel_count * 100.0) if total_pixel_count > 0 else 0.0
    )
    unchanged_percentage = (
        (unchanged_pixel_count / total_pixel_count * 100.0) if total_pixel_count > 0 else 0.0
    )

    global_stats = {
        "width": W,
        "height": H,
        "total_pixel_count": total_pixel_count,
        "changed_pixel_count": changed_pixel_count,
        "unchanged_pixel_count": unchanged_pixel_count,
        "changed_percentage": changed_percentage,
        "unchanged_percentage": unchanged_percentage,
        "changed_percentage_rounded": round(changed_percentage, 2),
        "unchanged_percentage_rounded": round(unchanged_percentage, 2),
    }

    # -------------------------------------------------------------------------
    # 2. Geo-Spatial Context & Physical Area
    # -------------------------------------------------------------------------
    # Check for genuine spatial resolution metadata
    gsd_meters: Optional[float] = None
    if "gsd_meters" in metadata and metadata["gsd_meters"] is not None:
        try:
            val = float(metadata["gsd_meters"])
            if val > 0:
                gsd_meters = val
        except (ValueError, TypeError):
            gsd_meters = None

    # Alternatively check pixel_resolution (rx, ry)
    pixel_res: Optional[Tuple[float, float]] = None
    if "pixel_resolution" in metadata and metadata["pixel_resolution"]:
        res_val = metadata["pixel_resolution"]
        if isinstance(res_val, (list, tuple)) and len(res_val) >= 2:
            try:
                rx, ry = float(abs(res_val[0])), float(abs(res_val[1]))
                if rx > 0 and ry > 0:
                    pixel_res = (rx, ry)
            except (ValueError, TypeError):
                pixel_res = None

    crs_str: Optional[str] = metadata.get("crs")
    epsg_code: Optional[int] = _extract_epsg_code(crs_str)
    image_bounds: Optional[Any] = metadata.get("bounds")
    transform: Optional[Any] = metadata.get("transform")

    # Determine effective pixel area if reliable resolution exists
    pixel_area_m2: Optional[float] = None
    if gsd_meters is not None:
        pixel_area_m2 = float(gsd_meters * gsd_meters)
    elif pixel_res is not None:
        pixel_area_m2 = float(pixel_res[0] * pixel_res[1])

    if pixel_area_m2 is not None and pixel_area_m2 > 0:
        changed_area_m2 = float(changed_pixel_count * pixel_area_m2)
        total_area_m2 = float(total_pixel_count * pixel_area_m2)
        changed_area_ha = float(changed_area_m2 / 10000.0)
        changed_area_km2 = float(changed_area_m2 / 1000000.0)

        physical_area_analytics = {
            "physical_area_available": True,
            "area_unavailable_reason": None,
            "pixel_area_m2": pixel_area_m2,
            "gsd_meters": gsd_meters if gsd_meters is not None else (pixel_res[0] if pixel_res else None),
            "changed_area_m2": changed_area_m2,
            "changed_area_ha": round(changed_area_ha, 4),
            "changed_area_km2": round(changed_area_km2, 6),
            "total_area_m2": total_area_m2,
            "formatted_summary": (
                f"{changed_area_ha:,.2f} ha ({changed_area_km2:,.4f} km²)"
                if changed_area_ha >= 1.0
                else f"{changed_area_m2:,.1f} m²"
            ),
        }
    else:
        physical_area_analytics = {
            "physical_area_available": False,
            "area_unavailable_reason": "Physical area unavailable because reliable spatial resolution metadata was not provided.",
            "pixel_area_m2": None,
            "gsd_meters": None,
            "changed_area_m2": None,
            "changed_area_ha": None,
            "changed_area_km2": None,
            "total_area_m2": None,
            "formatted_summary": "N/A — Spatial metadata unavailable",
        }

    geospatial_metadata = {
        "crs": crs_str,
        "epsg_code": epsg_code,
        "resolution": list(pixel_res) if pixel_res else ([gsd_meters, gsd_meters] if gsd_meters else None),
        "image_bounds": image_bounds,
        "transform": transform,
        "width": W,
        "height": H,
    }

    # -------------------------------------------------------------------------
    # 3. Change Hotspots (Connected Components)
    # -------------------------------------------------------------------------
    # 8-connectivity structure: connects diagonally adjacent pixels
    structure_8 = np.ones((3, 3), dtype=int)
    labeled_mask, num_features = ndimage.label(clean_mask, structure=structure_8)

    hotspots: List[Dict[str, Any]] = []
    if num_features > 0 and changed_pixel_count > 0:
        # Find slices for each connected component
        slices = ndimage.find_objects(labeled_mask)

        for i, slc in enumerate(slices, start=1):
            if slc is None:
                continue
            y_slice, x_slice = slc
            component_pixels = (labeled_mask[y_slice, x_slice] == i)
            pixel_area = int(component_pixels.sum())

            if pixel_area == 0:
                continue

            # Pixel bounding box
            x_min = int(x_slice.start)
            y_min = int(y_slice.start)
            x_max = int(x_slice.stop)  # exclusive
            y_max = int(y_slice.stop)  # exclusive
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min

            # Exact centroid in pixel space
            y_indices, x_indices = np.where(component_pixels)
            centroid_x = float(x_min + np.mean(x_indices))
            centroid_y = float(y_min + np.mean(y_indices))

            pct_of_total_changed = (
                (pixel_area / changed_pixel_count * 100.0) if changed_pixel_count > 0 else 0.0
            )

            hotspot_item: Dict[str, Any] = {
                "hotspot_id": i,
                "pixel_area": pixel_area,
                "percentage_of_total_changed": pct_of_total_changed,
                "percentage_of_total_changed_rounded": round(pct_of_total_changed, 2),
                "bounding_box": {
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "width": bbox_width,
                    "height": bbox_height,
                },
                "centroid_px": {
                    "x": round(centroid_x, 2),
                    "y": round(centroid_y, 2),
                },
            }

            # If physical area is available, calculate physical hotspot area
            if pixel_area_m2 is not None:
                hotspot_area_m2 = pixel_area * pixel_area_m2
                hotspot_item["area_m2"] = round(hotspot_area_m2, 2)
                hotspot_item["area_ha"] = round(hotspot_area_m2 / 10000.0, 4)

            hotspots.append(hotspot_item)

        # Sort hotspots descending by pixel area
        hotspots.sort(key=lambda h: h["pixel_area"], reverse=True)

        # Re-index rank IDs 1..N based on size order for clean reporting
        for rank, h in enumerate(hotspots, start=1):
            h["rank"] = rank

    # Filtered hotspots according to min_hotspot_size_px (without mutating mask)
    filtered_hotspots = [h for h in hotspots if h["pixel_area"] >= min_hotspot_size_px]

    # -------------------------------------------------------------------------
    # 4. Largest Change Region
    # -------------------------------------------------------------------------
    largest_region: Optional[Dict[str, Any]] = None
    if hotspots:
        largest = hotspots[0]  # Already sorted descending
        largest_region = {
            "hotspot_id": largest["hotspot_id"],
            "rank": largest.get("rank", 1),
            "pixel_area": largest["pixel_area"],
            "percentage_of_total_changed": largest["percentage_of_total_changed"],
            "percentage_of_total_changed_rounded": largest["percentage_of_total_changed_rounded"],
            "bounding_box": largest["bounding_box"],
            "centroid_px": largest["centroid_px"],
            "area_m2": largest.get("area_m2"),
            "area_ha": largest.get("area_ha"),
        }

    # -------------------------------------------------------------------------
    # 5. Spatial Change Density
    # -------------------------------------------------------------------------
    quadrant_density = _calculate_quadrant_density(clean_mask, H, W)
    grid_density = _calculate_grid_density(clean_mask, H, W, grid_rows=4, grid_cols=4)

    # Determine peak density quadrant
    peak_quadrant = max(quadrant_density.items(), key=lambda kv: kv[1])[0] if quadrant_density else "none"

    change_density_analytics = {
        "quadrant_density": quadrant_density,
        "grid_density_4x4": grid_density,
        "peak_density_quadrant": peak_quadrant,
        "mean_change_density_pct": round(changed_percentage, 2),
    }

    # -------------------------------------------------------------------------
    # Assemble Final Geo-Spatial Change Analytics Result
    # -------------------------------------------------------------------------
    return {
        "global_statistics": global_stats,
        "physical_area": physical_area_analytics,
        "geospatial_metadata": geospatial_metadata,
        "hotspots_count_total": len(hotspots),
        "hotspots_count_filtered": len(filtered_hotspots),
        "min_hotspot_size_applied": min_hotspot_size_px,
        "largest_change_region": largest_region,
        "hotspots": filtered_hotspots[:50],  # Return top 50 filtered hotspots for payload efficiency
        "change_density": change_density_analytics,
    }
