from typing import Any, Dict, List

from ..schemas import TaskType, AnalysisMode, BoundingBox
from ..logging_setup import logger

MOCK_PREFIX = "[MOCK — Phase 1, not real inference] "


def _make_vqa_result(query: str, mode: AnalysisMode, **kwargs: Any) -> Dict[str, Any]:
    q = query.strip()
    answer = (
        f"{MOCK_PREFIX}Analysis for query: \"{q}\". "
        f"Mode: {mode}. The satellite imagery shows a mixed landscape. "
        "Dominant visible features include urban built-up structures, agricultural parcels, "
        "and linear road networks. Vegetation cover is discernible throughout the scene. "
        "Key observations are placeholder-level only — a real VQA model will supply quantified class percentages, "
        "spatial distribution, and dataset-backed claims."
    )
    return {
        "answer": answer,
        "confidence": 0.72,
        "evidence": [
            "VQA mock ran on provided image inputs (no real model executed).",
            "Confidence score is a fixed placeholder, not a calibrated prediction.",
        ],
    }


def _make_caption_result(query: str, mode: AnalysisMode, **kwargs: Any) -> Dict[str, Any]:
    answer = (
        f"{MOCK_PREFIX}Caption: High-resolution satellite view of a mixed urban-agricultural region. "
        "Visible features include clustered building footprints, field parcels with distinct crop boundaries, "
        "linear road infrastructure, and a vegetated margin along the southern edge. "
        "Caption will be re-generated via a real RS captioning model in Phase 2."
    )
    return {
        "answer": answer,
        "confidence": 0.68,
        "evidence": ["Caption generated from template — not from image pixels."],
    }


def _make_grounding_result(query: str, mode: AnalysisMode, **kwargs: Any) -> Dict[str, Any]:
    boxes: List[BoundingBox] = [
        BoundingBox(x=0.10, y=0.15, width=0.06, height=0.05, label="Building", confidence=0.85),
        BoundingBox(x=0.22, y=0.10, width=0.04, height=0.04, label="Building", confidence=0.80),
        BoundingBox(x=0.35, y=0.20, width=0.07, height=0.06, label="Building", confidence=0.77),
        BoundingBox(x=0.50, y=0.30, width=0.05, height=0.04, label="Building", confidence=0.74),
        BoundingBox(x=0.65, y=0.18, width=0.03, height=0.03, label="Building", confidence=0.70),
        BoundingBox(x=0.15, y=0.55, width=0.12, height=0.10, label="Vegetation Cluster", confidence=0.82),
        BoundingBox(x=0.70, y=0.60, width=0.08, height=0.07, label="Water Body", confidence=0.88),
    ]
    answer = (
        f"{MOCK_PREFIX}Grounding detected {len(boxes)} placeholder objects across the scene. "
        "Bounding boxes are fixed template annotations (not produced by a real detector) and "
        "should be treated as visual examples only."
    )
    return {
        "answer": answer,
        "confidence": 0.70,
        "bounding_boxes": [b.model_dump() for b in boxes],
        "evidence": [
            f"{len(boxes)} mock bounding boxes returned.",
            "All coordinates are fixed template values — no object detection was performed.",
        ],
    }


def _make_change_detection_result(query: str, mode: AnalysisMode, **kwargs: Any) -> Dict[str, Any]:
    change_map = {
        "overlayUrl": None,
        "legend": [
            {"label": "New Urban / Built-up", "color": "#FF5C5C"},
            {"label": "Vegetation Loss", "color": "#FFB020"},
            {"label": "New Infrastructure", "color": "#3ED0FF"},
            {"label": "No Change", "color": "#1A2340"},
        ],
    }
    answer = (
        f"{MOCK_PREFIX}Change detection (mock, no pixel-level CVA/ML computation performed). "
        "Placeholder analysis suggests an estimated ~20% change mask across the temporal pair, "
        "split across semantic classes (urban gain, vegetation loss, new infrastructure). "
        "Real per-pixel change statistics and class-specific IoU/F1 require a deployed model."
    )
    return {
        "answer": answer,
        "confidence": 0.65,
        "change_map": change_map,
        "evidence": [
            "Change map legend is a template; no overlay raster was produced.",
            "Percentages above are representative placeholders, not measured values.",
        ],
    }


def _make_change_vqa_result(query: str, mode: AnalysisMode, **context: Any) -> Dict[str, Any]:
    # Try to use real pixel stats if the orchestrator passes them
    changed_pct = context.get("changed_pixel_pct")
    severity = context.get("severity", "unknown")
    exec_mode = context.get("execution_mode", "unknown")

    if changed_pct is not None:
        # We have real pixel stats — use them for a contextual (but still mock) answer
        answer = (
            f"[Contextual summary — real pixel statistics used, natural language requires a Change-VQA model] "
            f"The bi-temporal analysis measured **{changed_pct:.1f}%** of the scene as changed "
            f"(severity: {severity}). "
            f"The image pair shows a {severity}-level change signature. "
            f"Change detection was performed by: {exec_mode}. "
            f"Your specific question (\"{query}\") requires a Vision-Language model trained on "
            f"change-detection tasks to answer precisely — e.g., identifying *what* changed "
            f"(buildings, vegetation, roads) or *how* it changed. That model is not yet deployed."
        )
    else:
        answer = (
            f"{MOCK_PREFIX}Change-VQA response (mock). Your question about detected changes: \"{query}\". "
            "No pixel statistics available from this run. "
            "A real Change-VQA model is required to answer specific natural-language questions "
            "about what changed between the two images."
        )
    return {
        "answer": answer,
        "confidence": None,  # No calibrated confidence without a real model
        "evidence": [
            "Change-VQA answer is descriptive only — no Vision-Language model was run.",
            "Pixel statistics (if shown) are from real change detection, not fabricated.",
        ],
    }


def _make_optical_sar_result(query: str, mode: AnalysisMode, **kwargs: Any) -> Dict[str, Any]:
    boxes: List[BoundingBox] = [
        BoundingBox(x=0.15, y=0.25, width=0.09, height=0.07, label="SAR-only: Sub-canopy Structure", confidence=0.72),
        BoundingBox(x=0.55, y=0.40, width=0.10, height=0.08, label="Flooded Parcel (SAR-confirmed)", confidence=0.80),
        BoundingBox(x=0.30, y=0.15, width=0.06, height=0.05, label="Urban Expansion (confirmed both modalities)", confidence=0.83),
    ]
    answer = (
        f"{MOCK_PREFIX}Optical + SAR cross-modal analysis (mock, no real fusion run). "
        "Placeholders indicate (a) optical detections broadly confirmed by SAR backscatter trends, "
        "and (b) a small number of SAR-unique features. Phase 2 will compute actual "
        "phase-correlation alignment and weighted stack fusion."
    )
    return {
        "answer": answer,
        "confidence": 0.74,
        "bounding_boxes": [b.model_dump() for b in boxes],
        "evidence": [
            "Cross-modal confidence is a fixed placeholder value.",
            "SAR-unique feature boxes are template annotations, not real detections.",
        ],
    }


def _make_spatial_analyzer_result(query: str, mode: AnalysisMode, **kwargs: Any) -> Dict[str, Any]:
    answer = (
        f"{MOCK_PREFIX}Spatial analysis (mock): bounding box union area, road-network proximity, "
        "and zonal statistics are illustrative placeholders. Real spatial calculations will be "
        "performed in Phase 2 using vectorised geometry (shapely) and raster zonal tools (rasterio / rioxarray)."
    )
    return {
        "answer": answer,
        "confidence": 0.60,
        "evidence": [
            "No actual distance/area computation was performed.",
        ],
    }


TOOL_RUNNERS = {
    "rs_vqa": _make_vqa_result,
    "rs_caption": _make_caption_result,
    "rs_grounding": _make_grounding_result,
    "change_detector": _make_change_detection_result,
    "change_vqa": _make_change_vqa_result,
    "optical_sar_analyzer": _make_optical_sar_result,
    "spatial_analyzer": _make_spatial_analyzer_result,
}


def run_tool(tool_id: str, query: str, mode: AnalysisMode, **kwargs: Any) -> Dict[str, Any]:
    """Run a mock specialist tool by id. Returns structured dict always marked as mock."""
    runner = TOOL_RUNNERS.get(tool_id)
    if runner is None:
        raise ValueError(f"Unknown tool id: {tool_id}")
    logger.info(f"Running mock tool: {tool_id} for mode={mode}")
    result = runner(query, mode, **kwargs)
    result["tool_id"] = tool_id
    result["is_mock"] = True
    return result
