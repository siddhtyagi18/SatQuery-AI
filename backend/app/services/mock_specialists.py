from typing import Any, Dict, List

from ..schemas import TaskType, AnalysisMode, BoundingBox
from ..logging_setup import logger

MOCK_PREFIX = "[MOCK] "


def _make_vqa_result(query: str, mode: AnalysisMode, **kwargs: Any) -> Dict[str, Any]:
    q = query.strip()
    q_lower = q.lower()
    if any(k in q_lower for k in ["land cover", "landcover", "types", "terrain"]):
        body = (
            f"Multispectral satellite observation for query: \"{q}\":\n\n"
            "### Quantified Land Cover Breakdown:\n"
            "- **Vegetation & Canopy Cover**: **42.8%** — Tree stands, agricultural parcels, and green buffer corridors (mean NDVI surrogate: 0.44).\n"
            "- **Urban Built-up & Infrastructure**: **34.2%** — Engineered structures, paved road alignments, and clustered residential/commercial footprints.\n"
            "- **Barren Soil & Transitional Ground**: **18.5%** — Exposed soil parcels, ploughed fields, and open transitional land.\n"
            "- **Water Features & Hydrological Depressions**: **4.5%** — Localized drainage channels and low-albedo surface water.\n\n"
            "### Spatial Observations:\n"
            "The scene displays a peri-urban mixed landscape. Structural density is highest along the primary transit corridor, with distinct spatial separation between built parcels and vegetated plots."
        )
    elif any(k in q_lower for k in ["building", "structure", "urban"]):
        body = (
            f"Structural analysis for query: \"{q}\":\n\n"
            "Built-up features occupy approximately **34.2%** of the scene area. "
            "Structural clustering reveals 14 distinct building complexes concentrated along linear road networks. "
            "Footprints exhibit high edge-gradient contrast and sharp geometrical boundaries consistent with active infrastructure."
        )
    else:
        body = (
            f"Remote sensing intelligence synthesis for query: \"{q}\":\n\n"
            "The satellite imagery reveals a balanced mixed landscape containing **34.2% built-up infrastructure**, "
            "**42.8% vegetative canopy and agricultural cover**, **18.5% open soil**, and **4.5% water/shadow features**. "
            "Spatial morphology indicates active urban-rural transitional activity with clearly delineated parcel margins."
        )

    return {
        "answer": f"{MOCK_PREFIX}{body}",
        "confidence": None,
        "evidence": [
            f"Multispectral feature synthesis evaluated user query in mock execution mode: '{q}'.",
            "Spectral indices computed across visible and near-infrared bands.",
            "Confidence score remains strictly uncalibrated (null) under mock test mode.",
        ],
    }


def _make_caption_result(query: str, mode: AnalysisMode, **kwargs: Any) -> Dict[str, Any]:
    answer = (
        "High-resolution satellite view of a mixed urban-agricultural landscape. "
        "Visible features include clustered building footprints, field parcels with distinct crop boundaries, "
        "linear road infrastructure, and a vegetated margin along the southern sector."
    )
    return {
        "answer": answer,
        "confidence": None,
        "evidence": [
            "High-resolution semantic caption generated for scene context.",
            "Confidence score remains strictly uncalibrated (null) under mock test mode.",
        ],
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
        "confidence": None,
        "bounding_boxes": [b.model_dump() for b in boxes],
        "evidence": [
            f"{len(boxes)} mock bounding boxes returned.",
            "All coordinates are fixed template values — no object detection was performed.",
            "Confidence score is uncalibrated / null.",
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
        "confidence": None,
        "change_map": change_map,
        "evidence": [
            "Change map legend is a template; no overlay raster was produced.",
            "Percentages above are representative placeholders, not measured values.",
            "Confidence score is uncalibrated / null.",
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
            "Confidence score is a fixed placeholder, not a calibrated prediction.",
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
        "confidence": None,
        "bounding_boxes": [b.model_dump() for b in boxes],
        "evidence": [
            "Optical+SAR cross-modal model emits uncalibrated features; confidence=null.",
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
        "confidence": None,
        "evidence": [
            "No actual distance/area computation was performed.",
            "Confidence score is uncalibrated / null.",
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
