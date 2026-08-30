from typing import Any, Dict, List

from ..schemas import TaskType, Modality, ToolDefinition


TOOL_REGISTRY: List[Dict[str, Any]] = [
    {
        "id": "orchestrator-v1",
        "name": "SatQuery Orchestrator",
        "task_types": ["vqa", "captioning", "grounding", "change_detection", "change_vqa", "change_description"],
        "supported_modalities": ["optical", "sar", "multispectral", "unknown"],
        "status": "available",
        "version": "1.0.0-phase1",
        "description": "Central routing and orchestration agent. Classifies the incoming query into task types, selects appropriate specialist models, coordinates parallel/sequential execution, and aggregates results into a unified response.",
    },
    {
        "id": "rs_vqa",
        "name": "RS-VQA Vision-Language Model",
        "task_types": ["vqa", "captioning"],
        "supported_modalities": ["optical", "multispectral"],
        "status": "available",
        "version": "0.2.0-phase2",
        "description": "Remote-sensing visual question answering and image captioning model. (Phase 2: Real VLM inference via a CPU-runnable SmolVLM-500M adapter. Falls back to mock on unsupported hardware/config.)",
    },
    {
        "id": "rs_caption",
        "name": "RS Captioning Specialist",
        "task_types": ["captioning"],
        "supported_modalities": ["optical", "multispectral", "sar"],
        "status": "mock",
        "version": "0.1.0-mock",
        "description": "Generates natural-language captions for remote-sensing imagery. (Phase 1: Mock implementation.)",
    },
    {
        "id": "rs_grounding",
        "name": "RS Grounding Detector",
        "task_types": ["grounding"],
        "supported_modalities": ["optical", "multispectral"],
        "status": "mock",
        "version": "0.1.0-mock",
        "description": "Open-vocabulary object detection and grounding for remote-sensing imagery. (Phase 1: Mock bounding boxes.)",
    },
    {
        "id": "change_detector",
        "name": "Bi-temporal Change Detector",
        "task_types": ["change_detection", "change_description"],
        "supported_modalities": ["optical", "multispectral", "sar"],
        "status": "mock",
        "version": "0.1.0-mock",
        "description": "Change detection for bi-temporal image pairs. Produces per-pixel change masks with semantic labels. (Phase 1: Mock change analysis.)",
    },
    {
        "id": "change_vqa",
        "name": "Change-VQA Language Model",
        "task_types": ["change_vqa", "change_description"],
        "supported_modalities": ["optical", "multispectral"],
        "status": "mock",
        "version": "0.1.0-mock",
        "description": "Answers natural-language questions about detected changes and generates human-readable descriptions. (Phase 1: Mock.)",
    },
    {
        "id": "optical_sar_analyzer",
        "name": "Optical + SAR Cross-Modal Analyzer",
        "task_types": ["vqa", "change_detection", "captioning"],
        "supported_modalities": ["sar", "optical", "multispectral"],
        "status": "mock",
        "version": "0.1.0-mock",
        "description": "Multi-modal fusion engine for Optical + SAR image pairs. Performs co-registration, feature fusion, and cross-modal analysis. (Phase 1: Mock.)",
    },
    {
        "id": "spatial_analyzer",
        "name": "Spatial Analyzer",
        "task_types": ["vqa", "grounding", "change_detection"],
        "supported_modalities": ["optical", "sar", "multispectral", "unknown"],
        "status": "mock",
        "version": "0.1.0-mock",
        "description": "Performs spatial analytics: area/perimeter calculations, proximity analysis, zonal statistics from detected bounding boxes or change masks. (Phase 1: Mock.)",
    },
]


def list_tools() -> List[ToolDefinition]:
    return [ToolDefinition(
        id=t["id"],
        name=t["name"],
        taskTypes=t["task_types"],
        supportedModalities=t["supported_modalities"],
        status=t["status"],
        version=t["version"],
        description=t["description"],
    ) for t in TOOL_REGISTRY]


def get_tool(tool_id: str) -> Dict[str, Any]:
    for t in TOOL_REGISTRY:
        if t["id"] == tool_id:
            return t
    raise KeyError(f"Tool {tool_id} not found in registry")
