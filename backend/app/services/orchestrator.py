import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..schemas import AnalysisMode, TaskType, ToolInvocation
from .task_classifier import classify_task
from .tool_registry import get_tool, TOOL_REGISTRY
from .mock_specialists import run_tool as run_mock_tool
from ..logging_setup import logger


TASK_TO_PREFERRED_TOOL: Dict[TaskType, str] = {
    "vqa": "rs_vqa",
    "captioning": "rs_caption",
    "grounding": "rs_grounding",
    "change_detection": "change_detector",
    "change_vqa": "change_vqa",
    "change_description": "change_vqa",
}


REAL_VQA_TOOL_ID = "rs_vqa"


def select_tools_for_tasks(
    tasks: List[TaskType],
    mode: AnalysisMode,
) -> List[str]:
    """Deterministic tool selection (unchanged from Phase 1)."""
    selected: List[str] = []

    if mode == "optical_sar":
        if "optical_sar_analyzer" not in selected:
            selected.append("optical_sar_analyzer")
        for t in tasks:
            if t in ("grounding",):
                if "spatial_analyzer" not in selected:
                    selected.append("spatial_analyzer")
        for t in tasks:
            preferred = TASK_TO_PREFERRED_TOOL.get(t)
            if preferred and preferred not in selected and preferred != "optical_sar_analyzer":
                selected.append(preferred)
        return selected

    if mode == "bi_temporal":
        for t in tasks:
            preferred = TASK_TO_PREFERRED_TOOL.get(t)
            if preferred and preferred not in selected:
                selected.append(preferred)
        if "grounding" in tasks and "spatial_analyzer" not in selected:
            selected.append("spatial_analyzer")
        return selected

    for t in tasks:
        preferred = TASK_TO_PREFERRED_TOOL.get(t)
        if preferred and preferred not in selected:
            selected.append(preferred)
    if "grounding" in tasks and "spatial_analyzer" not in selected:
        selected.append("spatial_analyzer")

    return selected


def plan_execution(
    query: str,
    mode: AnalysisMode,
) -> Tuple[List[TaskType], List[str], Dict[str, Any], Dict[str, float]]:
    """Planning step: classify tasks, select tools, assemble default parameters."""
    tasks, scores = classify_task(query, mode)
    tool_ids = select_tools_for_tasks(tasks, mode)

    per_tool_params: Dict[str, Any] = {}
    for tid in tool_ids:
        meta = get_tool(tid)
        if tid == "rs_vqa":
            per_tool_params[tid] = {"temperature": 0.3, "max_tokens": 512, "beam_size": 4}
        elif tid == "rs_caption":
            per_tool_params[tid] = {"temperature": 0.4, "max_tokens": 256}
        elif tid == "rs_grounding":
            per_tool_params[tid] = {"confidence_threshold": 0.7, "nms_threshold": 0.45, "tile_size": 512}
        elif tid == "change_detector":
            per_tool_params[tid] = {"algorithm": "CVA+NDVI", "threshold": 0.15, "co_registration": True}
        elif tid == "change_vqa":
            per_tool_params[tid] = {"temperature": 0.2, "max_tokens": 768}
        elif tid == "optical_sar_analyzer":
            per_tool_params[tid] = {"polarisation": "VV", "fusion_method": "weighted_stack", "alignment": "phase_correlation"}
        elif tid == "spatial_analyzer":
            per_tool_params[tid] = {"compute_areas": True, "proximity_analysis": True}
        else:
            per_tool_params[tid] = {}

    logger.info(f"Plan: tasks={tasks} tools={tool_ids}")
    return tasks, tool_ids, per_tool_params, scores


def _mock_vqa_factory(query: str, mode: AnalysisMode):
    """Return a zero-arg closure that produces a mock VQA result."""
    def _factory():
        from .vqa_service import VQAServiceResult
        raw = run_mock_tool("rs_vqa", query, mode)
        return VQAServiceResult(
            answer=raw["answer"],
            confidence=raw.get("confidence"),
            evidence=raw.get("evidence", []),
            tool_id="rs_vqa",
            is_mock=True,
        )
    return _factory


def execute_plan(
    query: str,
    mode: AnalysisMode,
    tool_ids: List[str],
    per_tool_params: Dict[str, Any],
    tasks: Optional[List[TaskType]] = None,
    image_file_paths: Optional[List[Path]] = None,
) -> Tuple[str, Optional[float], List[ToolInvocation], List[Any], List[str], Any, Dict[str, str]]:
    """Run the planned tools with Phase 2 routing:

      - single-image rs_vqa → routes through real VQAService if enabled
      - all other tools remain MOCK (Phase 2 scope boundary)
      - every ToolInvocation carries an explicit executionMode: "real" or "mock"

    New extra return value: tool_execution_modes dict for trace-level visibility.
    """
    image_file_paths = image_file_paths or []
    tasks = tasks or []

    invocations: List[ToolInvocation] = []
    all_boxes: List[Any] = []
    all_evidence: List[str] = []
    answer_parts: List[str] = []
    confidences: List[float] = []
    change_map_out: Any = None
    tool_execution_modes: Dict[str, str] = {}

    from .vqa_service import get_vqa_service
    vqa_service = get_vqa_service()

    for tid in tool_ids:
        t_start = time.perf_counter()
        execution_mode: str = "mock"
        try:
            if tid == REAL_VQA_TOOL_ID and vqa_service.should_use_real_vqa(mode, tasks):
                mock_factory = _mock_vqa_factory(query, mode)
                try:
                    vqa_result = vqa_service.run_real_or_fallback(
                        query=query,
                        mode=mode,
                        image_file_paths=image_file_paths,
                        tasks=tasks,
                        mock_factory=mock_factory,
                    )
                    if vqa_result.run_context and vqa_result.run_context.execution_mode == "real":
                        execution_mode = "real"
                    tool_result = {
                        "answer": vqa_result.answer,
                        "confidence": vqa_result.confidence,
                        "evidence": vqa_result.evidence,
                        "tool_id": vqa_result.tool_id,
                        "is_mock": vqa_result.is_mock,
                        "bounding_boxes": vqa_result.bounding_boxes,
                    }
                    tool_execution_modes[tid] = execution_mode
                except Exception as e:
                    logger.exception(f"Real VQA tool {tid} failed; mock fallback was exhausted")
                    tool_result = {
                        "answer": f"[REAL VQA ERROR] Tool {tid} raised: {e}",
                        "confidence": None,
                        "evidence": [f"Tool {tid} failed during real execution: {type(e).__name__}"],
                        "tool_id": tid,
                        "is_mock": False,
                    }
                    tool_execution_modes[tid] = "real"
            else:
                tool_result = run_mock_tool(tid, query, mode)
                execution_mode = "mock"
                tool_execution_modes[tid] = "mock"
        except Exception as e:
            logger.exception(f"Tool {tid} failed")
            tool_result = {
                "answer": f"[ERROR] Tool {tid} raised: {e}",
                "confidence": None,
                "evidence": [f"Tool {tid} failed during execution: {type(e).__name__}"],
                "tool_id": tid,
                "is_mock": False,
            }
            tool_execution_modes[tid] = tool_execution_modes.get(tid, execution_mode)

        elapsed_ms = int((time.perf_counter() - t_start) * 1000) + (0 if execution_mode == "real" else 120)

        tool_meta = get_tool(tid)
        invocations.append(ToolInvocation(
            toolId=tid,
            toolName=tool_meta["name"],
            version=tool_meta["version"],
            taskType=next(iter(tool_meta["task_types"]), "vqa"),
            parameters=per_tool_params.get(tid, {}),
            processingTimeMs=elapsed_ms,
            executionMode=execution_mode,
        ))

        if "answer" in tool_result and tool_result["answer"]:
            answer_parts.append(tool_result["answer"])
        if "confidence" in tool_result and tool_result["confidence"] is not None:
            try:
                confidences.append(float(tool_result["confidence"]))
            except (TypeError, ValueError):
                pass
        if "bounding_boxes" in tool_result and tool_result["bounding_boxes"]:
            all_boxes.extend(tool_result["bounding_boxes"])
        if "evidence" in tool_result:
            all_evidence.extend(tool_result["evidence"])
        if "change_map" in tool_result and tool_result["change_map"] and change_map_out is None:
            change_map_out = tool_result["change_map"]

    merged_answer = "\n\n".join(answer_parts) if answer_parts else "[No tool produced an answer.]"
    agg_conf: Optional[float] = None
    if confidences:
        agg_conf = round(sum(confidences) / len(confidences), 3)
    return (
        merged_answer,
        agg_conf,
        invocations,
        all_boxes,
        all_evidence,
        change_map_out,
        tool_execution_modes,
    )
