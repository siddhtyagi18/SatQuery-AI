import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..schemas import AnalysisMode, TaskType, ToolInvocation
from ..config import get_settings
from .task_classifier import classify_task
from .tool_registry import get_tool, TOOL_REGISTRY
from . import mock_specialists
from .model_inference import run_change_detection, get_inference_mode
from ..logging_setup import logger

settings = get_settings()


# Tool ID that routes to the real CPU change detection service
REAL_CHANGE_DETECTOR_ID = "change_detector"


TASK_TO_PREFERRED_TOOL: Dict[TaskType, str] = {
    "vqa": "rs_vqa",
    "captioning": "rs_caption",
    "grounding": "rs_grounding",
    "change_detection": "change_detector",
    "change_vqa": "change_vqa",
    "change_description": "change_vqa",
}


REAL_VQA_TOOL_ID = "rs_vqa"
REAL_CAPTION_TOOL_ID = "rs_caption"
REAL_CHANGE_VQA_TOOL_ID = "change_vqa"
REAL_OPTICAL_SAR_TOOL_ID = "optical_sar_analyzer"


def _mock_caption_factory(query: str, mode: AnalysisMode):
    """Return a zero-arg closure that produces a mock caption result."""
    def _factory():
        from .vqa_service import VQAServiceResult
        raw = mock_specialists.run_tool("rs_caption", query, mode)
        return VQAServiceResult(
            answer=raw["answer"],
            confidence=raw.get("confidence"),
            evidence=raw.get("evidence", []),
            tool_id="rs_caption",
            is_mock=True,
        )
    return _factory


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
            elif t not in ("vqa", "captioning", "change_detection"):
                preferred = TASK_TO_PREFERRED_TOOL.get(t)
                if preferred and preferred not in selected and preferred != "optical_sar_analyzer":
                    selected.append(preferred)
        return selected

    if mode == "bi_temporal":
        # In bi-temporal mode, only multi-temporal change specialists should be executed.
        # Single-image VQA and captioning tools must not be selected to avoid injecting single-image reports.
        for t in tasks:
            if t in ("change_detection", "change_vqa", "change_description"):
                preferred = TASK_TO_PREFERRED_TOOL.get(t)
                if preferred and preferred not in selected:
                    selected.append(preferred)
        if "grounding" in tasks and "spatial_analyzer" not in selected:
            selected.append("spatial_analyzer")
        if not selected:
            selected = ["change_detector", "change_vqa"]
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
            per_tool_params[tid] = {
                "temperature": 0.3,
                "max_tokens": 512,
                "beam_size": 4,
                "provider": getattr(settings, "AI_PROVIDER", "auto"),
            }
        elif tid == "rs_caption":
            per_tool_params[tid] = {
                "temperature": 0.4,
                "max_tokens": 256,
                "provider": getattr(settings, "AI_PROVIDER", "auto"),
            }
        elif tid == "rs_grounding":
            per_tool_params[tid] = {"confidence_threshold": 0.7, "nms_threshold": 0.45, "tile_size": 512}
        elif tid == "change_detector":
            # Reflect the actual dispatch path so the trace shows the right algorithm
            actual_mode = get_inference_mode()
            if actual_mode == "model_checkpoint":
                per_tool_params[tid] = {
                    "algorithm": "siamese-unet-model",
                    "tile_size": 256,
                    "tile_overlap": 32,
                    "threshold": getattr(settings, "CHANGE_DETECTION_THRESHOLD", 0.70),
                }
            else:
                per_tool_params[tid] = {
                    "algorithm": "grayscale-abs-diff",
                    "threshold": 35,
                    "noise_cleanup": "3x3-morphological-opening",
                }
        elif tid == "change_vqa":
            per_tool_params[tid] = {
                "temperature": 0.2,
                "max_tokens": 768,
                "provider": getattr(settings, "AI_PROVIDER", "auto"),
            }
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
        raw = mock_specialists.run_tool("rs_vqa", query, mode)
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
    analysis_id: Optional[str] = None,
    compatibility_context: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[float], List[ToolInvocation], List[Any], List[str], Any, Dict[str, str], Dict[str, Any]]:
    """Run the planned tools with Phase 2 routing & Satellite Compatibility Guardrails:

      - single-image rs_vqa → routes through real VQAService if enabled
      - bi_temporal change_detector & change_vqa → routes through real specialists with domain checks
      - optical_sar_analyzer → routes through real Optical+SAR fusion
      - all other tools remain MOCK
      - every ToolInvocation carries an explicit executionMode: "real" or "mock"
      - unsupported or invalid inputs trigger transparent refusal without fabricated answers/confidence

    Returns: (merged_answer, agg_conf, invocations, all_boxes, all_evidence, change_map_out, tool_execution_modes, change_stats_out)
    """
    image_file_paths = image_file_paths or []
    tasks = tasks or []

    invocations: List[ToolInvocation] = []
    all_boxes: List[Any] = []
    all_evidence: List[str] = []
    answer_parts: List[str] = []
    confidences: List[float] = []
    change_map_out: Any = None
    change_stats_out: Dict[str, Any] = {}
    tool_execution_modes: Dict[str, str] = {}

    # Check compatibility guardrails
    if compatibility_context:
        c_status = compatibility_context.get("status")
        c_limits = compatibility_context.get("limitations", [])
        c_reasons = compatibility_context.get("reasons", [])
        c_warnings = compatibility_context.get("warnings", [])

        if c_status in ("invalid", "unsupported", "unsupported_for_reliable_inference", "needs_review"):
            refusal_reasons = c_reasons or [f"Input imagery is {c_status} for mode '{mode}'."]
            notice_type = "COMPATIBILITY NOTICE: NEEDS_REVIEW" if c_status == "needs_review" else f"CHANGE DETECTION NOTICE: {c_status.upper()}"
            headline = "Analysis halted for unidentified sensor modality:" if c_status == "needs_review" else "Change detection unavailable for this imagery:"
            refusal_text = (
                f"[{notice_type}]\n\n"
                f"{headline}\n"
                + "\n".join(f"- {r}" for r in refusal_reasons)
            )
            if c_limits:
                refusal_text += "\n\nValidated Model Domain & Limitations:\n" + "\n".join(f"- {l}" for l in c_limits)

            refusal_inv = ToolInvocation(
                toolId="satellite_compatibility_guardrail",
                toolName="Satellite Compatibility Guardrail",
                version="1.0.0",
                taskType="vqa",
                parameters={"status": c_status, "mode": mode},
                processingTimeMs=5,
                executionMode="real",
            )
            guard_evidence = [f"Compatibility status: {c_status}"] + refusal_reasons + c_limits
            return (
                refusal_text,
                None,
                [refusal_inv],
                [],
                guard_evidence,
                None,
                {"satellite_compatibility_guardrail": "real"},
                {"compatibility_status": c_status, "limitations": c_limits},
            )

        for w in c_warnings:
            all_evidence.append(f"Compatibility Warning: {w}")
        for l in c_limits:
            all_evidence.append(f"Model Limitation: {l}")

    from .vqa_service import get_vqa_service
    vqa_service = get_vqa_service()

    for tid in tool_ids:
        t_start = time.perf_counter()
        execution_mode: str = "mock"
        try:
            # ------------------------------------------------------------------
            # Real CPU change detection (bi_temporal mode, 2 image paths)
            # ------------------------------------------------------------------
            if (
                tid == REAL_CHANGE_DETECTOR_ID
                and mode == "bi_temporal"
                and len(image_file_paths) == 2
            ):
                try:
                    cd_params = per_tool_params.get(tid, {})
                    cd_threshold = cd_params.get("threshold")
                    cd_result = run_change_detection(
                        before_path=image_file_paths[0],
                        after_path=image_file_paths[1],
                        analysis_id=analysis_id or "unknown",
                        threshold=cd_threshold,
                    )
                    execution_mode = "real"
                    tool_result = {
                        "answer": cd_result.answer,
                        "confidence": cd_result.confidence,  # always None
                        "change_map": cd_result.change_map,
                        "evidence": cd_result.evidence,
                        "tool_id": tid,
                        "is_mock": False,
                    }
                    # Capture stats for step-6 meta injection
                    change_stats_out = cd_result.stats
                    tool_execution_modes[tid] = "real"
                    logger.info("[orchestrator] change_detector ran REAL CPU service")
                except Exception as cd_exc:
                    logger.exception("[orchestrator] Real change detection failed: %s", cd_exc)
                    tool_result = {
                        "answer": f"## Change Detection Error\n\nReal change detection model unavailable: {cd_exc}",
                        "confidence": None,
                        "change_map": None,
                        "evidence": [f"Real change detection model unavailable: {cd_exc}"],
                        "tool_id": tid,
                        "is_mock": False,
                    }
                    execution_mode = "real"
                    tool_execution_modes[tid] = "real"

            # ------------------------------------------------------------------
            # Real VQA / Captioning
            # ------------------------------------------------------------------
            elif tid in (REAL_VQA_TOOL_ID, REAL_CAPTION_TOOL_ID) and vqa_service.should_use_real_vqa(mode, tasks):
                mock_factory = _mock_caption_factory(query, mode) if tid == REAL_CAPTION_TOOL_ID else _mock_vqa_factory(query, mode)
                pref_provider = per_tool_params.get(tid, {}).get("provider")
                try:
                    vqa_result = vqa_service.run_real_or_fallback(
                        query=query,
                        mode=mode,
                        image_file_paths=image_file_paths,
                        tasks=tasks,
                        mock_factory=mock_factory,
                        tool_id=tid,
                        preferred_provider=pref_provider,
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
                    logger.exception(f"Real Vision-Language tool {tid} failed; mock fallback was exhausted")
                    tool_result = {
                        "answer": f"[REAL VLM ERROR] Tool {tid} raised: {e}",
                        "confidence": None,
                        "evidence": [f"Tool {tid} failed during real execution: {type(e).__name__}"],
                        "tool_id": tid,
                        "is_mock": False,
                    }
                    tool_execution_modes[tid] = "real"

            # ------------------------------------------------------------------
            # Real Optical + SAR Cross-Modal Analysis (2 image paths)
            # ------------------------------------------------------------------
            elif (
                tid == REAL_OPTICAL_SAR_TOOL_ID
                and mode == "optical_sar"
                and len(image_file_paths) >= 2
            ):
                try:
                    from .optical_sar import run_optical_sar_analysis
                    opt_p = image_file_paths[0]
                    sar_vv_p = image_file_paths[1]
                    sar_vh_p = image_file_paths[2] if len(image_file_paths) >= 3 else None
                    os_result = run_optical_sar_analysis(
                        optical_path=opt_p,
                        sar_path=sar_vv_p,
                        query=query,
                        sar_vh_path=sar_vh_p,
                        analysis_id=analysis_id or "unknown",
                    )
                    execution_mode = "real"
                    tool_result = {
                        "answer": os_result.answer,
                        "confidence": os_result.confidence,  # always None
                        "evidence": os_result.evidence,
                        "tool_id": tid,
                        "is_mock": False,
                    }
                    tool_execution_modes[tid] = "real"
                    logger.info("[orchestrator] optical_sar_analyzer ran REAL cross-modal service")
                except Exception as os_exc:
                    logger.warning(
                        "[orchestrator] Real Optical-SAR analysis failed (%s: %s); falling back to mock.",
                        type(os_exc).__name__, os_exc,
                    )
                    tool_result = mock_specialists.run_tool(tid, query, mode)
                    execution_mode = "mock"
                    tool_execution_modes[tid] = "mock"

            # ------------------------------------------------------------------
            # Real Change VQA (bi_temporal mode, 2 image paths)
            # ------------------------------------------------------------------
            elif (
                tid == REAL_CHANGE_VQA_TOOL_ID
                and mode == "bi_temporal"
                and len(image_file_paths) == 2
            ):
                pref_provider = per_tool_params.get(tid, {}).get("provider")
                try:
                    from .change_vqa import run_change_vqa
                    cvqa_result = run_change_vqa(
                        img_a_path=image_file_paths[0],
                        img_b_path=image_file_paths[1],
                        query=query,
                        change_stats=change_stats_out,
                        analysis_id=analysis_id or "unknown",
                        preferred_provider=pref_provider,
                    )
                    execution_mode = "mock" if cvqa_result.is_mock else "real"
                    tool_result = {
                        "answer": cvqa_result.answer,
                        "confidence": cvqa_result.confidence,  # always None
                        "evidence": cvqa_result.evidence,
                        "tool_id": tid,
                        "is_mock": cvqa_result.is_mock,
                    }
                    tool_execution_modes[tid] = execution_mode
                    logger.info(f"[orchestrator] change_vqa executed (mode={execution_mode})")
                except Exception as cvqa_exc:
                    logger.exception("[orchestrator] Real Change VQA failed: %s", cvqa_exc)
                    tool_result = {
                        "answer": f"### Bi-Temporal Scene Change Interpretation\n\nReal Change-VQA reasoning unavailable: {cvqa_exc}",
                        "confidence": None,
                        "evidence": [f"Real Change-VQA reasoning unavailable: {cvqa_exc}"],
                        "tool_id": tid,
                        "is_mock": False,
                    }
                    execution_mode = "real"
                    tool_execution_modes[tid] = "real"

            # ------------------------------------------------------------------
            # All other tools — mock
            # ------------------------------------------------------------------
            else:
                # For change_vqa, pass real change stats context so the mock can
                # produce a contextually accurate summary instead of a blank template.
                if tid == "change_vqa" and change_stats_out:
                    tool_result = mock_specialists.run_tool(
                        tid, query, mode,
                        changed_pixel_pct=change_stats_out.get("changed_pixel_pct"),
                        severity=change_stats_out.get("severity"),
                        execution_mode=change_stats_out.get("execution_mode"),
                    )
                else:
                    tool_result = mock_specialists.run_tool(tid, query, mode)
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
            ans_str = tool_result["answer"].strip()
            if ans_str and ans_str not in answer_parts:
                answer_parts.append(ans_str)
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
    # Strict Scientific Confidence Policy:
    # Unless a genuinely calibrated confidence model has been scientifically validated,
    # overall confidence MUST remain None across all analysis modes (single_image, bi_temporal, optical_sar).
    agg_conf = None
    return (
        merged_answer,
        agg_conf,
        invocations,
        all_boxes,
        all_evidence,
        change_map_out,
        tool_execution_modes,
        change_stats_out,
    )
