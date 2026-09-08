from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..config import get_settings
from ..crud import analysis_to_result, list_analyses
from ..database import get_db
from ..logging_setup import logger
from ..models import Analysis, AnalysisImage, UploadedFile
from ..schemas import (
    AnalysisMode,
    AnalysisResult,
    AnalysisStatus,
    ExecutionTraceOut,
    FollowUpRequest,
    FollowUpResponse,
    HistoryFilters,
    HistoryPage,
    ImageRole,
    SubmitAnalysisInput,
)
from ..services.firebase import FirebaseRepository, is_firebase_enabled
from ..services.follow_up import answer_follow_up
from ..services.orchestrator import execute_plan, plan_execution
from ..services.result_validation import validate_analysis_result_payload
from ..services.trace import build_trace_out, create_pending_trace, mark_step

settings = get_settings()
router = APIRouter(prefix="/api/analysis", tags=["analysis"])


def _roles_for_mode(mode: AnalysisMode) -> List[ImageRole]:
    if mode == "single_image":
        return ["single"]
    if mode == "bi_temporal":
        return ["before", "after"]
    if mode == "optical_sar":
        return ["optical", "sar"]
    return ["single"]


def _validate_input(db: Session, input_data: SubmitAnalysisInput) -> List[UploadedFile]:
    expected_roles = _roles_for_mode(input_data.mode)
    if len(input_data.imageIds) != len(expected_roles):
        raise HTTPException(
            status_code=400,
            detail=f"Mode {input_data.mode} requires exactly {len(expected_roles)} imageIds ({', '.join(expected_roles)})",
        )
    files: List[UploadedFile] = []
    for fid in input_data.imageIds:
        f = db.query(UploadedFile).filter(UploadedFile.id == fid).first()
        if f is None:
            raise HTTPException(status_code=404, detail=f"Uploaded file {fid} not found")
        if not Path(f.file_path).exists():
            raise HTTPException(
                status_code=400,
                detail=f"Uploaded file {fid} is missing on disk ({f.file_path}). Re-upload the image.",
            )
        files.append(f)
    if not input_data.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")
    return files


def _link_images(
    db: Session,
    analysis_id: str,
    mode: AnalysisMode,
    image_ids: List[str],
) -> None:
    roles = _roles_for_mode(mode)
    for fid, role in zip(image_ids, roles):
        db.add(AnalysisImage(analysis_id=analysis_id, file_id=fid, role=role))
    db.flush()


def _persist_to_firebase(
    repo: FirebaseRepository,
    analysis: Analysis,
    db: Session,
) -> None:
    """Best-effort Firestore sync. Never raises; failures are logged only."""
    try:
        if not is_firebase_enabled():
            return
        payload = analysis_to_result(db, analysis).model_dump(mode="json")
        payload["analysis_id"] = analysis.id
        payload["_schema_version"] = 2
        repo.save_analysis(analysis.id, payload)
    except Exception as e:
        logger.warning(f"Firestore persistence for {analysis.id} failed (non-fatal): {e}")


def _run_analysis_pipeline(db: Session, analysis: Analysis, files, input_data: SubmitAnalysisInput) -> None:
    analysis.status = "processing"
    db.flush()

    aid = analysis.id
    q = input_data.query
    mode = input_data.mode

    firebase_repo = FirebaseRepository()

    # Step 1 — Query Received
    mark_step(db, aid, "step-1", "done",
              detail=f'Query accepted: "{q[:80]}{"…" if len(q) > 80 else ""}" | Mode: {mode} | Images: {len(files)}')

    # Step 2 — Input Validation & Satellite Compatibility Inspection
    mark_step(db, aid, "step-2", "in_progress")
    from ..services.satellite_compatibility import (
        SatelliteImageInspector,
        SatelliteCompatibilityService,
    )

    validation_bits = []
    modalities: List[str] = []
    image_file_paths: List[Path] = [Path(f.file_path) for f in files]

    # Run satellite image inspections
    inspections = [SatelliteImageInspector.inspect(p) for p in image_file_paths]

    for f, p, insp in zip(files, image_file_paths, inspections):
        # Harmonize modality: if inspector determined modality, use it; otherwise preserve metadata
        raw_mod = insp.modality_hint if (insp.modality_hint and insp.modality_hint != "unknown") else (f.modality or "unknown")
        if "sar" in raw_mod:
            norm_mod = "sar"
        elif "multispectral" in raw_mod:
            norm_mod = "multispectral"
        elif any(k in raw_mod for k in ("optical", "rgb", "grayscale")):
            norm_mod = "optical"
        else:
            norm_mod = "unknown"

        if f.modality in (None, "unknown") and norm_mod != "unknown":
            f.modality = norm_mod
        effective_modality = raw_mod
        bits = [
            f"{f.file_name}: {f.file_format or '?'}",
            f"Modality={effective_modality}"
            + (f" ({int((f.modality_confidence or 0) * 100)}%)" if f.modality_confidence else ""),
        ]
        if p.exists():
            bits.append(f"file_exists=✓ ({f.file_size_bytes} B)")
        else:
            bits.append("file_exists=✗")
        if f.crs:
            bits.append(f"CRS={f.crs}")
        if f.width_px and f.height_px:
            bits.append(f"{f.width_px}×{f.height_px}px")
        if f.band_count:
            bits.append(f"bands={f.band_count}")
        validation_bits.append(" | ".join(bits))
        modalities.append(effective_modality)

    comp_dict: Dict[str, Any] = {}
    limits_list: List[str] = []

    if len(inspections) == 1:
        s_report = SatelliteCompatibilityService.check_single_compatibility(inspections[0])
        comp_dict = s_report.to_dict()
        limits_list = []
        # Input validation logic fix: if modality cannot be determined, status must be needs_review
        if modalities[0] == "unknown" or s_report.modality == "unknown":
            if comp_dict.get("status") in ("compatible", "adaptable"):
                comp_dict["status"] = "needs_review"
                comp_dict.setdefault("warnings", []).append("Sensor modality could not be determined; results may be unreliable.")
                comp_dict.setdefault("reasons", []).append("Sensor modality is unknown; specialist inference requires review.")
                limits_list.append("Sensor modality could not be determined; results may be unreliable. The system will not silently execute specialist models on unverified or unidentified imagery.")
        validation_bits.append(f"Compatibility: {comp_dict['status'].upper()}")
        if comp_dict.get("warnings"):
            validation_bits.append(f"Warnings: {len(comp_dict['warnings'])}")
    elif len(inspections) >= 2:
        p_report = SatelliteCompatibilityService.check_pair_compatibility(
            inspections[0], inspections[1], mode=mode, query=q
        )
        comp_dict = p_report.to_dict()
        limits_list = list(p_report.limitations or [])
        if any(m == "unknown" for m in modalities[:2]) or any(insp.modality_hint == "unknown" for insp in inspections[:2]):
            if comp_dict.get("status") in ("compatible", "adaptable"):
                comp_dict["status"] = "needs_review"
                comp_dict.setdefault("warnings", []).append("Sensor modality could not be determined; results may be unreliable.")
                comp_dict.setdefault("reasons", []).append("Sensor modality is unknown for one or both images.")
                if "Sensor modality could not be determined; results may be unreliable." not in limits_list:
                    limits_list.append("Sensor modality could not be determined; results may be unreliable. The system will not silently execute specialist models on unverified or unidentified imagery.")
        validation_bits.append(f"Pair Compatibility: {comp_dict['status'].upper()}")
        validation_bits.append(f"Temporal: {p_report.temporal_status}")
        if p_report.spatial_overlap_pct is not None:
            validation_bits.append(f"Overlap: {p_report.spatial_overlap_pct}%")
        if comp_dict.get("warnings"):
            validation_bits.append(f"Warnings: {len(comp_dict['warnings'])}")

    validation_detail = " ; ".join(validation_bits)
    mark_step(
        db, aid, "step-2", "done",
        detail=validation_detail,
        meta={
            "compatibility_status": comp_dict.get("status", "unknown"),
            "required_adaptations": comp_dict.get("required_adaptations", []),
            "warnings_count": len(comp_dict.get("warnings", [])),
            "temporal_status": comp_dict.get("temporal_status", "unknown"),
        },
    )

    # Step 3 — Task Classification
    mark_step(db, aid, "step-3", "in_progress")
    tasks, tool_ids, per_tool_params, class_scores = plan_execution(q, mode)
    if getattr(input_data, "provider", None):
        for tid in ("rs_vqa", "rs_caption", "change_vqa"):
            if tid in per_tool_params:
                per_tool_params[tid]["provider"] = input_data.provider
    class_str = " | ".join(f"[{t}: {class_scores.get(t, 0):.2f}]" for t in tasks)
    mark_step(db, aid, "step-3", "done",
              detail=f"Detected intents: {class_str} | Tasks: {', '.join(tasks)}",
              meta={k: round(v, 3) for k, v in class_scores.items() if v > 0})

    # Step 4 — Tool Selection (with REAL/MOCK indicators for each tool)
    mark_step(db, aid, "step-4", "in_progress")
    from ..services.tool_registry import get_tool
    from ..services.vqa_service import get_vqa_service
    vqa_service = get_vqa_service()
    selection_bits = []
    for tid in tool_ids:
        tm = get_tool(tid)
        is_real_vqa = (tid in ("rs_vqa", "rs_caption") and vqa_service.should_use_real_vqa(mode, tasks))
        is_real_change = (tid == "change_detector" and mode == "bi_temporal" and len(files) == 2)
        is_real_change_vqa = (tid == "change_vqa" and mode == "bi_temporal" and len(files) == 2)
        is_real_optical_sar = (tid == "optical_sar_analyzer" and mode == "optical_sar" and len(files) == 2)
        exec_label = "REAL" if (is_real_vqa or is_real_change or is_real_change_vqa or is_real_optical_sar) else "MOCK"
        selection_bits.append(f"{tid}[{exec_label}] → {tm['name']} {tm['version']}")
    mark_step(db, aid, "step-4", "done",
              detail=" || ".join(selection_bits) if selection_bits else "No tools selected",
              meta={"tool_count": len(tool_ids)})

    # Step 5 — Parameters
    mark_step(db, aid, "step-5", "in_progress")
    param_bits = []
    for tid, params in per_tool_params.items():
        if params:
            pstr = ", ".join(f"{k}={v}" for k, v in params.items())
            param_bits.append(f"{tid}: {pstr}")
    param_detail = " || ".join(param_bits) if param_bits else "default parameters"
    mark_step(db, aid, "step-5", "done", detail=param_detail,
              meta={k: v for kv in per_tool_params.values() for k, v in (kv or {}).items()})

    # Step 6 — Processing (real or mock inference)
    mark_step(db, aid, "step-6", "in_progress")
    db.commit()
    tool_exec_modes: Dict[str, str] = {}
    try:
        (
            merged_answer,
            agg_conf,
            invocations,
            all_boxes,
            all_evidence,
            change_map,
            tool_exec_modes,
            change_stats,
        ) = execute_plan(
            q, mode, tool_ids, per_tool_params,
            tasks=tasks,
            image_file_paths=image_file_paths,
            analysis_id=aid,
            compatibility_context=comp_dict,
        )
    except Exception as e:
        logger.exception("Pipeline execution failed")
        mark_step(db, aid, "step-6", "error",
                  detail=f"Execution error: {type(e).__name__}: {e}",
                  meta={"error_type": type(e).__name__})
        mark_step(db, aid, "step-7", "error", detail="Aborted due to upstream error")
        mark_step(db, aid, "step-8", "error", detail="Failed")
        analysis.status = "failed"
        analysis.error_reason = f"Pipeline error: {type(e).__name__}: {e}"
        db.flush()
        _persist_to_firebase(firebase_repo, analysis, db)
        return

    proc_bits = []
    total_ms = 0
    for inv in invocations:
        tms = inv.processingTimeMs or 0
        total_ms += tms
        mode_label = inv.executionMode.upper()
        proc_bits.append(f"{inv.toolName}[{mode_label}]: {tms}ms")
    if all_boxes:
        proc_bits.append(f"{len(all_boxes)} bounding boxes")
    if change_map:
        is_real_changemap = tool_exec_modes.get("change_detector") == "real"
        # Determine the specific algorithm used from change_stats
        _cd_exec_mode = change_stats.get("execution_mode", "cpu_classical") if change_stats else "cpu_classical"
        if is_real_changemap:
            if _cd_exec_mode == "model_checkpoint":
                cm_label = "real SiameseUNet model change map generated (LEVIR-CD trained checkpoint)"
            else:
                cm_label = "real CPU pixel-difference change map generated"
        else:
            cm_label = "change map generated (mock)"
        proc_bits.append(cm_label)
    # Build step-6 meta: always include tool counts; add scalar change stats when available
    step6_meta: Dict[str, Any] = {
        "tools_run": len(invocations),
        "total_tool_ms": total_ms,
        "boxes": len(all_boxes),
        "any_real": any(v == "real" for v in tool_exec_modes.values()),
    }
    # Inject scalar change stats so the frontend can display the ChangeStatsPanel
    # Only flat types (int, float, str, bool) — TypeScript meta is Record<string, string|number|boolean>
    _SCALAR_STATS_KEYS = {
        "changed_pixel_pct", "unchanged_pixel_pct", "changed_pixel_count",
        "unchanged_pixel_count", "total_pixel_count", "threshold_raw_255",
        "threshold_used",
        "processing_time_ms", "size_mismatch_corrected", "severity",
        "image_size_str", "overlay_url",
        # Execution provenance — always present from model_inference dispatcher
        "execution_mode", "checkpoint_path",
    }
    for k in _SCALAR_STATS_KEYS:
        if k in change_stats and isinstance(change_stats[k], (int, float, str, bool)):
            step6_meta[k] = change_stats[k]
    mark_step(db, aid, "step-6", "done",
              detail=" || ".join(proc_bits),
              meta=step6_meta)

    # Step 7 — Aggregation + Result validation (anti-fabrication)
    mark_step(db, aid, "step-7", "in_progress")
    any_mock = any(v == "mock" for v in tool_exec_modes.values())
    agg_detail_bits = [
        "Merging tool outputs",
        "Concatenating answer sections",
    ]
    if agg_conf is not None:
        agg_detail_bits.append(f"Confidence aggregation: mean of tool confidences → {agg_conf:.3f}")
    else:
        agg_detail_bits.append("Confidence: null (no calibrated model-reported score)")
    mark_step(db, aid, "step-7", "done", detail=" | ".join(agg_detail_bits))

    analysis.answer_text = merged_answer
    analysis.confidence = agg_conf
    analysis.bounding_boxes = all_boxes if all_boxes else None
    analysis.change_map = change_map
    analysis.detected_tasks = list(tasks)
    analysis.tool_invocations = [inv.model_dump() for inv in invocations]
    analysis.evidence = list(all_evidence)
    analysis.selected_tools = list(tool_ids)
    analysis.compatibility = comp_dict
    analysis.limitations = limits_list or (comp_dict.get("limitations") if comp_dict else [])
    analysis.specialist_selected = next(iter(tool_ids), None) if tool_ids else None
    analysis.input_summary = {
        "image_count": len(inspections),
        "inspections": [r.to_dict() for r in inspections],
    }

    # Step 8 — Completion + final validation + Firebase sync
    mark_step(db, aid, "step-8", "in_progress")
    trace = build_trace_out(db, aid, "completed")
    total_ms_actual = trace.totalElapsedMs or total_ms
    analysis.total_elapsed_ms = total_ms_actual
    analysis.status = "completed"

    # Final API-level anti-fabrication validation pass
    raw_result = analysis_to_result(db, analysis)
    try:
        raw_dict = raw_result.model_dump(mode="json")
    except Exception:
        raw_dict = {}
    validation_report = validate_analysis_result_payload(raw_dict, is_mock=any_mock)
    if validation_report.warnings:
        logger.info(f"Analysis {aid} validation warnings: {validation_report.warnings}")
        if "boundingBoxes" in validation_report.stripped_fields:
            analysis.bounding_boxes = None
        if "changeMap" in validation_report.stripped_fields:
            analysis.change_map = None

    status_bits = [
        f"Confidence: {analysis.confidence if analysis.confidence is not None else 'null'}",
        f"Total elapsed: {(total_ms_actual / 1000):.2f}s",
    ]
    if analysis.bounding_boxes:
        status_bits.append(f"{len(analysis.bounding_boxes)} boxes")
    mode_summary = ", ".join(f"{k}={v}" for k, v in tool_exec_modes.items()) or "none"
    status_bits.append(f"tools=[{mode_summary}]")
    if validation_report.warnings:
        status_bits.append(f"warnings={len(validation_report.warnings)}")
    mark_step(db, aid, "step-8", "done",
              detail="Analysis complete | " + " | ".join(status_bits),
              meta={
                  "confidence": analysis.confidence,
                  "elapsed_ms": total_ms_actual,
                  "boxes": len(analysis.bounding_boxes or []),
                  "tool_execution_modes": tool_exec_modes,
                  "validation_warnings": validation_report.warnings,
              })

    db.flush()
    _persist_to_firebase(firebase_repo, analysis, db)


@router.post("", response_model=dict)
def submit_analysis(input_data: SubmitAnalysisInput, db: Session = Depends(get_db)):
    files = _validate_input(db, input_data)

    # ------------------------------------------------------------------
    # Multilingual normalization layer (Hindi → English)
    # ------------------------------------------------------------------
    # Rules (per design):
    #   * analysis.query ALWAYS stores the user's ORIGINAL query (English,
    #     Hindi, or mixed) so the UI displays what the user typed.
    #   * If language=hi or Devanagari is detected, we translate to
    #     English BEFORE the existing pipeline runs. A mutated copy of
    #     input_data with the normalized English query is passed to
    #     _run_analysis_pipeline so task_classification, specialist
    #     routing, and model prompts all receive English (the only
    #     language the existing pipeline was designed for).
    #   * English / non-Hindi queries are passed through byte-for-byte
    #     identical to the pre-multilingual behaviour.
    from ..services.multilingual import translate_hindi_to_english

    original_query = input_data.query
    normalized_query, _ = translate_hindi_to_english(
        original_query, getattr(input_data, "language", None)
    )

    # Build a pipeline copy with the normalized English query.
    pipeline_input = SubmitAnalysisInput(
        mode=input_data.mode,
        imageIds=list(input_data.imageIds),
        query=normalized_query,
        provider=getattr(input_data, "provider", None),
        language=getattr(input_data, "language", None),
    )

    # analysis.query stores the USER'S original text (never overwritten).
    analysis = Analysis(
        mode=input_data.mode,
        query=original_query,
        status="queued",
    )
    db.add(analysis)
    db.flush()
    aid = analysis.id

    _link_images(db, aid, input_data.mode, input_data.imageIds)
    create_pending_trace(db, aid, input_data.mode, original_query)
    db.flush()

    try:
        _run_analysis_pipeline(db, analysis, files, pipeline_input)
    except HTTPException:
        raise
    except MemoryError as me:
        logger.exception("OOM during analysis pipeline")
        analysis.status = "failed"
        analysis.error_reason = f"Out of memory: {me}"
        mark_step(db, aid, "step-8", "error", detail=f"Failed: OOM")
        db.flush()
    except TimeoutError as te:
        logger.exception("Timeout during analysis pipeline")
        analysis.status = "failed"
        analysis.error_reason = f"Timeout: {te}"
        mark_step(db, aid, "step-8", "error", detail=f"Failed: timeout")
        db.flush()
    except Exception as e:
        logger.exception("Analysis pipeline threw")
        analysis.status = "failed"
        analysis.error_reason = f"Unhandled error: {type(e).__name__}: {e}"
        mark_step(db, aid, "step-8", "error", detail=f"Failed: {type(e).__name__}")
        db.flush()

    db.commit()
    logger.info(f"Created analysis {aid} status={analysis.status}")
    return {"analysisId": aid}


@router.get("/{analysis_id}", response_model=AnalysisResult)
def get_analysis(analysis_id: str, db: Session = Depends(get_db)):
    a = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if a is None:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found")
    result = analysis_to_result(db, a)
    # Lightweight anti-fabrication pass on read too.
    is_mock = any(
        (inv.executionMode == "mock") for inv in result.toolInvocations
    ) or not result.toolInvocations
    try:
        raw_dict = result.model_dump(mode="json")
        validate_analysis_result_payload(raw_dict, is_mock=is_mock)
    except Exception:
        pass
    return result


@router.get("/{analysis_id}/trace", response_model=ExecutionTraceOut)
def get_analysis_trace(analysis_id: str, db: Session = Depends(get_db)):
    a = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if a is None:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found")
    return build_trace_out(db, analysis_id, a.status)


@router.post("/{analysis_id}/follow-up", response_model=FollowUpResponse)
def ask_analysis_follow_up(
    analysis_id: str,
    payload: FollowUpRequest,
    db: Session = Depends(get_db),
):
    a = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if a is None:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found")
    if not payload.query or not payload.query.strip():
        raise HTTPException(status_code=400, detail="Follow-up query must not be empty")

    return answer_follow_up(
        analysis=a,
        query=payload.query,
        history=payload.conversationHistory,
        language=payload.language,
    )


@router.get("", response_model=HistoryPage)
def list_history(
    mode: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    dateFrom: Optional[str] = Query(None),
    dateTo: Optional[str] = Query(None),
    minConfidence: Optional[float] = Query(None),
    page: int = Query(1, ge=1),
    pageSize: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    filters = HistoryFilters(
        mode=mode, status=status, dateFrom=dateFrom, dateTo=dateTo,
        minConfidence=minConfidence, page=page, pageSize=pageSize,
    )
    return list_analyses(db, filters)


@router.delete("/{analysis_id}", status_code=204)
def delete_analysis(analysis_id: str, db: Session = Depends(get_db)):
    a = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if a is None:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found")
    db.delete(a)
    db.commit()
    return None
