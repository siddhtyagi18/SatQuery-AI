from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from .models import Analysis, AnalysisImage, ExecutionStep, UploadedFile
from .schemas import (
    AnalysisMode,
    AnalysisResult,
    AnalysisStatus,
    BoundingBox,
    ChangeMap,
    ExecutionTraceOut,
    HistoryFilters,
    HistoryPage,
    ImageMetadataType,
    ImageRole,
    MultilingualSummaries,
    ToolInvocation,
    UploadedImage,
)
from .services.trace import build_trace_out
from .config import get_settings

settings = get_settings()


def _attach_multilingual_summaries(
    db: Session,
    a: Analysis,
) -> Optional[MultilingualSummaries]:
    """Build-or-read multilingual summaries for an Analysis row.

    Strictly additive:
      * Never triggers re-analysis.
      * Only reads: answer_text, detected_tasks, confidence, query,
        adaptation (namespaced sub-key only).
      * Writes: Analysis.adaptation["multilingual_summaries"] ONLY if
        the key is missing.  Never overwrites any other adaptation key
        (teammate data protected).
      * Cache write is best-effort.  A cache failure is logged and does
        NOT break the API — we still return the freshly-built struct.
    """
    from .services.result_multilingual import (
        build_multilingual_summaries,
        cached_summaries,
        store_cached_summaries,
    )

    # Only populate once the analysis has produced an answer.
    # For failed / queued / processing rows, return None (no display).
    if a.status != "completed":
        return None
    if not a.answer_text:
        return None

    # 1) Fast path: already cached inside adaptation JSON.
    cached = cached_summaries(a)
    if cached:
        try:
            return MultilingualSummaries(**cached)
        except Exception:  # noqa: BLE001 — defensive, never break read
            cached = None

    # 2) Slow path: build summaries (dict fallback guaranteed to return
    #    valid text even if LLM is unreachable).
    try:
        summaries_dict = build_multilingual_summaries(
            answer_text=a.answer_text or "",
            detected_tasks=list(a.detected_tasks or []),
            confidence=a.confidence,
            query=a.query or "",
        )
    except Exception as exc:  # noqa: BLE001 — never break the read API
        from .logging_setup import logger
        logger.warning("[crud.multi] build failed (%s: %s)", type(exc).__name__, exc)
        return None

    # 3) Write-through cache into the namespaced adaptation key.
    #    store_cached_summaries calls db.flush() but does NOT commit.
    wrote_cache = False
    try:
        before = a.adaptation
        store_cached_summaries(db, a, summaries_dict)
        after = a.adaptation
        wrote_cache = (before != after)
    except Exception:  # noqa: BLE001 — caching is non-critical
        wrote_cache = False

    # If we mutated the adaptation column (cache miss), commit once so the
    # cache actually persists to disk.  This runs at most ONCE per analysis
    # row in its entire lifetime (subsequent reads hit the cached dict).
    if wrote_cache:
        try:
            db.commit()
        except Exception:  # noqa: BLE001 — persistence failure is non-fatal
            pass

    try:
        return MultilingualSummaries(**summaries_dict)
    except Exception:  # noqa: BLE001
        return None


def _iso(dt: Optional[datetime]) -> str:
    return dt.isoformat() if dt else None


def _norm_modality(m: Optional[str]) -> str:
    if not m or m == "unknown":
        return "unknown"
    ml = m.lower()
    if "sar" in ml:
        return "sar"
    if "multispectral" in ml:
        return "multispectral"
    if any(k in ml for k in ("optical", "rgb", "grayscale", "panchromatic")):
        return "optical"
    return "unknown"


def _file_to_meta(f: UploadedFile) -> ImageMetadataType:
    return ImageMetadataType(
        fileName=f.file_name,
        fileFormat=f.file_format or "TIFF",
        modality=_norm_modality(f.modality),
        modalityDetectionConfidence=f.modality_confidence,
        acquisitionDate=f.acquisition_date,
        widthPx=f.width_px,
        heightPx=f.height_px,
        bandCount=f.band_count,
        crs=f.crs,
        gsdMeters=f.gsd_meters,
        fileSizeBytes=f.file_size_bytes,
    )


def _file_to_uploaded_image(file: UploadedFile, role: ImageRole) -> UploadedImage:
    meta = _file_to_meta(file)
    is_geotiff = meta.fileFormat in ("GeoTIFF", "TIFF")
    preview_url = None if is_geotiff else f"/api/files/{file.id}"
    return UploadedImage(
        id=file.id,
        role=role,
        previewUrl=preview_url,
        metadata=meta,
    )


def analysis_to_result(db: Session, a: Analysis) -> AnalysisResult:
    imgs_db = (
        db.query(AnalysisImage)
        .filter(AnalysisImage.analysis_id == a.id)
        .all()
    )
    images: List[UploadedImage] = []
    for ai in imgs_db:
        f = db.query(UploadedFile).filter(UploadedFile.id == ai.file_id).first()
        if f:
            images.append(_file_to_uploaded_image(f, ai.role))

    trace: ExecutionTraceOut = build_trace_out(db, a.id, a.status)

    boxes: Optional[List[BoundingBox]] = None
    if a.bounding_boxes:
        boxes = [BoundingBox(**b) for b in a.bounding_boxes]

    change_map: Optional[ChangeMap] = None
    if a.change_map:
        change_map = ChangeMap(**a.change_map)

    tool_invocations: List[ToolInvocation] = []
    if a.tool_invocations:
        tool_invocations = [ToolInvocation(**t) for t in a.tool_invocations]

    detected_tasks = a.detected_tasks or []

    primary_task = detected_tasks[0] if detected_tasks else (a.mode or "vqa")
    selected_tools = a.selected_tools or []

    has_real = any(getattr(t, "executionMode", "mock") == "real" for t in tool_invocations)
    has_mock = any(getattr(t, "executionMode", "mock") == "mock" for t in tool_invocations)
    if has_real and has_mock:
        exec_mode = "mixed"
    elif has_real:
        exec_mode = "real"
    else:
        exec_mode = "mock"
    is_mock = not has_real

    return AnalysisResult(
        id=a.id,
        mode=a.mode,
        query=a.query,
        status=a.status,
        createdAt=_iso(a.created_at),
        images=images,
        detectedTasks=detected_tasks,
        answerText=a.answer_text,
        confidence=a.confidence,
        boundingBoxes=boxes,
        changeMap=change_map,
        toolInvocations=tool_invocations,
        executionTrace=trace,
        errorReason=a.error_reason,
        task=primary_task,
        selectedTools=selected_tools,
        evidence=a.evidence,
        analysisStatus=a.status,
        compatibility=getattr(a, "compatibility", None),
        limitations=getattr(a, "limitations", None),
        adaptation=getattr(a, "adaptation", None),
        specialistSelected=getattr(a, "specialist_selected", None),
        inputSummary=getattr(a, "input_summary", None),
        isMock=is_mock,
        executionMode=exec_mode,
        multilingualSummaries=_attach_multilingual_summaries(db, a),
    )


def list_analyses(db: Session, filters: HistoryFilters) -> HistoryPage:
    q = db.query(Analysis)
    if filters.mode:
        q = q.filter(Analysis.mode == filters.mode)
    if filters.status:
        q = q.filter(Analysis.status == filters.status)
    if filters.minConfidence is not None:
        q = q.filter(Analysis.confidence >= filters.minConfidence)
    if filters.dateFrom:
        try:
            df = datetime.fromisoformat(filters.dateFrom.replace("Z", "+00:00"))
            q = q.filter(Analysis.created_at >= df)
        except Exception:
            pass
    if filters.dateTo:
        try:
            dt = datetime.fromisoformat(filters.dateTo.replace("Z", "+00:00"))
            q = q.filter(Analysis.created_at <= dt)
        except Exception:
            pass
    total = q.count()
    page = max(1, filters.page)
    page_size = max(1, filters.pageSize)
    rows = (
        q.order_by(Analysis.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    items = [analysis_to_result(db, r) for r in rows]
    return HistoryPage(items=items, total=total)
