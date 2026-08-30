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
    ToolInvocation,
    UploadedImage,
)
from .services.trace import build_trace_out
from .config import get_settings

settings = get_settings()


def _iso(dt: Optional[datetime]) -> str:
    return dt.isoformat() if dt else None


def _file_to_meta(f: UploadedFile) -> ImageMetadataType:
    return ImageMetadataType(
        fileName=f.file_name,
        fileFormat=f.file_format or "TIFF",
        modality=f.modality or "unknown",
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
