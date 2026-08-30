from datetime import UTC, datetime
from typing import Any, Dict, List, Optional


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)

from sqlalchemy.orm import Session

from ..models import Analysis, ExecutionStep
from ..schemas import (
    AnalysisMode,
    AnalysisStatus,
    ExecutionStepOut,
    ExecutionTraceOut,
    StepStatus,
)
from ..logging_setup import logger


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def create_pending_trace(
    db: Session,
    analysis_id: str,
    mode: AnalysisMode,
    query: str,
) -> List[ExecutionStep]:
    """Create the 8 canonical pending steps for a new analysis."""
    qshort = query if len(query) <= 80 else query[:80] + "…"
    templates: List[Dict[str, Any]] = [
        {"id": "step-1", "title": "Query Received",
         "detail": f'Query: "{qshort}" | Mode: {mode}'},
        {"id": "step-2", "title": "Input Validation",
         "detail": "Checking file formats, modality, CRS metadata, and pair alignment…"},
        {"id": "step-3", "title": "Task Classification",
         "detail": "Mapping query to task types (VQA / Captioning / Grounding / Change Detection)…"},
        {"id": "step-4", "title": "Tool Selection",
         "detail": "Routing to specialist model(s) based on detected tasks…"},
        {"id": "step-5", "title": "Parameters",
         "detail": "Configuring inference parameters…"},
        {"id": "step-6", "title": "Processing",
         "detail": "Running model inference…"},
        {"id": "step-7", "title": "Aggregation",
         "detail": "Merging outputs and calibrating confidence…"},
        {"id": "step-8", "title": "Completion",
         "detail": "Finalising result…"},
    ]
    steps: List[ExecutionStep] = []
    for idx, t in enumerate(templates):
        steps.append(ExecutionStep(
            analysis_id=analysis_id,
            step_id=t["id"],
            order_index=idx,
            title=t["title"],
            detail=t["detail"],
            status="pending",
        ))
    db.add_all(steps)
    db.flush()
    return steps


def mark_step(
    db: Session,
    analysis_id: str,
    step_id: str,
    status: StepStatus,
    detail: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    step = (
        db.query(ExecutionStep)
        .filter(ExecutionStep.analysis_id == analysis_id, ExecutionStep.step_id == step_id)
        .first()
    )
    if step is None:
        logger.warning(f"mark_step: step {step_id} not found for analysis {analysis_id}")
        return
    step.status = status
    if status == "in_progress" and step.started_at is None:
        step.started_at = utcnow()
    if status in ("done", "error") and step.completed_at is None:
        step.completed_at = utcnow()
    if detail is not None:
        step.detail = detail
    if meta is not None:
        existing = step.meta or {}
        existing.update(meta)
        step.meta = existing
    db.flush()


def build_trace_out(db: Session, analysis_id: str, overall_status: AnalysisStatus) -> ExecutionTraceOut:
    steps_q = (
        db.query(ExecutionStep)
        .filter(ExecutionStep.analysis_id == analysis_id)
        .order_by(ExecutionStep.order_index)
        .all()
    )
    out_steps: List[ExecutionStepOut] = []
    started_ref: Optional[datetime] = None
    ended_ref: Optional[datetime] = None
    for s in steps_q:
        if started_ref is None and s.started_at:
            started_ref = s.started_at
        if s.completed_at:
            ended_ref = s.completed_at
        out_steps.append(ExecutionStepOut(
            id=s.step_id,
            title=s.title,
            detail=s.detail or "",
            status=s.status,
            startedAt=_iso(s.started_at),
            completedAt=_iso(s.completed_at),
            meta=s.meta,
        ))
    total_ms: Optional[int] = None
    if started_ref and ended_ref:
        total_ms = int((ended_ref - started_ref).total_seconds() * 1000)
    return ExecutionTraceOut(
        steps=out_steps,
        totalElapsedMs=total_ms,
        overallStatus=overall_status,
    )
