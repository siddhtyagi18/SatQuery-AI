import uuid
from datetime import UTC, datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, JSON, Boolean
from sqlalchemy.orm import relationship

from .database import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class UploadedFile(Base):
    __tablename__ = "uploaded_files"

    id = Column(String, primary_key=True, default=generate_uuid)
    file_name = Column(String, nullable=False)
    stored_name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_format = Column(String, nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    mime_type = Column(String, nullable=True)

    modality = Column(String, nullable=True)
    modality_confidence = Column(Float, nullable=True)
    acquisition_date = Column(String, nullable=True)

    width_px = Column(Integer, nullable=True)
    height_px = Column(Integer, nullable=True)
    band_count = Column(Integer, nullable=True)
    crs = Column(String, nullable=True)
    gsd_meters = Column(Float, nullable=True)
    bounds = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=utcnow, nullable=False)

    analyses = relationship("AnalysisImage", back_populates="file")


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(String, primary_key=True, default=generate_uuid)
    mode = Column(String, nullable=False)
    query = Column(Text, nullable=False)
    status = Column(String, nullable=False, default="queued")

    answer_text = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    bounding_boxes = Column(JSON, nullable=True)
    change_map = Column(JSON, nullable=True)
    detected_tasks = Column(JSON, nullable=True)
    tool_invocations = Column(JSON, nullable=True)
    evidence = Column(JSON, nullable=True)
    selected_tools = Column(JSON, nullable=True)

    error_reason = Column(Text, nullable=True)
    execution_trace_json = Column(JSON, nullable=True)
    total_elapsed_ms = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=utcnow, nullable=False)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)

    images = relationship("AnalysisImage", back_populates="analysis", cascade="all, delete-orphan")
    trace_steps = relationship("ExecutionStep", back_populates="analysis", cascade="all, delete-orphan", order_by="ExecutionStep.order_index")


class AnalysisImage(Base):
    __tablename__ = "analysis_images"

    id = Column(String, primary_key=True, default=generate_uuid)
    analysis_id = Column(String, ForeignKey("analyses.id"), nullable=False)
    file_id = Column(String, ForeignKey("uploaded_files.id"), nullable=False)
    role = Column(String, nullable=False)

    analysis = relationship("Analysis", back_populates="images")
    file = relationship("UploadedFile", back_populates="analyses")


class ExecutionStep(Base):
    __tablename__ = "execution_steps"

    id = Column(String, primary_key=True, default=generate_uuid)
    analysis_id = Column(String, ForeignKey("analyses.id"), nullable=False)
    step_id = Column(String, nullable=False)
    order_index = Column(Integer, nullable=False)
    title = Column(String, nullable=False)
    detail = Column(Text, nullable=True)
    status = Column(String, nullable=False, default="pending")
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    meta = Column(JSON, nullable=True)

    analysis = relationship("Analysis", back_populates="trace_steps")
