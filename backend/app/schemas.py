from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


AnalysisMode = Literal["single_image", "bi_temporal", "optical_sar"]
Modality = Literal["optical", "sar", "multispectral", "unknown"]
TaskType = Literal["vqa", "captioning", "grounding", "change_detection", "change_vqa", "change_description"]
AnalysisStatus = Literal["queued", "processing", "completed", "failed"]
StepStatus = Literal["pending", "in_progress", "done", "error"]
FileFormat = Literal["GeoTIFF", "TIFF", "PNG", "JPEG"]
ImageRole = Literal["single", "before", "after", "optical", "sar"]
ToolStatus = Literal["available", "mock", "planned"]


class ImageMetadataType(BaseModel):
    fileName: str
    fileFormat: FileFormat
    modality: Modality
    modalityDetectionConfidence: Optional[float] = None
    acquisitionDate: Optional[str] = None
    widthPx: Optional[int] = None
    heightPx: Optional[int] = None
    bandCount: Optional[int] = None
    crs: Optional[str] = None
    gsdMeters: Optional[float] = None
    fileSizeBytes: int


class UploadedImage(BaseModel):
    id: str
    role: ImageRole
    previewUrl: Optional[str] = None
    metadata: ImageMetadataType


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float
    label: str
    confidence: float


class ToolInvocation(BaseModel):
    toolId: str
    toolName: str
    version: str
    taskType: TaskType
    parameters: Dict[str, Any]
    processingTimeMs: Optional[int] = None
    executionMode: Literal["real", "mock"] = "mock"


class ExecutionStepOut(BaseModel):
    id: str
    title: str
    detail: str
    status: StepStatus
    startedAt: Optional[str] = None
    completedAt: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


class ExecutionTraceOut(BaseModel):
    steps: List[ExecutionStepOut]
    totalElapsedMs: Optional[int] = None
    overallStatus: AnalysisStatus


class ChangeMap(BaseModel):
    overlayUrl: Optional[str] = None
    legend: List[Dict[str, str]]


class AnalysisResult(BaseModel):
    id: str
    mode: AnalysisMode
    query: str
    status: AnalysisStatus
    createdAt: str
    images: List[UploadedImage]
    detectedTasks: List[TaskType]
    answerText: Optional[str] = None
    confidence: Optional[float] = None
    boundingBoxes: Optional[List[BoundingBox]] = None
    changeMap: Optional[ChangeMap] = None
    toolInvocations: List[ToolInvocation]
    executionTrace: ExecutionTraceOut
    errorReason: Optional[str] = None

    task: Optional[str] = None
    selectedTools: Optional[List[str]] = None
    evidence: Optional[List[str]] = None
    analysisStatus: Optional[AnalysisStatus] = None


class SubmitAnalysisInput(BaseModel):
    mode: AnalysisMode
    imageIds: List[str]
    query: str


class ToolDefinition(BaseModel):
    id: str
    name: str
    taskTypes: List[TaskType]
    supportedModalities: List[Modality]
    status: ToolStatus
    version: str
    description: str


class BenchmarkMetric(BaseModel):
    taskType: TaskType
    metricName: str
    value: Optional[float] = None
    datasetName: str
    evaluatedAt: Optional[str] = None


class HistoryFilters(BaseModel):
    mode: Optional[AnalysisMode] = None
    status: Optional[AnalysisStatus] = None
    dateFrom: Optional[str] = None
    dateTo: Optional[str] = None
    minConfidence: Optional[float] = None
    page: int = 1
    pageSize: int = 20


class HistoryPage(BaseModel):
    items: List[AnalysisResult]
    total: int


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    timestamp: str


class ErrorResponse(BaseModel):
    detail: str
