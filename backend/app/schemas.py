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
    legend: Optional[List[Dict[str, str]]] = None
    analytics: Optional[Dict[str, Any]] = None
    changedPixelPct: Optional[float] = None
    changedPixels: Optional[int] = None
    totalPixels: Optional[int] = None



class MultilingualSummaries(BaseModel):
    """Additive-only multilingual view of an analysis result.

    All fields are optional (None means "not computed / not applicable")
    so existing API consumers see no behavioural change.  The summaries
    are computed once and cached into Analysis.adaptation["multilingual_summaries"]
    so this struct is O(1) to serve on subsequent GETs.
    """

    language: Optional[Literal["en", "hi"]] = None
    summary_en: Optional[str] = None
    summary_hi: Optional[str] = None
    bullet_en: Optional[List[str]] = None
    bullet_hi: Optional[List[str]] = None
    generated_via_llm: Optional[bool] = None


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

    compatibility: Optional[Dict[str, Any]] = None
    limitations: Optional[List[str]] = None
    adaptation: Optional[Dict[str, Any]] = None
    specialistSelected: Optional[str] = None
    inputSummary: Optional[Dict[str, Any]] = None
    isMock: Optional[bool] = None
    executionMode: Optional[Literal["real", "mock", "mixed"]] = None

    multilingualSummaries: Optional[MultilingualSummaries] = None


class SubmitAnalysisInput(BaseModel):
    mode: AnalysisMode
    imageIds: List[str]
    query: str
    provider: Optional[str] = None  # 'local', 'gemini', 'openrouter', 'auto'
    language: Optional[str] = None  # 'en' (default), 'hi' — optional multilingual layer


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


class ROIAnalysisInput(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    is_normalized: bool = True
    run_vqa: bool = False
    vqa_query: Optional[str] = None
    provider: Optional[str] = None


class ROIPixelCoords(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    width: int
    height: int


class ROINormalizedCoords(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class ROIInfo(BaseModel):
    pixel_coordinates: ROIPixelCoords
    normalized_coordinates: ROINormalizedCoords
    image_dimensions: Dict[str, int]


class ROIStatistics(BaseModel):
    total_pixels: int
    changed_pixels: int
    unchanged_pixels: int
    changed_percentage: float
    unchanged_percentage: float


class ROIPhysicalArea(BaseModel):
    available: bool
    gsd_meters: Optional[float] = None
    roi_total_area_m2: Optional[float] = None
    roi_total_area_hectares: Optional[float] = None
    roi_total_area_sqkm: Optional[float] = None
    roi_changed_area_m2: Optional[float] = None
    roi_changed_area_hectares: Optional[float] = None
    roi_changed_area_sqkm: Optional[float] = None
    reason: Optional[str] = None
    metadata_source: Optional[str] = None


class ROIGlobalComparison(BaseModel):
    global_changed_percentage: Optional[float] = None
    roi_changed_percentage: float
    difference_percentage: Optional[float] = None
    relative_density_factor: Optional[float] = None
    summary: Optional[str] = None


class ROIHotspotsResult(BaseModel):
    hotspots_count_total: int
    hotspots_count_significant: int
    largest_hotspot: Optional[Dict[str, Any]] = None
    hotspots: List[Dict[str, Any]] = []


class ROIAnalysisOutput(BaseModel):
    roi: ROIInfo
    statistics: ROIStatistics
    physical_area: ROIPhysicalArea
    global_comparison: ROIGlobalComparison
    hotspots: ROIHotspotsResult
    vqa: Optional[Dict[str, Any]] = None


class ReportGenerationInput(BaseModel):
    roi: Optional[Dict[str, Any]] = None
    generate_pdf: bool = True


class ReportResponse(BaseModel):
    report: Dict[str, Any]
    pdf_url: Optional[str] = None
    pdf_filename: Optional[str] = None


class FollowUpHistoryItem(BaseModel):
    role: str
    text: str


class FollowUpRequest(BaseModel):
    query: str
    language: Optional[str] = "en"
    conversationHistory: Optional[List[FollowUpHistoryItem]] = None


class SpatialActionOut(BaseModel):
    action: str
    target: Optional[str] = None
    boxIndex: Optional[int] = None
    note: Optional[str] = None


class FollowUpResponse(BaseModel):
    answer: str
    answer_hi: Optional[str] = None
    language: str = "en"
    referencedMetrics: Optional[Dict[str, Any]] = None
    spatialAction: Optional[SpatialActionOut] = None
    rerunPerformed: bool = False

