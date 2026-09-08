// lib/types/analysis.ts
// Contract types shared between the mock API layer and future FastAPI backend.
// All UI components consume these types exclusively — never inline API shapes.

export type AnalysisMode = 'single_image' | 'bi_temporal' | 'optical_sar';
export type Modality = 'optical' | 'sar' | 'multispectral' | 'unknown';
export type TaskType =
  | 'vqa'
  | 'captioning'
  | 'grounding'
  | 'change_detection'
  | 'change_vqa'
  | 'change_description';
export type AnalysisStatus = 'queued' | 'processing' | 'completed' | 'failed';
export type StepStatus = 'pending' | 'in_progress' | 'done' | 'error';

export interface ImageMetadataType {
  fileName: string;
  fileFormat: 'GeoTIFF' | 'TIFF' | 'PNG' | 'JPEG';
  modality: Modality;
  modalityDetectionConfidence: number | null;
  acquisitionDate: string | null; // ISO date, null if not extractable
  widthPx: number | null;
  heightPx: number | null;
  bandCount: number | null;
  crs: string | null; // e.g. "EPSG:32643"
  gsdMeters: number | null;
  fileSizeBytes: number;
}

export interface UploadedImage {
  id: string;
  role: 'single' | 'before' | 'after' | 'optical' | 'sar';
  previewUrl: string | null; // null for non-renderable GeoTIFF
  metadata: ImageMetadataType;
}

export interface ExecutionStep {
  id: string;
  title: string;
  detail: string;
  status: StepStatus;
  startedAt: string | null;
  completedAt: string | null;
  meta?: Record<string, string | number | boolean | null>;
}

export interface ExecutionTrace {
  steps: ExecutionStep[];
  totalElapsedMs: number | null;
  overallStatus: AnalysisStatus;
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number; // normalized 0-1
  label: string;
  confidence: number;
}

export interface ToolInvocation {
  toolId: string;
  toolName: string;
  version: string;
  taskType: TaskType;
  parameters: Record<string, string | number | boolean>;
  processingTimeMs: number | null;
  executionMode?: 'real' | 'mock';
}

export interface AnalysisResult {
  id: string;
  mode: AnalysisMode;
  query: string;
  status: AnalysisStatus;
  createdAt: string;
  images: UploadedImage[];
  detectedTasks: TaskType[];
  answerText: string | null;
  confidence: number | null; // 0-1, null if not completed
  boundingBoxes: BoundingBox[] | null;
  changeMap: {
    overlayUrl: string | null;
    legend: { label: string; color: string }[];
  } | null;
  toolInvocations: ToolInvocation[];
  executionTrace: ExecutionTrace;
  errorReason: string | null;
  compatibility?: Record<string, any> | null;
  limitations?: string[] | null;
  adaptation?: Record<string, any> | null;
  specialistSelected?: string | null;
  inputSummary?: Record<string, any> | null;
  isMock?: boolean;
  executionMode?: 'real' | 'mock' | 'mixed';
  multilingualSummaries?: MultilingualSummaries | null;
}

export interface MultilingualSummaries {
  language?: 'en' | 'hi' | null;
  summary_en?: string | null;
  summary_hi?: string | null;
  bullet_en?: string[] | null;
  bullet_hi?: string[] | null;
  generated_via_llm?: boolean | null;
}

export interface ToolDefinition {
  id: string;
  name: string;
  taskTypes: TaskType[];
  supportedModalities: Modality[];
  status: 'available' | 'mock' | 'planned';
  version: string;
  description: string;
}

export interface BenchmarkMetric {
  taskType: TaskType;
  metricName: string; // e.g. "Accuracy", "BLEU-4", "mAP@0.5", "IoU"
  value: number | null; // null => render "Not evaluated yet"
  datasetName: string;
  evaluatedAt: string | null;
}

// ---- API input types ----

export interface SubmitAnalysisInput {
  mode: AnalysisMode;
  imageIds: string[];
  query: string;
  language?: 'en' | 'hi';
}

export interface HistoryFilters {
  mode?: AnalysisMode;
  status?: AnalysisStatus;
  dateFrom?: string;
  dateTo?: string;
  minConfidence?: number;
  page?: number;
  pageSize?: number;
}

// ---- Contextual Follow-up & Smart Insights Types ----

export interface SpatialAction {
  action: 'highlight' | 'zoom' | 'scroll' | 'none';
  target?: 'change_map' | 'bounding_box' | 'optical_sar';
  boxIndex?: number | null;
  note?: string | null;
}

export interface FollowUpMessage {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  timestamp: string;
  language?: 'en' | 'hi';
  spatialAction?: SpatialAction | null;
  referencedMetrics?: Record<string, string | number | null> | null;
}

export interface FollowUpRequest {
  query: string;
  language?: 'en' | 'hi';
  conversationHistory?: { role: 'user' | 'assistant'; text: string }[];
}

export interface FollowUpResponse {
  answer: string;
  answer_hi?: string | null;
  language: 'en' | 'hi';
  referencedMetrics?: Record<string, string | number | null> | null;
  spatialAction?: SpatialAction | null;
  rerunPerformed: false;
}

export interface SmartInsightsData {
  changeStatus?: string | null;
  changedArea?: string | null;
  severity?: string | null;
  confidence?: string | null;
  primaryFinding: string;
  dateComparison?: string | null;
  locationAoi?: string | null;
}
