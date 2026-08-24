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
  meta?: Record<string, string | number | boolean>;
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
