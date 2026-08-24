// lib/api/mock/mockApi.ts
// Full mock implementation of SatQueryApi.
// All async operations use staged setTimeout to simulate realistic latency
// and drive the execution trace step-by-step — the trace animation is real
// async state, not a CSS animation pretending to be data.

import type {
  AnalysisResult,
  AnalysisStatus,
  BenchmarkMetric,
  ExecutionStep,
  ExecutionTrace,
  HistoryFilters,
  ImageMetadataType,
  Modality,
  SubmitAnalysisInput,
  ToolDefinition,
  UploadedImage,
} from '@/lib/types/analysis';
import type { SatQueryApi } from '../index';
import { allFixtures, biTemporalResult, opticalSarResult, singleImageResult } from './fixtures';
import { generateId, sleep } from '@/lib/utils';

// In-memory store (resets on page refresh — intentional for demo)
const store = new Map<string, AnalysisResult>(
  allFixtures.map((r) => [r.id, { ...r }])
);

// ---- Tool registry ----
const MOCK_TOOLS: ToolDefinition[] = [
  {
    id: 'orchestrator-v1',
    name: 'SatQuery Orchestrator',
    taskTypes: ['vqa', 'captioning', 'grounding', 'change_detection', 'change_vqa', 'change_description'],
    supportedModalities: ['optical', 'sar', 'multispectral', 'unknown'],
    status: 'mock',
    version: '1.0.0-demo',
    description: 'Central routing and orchestration agent. Classifies the incoming query into task types, selects appropriate specialist models, coordinates parallel/sequential execution, and aggregates results into a unified response.',
  },
  {
    id: 'vqa-model-v1',
    name: 'RSVQA Vision-Language Model',
    taskTypes: ['vqa', 'captioning'],
    supportedModalities: ['optical', 'multispectral'],
    status: 'mock',
    version: '1.2.0-demo',
    description: 'Remote-sensing visual question answering and image captioning model. Fine-tuned on RSVQA-HR and RSITMD datasets. Accepts single optical/multispectral images and free-form natural language queries.',
  },
  {
    id: 'grounding-model-v1',
    name: 'RS-DINO Grounding Detector',
    taskTypes: ['grounding'],
    supportedModalities: ['optical', 'multispectral'],
    status: 'mock',
    version: '1.0.3-demo',
    description: 'Open-vocabulary object detection and grounding model for remote-sensing imagery. Based on Grounding DINO architecture, adapted for aerial/satellite image resolutions. Produces bounding boxes with label and confidence for arbitrary text-specified object classes.',
  },
  {
    id: 'change-det-v1',
    name: 'Bi-temporal Change Detection Model',
    taskTypes: ['change_detection', 'change_description'],
    supportedModalities: ['optical', 'multispectral'],
    status: 'mock',
    version: '2.1.0-demo',
    description: 'Change detection model for bi-temporal optical image pairs. Implements CVA (Change Vector Analysis) + NDVI/NDWI difference approaches. Produces per-pixel change masks with semantic class labels. Supports co-registration alignment for unregistered input pairs.',
  },
  {
    id: 'change-vqa-v1',
    name: 'Change-VQA Language Model',
    taskTypes: ['change_vqa', 'change_description'],
    supportedModalities: ['optical', 'multispectral'],
    status: 'mock',
    version: '1.1.0-demo',
    description: 'Specialised VQA head operating on bi-temporal image pairs and change masks. Answers natural-language questions about detected changes, generates human-readable change descriptions, and provides quantified change statistics (area, percentage, class-wise breakdown).',
  },
  {
    id: 'sar-optical-fusion-v1',
    name: 'SAR-Optical Cross-Modal Fusion Engine',
    taskTypes: ['vqa', 'change_detection'],
    supportedModalities: ['sar', 'optical', 'multispectral'],
    status: 'mock',
    version: '1.3.0-demo',
    description: 'Multi-modal fusion engine for Optical + SAR image pairs. Performs phase-correlation co-registration, weighted feature stack fusion, and cross-modal consistency analysis. Identifies features detectable only in SAR (sub-canopy structures, flooded areas, surface roughness patterns) vs optical-only features.',
  },
];

// ---- Benchmark metrics (all null — not evaluated yet) ----
const MOCK_METRICS: BenchmarkMetric[] = [
  { taskType: 'vqa', metricName: 'Accuracy', value: null, datasetName: 'RSVQA-HR', evaluatedAt: null },
  { taskType: 'vqa', metricName: 'F1 Score', value: null, datasetName: 'RSVQA-LR', evaluatedAt: null },
  { taskType: 'captioning', metricName: 'BLEU-4', value: null, datasetName: 'RSITMD', evaluatedAt: null },
  { taskType: 'captioning', metricName: 'CIDEr', value: null, datasetName: 'RSITMD', evaluatedAt: null },
  { taskType: 'captioning', metricName: 'METEOR', value: null, datasetName: 'UCM-Captions', evaluatedAt: null },
  { taskType: 'grounding', metricName: 'mAP@0.5', value: null, datasetName: 'DIOR-RSVG', evaluatedAt: null },
  { taskType: 'grounding', metricName: 'IoU (mean)', value: null, datasetName: 'DIOR-RSVG', evaluatedAt: null },
  { taskType: 'change_detection', metricName: 'F1 Score', value: null, datasetName: 'LEVIR-CD', evaluatedAt: null },
  { taskType: 'change_detection', metricName: 'IoU', value: null, datasetName: 'LEVIR-CD', evaluatedAt: null },
  { taskType: 'change_detection', metricName: 'Precision', value: null, datasetName: 'xBD', evaluatedAt: null },
  { taskType: 'change_detection', metricName: 'Recall', value: null, datasetName: 'xBD', evaluatedAt: null },
  { taskType: 'change_vqa', metricName: 'Accuracy', value: null, datasetName: 'LEVIR-CD-QA (custom)', evaluatedAt: null },
];

// ---- Modality detection heuristic (mock) ----
function detectModality(file: File): Modality {
  const name = file.name.toLowerCase();
  if (name.includes('sar') || name.includes('risat') || name.includes('sentinel-1')) return 'sar';
  if (name.includes('liss') || name.includes('msi') || name.includes('multispectral') || name.includes('s2')) return 'multispectral';
  if (name.includes('pan') || name.includes('optical') || name.includes('cartosat') || name.includes('rgb')) return 'optical';
  const ext = name.split('.').pop() ?? '';
  if (['tif', 'tiff'].includes(ext)) return 'optical'; // assume optical for generic GeoTIFF
  return 'unknown';
}

// ---- Build staged execution steps for a new analysis ----
function buildPendingSteps(mode: string, query: string): ExecutionStep[] {
  const now = new Date().toISOString();
  return [
    { id: 'step-1', title: 'Query Received', detail: `Query: "${query.slice(0, 80)}${query.length > 80 ? '…' : ''}" | Mode: ${mode}`, status: 'pending', startedAt: null, completedAt: null },
    { id: 'step-2', title: 'Input Validation', detail: 'Checking file formats, modality, CRS metadata, and pair alignment…', status: 'pending', startedAt: null, completedAt: null },
    { id: 'step-3', title: 'Task Classification', detail: 'Mapping query to task types (VQA / Captioning / Grounding / Change Detection)…', status: 'pending', startedAt: null, completedAt: null },
    { id: 'step-4', title: 'Tool Selection', detail: 'Routing to specialist model(s) based on detected tasks…', status: 'pending', startedAt: null, completedAt: null },
    { id: 'step-5', title: 'Parameters', detail: 'Configuring inference parameters…', status: 'pending', startedAt: null, completedAt: null },
    { id: 'step-6', title: 'Processing', detail: 'Running model inference…', status: 'pending', startedAt: null, completedAt: null },
    { id: 'step-7', title: 'Aggregation', detail: 'Merging outputs and calibrating confidence…', status: 'pending', startedAt: null, completedAt: null },
    { id: 'step-8', title: 'Completion', detail: 'Finalising result…', status: 'pending', startedAt: null, completedAt: null },
  ];
}

// ---- Pick fixture closest to the submitted mode for live-demo fill ----
function pickFixtureForMode(mode: string): AnalysisResult {
  if (mode === 'bi_temporal') return biTemporalResult;
  if (mode === 'optical_sar') return opticalSarResult;
  return singleImageResult;
}

// ---- Mock API implementation ----
export const mockApi: SatQueryApi = {
  async uploadImage(file: File, role: UploadedImage['role']): Promise<UploadedImage> {
    await sleep(400 + Math.random() * 300);
    const isGeoTiff = ['tif', 'tiff'].includes(file.name.split('.').pop()?.toLowerCase() ?? '');
    const previewUrl = isGeoTiff ? null : URL.createObjectURL(file);
    const modality = detectModality(file);
    const metadata: ImageMetadataType = {
      fileName: file.name,
      fileFormat: isGeoTiff ? 'GeoTIFF' : (file.type.includes('png') ? 'PNG' : 'JPEG'),
      modality,
      modalityDetectionConfidence: modality !== 'unknown' ? 0.75 + Math.random() * 0.2 : null,
      acquisitionDate: null,
      widthPx: isGeoTiff ? null : null, // Only extractable with actual decoder
      heightPx: null,
      bandCount: isGeoTiff ? null : (modality === 'sar' ? 1 : 3),
      crs: isGeoTiff ? null : null,
      gsdMeters: null,
      fileSizeBytes: file.size,
    };
    return { id: generateId(), role, previewUrl, metadata };
  },

  async submitAnalysis(input: SubmitAnalysisInput): Promise<{ analysisId: string }> {
    await sleep(300);
    const analysisId = `analysis-${generateId()}`;
    const fixture = pickFixtureForMode(input.mode);
    const steps = buildPendingSteps(input.mode, input.query);
    const newResult: AnalysisResult = {
      ...fixture,
      id: analysisId,
      mode: input.mode,
      query: input.query,
      status: 'queued',
      createdAt: new Date().toISOString(),
      answerText: null,
      confidence: null,
      boundingBoxes: null,
      changeMap: null,
      executionTrace: { steps, totalElapsedMs: null, overallStatus: 'queued' },
      errorReason: null,
    };
    store.set(analysisId, newResult);
    return { analysisId };
  },

  async getAnalysis(id: string): Promise<AnalysisResult> {
    await sleep(150);
    const result = store.get(id);
    if (!result) throw new Error(`Analysis ${id} not found`);
    return { ...result };
  },

  streamExecutionTrace(id: string, onUpdate: (trace: ExecutionTrace) => void): () => void {
    let cancelled = false;

    const run = async () => {
      const result = store.get(id);
      if (!result) return;

      const fixture = pickFixtureForMode(result.mode);
      const fixtureSteps = fixture.executionTrace.steps;
      const stepDelays = [200, 600, 400, 350, 300, 2800, 600, 300];

      // Mark queued → processing
      result.status = 'processing';
      result.executionTrace.overallStatus = 'processing';
      store.set(id, result);

      for (let i = 0; i < result.executionTrace.steps.length; i++) {
        if (cancelled) return;
        const step = result.executionTrace.steps[i];
        const fixtureStep = fixtureSteps[i];

        // Mark in_progress
        step.status = 'in_progress';
        step.startedAt = new Date().toISOString();
        const updated1 = { ...result.executionTrace, steps: [...result.executionTrace.steps] };
        onUpdate(updated1);

        await sleep(stepDelays[i] ?? 400);
        if (cancelled) return;

        // Mark done with fixture detail
        step.status = fixtureStep?.status === 'error' ? 'error' : 'done';
        step.completedAt = new Date().toISOString();
        step.detail = fixtureStep?.detail ?? step.detail;
        if (fixtureStep?.meta) step.meta = fixtureStep.meta;

        const updated2 = { ...result.executionTrace, steps: [...result.executionTrace.steps] };
        onUpdate(updated2);
        store.set(id, result);
      }

      // Finalise result
      await sleep(200);
      if (cancelled) return;
      const finalFixture = fixture;
      result.status = 'completed';
      result.answerText = finalFixture.answerText;
      result.confidence = finalFixture.confidence;
      result.boundingBoxes = finalFixture.boundingBoxes;
      result.changeMap = finalFixture.changeMap;
      result.toolInvocations = finalFixture.toolInvocations;
      result.detectedTasks = finalFixture.detectedTasks;
      result.executionTrace.overallStatus = 'completed';
      result.executionTrace.totalElapsedMs = stepDelays.reduce((a, b) => a + b, 0);
      store.set(id, result);
      onUpdate({ ...result.executionTrace });
    };

    run();
    return () => { cancelled = true; };
  },

  async listAnalysisHistory(filters: HistoryFilters): Promise<{ items: AnalysisResult[]; total: number }> {
    await sleep(300);
    let items = Array.from(store.values()).sort(
      (a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
    );
    if (filters.mode) items = items.filter((r) => r.mode === filters.mode);
    if (filters.status) items = items.filter((r) => r.status === filters.status);
    if (filters.minConfidence != null)
      items = items.filter((r) => r.confidence != null && r.confidence >= filters.minConfidence!);
    const total = items.length;
    const page = filters.page ?? 1;
    const pageSize = filters.pageSize ?? 20;
    items = items.slice((page - 1) * pageSize, page * pageSize);
    return { items, total };
  },

  async deleteAnalysis(id: string): Promise<void> {
    await sleep(200);
    store.delete(id);
  },

  async listTools(): Promise<ToolDefinition[]> {
    await sleep(200);
    return MOCK_TOOLS;
  },

  async getBenchmarkMetrics(): Promise<BenchmarkMetric[]> {
    await sleep(250);
    return MOCK_METRICS;
  },
};
