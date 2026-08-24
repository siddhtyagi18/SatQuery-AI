// lib/api/mock/fixtures/singleImageResult.ts
import type { AnalysisResult } from '@/lib/types/analysis';

export const singleImageResult: AnalysisResult = {
  id: 'analysis-001',
  mode: 'single_image',
  query: 'What land cover types are visible and locate all buildings in this image?',
  status: 'completed',
  createdAt: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
  images: [
    {
      id: 'img-001',
      role: 'single',
      previewUrl: '/demo/optical_sample.jpg',
      metadata: {
        fileName: 'RESOURCESAT-2_LISS-IV_2024-03-15.tif',
        fileFormat: 'GeoTIFF',
        modality: 'optical',
        modalityDetectionConfidence: 0.94,
        acquisitionDate: '2024-03-15',
        widthPx: 8192,
        heightPx: 8192,
        bandCount: 3,
        crs: 'EPSG:32643',
        gsdMeters: 5.8,
        fileSizeBytes: 67108864,
      },
    },
  ],
  detectedTasks: ['vqa', 'grounding'],
  answerText:
    'The image shows a mixed urban-agricultural landscape. The dominant land cover types are: (1) Dense urban settlement (~38% coverage) concentrated in the northwestern quadrant, characterised by clustered built structures with regular street grids; (2) Agricultural cropland (~41% coverage) in active cultivation phase, with field boundaries visible as linear boundaries; (3) Sparse vegetation / scrubland (~14%) along the southeastern margin; (4) Water body / reservoir (~7%) in the southern portion. A total of 127 individual buildings have been detected and localised with bounding box annotations, with confidence scores ranging from 0.71 to 0.96.',
  confidence: 0.87,
  boundingBoxes: [
    { x: 0.08, y: 0.12, width: 0.04, height: 0.03, label: 'Building', confidence: 0.94 },
    { x: 0.14, y: 0.09, width: 0.05, height: 0.04, label: 'Building', confidence: 0.91 },
    { x: 0.22, y: 0.15, width: 0.03, height: 0.03, label: 'Building', confidence: 0.88 },
    { x: 0.31, y: 0.08, width: 0.06, height: 0.05, label: 'Building', confidence: 0.85 },
    { x: 0.42, y: 0.11, width: 0.04, height: 0.03, label: 'Building', confidence: 0.83 },
    { x: 0.55, y: 0.18, width: 0.03, height: 0.04, label: 'Building', confidence: 0.79 },
    { x: 0.11, y: 0.31, width: 0.05, height: 0.04, label: 'Building', confidence: 0.76 },
    { x: 0.63, y: 0.24, width: 0.04, height: 0.03, label: 'Building', confidence: 0.74 },
    { x: 0.18, y: 0.42, width: 0.06, height: 0.05, label: 'Water Body', confidence: 0.96 },
    { x: 0.72, y: 0.35, width: 0.05, height: 0.04, label: 'Vegetation', confidence: 0.82 },
  ],
  changeMap: null,
  toolInvocations: [
    {
      toolId: 'vqa-model-v1',
      toolName: 'RSVQA Vision-Language Model',
      version: '1.2.0-mock',
      taskType: 'vqa',
      parameters: { temperature: 0.3, max_tokens: 512, beam_size: 4 },
      processingTimeMs: 2340,
    },
    {
      toolId: 'grounding-model-v1',
      toolName: 'RS-DINO Grounding Detector',
      version: '1.0.3-mock',
      taskType: 'grounding',
      parameters: { confidence_threshold: 0.7, nms_threshold: 0.45, tile_size: 512 },
      processingTimeMs: 4120,
    },
  ],
  executionTrace: {
    overallStatus: 'completed',
    totalElapsedMs: 7840,
    steps: [
      {
        id: 'step-1', title: 'Query Received',
        detail: 'Query accepted: "What land cover types are visible and locate all buildings?" | Mode: single_image | File: RESOURCESAT-2_LISS-IV_2024-03-15.tif',
        status: 'done', startedAt: new Date(Date.now() - 7840).toISOString(), completedAt: new Date(Date.now() - 7600).toISOString(),
      },
      {
        id: 'step-2', title: 'Input Validation',
        detail: 'File format: GeoTIFF ✓ | Modality: Optical (confidence 94%) ✓ | CRS present: EPSG:32643 ✓ | Dimensions: 8192×8192 px ✓ | Band count: 3 ✓',
        status: 'done', startedAt: new Date(Date.now() - 7600).toISOString(), completedAt: new Date(Date.now() - 7100).toISOString(),
      },
      {
        id: 'step-3', title: 'Task Classification',
        detail: 'Detected intents: [VQA: 0.91] [Grounding/Object Detection: 0.88] | Primary: VQA | Secondary: Grounding',
        status: 'done', startedAt: new Date(Date.now() - 7100).toISOString(), completedAt: new Date(Date.now() - 6800).toISOString(),
        meta: { vqa_confidence: 0.91, grounding_confidence: 0.88 },
      },
      {
        id: 'step-4', title: 'Tool Selection',
        detail: 'VQA task → RSVQA Vision-Language Model v1.2.0 | Grounding task → RS-DINO Grounding Detector v1.0.3',
        status: 'done', startedAt: new Date(Date.now() - 6800).toISOString(), completedAt: new Date(Date.now() - 6500).toISOString(),
      },
      {
        id: 'step-5', title: 'Parameters',
        detail: 'VQA: temperature=0.3, max_tokens=512, beam_size=4 | Grounding: confidence_threshold=0.70, nms_threshold=0.45, tile_size=512px',
        status: 'done', startedAt: new Date(Date.now() - 6500).toISOString(), completedAt: new Date(Date.now() - 6300).toISOString(),
        meta: { temperature: 0.3, max_tokens: 512, confidence_threshold: 0.7, tile_size: 512 },
      },
      {
        id: 'step-6', title: 'Processing',
        detail: 'RSVQA VLM: inference complete (2340ms) | RS-DINO: 127 objects detected across 16 tiles (4120ms)',
        status: 'done', startedAt: new Date(Date.now() - 6300).toISOString(), completedAt: new Date(Date.now() - 980).toISOString(),
        meta: { vlm_ms: 2340, detector_ms: 4120, objects_detected: 127 },
      },
      {
        id: 'step-7', title: 'Aggregation',
        detail: 'Merging VQA answer with grounding annotations | NMS applied across tiles | Confidence calibration applied',
        status: 'done', startedAt: new Date(Date.now() - 980).toISOString(), completedAt: new Date(Date.now() - 400).toISOString(),
      },
      {
        id: 'step-8', title: 'Completion',
        detail: 'Analysis complete | Overall confidence: 87.0% | Total elapsed: 7.84s | 127 bounding boxes generated',
        status: 'done', startedAt: new Date(Date.now() - 400).toISOString(), completedAt: new Date(Date.now() - 100).toISOString(),
        meta: { confidence: 0.87, elapsed_ms: 7840, boxes: 127 },
      },
    ],
  },
  errorReason: null,
};
