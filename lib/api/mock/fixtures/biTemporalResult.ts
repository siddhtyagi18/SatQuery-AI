// lib/api/mock/fixtures/biTemporalResult.ts
import type { AnalysisResult } from '@/lib/types/analysis';

export const biTemporalResult: AnalysisResult = {
  id: 'analysis-002',
  mode: 'bi_temporal',
  query: 'What changes occurred between these two dates? Has urban expansion affected vegetation cover?',
  status: 'completed',
  createdAt: new Date(Date.now() - 5 * 60 * 60 * 1000).toISOString(),
  images: [
    {
      id: 'img-002a',
      role: 'before',
      previewUrl: '/demo/optical_before.jpg',
      metadata: {
        fileName: 'CARTOSAT-3_PAN_2022-01-10_T1.tif',
        fileFormat: 'GeoTIFF',
        modality: 'optical',
        modalityDetectionConfidence: 0.97,
        acquisitionDate: '2022-01-10',
        widthPx: 4096,
        heightPx: 4096,
        bandCount: 1,
        crs: 'EPSG:32643',
        gsdMeters: 0.25,
        fileSizeBytes: 33554432,
      },
    },
    {
      id: 'img-002b',
      role: 'after',
      previewUrl: '/demo/optical_after.jpg',
      metadata: {
        fileName: 'CARTOSAT-3_PAN_2024-01-08_T2.tif',
        fileFormat: 'GeoTIFF',
        modality: 'optical',
        modalityDetectionConfidence: 0.97,
        acquisitionDate: '2024-01-08',
        widthPx: 4096,
        heightPx: 4096,
        bandCount: 1,
        crs: 'EPSG:32643',
        gsdMeters: 0.25,
        fileSizeBytes: 33554432,
      },
    },
  ],
  detectedTasks: ['change_detection', 'change_vqa', 'change_description'],
  answerText:
    'Significant land-use change is detected across the 2-year temporal baseline (Jan 2022 → Jan 2024). Key findings: (1) Urban built-up area increased by approximately 23.4%, with new construction identified predominantly in the northeastern corridor — 47 new building clusters detected; (2) Vegetation/green cover decreased by 18.7%, corresponding to an estimated 2.3 km² of cleared land; (3) A new road infrastructure corridor (~1.4 km linear feature) was constructed bisecting previously agricultural land; (4) Agricultural parcels in the southern sector show no significant change. The changes are consistent with planned urban expansion in this region. Change confidence: High (82%).',
  confidence: 0.82,
  boundingBoxes: null,
  changeMap: {
    overlayUrl: '/demo/change_mask.png',
    legend: [
      { label: 'New Urban / Built-up', color: '#FF5C5C' },
      { label: 'Vegetation Loss', color: '#FFB020' },
      { label: 'New Infrastructure', color: '#3ED0FF' },
      { label: 'No Change', color: '#1A2340' },
    ],
  },
  toolInvocations: [
    {
      toolId: 'change-det-v1',
      toolName: 'Bi-temporal Change Detection Model',
      version: '2.1.0-mock',
      taskType: 'change_detection',
      parameters: { algorithm: 'CVA+NDVI', threshold: 0.15, co_registration: true, band_combination: 'RGB' },
      processingTimeMs: 5670,
    },
    {
      toolId: 'change-vqa-v1',
      toolName: 'Change-VQA Language Model',
      version: '1.1.0-mock',
      taskType: 'change_vqa',
      parameters: { temperature: 0.2, max_tokens: 768 },
      processingTimeMs: 3120,
    },
  ],
  executionTrace: {
    overallStatus: 'completed',
    totalElapsedMs: 10240,
    steps: [
      {
        id: 'step-1', title: 'Query Received',
        detail: 'Query: "What changes occurred...?" | Mode: bi_temporal | T1: CARTOSAT-3_PAN_2022-01-10 | T2: CARTOSAT-3_PAN_2024-01-08',
        status: 'done', startedAt: new Date(Date.now() - 10240).toISOString(), completedAt: new Date(Date.now() - 9900).toISOString(),
      },
      {
        id: 'step-2', title: 'Input Validation',
        detail: 'T1 format: GeoTIFF ✓ | T2 format: GeoTIFF ✓ | CRS match: EPSG:32643 ✓ | Co-registration check: PASSED (RMSE < 0.5px) ✓ | Temporal baseline: 24 months ✓',
        status: 'done', startedAt: new Date(Date.now() - 9900).toISOString(), completedAt: new Date(Date.now() - 9200).toISOString(),
      },
      {
        id: 'step-3', title: 'Task Classification',
        detail: 'Detected intents: [Change Detection: 0.95] [Change VQA: 0.89] [Change Description: 0.84]',
        status: 'done', startedAt: new Date(Date.now() - 9200).toISOString(), completedAt: new Date(Date.now() - 8800).toISOString(),
      },
      {
        id: 'step-4', title: 'Tool Selection',
        detail: 'Change Detection → Bi-temporal Change Detection Model v2.1.0 | Semantic Interpretation → Change-VQA LM v1.1.0',
        status: 'done', startedAt: new Date(Date.now() - 8800).toISOString(), completedAt: new Date(Date.now() - 8500).toISOString(),
      },
      {
        id: 'step-5', title: 'Parameters',
        detail: 'algorithm=CVA+NDVI | threshold=0.15 | co_registration=true | band_combination=RGB | temporal_baseline=24mo',
        status: 'done', startedAt: new Date(Date.now() - 8500).toISOString(), completedAt: new Date(Date.now() - 8200).toISOString(),
        meta: { algorithm: 'CVA+NDVI', threshold: 0.15, co_registration: true },
      },
      {
        id: 'step-6', title: 'Processing',
        detail: 'CVA+NDVI change detection: 5670ms | Change mask generated: 3 classes | Change-VQA inference: 3120ms',
        status: 'done', startedAt: new Date(Date.now() - 8200).toISOString(), completedAt: new Date(Date.now() - 1200).toISOString(),
        meta: { change_pixels_pct: 23.4, vegetation_loss_pct: 18.7 },
      },
      {
        id: 'step-7', title: 'Aggregation',
        detail: 'Fusing change map with semantic description | Generating change legend | Computing per-class statistics',
        status: 'done', startedAt: new Date(Date.now() - 1200).toISOString(), completedAt: new Date(Date.now() - 400).toISOString(),
      },
      {
        id: 'step-8', title: 'Completion',
        detail: 'Analysis complete | Overall confidence: 82.0% | Total elapsed: 10.24s | 3 change classes detected',
        status: 'done', startedAt: new Date(Date.now() - 400).toISOString(), completedAt: new Date(Date.now() - 80).toISOString(),
        meta: { confidence: 0.82, elapsed_ms: 10240, change_classes: 3 },
      },
    ],
  },
  errorReason: null,
};
