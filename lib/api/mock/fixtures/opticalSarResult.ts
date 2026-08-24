// lib/api/mock/fixtures/opticalSarResult.ts
import type { AnalysisResult } from '@/lib/types/analysis';

export const opticalSarResult: AnalysisResult = {
  id: 'analysis-003',
  mode: 'optical_sar',
  query: 'Does the SAR data confirm the optical change detection? Identify features visible in SAR but not optical.',
  status: 'completed',
  createdAt: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
  images: [
    {
      id: 'img-003a',
      role: 'optical',
      previewUrl: '/demo/optical_sample.jpg',
      metadata: {
        fileName: 'RESOURCESAT-2A_LISS-IV_2024-02-20.tif',
        fileFormat: 'GeoTIFF',
        modality: 'multispectral',
        modalityDetectionConfidence: 0.91,
        acquisitionDate: '2024-02-20',
        widthPx: 6144,
        heightPx: 6144,
        bandCount: 4,
        crs: 'EPSG:32644',
        gsdMeters: 5.8,
        fileSizeBytes: 50331648,
      },
    },
    {
      id: 'img-003b',
      role: 'sar',
      previewUrl: '/demo/sar_sample.jpg',
      metadata: {
        fileName: 'RISAT-1A_SAR_C-band_VV_2024-02-22.tif',
        fileFormat: 'GeoTIFF',
        modality: 'sar',
        modalityDetectionConfidence: 0.96,
        acquisitionDate: '2024-02-22',
        widthPx: 6144,
        heightPx: 6144,
        bandCount: 1,
        crs: 'EPSG:32644',
        gsdMeters: 3.0,
        fileSizeBytes: 25165824,
      },
    },
  ],
  detectedTasks: ['vqa', 'change_detection'],
  answerText:
    'Cross-modal fusion analysis confirms and extends the optical change detection. Key findings: (1) SAR CONFIRMS optical detections: The urban expansion signal in the northeastern sector is corroborated by SAR backscatter increase (mean σ⁰ increase: +4.2 dB), consistent with new construction (double-bounce scattering mechanism); (2) SAR-UNIQUE features: Three sub-canopy linear structures (likely buried infrastructure or drainage channels) are detectable only in the SAR C-band VV channel — invisible in optical imagery due to surface vegetation; (3) A flooded agricultural parcel (7.3 ha) in the western sector shows specular reflection (near-zero backscatter) in SAR, confirming standing water not easily discerned in optical; (4) No contradictions between modalities found. Cross-modal confidence: Very High (91%).',
  confidence: 0.91,
  boundingBoxes: [
    { x: 0.15, y: 0.22, width: 0.08, height: 0.06, label: 'SAR-only: Sub-canopy Structure', confidence: 0.78 },
    { x: 0.62, y: 0.45, width: 0.12, height: 0.09, label: 'Flooded Parcel (SAR confirmed)', confidence: 0.93 },
    { x: 0.38, y: 0.18, width: 0.07, height: 0.05, label: 'Urban Expansion (confirmed)', confidence: 0.89 },
  ],
  changeMap: null,
  toolInvocations: [
    {
      toolId: 'sar-optical-fusion-v1',
      toolName: 'SAR-Optical Cross-Modal Fusion Engine',
      version: '1.3.0-mock',
      taskType: 'vqa',
      parameters: { polarisation: 'VV', fusion_method: 'weighted_stack', alignment: 'phase_correlation' },
      processingTimeMs: 6890,
    },
    {
      toolId: 'sar-vqa-v1',
      toolName: 'Multi-modal VQA Head',
      version: '1.0.2-mock',
      taskType: 'vqa',
      parameters: { temperature: 0.25, modalities: 'optical+sar' },
      processingTimeMs: 3450,
    },
  ],
  executionTrace: {
    overallStatus: 'completed',
    totalElapsedMs: 12100,
    steps: [
      {
        id: 'step-1', title: 'Query Received',
        detail: 'Query: "Does the SAR data confirm...?" | Mode: optical_sar | Optical: LISS-IV 2024-02-20 | SAR: RISAT-1A 2024-02-22',
        status: 'done', startedAt: new Date(Date.now() - 12100).toISOString(), completedAt: new Date(Date.now() - 11700).toISOString(),
      },
      {
        id: 'step-2', title: 'Input Validation',
        detail: 'Optical: GeoTIFF ✓ | SAR: GeoTIFF ✓ | Modality match: Optical+SAR ✓ | CRS match: EPSG:32644 ✓ | Acquisition delta: 2 days ✓',
        status: 'done', startedAt: new Date(Date.now() - 11700).toISOString(), completedAt: new Date(Date.now() - 10900).toISOString(),
      },
      {
        id: 'step-3', title: 'Task Classification',
        detail: 'Detected intents: [Cross-modal VQA: 0.93] [SAR Feature Analysis: 0.87] | Multi-modal fusion required',
        status: 'done', startedAt: new Date(Date.now() - 10900).toISOString(), completedAt: new Date(Date.now() - 10400).toISOString(),
      },
      {
        id: 'step-4', title: 'Tool Selection',
        detail: 'Fusion → SAR-Optical Cross-Modal Fusion Engine v1.3.0 | Interpretation → Multi-modal VQA Head v1.0.2',
        status: 'done', startedAt: new Date(Date.now() - 10400).toISOString(), completedAt: new Date(Date.now() - 10000).toISOString(),
      },
      {
        id: 'step-5', title: 'Parameters',
        detail: 'polarisation=VV | fusion_method=weighted_stack | alignment=phase_correlation | modalities=optical+sar',
        status: 'done', startedAt: new Date(Date.now() - 10000).toISOString(), completedAt: new Date(Date.now() - 9700).toISOString(),
        meta: { polarisation: 'VV', fusion_method: 'weighted_stack' },
      },
      {
        id: 'step-6', title: 'Processing',
        detail: 'Phase correlation alignment: 1240ms | Fusion stack computation: 5650ms | Multi-modal VQA inference: 3450ms',
        status: 'done', startedAt: new Date(Date.now() - 9700).toISOString(), completedAt: new Date(Date.now() - 1000).toISOString(),
        meta: { sar_unique_features: 3, confirmed_changes: 1, contradictions: 0 },
      },
      {
        id: 'step-7', title: 'Aggregation',
        detail: 'Merging cross-modal detections | Generating SAR-unique feature annotations | Reconciling modality reports',
        status: 'done', startedAt: new Date(Date.now() - 1000).toISOString(), completedAt: new Date(Date.now() - 350).toISOString(),
      },
      {
        id: 'step-8', title: 'Completion',
        detail: 'Analysis complete | Overall confidence: 91.0% | Total elapsed: 12.10s | 3 cross-modal annotations',
        status: 'done', startedAt: new Date(Date.now() - 350).toISOString(), completedAt: new Date(Date.now() - 60).toISOString(),
        meta: { confidence: 0.91, elapsed_ms: 12100 },
      },
    ],
  },
  errorReason: null,
};
