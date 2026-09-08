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
    '### Bi-Temporal Scene Change Interpretation\n\n' +
    '**Qualitative Visual Interpretation (VLM):**\n' +
    'Comparing the earlier acquisition (T1 / Left) with the later acquisition (T2 / Right), localized structural changes are observable in the scene. Engineered building structures and ground clearing activities appear along transit access boundaries.\n\n' +
    '**Quantitative Detection Telemetry:**\n' +
    '- **Detected Changed Area:** `3.14%` (Severity: **low**)\n' +
    '- **Change Detection Threshold:** `0.70` (Siamese U-Net)\n' +
    '- **Detector Model:** SiameseUNet (~490K parameters, LEVIR-CD trained checkpoint)\n' +
    '- **Confidence:** Not calibrated for this bi-temporal analysis (confidence = null)\n' +
    '- **Inference Provenance:** Vision-Language Model (SmolVLM-500M-Instruct + LoRA domain adapter)',
  confidence: null,
  isMock: false,
  executionMode: 'real',
  boundingBoxes: null,
  changeMap: {
    overlayUrl: '/demo/change_mask.png',
    legend: [
      { label: 'Detected Changed Region (3.1% area)', color: '#FF3C3C' },
      { label: 'Unchanged Region (96.9% area)', color: 'transparent' },
    ],
    analytics: {
      global_statistics: {
        width: 1024,
        height: 768,
        total_pixel_count: 786432,
        changed_pixel_count: 24694,
        unchanged_pixel_count: 761738,
        changed_percentage: 3.14,
        unchanged_percentage: 96.86,
        changed_percentage_rounded: 3.14,
        unchanged_percentage_rounded: 96.86,
      },
      physical_area: {
        physical_area_available: false,
        area_unavailable_reason: 'Physical area unavailable because reliable spatial resolution metadata was not provided.',
        pixel_area_m2: null,
        gsd_meters: null,
        changed_area_m2: null,
        changed_area_ha: null,
        changed_area_km2: null,
        total_area_m2: null,
        formatted_summary: 'N/A — Spatial metadata unavailable',
      },
      geospatial_metadata: {
        crs: null,
        epsg_code: null,
        resolution: null,
        image_bounds: null,
        transform: null,
        width: 1024,
        height: 768,
      },
      hotspots_count_total: 12,
      hotspots_count_filtered: 8,
      min_hotspot_size_applied: 10,
      largest_change_region: {
        hotspot_id: 1,
        rank: 1,
        pixel_area: 5820,
        percentage_of_total_changed: 23.57,
        percentage_of_total_changed_rounded: 23.57,
        bounding_box: { x_min: 140, y_min: 210, x_max: 290, y_max: 330, width: 150, height: 120 },
        centroid_px: { x: 215.4, y: 270.8 },
      },
      change_density: {
        quadrant_density: {
          northwest: 4.82,
          northeast: 1.15,
          southwest: 5.21,
          southeast: 1.38,
        },
        peak_density_quadrant: 'southwest',
        mean_change_density_pct: 3.14,
      },
    },
  },
  toolInvocations: [
    {
      toolId: 'change_detector',
      toolName: 'Bi-temporal Change Detector',
      version: '0.3.0-p0',
      taskType: 'change_detection',
      parameters: { algorithm: 'siamese-unet-model', threshold: 0.70, checkpoint: 'experiment_01/best_model.pt' },
      processingTimeMs: 4120,
      executionMode: 'real',
    },
    {
      toolId: 'change_vqa',
      toolName: 'Change-VQA Language Model',
      version: '0.3.0-p0',
      taskType: 'change_vqa',
      parameters: { model: 'SmolVLM-500M-Instruct', lora: 'vqa_lora_experiment_01/best' },
      processingTimeMs: 2840,
      executionMode: 'real',
    },
  ],
  executionTrace: {
    overallStatus: 'completed',
    totalElapsedMs: 6960,
    steps: [
      {
        id: 'step-1', title: 'Query Received',
        detail: 'Query: "What changes occurred...?" | Mode: bi_temporal | T1: CARTOSAT-3_PAN_2022-01-10 | T2: CARTOSAT-3_PAN_2024-01-08',
        status: 'done', startedAt: new Date(Date.now() - 6960).toISOString(), completedAt: new Date(Date.now() - 6700).toISOString(),
      },
      {
        id: 'step-2', title: 'Input Validation',
        detail: 'T1 format: GeoTIFF ✓ | T2 format: GeoTIFF ✓ | CRS match: EPSG:32643 ✓ | Co-registration check: PASSED (RMSE < 0.5px) ✓ | Temporal baseline: 24 months ✓',
        status: 'done', startedAt: new Date(Date.now() - 6700).toISOString(), completedAt: new Date(Date.now() - 6100).toISOString(),
      },
      {
        id: 'step-3', title: 'Task Classification',
        detail: 'Detected intents: [Change Detection: 0.95] [Change VQA: 0.89] [Change Description: 0.84]',
        status: 'done', startedAt: new Date(Date.now() - 6100).toISOString(), completedAt: new Date(Date.now() - 5800).toISOString(),
      },
      {
        id: 'step-4', title: 'Tool Selection',
        detail: 'Change Detection → Bi-temporal Change Detector v0.3.0-p0 | Semantic Interpretation → Change-VQA LM v0.3.0-p0',
        status: 'done', startedAt: new Date(Date.now() - 5800).toISOString(), completedAt: new Date(Date.now() - 5500).toISOString(),
      },
      {
        id: 'step-5', title: 'Parameters',
        detail: 'model=SiameseUNet | threshold=0.70 | checkpoint=experiment_01/best_model.pt | VLM=SmolVLM+LoRA',
        status: 'done', startedAt: new Date(Date.now() - 5500).toISOString(), completedAt: new Date(Date.now() - 5200).toISOString(),
        meta: { algorithm: 'siamese-unet-model', threshold: 0.70, checkpoint: 'experiment_01/best_model.pt' },
      },
      {
        id: 'step-6', title: 'Processing',
        detail: 'SiameseUNet inference: 4120ms | Change mask generated at threshold 0.70 | Change-VQA inference: 2840ms',
        status: 'done', startedAt: new Date(Date.now() - 5200).toISOString(), completedAt: new Date(Date.now() - 800).toISOString(),
        meta: { changed_pixel_pct: 3.14, threshold: 0.70 },
      },
      {
        id: 'step-7', title: 'Aggregation',
        detail: 'Fusing change map with semantic description | Generating change legend | Computing per-class statistics',
        status: 'done', startedAt: new Date(Date.now() - 800).toISOString(), completedAt: new Date(Date.now() - 200).toISOString(),
      },
      {
        id: 'step-8', title: 'Completion',
        detail: 'Analysis complete | Confidence: uncalibrated (null) | Total elapsed: 6.96s',
        status: 'done', startedAt: new Date(Date.now() - 200).toISOString(), completedAt: new Date(Date.now() - 50).toISOString(),
        meta: { confidence: null, elapsed_ms: 6960 },
      },
    ],
  },
  errorReason: null,
  multilingualSummaries: {
    language: 'en',
    summary_en:
      'Comparing the earlier acquisition (T1) with the later acquisition (T2), localized structural changes are observable in the scene. The quantitative analysis indicates a detected changed area of 3.14% along transit access boundaries.',
    summary_hi:
      'पहले (T1) और बाद (T2) की उपग्रह तस्वीरों की तुलना करने पर, इस क्षेत्र में कुछ जगहों पर संरचनात्मक बदलाव दिखाई दे रहे हैं। बदला हुआ क्षेत्र: 3.14% है, जो रास्तों और नए निर्माण के पास देखा गया है।',
    bullet_en: [
      'Localized structural changes are observable in the scene.',
      'Detected Changed Area: 3.14% across the temporal pair.',
      'Engineered building structures appear along transit access boundaries.',
    ],
    bullet_hi: [
      'इस क्षेत्र में कुछ जगहों पर संरचनात्मक बदलाव दिखाई दे रहे हैं।',
      'बदला हुआ क्षेत्र: 3.14% (कम फैलाव)।',
      'रास्तों के किनारे नए भवन और निर्माण कार्य पहचाने गए हैं।',
    ],
    generated_via_llm: true,
  },
};
