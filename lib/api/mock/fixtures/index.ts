// lib/api/mock/fixtures/index.ts
export { singleImageResult } from './singleImageResult';
export { biTemporalResult } from './biTemporalResult';
export { opticalSarResult } from './opticalSarResult';
export { failedResult } from './failedResult';

import { singleImageResult } from './singleImageResult';
import { biTemporalResult } from './biTemporalResult';
import { opticalSarResult } from './opticalSarResult';
import { failedResult } from './failedResult';
import type { AnalysisResult } from '@/lib/types/analysis';

export const allFixtures: AnalysisResult[] = [
  singleImageResult,
  biTemporalResult,
  opticalSarResult,
  failedResult,
];
