// lib/api/index.ts
//
// ============================================================
// HOW TO SWAP TO A REAL FASTAPI BACKEND
// ============================================================
// 1. Change `API_MODE` in `lib/config.ts` from 'mock' to 'live'
// 2. Set the environment variable NEXT_PUBLIC_API_URL to your
//    FastAPI base URL, e.g.:
//      NEXT_PUBLIC_API_URL=https://satquery-ai-api.isro.gov.in
// 3. Implement `liveApi` below conforming to `SatQueryApi` by
//    replacing each method with a fetch() call to the FastAPI
//    endpoint. The response shapes MUST match the types in
//    lib/types/analysis.ts exactly.
//    Example FastAPI endpoints to implement:
//      POST   /api/upload             → uploadImage()
//      POST   /api/analysis           → submitAnalysis()
//      GET    /api/analysis/{id}      → getAnalysis()
//      GET    /api/analysis/{id}/stream (SSE) → streamExecutionTrace()
//      GET    /api/analysis?page=&mode=... → listAnalysisHistory()
//      DELETE /api/analysis/{id}      → deleteAnalysis()
//      GET    /api/tools              → listTools()
//      GET    /api/benchmark          → getBenchmarkMetrics()
// ============================================================

import type {
  AnalysisResult,
  BenchmarkMetric,
  ExecutionTrace,
  HistoryFilters,
  SubmitAnalysisInput,
  ToolDefinition,
  UploadedImage,
} from '@/lib/types/analysis';
import { API_MODE } from '@/lib/config';
import { mockApi } from './mock/mockApi';
import { liveApi } from './liveApi';

export interface SatQueryApi {
  uploadImage(file: File, role: UploadedImage['role']): Promise<UploadedImage>;
  submitAnalysis(input: SubmitAnalysisInput): Promise<{ analysisId: string }>;
  getAnalysis(id: string): Promise<AnalysisResult>;
  /** Returns an unsubscribe function. Call it to stop listening. */
  streamExecutionTrace(id: string, onUpdate: (trace: ExecutionTrace) => void): () => void;
  listAnalysisHistory(filters: HistoryFilters): Promise<{ items: AnalysisResult[]; total: number }>;
  deleteAnalysis(id: string): Promise<void>;
  listTools(): Promise<ToolDefinition[]>;
  getBenchmarkMetrics(): Promise<BenchmarkMetric[]>;
}

// Automatically uses liveApi when API_MODE === 'live', mock otherwise.
export const api: SatQueryApi = API_MODE === 'live' ? liveApi : mockApi;

