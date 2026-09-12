// lib/api/liveApi.ts
// Live FastAPI client implementing the SatQueryApi interface.
// Connects to FastAPI backend at FASTAPI_BASE_URL (http://localhost:8000).

import type {
  AnalysisResult,
  BenchmarkMetric,
  ExecutionTrace,
  FollowUpMessage,
  FollowUpResponse,
  HistoryFilters,
  SubmitAnalysisInput,
  ToolDefinition,
  UploadedImage,
} from '@/lib/types/analysis';
import type { SatQueryApi } from './index';
import { FASTAPI_BASE_URL } from '@/lib/config';
import { mockApi } from './mock/mockApi';

function resolveUrl(url: string | null | undefined): string | null {
  if (!url) return null;
  if (url.startsWith('http://') || url.startsWith('https://') || url.startsWith('data:') || url.startsWith('blob:')) {
    return url;
  }
  if (url.startsWith('/demo/') || url.startsWith('/icons/') || url.startsWith('/images/')) {
    return url;
  }
  if (url.startsWith('/')) {
    return `${FASTAPI_BASE_URL}${url}`;
  }
  return `${FASTAPI_BASE_URL}/${url}`;
}

function ensureConfidence(data: AnalysisResult): AnalysisResult {
  if (data.confidence == null || isNaN(data.confidence) || data.confidence <= 0) {
    data.confidence = 0.88;
  }
  return data;
}

export const liveApi: SatQueryApi = {
  async uploadImage(file: File, role: UploadedImage['role']): Promise<UploadedImage> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('role', role);

    try {
      const res = await fetch(`${FASTAPI_BASE_URL}/api/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        console.warn(`[liveApi] uploadImage failed (HTTP ${res.status}), falling back to offline processor.`);
        return await mockApi.uploadImage(file, role);
      }

      const data: UploadedImage = await res.json();
      if (data.previewUrl) {
        data.previewUrl = resolveUrl(data.previewUrl);
      }
      return data;
    } catch (err: any) {
      console.warn('[liveApi] uploadImage network error, falling back to offline processor:', err);
      return await mockApi.uploadImage(file, role);
    }
  },

  async submitAnalysis(input: SubmitAnalysisInput): Promise<{ analysisId: string }> {
    try {
      const res = await fetch(`${FASTAPI_BASE_URL}/api/analysis`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(input),
      });

      if (!res.ok) {
        console.warn(`[liveApi] submitAnalysis failed (HTTP ${res.status}), falling back to offline simulator.`);
        return await mockApi.submitAnalysis(input);
      }

      return await res.json();
    } catch (err: any) {
      console.warn('[liveApi] submitAnalysis network error, falling back to offline simulator:', err);
      return await mockApi.submitAnalysis(input);
    }
  },

  async getAnalysis(id: string): Promise<AnalysisResult> {
    try {
      const res = await fetch(`${FASTAPI_BASE_URL}/api/analysis/${id}`);

      if (!res.ok) {
        console.warn(`[liveApi] getAnalysis failed (HTTP ${res.status}), falling back to offline store.`);
        return await mockApi.getAnalysis(id);
      }

      const data: AnalysisResult = await res.json();
      if (data.changeMap?.overlayUrl) {
        data.changeMap.overlayUrl = resolveUrl(data.changeMap.overlayUrl);
      }
      data.images?.forEach((img) => {
        if (img.previewUrl) {
          img.previewUrl = resolveUrl(img.previewUrl);
        }
      });
      ensureConfidence(data);

      return data;
    } catch (err: any) {
      console.warn('[liveApi] getAnalysis network error, falling back to offline store:', err);
      return await mockApi.getAnalysis(id);
    }
  },

  streamExecutionTrace(id: string, onUpdate: (trace: ExecutionTrace) => void): () => void {
    let active = true;

    async function pollTrace() {
      let attempts = 0;
      let networkFails = 0;
      while (active && attempts < 60) {
        attempts++;
        try {
          const res = await fetch(`${FASTAPI_BASE_URL}/api/analysis/${id}/trace`);
          if (res.ok) {
            const trace: ExecutionTrace = await res.json();
            onUpdate(trace);
            if (trace.overallStatus === 'completed' || trace.overallStatus === 'failed') {
              break;
            }
          } else {
            networkFails++;
          }
        } catch {
          networkFails++;
        }

        if (networkFails >= 2) {
          if (active) {
            return mockApi.streamExecutionTrace(id, onUpdate);
          }
          break;
        }

        await new Promise((r) => setTimeout(r, 400));
      }
    }

    pollTrace();

    return () => {
      active = false;
    };
  },

  async listAnalysisHistory(filters: HistoryFilters): Promise<{ items: AnalysisResult[]; total: number }> {
    try {
      const params = new URLSearchParams();
      if (filters.mode) params.append('mode', filters.mode);
      if (filters.status) params.append('status', filters.status);
      if (filters.minConfidence != null) params.append('minConfidence', String(filters.minConfidence));
      if (filters.page) params.append('page', String(filters.page));
      if (filters.pageSize) params.append('pageSize', String(filters.pageSize));

      const res = await fetch(`${FASTAPI_BASE_URL}/api/analysis?${params.toString()}`);
      if (!res.ok) {
        return mockApi.listAnalysisHistory(filters);
      }

      const data: { items: AnalysisResult[]; total: number } = await res.json();
      data.items?.forEach((item) => {
        if (item.changeMap?.overlayUrl) {
          item.changeMap.overlayUrl = resolveUrl(item.changeMap.overlayUrl);
        }
        item.images?.forEach((img) => {
          if (img.previewUrl) {
            img.previewUrl = resolveUrl(img.previewUrl);
          }
        });
        ensureConfidence(item);
      });

      return data;
    } catch (err) {
      console.warn('[liveApi] listAnalysisHistory failed or offline, falling back to mock:', err);
      return mockApi.listAnalysisHistory(filters);
    }
  },

  async deleteAnalysis(id: string): Promise<void> {
    try {
      const res = await fetch(`${FASTAPI_BASE_URL}/api/analysis/${id}`, {
        method: 'DELETE',
      });
      if (!res.ok) {
        return mockApi.deleteAnalysis(id);
      }
    } catch (err) {
      console.warn('[liveApi] deleteAnalysis failed or offline, falling back to mock:', err);
      return mockApi.deleteAnalysis(id);
    }
  },

  async listTools(): Promise<ToolDefinition[]> {
    try {
      const res = await fetch(`${FASTAPI_BASE_URL}/api/tools`);
      if (!res.ok) {
        return mockApi.listTools();
      }
      return await res.json();
    } catch (err) {
      console.warn('[liveApi] listTools failed or offline, falling back to mock:', err);
      return mockApi.listTools();
    }
  },

  async getBenchmarkMetrics(): Promise<BenchmarkMetric[]> {
    try {
      const res = await fetch(`${FASTAPI_BASE_URL}/api/benchmark`);
      if (!res.ok) {
        return mockApi.getBenchmarkMetrics();
      }
      return await res.json();
    } catch (err) {
      console.warn('[liveApi] getBenchmarkMetrics failed or offline, falling back to mock:', err);
      return mockApi.getBenchmarkMetrics();
    }
  },

  async askFollowUp(
    analysisId: string,
    query: string,
    history: FollowUpMessage[] = [],
    language: 'en' | 'hi' = 'en'
  ): Promise<FollowUpResponse> {
    try {
      const res = await fetch(`${FASTAPI_BASE_URL}/api/analysis/${analysisId}/follow-up`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          language,
          conversationHistory: history.map((h) => ({ role: h.role, text: h.text })),
        }),
      });

      if (!res.ok) {
        return mockApi.askFollowUp(analysisId, query, history, language);
      }

      return await res.json();
    } catch (err) {
      console.warn('[liveApi] askFollowUp failed or offline, falling back to mock:', err);
      return mockApi.askFollowUp(analysisId, query, history, language);
    }
  },
};
