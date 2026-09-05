import type {
  AnalysisResult,
  BenchmarkMetric,
  ExecutionStep,
  ExecutionTrace,
  HistoryFilters,
  SubmitAnalysisInput,
  ToolDefinition,
  UploadedImage,
} from '@/lib/types/analysis';
import type { SatQueryApi } from '@/lib/api';
import { supabaseAnalysisService, SUPABASE_PERSISTENCE_ENABLED } from '@/lib/supabase/services';
import { getCurrentUserId } from '@/lib/authService';

const pendingUploads = new Map<string, { file: File; role: UploadedImage['role'] }>();
const traceStepSeen = new Map<string, Set<string>>();
const submittedContext = new Map<
  string,
  { mode: SubmitAnalysisInput['mode']; query: string; imageIds: string[]; userId: string | null }
>();

async function resolveUserId(): Promise<string | null> {
  try {
    return await getCurrentUserId();
  } catch {
    return null;
  }
}

function getResultByIdFromStore(store: SatQueryApi, id: string): Promise<AnalysisResult> {
  return store.getAnalysis(id);
}

async function mergeHistoryLists(
  fromSupabase: AnalysisResult[],
  fromStore: AnalysisResult[]
): Promise<{ items: AnalysisResult[]; total: number }> {
  const byId = new Map<string, AnalysisResult>();
  for (const item of fromSupabase) byId.set(item.id, item);
  for (const item of fromStore) {
    if (!byId.has(item.id)) byId.set(item.id, item);
  }
  const sorted = Array.from(byId.values()).sort(
    (a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
  );
  return { items: sorted, total: sorted.length };
}

export function createPersistedApi(underlying: SatQueryApi): SatQueryApi {
  const service = SUPABASE_PERSISTENCE_ENABLED ? supabaseAnalysisService : null;

  const enabled = service !== null;

  const uploadImage: SatQueryApi['uploadImage'] = async (file, role) => {
    const result = await underlying.uploadImage(file, role);
    if (!enabled) return result;
    pendingUploads.set(result.id, { file, role });
    return result;
  };

  const submitAnalysis: SatQueryApi['submitAnalysis'] = async (input) => {
    const result = await underlying.submitAnalysis(input);
    if (!enabled) return result;
    try {
      const userId = await resolveUserId();
      const analysisId = result.analysisId;
      const images: UploadedImage[] = [];
      for (const iid of input.imageIds) {
        const up = pendingUploads.get(iid);
        if (up) {
          try {
            const fmt = (file: File): UploadedImage['metadata']['fileFormat'] =>
              file.type.includes('png')
                ? 'PNG'
                : file.type.includes('jpeg') || file.type.includes('jpg')
                ? 'JPEG'
                : file.name.toLowerCase().endsWith('.tif') || file.name.toLowerCase().endsWith('.tiff')
                ? 'GeoTIFF'
                : 'PNG';
            const modality: UploadedImage['metadata']['modality'] =
              up.role === 'sar' ? 'sar' : 'optical';
            const md: UploadedImage['metadata'] = {
              fileName: up.file.name,
              fileFormat: fmt(up.file),
              modality,
              modalityDetectionConfidence: null,
              acquisitionDate: null,
              widthPx: null,
              heightPx: null,
              bandCount: null,
              crs: null,
              gsdMeters: null,
              fileSizeBytes: up.file.size,
            };
            images.push({
              id: iid,
              role: up.role,
              previewUrl: null,
              metadata: md,
            });
            if (userId && service) {
              void service.uploadInputFile({
                userId,
                analysisId,
                role: up.role,
                file: up.file,
              });
            }
          } catch {
            // noop
          }
        }
      }

      submittedContext.set(analysisId, {
        mode: input.mode,
        query: input.query,
        imageIds: input.imageIds,
        userId,
      });

      if (userId) {
        await service.saveAnalysisStarted({
          analysisId,
          userId,
          query: input.query,
          analysisType: input.mode,
          images,
        });
      }
    } catch (err) {
      console.warn('[persistenceBridge] submitAnalysis side-effect failed:', err);
    }
    return result;
  };

  const getAnalysis: SatQueryApi['getAnalysis'] = async (id) => {
    if (!enabled) return underlying.getAnalysis(id);
    try {
      const userId = await resolveUserId();
      const sb = userId ? await service.getAnalysis(id, userId) : null;
      if (sb) return sb;
    } catch (err) {
      console.warn('[persistenceBridge] getAnalysis supabase failed, falling back:', err);
    }
    return underlying.getAnalysis(id);
  };

  const streamExecutionTrace: SatQueryApi['streamExecutionTrace'] = (id, onUpdate) => {
    const seen = traceStepSeen.get(id) ?? new Set<string>();
    traceStepSeen.set(id, seen);

    const saveStep = async (stepIdx: number, step: ExecutionStep) => {
      if (!enabled || !service) return;
      const key = `${stepIdx}:${step.status}`;
      if (seen.has(key)) return;
      seen.add(key);
      try {
        await service.saveTraceStep({
          analysisId: id,
          stepIndex: stepIdx,
          step,
        });
      } catch (err) {
        console.warn('[persistenceBridge] saveTraceStep failed:', err);
      }
    };

    let finalised = false;
    const finaliseOnce = async (trace: ExecutionTrace) => {
      if (finalised) return;
      finalised = true;
      try {
        const res = await getResultByIdFromStore(underlying, id);
        if (enabled && service) {
          const ctx = submittedContext.get(id);
          const userId = ctx?.userId ?? (await resolveUserId());
          if (userId) {
            await service.saveAnalysisCompleted({
              analysisId: id,
              result: res,
            });
          }
        }
      } catch (err) {
        console.warn('[persistenceBridge] saveAnalysisCompleted failed:', err);
      }
    };

    const unsubscribe = underlying.streamExecutionTrace(id, (trace) => {
      if (enabled && service) {
        trace.steps.forEach((step, i) => {
          if (step.status === 'done' || step.status === 'in_progress' || step.status === 'error') {
            void saveStep(i, step);
          }
        });
      }
      onUpdate(trace);
      if (trace.overallStatus === 'completed' || trace.overallStatus === 'failed') {
        void finaliseOnce(trace);
      }
    });

    return () => {
      unsubscribe();
      traceStepSeen.delete(id);
    };
  };

  const listAnalysisHistory: SatQueryApi['listAnalysisHistory'] = async (filters) => {
    if (!enabled) return underlying.listAnalysisHistory(filters);
    try {
      const userId = await resolveUserId();
      if (userId) {
        const page = filters.page ?? 1;
        const pageSize = filters.pageSize ?? 20;
        const sb = await service.listHistory(filters, userId);
        const st = await underlying.listAnalysisHistory(filters);
        const merged = await mergeHistoryLists(sb.items, st.items);
        const pagedItems = merged.items.slice((page - 1) * pageSize, page * pageSize);
        return { items: pagedItems, total: merged.total };
      }
    } catch (err) {
      console.warn('[persistenceBridge] listAnalysisHistory supabase failed, falling back:', err);
    }
    return underlying.listAnalysisHistory(filters);
  };

  const deleteAnalysis: SatQueryApi['deleteAnalysis'] = async (id) => {
    const tasks: Promise<unknown>[] = [underlying.deleteAnalysis(id)];
    if (enabled && service) {
      tasks.push(
        (async () => {
          try {
            const userId = await resolveUserId();
            if (userId) await service.deleteAnalysis(id, userId);
          } catch (err) {
            console.warn('[persistenceBridge] deleteAnalysis supabase side-effect failed:', err);
          }
        })()
      );
    }
    await Promise.all(tasks);
  };

  const listTools: SatQueryApi['listTools'] = () => underlying.listTools();
  const getBenchmarkMetrics: SatQueryApi['getBenchmarkMetrics'] = () => underlying.getBenchmarkMetrics();

  return {
    uploadImage,
    submitAnalysis,
    getAnalysis,
    streamExecutionTrace,
    listAnalysisHistory,
    deleteAnalysis,
    listTools,
    getBenchmarkMetrics,
  };
}
