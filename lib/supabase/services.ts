import { supabase, HAS_SUPABASE } from '@/lib/supabase';
import { DEMO_MODE } from '@/lib/config';
import type {
  AnalysisMode,
  AnalysisResult,
  AnalysisStatus,
  ExecutionStep,
  HistoryFilters,
  ImageMetadataType,
  UploadedImage,
  TaskType,
  ToolInvocation,
  BoundingBox,
  Modality,
} from '@/lib/types/analysis';

const STORAGE_BUCKETS = {
  inputs: 'satquery-inputs',
  results: 'satquery-results',
  reports: 'satquery-reports',
} as const;

type StorageBucket = typeof STORAGE_BUCKETS[keyof typeof STORAGE_BUCKETS];

export interface SupabaseAnalysisService {
  ensureProfileForUser(userId: string, email: string, fullName?: string): Promise<void>;

  saveAnalysisStarted(input: {
    analysisId: string;
    userId: string;
    query: string;
    analysisType: AnalysisMode;
    images: UploadedImage[];
  }): Promise<void>;

  saveTraceStep(input: {
    analysisId: string;
    stepIndex: number;
    step: ExecutionStep;
  }): Promise<void>;

  saveAnalysisCompleted(input: {
    analysisId: string;
    result: AnalysisResult;
  }): Promise<void>;

  listHistory(
    filters: HistoryFilters,
    userId: string
  ): Promise<{ items: AnalysisResult[]; total: number }>;

  getAnalysis(id: string, userId: string): Promise<AnalysisResult | null>;

  deleteAnalysis(id: string, userId: string): Promise<void>;

  uploadInputFile(args: {
    userId: string;
    analysisId: string;
    role: UploadedImage['role'];
    file: File;
  }): Promise<{ storagePath: string; signedUrl?: string }>;

  uploadResultBlob(args: {
    userId: string;
    analysisId: string;
    blobName: string;
    blob: Blob;
  }): Promise<{ storagePath: string }>;

  getSignedUrl(bucket: 'inputs' | 'results' | 'reports', path: string): Promise<string | null>;

  getCurrentUserId(): Promise<string | null>;
}

const ANALYSIS_ID_PREFIX = 'analysis-';

function normalizeAnalysisId(rawId: string): string {
  if (rawId.startsWith(ANALYSIS_ID_PREFIX)) {
    const tail = rawId.slice(ANALYSIS_ID_PREFIX.length);
    if (/^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$/.test(tail)) {
      return tail;
    }
    const padded = tail.padEnd(32, '0').slice(0, 32);
    return `${padded.slice(0, 8)}-${padded.slice(8, 12)}-${padded.slice(12, 16)}-${padded.slice(16, 20)}-${padded.slice(20, 32)}`;
  }
  return rawId;
}

function denormalizeAnalysisId(dbUuid: string, fallback: string): string {
  return fallback.startsWith(ANALYSIS_ID_PREFIX) ? fallback : `${ANALYSIS_ID_PREFIX}${dbUuid.replace(/-/g, '').slice(0, 9)}`;
}

function mapAnalysisType(mode: AnalysisMode): string {
  return mode;
}

function mapModality(mod: Modality): string {
  return mod;
}

function mapFileFormat(fmt: ImageMetadataType['fileFormat']): string {
  return fmt;
}

function mapStepStatus(s: ExecutionStep['status']): string {
  switch (s) {
    case 'pending':
      return 'pending';
    case 'in_progress':
      return 'in_progress';
    case 'done':
      return 'done';
    case 'error':
      return 'error';
    default:
      return 'pending';
  }
}

function reverseStepStatus(db: string): ExecutionStep['status'] {
  if (db === 'done') return 'done';
  if (db === 'in_progress') return 'in_progress';
  if (db === 'error') return 'error';
  return 'pending';
}

function sanitizeFilename(name: string): string {
  return name.replace(/[^a-zA-Z0-9._-]/g, '_');
}

function logWarn(ctx: string, err: unknown): void {
  try {
    console.warn(`[supabase-service] ${ctx} failed:`, err);
  } catch {
    // ignore
  }
}

class RealSupabaseAnalysisService implements SupabaseAnalysisService {
  async getCurrentUserId(): Promise<string | null> {
    try {
      if (!supabase) return null;
      const { data } = await supabase.auth.getUser();
      return data.user?.id ?? null;
    } catch (err) {
      logWarn('getCurrentUserId', err);
      return null;
    }
  }

  async ensureProfileForUser(userId: string, email: string, fullName?: string): Promise<void> {
    try {
      if (!supabase) return;
      const now = new Date().toISOString();
      const { error } = await supabase.from('profiles').upsert(
        {
          id: userId,
          email,
          full_name: fullName ?? null,
          role: 'operator',
          created_at: now,
        },
        { onConflict: 'id', ignoreDuplicates: true }
      );
      if (error) throw error;
    } catch (err) {
      logWarn('ensureProfileForUser', err);
      throw err;
    }
  }

  async saveAnalysisStarted(input: {
    analysisId: string;
    userId: string;
    query: string;
    analysisType: AnalysisMode;
    images: UploadedImage[];
  }): Promise<void> {
    try {
      if (!supabase) return;
      const uuid = normalizeAnalysisId(input.analysisId);
      const createdAt = new Date().toISOString();

      const { error: analysisErr } = await supabase.from('analyses').upsert(
        {
          id: uuid,
          user_id: input.userId,
          query: input.query,
          analysis_type: mapAnalysisType(input.analysisType),
          status: 'queued',
          answer: null,
          confidence: null,
          created_at: createdAt,
          completed_at: null,
        },
        { onConflict: 'id' }
      );
      if (analysisErr) throw analysisErr;

      for (const img of input.images) {
        const meta = img.metadata;
        const storagePath = meta?.fileSizeBytes
          ? `inputs/${input.userId}/${uuid}/${img.role}-${sanitizeFilename(meta.fileName)}`
          : `inputs/${input.userId}/${uuid}/${img.role}-${img.id}`;
        const { error: inputErr } = await supabase.from('analysis_inputs').insert({
          analysis_id: uuid,
          filename: meta?.fileName ?? `img-${img.role}`,
          modality: mapModality(meta?.modality ?? 'unknown'),
          format: meta ? mapFileFormat(meta.fileFormat) : 'unknown',
          acquisition_date: meta?.acquisitionDate ? meta.acquisitionDate.slice(0, 10) : null,
          sensor: meta?.fileName?.toLowerCase().includes('sentinel') ? 'Sentinel-2' : null,
          resolution: meta?.gsdMeters ?? null,
          crs: meta?.crs ?? null,
          storage_path: storagePath,
          metadata: {
            ...(meta ?? {}),
            role: img.role,
            imageId: img.id,
            previewUrl: img.previewUrl,
          },
          created_at: createdAt,
        });
        if (inputErr) throw inputErr;
      }
    } catch (err) {
      logWarn('saveAnalysisStarted', err);
    }
  }

  async saveTraceStep(input: {
    analysisId: string;
    stepIndex: number;
    step: ExecutionStep;
  }): Promise<void> {
    try {
      if (!supabase) return;
      const uuid = normalizeAnalysisId(input.analysisId);
      const step = Math.max(1, input.stepIndex + 1);
      const toolName = (input.step.title ?? 'pipeline-step').slice(0, 255);
      const parameters = (input.step.meta ?? {}) as Record<string, unknown>;
      const output = {
        detail: input.step.detail ?? null,
        startedAt: input.step.startedAt ?? null,
        completedAt: input.step.completedAt ?? null,
      } as Record<string, unknown>;

      const { error } = await supabase.from('analysis_trace').upsert(
        {
          analysis_id: uuid,
          step,
          tool_name: toolName,
          status: mapStepStatus(input.step.status),
          parameters,
          output,
          created_at: new Date().toISOString(),
        },
        { onConflict: 'analysis_id,step' }
      );
      if (error) throw error;
    } catch (err) {
      logWarn('saveTraceStep', err);
    }
  }

  async saveAnalysisCompleted(input: {
    analysisId: string;
    result: AnalysisResult;
  }): Promise<void> {
    try {
      if (!supabase) return;
      const uuid = normalizeAnalysisId(input.analysisId);
      const r = input.result;
      const now = new Date().toISOString();
      const terminalStatus: AnalysisStatus = r.status === 'failed' ? 'failed' : 'completed';

      const confidence = typeof r.confidence === 'number' ? r.confidence : null;

      const { error: analysisErr } = await supabase
        .from('analyses')
        .update({
          status: terminalStatus,
          answer: r.answerText ?? null,
          confidence,
          completed_at: now,
        })
        .eq('id', uuid);
      if (analysisErr) throw analysisErr;

      const evidence = (r.boundingBoxes ?? []).map((bb: BoundingBox) => ({
        x: bb.x,
        y: bb.y,
        width: bb.width,
        height: bb.height,
        label: bb.label,
        confidence: bb.confidence,
      }));

      const statistics: Record<string, unknown> = {};
      const step6 = r.executionTrace?.steps?.find((s) => s.id === 'step-6');
      if (step6?.meta) {
        for (const [k, v] of Object.entries(step6.meta)) {
          if (
            k.startsWith('changed') ||
            k.startsWith('unchanged') ||
            k.startsWith('total') ||
            k.startsWith('percent') ||
            k.startsWith('class') ||
            k === 'execution_mode'
          ) {
            statistics[k] = v;
          }
        }
      }

      const resultMetadata: Record<string, unknown> = {
        schemaVersion: 1,
        detectedTasks: (r.detectedTasks ?? []) as TaskType[],
        toolInvocations: (r.toolInvocations ?? []) as ToolInvocation[],
        createdAt: r.createdAt ?? now,
        errorReason: r.errorReason ?? null,
      };

      const { error: resultsErr } = await supabase.from('analysis_results').upsert(
        {
          analysis_id: uuid,
          answer: r.answerText ?? null,
          confidence,
          evidence,
          statistics,
          result_metadata: resultMetadata,
          change_map_path: r.changeMap?.overlayUrl ?? null,
          report_path: null,
          created_at: now,
        },
        { onConflict: 'analysis_id' }
      );
      if (resultsErr) throw resultsErr;

      if (Object.keys(statistics).length > 0) {
        const { error: updateErr } = await supabase
          .from('analysis_results')
          .update({ statistics })
          .eq('analysis_id', uuid);
        if (updateErr) throw updateErr;
      }
    } catch (err) {
      logWarn('saveAnalysisCompleted', err);
    }
  }

  async listHistory(
    filters: HistoryFilters,
    userId: string
  ): Promise<{ items: AnalysisResult[]; total: number }> {
    try {
      if (!supabase) return { items: [], total: 0 };

      let query = supabase
        .from('analyses')
        .select(
          `id, user_id, query, analysis_type, status, answer, confidence, created_at, completed_at`,
          { count: 'exact' }
        )
        .eq('user_id', userId)
        .order('created_at', { ascending: false });

      if (filters.mode) {
        query = query.eq('analysis_type', filters.mode);
      }
      if (filters.status) {
        query = query.eq('status', filters.status);
      }
      if (filters.minConfidence != null) {
        query = query.gte('confidence', filters.minConfidence);
      }

      const { data, count, error } = await query;
      if (error) throw error;
      if (!data) return { items: [], total: 0 };

      const page = filters.page ?? 1;
      const pageSize = filters.pageSize ?? 20;
      const total = count ?? data.length;

      const items: AnalysisResult[] = [];
      for (const row of data) {
        const detail = await this.getAnalysisInner(row.id, userId, row);
        if (detail) items.push(detail);
      }

      const paged = items.slice((page - 1) * pageSize, page * pageSize);
      return { items: paged, total };
    } catch (err) {
      logWarn('listHistory', err);
      return { items: [], total: 0 };
    }
  }

  private async getAnalysisInner(
    uuid: string,
    userId: string,
    analysisRow?: any
  ): Promise<AnalysisResult | null> {
    try {
      if (!supabase) return null;

      let row: any = analysisRow;
      if (!row) {
        const { data, error } = await supabase
          .from('analyses')
          .select('*')
          .eq('id', uuid)
          .eq('user_id', userId)
          .maybeSingle();
        if (error) throw error;
        row = data;
      }
      if (!row) return null;

      const { data: inputs, error: inputsErr } = await supabase
        .from('analysis_inputs')
        .select('*')
        .eq('analysis_id', uuid)
        .order('created_at', { ascending: true });
      if (inputsErr) throw inputsErr;

      const { data: traceRows, error: traceErr } = await supabase
        .from('analysis_trace')
        .select('*')
        .eq('analysis_id', uuid)
        .order('step', { ascending: true });
      if (traceErr) throw traceErr;

      const { data: resultRow, error: resultErr } = await supabase
        .from('analysis_results')
        .select('*')
        .eq('analysis_id', uuid)
        .maybeSingle();
      if (resultErr) throw resultErr;

      const mode = (['single_image', 'bi_temporal', 'optical_sar'].includes(row.analysis_type)
        ? row.analysis_type
        : 'single_image') as AnalysisMode;

      const displayId = denormalizeAnalysisId(uuid, uuid);

      const images: UploadedImage[] = (inputs ?? []).map((inp) => {
        const meta = (inp.metadata ?? {}) as any;
        const role = (meta.role ?? 'single') as UploadedImage['role'];
        const md: ImageMetadataType = {
          fileName: inp.filename ?? meta.fileName ?? `img-${role}`,
          fileFormat: (['GeoTIFF', 'TIFF', 'PNG', 'JPEG'].includes(inp.format)
            ? inp.format
            : 'PNG') as ImageMetadataType['fileFormat'],
          modality: (['optical', 'sar', 'multispectral', 'unknown'].includes(inp.modality)
            ? inp.modality
            : 'unknown') as Modality,
          modalityDetectionConfidence: meta.modalityDetectionConfidence ?? null,
          acquisitionDate: inp.acquisition_date ? new Date(inp.acquisition_date).toISOString() : meta.acquisitionDate ?? null,
          widthPx: meta.widthPx ?? null,
          heightPx: meta.heightPx ?? null,
          bandCount: meta.bandCount ?? null,
          crs: inp.crs ?? meta.crs ?? null,
          gsdMeters: inp.resolution ?? meta.gsdMeters ?? null,
          fileSizeBytes: typeof meta.fileSizeBytes === 'number' ? meta.fileSizeBytes : 0,
        };
        return {
          id: meta.imageId ?? `img-${inp.id}`,
          role,
          previewUrl: meta.previewUrl ?? null,
          metadata: md,
        };
      });

      const steps: ExecutionStep[] = (traceRows ?? []).map((tr, i) => {
        const out = (tr.output ?? {}) as any;
        return {
          id: `step-${i + 1}`,
          title: tr.tool_name ?? 'Pipeline Step',
          detail: out.detail ?? '',
          status: reverseStepStatus(tr.status),
          startedAt: out.startedAt ?? null,
          completedAt: out.completedAt ?? null,
          meta: (tr.parameters ?? {}) as Record<string, string | number | boolean>,
        };
      });

      const overallStatus = (['queued', 'processing', 'completed', 'failed'].includes(row.status)
        ? row.status
        : 'queued') as AnalysisStatus;

      const totalElapsedMs = resultRow?.result_metadata?.totalElapsedMs ??
        steps.reduce<number | null>((acc, s) => {
          if (s.startedAt && s.completedAt) {
            const d = new Date(s.completedAt).getTime() - new Date(s.startedAt).getTime();
            return (acc ?? 0) + d;
          }
          return acc;
        }, null);

      const evidence = (resultRow?.evidence ?? []) as BoundingBox[];
      const detectedTasks = (resultRow?.result_metadata?.detectedTasks ?? []) as TaskType[];
      const toolInvocations = (resultRow?.result_metadata?.toolInvocations ?? []) as ToolInvocation[];

      return {
        id: displayId,
        mode,
        query: row.query,
        status: overallStatus,
        createdAt: row.created_at ?? new Date().toISOString(),
        images,
        detectedTasks,
        answerText: resultRow?.answer ?? row.answer ?? null,
        confidence: resultRow ? resultRow.confidence : row.confidence,
        boundingBoxes: evidence.length > 0 ? evidence : null,
        changeMap: resultRow?.change_map_path
          ? { overlayUrl: resultRow.change_map_path, legend: [] }
          : null,
        toolInvocations,
        executionTrace: {
          steps,
          totalElapsedMs,
          overallStatus,
        },
        errorReason: resultRow?.result_metadata?.errorReason ?? null,
      };
    } catch (err) {
      logWarn('getAnalysisInner', err);
      return null;
    }
  }

  async getAnalysis(id: string, userId: string): Promise<AnalysisResult | null> {
    try {
      const uuid = normalizeAnalysisId(id);
      return await this.getAnalysisInner(uuid, userId);
    } catch (err) {
      logWarn('getAnalysis', err);
      return null;
    }
  }

  async deleteAnalysis(id: string, userId: string): Promise<void> {
    try {
      if (!supabase) return;
      const uuid = normalizeAnalysisId(id);
      const { error } = await supabase
        .from('analyses')
        .delete()
        .eq('id', uuid)
        .eq('user_id', userId);
      if (error) throw error;
    } catch (err) {
      logWarn('deleteAnalysis', err);
    }
  }

  async uploadInputFile(args: {
    userId: string;
    analysisId: string;
    role: UploadedImage['role'];
    file: File;
  }): Promise<{ storagePath: string; signedUrl?: string }> {
    try {
      if (!supabase) return { storagePath: '' };
      const uuid = normalizeAnalysisId(args.analysisId);
      const path = `${args.userId}/${uuid}/${args.role}-${sanitizeFilename(args.file.name)}`;
      const { error } = await supabase.storage
        .from(STORAGE_BUCKETS.inputs)
        .upload(path, args.file, {
          cacheControl: '31536000',
          upsert: true,
        });
      if (error) throw error;
      const signed = await this.getSignedUrl('inputs', path);
      return { storagePath: path, signedUrl: signed ?? undefined };
    } catch (err) {
      logWarn('uploadInputFile', err);
      return { storagePath: '' };
    }
  }

  async uploadResultBlob(args: {
    userId: string;
    analysisId: string;
    blobName: string;
    blob: Blob;
  }): Promise<{ storagePath: string }> {
    try {
      if (!supabase) return { storagePath: '' };
      const uuid = normalizeAnalysisId(args.analysisId);
      const bucket: StorageBucket = /report/i.test(args.blobName)
        ? STORAGE_BUCKETS.reports
        : STORAGE_BUCKETS.results;
      const path = `${args.userId}/${uuid}/${sanitizeFilename(args.blobName)}`;
      const { error } = await supabase.storage.from(bucket).upload(path, args.blob, {
        cacheControl: '31536000',
        upsert: true,
        contentType: args.blob.type || undefined,
      });
      if (error) throw error;
      return { storagePath: path };
    } catch (err) {
      logWarn('uploadResultBlob', err);
      return { storagePath: '' };
    }
  }

  async getSignedUrl(bucket: 'inputs' | 'results' | 'reports', path: string): Promise<string | null> {
    try {
      if (!supabase) return null;
      const bucketName: StorageBucket = STORAGE_BUCKETS[bucket];
      const { data, error } = await supabase.storage
        .from(bucketName)
        .createSignedUrl(path, 3600);
      if (error) throw error;
      return data?.signedUrl ?? null;
    } catch (err) {
      logWarn('getSignedUrl', err);
      return null;
    }
  }
}

class NoopSupabaseAnalysisService implements SupabaseAnalysisService {
  async getCurrentUserId(): Promise<string | null> {
    return null;
  }
  async ensureProfileForUser(): Promise<void> {}
  async saveAnalysisStarted(): Promise<void> {}
  async saveTraceStep(): Promise<void> {}
  async saveAnalysisCompleted(): Promise<void> {}
  async listHistory(): Promise<{ items: AnalysisResult[]; total: number }> {
    return { items: [], total: 0 };
  }
  async getAnalysis(): Promise<AnalysisResult | null> {
    return null;
  }
  async deleteAnalysis(): Promise<void> {}
  async uploadInputFile(): Promise<{ storagePath: string; signedUrl?: string }> {
    return { storagePath: '' };
  }
  async uploadResultBlob(): Promise<{ storagePath: string }> {
    return { storagePath: '' };
  }
  async getSignedUrl(): Promise<string | null> {
    return null;
  }
}

const ENABLE_SUPABASE_PERSISTENCE: boolean =
  HAS_SUPABASE && !DEMO_MODE && !!process.env.NEXT_PUBLIC_SUPABASE_URL;

export const supabaseAnalysisService: SupabaseAnalysisService =
  ENABLE_SUPABASE_PERSISTENCE
    ? new RealSupabaseAnalysisService()
    : new NoopSupabaseAnalysisService();

export const SUPABASE_PERSISTENCE_ENABLED = ENABLE_SUPABASE_PERSISTENCE;
