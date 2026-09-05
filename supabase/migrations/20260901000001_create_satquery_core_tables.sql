-- ============================================================
-- SatQuery-AI — Supabase Database Schema (Migration 001)
-- ============================================================
-- Scope:    Core public tables, FKs, indexes, RLS + policies
-- Engine:   Supabase / PostgreSQL 15+ (pgcrypto preinstalled)
-- Users:    Supabase Auth owns auth.users; public.profiles mirrors it
--
-- WARNING:  Enable RLS BEFORE INSERTING ANY ROWS.
--           policies are deliberately restrictive — no anonymous
--           read access anywhere. Service role bypasses RLS for
--           backend writes; frontend must use anon key + login.
-- ============================================================

-- pgcrypto ships with Supabase but be explicit for local pg installs.
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================
-- 1. profiles — user directory (one row per auth.users entry)
-- ============================================================
CREATE TABLE IF NOT EXISTS public.profiles (
    id          UUID PRIMARY KEY REFERENCES auth.users (id) ON DELETE CASCADE,
    email       TEXT NOT NULL,
    full_name   TEXT,
    role        TEXT NOT NULL DEFAULT 'operator'
                CONSTRAINT profiles_role_check
                CHECK (role IN ('operator', 'analyst', 'specialist', 'admin')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE  public.profiles     IS 'Per-user directory row; id links 1:1 to Supabase Auth user.';
COMMENT ON COLUMN public.profiles.id  IS 'FK → auth.users.id (one profile per authenticated user).';

-- ============================================================
-- 2. analyses — top-level analysis request (per user)
-- ============================================================
CREATE TABLE IF NOT EXISTS public.analyses (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES public.profiles (id) ON DELETE CASCADE,
    query           TEXT NOT NULL,
    analysis_type   TEXT NOT NULL
                    CONSTRAINT analyses_analysis_type_check
                    CHECK (analysis_type IN (
                        'single_image', 'bi_temporal', 'optical_sar',
                        'captioning', 'grounding', 'change_detection',
                        'change_vqa', 'change_description'
                    )),
    status          TEXT NOT NULL DEFAULT 'queued'
                    CONSTRAINT analyses_status_check
                    CHECK (status IN ('queued', 'processing', 'completed', 'failed')),
    answer          TEXT,
    confidence      NUMERIC(5, 4)
                    CONSTRAINT analyses_confidence_check
                    CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at    TIMESTAMPTZ
);

COMMENT ON TABLE  public.analyses IS 'Top-level analysis record owned by a profile.';
COMMENT ON COLUMN public.analyses.confidence IS 'Optional 0..1 scalar; NULL when anti-fabrication rule disallows output.';

-- ============================================================
-- 3. analysis_inputs — per-analysis uploaded image / sensor file
-- ============================================================
CREATE TABLE IF NOT EXISTS public.analysis_inputs (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id      UUID NOT NULL REFERENCES public.analyses (id) ON DELETE CASCADE,
    filename         TEXT NOT NULL,
    modality         TEXT NOT NULL DEFAULT 'unknown'
                     CONSTRAINT analysis_inputs_modality_check
                     CHECK (modality IN ('optical', 'sar', 'multispectral', 'hyperspectral', 'thermal', 'unknown')),
    format           TEXT NOT NULL DEFAULT 'unknown'
                     CONSTRAINT analysis_inputs_format_check
                     CHECK (format IN ('GeoTIFF', 'TIFF', 'PNG', 'JPEG', 'JP2', 'COG', 'NETCDF', 'HDF5', 'unknown')),
    acquisition_date DATE,
    sensor           TEXT,
    resolution       NUMERIC(10, 3),
    crs              TEXT,
    storage_path     TEXT NOT NULL,
    metadata         JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE  public.analysis_inputs IS 'Every image / sensor file associated with one analysis.';
COMMENT ON COLUMN public.analysis_inputs.storage_path IS 'Bucket prefix path, e.g. uploads/<uuid>/scene_before.tif';
COMMENT ON COLUMN public.analysis_inputs.metadata   IS 'Flexible per-image metadata (band_count, gsd_meters, bounds, file_size_bytes, etc.).';

-- ============================================================
-- 4. analysis_trace — per-step execution trace (SatQuery pipeline)
-- ============================================================
CREATE TABLE IF NOT EXISTS public.analysis_trace (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id UUID NOT NULL REFERENCES public.analyses (id) ON DELETE CASCADE,
    step        INTEGER NOT NULL,
    tool_name   TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending'
                CONSTRAINT analysis_trace_status_check
                CHECK (status IN ('pending', 'in_progress', 'done', 'error', 'skipped')),
    parameters  JSONB NOT NULL DEFAULT '{}'::jsonb,
    output      JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS analysis_trace_analysis_step_idx
    ON public.analysis_trace (analysis_id, step);

COMMENT ON TABLE public.analysis_trace IS 'Ordered per-step trace (validation → metadata → tools → preprocessing → inference → validation → persist).';

-- ============================================================
-- 5. analysis_results — terminal output blob(s) per analysis
-- ============================================================
CREATE TABLE IF NOT EXISTS public.analysis_results (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id       UUID NOT NULL UNIQUE REFERENCES public.analyses (id) ON DELETE CASCADE,
    answer            TEXT,
    confidence        NUMERIC(5, 4)
                      CONSTRAINT analysis_results_confidence_check
                      CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
    evidence          JSONB NOT NULL DEFAULT '[]'::jsonb,
    statistics        JSONB NOT NULL DEFAULT '{}'::jsonb,
    result_metadata   JSONB NOT NULL DEFAULT '{}'::jsonb,
    change_map_path   TEXT,
    report_path       TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE  public.analysis_results IS 'Single terminal result row per completed (or failed) analysis.';
COMMENT ON COLUMN public.analysis_results.evidence IS 'Array of bounding-box / grounding evidence entries (normalised coords + label + confidence).';
COMMENT ON COLUMN public.analysis_results.statistics IS 'Per-task stats: change-detection pixel counts, per-class area, etc.';
COMMENT ON COLUMN public.analysis_results.result_metadata IS 'Arbitrary schema-version, tool-versions, inference timings, anti-fabrication flags.';

-- ============================================================
-- INDEXES  (read-heavy: user owns rows, list by user+created desc)
-- ============================================================
CREATE INDEX IF NOT EXISTS profiles_role_idx         ON public.profiles  (role);

CREATE INDEX IF NOT EXISTS analyses_user_created_idx  ON public.analyses  (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS analyses_status_idx        ON public.analyses  (status);
CREATE INDEX IF NOT EXISTS analyses_type_idx          ON public.analyses  (analysis_type);
CREATE INDEX IF NOT EXISTS analyses_user_status_idx   ON public.analyses  (user_id, status) WHERE status <> 'completed';

CREATE INDEX IF NOT EXISTS inputs_analysis_idx        ON public.analysis_inputs (analysis_id);
CREATE INDEX IF NOT EXISTS inputs_modality_idx        ON public.analysis_inputs (modality);

CREATE INDEX IF NOT EXISTS results_analysis_idx       ON public.analysis_results (analysis_id);

-- JSONB GIN indexes — useful when backend queries metadata / evidence fields.
CREATE INDEX IF NOT EXISTS inputs_metadata_gin        ON public.analysis_inputs  USING GIN (metadata);
CREATE INDEX IF NOT EXISTS results_evidence_gin       ON public.analysis_results USING GIN (evidence);
CREATE INDEX IF NOT EXISTS results_statistics_gin     ON public.analysis_results USING GIN (statistics);
CREATE INDEX IF NOT EXISTS trace_parameters_gin       ON public.analysis_trace   USING GIN (parameters);

-- ============================================================
-- ROW LEVEL SECURITY  —  enabled on EVERY public table.
-- ============================================================
ALTER TABLE public.profiles          ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.analyses          ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.analysis_inputs   ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.analysis_trace    ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.analysis_results  ENABLE ROW LEVEL SECURITY;

-- ============================================================
-- RLS POLICIES
--
-- Auth model:
--   * anon key   → NO access (all policies fail because auth.uid() is NULL)
--   * user key   → Can see / touch ONLY rows reachable via profiles.id = auth.uid()
--   * service_role / pgbouncer / postgres → RLS bypass (backend + migrations only)
--
-- profiles:   each user can see/edit their own directory row ONLY.
--             (No public listing, no anonymous read.)
-- analyses / inputs / trace / results:
--             SELECT — ownership chain user_id → analyses.id → child tables.
--             INSERT — user_id MUST equal auth.uid() (for analyses)
--                      and children must reference an analysis owned by the user.
--             UPDATE — only allowed when status IN ('queued','processing') AND owner.
--             DELETE — only owner; deletes cascade to children.
-- ============================================================

-- ------------------------------------------------------------
-- profiles  (1:1 with auth.users)
-- ------------------------------------------------------------
CREATE POLICY "profiles_select_own"
    ON public.profiles FOR SELECT
    USING (auth.uid() = id);

CREATE POLICY "profiles_insert_own"
    ON public.profiles FOR INSERT
    WITH CHECK (auth.uid() = id AND auth.email() IS NOT DISTINCT FROM email);

CREATE POLICY "profiles_update_own"
    ON public.profiles FOR UPDATE
    USING (auth.uid() = id)
    WITH CHECK (auth.uid() = id AND auth.email() IS NOT DISTINCT FROM email);

-- ------------------------------------------------------------
-- analyses
-- ------------------------------------------------------------
CREATE POLICY "analyses_select_owner"
    ON public.analyses FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "analyses_insert_owner"
    ON public.analyses FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "analyses_update_owner_active"
    ON public.analyses FOR UPDATE
    USING (auth.uid() = user_id AND status IN ('queued', 'processing', 'failed'))
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "analyses_delete_owner"
    ON public.analyses FOR DELETE
    USING (auth.uid() = user_id);

-- ------------------------------------------------------------
-- analysis_inputs  (inherits visibility from parent analysis)
-- ------------------------------------------------------------
CREATE POLICY "inputs_select_owner"
    ON public.analysis_inputs FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM public.analyses a
             WHERE a.id = analysis_inputs.analysis_id
               AND a.user_id = auth.uid()
        )
    );

CREATE POLICY "inputs_insert_owner"
    ON public.analysis_inputs FOR INSERT
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.analyses a
             WHERE a.id = analysis_inputs.analysis_id
               AND a.user_id = auth.uid()
        )
    );

CREATE POLICY "inputs_update_owner"
    ON public.analysis_inputs FOR UPDATE
    USING (
        EXISTS (
            SELECT 1 FROM public.analyses a
             WHERE a.id = analysis_inputs.analysis_id
               AND a.user_id = auth.uid()
        )
    )
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.analyses a
             WHERE a.id = analysis_inputs.analysis_id
               AND a.user_id = auth.uid()
        )
    );

CREATE POLICY "inputs_delete_owner"
    ON public.analysis_inputs FOR DELETE
    USING (
        EXISTS (
            SELECT 1 FROM public.analyses a
             WHERE a.id = analysis_inputs.analysis_id
               AND a.user_id = auth.uid()
        )
    );

-- ------------------------------------------------------------
-- analysis_trace
-- ------------------------------------------------------------
CREATE POLICY "trace_select_owner"
    ON public.analysis_trace FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM public.analyses a
             WHERE a.id = analysis_trace.analysis_id
               AND a.user_id = auth.uid()
        )
    );

CREATE POLICY "trace_insert_owner"
    ON public.analysis_trace FOR INSERT
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.analyses a
             WHERE a.id = analysis_trace.analysis_id
               AND a.user_id = auth.uid()
        )
    );

CREATE POLICY "trace_update_owner"
    ON public.analysis_trace FOR UPDATE
    USING (
        EXISTS (
            SELECT 1 FROM public.analyses a
             WHERE a.id = analysis_trace.analysis_id
               AND a.user_id = auth.uid()
        )
    )
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.analyses a
             WHERE a.id = analysis_trace.analysis_id
               AND a.user_id = auth.uid()
        )
    );

CREATE POLICY "trace_delete_owner"
    ON public.analysis_trace FOR DELETE
    USING (
        EXISTS (
            SELECT 1 FROM public.analyses a
             WHERE a.id = analysis_trace.analysis_id
               AND a.user_id = auth.uid()
        )
    );

-- ------------------------------------------------------------
-- analysis_results
-- ------------------------------------------------------------
CREATE POLICY "results_select_owner"
    ON public.analysis_results FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM public.analyses a
             WHERE a.id = analysis_results.analysis_id
               AND a.user_id = auth.uid()
        )
    );

CREATE POLICY "results_insert_owner"
    ON public.analysis_results FOR INSERT
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.analyses a
             WHERE a.id = analysis_results.analysis_id
               AND a.user_id = auth.uid()
        )
    );

CREATE POLICY "results_update_owner"
    ON public.analysis_results FOR UPDATE
    USING (
        EXISTS (
            SELECT 1 FROM public.analyses a
             WHERE a.id = analysis_results.analysis_id
               AND a.user_id = auth.uid()
        )
    )
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.analyses a
             WHERE a.id = analysis_results.analysis_id
               AND a.user_id = auth.uid()
        )
    );

CREATE POLICY "results_delete_owner"
    ON public.analysis_results FOR DELETE
    USING (
        EXISTS (
            SELECT 1 FROM public.analyses a
             WHERE a.id = analysis_results.analysis_id
               AND a.user_id = auth.uid()
        )
    );
