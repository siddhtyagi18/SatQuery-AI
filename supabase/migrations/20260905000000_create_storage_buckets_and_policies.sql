-- ============================================================
-- SatQuery-AI — Supabase Storage Buckets & Policies (Migration 002)
-- ============================================================
-- Prerequisite: Migration 001 (core tables + RLS) applied first.
-- Scope:
--   * Create 3 private storage buckets (satquery-inputs, satquery-results, satquery-reports)
--   * Idempotent (ON CONFLICT DO NOTHING)
--   * RLS on storage.objects — user owns paths whose first folder == auth.uid()
--   * Anon has NO access to any bucket (no permissive policies)
--
-- Bucket private flag: (buckets.public = false) — requires auth for any GET /object
-- ============================================================

-- ============================================================
-- 1. BUCKETS  (private; size limits; allowed MIME types)
-- ============================================================

INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types, created_at)
VALUES
(
  'satquery-inputs',
  'satquery-inputs',
  FALSE,
  524288000,
  ARRAY[
    'image/png',
    'image/jpeg',
    'image/tiff',
    'image/tiff; application=geotiff',
    'image/x-hdf5',
    'application/x-hdf5',
    'application/x-netcdf',
    'application/netcdf',
    'application/octet-stream',
    'application/json'
  ]::text[],
  now()
)
ON CONFLICT (id) DO NOTHING;

INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types, created_at)
VALUES
(
  'satquery-results',
  'satquery-results',
  FALSE,
  52428800,
  ARRAY[
    'image/png',
    'image/jpeg',
    'image/tiff',
    'application/json',
    'text/csv'
  ]::text[],
  now()
)
ON CONFLICT (id) DO NOTHING;

INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types, created_at)
VALUES
(
  'satquery-reports',
  'satquery-reports',
  FALSE,
  20971520,
  ARRAY[
    'application/json',
    'application/pdf',
    'text/html',
    'text/markdown'
  ]::text[],
  now()
)
ON CONFLICT (id) DO NOTHING;

-- ============================================================
-- 2. ENABLE RLS ON storage.objects  (Supabase default is ON — be explicit)
-- ============================================================

ALTER TABLE storage.objects ENABLE ROW LEVEL SECURITY;

-- ============================================================
-- 3. POLICIES  — one set of 4 ops per bucket
--
-- Ownership rule (all buckets):
--   Path format inside a bucket:  {auth.uid()}/{analysis_uuid}/...
--   storage.foldername(name)[1] returns the FIRST path segment == user UUID.
--
-- NOTE:
--   * storage.foldername(text) → text[]  (1-indexed in Postgres)
--   * [1] = first folder == user_id (scoping)
-- ============================================================

-- ------------------------------------------------------------
-- satquery-inputs
-- ------------------------------------------------------------

CREATE POLICY "satquery_inputs_select_owner"
    ON storage.objects FOR SELECT
    USING (
      bucket_id = 'satquery-inputs'
      AND auth.role() = 'authenticated'
      AND (storage.foldername(name))[1] = auth.uid()::text
    );

CREATE POLICY "satquery_inputs_insert_owner"
    ON storage.objects FOR INSERT
    WITH CHECK (
      bucket_id = 'satquery-inputs'
      AND auth.role() = 'authenticated'
      AND (storage.foldername(name))[1] = auth.uid()::text
    );

CREATE POLICY "satquery_inputs_update_owner"
    ON storage.objects FOR UPDATE
    USING (
      bucket_id = 'satquery-inputs'
      AND auth.role() = 'authenticated'
      AND (storage.foldername(name))[1] = auth.uid()::text
    )
    WITH CHECK (
      bucket_id = 'satquery-inputs'
      AND auth.role() = 'authenticated'
      AND (storage.foldername(name))[1] = auth.uid()::text
    );

CREATE POLICY "satquery_inputs_delete_owner"
    ON storage.objects FOR DELETE
    USING (
      bucket_id = 'satquery-inputs'
      AND auth.role() = 'authenticated'
      AND (storage.foldername(name))[1] = auth.uid()::text
    );

-- ------------------------------------------------------------
-- satquery-results
-- ------------------------------------------------------------

CREATE POLICY "satquery_results_select_owner"
    ON storage.objects FOR SELECT
    USING (
      bucket_id = 'satquery-results'
      AND auth.role() = 'authenticated'
      AND (storage.foldername(name))[1] = auth.uid()::text
    );

CREATE POLICY "satquery_results_insert_owner"
    ON storage.objects FOR INSERT
    WITH CHECK (
      bucket_id = 'satquery-results'
      AND auth.role() = 'authenticated'
      AND (storage.foldername(name))[1] = auth.uid()::text
    );

CREATE POLICY "satquery_results_update_owner"
    ON storage.objects FOR UPDATE
    USING (
      bucket_id = 'satquery-results'
      AND auth.role() = 'authenticated'
      AND (storage.foldername(name))[1] = auth.uid()::text
    )
    WITH CHECK (
      bucket_id = 'satquery-results'
      AND auth.role() = 'authenticated'
      AND (storage.foldername(name))[1] = auth.uid()::text
    );

CREATE POLICY "satquery_results_delete_owner"
    ON storage.objects FOR DELETE
    USING (
      bucket_id = 'satquery-results'
      AND auth.role() = 'authenticated'
      AND (storage.foldername(name))[1] = auth.uid()::text
    );

-- ------------------------------------------------------------
-- satquery-reports
-- ------------------------------------------------------------

CREATE POLICY "satquery_reports_select_owner"
    ON storage.objects FOR SELECT
    USING (
      bucket_id = 'satquery-reports'
      AND auth.role() = 'authenticated'
      AND (storage.foldername(name))[1] = auth.uid()::text
    );

CREATE POLICY "satquery_reports_insert_owner"
    ON storage.objects FOR INSERT
    WITH CHECK (
      bucket_id = 'satquery-reports'
      AND auth.role() = 'authenticated'
      AND (storage.foldername(name))[1] = auth.uid()::text
    );

CREATE POLICY "satquery_reports_update_owner"
    ON storage.objects FOR UPDATE
    USING (
      bucket_id = 'satquery-reports'
      AND auth.role() = 'authenticated'
      AND (storage.foldername(name))[1] = auth.uid()::text
    )
    WITH CHECK (
      bucket_id = 'satquery-reports'
      AND auth.role() = 'authenticated'
      AND (storage.foldername(name))[1] = auth.uid()::text
    );

CREATE POLICY "satquery_reports_delete_owner"
    ON storage.objects FOR DELETE
    USING (
      bucket_id = 'satquery-reports'
      AND auth.role() = 'authenticated'
      AND (storage.foldername(name))[1] = auth.uid()::text
    );

-- ------------------------------------------------------------
-- Explicitly deny anon for the 3 buckets (belt + suspenders)
-- ------------------------------------------------------------

CREATE POLICY "satquery_all_buckets_anon_deny"
    ON storage.objects FOR SELECT
    USING (
      bucket_id IN ('satquery-inputs', 'satquery-results', 'satquery-reports')
      AND auth.role() = 'anon'
      AND FALSE
    );
