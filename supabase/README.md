# SatQuery-AI — Supabase Database Schema

> This folder describes the Supabase Postgres schema for the SatQuery-AI
> cloud storage layer. The FastAPI backend remains the primary system of
> record; Supabase is an optional, opt-in secondary store used for:
> (1) per-user cloud history persistence, (2) signed storage URLs for
> imagery + change-map blobs, (3) server-side auth policies.

## Files

| File | Purpose |
|---|---|
| `migrations/20260901000001_create_satquery_core_tables.sql` | Idempotent migration. Run once on any Supabase project via the SQL Editor, `supabase db push`, or `psql`. |

## Apply

```bash
# (A) Via Supabase SQL Editor — paste the migration SQL into a New Query and run.
#
# (B) Via Supabase CLI
supabase link   --project-ref <your-ref>
supabase db push
```

## Tables

### 1. `public.profiles` (1 row per Supabase Auth user)

| Column | Type | Notes |
|---|---|---|
| `id` | `uuid` (PK) | `REFERENCES auth.users(id) ON DELETE CASCADE` — 1:1 mirror of a Supabase Auth user. |
| `email` | `text NOT NULL` | Mirror of `auth.users.email` (RLS-visible to the user only). |
| `full_name` | `text` | Optional display name. |
| `role` | `text NOT NULL DEFAULT 'operator'` | `operator \| analyst \| specialist \| admin` (CHECK constraint). |
| `created_at` | `timestamptz NOT NULL DEFAULT now()` | Profile creation time. |

### 2. `public.analyses` (Top-level analysis request)

| Column | Type | Notes |
|---|---|---|
| `id` | `uuid` (PK) | `DEFAULT gen_random_uuid()` |
| `user_id` | `uuid NOT NULL` (FK → `profiles.id` `ON DELETE CASCADE`) | Owner — every analysis is scoped to exactly one user. |
| `query` | `text NOT NULL` | Free-text prompt. |
| `analysis_type` | `text NOT NULL` | `single_image \| bi_temporal \| optical_sar \| captioning \| grounding \| change_detection \| change_vqa \| change_description` (CHECK). |
| `status` | `text NOT NULL DEFAULT 'queued'` | `queued \| processing \| completed \| failed` (CHECK). |
| `answer` | `text NULL` | Final answer text (denormalised from `analysis_results.answer` for fast list-view filtering). |
| `confidence` | `numeric(5,4) NULL` | 0..1 scalar; NULL when anti-fabrication rules suppress a value (CHECK 0..1). |
| `created_at` | `timestamptz NOT NULL DEFAULT now()` | |
| `completed_at` | `timestamptz NULL` | Set when status transitions to `completed` / `failed`. |

### 3. `public.analysis_inputs` (Per-analysis uploaded imagery / sensor files)

| Column | Type | Notes |
|---|---|---|
| `id` | `uuid` (PK) | |
| `analysis_id` | `uuid NOT NULL` (FK → `analyses.id` `ON DELETE CASCADE`) | Parent analysis. |
| `filename` | `text NOT NULL` | Original upload filename. |
| `modality` | `text NOT NULL DEFAULT 'unknown'` | `optical \| sar \| multispectral \| hyperspectral \| thermal \| unknown` (CHECK). |
| `format` | `text NOT NULL DEFAULT 'unknown'` | `GeoTIFF \| TIFF \| PNG \| JPEG \| JP2 \| COG \| NETCDF \| HDF5 \| unknown` (CHECK). |
| `acquisition_date` | `date NULL` | Scene date. |
| `sensor` | `text NULL` | e.g. `Sentinel-2B`, `RISAT-2BR1`. |
| `resolution` | `numeric(10,3) NULL` | Ground sample distance in metres (if known). |
| `crs` | `text NULL` | EPSG string, e.g. `EPSG:32644`. |
| `storage_path` | `text NOT NULL` | Bucket prefix path (Supabase Storage `uploads/` bucket), signed URL generated from this. |
| `metadata` | `jsonb NOT NULL DEFAULT '{}'` | Flexible bag: `band_count`, `gsd_meters`, `bounds_wkt`, `width_px`, `height_px`, `file_size_bytes`, vendor tags, etc. |
| `created_at` | `timestamptz NOT NULL DEFAULT now()` | |

### 4. `public.analysis_trace` (Per-step ordered SatQuery pipeline trace)

| Column | Type | Notes |
|---|---|---|
| `id` | `uuid` (PK) | |
| `analysis_id` | `uuid NOT NULL` (FK → `analyses.id` `ON DELETE CASCADE`) | |
| `step` | `integer NOT NULL` | Pipeline step index (0..N). |
| `tool_name` | `text NOT NULL` | e.g. `rs_vqa`, `change_detector`, `spatial_analyzer`. |
| `status` | `text NOT NULL DEFAULT 'pending'` | `pending \| in_progress \| done \| error \| skipped` (CHECK). |
| `parameters` | `jsonb NOT NULL DEFAULT '{}'` | Inputs fed to the step. |
| `output` | `jsonb NOT NULL DEFAULT '{}'` | Outputs produced (scalars, references into storage paths, stats). |
| `created_at` | `timestamptz NOT NULL DEFAULT now()` | |

Unique constraint: `(analysis_id, step)` — each step exists at most once per analysis.

### 5. `public.analysis_results` (Terminal output of a completed/failed analysis)

| Column | Type | Notes |
|---|---|---|
| `id` | `uuid` (PK) | |
| `analysis_id` | `uuid NOT NULL UNIQUE` (FK → `analyses.id` `ON DELETE CASCADE`) | Exactly 0 or 1 result rows per analysis. |
| `answer` | `text NULL` | Final natural-language answer. |
| `confidence` | `numeric(5,4) NULL` | 0..1 scalar; NULL when anti-fabrication rules apply (CHECK 0..1). |
| `evidence` | `jsonb NOT NULL DEFAULT '[]'` | JSON array of grounding / bounding-box entries — `[{x, y, width, height, label, confidence}]` (normalised 0..1 coords). |
| `statistics` | `jsonb NOT NULL DEFAULT '{}'` | Per-task statistics (change-detection class histogram, per-class m², etc.). |
| `result_metadata` | `jsonb NOT NULL DEFAULT '{}'` | Freeform: schema version, tool versions, total elapsed ms, anti-fabrication flags. |
| `change_map_path` | `text NULL` | Storage bucket prefix for a change-map PNG/GeoTIFF (if change task). |
| `report_path` | `text NULL` | Storage bucket prefix for a PDF or HTML mission report (if generated). |
| `created_at` | `timestamptz NOT NULL DEFAULT now()` | |

---

## Relationships (ER)

```
auth.users ──1:1──▶ profiles.id ──1:N──▶ analyses.user_id
                                       │
                                       ├──1:N──▶ analysis_inputs.analysis_id
                                       ├──1:N──▶ analysis_trace.analysis_id
                                       └──1:1──▶ analysis_results.analysis_id (UNIQUE)
```

All child FKs are `ON DELETE CASCADE` — deleting a user or analysis also cleans up every dependent row.

---

## Indexes (17 total)

### Scalar B-trees
| Table | Index | Why |
|---|---|---|
| `profiles` | `profiles_role_idx` (role) | Admin list-by-role. |
| `analyses` | `analyses_user_created_idx` (user_id, created_at DESC) | History "List analyses for me newest-first". |
| `analyses` | `analyses_status_idx` (status) | Queue queries. |
| `analyses` | `analyses_type_idx` (analysis_type) | Per-mode filtering. |
| `analyses` | `analyses_user_status_idx` (user_id, status PARTIAL WHERE status <> 'completed') | Dashboard "My active analyses". |
| `analysis_inputs` | `inputs_analysis_idx` (analysis_id) | Join support. |
| `analysis_inputs` | `inputs_modality_idx` (modality) | Per-modality queries. |
| `analysis_results` | `results_analysis_idx` (analysis_id) | Join support (also covered by UNIQUE FK, kept for clarity). |
| `analysis_trace` | `analysis_trace_analysis_step_idx` UNIQUE (analysis_id, step) | Deterministic step ordering. |

### JSONB GIN indexes
Useful when backend queries metadata / evidence / statistics fields.
| Table | Column |
|---|---|
| `analysis_inputs` | `metadata` (GIN) |
| `analysis_results` | `evidence` (GIN) |
| `analysis_results` | `statistics` (GIN) |
| `analysis_trace` | `parameters` (GIN) |

---

## Row Level Security — Summary

RLS is **enabled** on ALL five `public.*` tables. There are NO `true` / `USING (true)` policies — so the `anon` key (and any unauthenticated caller) sees zero rows everywhere.

### Access model

| Key type | Access |
|---|---|
| `anon` | None (all policies fail; `auth.uid() IS NULL`). |
| Authenticated user (`auth.uid()` set) | Only rows that transitively belong to `profiles.id = auth.uid()`. |
| `service_role` / `postgres` superusers | RLS bypassed — backend writes happen via service role only. |

### Policies per table (22 total)

All child tables (`analysis_inputs`, `analysis_trace`, `analysis_results`) use the same **ownership-via-parent pattern**:

```
USING ( EXISTS ( SELECT 1
                   FROM public.analyses a
                  WHERE a.id = <child>.analysis_id
                    AND a.user_id = auth.uid() ) )
```

This policy pattern appears 4 times per table (SELECT / INSERT / UPDATE / DELETE) — meaning an attacker who guesses a UUID still cannot see or modify rows belonging to another user.

| Table | SELECT | INSERT | UPDATE | DELETE |
|---|---|---|---|---|
| `profiles` | `id = auth.uid()` | `id = auth.uid()` + email matches `auth.email()` | Same constraints | (none — users keep their row) |
| `analyses` | `user_id = auth.uid()` | `user_id = auth.uid()` | Owner AND `status IN ('queued','processing','failed')` — prevents mutating already-finalised rows | Owner only |
| `analysis_inputs` | Owner via parent analysis | Owner via parent analysis | Owner via parent analysis | Owner via parent analysis |
| `analysis_trace` | Owner via parent analysis | Owner via parent analysis | Owner via parent analysis | Owner via parent analysis |
| `analysis_results` | Owner via parent analysis | Owner via parent analysis | Owner via parent analysis | Owner via parent analysis |

### Important — no public read anywhere

There is no `FOR SELECT USING (true)` policy on any table. The only way to read a row is:
1. Be an authenticated user (via Supabase Auth email/OAuth), AND
2. Be the owner of the parent `analyses.user_id` (or the `profiles.id` row itself).

### Frontend contract

- The Next.js `lib/supabase.ts` client MUST be initialised with the **anon key** only
  (via `NEXT_PUBLIC_SUPABASE_ANON_KEY`).
- The **service_role key** must NEVER leave the backend FastAPI server environment
  (`backend/.env` only — already blacklisted by `.gitignore`).
- Backend writes (trace steps, results, final confidence) happen via service role
  (RLS bypassed) — this is necessary because the AI orchestrator updates rows on
  behalf of the user after the initial HTTP request has returned.

---

## Supabase Storage (manual step — not SQL-migratable)

After running this migration, create two **private** buckets in Supabase Studio
→ Storage (NOT PUBLIC, they must require Policy):

1. `uploads`    — stores `analysis_inputs.storage_path` (per-user uploaded scenes)
2. `results`    — stores `analysis_results.change_map_path` / `.report_path`

Recommended bucket-level RLS policies (add via Studio / SQL as `storage.objects` policies):
- `uploads`: user can read/write/delete objects whose `name` starts with
  `<auth.uid()>/` (backend uploads via service role should also namespace by uid).
- `results`: same `name LIKE (auth.uid() || '/%')` ownership pattern.

Signed URLs are then handed out by the backend via service role for in-browser display.
