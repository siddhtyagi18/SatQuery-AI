# SatQuery-AI — Supabase Integration Specification

## 1. Problem & Goals

### Problem
SatQuery-AI currently uses:
- **Auth**: Mock sessionStorage-based auth (or partial Supabase auth in `authService.ts` that lacks profile creation on signup)
- **Persistence**: In-memory `Map<string, AnalysisResult>` store (mockApi) or FastAPI backend (liveApi) — neither persists user analyses to Supabase
- **Storage**: No Supabase Storage usage; uploaded images use `URL.createObjectURL()` or FastAPI file storage
- **History**: Lost on page refresh (mock mode) or stored in FastAPI only (live mode)

### Goals
1. Wire the existing Login UI to real Supabase Auth (signup/login/logout/session) while preserving DEMO/MOCK fallback
2. Create a `profiles` row for every new signup, secured by RLS
3. Save & load analysis history to/from Supabase `analyses` table, preserving current UI entirely
4. Persist image/modality/date/sensor/CRS metadata to `analysis_inputs`
5. Persist the REAL execution trace (not fabricated steps) to `analysis_trace`
6. Persist answer/confidence/evidence/result metadata to `analysis_results`
7. Use Supabase Storage with private buckets for inputs/results/reports, scoped by user & analysis
8. Pass typecheck/lint/build; verify no secret keys leak to frontend; confirm RLS enforces row ownership

### Non-Goals
- ❌ NO changes to existing AI models, FastAPI backend, API contracts, or analysis workflows
- ❌ NO UI redesign, styling changes, or component layout changes
- ❌ NO breaking changes to DEMO/MOCK mode (must continue working when Supabase env vars are unset or `NEXT_PUBLIC_API_MODE=mock`)
- ❌ NO fabrication of trace steps, confidence values, coordinates, or masks
- ❌ NO exposure of `service_role` or secret keys to frontend/bundled code
- ❌ NO Git commits or pushes

---

## 2. Users & User Stories

| Actor | Story |
|---|---|
| **Ground Station Operator** | "I can sign up with email+password, log in, log out, and my session persists between reloads." |
| **Operator** | "After signup, my profile row is created automatically and I can only see/edit my own data." |
| **Operator** | "Every analysis I run is saved to the archive, I can search/filter/delete old analyses, and re-open past results." |
| **Operator** | "Uploaded images and generated reports are stored securely and only I can download them." |
| **Developer** | "When `NEXT_PUBLIC_SUPABASE_URL` or `NEXT_PUBLIC_SUPABASE_ANON_KEY` is missing OR when `DEMO_MODE=true`, the app falls back to the existing mock behaviour without errors." |
| **Security Auditor** | "RLS blocks anon/unauth users. A user cannot read, write, or delete another user's rows. No Supabase secret key is ever sent to the browser." |

---

## 3. Functional Requirements (FR)

### FR-1 — Supabase Authentication
- **FR-1.1**: Existing Login page's "Sign In" and "Request ground-station access" buttons MUST call `signIn()` / `signUp()` from `authService.ts`, which already has a working `SupabaseAuthService`.
- **FR-1.2**: `onAuthStateChange` MUST fire `INITIAL_SESSION`, `SIGNED_IN`, `SIGNED_OUT`, `TOKEN_REFRESHED`, `USER_UPDATED` events correctly.
- **FR-1.3**: `AuthContext` (`authContext.tsx`) MUST honor Supabase session (not just mock sessionStorage) for redirect guards.
- **FR-1.4**: Logout button in `TopBar` MUST call `signOut()` and clear both Supabase session + mock storage.
- **FR-1.5**: Session persistence across page reloads MUST work via Supabase's built-in storage (not our sessionStorage) when in supabase auth mode.

### FR-2 — User Profiles
- **FR-2.1**: On successful signup (after Supabase `auth.signUp` resolves), a row MUST be inserted into `public.profiles (id, email, full_name, role, created_at)` where `id = auth.uid()`.
- **FR-2.2**: The existing RLS policy `profiles_select_own` / `profiles_insert_own` / `profiles_update_own` in migration 001 MUST remain unchanged and MUST be the sole enforcement of profile visibility.
- **FR-2.3**: Profile creation MUST survive auto-confirmation (Supabase project with `Disable email confirmations` ON) and email-confirmation flows.
- **FR-2.4**: If profile insert fails (e.g. RLS), the signup flow MUST surface an error to the user, not silently succeed without a profile.

### FR-3 — Analysis History
- **FR-3.1**: On every `submitAnalysis()` success, an `analyses` row MUST be upserted/inserted with `user_id = auth.uid()`, `query`, `analysis_type`, `status='queued'`, `created_at=now()`.
- **FR-3.2**: On `streamExecutionTrace` completion (status = completed or failed), the `analyses` row MUST be updated with final `status`, `answer`, `confidence`, `completed_at`.
- **FR-3.3**: `listAnalysisHistory()` MUST fetch from Supabase `analyses` (ordered by `created_at DESC`) when auth mode is supabase; fall back to in-memory store when in mock mode.
- **FR-3.4**: `deleteAnalysis(id)` MUST delete from Supabase `analyses` (cascade deletes children via FK `ON DELETE CASCADE`); mock store fallback preserved.
- **FR-3.5**: `getAnalysis(id)` MUST load from Supabase (joins analyses + inputs + trace + results), reconstructing a full `AnalysisResult` matching the existing type contract.

### FR-4 — Analysis Metadata (analysis_inputs)
- **FR-4.1**: For every `UploadedImage` in `submitAnalysis()`, insert one `analysis_inputs` row with:
  - `analysis_id` → parent analyses UUID
  - `filename`, `modality`, `format`
  - `acquisition_date`, `sensor`, `resolution`, `crs` (from extracted metadata if available, else NULL)
  - `storage_path` = bucket path (`inputs/{user_id}/{analysis_id}/{role}-{filename}`)
  - `metadata` JSONB = full `ImageMetadataType` serialized
- **FR-4.2**: When loading `getAnalysis(id)`, `analysis_inputs` rows MUST be mapped back to `AnalysisResult.images[]` (UploadedImage[]), preserving `role` via the storage_path suffix or a metadata field.

### FR-5 — Agent Execution Trace (analysis_trace)
- **FR-5.1**: Each `ExecutionStep` in `ExecutionTrace` MUST be persisted to `analysis_trace` (one row per step) ONLY as the step actually transitions state — NEVER pre-populate fabricated steps.
- **FR-5.2**: Step mapping:
  - `step.step_number` ↔ integer order of the step in `trace.steps[]`
  - `step.tool_name` ↔ `ExecutionStep.title` (truncated if >255 or mapped via lookup)
  - `step.parameters` JSONB ↔ `ExecutionStep.meta` (or `{}` if absent)
  - `step.output` JSONB ↔ `{ detail, startedAt, completedAt }`
  - `step.status` ↔ `pending|in_progress|done|error|skipped` (map frontend `StepStatus` via: pending→pending, in_progress→in_progress, done→done, error→error)
- **FR-5.3**: On trace load, rows MUST be re-ordered by `step ASC` and materialised back into `ExecutionTrace.steps[]` in order.

### FR-6 — Analysis Results (analysis_results)
- **FR-6.1**: When an analysis reaches a terminal state (completed / failed), UPSERT one `analysis_results` row:
  - `answer` = `AnalysisResult.answerText` (NULL if failed)
  - `confidence` = `AnalysisResult.confidence` (NULL if not available or anti-fabrication disallows)
  - `evidence` JSONB = serialize `AnalysisResult.boundingBoxes[]` as `[{x,y,width,height,label,confidence}]`
  - `statistics` JSONB = change-detection pixel/class stats if present; else `{}`
  - `result_metadata` JSONB = `{ detectedTasks, toolInvocations, schemaVersion: 1, antiFabricationFlags: {...} }`
  - `change_map_path` = storage path of change map overlay (if applicable)
  - `report_path` = storage path of exported JSON report (if user generates one)
- **FR-6.2**: NULL confidence MUST be stored as SQL NULL (not 0) to honor the anti-fabrication rule.

### FR-7 — Supabase Storage
- **FR-7.1**: Create 3 private buckets via migration (or verify programmatically on startup):
  - `satquery-inputs` — private; user uploads
  - `satquery-results` — private; generated change maps, overlays, figures
  - `satquery-reports` — private; exported JSON/PDF reports
- **FR-7.2**: Bucket-level policies MUST allow read/write ONLY to objects where the path prefix matches `auth.uid()`:
  - `satquery-inputs/{user_id}/*` → only that user
  - `satquery-results/{user_id}/*` → only that user
  - `satquery-reports/{user_id}/*` → only that user
- **FR-7.3**: All public-access policies MUST be absent. `SELECT` on `storage.objects` MUST be denied for paths outside own scope.
- **FR-7.4**: Uploads MUST route through the Supabase JS client (signed URL fallback if direct upload fails). `storage_path` in DB rows MUST match the actual object key.
- **FR-7.5**: For mock/demo mode, storage operations MUST be no-ops or use the existing `URL.createObjectURL` approach without errors.

### FR-8 — DEMO / MOCK Mode Preservation
- **FR-8.1**: If `!HAS_SUPABASE` OR `DEMO_MODE === true` OR `API_MODE === 'mock'`, all new Supabase persistence calls MUST short-circuit and delegate to the existing in-memory store / sessionStorage mock flow.
- **FR-8.2**: No Supabase client method may be called when the client is `null`; guard with `if (supabase)`.

---

## 4. Non-Functional Requirements (NFR)

- **NFR-1 — Zero UI redesign**: The existing appearance, layout, styling, text, and UX of every page MUST remain byte-for-byte identical. Only *data plumbing* changes.
- **NFR-2 — Backend-agnostic**: FastAPI backend code in `/backend` MUST NOT be modified. The FastAPI <-> Supabase integration is explicitly OUT OF SCOPE for this task.
- **NFR-3 — TypeScript strict**: All new/edited code MUST pass `tsc --noEmit` with the project's `tsconfig.json`.
- **NFR-4 — Lint-clean**: `next lint` / `eslint` MUST report zero new warnings or errors.
- **NFR-5 — Build-clean**: `next build` MUST succeed.
- **NFR-6 — Secrets hygiene**: No file under `/app`, `/components`, `/lib` may reference or embed a Supabase `service_role` key. Only `NEXT_PUBLIC_SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_ANON_KEY` may appear in frontend-facing code, and they MUST be read from `process.env`.
- **NFR-7 — RLS verification**: Every table (`profiles`, `analyses`, `analysis_inputs`, `analysis_trace`, `analysis_results`) has RLS `ENABLE`d and restrictive policies. Anon user MUST receive zero rows on any SELECT.
- **NFR-8 — Isolated services**: All Supabase persistence logic MUST live in a dedicated module (e.g. `lib/supabase/services.ts` or `lib/supabaseAnalysisService.ts`). Do NOT scatter `supabase.from(...)` calls across UI page components.

---

## 5. Constraints & Dependencies

### Hard Constraints
1. **Do not modify AI / FastAPI backend.** Backend code is out of scope.
2. **Do not redesign the UI.** If the existing visual design needs a Supabase-related status indicator, use a minimal non-intrusive badge only where one already exists.
3. **Keep mock mode working.** Any user without Supabase env vars set must get the same pre-Supabase demo experience.
4. **Never fabricate data.** If a metadata field (coordinates, CRS, sensor, confidence) isn't in the upstream result, store SQL NULL / empty JSONB — never invent values.
5. **No commits/pushes.** Git state MUST remain unchanged at the end.

### Dependencies
- `@supabase/supabase-js` `^2.112.4` — already in `package.json` ✅
- Supabase project at `NEXT_PUBLIC_SUPABASE_URL` with migration 001 already applied (tables + RLS policies exist)
- Storage buckets either created via a new migration (preferred) or checked/created at runtime

### Assumptions
- `.env.local` exists with valid `NEXT_PUBLIC_SUPABASE_URL` + `NEXT_PUBLIC_SUPABASE_ANON_KEY` (as per user instruction)
- Migration `20260901000001_create_satquery_core_tables.sql` has already been applied to the Supabase project
- The Supabase project is configured with email auth enabled; email confirmation is either disabled or handled per Supabase defaults

---

## 6. Acceptance Criteria

Every AC below is typed **rule** or **rubric** per Spec Mode vocabulary.

| ID | Type | Criterion | Evidence source |
|---|---|---|---|
| **AC-1** | `rule` | SignUp with a new email → auth.users row + profiles row both exist; profiles.id = auth.uid(); profiles.email matches. | Supabase dashboard SQL query on auth.users JOIN public.profiles |
| **AC-2** | `rule` | SignUp → app redirects to `/` (home); refresh the page → session still authenticated; Logout → redirected to `/login` and unauthenticated. | Manual browser test + screenshot |
| **AC-3** | `rule` | SignIn with email of user A → listAnalysisHistory returns ONLY analyses where user_id = A's id. SignIn as B → same test, zero rows from A. | RLS test: two accounts + Supabase SQL console |
| **AC-4** | `rule` | Anon (signed out) request to `supabase.from('analyses').select('*')` → 0 rows (not error). | Supabase anon-key test via curl or JS snippet |
| **AC-5** | `rule` | Run one full analysis in `single_image` mode → after completion: 1 analyses row + N analysis_inputs rows + M analysis_trace rows (= number of actual steps executed) + 1 analysis_results row; all rows share analysis_id. | Supabase SQL query after analysis |
| **AC-6** | `rule` | `analysis_trace` rows are written only as steps actually transition (no pre-populated "done" rows written upfront). step column is sequential 1..M. | Check trace rows mid-execution (before pipeline completes) |
| **AC-7** | `rule` | In `analysis_results`, if the original result had `confidence = null`, the DB column is SQL NULL (not 0, not 0.0). | SQL query: `SELECT confidence IS NULL FROM analysis_results ...` |
| **AC-8** | `rule` | Storage object paths in `satquery-inputs` match `inputs/{user_id}/{analysis_id}/...`; a user cannot list/read a different user's object (policy blocks it). | Supabase storage policy test + signed URL attempt |
| **AC-9** | `rule` | Remove `.env.local` (or unset Supabase env vars) → reload app → Login page uses mock auth; history loads from in-memory store; no console errors from supabase=null. | Local test with modified env |
| **AC-10** | `rule` | `next build` exits 0. | CI-style build log |
| **AC-11** | `rule` | `next lint` exits 0. | ESLint output |
| **AC-12** | `rule` | Grep for `service_role`, `SUPABASE_SERVICE`, `sb_secret_`, or any string matching a Supabase secret key pattern across all of `/app`, `/components`, `/lib` → 0 matches. | `grep -R` output |
| **AC-13** | `rubric` | **Isolation of Supabase concerns (0–2)**. `2`: all Supabase data-access calls routed through a single dedicated service module; zero `supabase.from(...)` in page components. `1`: one or two page components have direct supabase calls but they are minimal and well-guarded. `0`: supabase calls scattered across UI. | Code review |
| **AC-14** | `rubric` | **Mock-mode fidelity (0–2)**. `2`: in DEMO_MODE, every new code path falls through cleanly to pre-existing mock behaviour; identical UX and data. `1`: minor cosmetic difference (e.g. one extra toast) but behaviour correct. `0`: mock mode is broken or requires env configuration to avoid crashing. | Manual mock-mode run |
