# SatQuery-AI — Supabase Integration Implementation Tasks

Task dependencies: `T1 → T2 → T3 → T4 → T5 → T6 → T7 → T8 → T9` (serial pipeline with some parallelisable sub-tasks marked).

---

## Task 1: Supabase Dedicated Persistence Service Module

**Status**: `pending`
**Priority**: `high`
**Acceptance Criteria covered**: AC-13 (isolation of concerns)

### Scope
Create a single new module `lib/supabase/services.ts` (DO NOT scatter Supabase calls anywhere else) that exposes:

```ts
export interface SupabaseAnalysisService {
  // Auth & Profiles
  ensureProfileForUser(userId: string, email: string, fullName?: string): Promise<void>;

  // Analysis lifecycle
  saveAnalysisStarted(input: {
    analysisId: string;
    userId: string;
    query: string;
    analysisType: AnalysisMode;
    images: UploadedImage[];
  }): Promise<void>;

  saveTraceStep(input: {
    analysisId: string;
    stepIndex: number;  // 1-based
    step: ExecutionStep;
  }): Promise<void>;

  saveAnalysisCompleted(input: {
    analysisId: string;
    result: AnalysisResult;
  }): Promise<void>;

  // Queries
  listHistory(filters: HistoryFilters, userId: string): Promise<{ items: AnalysisResult[]; total: number }>;
  getAnalysis(id: string, userId: string): Promise<AnalysisResult | null>;
  deleteAnalysis(id: string, userId: string): Promise<void>;

  // Storage
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
  getSignedUrl(bucket: 'inputs'|'results'|'reports', path: string): Promise<string | null>;
}
```

Also export a singleton factory:
```ts
export const supabaseAnalysisService: SupabaseAnalysisService | null = HAS_SUPABASE ? ... : null;
```

And a "noop" instance for when HAS_SUPABASE=false — every method is a no-op or returns null/empty. This way callers never need `if (service)` guards in multiple places.

### Implementation Notes
- Map `AnalysisMode` → DB `analysis_type` column enum exactly as in the migration.
- Map `UploadedImage.metadata` → `analysis_inputs.metadata` JSONB; extract flat columns (crs, resolution, acquisition_date, sensor, modality, format, filename) from ImageMetadataType where available.
- For `analysis_trace.tool_name` column: use `step.title` (max 255 chars; truncate if needed). `parameters` = `step.meta ?? {}`. `output` = `{ detail, startedAt, completedAt }`.
- For `analysis_results.evidence` = `boundingBoxes` mapped to JSONB array of `{x,y,width,height,label,confidence}`.
- For `analysis_results.statistics` = `{}` if no change-det stats exist.
- For `analysis_results.result_metadata` = `{ detectedTasks, toolInvocations, schemaVersion: 1 }`.
- ALL methods must have internal `if (!supabase) return ...` short-circuits.
- ALL errors MUST be caught and logged (`console.error`) but MUST NOT propagate to break the user workflow (persistence failures should be silent, not crash analysis).

### Files created
- `lib/supabase/services.ts`

### Local Test Requirements (TR)
| ID | Type | Requirement |
|---|---|---|
| T1-TR1 | `rule` | `tsc --noEmit` passes for the new module with no new TS errors. |
| T1-TR2 | `rule` | With `HAS_SUPABASE=false`, the noop service instance every method returns without throwing. |
| T1-TR3 | `rule` | No `supabase.from(...)` or `supabase.storage.from(...)` calls exist outside this module. |

### Completion Evidence
(To be filled during Implement phase)

---

## Task 2: Auth — Profile Auto-Creation on Signup & Session Guard Fixes

**Status**: `pending`
**Priority**: `high`
**Depends on**: T1
**Acceptance Criteria covered**: AC-1, AC-2, FR-1, FR-2

### Scope
- **`lib/authService.ts`**: In `SupabaseAuthService.signUp()`, after a successful `signUp` that produces a user (regardless of whether auto-confirmation created a session), call `supabaseAnalysisService.ensureProfileForUser(userId, email, fullName)`. Handle the case where the profile insert races or is otherwise done by a trigger — upsert semantics preferred (`ON CONFLICT (id) DO NOTHING`).
- **`lib/authService.ts`**: In `getCurrentSession()` and `onAuthStateChange()`, map Supabase session to AuthSession for INITIAL_SESSION so the guard in authContext.tsx redirects correctly.
- **`lib/authContext.tsx`**: Ensure `onAuthStateChange()` handler in useEffect correctly updates `user` from Supabase session (already partial — verify `SIGNED_IN` after signup sets user; if `auth.uid()` from mock is overriding, fix ordering). The existing mock `login()` function in AuthContext is for OAuth buttons only — verify it doesn't interfere with real signIn.
- **`components/layout/TopBar.tsx`**: No UI changes — just confirm logout flow uses `authSignOut()` correctly (it already does; verify no regression).

### Files changed
- `lib/authService.ts` (edit)
- `lib/authContext.tsx` (edit — minimal, if needed)

### Local Test Requirements (TR)
| ID | Type | Requirement |
|---|---|---|
| T2-TR1 | `rule` | After Supabase signUp() in test: `SELECT count(*) FROM public.profiles WHERE id = auth_uid` → 1. |
| T2-TR2 | `rule` | After signUp, refresh page → still authenticated (session persisted via Supabase). |
| T2-TR3 | `rule` | Click logout in TopBar → user is null, redirected to /login, localStorage supabase session cleared. |
| T2-TR4 | `rule` | In DEMO_MODE env, authService uses MockAuthService — no Supabase auth call is made. |

### Completion Evidence
(To be filled during Implement phase)

---

## Task 3: Storage Buckets & Policies (Second Migration)

**Status**: `pending`
**Priority**: `high`
**Depends on**: T1
**Acceptance Criteria covered**: AC-8, FR-7

### Scope
Create a NEW migration file under `supabase/migrations/` (do NOT edit migration 001). Name it with a timestamp before the current one OR just use a higher timestamp than the first.

The migration MUST:
1. Create 3 private buckets via `insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types) values ... ON CONFLICT DO NOTHING;`
   - `satquery-inputs` (private, max 500MB, allow image/* + application/json + image/tiff + image/x-hdf5 + application/netcdf)
   - `satquery-results` (private, max 50MB)
   - `satquery-reports` (private, max 20MB, application/json + application/pdf)
2. Create storage.object-level RLS policies (Supabase-specific) that:
   - For each bucket, allow `SELECT` only when `(storage.foldername(name))[1] = auth.uid()::text` (user owns the first path segment)
   - Allow `INSERT` under same ownership condition
   - Allow `UPDATE` and `DELETE` under same ownership condition
3. Ensure no anonymous policy exists — anon has NO access to any bucket.

### Files created
- `supabase/migrations/20260905000000_create_storage_buckets_and_policies.sql`

### Local Test Requirements (TR)
| ID | Type | Requirement |
|---|---|---|
| T3-TR1 | `rule` | Migration file is valid SQL; running it against Supabase via Dashboard SQL Editor produces no errors. |
| T3-TR2 | `rule` | Upload object to satquery-inputs as user A. Attempt to GET signed URL (or list) as user B → blocked by policy. |

### Completion Evidence
(To be filled during Implement phase)

---

## Task 4: Wire Persistence into Analysis Flows (New Analysis Submission)

**Status**: `pending`
**Priority**: `high`
**Depends on**: T1, T2
**Acceptance Criteria covered**: AC-3, AC-5, AC-6, FR-3.1, FR-3.2, FR-4, FR-5.1

### Scope
Create an `AppPersistenceBridge` module (`lib/persistenceBridge.ts`) that exposes thin wrappers around the `api` object, calling both the underlying `api` method AND the `supabaseAnalysisService` method when the service is non-null.

The bridge MUST:
- NOT change the `SatQueryApi` interface signature — it can expose the same interface but just decorate.
- `uploadImage()`: after `api.uploadImage()` returns the `UploadedImage`, also call storage upload on the same file and tag the image with a `storagePath` for later (hold in module cache keyed by `uploadedImage.id` — since UploadedImage is a plain type, extend via a WeakMap).
- `submitAnalysis()`: after `api.submitAnalysis()` returns `{ analysisId }`, call `saveAnalysisStarted()` with the images + query + mode. Also at this point we know the `userId` from auth context — read it via a `getUserId(): string | null` helper exported from `authService.ts`/`authContext.tsx`.
- `streamExecutionTrace()`: wrap the original subscriber. For every trace update callback, for each step that changed status, call `saveTraceStep(analysisId, stepIndex, step)` ONCE per transition. De-duplicate via a `Map<analysisId, Set<stepIndex+status>>` cache so repeated identical callbacks don't write duplicate rows.

### Files created / changed
- `lib/persistenceBridge.ts` (new)
- `lib/api/index.ts` (edit — minimal: export a `persistedApi` wrapper that replaces `api` if persistence is active; otherwise `api` as before. Keep `api` symbol intact; pages can import either unchanged OR we make `api` itself point to the decorated one — **preferred** is to make `api = persistenceDecorator(underlyingApi)` so all existing call-sites gain persistence with ZERO page component changes).

### Local Test Requirements (TR)
| ID | Type | Requirement |
|---|---|---|
| T4-TR1 | `rule` | After `submitAnalysis()` returns (pre-stream), 1 analyses row exists with status='queued' and correct user_id. |
| T4-TR2 | `rule` | Mid-stream (after 2 steps transition to 'done'), exactly 2 trace rows have status='done'. Remaining steps (if any) have status as actually seen (not pre-filled). |
| T4-TR3 | `rule` | In DEMO_MODE mode, bridge calls skip Supabase service entirely — no write attempts. |

### Completion Evidence
(To be filled during Implement phase)

---

## Task 5: Wire Persistence into History & Detail Pages

**Status**: `pending`
**Priority**: `high`
**Depends on**: T4
**Acceptance Criteria covered**: AC-3, AC-5, FR-3.3, FR-3.4, FR-3.5, FR-6

### Scope
- In the persistence decorator:
  - `listAnalysisHistory()`: Try `supabaseAnalysisService.listHistory()` first if available; fall back to underlying api (mockApi store / FastAPI) if Supabase returns empty or fails. Merge strategy: **Supabase wins if non-empty**, because it's the long-term archive. But if user has in-flight analyses NOT yet persisted, include from underlying store too.
  - `getAnalysis()`: Same merge pattern — prefer Supabase, fall back to underlying api.
  - `deleteAnalysis()`: Call both Supabase delete AND underlying store delete.
- When analysis reaches 'completed'/'failed' in stream, decorator MUST call `saveAnalysisCompleted()` with the final `AnalysisResult`.

### Files changed
- `lib/persistenceBridge.ts` (edit, completing T4 module)
- `lib/api/index.ts` (edit — ensure `api` symbol is decorated)

### Local Test Requirements (TR)
| ID | Type | Requirement |
|---|---|---|
| T5-TR1 | `rule` | After 1 full analysis completes, refresh page → history row appears via `listAnalysisHistory()` Supabase fetch. |
| T5-TR2 | `rule` | Click history row → `getAnalysis(id)` loads correct data; images, trace steps, answer all populated. |
| T5-TR3 | `rule` | Delete analysis → row gone from history page AND Supabase SQL `SELECT * FROM analyses WHERE id=...` returns 0 rows. |
| T5-TR4 | `rule` | If original result confidence=null → DB stores NULL (AC-7). |

### Completion Evidence
(To be filled during Implement phase)

---

## Task 6: Storage Upload for Inputs & Results

**Status**: `pending`
**Priority**: `medium`
**Depends on**: T3, T4
**Acceptance Criteria covered**: AC-8, FR-7

### Scope
Finish implementing the storage methods in `supabaseAnalysisService`:

- `uploadInputFile`: Upload to `satquery-inputs/{userId}/{analysisId}/{role}-{sanitizedFilename}` via `supabase.storage.from('satquery-inputs').upload(path, file, { cacheControl: '31536000', upsert: false })`. Return storage path.
- `uploadResultBlob`: Upload to `satquery-results/{userId}/{analysisId}/{blobName}`.
- `getSignedUrl`: Call `createSignedUrl(path, expiresIn=3600)`.
- In the persistence bridge, hook `uploadImage()` to actually call storage upload (not just cache path), and store the path in `analysis_inputs.storage_path` column on `saveAnalysisStarted()`.
- For the "Export Report" button in `app/analysis/[id]/page.tsx`, after the browser download also call `uploadResultBlob` to archive the report to `satquery-reports/{userId}/{analysisId}/SatQuery_${id}_report.json`. Do NOT change the user-visible download behaviour; the archive save is a silent side-effect.

### Files changed
- `lib/supabase/services.ts` (complete storage impl)
- `lib/persistenceBridge.ts` (wire storage upload calls)
- `app/analysis/[id]/page.tsx` (edit `handleDownloadReport` — add silent `supabaseAnalysisService.uploadResultBlob(...)` call; zero UI changes)

### Local Test Requirements (TR)
| ID | Type | Requirement |
|---|---|---|
| T6-TR1 | `rule` | After uploadImage + submitAnalysis, object exists in Supabase Storage at the expected path. |
| T6-TR2 | `rule` | Second user cannot list/read the first user's objects (policy enforcement — storage returns 403 or empty). |
| T6-TR3 | `rule` | After Export Report click → report JSON saved to satquery-reports bucket AND browser download still works. |

### Completion Evidence
(To be filled during Implement phase)

---

## Task 7: Mock / DEMO Mode Hardening

**Status**: `pending`
**Priority**: `high`
**Depends on**: T4, T5
**Acceptance Criteria covered**: AC-9, AC-14, FR-8, NFR-1

### Scope
- Audit ALL new code paths (services.ts, persistenceBridge.ts, authService.ts changes) for `if (HAS_SUPABASE)` / null-service guards.
- Verify that with `NEXT_PUBLIC_SUPABASE_URL=''` (or absent):
  - Login page still works with auto-fill demo credentials
  - OAuth buttons still call mock `login()` in authContext
  - New analysis runs, history shows, trace animates, detail page loads — all from in-memory store
  - No console errors of the form "Cannot read property 'from' of null"
- Ensure that in the persistence bridge, if the underlying `api` call throws, we do NOT swallow it — let propagate to existing toast handlers. Persistence side-effects MUST NOT throw, ever — wrap in try/catch inside the bridge.

### Files changed
- `lib/persistenceBridge.ts` (add robust try/catch + logging)
- `lib/supabase/services.ts` (audit guards)
- `lib/authService.ts` (audit guards)

### Local Test Requirements (TR)
| ID | Type | Requirement |
|---|---|---|
| T7-TR1 | `rule` | Run app with empty SUPABASE env → full demo flow: sign in → new analysis → history → detail → export. Zero console errors. |
| T7-TR2 | `rule` | Persistence-side failure (simulate by mock-throw in service method) → user-facing analysis still succeeds; error logged to console only. |

### Completion Evidence
(To be filled during Implement phase)

---

## Task 8: Typecheck, Lint, Build

**Status**: `pending`
**Priority**: `high`
**Depends on**: T1–T7 (all preceding)
**Acceptance Criteria covered**: AC-10, AC-11, NFR-3, NFR-4, NFR-5

### Scope
Run and fix:
1. `npm run lint` → fix any new ESLint warnings
2. TypeScript strict check via: `npx tsc --noEmit` → fix any new TS errors
3. `npm run build` → fix any Next.js build errors

Rules:
- If a new import causes "React not defined" or similar, import correctly.
- If a JSONB-to-TS-type mapping is inexact, add explicit type casts / wrappers.
- Do NOT weaken TypeScript strictness globally.

### Files changed
- Any files needing TS/ESLint fixes (usually small edits).

### Local Test Requirements (TR)
| ID | Type | Requirement |
|---|---|---|
| T8-TR1 | `rule` | `npm run lint` exits 0. |
| T8-TR2 | `rule` | `npx tsc --noEmit` exits 0. |
| T8-TR3 | `rule` | `npm run build` exits 0 with successful build artifact list. |

### Completion Evidence
(To be filled during Implement phase)

---

## Task 9: Security Verification & Secrets Audit

**Status**: `pending`
**Priority**: `high`
**Depends on**: T8
**Acceptance Criteria covered**: AC-4, AC-12, NFR-6, NFR-7

### Scope — Manual + automated checks:
1. **Secrets audit** — Grep across `/app`, `/components`, `/lib`, `/public`, `/pages` (if any) for patterns:
   - `sb_secret_`
   - `service_role`
   - `SUPABASE_SERVICE`
   - `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9` (typical JWT — catches leaked anon keys too — but ANON is allowed; just verify it's the NEXT_PUBLIC one from env, not a hardcoded copy)
   - Any string in code starting with `sb_private_` or `sk_`
2. **RLS verification** — For every public table, verify `relrowsecurity = true` and no permissive policies for anon (via a SQL check script, or at least confirm the migration has RLS enabled on all 5 tables).
3. **Frontend key check** — `lib/supabase.ts` must ONLY read `process.env.NEXT_PUBLIC_*`. Confirm there's no import of a `.env` file that might bring in server vars to client.
4. **Storage buckets privacy** — Confirm `public = false` in bucket metadata (migration sets this).

### Files changed
- None expected. If any issue is found, fix it in-place and record.

### Local Test Requirements (TR)
| ID | Type | Requirement |
|---|---|---|
| T9-TR1 | `rule` | grep returns 0 matches for any secret/service_role key pattern. 0 instances of hardcoded `sb_publishable_` string (only read via env). |
| T9-TR2 | `rule` | Anon request: `const { data } = await createClient(url, anonKey).from('analyses').select('*')` → data = [] (not error, not rows). |
| T9-TR3 | `rule` | RLS enabled boolean confirmed for all 5 tables. |

### Completion Evidence
(To be filled during Implement phase)

---

## Task 10: Final End-to-End Integration Runs (Manual Checklists)

**Status**: `pending`
**Priority**: `high`
**Depends on**: T9
**Acceptance Criteria covered**: AC-1, AC-2, AC-3, AC-5, AC-6, AC-7, AC-8, AC-9

### Scope
Execute each workflow against a running dev server (`npm run dev`):

1. **Wizard A — Supabase mode, Signup → Signout → Signin cycle**
   - New email, password ≥6 chars → verify profile & auth rows.
   - Refresh → still logged in.
   - Logout → /login page, session gone.
   - Sign back in → works.

2. **Wizard B — Supabase mode, all 3 analysis modes end-to-end**
   - single_image: Load Demo Preset → Run → wait completion → verify analyses + inputs + trace + results rows (SQL counts).
   - bi_temporal: same.
   - optical_sar: same.
   - For each, confirm History page shows 3 rows.
   - Open one detail → data correct → Delete → gone.

3. **Wizard C — DEMO_MODE parity**
   - Unset Supabase env (or edit .env.local to empty) → restart → same flows work identically.

### Files changed
- None expected.

### Local Test Requirements (TR)
| ID | Type | Requirement |
|---|---|---|
| T10-TR1 | `rule` | All 3 wizards pass with no uncaught exceptions / 5xx responses. |
| T10-TR2 | `rubric` | Mock vs Supabase UX parity (0-2). 2 = identical UI. 1 = 1 trivial difference. 0 = broken. |

### Completion Evidence
(To be filled during Implement phase)
