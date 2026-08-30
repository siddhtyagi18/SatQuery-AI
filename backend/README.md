# SatQuery-AI Backend — Phase 1

FastAPI backend for SatQuery-AI (SIH project). This is Phase 1: a CPU-only
foundation with mock specialist services, ready to plug in real ML models in
Phase 2.

## Architecture

```
Frontend (Next.js)
    │  HTTP / JSON (contracts in lib/types/analysis.ts)
    ▼
FastAPI (app/main.py)
    │  endpoints: upload, analysis, tools, benchmark, health
    ▼
Orchestrator (app/services/orchestrator.py)
    ├─ Task Classifier  → keyword-based deterministic routing
    ├─ Tool Registry    → 7 replaceable specialist tool definitions
    └─ Tool Selection   → mode-aware mapping tasks → tool ids
    ▼
Specialist Services (app/services/mock_specialists.py)
    • RS_VQA              RS_CAPTION
    • RS_GROUNDING        CHANGE_DETECTOR
    • CHANGE_VQA          OPTICAL_SAR_ANALYZER
    • SPATIAL_ANALYZER
    ▼
Storage: SQLite (SQLAlchemy) + local filesystem for uploads
```

All specialist tools in Phase 1 return structured **mock** results clearly
marked with a `[MOCK — Phase 1, not real inference]` prefix. Confidence scores
are fixed placeholders, never fabricated scientific results.

## Project Structure

```
backend/
├── app/
│   ├── main.py                FastAPI app + CORS + error handlers + routers
│   ├── config.py              Settings via .env / env vars
│   ├── database.py            SQLAlchemy engine + session
│   ├── models.py              DB tables (UploadedFile, Analysis, AnalysisImage, ExecutionStep)
│   ├── schemas.py             Pydantic types matching frontend contract (analysis.ts)
│   ├── crud.py                DB → API result conversion + history listing
│   ├── logging_setup.py       Structured logger (stdout + rotating file)
│   ├── routers/
│   │   ├── upload.py          POST /api/upload
│   │   ├── analysis.py        POST/GET/DELETE /api/analysis + /trace
│   │   ├── tools.py           GET /api/tools
│   │   ├── benchmark.py       GET /api/benchmark
│   │   ├── health.py          GET /health
│   │   └── files.py           GET /api/files/{id}  (preview images)
│   └── services/
│       ├── metadata.py        Upload validation + modality/format/CRS/bands extraction
│       ├── task_classifier.py Keyword-based task → task type classification
│       ├── tool_registry.py   7 tool definitions + list_tools()
│       ├── orchestrator.py    plan_execution() + execute_plan()
│       ├── mock_specialists.py 7 mock tool runners → marked mock results
│       └── trace.py           8-step execution trace lifecycle
├── tests/
│   ├── conftest.py            Test DB + fixtures
│   ├── test_basic.py          Health, tools, benchmark
│   ├── test_upload.py         Upload success / validation / modality detection
│   └── test_analysis.py       3 analysis modes, trace, history, delete, validation
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Requirements

- Python **3.11+** (tested on 3.13 Windows per project constraints)
- CPU only — no GPU / large AI model downloads required
- `rasterio` is optional but recommended for GeoTIFF metadata (it will
  gracefully degrade if not installed)

## Setup

### 1. Create a virtual environment (PowerShell on Windows)

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

If `rasterio` fails on Windows, install the wheel from
[Unofficial Windows Wheels](https://www.lfd.uci.edu/~gohlke/pythonlibs/#rasterio)
or use:

```powershell
pip install rasterio --only-binary :all:
```

If you can't install rasterio at all, simply skip it — PNG/JPEG uploads still
work (GeoTIFF metadata extraction degrades gracefully).

### 3. Configure environment

```powershell
copy .env.example .env
```

Defaults are sensible for local dev. Key variables:

| Variable              | Default                          | Purpose                                |
|-----------------------|----------------------------------|----------------------------------------|
| `DATABASE_URL`        | `sqlite:///./satquery.db`        | SQLite path                            |
| `UPLOAD_DIR`          | `./uploads`                      | Where uploaded files live on disk      |
| `MAX_UPLOAD_SIZE_MB`  | `512`                            | Max upload size per file               |
| `CORS_ORIGINS`        | `http://localhost:3000,…`        | Frontend origins allowed               |
| `PORT`                | `8000`                           | Uvicorn bind port                      |

### 4. Run the backend

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then open:

- Swagger UI: <http://localhost:8000/docs>
- ReDoc:     <http://localhost:8000/redoc>
- Health:    <http://localhost:8000/health>

### 5. Run tests

```powershell
pytest -v
```

The test suite uses a throwaway SQLite DB `test_satquery.db` and `test_uploads/`
directory, cleaned automatically between runs.

## Connecting the Frontend

The frontend `lib/config.ts` already points at `http://localhost:8000`. Swap
it to live mode:

```ts
// lib/config.ts
export const API_MODE: 'mock' | 'live' = 'live';
```

## API Reference

All response shapes exactly match the TypeScript contracts in
[`lib/types/analysis.ts`](../lib/types/analysis.ts).

### `GET /health`

```json
{ "status": "ok", "app_name": "SatQuery-AI Backend", "version": "0.1.0-phase1", "timestamp": "…" }
```

### `POST /api/upload`

Multipart form: `file` (required, `.tif/.tiff/.png/.jpg/.jpeg`) + `role`
(`single` | `before` | `after` | `optical` | `sar`).

**Example request (curl):**

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@optical_sample_2024-03-15.png;type=image/png" \
  -F "role=single"
```

**Example response:**

```json
{
  "id": "f0a3b…",
  "role": "single",
  "previewUrl": "/api/files/f0a3b…",
  "metadata": {
    "fileName": "optical_sample_2024-03-15.png",
    "fileFormat": "PNG",
    "modality": "optical",
    "modalityDetectionConfidence": 0.88,
    "acquisitionDate": "2024-03-15",
    "widthPx": 2048,
    "heightPx": 2048,
    "bandCount": 3,
    "crs": null,
    "gsdMeters": null,
    "fileSizeBytes": 1834922
  }
}
```

**Validation:**

- 400 on unsupported extension, oversized file (> 512 MB default)
- Extracts format, modality, date, width/height, bands (rasterio for GeoTIFF, Pillow otherwise)
- Extracts CRS, GSD (resolution), bounds when GeoTIFF contains them

### `POST /api/analysis`

Body:

```json
{
  "mode": "single_image",
  "imageIds": ["id-from-upload-1"],
  "query": "What land cover types are visible and locate all buildings?"
}
```

Mode → expected image count:

- `single_image` → 1 image
- `bi_temporal` → 2 images (`before`, `after`)
- `optical_sar` → 2 images (`optical`, `sar`)

Response (synchronous in Phase 1 — the pipeline runs inline):

```json
{ "analysisId": "analysis-4af2…" }
```

### `GET /api/analysis/{id}`

Returns the full `AnalysisResult` object including `answerText`, `confidence`,
`boundingBoxes`, `changeMap`, `detectedTasks`, `toolInvocations`,
`executionTrace`, and the extra Phase-1 fields `task`, `selectedTools`,
`evidence`, `analysisStatus`.

All answers are clearly prefixed with `[MOCK — Phase 1, not real inference]`.
Confidence values are fixed placeholder means of the participating tools
(never fabricated scientific scores).

### `GET /api/analysis/{id}/trace`

Returns the standalone `ExecutionTrace` object: 8 steps each with `status`,
`startedAt`, `completedAt`, `detail`, optional `meta`:

```
step-1 Query Received   → step-2 Validation  → step-3 Classification
→ step-4 Tool Selection → step-5 Parameters  → step-6 Processing
→ step-7 Aggregation    → step-8 Completion
```

### `GET /api/analysis`

Query params: `mode`, `status`, `dateFrom`, `dateTo`, `minConfidence`, `page`,
`pageSize` (default 20). Returns:

```json
{ "items": [ /* AnalysisResult[] */ ], "total": 42 }
```

### `DELETE /api/analysis/{id}` → 204 No Content

### `GET /api/tools`

7 specialist tools + orchestrator metadata. All tools report `status: "mock"`
in Phase 1 (except the orchestrator which is `available`).

### `GET /api/benchmark`

12 benchmark metric rows (Accuracy, BLEU-4, mAP, IoU, F1, …) all with
`value: null`, `evaluatedAt: null` — deliberately not evaluated.

### `GET /api/files/{id}`

Serves an uploaded file as an HTTP response (for frontend previews of PNG/JPEG).
GeoTIFF previews return `previewUrl: null` from the upload endpoint.

## Example End-to-End Flow (HTTPie)

```bash
# 1. Upload optical image
IMG_ID=$(http -f POST :8000/api/upload file@optical_sample.png role=single | jq -r .id)

# 2. Run analysis
AID=$(http POST :8000/api/analysis mode=single_image "imageIds:=[\"$IMG_ID\"]" \
  query='What land cover types are visible? Locate buildings.' | jq -r .analysisId)

# 3. Get result
http GET :8000/api/analysis/$AID

# 4. Inspect just the trace
http GET :8000/api/analysis/$AID/trace
```

## Result Format Contract

From `GET /api/analysis/{id}` the top-level keys are:

```
answer          → answerText
task            → primary detected task (e.g. "vqa")
selectedTools   → ["rs_vqa", "rs_grounding", …]
evidence        → list of strings (all marked mock in Phase 1)
confidence      → float 0-1 or null
executionTrace  → {steps: […], totalElapsedMs: n, overallStatus: "completed"}
analysisStatus  → "queued" | "processing" | "completed" | "failed"
```

## Error Handling

- 400: upload validation / missing required images / empty query
- 404: missing uploaded file id or analysis id
- 500: unhandled exceptions (logged to stdout + `logs/satquery.log`)
- CORS: origins allowed via `CORS_ORIGINS` env (default `localhost:3000,8080`)

## Tests Summary

Tests live in `backend/tests/`. Run with:

```
pytest -v
```

Test coverage (Phase 1):

| Test File          | Count | Purpose                                              |
|--------------------|-------|------------------------------------------------------|
| `test_basic.py`    | 4     | `/health`, `/`, `/api/tools`, `/api/benchmark`       |
| `test_upload.py`   | 4     | PNG upload, bad extension, size limit, SAR modality  |
| `test_analysis.py` | 10    | single_image, bi_temporal, optical_sar, trace, history, validation, delete |
| **Total**          | **18**|                                                      |

## What's Next — Phase 2

This backend is designed so each mock specialist can be swapped **independently**
for a real implementation. The orchestrator, classifier, tool registry,
database, and execution trace do NOT need to change.

**Phase 2 checklist:**

1. **Real RS-VQA / Captioning model** — replace `_make_vqa_result` +
   `_make_caption_result` in `services/mock_specialists.py` with calls to a
   deployed model (e.g. RSVQA-HR fine-tune, LLaVA-RS adapter). Keep the same
   `(query, mode) -> {answer, confidence, evidence}` signature.

2. **Real Grounding detector** — replace `_make_grounding_result` with a
   Grounding DINO / RS-DINO variant producing actual bboxes on the image
   raster; return the same `{answer, confidence, bounding_boxes, evidence}` shape.

3. **Real Bi-temporal Change Detection** — plug in CVA + deep change model
   (e.g. BIT, SNUNet, ChangeFormer). Produce a per-pixel mask PNG and save it
   to disk so `changeMap.overlayUrl` can point at it.

4. **Change-VQA head** — run change mask + image pair through a VLM to answer
   the user's natural-language questions about change.

5. **Optical+SAR Fusion** — real phase-correlation co-registration, weighted
   stack fusion, SAR-unique backscatter feature extraction.

6. **Spatial Analyzer** — shapely + rasterio zonal stats producing real area,
   perimeter, proximity numbers from bboxes/masks.

7. **Swap tool `status`** in `tool_registry.py` from `"mock"` → `"available"`
   as each specialist is implemented (frontend's `ToolRegistry` page will
   render the badge colour change).

8. **Populate benchmark** — once real specialists exist, run each one against
   public RS datasets (RSVQA-HR, RSITMD, DIOR-RSVG, LEVIR-CD, xBD) and fill
   the `value` + `evaluatedAt` fields in `routers/benchmark.py`.

9. **Asynchronous execution** — Phase 1 runs the whole pipeline inline in
   `POST /api/analysis`. Phase 2 should enqueue the work (TaskIQ, RQ, or
   FastAPI BackgroundTasks + a worker) and stream trace updates via SSE to
   match the frontend's `streamExecutionTrace` interface.

10. **Authentication + rate limiting** before any public deployment.

11. **Persistent object storage** — replace the local `UPLOAD_DIR` with S3 /
    MinIO once upload volume exceeds a single disk.

12. **Geospatial metadata enrichment** — parse additional GeoTIFF / GML tags
    (`TIFFTAG_DATETIME`, RPCs, sensor band descriptions) when available.

## Key Design Choices

- **Flat monolith** (no microservices) — keeps Phase 1 / 2 transitions simple;
  split into services later only if deployment requires it.
- **Frontend contract is the source of truth.** Pydantic schemas mirror
  `analysis.ts` exactly. Swapping the API mode in `lib/config.ts` to `live`
  should be a zero-code change on the UI.
- **Tool runners are pure functions.** They accept `(tool_id, query, mode, …)`
  and return `dict`. Easy to unit-test, easy to replace for ML-backed ones.
- **No fake science.** All answers, confidence scores, and bounding boxes are
  explicitly labelled MOCK. Benchmark metrics are `null` (not placeholder
  numbers) so the UI says "Not evaluated yet".
- **Trace-first design.** Every pipeline step is a persisted row in
  `execution_steps` so streaming animation (Phase 2 SSE) just needs to push
  row updates — not recompute history.
- **SQLite is the right choice** for Phase 1 (single-node, zero ops). Upgrade
  to Postgres only when concurrent writes / multi-node deployments happen.
