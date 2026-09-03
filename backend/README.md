# SatQuery-AI Backend — Phase 3 (Real ML Models & Dataset Pipeline)

FastAPI backend for SatQuery-AI (SIH project). Features real bi-temporal change detection (trained Siamese U-Net on LEVIR-CD), SmolVLM VQA adapter, geospatial metadata extraction, and structured deterministic execution tracing.

## Architecture & Real ML Dispatch

```
Frontend (Next.js)
    │  HTTP / JSON (contracts in lib/types/analysis.ts)
    ▼
FastAPI (app/main.py)
    │  endpoints: upload, analysis, datasets, tools, benchmark, health
    ▼
Orchestrator (app/services/orchestrator.py)
    ├─ Task Classifier  → keyword-based deterministic routing
    ├─ Tool Registry    → 7 specialist tool definitions
    └─ Tool Selection   → mode-aware mapping tasks → tool ids
    ▼
Inference & Specialist Services
    • CHANGE_DETECTOR     → Real SiameseUNet (`checkpoints/best_model.pt`) with sliding window
    • RS_VQA              → Real SmolVLM adapter (`vqa_service.py`) / fallback
    • RS_GROUNDING        → Structured mock service
    • RS_CAPTION          → Structured mock service
    • CHANGE_VQA          → Structured mock service
    • OPTICAL_SAR_ANALYZER→ Structured mock service
    • SPATIAL_ANALYZER    → Structured mock service
    ▼
Storage: SQLite (SQLAlchemy) + local filesystem uploads + optional Firebase Firestore/Storage
```

### Real vs Mock Distinction
- **Real ML Change Detection**: When `CHANGE_DETECTION_CHECKPOINT=./checkpoints/best_model.pt` is present, bi-temporal analysis executes the trained Siamese U-Net across full-resolution imagery via 256×256 sliding windows with 32px overlap smoothing.
- **Classical Difference Fallback**: If checkpoint is missing, gracefully runs CPU perceptual differencing.
- **Mock Specialists**: Unimplemented tools explicitly output structured mock results labeled `[MOCK]` to ensure scientific validity without fabricating fake confidence values.

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

## Phase 3 — Real Dataset Integration & Change Detection Pipeline

SatQuery-AI connects real remote sensing datasets and model pipelines without requiring large local GPU hardware for development.

### Supported Datasets

#### 1. LEVIR-CD (Bi-Temporal Building Change Detection)
Extracted dataset directory containing train/val/test splits with 1024×1024 optical RGB images:
```
<LEVIR_CD_ROOT>/
├── train/
│   ├── A/      # 445 pre-change RGB images (PNG)
│   ├── B/      # 445 post-change RGB images (PNG)
│   └── label/  # 445 binary change masks (PNG, 0=no change, 255=change)
├── val/
│   ├── A/      # 64 pre-change RGB images
│   ├── B/      # 64 post-change RGB images
│   └── label/  # 64 binary change masks
└── test/
    ├── A/      # 128 pre-change RGB images
    ├── B/      # 128 post-change RGB images
    └── label/  # 128 binary change masks
```
- **Validator**: `validate_levir_cd(root)` verifies directory structure, file counts, matching filenames, sample readable formats, and binary masks {0, 255} without loading all images into RAM.
- **PyTorch Dataset**: `LEVIRDataset` (`app.services.datasets.levir_cd.py`) loads images lazily, crops to configurable patch sizes (e.g. 256×256), normalizes masks to {0.0, 1.0}, and applies data augmentations during training.

#### 2. BigEarthNet (VQA & Land Cover Annotations)
- `BigEarthNet.txt.parquet`: 9,553,962 rows of VQA-style question-answer text annotations across 13 columns (`ID`, `patch_id`, `s1_name`, `input`, `output`, `type`, `category`, `split`, `country`, `season`, `climate_zone`, `latitude`, `longitude`).
- `metadata.parquet`: 480,038 rows of patch-level metadata including multi-label land-cover classifications.
- **Important**: These parquet files contain text annotations and metadata, **NOT** image pixel rasters. The service uses `pyarrow` to inspect schema, row counts, and summary distributions memory-efficiently.

### Dataset Environment Configuration

Configure your dataset and model paths in `backend/.env`:
```env
# Path to extracted LEVIR-CD directory (or set LEVIR_CD_ROOT)
LEVIR_CD_DATASET_PATH=/path/to/LEVIR-CD

# Parquet annotation and metadata files (optional)
BIGEARTHNET_TXT_PARQUET=/path/to/BigEarthNet.txt.parquet
BIGEARTHNET_METADATA_PARQUET=/path/to/metadata.parquet

DATASET_CACHE_DIR=./data/cache
CHECKPOINT_DIR=./checkpoints

# Path to trained SiameseUNet checkpoint for real change detection inference:
CHANGE_DETECTION_CHECKPOINT=./checkpoints/best_model.pt
```

### Dataset API Endpoints

- `GET /api/datasets/status` — Quick status check of dataset and checkpoint configurations.
- `GET /api/datasets/levir-cd/validate` — Validates LEVIR-CD split file alignment and format integrity.
- `GET /api/datasets/bigearthnet/summary` — Memory-safe summary of BigEarthNet VQA annotations & metadata schema.

### Siamese U-Net Model & Trained Checkpoints

- **Model**: `SiameseUNet` (`app.services.models.siamese_unet.py`)
- **Inputs**: 6-channel concatenated RGB image pair `[img_A, img_B]` of shape `(B, 6, H, W)`.
- **Architecture**: 3-level encoder with skip connections + upsampling decoder + 1-channel logit head (~490K trainable parameters, ~1.48 MB checkpoint size).
- **Loss**: Differentiable BCE + Dice combined loss for class-imbalanced change detection.
- **Trained Experiments Summary**:
  - **Baseline (50 Epochs)**: Best checkpoint at Epoch 48 (Val IoU: `0.4875`, Val F1: `0.5429`, Precision: `0.9691`, Accuracy: `97.63%`).
  - **Experiment 01 (Hybrid Imbalance Loss, 50 Epochs)**: Best Validation F1 = `0.6245`, Best Validation IoU = `0.4638`.
  - **Experiment 01 Final Test Split Evaluation (128 samples, threshold 0.70)**:
    - **Test Micro IoU**: `58.06%` (`0.5806`)
    - **Test Micro F1/Dice**: `73.47%` (`0.7347`)
    - **Test Precision**: `73.62%` (`0.7362`)
    - **Test Recall**: `73.32%` (`0.7332`)
    - **Test Pixel Accuracy**: `97.34%` (`0.9734`)
  - Checkpoint files:
    - `checkpoints/best_model.pt`: Best model weights (~1.48 MB).
    - `checkpoints/last_model.pt`: Epoch 50 weights with optimizer & scheduler state (~1.48 MB).
    - `checkpoints/training_log.json`: 50-epoch loss and evaluation history.
- **Inference Mode Dispatcher**:
  - If `CHANGE_DETECTION_CHECKPOINT` is configured → runs model tiled sliding-window inference with 32px overlap averaging.
  - If unset or missing → transparently falls back to the CPU classical pixel-difference baseline.
  - Results are never fabricated; execution mode is explicitly declared in evidence and stats.

### Running the CPU Smoke Test

To verify the training and inference pipeline end-to-end on CPU with zero GPU requirements:
```powershell
python scripts/train_change_detector.py --smoke-test
```

### Continuing / Resuming Model Training

To resume training from the 50-epoch checkpoint on a GPU/cloud machine (Colab, Kaggle, Cloud VM):
```powershell
python scripts/train_change_detector.py `
    --data-root "/path/to/LEVIR-CD" `
    --resume ./checkpoints/last_model.pt `
    --epochs 100 `
    --batch-size 4 `
    --img-size 256 `
    --lr 0.001 `
    --checkpoint-dir ./checkpoints
```

### Evaluating Trained Checkpoint

To evaluate the best checkpoint on validation and test splits without training:
```powershell
python scripts/train_change_detector.py `
    --data-root "/path/to/LEVIR-CD" `
    --resume ./checkpoints/best_model.pt `
    --eval-only
```

### Running Test Suite

```powershell
pytest tests/ -v
```
All 88 unit and integration tests run on CPU in under 10 seconds.
