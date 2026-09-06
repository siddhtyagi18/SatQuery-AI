# SatQuery-AI 🌍🛰️

An AI-powered satellite imagery query and multi-temporal analysis platform for the Smart India Hackathon (SIH).

---

## 🚀 Project Status & Real ML Milestones

SatQuery-AI features an end-to-end pipeline spanning geospatial pre-processing, deterministic task routing, deep learning change detection, multi-modal VQA adapters, and interactive Next.js visualization.

### Real Training Experiments (LEVIR-CD Dataset)

We have completed **7 training experiments** on the LEVIR-CD building change detection benchmark using a progressively refined Siamese U-Net architecture (119,025–490,561 parameters):

| Experiment | Architecture | Params | Target Epochs | Actual Epochs | Loss | Best Val F1 | Best Val IoU | Status |
|---|---|---|---|---|---|---|---|---|
| **Baseline / Root** | Siamese U-Net | 490,561 | 50 | 50 | BCE + Dice | 0.5429 (E48) | 0.4875 (E48) | ✅ Complete |
| **experiment_01** | Siamese U-Net | 490,561 | 50 | 50 | Hybrid Imbalance | 0.6245 | 0.4638 | ✅ Complete |
| **experiment_02** | Siamese U-Net (base=16) | 119,025 | 5 | 5 | hybrid_v2 | 0.4116 (E4) | 0.2651 (E4) | ✅ Complete |
| **experiment_03** | Siamese U-Net (base=16) | 119,025 | 60 | 60 | hybrid_v2 | **0.6435** (E60) | **0.4776** (E60) | ✅ Complete 🏆 |
| **experiment_04** | Siamese U-Net (base=16) | 119,025 | 75 | 75 | hybrid_v2 | 0.6401 (E75) | 0.4738 (E75) | ✅ Complete |
| **experiment_A_mini** | Siamese U-Net (base=16) | 119,025 | — | — | — | — | — | ✅ Checkpoints present |
| **experiment_controlled** | Siamese U-Net (base=16) | 119,025 | 51 | 51 | hybrid_v2 | 0.6278 (E51) | 0.4604 (E51) | ✅ Complete |

**🏆 Best Model**: experiment_03 — Val F1 = **0.6435**, Val IoU = **0.4776** at epoch 60.

### Official Test Evaluation Benchmarks

#### Baseline Run (Root Checkpoint)
Evaluated across the full 128-sample LEVIR-CD test split using the validation-selected optimal threshold:
- **Test Micro IoU (Jaccard Index)**: **`58.06%`** (`0.5806`)
- **Test Micro F1 / Dice Score**: **`73.47%`** (`0.7347`)
- **Test Precision**: **`73.62%`** (`0.7362`)
- **Test Recall**: **`73.32%`** (`0.7332`)
- **Test Pixel Accuracy**: **`97.34%`** (`0.9734`)

#### experiment_03 Full Test Eval
Full 128-sample test split evaluation with threshold sweep is available in [`evaluation_results/experiment_03_eval/`](./evaluation_results/experiment_03_eval/).

#### experiment_04 Full Test Eval
Full 128-sample test split evaluation with threshold sweep is available in [`evaluation_results/experiment_04_eval/`](./evaluation_results/experiment_04_eval/).

Full evaluation logs, per-sample qualitative prediction PNGs, and JSON validation sweeps are available in [`evaluation_results/`](./evaluation_results/):
- [`EXPERIMENT_01_RESULTS.md`](./evaluation_results/EXPERIMENT_01_RESULTS.md)
- [`EXPERIMENT_03_RESULTS.md`](./evaluation_results/EXPERIMENT_03_RESULTS.md)
- [`EXPERIMENT_04_RESULTS.md`](./evaluation_results/EXPERIMENT_04_RESULTS.md)

---

## 🔍 Real vs. Mock Specialist Capabilities

To ensure scientific honesty and prevent fabricated metrics ("no fake science"), all backend services clearly delineate real ML capabilities from mock tools:

| Specialist Tool / Capability | Status | Execution Engine | Output Guarantee |
|---|---|---|---|
| **Change Detection (Bi-temporal)** | 🟢 **REAL ML** | Trained Siamese U-Net (`checkpoints/best_model.pt`) with tiled 256×256 sliding-window inference + 32px overlap smoothing. | Real binary change mask, percentage changed, pixel confusion stats. |
| **Classical Difference (Fallback)** | 🟢 **REAL ALG** | CPU perceptual luminance difference + adaptive thresholding. | Active if checkpoint is unconfigured. |
| **Geospatial Preprocessing** | 🟢 **REAL** | Pillow + Rasterio/PyProj GeoTIFF bounds, CRS, dimensions, and band normalization. | Real metadata extraction. |
| **VQA Adapter Pipeline** | 🟢 **REAL** | HuggingFace `SmolVLM-500M-Instruct` adapter with dynamic device fallback (CUDA/CPU) & model caching. | Real text synthesis when enabled (`VQA_MODE=real/auto`). |
| **Dataset Validators** | 🟢 **REAL** | LEVIR-CD directory layout & file alignment validator; BigEarthNet parquet schema validator. | Real split checks & patch count summaries. |
| **RS Captioning / Grounding** | 🟡 *Mock* | Structured mock specialist service (clearly marked `[MOCK]`). | Bounding boxes/coords return `null` when unverified. |
| **Optical / SAR Fusion** | 🟡 *Mock* | Structured mock specialist service. | Delineated mock summary. |

---

## 📂 Repository Structure

```
SatQuery-AI/
├── app/                              # Next.js 15 Frontend (App Router, Tailwind CSS, Dark Mode)
│   ├── analysis/                     # Analysis flows ([id] details, history, new submission)
│   ├── benchmark/                    # Benchmark metrics dashboard
│   ├── registry/                     # Specialist tool registry inspection page
│   ├── login/                        # Authentication guard & access
│   ├── analysis/                     # (see above) Analysis workflow pages
│   ├── page.tsx                      # Landing page & quick launch
│   └── layout.tsx                    # Root layout with theme provider & header
├── backend/                          # FastAPI Backend
│   ├── app/
│   │   ├── routers/                  # API endpoints (upload, analysis, datasets, tools, benchmark, health, files)
│   │   ├── services/                 # Core services (orchestrator, model_inference, datasets, vqa_service,
│   │   │                             #               preprocessing, firebase, metadata, models, trace,
│   │   │                             #               change_detection, model_manager)
│   │   ├── main.py                   # FastAPI app entry + CORS + error handlers
│   │   ├── config.py                 # Pydantic Settings (LEVIR_CD_DATASET_PATH, CHECKPOINT_DIR, etc.)
│   │   ├── models.py                 # SQLAlchemy SQLite models
│   │   └── schemas.py                # Pydantic validation schemas matching TypeScript contracts
│   ├── checkpoints/                  # Model weights (versioned in Git, ~1 MB each)
│   │   ├── best_model.pt             # Baseline best checkpoint (Epoch 48, F1=0.5429)
│   │   ├── last_model.pt             # Baseline Epoch 50 (resume-ready)
│   │   ├── baseline_epoch48_best_model.pt  # Explicit epoch-48 baseline copy
│   │   ├── training_log.json         # Baseline 50-epoch training curves
│   │   ├── experiment_01/            # Hybrid Loss (F1=0.6245) — best_model.pt / last_model.pt / log / config
│   │   ├── experiment_02/            # 5-epoch quick run (F1=0.4116) — best_model.pt / last_model.pt / log / config
│   │   ├── experiment_03/            # 🏆 Best — 60 epochs (F1=0.6435) — best_model.pt / last_model.pt / log / config
│   │   ├── experiment_04/            # 75 epochs (F1=0.6401) — best_model.pt / last_model.pt / log / config
│   │   ├── experiment_A_mini/        # Lightweight experiment — best_model.pt / last_model.pt
│   │   └── experiment_controlled/    # Controlled 51-epoch run (F1=0.6278) — best/last/log/config
│   ├── scripts/                      # Standalone CLI tools for training & evaluation
│   │   ├── train_change_detector.py  # Full training, resume training, smoke-test, eval-only CLI
│   │   ├── evaluate_full_test_and_val.py  # Full 128-test split evaluation & threshold sweeps
│   │   ├── visualize_change_predictions.py  # 6-panel qualitative visual evaluation generator
│   │   ├── _baseline_eval.py         # Baseline checkpoint full-split evaluation (standalone)
│   │   ├── _baseline_fullres.py      # Full-resolution 1024×1024 evaluation script
│   │   ├── _estimate_time.py         # Epoch time estimator for training planning
│   │   ├── _inspect_ckpt.py          # Checkpoint inspector (state dict, sizes, epoch metadata)
│   │   └── _train_expAmini.py        # Short A_mini experiment launch script
│   ├── evaluation_results/           # Per-experiment quantitative & qualitative evaluation artifacts
│   │   ├── baseline_run/             # Baseline test split metrics + per-sample PNG visuals
│   │   ├── experiment_03_eval/       # experiment_03 full test split eval + threshold sweep
│   │   ├── experiment_04_eval/       # experiment_04 full test split eval + threshold sweep
│   │   └── visuals/                  # Shared 6-panel prediction PNGs
│   ├── tests/                        # Automated unit & integration tests
│   ├── requirements.txt              # Backend dependencies
│   └── .env.example                  # Backend environment variable template
├── components/                       # React UI components (SatelliteViewer, ChangeStatsPanel, Trace UI, etc.)
├── evaluation_results/               # Top-level evaluation reports (root)
│   ├── EXPERIMENT_01_RESULTS.md      # Detailed experiment 01 logs & benchmark comparison
│   ├── EXPERIMENT_03_RESULTS.md      # Detailed experiment 03 logs & benchmark comparison
│   ├── EXPERIMENT_04_RESULTS.md      # Detailed experiment 04 logs & benchmark comparison
│   ├── test_full_results.json        # Full 128-sample test evaluation metrics
│   └── val_threshold_sweep.json      # Validation threshold sweep metrics [0.30 - 0.70]
├── lib/                              # API client (`liveApi.ts`, `mockApi.ts`) & TypeScript interfaces + config.ts
├── public/                           # Static demo assets & sample imagery
├── .env.example                      # Frontend environment variable template
└── README.md                         # Project documentation
```

---

## 👥 Team Setup & Quickstart Guide

### 1. Clone & Frontend Setup
```bash
# Clone the repository
git clone https://github.com/siddhtyagi18/SatQuery-AI.git
cd SatQuery-AI

# Configure frontend environment
cp .env.example .env.local

# Install dependencies and start development server
npm install
npm run dev
# Frontend is now running at: http://localhost:3000
```

### 2. Backend Setup
```bash
# In a new terminal, navigate to backend:
cd backend

# Create and activate Python virtual environment (Python 3.11+ recommended)
python -m venv .venv

# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
```

### 3. Local Dataset & Checkpoint Configuration

Edit `backend/.env` with your local paths:

```env
# Path to trained SiameseUNet checkpoint file (included in repository)
# Default: baseline checkpoint (Val F1=0.5429)
# For BEST PERFORMANCE, use experiment_03 (Val F1=0.6435):
# CHANGE_DETECTION_CHECKPOINT=./checkpoints/experiment_03/best_model.pt
# Or experiment_04 (Val F1=0.6401):
# CHANGE_DETECTION_CHECKPOINT=./checkpoints/experiment_04/best_model.pt
CHANGE_DETECTION_CHECKPOINT=./checkpoints/best_model.pt

# Optional: Path to local LEVIR-CD dataset (if validating or running training/eval scripts)
# Do NOT commit your local dataset path into version control
LEVIR_CD_DATASET_PATH=/path/to/LEVIR-CD

# Storage & VQA configuration
STORAGE_BACKEND=local
VQA_MODE=auto   # "mock" (fast), "real" (SmolVLM 500M), "auto" (hybrid)
```

### 4. Run Backend Server
```bash
uvicorn app.main:app --reload --port 8000
# Backend API & Interactive Docs: http://localhost:8000/docs
```

### 5. Run Automated Test Suite
```bash
cd backend
pytest tests/ -v
# All 88 tests execute locally in under 15 seconds
```

---

## 🏋️ Training & Evaluation CLI Commands

### Run Full Test Split Evaluation
```bash
cd backend
python scripts/evaluate_full_test_and_val.py \
    --checkpoint ./checkpoints/best_model.pt \
    --data-root /path/to/LEVIR-CD \
    --threshold 0.70
```

### Generate 6-Panel Prediction Visualizations
```bash
cd backend
python scripts/visualize_change_predictions.py \
    --checkpoint ./checkpoints/best_model.pt \
    --data-root /path/to/LEVIR-CD \
    --num-samples 10 \
    --threshold 0.70
```

### Continue / Resume Training
```bash
cd backend
python scripts/train_change_detector.py \
    --data-root /path/to/LEVIR-CD \
    --resume ./checkpoints/last_model.pt \
    --epochs 100 \
    --batch-size 4
```

---

## 💾 Model Weights & Git Storage Strategy

- **Current Checkpoints**: The Siamese U-Net weights (`best_model.pt` and `last_model.pt`) are **~1.48 MB** each. Because they are well below GitHub's 50 MB / 100 MB limits, they are versioned directly in Git under `backend/checkpoints/` for zero-friction team onboarding.
- **Large Transformer Models**: If larger foundational models or Vision-Language Transformers (>50 MB) are added in future iterations, they should be stored via **Git LFS** (`git lfs track "*.pt"`), **GitHub Releases**, or a shared cloud storage bucket (e.g. Google Cloud Storage / Hugging Face Model Hub).
- **Datasets**: The LEVIR-CD dataset (~5-10 GB) and BigEarthNet parquet files are strictly excluded from git via `.gitignore`. Each teammate configures their local dataset path via `LEVIR_CD_DATASET_PATH` in `.env`.

---

*SatQuery-AI — Smart India Hackathon (SIH) Project*
