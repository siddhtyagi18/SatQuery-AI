# SatQuery-AI 🌍🛰️

An AI-powered satellite imagery query and multi-temporal analysis platform for the Smart India Hackathon (SIH).

---

## 🚀 Project Status & Real ML Milestones

SatQuery-AI features an end-to-end pipeline spanning geospatial pre-processing, deterministic task routing, deep learning change detection, multi-modal VQA adapters, and interactive Next.js visualization.

### Real Training Experiments (LEVIR-CD Dataset)

We have completed two full real training experiments on the LEVIR-CD building change detection benchmark:

| Metric / Parameter | Experiment 00 (Baseline) | Experiment 01 (Hybrid Imbalance Loss) |
|---|---|---|
| **Architecture** | Siamese U-Net (6-channel bitemporal) | Siamese U-Net (6-channel bitemporal) |
| **Parameters** | 490,561 (~1.48 MB) | 490,561 (~1.48 MB) |
| **Epochs** | 50 (Best Checkpoint: Epoch 48) | 50 (Best Validation Checkpoint: Epoch 48/50) |
| **Loss Formulation** | BCE + Dice Loss | Hybrid Weighted BCE + Soft Dice + Boundary Loss |
| **Best Validation F1** | 0.5429 | **0.6245** |
| **Best Validation IoU** | 0.4875 | **0.4638** |

### Official Test Evaluation Benchmark (Experiment 01)

Evaluated across the full 128-sample LEVIR-CD test split using the validation-selected optimal threshold **0.70**:

- **Test Micro IoU (Jaccard Index)**: **`58.06%`** (`0.5806`)
- **Test Micro F1 / Dice Score**: **`73.47%`** (`0.7347`)
- **Test Precision**: **`73.62%`** (`0.7362`)
- **Test Recall**: **`73.32%`** (`0.7332`)
- **Test Pixel Accuracy**: **`97.34%`** (`0.9734`)

Full evaluation logs and per-sample benchmark tables are available in [`evaluation_results/`](./evaluation_results/).

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
│   ├── layout.tsx                    # Root layout with theme provider & header
│   └── page.tsx                      # Landing page & quick launch
├── backend/                          # FastAPI Backend
│   ├── app/
│   │   ├── routers/                  # API endpoints (upload, analysis, datasets, tools, benchmark, health)
│   │   ├── services/                 # Core services (orchestrator, model_inference, datasets, vqa_service,
│   │   │                             #               preprocessing, firebase, metadata, models, trace)
│   │   ├── main.py                   # FastAPI app entry + CORS + error handlers
│   │   ├── config.py                 # Pydantic Settings (LEVIR_CD_DATASET_PATH, CHECKPOINT_DIR, etc.)
│   │   ├── models.py                 # SQLAlchemy SQLite models
│   │   └── schemas.py                # Pydantic validation schemas matching TypeScript contracts
│   ├── checkpoints/                  # Model weights (1.48 MB each, versioned in Git)
│   │   ├── best_model.pt             # Best checkpoint (Epoch 48 weights)
│   │   ├── last_model.pt             # Epoch 50 checkpoint (ready for resume training)
│   │   ├── training_log.json         # 50-epoch loss and evaluation curves
│   │   └── experiment_01/            # Experiment 01 directory copy for automated tooling
│   ├── scripts/                      # Standalone CLI tools for training & evaluation
│   │   ├── train_change_detector.py  # Full training, resume training, and smoke-test CLI
│   │   ├── evaluate_full_test_and_val.py # Full 128-test split evaluation & threshold sweeps
│   │   └── visualize_change_predictions.py # 6-panel qualitative visual evaluation generator
│   ├── tests/                        # 88 automated unit & integration tests
│   ├── requirements.txt              # Backend dependencies
│   └── .env.example                  # Backend environment variable template
├── components/                       # React UI components (SatelliteViewer, ChangeStatsPanel, Trace UI, etc.)
├── evaluation_results/               # Quantitative benchmark metrics & JSON validation sweeps
│   ├── EXPERIMENT_01_RESULTS.md      # Detailed experiment logs & benchmark comparison
│   ├── test_full_results.json        # Full 128-sample test evaluation metrics
│   └── val_threshold_sweep.json      # Validation threshold sweep metrics [0.30 - 0.70]
├── lib/                              # API client (`liveApi.ts`, `mockApi.ts`) & TypeScript interfaces
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
CHANGE_DETECTION_CHECKPOINT=./checkpoints/best_model.pt

# Optional: Path to local LEVIR-CD dataset (if validating or running training/eval scripts)
# Do NOT commit your local dataset path into version control
LEVIR_CD_DATASET_PATH=/path/to/LEVIR-CD

# Storage & VQA configuration
STORAGE_BACKEND=local
VQA_MODE=auto
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
