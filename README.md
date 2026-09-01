# SatQuery-AI 🌍🛰️

An AI-powered satellite imagery query and analysis platform (SIH Project).

## Status: Phase 3 In Progress (Trained Model Checkpoint Ready) 🚀

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1** | ✅ Completed | Full-stack foundation with Next.js frontend + FastAPI backend, mock specialist services, SQLite persistence, test suite, execution trace pipeline, 7-tool registry |
| **Phase 2** | ✅ Completed | Real VQA service via Model Adapter architecture, Firebase Firestore + Storage integration, Rasterio/PyProj/Pillow geospatial preprocessing, Model Manager caching |
| **Phase 3** | 🔄 In Progress | Real Change Detection Siamese U-Net model (**50 epochs trained on LEVIR-CD, best checkpoint at Epoch 48 with IoU 0.4875**), tiled sliding-window inference dispatcher, dataset validators, resume training pipeline, BigEarthNet metadata schema analysis |

---

## Project Structure

```
SatQuery-AI/
├── app/                          # Next.js Frontend (App Router)
│   ├── analysis/                 # Analysis flows (new, history, detail)
│   ├── benchmark/                # Benchmark metrics page
│   ├── registry/                 # Tool registry page
│   ├── layout.tsx                # Root layout
│   └── page.tsx                  # Landing page
├── backend/                      # FastAPI Backend
│   ├── app/
│   │   ├── routers/              # API endpoints (upload, analysis, datasets, tools, benchmark, health)
│   │   ├── services/             # Core services (orchestrator, model_inference, datasets, vqa_service,
│   │   │                         #               preprocessing, firebase, metadata, models, trace)
│   │   ├── main.py               # FastAPI app entry
│   │   ├── config.py             # Configurable settings via .env (no hardcoded paths)
│   │   ├── models.py             # SQLAlchemy models + DB schema
│   │   └── schemas.py            # Pydantic schemas matching frontend contracts
│   ├── checkpoints/              # Trained PyTorch weights (~1.48 MB each)
│   │   ├── best_model.pt         # Best checkpoint (Epoch 48, IoU: 0.4875)
│   │   ├── last_model.pt         # Epoch 50 checkpoint (ready for resume training)
│   │   └── training_log.json     # Full 50-epoch training history log
│   ├── scripts/
│   │   └── train_change_detector.py # Training, resume training & evaluation script
│   ├── tests/                    # 88 unit & integration tests
│   ├── requirements.txt          # Python dependencies
│   └── README.md                 # Detailed backend docs
├── components/                   # React UI components (SatelliteViewer, ChangeStatsPanel, Trace UI, etc.)
├── lib/                          # API client + TypeScript contracts
├── public/                       # Static demo imagery
├── .env.example                  # Frontend environment configuration template
└── README.md                     # Root project documentation
```

---

## Model & Checkpoints Summary

- **Architecture**: Siamese U-Net with 6-channel input `[Image_A, Image_B]`, 3-level skip connections, ~490K parameters (~1.48 MB).
- **Dataset**: Trained on LEVIR-CD bi-temporal building change detection dataset.
- **Checkpoints**:
  - `backend/checkpoints/best_model.pt` — Best model checkpoint (Epoch 48: IoU `0.4875`, F1 `0.5429`, Precision `0.9691`, Accuracy `97.63%`).
  - `backend/checkpoints/last_model.pt` — Epoch 50 checkpoint with full model, optimizer, and scheduler states for seamless resume training.
  - `backend/checkpoints/training_log.json` — 50-epoch loss and evaluation history.
- **Inference**: High-resolution 1024×1024 images are processed via 256×256 tiled sliding windows with 32px overlap averaging.

---

## Team Setup & Quick Start

### 1. Frontend Setup (Next.js)
```bash
# In repository root:
cp .env.example .env.local
npm install
npm run dev
# → http://localhost:3000
```

### 2. Backend Setup (FastAPI)
```bash
cd backend
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env

# Run FastAPI dev server:
uvicorn app.main:app --reload --port 8000
# → http://localhost:8000/docs (Interactive Swagger UI)
```

### 3. Change Detection Model Configuration (.env)
To enable the trained change detection model for inference, configure in `backend/.env`:
```env
CHANGE_DETECTION_CHECKPOINT=./checkpoints/best_model.pt
LEVIR_CD_DATASET_PATH=/path/to/LEVIR-CD
```

### 4. Resuming Training
To continue training beyond Epoch 50:
```bash
cd backend
python scripts/train_change_detector.py \
    --data-root "/path/to/LEVIR-CD" \
    --resume ./checkpoints/last_model.pt \
    --epochs 100 \
    --batch-size 4
```

### 5. Running Tests
```bash
cd backend
pytest tests/ -v
```
All 88 unit & integration tests execute locally in under 10 seconds.

---

## API Quick Reference

| Endpoint | Method | Phase | Description |
|----------|--------|-------|-------------|
| `/health` | GET | 1 | Health check + version |
| `/api/upload` | POST | 1+2 | Upload satellite image (PNG/JPG/GeoTIFF) |
| `/api/analysis` | POST | 1+2 | Submit analysis request |
| `/api/analysis` | GET | 1+2 | List analysis history (paginated, filtered) |
| `/api/analysis/{id}` | GET | 1+2 | Get full analysis result |
| `/api/analysis/{id}/trace` | GET | 1+2 | Get execution trace only |
| `/api/analysis/{id}` | DELETE | 1+2 | Delete analysis record |
| `/api/datasets/status` | GET | 3 | Status of configured datasets & checkpoints |
| `/api/datasets/levir-cd/validate` | GET | 3 | Validate LEVIR-CD directory splits & file alignment |
| `/api/datasets/bigearthnet/summary` | GET | 3 | Schema & statistics summary for BigEarthNet annotations |
| `/api/tools` | GET | 1+2 | List specialist tools + status |
| `/api/benchmark` | GET | 1+2 | 12 benchmark metric rows |
| `/api/files/{id}` | GET | 1+2 | Serve uploaded file preview |

---

## Design Principles

- **No Fake Science** — All mock outputs are explicitly labelled; confidence/coords/masks return `null` when unreliable.
- **Frontend Contract as Source of Truth** — Pydantic schemas mirror `lib/types/analysis.ts` exactly.
- **Deterministic Routing & Full Tracing** — Per-request multi-step execution traces for auditable workflows.
- **Lightweight Checkpoints** — Trained model weights (~1.48 MB) fit cleanly into git for immediate team collaboration without large LFS overhead.

---

*SatQuery-AI — SIH Project*
