# SatQuery-AI 🌍🛰️

An AI-powered satellite imagery query and analysis platform (SIH Project).

## Status: Phase 2 Completed ✅

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1** | ✅ Completed | Full-stack foundation with Next.js frontend + FastAPI backend, mock specialist services, SQLite persistence, 18 test suite, execution trace pipeline, 7-tool registry (VQA, Captioning, Grounding, Change Detection, Change-VQA, Optical-SAR, Spatial Analyzer) |
| **Phase 2** | ✅ Completed | Real VQA service via Model Adapter architecture, Firebase Firestore + Storage integration, Rasterio/PyProj/Pillow geospatial preprocessing, Model Manager caching, clean Firebase service/repository layer, per-request detailed execution traces (validation → metadata → tool selection → preprocessing → inference → result validation) |
| Phase 3 | ⏳ Planned | Real Grounding + Change Detection + Optical-SAR specialists, async SSE streaming, auth + rate limiting, benchmark evaluation on public RS datasets |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│  Frontend (Next.js 15 App Router)               │
│  • SatelliteViewer · AnalysisModeSelector · Trace UI   │
│  • History · Benchmark · Tool Registry         │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP / JSON (analysis.ts contracts)
                       ▼
┌─────────────────────────────────────────────────────┐
│  FastAPI Backend                                │
│  • /api/upload  /api/analysis  /api/tools      │
│  • /api/benchmark  /health  /api/files        │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  Orchestrator Layer                              │
│  • Task Classifier · Tool Registry                │
│  • Execution Trace (6 persisted steps)             │
└──────────────────────┬──────────────────────────────┘
          ┌────────────┴────────────┐
          ▼                         ▼
┌─────────────────────┐   ┌─────────────────────────┐
│  Real VQA Service   │   │  Mock Specialists      │
│  • VQAService       │   │  • RS_CAPTION         │
│  • VQAModelAdapter  │   │  • RS_GROUNDING       │
│  • Model Manager    │   │  • CHANGE_DETECTOR    │
│  • Preprocessing    │   │  • CHANGE_VQA         │
│  • Result Valid.    │   │  • OPTICAL_SAR        │
│                     │   │  • SPATIAL_ANALYZER  │
└─────────────────────┘   └─────────────────────────┘
          │                         │
          └────────────┬────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  Persistence Layer (Phase 2)                    │
│  • Firebase Firestore — analysis records, traces │
│  • Firebase Storage — uploaded images                │
│  • Environment-based credentials (no hardcodes) │
└─────────────────────────────────────────────────────┘
```

---

## Project Structure

```
SatQuery-AI/
├── app/                          # Next.js Frontend
│   ├── analysis/                 # Analysis flows (new, history, detail)
│   ├── benchmark/                # Benchmark metrics page
│   ├── registry/                 # Tool registry page
│   ├── layout.tsx             # Root layout
│   └── page.tsx               # Landing page
├── backend/                      # FastAPI Backend
│   ├── app/
│   │   ├── routers/          # API endpoints (upload, analysis, tools, benchmark, files, health
│   │   ├── services/       # Core services (orchestrator, vqa_service, vqa_adapter, model_manager,
│   │   │                   #              preprocessing, firebase, metadata, task_classifier,
│   │   │                   #              tool_registry, trace, result_validation, mock_specialists)
│   │   ├── main.py         # FastAPI app entry
│   │   ├── config.py       # Settings via .env
│   │   ├── models.py       # SQLAlchemy tables (Phase 1) + Firebase abstractions (Phase 2)
│   │   └── schemas.py      # Pydantic types matching frontend contract
│   ├── tests/                # 18+ tests including test_phase2.py
│   ├── requirements.txt   # Python 3.13 compatible
│   └── README.md           # Detailed backend docs
├── components/                 # React UI components
├── lib/                        # API client + types
├── public/                     # Static assets + demo images
├── AGENTS.md                  # Workspace agent rules
├── package.json
└── next.config.ts
```

---

## Phase 1 Deliverables (Completed) ✅

- ✅ **Next.js Frontend** — Satellite imagery viewer, analysis mode selector (single/bi-temporal/optical-SAR), execution trace animation, history browser, benchmark dashboard, tool registry page
- ✅ **FastAPI Backend** — Upload validation, 7-tool specialist registry, orchestrator with deterministic task routing, SQLite + SQLAlchemy persistence
- ✅ **7 Mock Specialist Tools** — VQA, Captioning, Grounding, Change Detection, Change-VQA, Optical-SAR Analyzer, Spatial Analyzer (all explicitly marked `[MOCK]`)
- ✅ **8-Step Execution Trace** — Query Received → Validation → Classification → Tool Selection → Parameters → Processing → Aggregation → Completion
- ✅ **18 Test Cases** — Coverage for upload validation, 3 analysis modes, trace, history, delete, benchmark, health, tools
- ✅ **Frontend Contract as Source of Truth** — Pydantic schemas mirror `lib/types/analysis.ts` exactly; `API_MODE` toggle in `lib/config.ts`

## Phase 2 Deliverables (Completed) ✅

- ✅ **Real VQA Pipeline** — Single-image VQA queries routed to real AI service via Model Adapter pattern
- ✅ **VQA Service Layer** — `VQAService` → `VQAModelAdapter` → Model (clean separation, swappable)
- ✅ **Model Manager** — Model caching system (load once, reuse across requests, no per-request overhead)
- ✅ **Geospatial Preprocessing** — Rasterio + PyProj + Pillow for remote-sensing image preprocessing (CRS, bands, GSD, metadata extraction)
- ✅ **Firebase Integration** — Firestore for analysis records (status, traces, results); Firebase Storage for binary uploads
- ✅ **Clean Firebase Layer** — Service/repository abstraction, credentials via environment variables only (no hardcoding)
- ✅ **6-Step Detailed Execution Traces** — validation → metadata extraction → tool selection → preprocessing → inference → result validation
- ✅ **Strict Non-Fabrication** — Confidence scores, coordinates, masks return `null` when unreliable; no fake values
- ✅ **Python 3.13 Windows Compatibility** — CPU-runnable architecture maintained
- ✅ **Phase 2 Test Suite** — `tests/test_phase2.py validating new VQA + Firebase flows
- 🔒 **Remaining Mocks** — Captioning, Grounding, Change Detection, Change-VQA, Optical-SAR, Spatial Analyzer remain mocked (per Phase 2 scope)

---

## Quick Start

### Prerequisites
- **Node.js 18+** (for frontend)
- **Python 3.11+** (3.13 recommended on Windows)
- **Firebase project** (Phase 2 — optional; backend falls back to SQLite for Phase 1 mode)

### Frontend (Next.js)
```bash
npm install
npm run dev
# → http://localhost:3000
```

Toggle `API_MODE` in `lib/config.ts`:
```ts
export const API_MODE: 'mock' | 'live' = 'live';  // use FastAPI backend
```

### Backend (FastAPI) — see [backend/README.md](backend/README.md) for full docs
```bash
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # Windows PowerShell
pip install -r requirements.txt
copy .env.example .env            # fill Firebase creds (Phase 2) or skip for Phase 1 SQLite
uvicorn app.main:app --reload --port 8000
# → http://localhost:8000/docs  (Swagger UI)
```

### Tests
```bash
cd backend
pytest -v
```

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
| `/api/tools` | GET | 1+2 | List 7 specialist tools + status |
| `/api/benchmark` | GET | 1+2 | 12 benchmark metric rows |
| `/api/files/{id}` | GET | 1+2 | Serve uploaded file preview |

---

## Design Principles

- **No Fake Science** — All mock outputs are explicitly labelled; confidence/coords/masks return `null` when unreliable.
- **Frontend Contract is King** — Pydantic schemas mirror `analysis.ts` exactly.
- **Model Adapter Pattern** — Swappable model backends; clean `VQAService → VQAModelAdapter → Model chain.
- **Trace-First Design** — Every pipeline step is persisted for audit + streaming animation.
- **Clean Firebase Abstraction** — Firebase access behind service/repository layer.
- **CPU-First** — System runs on CPU; no GPU required.
- **Python 3.13 Windows Compatible** — All dependencies verified on Windows.

---

*SatQuery-AI — SIH Project • Phase 2 Complete
