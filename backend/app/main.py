from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .database import Base, engine
from .logging_setup import logger
from .routers import upload, analysis, tools, benchmark, health, files

settings = get_settings()

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "SatQuery-AI Phase 2 Backend (FastAPI). "
        "REAL: VQA (SmolVLM-500M via HuggingFace transformers, CPU-runnable), "
        "Change Detection (CPU-only classical pixel-difference, Pillow+NumPy). "
        "MOCK: Captioning, Grounding, Change VQA, Optical-SAR Analysis. "
        "Firebase (Firestore + Cloud Storage) persistence is optional via env vars."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled error on {request.method} {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {exc}"},
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    logger.warning(f"ValueError on {request.method} {request.url.path}: {exc}")
    return JSONResponse(status_code=400, content={"detail": str(exc)})


app.include_router(upload.router)
app.include_router(analysis.router)
app.include_router(tools.router)
app.include_router(benchmark.router)
app.include_router(health.router)
app.include_router(files.router)

# Serve generated change-mask PNGs at /api/results/<filename>
_results_dir = Path(__file__).resolve().parent.parent / "data" / "results"
_results_dir.mkdir(parents=True, exist_ok=True)
app.mount("/api/results", StaticFiles(directory=str(_results_dir)), name="results")


@app.get("/", tags=["root"])
def root():
    from .services.vqa_service import get_vqa_service
    from .services.firebase import is_firebase_enabled
    vqa_svc = get_vqa_service()
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "phase": 2,
        "docs": "/docs",
        "health": "/health",
        "vqa_mode": settings.VQA_MODE,
        "vqa_model_id": settings.VQA_MODEL_ID,
        "vqa_device": settings.VQA_DEVICE,
        "vqa_real_enabled_single_image": vqa_svc.should_use_real_vqa("single_image", tasks=["vqa"]),
        "firebase_enabled": is_firebase_enabled(),
        "real_tools": ["rs_vqa", "change_detector"],
        "mock_tools": ["rs_caption", "rs_grounding", "change_vqa", "optical_sar_analyzer", "spatial_analyzer"],
    }
