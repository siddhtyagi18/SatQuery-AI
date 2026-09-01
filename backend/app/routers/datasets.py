"""
backend/app/routers/datasets.py
================================
Dataset validation and inspection API routes.

Routes
------
GET /api/datasets/levir-cd/validate
    Validates the LEVIR-CD dataset at the configured LEVIR_CD_ROOT path.
    Returns a structured validation report without loading any images into RAM.
    If LEVIR_CD_ROOT is not configured, returns a clear error response.

GET /api/datasets/bigearthnet/summary
    Reads the schema and a small sample of rows from the BigEarthNet parquet files.
    Uses pyarrow and reads only metadata — does NOT load all 9.5M rows into RAM.
    Clarifies that these are annotation files, NOT the actual Sentinel image data.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter

from ..config import get_settings
from ..logging_setup import logger
from ..services.datasets.levir_cd import validate_levir_cd
from ..services.datasets.bigearthnet_txt import get_bigearthnet_summary
from ..services.model_inference import get_inference_mode

router = APIRouter(prefix="/api/datasets", tags=["datasets"])
settings = get_settings()


@router.get("/levir-cd/validate", response_model=Dict[str, Any])
def validate_levir_cd_endpoint() -> Dict[str, Any]:
    """
    Validate the LEVIR-CD dataset at the configured LEVIR_CD_ROOT.

    Returns a structured validation report including:
    - Per-split image counts (A/B/label)
    - Matched triplet counts
    - Sample image format/dimension checks
    - Corrupt or mismatched file reports

    Does NOT load entire dataset into RAM. Samples at most 3 images per split.

    If LEVIR_CD_ROOT is not configured, returns a clear error response.
    """
    # Support both LEVIR_CD_DATASET_PATH and LEVIR_CD_ROOT
    levir_root_cfg = None
    for attr in ("LEVIR_CD_DATASET_PATH", "LEVIR_CD_ROOT"):
        val = getattr(settings, attr, None)
        if isinstance(val, (str, Path)) and str(val).strip():
            levir_root_cfg = str(val).strip()
            break

    if not levir_root_cfg:
        logger.info("[datasets] /levir-cd/validate: LEVIR_CD_DATASET_PATH / LEVIR_CD_ROOT not configured")
        return {
            "valid": False,
            "error": (
                "LEVIR_CD_DATASET_PATH (or LEVIR_CD_ROOT) environment variable is not set. "
                "Add it to your .env file pointing to the extracted LEVIR-CD folder. "
                "Expected structure: <root>/{train,val,test}/{A,B,label}/"
            ),
            "root": None,
            "splits": {},
            "total_triplets": 0,
            "global_errors": ["LEVIR_CD_DATASET_PATH not configured"],
        }

    levir_root = Path(levir_root_cfg)
    logger.info("[datasets] Validating LEVIR-CD at: %s", levir_root)

    try:
        result = validate_levir_cd(levir_root)
        report = result.to_dict()
        report["inference_mode"] = get_inference_mode()
        return report
    except Exception as e:
        logger.exception("[datasets] LEVIR-CD validation failed unexpectedly: %s", e)
        return {
            "valid": False,
            "error": f"Validation raised an unexpected error: {type(e).__name__}: {e}",
            "root": str(levir_root),
            "splits": {},
            "total_triplets": 0,
            "global_errors": [f"{type(e).__name__}: {e}"],
        }


@router.get("/bigearthnet/summary", response_model=Dict[str, Any])
def bigearthnet_summary_endpoint() -> Dict[str, Any]:
    """
    Summarise BigEarthNet annotation and metadata parquet files.

    IMPORTANT: These are annotation/metadata files, NOT image data.
    The actual Sentinel-1/Sentinel-2 image patches are a separate ~65GB download
    and are not required for this summary.

    Returns:
    - File availability and schema for BigEarthNet.txt.parquet
      (9.5M rows of VQA-style question-answer annotations)
    - File availability and schema for metadata.parquet
      (480K rows of patch-level metadata including labels, split, country, etc.)
    - Top distributions for key columns (split, type, category, country, season)

    Reads only schema + 3 sample rows per file — does NOT load all rows into RAM.
    """
    txt_cfg = settings.BIGEARTHNET_TXT_PARQUET
    meta_cfg = settings.BIGEARTHNET_METADATA_PARQUET

    txt_path = Path(txt_cfg) if txt_cfg else None
    meta_path = Path(meta_cfg) if meta_cfg else None

    logger.info(
        "[datasets] BigEarthNet summary: txt=%s meta=%s",
        txt_path, meta_path,
    )

    try:
        summary = get_bigearthnet_summary(txt_path, meta_path)
        return summary.to_dict()
    except Exception as e:
        logger.exception("[datasets] BigEarthNet summary failed: %s", e)
        return {
            "note": "Error computing BigEarthNet summary.",
            "pyarrow_available": False,
            "error": f"{type(e).__name__}: {e}",
            "txt_annotation": {"path": str(txt_path), "available": False, "error": str(e)},
            "metadata": {"path": str(meta_path), "available": False, "error": str(e)},
        }


@router.get("/status", response_model=Dict[str, Any])
def datasets_status_endpoint() -> Dict[str, Any]:
    """
    Quick status check for all configured datasets and the inference mode.

    Returns configuration status without actually reading any dataset files.
    """
    levir_root_cfg = None
    for attr in ("LEVIR_CD_DATASET_PATH", "LEVIR_CD_ROOT"):
        val = getattr(settings, attr, None)
        if isinstance(val, (str, Path)) and str(val).strip():
            levir_root_cfg = str(val).strip()
            break
    txt_cfg = settings.BIGEARTHNET_TXT_PARQUET
    meta_cfg = settings.BIGEARTHNET_METADATA_PARQUET
    ckpt_cfg = settings.CHANGE_DETECTION_CHECKPOINT
    ckpt_dir_cfg = settings.CHECKPOINT_DIR

    levir_exists = Path(levir_root_cfg).exists() if levir_root_cfg else False
    txt_exists = Path(txt_cfg).exists() if txt_cfg else False
    meta_exists = Path(meta_cfg).exists() if meta_cfg else False
    ckpt_exists = Path(ckpt_cfg).exists() if ckpt_cfg else False

    return {
        "inference_mode": get_inference_mode(),
        "datasets": {
            "levir_cd": {
                "configured": bool(levir_root_cfg),
                "path": levir_root_cfg,
                "exists": levir_exists,
            },
            "bigearthnet_txt": {
                "configured": bool(txt_cfg),
                "path": txt_cfg,
                "exists": txt_exists,
            },
            "bigearthnet_metadata": {
                "configured": bool(meta_cfg),
                "path": meta_cfg,
                "exists": meta_exists,
            },
        },
        "checkpoint": {
            "configured": bool(ckpt_cfg),
            "path": ckpt_cfg,
            "exists": ckpt_exists,
            "checkpoint_dir": ckpt_dir_cfg,
        },
    }
