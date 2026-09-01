"""
tests/test_datasets_api.py
--------------------------
Tests for dataset API endpoints:
- GET /api/datasets/status
- GET /api/datasets/levir-cd/validate
- GET /api/datasets/bigearthnet/summary
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))


def test_datasets_status_endpoint(client: TestClient):
    """
    GET /api/datasets/status should return configuration and existence status
    for datasets and checkpoints without crashing.
    """
    resp = client.get("/api/datasets/status")
    assert resp.status_code == 200
    data = resp.json()

    assert "inference_mode" in data
    assert data["inference_mode"] in ("model_checkpoint", "cpu_classical")
    assert "datasets" in data
    assert "levir_cd" in data["datasets"]
    assert "bigearthnet_txt" in data["datasets"]
    assert "bigearthnet_metadata" in data["datasets"]
    assert "checkpoint" in data


def test_levir_validate_endpoint_unconfigured(client: TestClient, monkeypatch):
    """
    When LEVIR_CD_ROOT is None or empty, the endpoint should return 200
    with valid=False and a helpful error message.
    """
    from app.routers import datasets

    mock_settings = MagicMock()
    mock_settings.LEVIR_CD_ROOT = None
    mock_settings.LEVIR_CD_DATASET_PATH = None
    monkeypatch.setattr(datasets, "settings", mock_settings)

    resp = client.get("/api/datasets/levir-cd/validate")
    assert resp.status_code == 200
    data = resp.json()

    assert data["valid"] is False
    assert "error" in data
    assert "LEVIR_CD_ROOT" in data["error"]


def test_levir_validate_endpoint_valid_fixture(client: TestClient, tmp_path, monkeypatch):
    """
    When LEVIR_CD_ROOT points to a valid LEVIR-CD directory fixture,
    the endpoint should return valid=True with full split statistics.
    """
    from app.routers import datasets
    from PIL import Image
    import numpy as np

    # Build tiny fixture
    root = tmp_path / "LEVIR_TEST"
    for split in ("train", "val", "test"):
        for sub in ("A", "B", "label"):
            d = root / split / sub
            d.mkdir(parents=True)
            if sub in ("A", "B"):
                Image.new("RGB", (16, 16), color=(50, 100, 150)).save(d / "img_1.png")
            else:
                arr = np.zeros((16, 16), dtype=np.uint8)
                arr[4:8, 4:8] = 255
                Image.fromarray(arr, mode="L").save(d / "img_1.png")

    mock_settings = MagicMock()
    mock_settings.LEVIR_CD_ROOT = str(root)
    monkeypatch.setattr(datasets, "settings", mock_settings)

    resp = client.get("/api/datasets/levir-cd/validate")
    assert resp.status_code == 200
    data = resp.json()

    assert data["valid"] is True
    assert data["total_triplets"] == 3
    assert "splits" in data
    assert "train" in data["splits"]
    assert data["splits"]["train"]["matched_triplets"] == 1


def test_bigearthnet_summary_endpoint_unconfigured(client: TestClient, monkeypatch):
    """
    When BigEarthNet parquet paths are not set, the endpoint should return 200
    with available=False and helpful descriptions.
    """
    from app.routers import datasets

    mock_settings = MagicMock()
    mock_settings.BIGEARTHNET_TXT_PARQUET = None
    mock_settings.BIGEARTHNET_METADATA_PARQUET = None
    monkeypatch.setattr(datasets, "settings", mock_settings)

    resp = client.get("/api/datasets/bigearthnet/summary")
    assert resp.status_code == 200
    data = resp.json()

    assert "note" in data
    assert "txt_annotation" in data
    assert data["txt_annotation"]["available"] is False
    assert "metadata" in data
    assert data["metadata"]["available"] is False


def test_root_endpoint_includes_dataset_info(client: TestClient):
    """
    GET / should include dataset routes and phase 3 status information.
    """
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()

    assert data.get("phase") == 3
    assert "change_detection_inference_mode" in data
    assert "dataset_routes" in data
    assert "/api/datasets/levir-cd/validate" in data["dataset_routes"]
