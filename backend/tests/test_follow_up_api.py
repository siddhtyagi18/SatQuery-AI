"""
Integration Test for Follow-up API Router
=========================================

Tests POST /api/analysis/{analysis_id}/follow-up via FastAPI TestClient.
Verifies HTTP 200, zero model re-run, correct metric grounding, and HTTP 404/400 validation.
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.database import get_db
from backend.app.models import Analysis

client = TestClient(app)


def test_follow_up_api_flow():
    # 1. Create a test analysis row directly in DB session
    from backend.app.database import SessionLocal
    db = SessionLocal()
    try:
        a = Analysis(
            id="test-api-follow-up-01",
            mode="bi_temporal",
            query="What changes occurred between T1 and T2?",
            status="completed",
            confidence=None,
        )
        a.answer_text = (
            "Detected Changed Area: `3.14%` (Severity: **low**)\n"
            "Threshold: `0.70`\n"
            "Confidence: Not calibrated"
        )
        a.detected_tasks = ["change_detection"]
        a.change_map = {"overlayUrl": "/demo/mask.png"}
        db.add(a)
        db.commit()

        # 2. Test Area Query via API
        res = client.post(
            "/api/analysis/test-api-follow-up-01/follow-up",
            json={"query": "How much area changed?", "language": "en"}
        )
        assert res.status_code == 200
        data = res.json()
        assert data["rerunPerformed"] is False
        assert "3.14%" in data["answer"]
        assert data["referencedMetrics"]["changed_area"] == "3.14%"

        # 3. Test Spatial Query via API
        res_spatial = client.post(
            "/api/analysis/test-api-follow-up-01/follow-up",
            json={"query": "Where did this happen?"}
        )
        assert res_spatial.status_code == 200
        data_spatial = res_spatial.json()
        assert data_spatial["spatialAction"]["action"] == "highlight"

        # 4. Test Hindi Query via API
        res_hi = client.post(
            "/api/analysis/test-api-follow-up-01/follow-up",
            json={"query": "कितना क्षेत्र बदला?", "language": "hi"}
        )
        assert res_hi.status_code == 200
        assert "3.14%" in res_hi.json()["answer"]

        # 5. Test 404 for non-existent analysis
        res_404 = client.post(
            "/api/analysis/non-existent-id-999/follow-up",
            json={"query": "Any changes?"}
        )
        assert res_404.status_code == 404

        # 6. Test 400 for empty query
        res_400 = client.post(
            "/api/analysis/test-api-follow-up-01/follow-up",
            json={"query": "   "}
        )
        assert res_400.status_code == 400

    finally:
        # Cleanup
        to_del = db.query(Analysis).filter(Analysis.id == "test-api-follow-up-01").first()
        if to_del:
            db.delete(to_del)
            db.commit()
        db.close()
