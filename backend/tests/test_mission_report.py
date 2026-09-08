"""
backend/tests/test_mission_report.py
------------------------------------
Unit & regression tests for AI Mission Report and Evidence-backed PDF Generation.

Covers prompt-specified test cases A through O:
A. report generation from completed analysis
B. missing optional metadata
C. no ROI case
D. ROI present case
E. no VQA case
F. VQA present case
G. confidence remains None
H. no fabricated physical area
I. report contains threshold 0.70
J. report contains existing checkpoint
K. report does not trigger Change Detection
L. report does not trigger VQA
M. PDF generation succeeds
N. malformed/missing analysis handling
O. existing API responses remain compatible
"""
from __future__ import annotations

import datetime
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.database import get_db
from app.main import app
from app.models import Analysis, UploadedFile, AnalysisImage
from app.services.change_detection import _ensure_results_dir
from app.services.mission_report import (
    build_mission_report_data,
    generate_mission_report_pdf,
)


@pytest.fixture
def test_db():
    db_gen = get_db()
    db = next(db_gen)
    yield db


@pytest.fixture
def completed_analysis(test_db):
    """Create or retrieve a completed bi-temporal analysis with full telemetry."""
    aid = "test_mission_report_analysis"
    analysis = test_db.query(Analysis).filter(Analysis.id == aid).first()
    if not analysis:
        analysis = Analysis(
            id=aid,
            mode="bi_temporal",
            query="Analyze bi-temporal changes in industrial sector",
            status="completed",
            answer_text="Vegetation clearing observed along the southeastern corridor.",
            confidence=None,  # Strictly None
            change_map={
                "overlayUrl": "/api/results/regression_analytics_test_changemap.png",
                "changed_pixel_pct": 9.09,
                "total_pixels": 786432,
                "changed_pixels": 71495,
                "analytics": {
                    "global_statistics": {
                        "total_pixel_count": 786432,
                        "changed_pixel_count": 71495,
                        "changed_pixel_percentage": 9.09,
                        "physical_area": {
                            "available": False,
                            "reason": "Physical area unavailable — reliable spatial resolution metadata was not provided.",
                        },
                    },
                    "geospatial_metadata": {"crs": None, "resolution": None},
                    "hotspots": [
                        {"id": 1, "pixel_area": 1200, "centroid_px": [150.0, 200.0], "bbox_px": [100, 180, 200, 220]}
                    ],
                },
            },
            tool_invocations=[
                {"toolId": "change_detector", "executionMode": "real"},
                {"toolId": "change_vqa", "executionMode": "real"},
            ],
            evidence=["Bi-temporal contrast localized across coordinates."],
        )
        test_db.add(analysis)
        test_db.commit()
    return analysis


# ---------------------------------------------------------------------------
# Test A: Report Generation from Completed Analysis
# ---------------------------------------------------------------------------
def test_a_report_generation_from_completed_analysis(test_db, completed_analysis):
    """Verify report data is cleanly generated from completed analysis."""
    report = build_mission_report_data(test_db, completed_analysis)

    assert report["analysis_id"] == completed_analysis.id
    assert report["mode"] == "bi_temporal"
    assert report["status"] == "completed"
    assert "9.09%" in report["executive_summary"]
    assert report["change_detection"]["changed_pixel_pct"] == 9.09
    assert report["change_detection"]["changed_pixel_count"] == 71495


# ---------------------------------------------------------------------------
# Test B: Missing Optional Metadata
# ---------------------------------------------------------------------------
def test_b_missing_optional_metadata(test_db):
    """Report handles analysis missing optional metadata without throwing."""
    bare_aid = "test_bare_analysis"
    analysis = test_db.query(Analysis).filter(Analysis.id == bare_aid).first()
    if not analysis:
        analysis = Analysis(
            id=bare_aid,
            mode="bi_temporal",
            query="Detect change",
            status="completed",
            change_map=None,
            input_summary=None,
            confidence=None,
        )
        test_db.add(analysis)
        test_db.commit()

    report = build_mission_report_data(test_db, analysis)
    assert report["analysis_id"] == bare_aid
    assert report["change_detection"]["changed_pixel_pct"] is None
    assert "Physical area unavailable" in " ".join(report["limitations"])


# ---------------------------------------------------------------------------
# Test C: No ROI Case
# ---------------------------------------------------------------------------
def test_c_no_roi_case(test_db, completed_analysis):
    """When no ROI was selected, report explicitly reflects this state."""
    report = build_mission_report_data(test_db, completed_analysis, roi_data=None)

    assert report["roi_investigation"]["performed"] is False
    assert report["roi_investigation"]["details"] is None
    assert "ROI investigation was not performed" in " ".join(report["limitations"])


# ---------------------------------------------------------------------------
# Test D: ROI Present Case
# ---------------------------------------------------------------------------
def test_d_roi_present_case(test_db, completed_analysis):
    """When ROI was selected, report includes ROI metrics and comparison."""
    sample_roi = {
        "roi": {
            "pixel_coordinates": {"x1": 100, "y1": 100, "x2": 300, "y2": 300, "width": 200, "height": 200}
        },
        "statistics": {
            "total_pixels": 40000,
            "changed_pixels": 10000,
            "changed_percentage": 25.0,
            "unchanged_percentage": 75.0,
        },
        "global_comparison": {
            "global_changed_percentage": 9.09,
            "roi_changed_percentage": 25.0,
            "difference_percentage": 15.91,
            "summary": "Selected ROI has 2.75x higher change density.",
        },
        "physical_area": {"available": False, "reason": "No GSD"},
        "hotspots": {"hotspots_count_total": 5},
    }

    report = build_mission_report_data(test_db, completed_analysis, roi_data=sample_roi)
    assert report["roi_investigation"]["performed"] is True
    assert report["roi_investigation"]["details"]["statistics"]["changed_percentage"] == 25.0
    assert "25.00%" in report["executive_summary"]


# ---------------------------------------------------------------------------
# Test E: No VQA Case
# ---------------------------------------------------------------------------
def test_e_no_vqa_case(test_db):
    """When Change VQA was not requested, report faithfully declares it."""
    no_vqa_aid = "test_no_vqa_analysis"
    analysis = test_db.query(Analysis).filter(Analysis.id == no_vqa_aid).first()
    if not analysis:
        analysis = Analysis(
            id=no_vqa_aid,
            mode="bi_temporal",
            query="Detect change without VQA",
            status="completed",
            change_map={"changed_pixel_pct": 5.0},
            tool_invocations=[{"toolId": "change_detector", "executionMode": "real"}],
            confidence=None,
        )
        test_db.add(analysis)
        test_db.commit()

    report = build_mission_report_data(test_db, analysis, roi_data=None)
    assert report["ai_interpretation"]["executed"] is False
    assert "Natural-language AI interpretation (Change VQA) was not requested" in " ".join(report["limitations"])


# ---------------------------------------------------------------------------
# Test F: VQA Present Case
# ---------------------------------------------------------------------------
def test_f_vqa_present_case(test_db, completed_analysis):
    """When VQA is present, report includes VQA interpretation and uncalibrated label."""
    report = build_mission_report_data(test_db, completed_analysis)

    assert report["ai_interpretation"]["executed"] is True
    assert "Vegetation clearing" in report["ai_interpretation"]["answer"]
    assert report["ai_interpretation"]["confidence_label"] == "N/A — Uncalibrated"


# ---------------------------------------------------------------------------
# Test G: Confidence Remains None
# ---------------------------------------------------------------------------
def test_g_confidence_remains_none(test_db, completed_analysis):
    """Report strictly preserves None confidence without fabricating numbers."""
    report = build_mission_report_data(test_db, completed_analysis)

    assert report["change_detection"]["confidence"] is None
    assert report["change_detection"]["confidence_label"] == "N/A — Uncalibrated"
    assert "strictly reported as N/A — Uncalibrated" in " ".join(report["limitations"])


# ---------------------------------------------------------------------------
# Test H: No Fabricated Physical Area
# ---------------------------------------------------------------------------
def test_h_no_fabricated_physical_area(test_db, completed_analysis):
    """Physical area is marked unavailable without fabricating assumptions."""
    report = build_mission_report_data(test_db, completed_analysis)

    pa = report["geospatial_analytics"]["physical_area"]
    assert pa.get("available") in (True, False)
    assert "reliable spatial resolution metadata was not provided" in pa.get("reason", "")


# ---------------------------------------------------------------------------
# Test I: Report Contains Threshold 0.70
# ---------------------------------------------------------------------------
def test_i_report_contains_threshold_070(test_db, completed_analysis):
    """Report strictly records threshold 0.70."""
    report = build_mission_report_data(test_db, completed_analysis)
    assert report["change_detection"]["threshold_used"] == 0.70


# ---------------------------------------------------------------------------
# Test J: Report Contains Existing Checkpoint
# ---------------------------------------------------------------------------
def test_j_report_contains_existing_checkpoint(test_db, completed_analysis):
    """Report specifies the LEVIR-CD trained best_model.pt checkpoint."""
    report = build_mission_report_data(test_db, completed_analysis)
    assert "best_model.pt" in report["change_detection"]["checkpoint"]


# ---------------------------------------------------------------------------
# Test K: Report Does Not Trigger Change Detection
# ---------------------------------------------------------------------------
def test_k_report_does_not_trigger_change_detection(test_db, completed_analysis, monkeypatch):
    """Building the report must never rerun Siamese U-Net or change detector."""
    import app.services.model_inference as mi

    def _forbidden(*args, **kwargs):
        raise AssertionError("Illegal call to change detection during report generation!")

    monkeypatch.setattr(mi, "run_change_detection", _forbidden)
    # Should complete without error
    report = build_mission_report_data(test_db, completed_analysis)
    assert report["analysis_id"] == completed_analysis.id


# ---------------------------------------------------------------------------
# Test L: Report Does Not Trigger VQA
# ---------------------------------------------------------------------------
def test_l_report_does_not_trigger_vqa(test_db, completed_analysis, monkeypatch):
    """Building the report must never rerun vision-language model inference."""
    import app.services.change_vqa as cvqa

    def _forbidden_vqa(*args, **kwargs):
        raise AssertionError("Illegal call to Change VQA during report generation!")

    monkeypatch.setattr(cvqa, "run_change_vqa", _forbidden_vqa)
    # Should complete without error
    report = build_mission_report_data(test_db, completed_analysis)
    assert report["ai_interpretation"]["executed"] is True


# ---------------------------------------------------------------------------
# Test M: PDF Generation Succeeds
# ---------------------------------------------------------------------------
def test_m_pdf_generation_succeeds(test_db, completed_analysis, tmp_path):
    """PDF generation produces a valid, readable PDF document."""
    report = build_mission_report_data(test_db, completed_analysis)
    pdf_path = tmp_path / "test_report.pdf"

    res_path = generate_mission_report_pdf(report, pdf_path)
    assert res_path.exists()
    assert res_path.stat().st_size > 1000  # Non-trivial PDF size

    # Verify PDF magic bytes
    with open(res_path, "rb") as f:
        header = f.read(5)
    assert header == b"%PDF-"


# ---------------------------------------------------------------------------
# Test N: Malformed / Missing Analysis Handling
# ---------------------------------------------------------------------------
def test_n_missing_analysis_handling():
    """Requesting report for non-existent analysis returns 404."""
    client = TestClient(app)
    resp = client.post("/api/analysis/non_existent_mission_id/report")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Test O: Existing API Responses Remain Compatible
# ---------------------------------------------------------------------------
def test_o_existing_api_responses_remain_compatible(test_db, completed_analysis):
    """POST /api/analysis/{analysis_id}/report returns expected schema without breaking others."""
    client = TestClient(app)
    resp = client.post(
        f"/api/analysis/{completed_analysis.id}/report",
        json={"generate_pdf": True},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "report" in data
    assert "pdf_url" in data
    assert data["pdf_url"] is not None
    assert f"mission_report_{completed_analysis.id}.pdf" in data["pdf_url"]
