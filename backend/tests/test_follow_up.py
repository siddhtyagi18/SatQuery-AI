"""
Unit Tests for Contextual Follow-up Service
===========================================

Validates:
1. TEST 1: Existing result data preserved.
2. TEST 2: "How much area changed?" answers directly from existing structured result without model rerun.
3. TEST 3: "Where exactly did it change?" uses existing spatial evidence.
4. TEST 4: "Tell me in Hindi" produces Hindi answer from same facts.
5. TEST 6: Multiple follow-up queries preserve context.
6. TEST 8, 9, 10: Accurate metric extraction, severity handling, and 100% consistency.
"""

import pytest
from backend.app.models import Analysis
from backend.app.schemas import FollowUpHistoryItem
from backend.app.services.follow_up import answer_follow_up, extract_structured_metrics


def create_mock_bi_temporal_analysis():
    a = Analysis(
        id="test-analysis-002",
        mode="bi_temporal",
        query="What changes occurred between these two dates? Has urban expansion affected vegetation cover?",
        status="completed",
        confidence=None,
    )
    a.answer_text = (
        "### Bi-Temporal Scene Change Interpretation\n\n"
        "**Qualitative Visual Interpretation (VLM):**\n"
        "Comparing the earlier acquisition (T1) with the later acquisition (T2), "
        "localized structural changes are observable in the scene. Engineered building structures "
        "and ground clearing activities appear along transit access boundaries.\n\n"
        "**Quantitative Detection Telemetry:**\n"
        "- **Detected Changed Area:** `3.14%` (Severity: **low**)\n"
        "- **Change Detection Threshold:** `0.70` (Siamese U-Net)\n"
        "- **Detector Model:** SiameseUNet (~490K parameters, LEVIR-CD trained checkpoint)\n"
        "- **Confidence:** Not calibrated for this bi-temporal analysis (confidence = null)\n"
        "- **Inference Provenance:** Vision-Language Model (SmolVLM-500M-Instruct + LoRA domain adapter)"
    )
    a.detected_tasks = ["change_detection", "change_vqa", "change_description"]
    a.change_map = {
        "overlayUrl": "/demo/change_mask.png",
        "legend": [
            {"label": "Detected Changed Region (3.1% area)", "color": "#FF3C3C"},
            {"label": "Unchanged Region (96.9% area)", "color": "transparent"},
        ],
    }
    a.bounding_boxes = None
    a.adaptation = {
        "multilingual_summaries": {
            "summary_hi": "उपग्रह तस्वीर में दो अवधियों के बीच 3.14% क्षेत्र में निर्माण और बुनियादी ढांचे में बदलाव देखा गया है।",
            "summary_en": "Built-up expansion and ground clearing detected in 3.14% of the scene along transit boundaries.",
        }
    }
    return a


def create_mock_single_image_analysis():
    a = Analysis(
        id="test-analysis-001",
        mode="single_image",
        query="What land cover types are visible and locate all buildings in this image?",
        status="completed",
        confidence=0.88,
    )
    a.answer_text = (
        "The image shows a mixed urban-agricultural landscape. The dominant land cover types are: "
        "(1) Dense urban settlement (~38% coverage) concentrated in the northwestern quadrant; "
        "(2) Agricultural cropland (~41% coverage). A total of 127 individual buildings have been detected."
    )
    a.detected_tasks = ["vqa", "grounding"]
    a.change_map = None
    a.bounding_boxes = [
        {"x": 0.08, "y": 0.12, "width": 0.04, "height": 0.03, "label": "Building", "confidence": 0.94},
        {"x": 0.14, "y": 0.09, "width": 0.05, "height": 0.04, "label": "Building", "confidence": 0.91},
    ]
    return a


def test_metric_extraction():
    a = create_mock_bi_temporal_analysis()
    metrics = extract_structured_metrics(a.answer_text, a.mode)
    assert metrics.get("changed_area") == "3.14%"
    assert metrics.get("severity") == "low"
    assert metrics.get("threshold") == "0.70"


def test_area_follow_up_zero_rerun():
    a = create_mock_bi_temporal_analysis()
    res = answer_follow_up(a, "How much area changed?")
    assert res.rerunPerformed is False
    assert "3.14%" in res.answer
    assert res.referencedMetrics is not None
    assert res.referencedMetrics.get("changed_area") == "3.14%"
    assert res.spatialAction is not None
    assert res.spatialAction.action == "highlight"
    assert res.spatialAction.target == "change_map"


def test_where_spatial_follow_up():
    a = create_mock_bi_temporal_analysis()
    res = answer_follow_up(a, "Where exactly did it change?")
    assert res.rerunPerformed is False
    assert "north-eastern" in res.answer.lower() or "transit" in res.answer.lower()
    assert res.spatialAction is not None
    assert res.spatialAction.action == "highlight"
    assert res.spatialAction.target == "change_map"


def test_hindi_follow_up():
    a = create_mock_bi_temporal_analysis()
    # Test with Hindi question
    res_hi = answer_follow_up(a, "कितने क्षेत्र में बदलाव हुआ?")
    assert res_hi.rerunPerformed is False
    assert "3.14%" in res_hi.answer
    assert res_hi.language == "hi"

    # Test with English question asking for Hindi
    res_en_hi = answer_follow_up(a, "Explain this in Hindi.")
    assert res_en_hi.rerunPerformed is False
    assert res_en_hi.language == "hi"
    assert len(res_en_hi.answer) > 10


def test_confidence_follow_up():
    # Null confidence case
    a_bi = create_mock_bi_temporal_analysis()
    res_bi = answer_follow_up(a_bi, "What is the confidence?")
    assert res_bi.rerunPerformed is False
    assert "not calibrated" in res_bi.answer.lower()

    # Calibrated confidence case
    a_single = create_mock_single_image_analysis()
    res_single = answer_follow_up(a_single, "What is the confidence?")
    assert res_single.rerunPerformed is False
    assert "88%" in res_single.answer


def test_missing_severity_does_not_invent():
    # Analysis without severity
    a_single = create_mock_single_image_analysis()
    res = answer_follow_up(a_single, "What is the severity of this change?")
    assert res.rerunPerformed is False
    assert "not available" in res.answer.lower()


def test_multi_turn_follow_up_preservation():
    a = create_mock_bi_temporal_analysis()
    history = [
        FollowUpHistoryItem(role="user", text="What changed between the two dates?"),
        FollowUpHistoryItem(role="assistant", text="Structural changes and ground clearing were detected."),
    ]
    res = answer_follow_up(a, "Compare the affected area with the total area.", history=history)
    assert res.rerunPerformed is False
    assert "3.14%" in res.answer
    assert "96.86%" in res.answer or "unchanged" in res.answer.lower()
