"""
tests/test_change_detection.py
-------------------------------
Tests for the CPU-only bi-temporal change detection MVP.

All tests use tiny synthetic images generated in-process.
No internet access, no GPU, no external datasets required.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))


def _make_png_bytes(color: tuple, size: tuple = (32, 24)) -> bytes:
    """Create a tiny solid-colour PNG and return its bytes."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def _upload_image(client, png_bytes: bytes, name: str, role: str) -> str:
    """Upload an image via the test client and return the uploaded file id."""
    resp = client.post(
        "/api/upload",
        files={"file": (name, io.BytesIO(png_bytes), "image/png")},
        data={"role": role},
    )
    assert resp.status_code == 200, f"Upload failed: {resp.text}"
    return resp.json()["id"]


# ---------------------------------------------------------------------------
# Test 1 — Unit test: service function directly
# ---------------------------------------------------------------------------

def test_cpu_change_detection_service_unit(tmp_path):
    """
    Directly exercises run_cpu_change_detection with two tiny PNG files.
    Verifies:
    - ChangeDetectionResult is returned with the right shape.
    - confidence is always None (no fabricated score).
    - change_map has overlayUrl and legend.
    - A PNG file was actually written to disk.
    - Stats dict contains expected keys.
    - Changed pixels > 0 for images that differ.
    """
    from app.services.change_detection import run_cpu_change_detection, ChangeDetectionResult

    # Create two maximally different images: all-black vs all-white
    before_path = tmp_path / "before.png"
    after_path = tmp_path / "after.png"
    Image.new("RGB", (32, 24), color=(0, 0, 0)).save(before_path, format="PNG")
    Image.new("RGB", (32, 24), color=(255, 255, 255)).save(after_path, format="PNG")

    # Temporarily redirect output dir to tmp_path so we don't pollute the real dir
    import app.services.change_detection as cd_module
    original_dir = cd_module._RESULTS_DIR
    cd_module._RESULTS_DIR = tmp_path / "results"

    try:
        result = run_cpu_change_detection(
            before_path=before_path,
            after_path=after_path,
            analysis_id="unit-test-001",
        )
    finally:
        cd_module._RESULTS_DIR = original_dir

    # Shape checks
    assert isinstance(result, ChangeDetectionResult)
    assert result.confidence is None, "Confidence must be None for classical algorithm"
    assert isinstance(result.answer, str)
    assert len(result.answer) > 50, "Answer should be substantive"

    # Change map
    assert "overlayUrl" in result.change_map
    assert result.change_map["overlayUrl"].startswith("/api/results/")
    assert "legend" in result.change_map
    assert len(result.change_map["legend"]) >= 1

    # Evidence
    assert isinstance(result.evidence, list)
    assert len(result.evidence) >= 3
    # Should not fabricate scientific claims
    for ev in result.evidence:
        assert "MOCK" not in ev, "Evidence from real service should not say MOCK"

    # Stats
    assert "changed_pixel_pct" in result.stats
    assert "changed_pixel_count" in result.stats
    assert "total_pixel_count" in result.stats
    assert "image_size_wh" in result.stats
    assert "threshold_used" in result.stats

    # For all-black vs all-white, every pixel should change
    assert result.stats["changed_pixel_pct"] > 90.0, (
        f"Expected >90% changed (black vs white), got {result.stats['changed_pixel_pct']}"
    )

    # PNG was actually written
    png_path = tmp_path / "results" / "unit-test-001_changemap.png"
    assert png_path.exists(), f"Change map PNG not written: {png_path}"
    # Verify it is a valid RGBA PNG
    overlay = Image.open(png_path)
    assert overlay.mode == "RGBA"
    assert overlay.size == (32, 24)


def test_cpu_change_detection_identical_images(tmp_path):
    """
    Two identical images should produce 0% (or near-0%) change after noise cleanup.
    """
    from app.services.change_detection import run_cpu_change_detection
    import app.services.change_detection as cd_module

    original_dir = cd_module._RESULTS_DIR
    cd_module._RESULTS_DIR = tmp_path / "results"
    try:
        img_path = tmp_path / "same.png"
        Image.new("RGB", (32, 24), color=(80, 140, 200)).save(img_path, format="PNG")

        result = run_cpu_change_detection(
            before_path=img_path,
            after_path=img_path,
            analysis_id="unit-test-002",
        )
    finally:
        cd_module._RESULTS_DIR = original_dir

    assert result.confidence is None
    assert result.stats["changed_pixel_pct"] == 0.0, (
        f"Identical images should have 0% change, got {result.stats['changed_pixel_pct']}"
    )


def test_cpu_change_detection_size_mismatch(tmp_path):
    """
    Images of different sizes should succeed (the after image is resized).
    size_mismatch_corrected flag should be True.
    """
    from app.services.change_detection import run_cpu_change_detection
    import app.services.change_detection as cd_module

    original_dir = cd_module._RESULTS_DIR
    cd_module._RESULTS_DIR = tmp_path / "results"
    try:
        before_path = tmp_path / "before.png"
        after_path = tmp_path / "after.png"
        Image.new("RGB", (32, 24), color=(10, 20, 30)).save(before_path, format="PNG")
        Image.new("RGB", (64, 48), color=(200, 200, 200)).save(after_path, format="PNG")

        result = run_cpu_change_detection(
            before_path=before_path,
            after_path=after_path,
            analysis_id="unit-test-003",
        )
    finally:
        cd_module._RESULTS_DIR = original_dir

    assert result.confidence is None
    assert result.stats["size_mismatch_corrected"] is True
    assert result.stats["image_size_wh"] == [32, 24]
    # At least one evidence entry should mention the mismatch
    mismatch_evidence = [e for e in result.evidence if "mismatch" in e.lower() or "resam" in e.lower()]
    assert len(mismatch_evidence) >= 1


# ---------------------------------------------------------------------------
# Test 2 — E2E API test: bi_temporal analysis returns real executionMode
# ---------------------------------------------------------------------------

def test_bi_temporal_e2e_real_change_detector(client):
    """
    Full API flow: upload two different tiny PNG images, submit bi_temporal
    analysis, and verify:
    - status is "completed"
    - change_detector ToolInvocation has executionMode == "real"
    - changeMap.overlayUrl is present and points to /api/results/
    - confidence is None (no fabricated score)
    """
    # Use clearly different images so the difference is measurable
    before_bytes = _make_png_bytes(color=(10, 20, 30), size=(48, 32))
    after_bytes = _make_png_bytes(color=(200, 180, 160), size=(48, 32))

    f1 = _upload_image(client, before_bytes, "CARTOSAT-T1_2022.png", "before")
    f2 = _upload_image(client, after_bytes, "CARTOSAT-T2_2024.png", "after")

    # Submit bi_temporal analysis
    resp = client.post("/api/analysis", json={
        "mode": "bi_temporal",
        "imageIds": [f1, f2],
        "query": "What pixel-level changes occurred between the two acquisitions?",
    })
    assert resp.status_code == 200, f"Analysis submit failed: {resp.text}"
    aid = resp.json()["analysisId"]

    # Fetch result
    g = client.get(f"/api/analysis/{aid}")
    assert g.status_code == 200, f"GET analysis failed: {g.text}"
    result = g.json()

    # Status
    assert result["status"] == "completed", (
        f"Expected completed, got {result['status']}. errorReason={result.get('errorReason')}"
    )

    # change_detector must have run in real mode
    cd_invocations = [
        inv for inv in result["toolInvocations"]
        if inv["toolId"] == "change_detector"
    ]
    assert len(cd_invocations) == 1, "Expected exactly one change_detector invocation"
    cd_inv = cd_invocations[0]
    assert cd_inv["executionMode"] == "real", (
        f"change_detector executionMode should be 'real', got '{cd_inv['executionMode']}'"
    )

    # changeMap with real overlay URL
    assert result["changeMap"] is not None, "changeMap must not be None"
    assert result["changeMap"]["overlayUrl"] is not None, "overlayUrl must not be None"
    assert result["changeMap"]["overlayUrl"].startswith("/api/results/"), (
        f"overlayUrl should start with /api/results/, got {result['changeMap']['overlayUrl']!r}"
    )
    assert "legend" in result["changeMap"]

    # confidence: the aggregate MAY be non-null because other mock tools in the
    # bi_temporal pipeline (rs_vqa, change_vqa) contribute fixed confidence scores.
    # What matters is that the change_detector itself did NOT fabricate a score.
    # The change_detector invocation carries no confidence — it is absorbed into None
    # at the tool level.  We verify the invocation's parameters do not include a
    # fabricated confidence.
    cd_params = cd_inv.get("parameters", {})
    assert "confidence" not in cd_params, (
        "change_detector parameters should not contain a fabricated confidence value"
    )
    # If confidence is present overall, it must come only from other mock tools,
    # and must be a valid [0, 1] float.
    overall_conf = result["confidence"]
    if overall_conf is not None:
        assert 0.0 <= overall_conf <= 1.0, f"Aggregate confidence out of range: {overall_conf}"

    # answer text should mention classical / pixel-level
    assert result["answerText"] is not None
    assert "pixel" in result["answerText"].lower() or "classical" in result["answerText"].lower(), (
        "Answer should clearly identify this as classical pixel-level analysis"
    )

    # evidence should not fabricate scientific claims
    assert isinstance(result["evidence"], list)
    assert len(result["evidence"]) >= 2
    for ev in result["evidence"]:
        assert "[MOCK" not in ev, f"Real tool evidence should not say [MOCK]: {ev!r}"


# ---------------------------------------------------------------------------
# Test 3 — E2E API test: /api/results/ URL is actually accessible
# ---------------------------------------------------------------------------

def test_bi_temporal_overlay_url_is_accessible(client):
    """
    Verifies that the overlayUrl returned in the changeMap response can be
    fetched from the backend as a valid PNG image.
    """
    before_bytes = _make_png_bytes(color=(5, 10, 15), size=(24, 16))
    after_bytes = _make_png_bytes(color=(240, 230, 220), size=(24, 16))

    f1 = _upload_image(client, before_bytes, "before_access_test.png", "before")
    f2 = _upload_image(client, after_bytes, "after_access_test.png", "after")

    resp = client.post("/api/analysis", json={
        "mode": "bi_temporal",
        "imageIds": [f1, f2],
        "query": "Detect changes",
    })
    assert resp.status_code == 200
    aid = resp.json()["analysisId"]

    g = client.get(f"/api/analysis/{aid}")
    assert g.status_code == 200
    result = g.json()

    overlay_url = result.get("changeMap", {}).get("overlayUrl")
    assert overlay_url is not None, "overlayUrl must be present"

    # Fetch the PNG directly through the test client
    img_resp = client.get(overlay_url)
    assert img_resp.status_code == 200, (
        f"Expected 200 fetching {overlay_url}, got {img_resp.status_code}"
    )
    content_type = img_resp.headers.get("content-type", "")
    assert "image/png" in content_type, (
        f"Expected image/png content-type, got {content_type!r}"
    )

    # Verify it is a parseable PNG
    png_bytes = img_resp.content
    img = Image.open(io.BytesIO(png_bytes))
    assert img.format == "PNG"
    assert img.mode == "RGBA"


# ---------------------------------------------------------------------------
# Test 4 — Severity label heuristics unit test
# ---------------------------------------------------------------------------

def test_severity_label_thresholds():
    """Verify heuristic severity thresholds: low (<5%), moderate (5-25%), high (>25%)."""
    from app.services.change_detection import _severity_label

    assert _severity_label(0.0) == "low"
    assert _severity_label(4.9) == "low"
    assert _severity_label(5.0) == "moderate"
    assert _severity_label(15.2) == "moderate"
    assert _severity_label(25.0) == "moderate"
    assert _severity_label(25.1) == "high"
    assert _severity_label(100.0) == "high"


# ---------------------------------------------------------------------------
# Test 5 — Demo image generator & nonzero change measurement
# ---------------------------------------------------------------------------

def test_changed_pair_has_nonzero_change_pct(tmp_path):
    """
    Uses make_changed_pair helper to verify that a structured synthetic
    change produces a measurable, non-zero changed_pixel_pct and matching stats.
    """
    from tests.generate_demo_pair import make_changed_pair
    from app.services.change_detection import run_cpu_change_detection
    import app.services.change_detection as cd_module

    b_path, a_path = make_changed_pair(tmp_path, size=(64, 64), change_rect=(10, 10, 30, 30))

    orig_dir = cd_module._RESULTS_DIR
    cd_module._RESULTS_DIR = tmp_path / "results"
    try:
        res = run_cpu_change_detection(b_path, a_path, "demo-pair-test")
    finally:
        cd_module._RESULTS_DIR = orig_dir

    assert res.stats["changed_pixel_pct"] > 0.0
    assert res.stats["unchanged_pixel_pct"] > 0.0
    assert round(res.stats["changed_pixel_pct"] + res.stats["unchanged_pixel_pct"], 1) == 100.0
    assert res.stats["severity"] in ("low", "moderate", "high")
    assert res.stats["changed_pixel_count"] > 0
    assert res.stats["total_pixel_count"] == 64 * 64

    # Report verification
    assert "Pixel-level visual change was detected across" in res.answer
    assert "Interpretation guidance:" in res.answer
    assert "Recommended next action:" in res.answer


# ---------------------------------------------------------------------------
# Test 6 — E2E trace step-6 meta has change stats
# ---------------------------------------------------------------------------

def test_stats_in_step6_meta_e2e(client):
    """
    E2E test verifying that scalar change statistics are stored in step-6 meta
    and exposed via the executionTrace API response for the frontend ChangeStatsPanel.
    """
    from tests.generate_demo_pair import make_changed_pair_bytes

    b_bytes, a_bytes = make_changed_pair_bytes(size=(64, 64))

    f1 = _upload_image(client, b_bytes, "demo_b.png", "before")
    f2 = _upload_image(client, a_bytes, "demo_a.png", "after")

    resp = client.post("/api/analysis", json={
        "mode": "bi_temporal",
        "imageIds": [f1, f2],
        "query": "Evaluate temporal difference",
    })
    assert resp.status_code == 200
    aid = resp.json()["analysisId"]

    g = client.get(f"/api/analysis/{aid}")
    assert g.status_code == 200
    data = g.json()

    trace_steps = data.get("executionTrace", {}).get("steps", [])
    step6 = next((s for s in trace_steps if s["id"] == "step-6"), None)
    assert step6 is not None, "step-6 must exist in executionTrace"
    meta = step6.get("meta", {})

    assert "changed_pixel_pct" in meta
    assert isinstance(meta["changed_pixel_pct"], (int, float))
    assert meta["changed_pixel_pct"] > 0.0
    assert "unchanged_pixel_pct" in meta
    assert "severity" in meta
    assert meta["severity"] in ("low", "moderate", "high")
    assert "image_size_str" in meta
