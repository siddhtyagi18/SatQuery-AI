from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_satquery_phase2.db")
os.environ.setdefault("UPLOAD_DIR", "./test_uploads_p2")
os.environ.setdefault("DEBUG", "true")
os.environ.setdefault("VQA_MODE", "mock")
os.environ.setdefault("FIREBASE_ENABLED", "false")

from app.config import get_settings

_settings = get_settings()
_settings.DATABASE_URL = "sqlite:///./test_satquery_phase2.db"
_settings.UPLOAD_DIR = "./test_uploads_p2"
_settings.VQA_MODE = "mock"
_settings.FIREBASE_ENABLED = False

from app.database import Base, engine, get_db
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

_TEST_DB = BACKEND_ROOT / "test_satquery_phase2.db"
if _TEST_DB.exists():
    _TEST_DB.unlink()
_TEST_UPLOADS = BACKEND_ROOT / "test_uploads_p2"
if _TEST_UPLOADS.exists():
    import shutil
    shutil.rmtree(_TEST_UPLOADS, ignore_errors=True)


@pytest.fixture(scope="session")
def db_engine():
    engine = create_engine(
        "sqlite:///./test_satquery_phase2.db", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def db(db_engine):
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def client(db):
    from app.main import app

    def _override_get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = _override_get_db
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture()
def png_file():
    from PIL import Image

    buf = io.BytesIO()
    img = Image.new("RGB", (64, 48), color=(30, 80, 120))
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


@pytest.fixture()
def sample_png_name():
    return "optical_sample_2024-03-15.png"


def _upload_image(client, png_file, name, role):
    resp = client.post(
        "/api/upload",
        files={"file": (name, png_file, "image/png")},
        data={"role": role},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["id"]


# ---------------------------------------------------------------------------
# 1. Firebase service layer (mocked, never hits real GCP)
# ---------------------------------------------------------------------------


def test_firebase_service_disabled_by_default():
    from app.services.firebase import (
        FirebaseRepository,
        is_firebase_enabled,
    )
    assert is_firebase_enabled() is False
    repo = FirebaseRepository()
    # Graceful no-op when disabled — should not raise, returns None.
    assert repo.save_analysis("any-id", {"foo": 1}) is None
    assert repo.get_analysis("any-id") is None
    assert repo.update_analysis("any-id", {"a": 1}) is None


def test_firebase_repo_mocked_enabled(db):
    """Simulate FIREBASE_ENABLED=true with a mocked firestore client."""
    from app.services.firebase import FirebaseRepository, _firestore_client, _firebase_initialized, _ensure_initialized
    import app.services.firebase as fb_mod

    # Reset module-level cache so our override is seen.
    orig_client = fb_mod._firestore_client
    orig_init = fb_mod._firebase_initialized
    try:
        fake_db = MagicMock()
        fake_collection = MagicMock()
        fake_doc = MagicMock()
        fake_collection.document.return_value = fake_doc
        fake_db.collection.return_value = fake_collection

        fb_mod._firestore_client = fake_db
        fb_mod._firebase_initialized = True

        repo = FirebaseRepository()
        # save should delegate to mocked client
        repo.save_analysis("aid1", {"a": 1})
        fake_db.collection.assert_called_once()
        fake_doc.set.assert_called_once_with({"a": 1})

        # get — simulate exists
        fake_doc_get = MagicMock()
        fake_doc_get.exists = True
        fake_doc_get.to_dict.return_value = {"a": 1, "b": 2}
        fake_doc.get.return_value = fake_doc_get
        got = repo.get_analysis("aid1")
        assert got == {"a": 1, "b": 2}

        # update
        repo.update_analysis("aid1", {"b": 3})
        fake_doc.update.assert_called_once_with({"b": 3})
    finally:
        fb_mod._firestore_client = orig_client
        fb_mod._firebase_initialized = orig_init


def test_firebase_storage_service_no_op_when_disabled(tmp_path):
    from app.services.firebase import FirebaseStorageService

    svc = FirebaseStorageService()
    f = tmp_path / "foo.bin"
    f.write_bytes(b"hello")
    # Without firebase enabled, upload is a no-op returning None.
    assert svc.upload_file(f, "storage/path.bin") is None
    assert svc.download_to_temp("storage/path.bin") is None


# ---------------------------------------------------------------------------
# 2. Upload + local storage (Phase 1 preserved + file-exists validation)
# ---------------------------------------------------------------------------


def test_upload_png_extracts_metadata(client, png_file, db):
    fid = _upload_image(client, png_file, "RESOURCESAT-2_LISS-IV_2024-03-15.png", "single")
    # Reading it back via analysis endpoint isn't needed yet; we just need the
    # upload service to have populated metadata fields.
    from app.models import UploadedFile

    f = db.query(UploadedFile).filter(UploadedFile.id == fid).first()
    assert f is not None
    assert f.file_format == "PNG"
    assert f.modality == "multispectral" or f.modality == "optical"
    assert f.acquisition_date == "2024-03-15"
    assert f.width_px == 64
    assert f.height_px == 48
    assert Path(f.file_path).exists()


def test_upload_unsupported_extension_rejected(client):
    buf = io.BytesIO(b"not a real file")
    resp = client.post(
        "/api/upload",
        files={"file": ("bad.exe", buf, "application/octet-stream")},
        data={"role": "single"},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# 3. Analysis creation + submission preserves Phase 1 contracts
# ---------------------------------------------------------------------------


def test_submit_analysis_returns_analysis_id_and_creates_record(client, png_file):
    fid = _upload_image(client, png_file, "optical_sample_2024-03-15.png", "single")
    resp = client.post(
        "/api/analysis",
        json={
            "mode": "single_image",
            "imageIds": [fid],
            "query": "What land cover types are visible?",
        },
    )
    assert resp.status_code == 200
    aid = resp.json()["analysisId"]
    assert aid and len(aid) > 0
    g = client.get(f"/api/analysis/{aid}")
    assert g.status_code == 200
    r = g.json()
    assert r["id"] == aid
    assert r["status"] == "completed"
    # Phase 1-preserved fields
    assert r["mode"] == "single_image"
    assert isinstance(r["detectedTasks"], list) and len(r["detectedTasks"]) >= 1
    assert isinstance(r["toolInvocations"], list) and len(r["toolInvocations"]) >= 1
    assert isinstance(r["executionTrace"]["steps"], list)
    # Phase 2: each ToolInvocation now has an executionMode.
    for inv in r["toolInvocations"]:
        assert "executionMode" in inv
        assert inv["executionMode"] in ("real", "mock")


# ---------------------------------------------------------------------------
# 4. VQA routing logic — should_use_real_vqa respects VQA_MODE
# ---------------------------------------------------------------------------


def test_vqa_service_should_use_real_vqa_respects_settings():
    from app.services.vqa_service import VQAService

    svc = VQAService()

    from app.config import get_settings

    s = get_settings()
    orig = s.VQA_MODE
    try:
        s.VQA_MODE = "mock"
        assert svc.should_use_real_vqa("single_image", ["vqa"]) is False

        s.VQA_MODE = "real"
        assert svc.should_use_real_vqa("single_image", ["vqa"]) is True

        s.VQA_MODE = "auto"
        # auto + single_image + vqa = real
        assert svc.should_use_real_vqa("single_image", ["vqa"]) is True
        # auto + bi_temporal = still mock (only VQA on single images is Phase-2 real)
        assert svc.should_use_real_vqa("bi_temporal", ["vqa"]) is False
        # auto + single_image but no vqa task → caption still uses real (per implementation)
        # but grounding-only → still mock? actually caption also in real check
        assert svc.should_use_real_vqa("single_image", ["grounding"]) is False
    finally:
        s.VQA_MODE = orig


# ---------------------------------------------------------------------------
# 5. Real VQA service interface — with mocked model adapter (no downloads)
# ---------------------------------------------------------------------------


def test_vqa_service_mock_fallback_when_disabled(tmp_path, png_file):
    from app.services.vqa_service import VQAService, VQAServiceResult

    # Write png_file to a temporary path
    img_path = tmp_path / "test.png"
    from PIL import Image

    Image.new("RGB", (32, 32), color="blue").save(img_path, format="PNG")

    svc = VQAService()
    # When VQA_MODE=mock (the test default), result is a mock even when real factory would apply
    res = svc.run_real_or_fallback(
        query="What is in this image?",
        mode="single_image",
        image_file_paths=[img_path],
        tasks=["vqa"],
    )
    assert isinstance(res, VQAServiceResult)
    assert res.is_mock is True
    assert res.answer and len(res.answer) > 0
    assert res.run_context is not None
    assert res.run_context.execution_mode == "mock"
    assert res.confidence is not None  # mock produces a placeholder confidence


def test_vqa_service_real_pipeline_with_mocked_adapter(tmp_path):
    """Test the real pipeline code path using a fully mocked ModelManager + adapter.

    This ensures the preprocessing → adapter.load → preprocess_input → infer →
    validation codepath is exercised without touching HuggingFace servers or
    allocating a 500M-param model.
    """
    from PIL import Image
    from app.services import model_manager as mm_mod
    from app.services import vqa_service as vqa_mod
    from app.services.vqa_adapter import (
        VQAInferenceInput,
        VQAInferenceOutput,
    )
    from app.services.model_manager import LoadedModel

    # Build image
    img_path = tmp_path / "scene.png"
    Image.new("RGB", (100, 80), color=(70, 120, 30)).save(img_path, format="PNG")

    # Build a fake LoadedModel entry that a fake adapter will populate
    fake_model = MagicMock()
    fake_processor = MagicMock()
    fake_load_meta = {"fake": True, "device_actual": "cpu", "num_params_millions": 0}

    loaded_entry = LoadedModel(
        model_id="fake-model",
        model_object=fake_model,
        processor_object=fake_processor,
        load_duration_sec=0.01,
        metadata=fake_load_meta,
    )

    # Override the adapter used by vqa_service.get_adapter_for_model
    fake_adapter = MagicMock()
    fake_adapter.supports_model.return_value = True

    def fake_load():
        return fake_model, fake_processor, fake_load_meta

    fake_adapter.load = fake_load

    def fake_preprocess(inp, loaded):
        return {"tensors": "ok"}

    fake_adapter.preprocess_input = fake_preprocess

    def fake_infer(prep, loaded, inp):
        return VQAInferenceOutput(
            answer_text="This is a synthetic answer from the mocked adapter pipeline.",
            confidence=None,  # model does not emit calibrated confidence
            model_id="fake-model",
            inference_meta={"generated_token_count": 15},
        )

    fake_adapter.infer = fake_infer

    # Swap registries and model manager for this test
    with patch.object(vqa_mod, "get_adapter_for_model", return_value=fake_adapter):
        manager = mm_mod.get_model_manager()
        manager.reset()
        # Pre-seed the manager with our fake LoadedModel so the real codepath
        # runs through adapter.preprocess_input and adapter.infer without downloading.
        manager._cache["fake-model"] = loaded_entry
        fake_adapter.supports_model.return_value = True

        # The service should pick the model id from settings. Temporarily override.
        from app.config import get_settings

        s = get_settings()
        orig_model = s.VQA_MODEL_ID
        orig_mode = s.VQA_MODE
        try:
            s.VQA_MODEL_ID = "fake-model"
            s.VQA_MODE = "real"
            svc = vqa_mod.VQAService()
            res = svc.run_real_or_fallback(
                query="Describe this scene.",
                mode="single_image",
                image_file_paths=[img_path],
                tasks=["vqa"],
            )
            assert res.is_mock is False
            assert "synthetic answer" in res.answer
            assert res.confidence is None  # model-reported None preserved (no fabrication)
            assert res.run_context.execution_mode == "real"
            # Evidence should include no-confidence message and tokens count.
            joined_evidence = " ".join(res.evidence)
            assert "confidence=null" in joined_evidence.lower() or "confidence score" in joined_evidence.lower()
        finally:
            s.VQA_MODEL_ID = orig_model
            s.VQA_MODE = orig_mode
            manager.reset()


# ---------------------------------------------------------------------------
# 6. Model manager (singleton cache, cooldown on failure, thread-safe API)
# ---------------------------------------------------------------------------


def test_model_manager_singleton_and_cache_hit():
    from app.services.model_manager import ModelManager, get_model_manager, ModelLoadingError

    m1 = get_model_manager()
    m2 = get_model_manager()
    assert m1 is m2

    m1.reset()

    def slow_load():
        return ("MODEL", "PROC", {"k": 1})

    t0 = m1.load("my-model", slow_load)
    assert m1.is_loaded("my-model")
    # Second call should return cached entry without re-running load_fn
    call_count = [0]

    def counting_load():
        call_count[0] += 1
        return ("M", "P", {})

    m1.load("my-model", counting_load)
    assert call_count[0] == 0

    # unload + reload should re-run load
    assert m1.unload("my-model") is True
    m1.load("my-model", counting_load)
    assert call_count[0] == 1

    m1.reset()


def test_model_manager_failure_cooldown():
    from app.services.model_manager import ModelManager, ModelLoadingError, get_model_manager

    mgr = get_model_manager()
    mgr.reset()
    _failure_counter = [0]

    def always_fails():
        _failure_counter[0] += 1
        raise ModelLoadingError("boom")

    with pytest.raises(ModelLoadingError):
        mgr.load("flaky-model", always_fails)
    assert _failure_counter[0] == 1
    # Within the cooldown, subsequent calls immediately raise without re-invoking load_fn.
    with pytest.raises(ModelLoadingError):
        mgr.load("flaky-model", always_fails)
    assert _failure_counter[0] == 1
    # force_reload bypasses cooldown
    with pytest.raises(ModelLoadingError):
        mgr.load("flaky-model", always_fails, force_reload=True)
    assert _failure_counter[0] == 2
    mgr.reset()


# ---------------------------------------------------------------------------
# 7. Inference failure handling (via execute_plan mock injection)
# ---------------------------------------------------------------------------


def test_execute_plan_handles_tool_exception_gracefully(db):
    """execute_plan should catch per-tool exceptions and convert them into an
    answer/evidence payload rather than bubbling up."""
    from app.services.orchestrator import execute_plan
    from app.services import mock_specialists

    original_run = mock_specialists.run_tool

    def _explosive(tool_id, query, mode, **kwargs):
        raise RuntimeError(f"oops in {tool_id}")

    try:
        mock_specialists.run_tool = _explosive
        ans, conf, invs, boxes, ev, change_map, exec_modes, change_stats = execute_plan(
            query="any question",
            mode="single_image",
            tool_ids=["rs_caption", "rs_grounding"],
            per_tool_params={"rs_caption": {}, "rs_grounding": {}},
            tasks=["captioning", "grounding"],
            image_file_paths=[],
        )
    finally:
        mock_specialists.run_tool = original_run

    # No exceptions bubbled
    assert len(invs) == 2
    for inv in invs:
        assert inv.executionMode == "mock"
        assert inv.processingTimeMs is not None
    # Answer should contain error messages
    assert "ERROR" in ans or "raised" in ans
    # Evidence should mention failures
    joined_ev = " ".join(ev)
    assert "failed" in joined_ev.lower()
    # Exec modes dict populated (rs_caption + rs_grounding both mock)
    assert "rs_caption" in exec_modes and exec_modes["rs_caption"] == "mock"


# ---------------------------------------------------------------------------
# 8. Execution trace — 8 canonical steps, statuses all done on success
# ---------------------------------------------------------------------------


def test_execution_trace_has_8_steps_and_exec_modes(client, png_file):
    fid = _upload_image(client, png_file, "sample_2024-01-01.png", "single")
    resp = client.post(
        "/api/analysis",
        json={
            "mode": "single_image",
            "imageIds": [fid],
            "query": "What is the weather?",
        },
    )
    aid = resp.json()["analysisId"]
    t = client.get(f"/api/analysis/{aid}/trace").json()
    assert len(t["steps"]) == 8
    titles = [s["title"] for s in t["steps"]]
    expected_sequence = [
        "Query Received",
        "Input Validation",
        "Task Classification",
        "Tool Selection",
        "Parameters",
        "Processing",
        "Aggregation",
        "Completion",
    ]
    assert titles == expected_sequence
    for step in t["steps"]:
        assert step["status"] == "done", f"Step {step['title']} is {step['status']}"
    assert t["overallStatus"] == "completed"
    assert isinstance(t["totalElapsedMs"], int) and t["totalElapsedMs"] > 0

    # Tool-selection step should mention REAL or MOCK for each tool.
    toolsel = t["steps"][3]
    assert ("[MOCK]" in toolsel["detail"]) or ("[REAL]" in toolsel["detail"])

    # Processing step detail should include per-tool execution labels
    proc = t["steps"][5]
    assert "[MOCK]" in proc["detail"] or "[REAL]" in proc["detail"]

    # Result-level: rs_vqa ToolInvocation.executionMode present
    r = client.get(f"/api/analysis/{aid}").json()
    inv_modes = {inv["toolId"]: inv["executionMode"] for inv in r["toolInvocations"]}
    assert "rs_vqa" in inv_modes
    # With VQA_MODE=mock in tests, executionMode must be mock.
    assert inv_modes["rs_vqa"] == "mock"


# ---------------------------------------------------------------------------
# 9. Result validation — anti-fabrication guardrails
# ---------------------------------------------------------------------------


def test_vqa_output_validation_does_not_fabricate_confidence():
    from app.services.result_validation import validate_vqa_output
    from app.services.vqa_adapter import VQAInferenceOutput

    # Model reports NO confidence → validator keeps it as None (doesn't invent one)
    out = VQAInferenceOutput(
        answer_text="  A reasonable answer containing multiple words.  ",
        confidence=None,
    )
    res = validate_vqa_output(out)
    assert res.valid is True
    assert "Confidence not produced" in " ".join(res.warnings) or "confidence" in " ".join(res.warnings).lower()
    assert out.confidence is None

    # Out-of-range confidence is discarded to null (not clamped)
    out2 = VQAInferenceOutput(answer_text="foo bar baz", confidence=1.5)
    res2 = validate_vqa_output(out2)
    assert out2.confidence is None
    assert any("outside" in w for w in res2.warnings)

    # Too-short answer is INVALID
    out3 = VQAInferenceOutput(answer_text=" ", confidence=None)
    res3 = validate_vqa_output(out3)
    assert res3.valid is False


def test_analysis_result_validation_strips_unsourced_bounding_boxes_for_real_runs():
    from app.services.result_validation import validate_analysis_result_payload

    # Simulate a real VQA run that spuriously includes boxes and a change map
    payload = {
        "status": "completed",
        "images": [{"role": "single"}],
        "selectedTools": ["rs_vqa"],
        "boundingBoxes": [{"x": 0.1, "y": 0.2, "w": 0.05, "h": 0.05, "label": "Building", "confidence": 0.9}],
        "changeMap": {"overlayUrl": None, "legend": []},
        "answerText": "answer",
        "confidence": 1.4,  # also test out-of-range confidence stripping
    }
    report = validate_analysis_result_payload(payload, is_mock=False)
    assert "boundingBoxes" in report.stripped_fields
    assert payload["boundingBoxes"] is None
    assert "changeMap" in report.stripped_fields
    assert payload["changeMap"] is None
    assert "confidence" in report.stripped_fields
    assert payload["confidence"] is None

    # Mock runs: existing payload preserved for Phase 1 compatibility.
    payload2 = {
        "status": "completed",
        "images": [{"role": "single"}],
        "selectedTools": ["rs_grounding"],
        "boundingBoxes": [{"x": 0.1, "y": 0.2, "w": 0.05, "h": 0.05, "label": "Building", "confidence": 0.9}],
        "answerText": "mock answer",
    }
    report2 = validate_analysis_result_payload(payload2, is_mock=True)
    # rs_grounding is a grounding-capable tool even in mock → boxes not stripped
    assert "boundingBoxes" not in report2.stripped_fields
    assert payload2["boundingBoxes"] is not None


# ---------------------------------------------------------------------------
# 10. Preprocessing service — imagery in → PIL RGB out
# ---------------------------------------------------------------------------


def test_preprocessing_png_via_pillow(tmp_path):
    from PIL import Image
    from app.services.preprocessing import preprocess_imagery_for_vqa, PreprocessingResult

    img_path = tmp_path / "test_optical_2024-01-01.png"
    Image.new("RGB", (100, 80), color=(50, 100, 150)).save(img_path, format="PNG")

    res = preprocess_imagery_for_vqa(img_path)
    assert isinstance(res, PreprocessingResult)
    assert res.rgb_image.mode == "RGB"
    assert isinstance(res.preprocessing_meta, dict)
    assert res.preprocessing_meta["backend"] == "pillow"
    # Percent stretch applied, shape set
    assert "output_shape" in res.preprocessing_meta


def test_preprocessing_rejects_missing_file(tmp_path):
    from app.services.preprocessing import ImageryPreprocessingError, preprocess_imagery_for_vqa

    with pytest.raises(ImageryPreprocessingError):
        preprocess_imagery_for_vqa(tmp_path / "does_not_exist.png")


def test_preprocessing_rejects_empty_file(tmp_path):
    from app.services.preprocessing import ImageryPreprocessingError, preprocess_imagery_for_vqa

    p = tmp_path / "empty.png"
    p.write_bytes(b"")
    with pytest.raises(ImageryPreprocessingError):
        preprocess_imagery_for_vqa(p)


# ---------------------------------------------------------------------------
# 11. Phase 1 compatibility: existing bi-temporal + optical_sar flows still work
# ---------------------------------------------------------------------------


def test_bi_temporal_submission_still_works(client, png_file):
    f1 = _upload_image(client, png_file, "CARTOSAT-3_PAN_2022-01-10_T1.png", "before")
    f2 = _upload_image(client, png_file, "CARTOSAT-3_PAN_2024-01-08_T2.png", "after")
    resp = client.post(
        "/api/analysis",
        json={
            "mode": "bi_temporal",
            "imageIds": [f1, f2],
            "query": "What changes occurred? Has urban expansion affected vegetation?",
        },
    )
    assert resp.status_code == 200
    aid = resp.json()["analysisId"]
    r = client.get(f"/api/analysis/{aid}").json()
    assert r["status"] == "completed"
    assert "change_detection" in r["detectedTasks"] or "change_vqa" in r["detectedTasks"]
    # executionMode on all tools must still be populated (they're mock tools)
    for inv in r["toolInvocations"]:
        assert inv["executionMode"] in ("real", "mock")


def test_optical_sar_submission_still_works(client, png_file):
    f1 = _upload_image(client, png_file, "optical_sample_2024-02-20.png", "optical")
    f2 = _upload_image(client, png_file, "RISAT-1A_SAR_C-band_VV_2024-02-22.png", "sar")
    resp = client.post(
        "/api/analysis",
        json={
            "mode": "optical_sar",
            "imageIds": [f1, f2],
            "query": "Does SAR confirm optical detections? Find SAR-only features.",
        },
    )
    assert resp.status_code == 200
    aid = resp.json()["analysisId"]
    r = client.get(f"/api/analysis/{aid}").json()
    assert r["status"] == "completed"
    assert "optical_sar_analyzer" in r["selectedTools"]


# ---------------------------------------------------------------------------
# 12. Root endpoint reports Phase 2 info
# ---------------------------------------------------------------------------


def test_root_endpoint_reports_phase2(client):
    r = client.get("/").json()
    assert r["phase"] in (2, 3)
    assert "rs_vqa" in r["real_tools"]
    assert "rs_grounding" in r["mock_tools"]
    assert "vqa_mode" in r
    assert "vqa_model_id" in r
