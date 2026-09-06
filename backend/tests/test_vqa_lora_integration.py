"""
backend/tests/test_vqa_lora_integration.py
==========================================
Verification suite for SatQuery-AI VLM LoRA integration:
1. Checkpoint discovery & existence of trained Experiment 01 weights
2. Observable provider selection: 'local' (SmolVLM+LoRA), 'gemini', 'openrouter', 'auto'
3. Remote sensing scene description / captioning prompt specialization
4. Strict non-fabrication of confidence (confidence=None)
5. Clear, observable mock fallback flagging
6. Tool ID preservation across VQA and captioning
7. Trace and metadata integrity
"""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.config import get_settings
from app.services.vqa_adapter import (
    resolve_vqa_lora_checkpoint_path,
    VQAInferenceInput,
    VQAInferenceOutput,
)
from app.services.vqa_service import VQAService, VQAServiceResult, VQARunContext
from app.services.orchestrator import plan_execution, execute_plan
from app.schemas import AnalysisMode

settings = get_settings()
_BACKEND_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_satellite_image(tmp_path: Path) -> Path:
    """Create a temporary dummy RGB satellite image for unit tests."""
    img_path = tmp_path / "test_satellite_tile.png"
    img = Image.new("RGB", (120, 120), color=(50, 110, 70))
    img.save(img_path, format="PNG")
    return img_path


# ---------------------------------------------------------------------------
# 1. Checkpoint Discovery & Integrity
# ---------------------------------------------------------------------------

def test_resolve_vqa_lora_checkpoint_path_finds_experiment_01():
    """Verify that the repository-relative resolver locates the trained LoRA checkpoint."""
    ckpt_path = resolve_vqa_lora_checkpoint_path()
    assert ckpt_path is not None, "Failed to resolve trained VLM LoRA checkpoint"
    assert ckpt_path.exists(), f"Resolved path does not exist: {ckpt_path}"
    assert ckpt_path.is_dir(), f"Resolved path is not a directory: {ckpt_path}"

    # Check for weights and configuration
    safetensors = ckpt_path / "adapter_model.safetensors"
    config_json = ckpt_path / "adapter_config.json"
    assert safetensors.exists(), f"Missing adapter weights in {ckpt_path}"
    assert config_json.exists(), f"Missing adapter_config.json in {ckpt_path}"
    assert safetensors.stat().st_size > 1_000_000, "adapter_model.safetensors is too small"


# ---------------------------------------------------------------------------
# 2. Observable Provider Selection
# ---------------------------------------------------------------------------

def test_provider_selection_local_routes_to_local_vlm(sample_satellite_image: Path):
    """Verify that provider='local' invokes local SmolVLM without calling cloud gateway."""
    service = VQAService()

    dummy_loaded = MagicMock()
    dummy_loaded.model_id = "HuggingFaceTB/SmolVLM-500M-Instruct"
    dummy_loaded.age_sec = 10.0
    dummy_loaded.load_duration_sec = 0.5
    dummy_loaded.metadata = {
        "num_params_millions": 500.0,
        "device_actual": "cpu",
        "lora_adapted": True,
        "lora_checkpoint": str(_BACKEND_ROOT / "checkpoints" / "vqa_lora_experiment_01"),
    }

    dummy_adapter = MagicMock()
    dummy_adapter.preprocess_input.return_value = {"input_ids": []}
    dummy_adapter.infer.return_value = VQAInferenceOutput(
        answer_text="Water bodies and broad-leaved forest detected.",
        confidence=None,
        model_id="HuggingFaceTB/SmolVLM-500M-Instruct",
        inference_meta={"generated_token_count": 12},
    )

    with patch.object(service._manager, "load", return_value=dummy_loaded), \
         patch("app.services.vqa_service.get_adapter_for_model", return_value=dummy_adapter), \
         patch("app.services.vqa_service.settings.VQA_MODE", "real"):

        result = service.run_real_or_fallback(
            query="What land cover is visible?",
            mode="single_image",
            image_file_paths=[sample_satellite_image],
            tool_id="rs_vqa",
            preferred_provider="local",
        )

        assert not result.is_mock
        assert result.confidence is None
        assert "Water bodies" in result.answer
        assert any("Provider: local" in ev for ev in result.evidence)
        assert any("PEFT LoRA domain adaptation active" in ev for ev in result.evidence)
        assert any("confidence=null" in ev for ev in result.evidence)


def test_provider_selection_cloud_gemini_routes_to_gateway(sample_satellite_image: Path):
    """Verify that provider='gemini' delegates to Cloud AI Gateway."""
    service = VQAService()

    mock_gateway_res = {
        "answer": "Cloud analysis identifies agricultural fields.",
        "provider": "gemini",
        "model": "gemini-3.6-flash",
        "confidence": None,
        "evidence": ["Gemini cloud call"],
        "raw_meta": {"model": "gemini-3.6-flash"},
    }

    mock_gateway = MagicMock()
    mock_gateway.generate.return_value = mock_gateway_res

    with patch("app.services.vqa_service.get_ai_gateway", return_value=mock_gateway), \
         patch("app.services.vqa_service.settings.VQA_MODE", "real"):

        result = service.run_real_or_fallback(
            query="Describe this scene",
            mode="single_image",
            image_file_paths=[sample_satellite_image],
            tool_id="rs_caption",
            preferred_provider="gemini",
        )

        assert not result.is_mock
        assert result.tool_id == "rs_caption"
        assert result.confidence is None
        assert "Cloud analysis" in result.answer
        assert any("Provider: gemini" in ev for ev in result.evidence)
        mock_gateway.generate.assert_called_once()
        assert mock_gateway.generate.call_args.kwargs.get("preferred_provider") == "gemini"


def test_provider_selection_cloud_openrouter_routes_to_gateway(sample_satellite_image: Path):
    """Verify that provider='openrouter' delegates to OpenRouter through Cloud AI Gateway."""
    service = VQAService()

    mock_gateway_res = {
        "answer": "OpenRouter reasoning indicates dense urban area.",
        "provider": "openrouter",
        "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        "confidence": None,
        "evidence": ["OpenRouter reasoning trace"],
        "raw_meta": {"provider": "openrouter"},
    }

    mock_gateway = MagicMock()
    mock_gateway.generate.return_value = mock_gateway_res

    with patch("app.services.vqa_service.get_ai_gateway", return_value=mock_gateway), \
         patch("app.services.vqa_service.settings.VQA_MODE", "real"):

        result = service.run_real_or_fallback(
            query="What infrastructure is shown?",
            mode="single_image",
            image_file_paths=[sample_satellite_image],
            tool_id="rs_vqa",
            preferred_provider="openrouter",
        )

        assert not result.is_mock
        assert result.confidence is None
        assert "OpenRouter" in result.answer
        assert any("Provider: openrouter" in ev for ev in result.evidence)
        assert mock_gateway.generate.call_args.kwargs.get("preferred_provider") == "openrouter"


# ---------------------------------------------------------------------------
# 3. Captioning Specialization & Domain Prompt Conditioning
# ---------------------------------------------------------------------------

def test_captioning_prompt_specialization(sample_satellite_image: Path):
    """Verify that rs_caption conditions the prompt with comprehensive remote sensing instructions."""
    service = VQAService()

    captured_inputs = []

    dummy_loaded = MagicMock()
    dummy_loaded.metadata = {"num_params_millions": 500.0, "device_actual": "cpu", "lora_adapted": False}
    dummy_loaded.age_sec = 5.0
    dummy_loaded.load_duration_sec = 0.2

    dummy_adapter = MagicMock()
    def mock_preprocess(inf_input, loaded):
        captured_inputs.append(inf_input)
        return {"input_ids": []}
    dummy_adapter.preprocess_input = mock_preprocess
    dummy_adapter.infer.return_value = VQAInferenceOutput(
        answer_text="Detailed landscape overview.",
        confidence=None,
        model_id="SmolVLM",
    )

    with patch.object(service._manager, "load", return_value=dummy_loaded), \
         patch("app.services.vqa_service.get_adapter_for_model", return_value=dummy_adapter), \
         patch("app.services.vqa_service.settings.VQA_MODE", "real"):

        result = service.run_real_or_fallback(
            query="Give an overview of this image",
            mode="single_image",
            image_file_paths=[sample_satellite_image],
            tool_id="rs_caption",
            preferred_provider="local",
        )

        assert result.tool_id == "rs_caption"
        assert len(captured_inputs) == 1
        effective_query = captured_inputs[0].query_text
        assert "Provide a detailed, comprehensive remote-sensing scene description" in effective_query
        assert "land-cover categories" in effective_query


# ---------------------------------------------------------------------------
# 4. Strict Non-Fabrication of Confidence
# ---------------------------------------------------------------------------

def test_confidence_is_strictly_none_and_explained(sample_satellite_image: Path):
    """Verify that generative VQA and captioning never fabricate a confidence score."""
    service = VQAService()

    dummy_loaded = MagicMock()
    dummy_loaded.metadata = {"lora_adapted": True, "num_params_millions": 500}
    dummy_loaded.age_sec = 1.0
    dummy_loaded.load_duration_sec = 0.1

    dummy_adapter = MagicMock()
    dummy_adapter.preprocess_input.return_value = {}
    dummy_adapter.infer.return_value = VQAInferenceOutput(
        answer_text="Pastures and agricultural land.",
        confidence=None,
        model_id="SmolVLM",
    )

    with patch.object(service._manager, "load", return_value=dummy_loaded), \
         patch("app.services.vqa_service.get_adapter_for_model", return_value=dummy_adapter), \
         patch("app.services.vqa_service.settings.VQA_MODE", "real"):

        result = service.run_real_or_fallback(
            query="What is here?",
            mode="single_image",
            image_file_paths=[sample_satellite_image],
            tool_id="rs_vqa",
            preferred_provider="local",
        )

        assert result.confidence is None
        assert any("confidence=null" in ev for ev in result.evidence)


# ---------------------------------------------------------------------------
# 5. Observable Mock Fallback
# ---------------------------------------------------------------------------

def test_mock_fallback_is_explicitly_flagged_in_evidence(sample_satellite_image: Path):
    """Verify that when real inference fails in auto mode, mock fallback is clearly marked."""
    service = VQAService()

    with patch("app.services.vqa_service.preprocess_imagery_for_vqa", side_effect=RuntimeError("Simulated CUDA OOM")), \
         patch("app.services.vqa_service.settings.VQA_MODE", "auto"):

        result = service.run_real_or_fallback(
            query="What land cover?",
            mode="single_image",
            image_file_paths=[sample_satellite_image],
            tasks=["vqa"],
            tool_id="rs_vqa",
            preferred_provider="local",
        )

        assert result.is_mock is True
        assert any("[MOCK FALLBACK]" in ev for ev in result.evidence)
        assert any("Simulated CUDA OOM" in err for err in result.run_context.errors)


# ---------------------------------------------------------------------------
# 6. Orchestrator Integration with Provider Support
# ---------------------------------------------------------------------------

def test_orchestrator_passes_provider_to_tools():
    """Verify plan_execution attaches the configured provider to rs_vqa and rs_caption parameters."""
    with patch("app.services.orchestrator.settings.AI_PROVIDER", "local"):
        tasks, tools, params, scores = plan_execution("Summarize this satellite image", "single_image")
        assert "rs_caption" in tools or "rs_vqa" in tools
        for tid in ("rs_vqa", "rs_caption"):
            if tid in params:
                assert params[tid].get("provider") == "local"
