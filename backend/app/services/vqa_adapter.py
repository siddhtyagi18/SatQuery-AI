from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

from PIL import Image

from ..config import get_settings
from ..logging_setup import logger
from .model_manager import ModelLoadingError, InferenceRuntimeError, get_model_manager

settings = get_settings()


# ---------------------------------------------------------------------------
# Data contract shared by all VQA model adapters
# ---------------------------------------------------------------------------

@dataclass
class VQAInferenceInput:
    """Normalized VQA input delivered to a VQAModelAdapter.

    All adapter implementations receive this structured input — they do not
    talk directly to the filesystem, rasterio, or the preprocessor. That
    separation keeps model adapters self-contained and swappable.
    """
    rgb_image: Image.Image
    query_text: str
    max_new_tokens: int = 512
    temperature: float = 0.2


@dataclass
class VQAInferenceOutput:
    """Structured output of a VQAModelAdapter.infer() call.

    IMPORTANT — these fields reflect ONLY what the model actually emits:
      - `answer_text` is always present (the generated string)
      - `confidence` is None unless the adapter can extract a calibrated,
        model-native confidence signal. We **never fabricate** a score.
      - `raw_output` is optional debug info, not part of the API contract.
    """
    answer_text: str
    confidence: Optional[float] = None
    model_id: Optional[str] = None
    inference_meta: Optional[Dict[str, Any]] = None


class VQAModelAdapter(ABC):
    """Abstract base for VQA / VLM model adapters.

    Architectural contract (VQAService → VQAModelAdapter → Actual model):
      1. `load()` returns a (model, processor, metadata) tuple once.
      2. `preprocess_input()` converts a VQAInferenceInput into model-specific
         tensors/dicts — this is the **adapter's chance** to apply any
         model-specific transforms (prompt templates, tokenization, image
         featurization) before the actual forward pass.
      3. `infer()` runs the model on the preprocessed input and returns a
         clean VQAInferenceOutput.
      4. `supports_model()` declares which HuggingFace/registry ids this
         adapter can drive — used by `get_adapter_for_model()`.
    """

    @classmethod
    @abstractmethod
    def supports_model(cls, model_id: str) -> bool:
        """Return True if this adapter knows how to drive `model_id`."""

    @abstractmethod
    def load(self):
        """Load (model_obj, processor_obj, metadata_dict) from the model id.

        The returned triple is cached by ModelManager. Raises
        `ModelLoadingError` on any failure.
        """

    @abstractmethod
    def preprocess_input(
        self,
        inference_input: VQAInferenceInput,
        loaded,
    ) -> Any:
        """Convert VQAInferenceInput into model-ready tensors/inputs.

        `loaded` is the `LoadedModel` entry from ModelManager.
        """

    @abstractmethod
    def infer(
        self,
        preprocessed_inputs: Any,
        loaded,
        inference_input: VQAInferenceInput,
    ) -> VQAInferenceOutput:
        """Run inference on preprocessed inputs; return structured output."""


# ---------------------------------------------------------------------------
# Concrete adapter: HuggingFace Idefics3/SmolVLM-family (AutoModelForVision2Seq)
# ---------------------------------------------------------------------------

class SmolVLMHuggingFaceAdapter(VQAModelAdapter):
    """Adapter for HuggingFace TB SmolVLM-500M-Instruct and similar
    `AutoModelForVision2Seq` VLMs that expose the Idefics3 chat template.
    """

    _SUPPORTED_PREFIXES = (
        "HuggingFaceTB/SmolVLM",
        "HuggingFaceM4/Idefics3",
    )

    @classmethod
    def supports_model(cls, model_id: str) -> bool:
        mid = (model_id or "").strip()
        return any(mid.startswith(p) for p in cls._SUPPORTED_PREFIXES)

    # -- loading ------------------------------------------------------------
    def load(self):
        try:
            import torch
            from transformers import (
                AutoProcessor,
                AutoModelForVision2Seq,
            )
        except Exception as e:
            raise ModelLoadingError(
                f"transformers/torch package missing or broken: {e}. "
                f"Install with: pip install transformers>=4.49.0 torch>=2.4.0"
            ) from e

        model_id = settings.VQA_MODEL_ID
        hf_token = settings.VQA_HF_TOKEN or os.environ.get("HF_TOKEN")
        cache_dir = str(settings.vqa_cache_dir_path) if settings.vqa_cache_dir_path else None
        device_str = settings.VQA_DEVICE.lower()

        if device_str == "cuda" and not torch.cuda.is_available():
            logger.warning(
                f"[VQAAdapter] VQA_DEVICE='cuda' requested but torch reports no CUDA; "
                f"falling back to CPU."
            )
            device_str = "cpu"

        precision_str = settings.VQA_PRECISION.lower()
        if device_str == "cpu" and precision_str in ("fp16", "bf16"):
            logger.warning(
                f"[VQAAdapter] precision {precision_str} on CPU is not supported; "
                f"falling back to fp32."
            )
            precision_str = "fp32"

        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(precision_str, torch.float32)

        load_meta: Dict[str, Any] = {
            "model_id": model_id,
            "device": device_str,
            "precision": precision_str,
            "hf_token_provided": bool(hf_token),
            "custom_cache_dir": cache_dir is not None,
        }

        try:
            logger.info(f"[VQAAdapter] Downloading/loading processor for {model_id} ...")
            processor = AutoProcessor.from_pretrained(
                model_id,
                token=hf_token,
                cache_dir=cache_dir,
            )
            logger.info(f"[VQAAdapter] Downloading/loading model weights for {model_id} (this may take a while on first run) ...")
            model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                token=hf_token,
                cache_dir=cache_dir,
                torch_dtype=torch_dtype,
            )
            model.to(device_str)
            model.eval()
            load_meta["num_params_millions"] = round(
                sum(p.numel() for p in model.parameters()) / 1_000_000, 1
            )
            load_meta["device_actual"] = str(model.device)
        except Exception as e:
            raise ModelLoadingError(
                f"Failed to load SmolVLM-style model '{model_id}': {e}"
            ) from e

        return model, processor, load_meta

    # -- preprocessing ------------------------------------------------------
    def preprocess_input(self, inference_input: VQAInferenceInput, loaded) -> Any:
        processor = loaded.processor_object
        model = loaded.model_object
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": inference_input.query_text.strip()},
                ],
            }
        ]
        try:
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        except Exception as e:
            raise InferenceRuntimeError(
                f"Chat template application failed: {e}"
            ) from e

        try:
            inputs = processor(
                text=prompt,
                images=[inference_input.rgb_image],
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)
            return inputs
        except Exception as e:
            raise InferenceRuntimeError(
                f"Processor tokenization/featurization failed: {e}"
            ) from e

    # -- inference ----------------------------------------------------------
    def infer(
        self,
        preprocessed_inputs: Any,
        loaded,
        inference_input: VQAInferenceInput,
    ) -> VQAInferenceOutput:
        processor = loaded.processor_object
        model = loaded.model_object

        import torch
        generate_kwargs: Dict[str, Any] = {
            "max_new_tokens": max(1, min(inference_input.max_new_tokens, 4096)),
            "temperature": max(0.0, min(inference_input.temperature, 2.0)),
        }
        if generate_kwargs["temperature"] < 1e-3:
            generate_kwargs["do_sample"] = False
            generate_kwargs.pop("temperature", None)
        else:
            generate_kwargs["do_sample"] = True

        meta: Dict[str, Any] = {"generation_kwargs": generate_kwargs.copy()}
        input_len = preprocessed_inputs.get("input_ids", torch.tensor([])).shape[-1]

        try:
            with torch.inference_mode():
                generated_ids = model.generate(**preprocessed_inputs, **generate_kwargs)
        except Exception as e:
            raise InferenceRuntimeError(f"Model.generate() raised: {e}") from e

        try:
            new_tokens_only = generated_ids[:, input_len:]
            answer_text = processor.batch_decode(
                new_tokens_only, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]
        except Exception as e:
            raise InferenceRuntimeError(f"Token decoding failed: {e}") from e

        answer_text = (answer_text or "").strip()
        meta["generated_token_count"] = int(new_tokens_only.shape[-1])

        return VQAInferenceOutput(
            answer_text=answer_text,
            confidence=None,
            model_id=loaded.model_id,
            inference_meta=meta,
        )


# ---------------------------------------------------------------------------
# Adapter registry / factory
# ---------------------------------------------------------------------------

_REGISTERED_ADAPTERS = [SmolVLMHuggingFaceAdapter]


def register_vqa_adapter(cls) -> None:
    """Register an additional VQAModelAdapter subclass at runtime."""
    if cls not in _REGISTERED_ADAPTERS:
        _REGISTERED_ADAPTERS.append(cls)


def get_adapter_for_model(model_id: str) -> VQAModelAdapter:
    """Return a concrete VQAModelAdapter that supports `model_id`.

    Raises `ModelLoadingError` if no registered adapter claims the model id.
    """
    for cls in _REGISTERED_ADAPTERS:
        if cls.supports_model(model_id):
            return cls()
    raise ModelLoadingError(
        f"No registered VQAModelAdapter supports model_id='{model_id}'. "
        f"Registered adapters claim: "
        + ", ".join(
            getattr(c, "_SUPPORTED_PREFIXES", (c.__name__,))[0]
            for c in _REGISTERED_ADAPTERS
        )
    )
