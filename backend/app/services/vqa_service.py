from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..config import get_settings
from ..logging_setup import logger
from .ai_provider import get_ai_gateway, AIProviderError
from .model_manager import InferenceRuntimeError, ModelLoadingError, get_model_manager
from .preprocessing import ImageryPreprocessingError, preprocess_imagery_for_vqa
from .vqa_adapter import (
    VQAInferenceInput,
    VQAInferenceOutput,
    get_adapter_for_model,
)

settings = get_settings()


@dataclass
class VQARunContext:
    """Detailed per-run info surfaced to the orchestrator for traces/tool invocations."""

    execution_mode: str
    model_id: Optional[str] = None
    preprocessing_meta: Optional[Dict[str, Any]] = None
    model_load_meta: Optional[Dict[str, Any]] = None
    inference_meta: Optional[Dict[str, Any]] = None
    evidence: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    total_time_ms: int = 0


@dataclass
class VQAServiceResult:
    """The unified return type from the VQA service, used by the orchestrator.

    This intentionally mirrors the mock specialist dict keys so the existing
    `execute_plan` aggregation logic works without changes.
    """

    answer: str
    confidence: Optional[float]
    evidence: List[str]
    tool_id: str = "rs_vqa"
    is_mock: bool = False
    bounding_boxes: Optional[List[Any]] = None
    run_context: Optional[VQARunContext] = None


def _fabricated_confidence_error() -> None:
    """Guard: the VQA service MUST NOT fabricate a confidence value."""
    return None


class VQAService:
    """Top-level VQA service entry point.

    Lifecycle for a single call:
      1. Check settings + availability → decide real vs. mock
      2. (real) Run the dedicated imagery preprocessor
      3. (real) Obtain a loaded model via ModelManager (first-call = download+init,
         subsequent calls = cached)
      4. (real) Delegate to the VQAModelAdapter for preprocessing + inference
      5. (real) Validate the adapter's output with ResultValidator
      6. Wrap everything into a VQAServiceResult + VQARunContext for the trace.

    On ANY real-path failure, the service falls back to the mock answer UNLESS
    the caller has explicitly set VQA_MODE="real". This keeps Phase 1 behaviour
    intact for environments where transformers/torch aren't installed.
    """

    TOOL_ID = "rs_vqa"

    def __init__(self) -> None:
        self._manager = get_model_manager()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def should_use_real_vqa(self, mode: str, tasks: Optional[List[str]] = None) -> bool:
        """Decide whether to run real inference or fall back to mock.

        Rules:
          - VQA_MODE=mock  → never real
          - VQA_MODE=real  → always attempt real (fallback on error if not forced)
          - VQA_MODE=auto  → real only for single_image + task list includes "vqa"
        """
        mode_setting = (settings.VQA_MODE or "auto").lower()
        if mode_setting == "mock":
            return False
        if mode_setting == "real":
            return True
        tasks = tasks or []
        return mode == "single_image" and ("vqa" in tasks or "captioning" in tasks)

    def run_real_or_fallback(
        self,
        query: str,
        mode: str,
        image_file_paths: List[Path],
        tasks: Optional[List[str]] = None,
        mock_factory: Optional[Callable[[], VQAServiceResult]] = None,
        tool_id: str = "rs_vqa",
        preferred_provider: Optional[str] = None,
    ) -> VQAServiceResult:
        """Main entry point used by the orchestrator.

        image_file_paths: one or more local file paths. For single-image VQA
            we use the first image; future multi-image adapters can consume more.
        mock_factory: zero-arg callable returning a mock VQAServiceResult.
            If None, a minimal mock answer is synthesized.
        preferred_provider: 'local' (SmolVLM+LoRA), 'gemini', 'openrouter', or 'auto'.
        """
        t0 = time.perf_counter()
        use_real = self.should_use_real_vqa(mode, tasks)
        if not use_real:
            res = self._mock_result(query, mode, reason="VQA_MODE setting disabled real inference", mock_factory=mock_factory, tool_id=tool_id)
            res.run_context.total_time_ms = int((time.perf_counter() - t0) * 1000)
            return res

        if not image_file_paths:
            res = self._mock_result(query, mode, reason="No input images available for real VQA", mock_factory=mock_factory, tool_id=tool_id)
            res.run_context.total_time_ms = int((time.perf_counter() - t0) * 1000)
            return res

        try:
            return self._run_real_pipeline(
                query, mode, image_file_paths, t0,
                tasks=tasks, tool_id=tool_id, preferred_provider=preferred_provider,
            )
        except Exception as e:
            logger.exception(f"Real VQA pipeline failed; falling back to mock: {e}")
            force_real = (settings.VQA_MODE or "auto").lower() == "real"
            if force_real:
                raise
            res = self._mock_result(
                query,
                mode,
                reason=f"[MOCK FALLBACK] Real VQA failed ({type(e).__name__}: {e}); mock fallback used.",
                mock_factory=mock_factory,
                tool_id=tool_id,
            )
            res.run_context.errors.append(f"Real pipeline failure: {type(e).__name__}: {e}")
            res.run_context.total_time_ms = int((time.perf_counter() - t0) * 1000)
            return res

    # ------------------------------------------------------------------
    # Real pipeline dispatcher
    # ------------------------------------------------------------------
    def _run_real_pipeline(
        self,
        query: str,
        mode: str,
        image_file_paths: List[Path],
        t0: float,
        tasks: Optional[List[str]] = None,
        tool_id: str = "rs_vqa",
        preferred_provider: Optional[str] = None,
    ) -> VQAServiceResult:
        ctx = VQARunContext(execution_mode="real")

        # --- Preprocessing -------------------------------------------------
        preproc_path = image_file_paths[0]
        try:
            preproc = preprocess_imagery_for_vqa(preproc_path)
            ctx.preprocessing_meta = preproc.preprocessing_meta
            ctx.evidence.append(
                f"Preprocessed {preproc_path.name} via {preproc.preprocessing_meta.get('backend','?')} backend; "
                f"output shape={preproc.preprocessing_meta.get('output_shape')}."
            )
        except ImageryPreprocessingError as e:
            raise ImageryPreprocessingError(str(e)) from e

        provider_pref = (preferred_provider or settings.AI_PROVIDER or "auto").lower()

        # 1. Explicit local VLM requested (SmolVLM-500M + domain-adapted LoRA)
        if provider_pref in ("local", "lora", "smolvlm"):
            logger.info("[VQAService] Provider explicitly set to 'local' -> executing local SmolVLM+LoRA...")
            return self._execute_local_vlm(
                query=query, preproc=preproc, preproc_path=preproc_path,
                ctx=ctx, tasks=tasks, tool_id=tool_id, t0=t0,
            )

        # 2. Explicit cloud provider requested (Gemini or OpenRouter)
        if provider_pref in ("gemini", "openrouter", "cloud"):
            logger.info(f"[VQAService] Provider explicitly set to '{provider_pref}' -> executing Cloud AI Gateway...")
            pref = provider_pref if provider_pref in ("gemini", "openrouter") else None
            return self._execute_cloud_gateway(
                query=query, preproc_path=preproc_path, ctx=ctx,
                tasks=tasks, tool_id=tool_id, t0=t0, preferred_cloud_provider=pref,
            )

        # 3. 'auto' mode:
        # Check if any cloud provider credentials are configured.
        gateway = get_ai_gateway()
        has_cloud = (
            gateway.get_provider("openrouter").is_available()
            or gateway.get_provider("gemini").is_available()
        )

        if has_cloud:
            try:
                logger.info("[VQAService] Auto-routing: cloud credentials found -> attempting Cloud AI Gateway...")
                return self._execute_cloud_gateway(
                    query=query, preproc_path=preproc_path, ctx=ctx,
                    tasks=tasks, tool_id=tool_id, t0=t0,
                )
            except Exception as cloud_err:
                logger.warning(
                    f"[VQAService] Cloud AI Gateway failed ({cloud_err}); attempting local VLM fallback..."
                )
                ctx.evidence.append(f"[fallback] Cloud AI Gateway failed ({cloud_err}); fell back to local VLM.")

        # Fall back to local SmolVLM + LoRA execution
        logger.info("[VQAService] Executing local VLM (SmolVLM+LoRA)...")
        return self._execute_local_vlm(
            query=query, preproc=preproc, preproc_path=preproc_path,
            ctx=ctx, tasks=tasks, tool_id=tool_id, t0=t0,
        )

    # ------------------------------------------------------------------
    # Local VLM (SmolVLM-500M + LoRA) Execution
    # ------------------------------------------------------------------
    def _execute_local_vlm(
        self,
        query: str,
        preproc,
        preproc_path: Path,
        ctx: VQARunContext,
        tasks: Optional[List[str]],
        tool_id: str,
        t0: float,
    ) -> VQAServiceResult:
        model_id = settings.VQA_MODEL_ID
        try:
            adapter = get_adapter_for_model(model_id)
        except ModelLoadingError:
            raise

        load_start = time.perf_counter()
        loaded = self._manager.load(model_id, adapter.load)
        ctx.model_load_meta = dict(loaded.metadata)
        ctx.model_load_meta["cache_hit"] = loaded.age_sec > (time.perf_counter() - load_start)
        ctx.model_load_meta["load_duration_sec"] = loaded.load_duration_sec

        is_lora = ctx.model_load_meta.get("lora_adapted", False)
        lora_ckpt = ctx.model_load_meta.get("lora_checkpoint")

        ctx.model_id = f"local:{model_id}"
        ctx.evidence.append(
            f"Provider: local (SmolVLM {'with domain-adapted LoRA' if is_lora else 'base model'})."
        )
        ctx.evidence.append(
            f"Model '{model_id}' ready ({loaded.load_duration_sec:.1f}s load; "
            f"params={loaded.metadata.get('num_params_millions','?')}M, device={loaded.metadata.get('device_actual','?')})."
        )
        if is_lora:
            ctx.evidence.append(f"PEFT LoRA domain adaptation active from checkpoint: {lora_ckpt} (trained on BigEarthNet).")
        else:
            ctx.evidence.append("Base vision-language model active (no LoRA adapter attached).")

        # --- Domain-adapted prompt conditioning for captioning -------------
        effective_query = query.strip()
        if tool_id == "rs_caption" or ("captioning" in (tasks or []) and "vqa" not in (tasks or [])):
            effective_query = (
                f"Provide a detailed, comprehensive remote-sensing scene description for this satellite image, "
                f"identifying dominant land-cover categories, terrain morphology, infrastructure, and visible objects. "
                f"User request: {query.strip()}"
            )

        inf_input = VQAInferenceInput(
            rgb_image=preproc.rgb_image,
            query_text=effective_query,
            max_new_tokens=settings.VQA_MAX_NEW_TOKENS,
            temperature=settings.VQA_TEMPERATURE,
        )
        try:
            model_inputs = adapter.preprocess_input(inf_input, loaded)
        except InferenceRuntimeError as e:
            raise InferenceRuntimeError(f"adapter.preprocess_input: {e}") from e

        # --- Inference ----------------------------------------------------
        try:
            inf_output: VQAInferenceOutput = adapter.infer(
                model_inputs, loaded, inf_input
            )
        except InferenceRuntimeError as e:
            raise InferenceRuntimeError(f"adapter.infer: {e}") from e

        ctx.inference_meta = inf_output.inference_meta or {}
        ctx.inference_meta["provider"] = "local"
        ctx.inference_meta["model"] = model_id
        ctx.inference_meta["lora_adapted"] = is_lora
        ctx.inference_meta["lora_checkpoint"] = lora_ckpt

        # Confidence: never fabricate confidence
        ctx.evidence.append("Model does not emit a calibrated confidence score; confidence=null.")

        # --- Result validation --------------------------------------------
        from .result_validation import validate_vqa_output
        validation = validate_vqa_output(inf_output)
        if validation.warnings:
            for w in validation.warnings:
                ctx.evidence.append(f"[validation] {w}")
        if not validation.valid:
            raise InferenceRuntimeError(
                "VQA output failed validation: " + "; ".join(validation.warnings)
            )

        answer = validation.cleaned_answer or inf_output.answer_text
        ctx.evidence.append(
            f"Local inference completed; generated ~{ctx.inference_meta.get('generated_token_count','?')} tokens."
        )
        ctx.total_time_ms = int((time.perf_counter() - t0) * 1000)

        return VQAServiceResult(
            answer=answer,
            confidence=None,  # Do not fabricate confidence
            evidence=list(ctx.evidence),
            tool_id=tool_id,
            is_mock=False,
            run_context=ctx,
        )

    # ------------------------------------------------------------------
    # Cloud AI Gateway (Gemini / OpenRouter) Execution
    # ------------------------------------------------------------------
    def _execute_cloud_gateway(
        self,
        query: str,
        preproc_path: Path,
        ctx: VQARunContext,
        tasks: Optional[List[str]],
        tool_id: str,
        t0: float,
        preferred_cloud_provider: Optional[str] = None,
    ) -> VQAServiceResult:
        gateway = get_ai_gateway()

        effective_query = query.strip()
        if tool_id == "rs_caption" or ("captioning" in (tasks or []) and "vqa" not in (tasks or [])):
            effective_query = (
                f"Provide a detailed, comprehensive remote-sensing scene description for this satellite image, "
                f"identifying dominant land-cover categories, terrain morphology, infrastructure, and visible objects. "
                f"User request: {query.strip()}"
            )

        gateway_res = gateway.generate(
            prompt=effective_query,
            image_path=preproc_path,
            preferred_provider=preferred_cloud_provider,
            max_tokens=settings.VQA_MAX_NEW_TOKENS,
            temperature=settings.VQA_TEMPERATURE,
        )

        answer = gateway_res.get("answer", "").strip()
        provider_name = gateway_res.get("provider", "cloud")
        model_name = gateway_res.get("model", "unknown")

        ctx.model_id = f"{provider_name}:{model_name}"
        ctx.inference_meta = gateway_res.get("raw_meta") or {}
        ctx.inference_meta["provider"] = provider_name
        ctx.inference_meta["model"] = model_name
        ctx.inference_meta["lora_adapted"] = False

        ctx.evidence.append(f"Provider: {provider_name} (Cloud AI Gateway).")
        ctx.evidence.append(f"Model: {model_name}.")
        ctx.evidence.extend(gateway_res.get("evidence", []))
        ctx.evidence.append("Model does not emit a calibrated confidence score; confidence=null.")
        ctx.total_time_ms = int((time.perf_counter() - t0) * 1000)

        return VQAServiceResult(
            answer=answer,
            confidence=None,  # Do not fabricate confidence
            evidence=list(ctx.evidence),
            tool_id=tool_id,
            is_mock=False,
            run_context=ctx,
        )

    # ------------------------------------------------------------------
    # Mock fallback
    # ------------------------------------------------------------------
    def _mock_result(
        self,
        query: str,
        mode: str,
        reason: str,
        mock_factory: Optional[Callable[[], VQAServiceResult]] = None,
        tool_id: str = "rs_vqa",
    ) -> VQAServiceResult:
        if mock_factory is not None:
            res = mock_factory()
            if res.run_context is None:
                res.run_context = VQARunContext(execution_mode="mock")
            res.run_context.execution_mode = "mock"
            res.run_context.evidence.append(reason)
            res.tool_id = tool_id
            return res
        from .mock_specialists import _make_vqa_result
        raw = _make_vqa_result(query, mode)
        ctx = VQARunContext(execution_mode="mock")
        ctx.evidence.append(reason)
        ctx.evidence.extend(raw.get("evidence", []))
        return VQAServiceResult(
            answer=raw["answer"],
            confidence=raw.get("confidence"),
            evidence=list(ctx.evidence),
            tool_id=tool_id,
            is_mock=True,
            run_context=ctx,
        )


_vqa_service_singleton: Optional[VQAService] = None


def get_vqa_service() -> VQAService:
    global _vqa_service_singleton
    if _vqa_service_singleton is None:
        _vqa_service_singleton = VQAService()
    return _vqa_service_singleton
