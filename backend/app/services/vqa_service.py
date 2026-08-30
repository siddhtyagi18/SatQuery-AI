from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..config import get_settings
from ..logging_setup import logger
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
    ) -> VQAServiceResult:
        """Main entry point used by the orchestrator.

        image_file_paths: one or more local file paths. For single-image VQA
            we use the first image; future multi-image adapters can consume more.
        mock_factory: zero-arg callable returning a mock VQAServiceResult.
            If None, a minimal mock answer is synthesized.
        """
        t0 = time.perf_counter()
        use_real = self.should_use_real_vqa(mode, tasks)
        if not use_real:
            res = self._mock_result(query, mode, reason="VQA_MODE setting disabled real inference", mock_factory=mock_factory)
            res.run_context.total_time_ms = int((time.perf_counter() - t0) * 1000)
            return res

        if not image_file_paths:
            res = self._mock_result(query, mode, reason="No input images available for real VQA", mock_factory=mock_factory)
            res.run_context.total_time_ms = int((time.perf_counter() - t0) * 1000)
            return res

        try:
            return self._run_real_pipeline(query, mode, image_file_paths, t0)
        except Exception as e:
            logger.exception(f"Real VQA pipeline failed; falling back to mock: {e}")
            force_real = (settings.VQA_MODE or "auto").lower() == "real"
            if force_real:
                raise
            res = self._mock_result(
                query,
                mode,
                reason=f"Real VQA failed ({type(e).__name__}: {e}); mock fallback used.",
                mock_factory=mock_factory,
            )
            res.run_context.errors.append(f"Real pipeline failure: {type(e).__name__}: {e}")
            res.run_context.total_time_ms = int((time.perf_counter() - t0) * 1000)
            return res

    # ------------------------------------------------------------------
    # Real pipeline
    # ------------------------------------------------------------------
    def _run_real_pipeline(
        self,
        query: str,
        mode: str,
        image_file_paths: List[Path],
        t0: float,
    ) -> VQAServiceResult:
        ctx = VQARunContext(execution_mode="real")
        model_id = settings.VQA_MODEL_ID
        ctx.model_id = model_id

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

        # --- Model loading / cache lookup ----------------------------------
        try:
            adapter = get_adapter_for_model(model_id)
        except ModelLoadingError:
            raise

        load_start = time.perf_counter()
        loaded = self._manager.load(model_id, adapter.load)
        ctx.model_load_meta = dict(loaded.metadata)
        ctx.model_load_meta["cache_hit"] = loaded.age_sec > (time.perf_counter() - load_start)
        ctx.model_load_meta["load_duration_sec"] = loaded.load_duration_sec
        ctx.evidence.append(
            f"Model '{model_id}' ready (took {loaded.load_duration_sec:.1f}s on first load; "
            f"params={loaded.metadata.get('num_params_millions','?')}M, device={loaded.metadata.get('device_actual','?')})."
        )

        # --- Adapter preprocessing ----------------------------------------
        inf_input = VQAInferenceInput(
            rgb_image=preproc.rgb_image,
            query_text=query,
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
        if inf_output.confidence is not None:
            ctx.evidence.append(
                f"Model-reported confidence: {inf_output.confidence:.3f}."
            )
        else:
            ctx.evidence.append(
                "Model does not emit a calibrated confidence score; confidence=null."
            )

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
            f"Inference completed; generated ~{ctx.inference_meta.get('generated_token_count','?')} tokens."
        )
        ctx.total_time_ms = int((time.perf_counter() - t0) * 1000)

        return VQAServiceResult(
            answer=answer,
            confidence=inf_output.confidence,
            evidence=list(ctx.evidence),
            tool_id=self.TOOL_ID,
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
    ) -> VQAServiceResult:
        if mock_factory is not None:
            res = mock_factory()
            if res.run_context is None:
                res.run_context = VQARunContext(execution_mode="mock")
            res.run_context.execution_mode = "mock"
            res.run_context.evidence.append(reason)
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
            tool_id=self.TOOL_ID,
            is_mock=True,
            run_context=ctx,
        )


_vqa_service_singleton: Optional[VQAService] = None


def get_vqa_service() -> VQAService:
    global _vqa_service_singleton
    if _vqa_service_singleton is None:
        _vqa_service_singleton = VQAService()
    return _vqa_service_singleton
