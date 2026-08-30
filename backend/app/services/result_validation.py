from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..schemas import AnalysisResult, BoundingBox
from .vqa_adapter import VQAInferenceOutput
from ..logging_setup import logger


# ---------------------------------------------------------------------------
# VQA output validation (no fabrication)
# ---------------------------------------------------------------------------

@dataclass
class VQAValidationResult:
    valid: bool
    cleaned_answer: Optional[str]
    warnings: List[str] = field(default_factory=list)


_VALID_ANSWER_MIN_CHARS = 2
_VALID_ANSWER_MAX_CHARS = 16000


def validate_vqa_output(output: VQAInferenceOutput) -> VQAValidationResult:
    """Validate a real VQA adapter output.

    **No fabrication rule**: the validator never invents confidence, masks,
    coordinates, or statistics. It only cleans + sanity-checks text that was
    actually produced by the model. If the model provided no confidence,
    confidence stays None.
    """
    warnings: List[str] = []
    answer = output.answer_text
    if answer is None:
        warnings.append("Model returned None answer_text; substituting empty string.")
        answer = ""

    cleaned = _clean_answer_text(answer)

    if len(cleaned.strip()) < _VALID_ANSWER_MIN_CHARS:
        warnings.append(
            f"Answer is shorter than {_VALID_ANSWER_MIN_CHARS} chars after cleanup; "
            f"treating as suspicious/invalid."
        )
        return VQAValidationResult(valid=False, cleaned_answer=None, warnings=warnings)

    if len(cleaned) > _VALID_ANSWER_MAX_CHARS:
        warnings.append(
            f"Answer exceeds {_VALID_ANSWER_MAX_CHARS} chars ({len(cleaned)}); "
            f"truncating for output."
        )
        cleaned = cleaned[:_VALID_ANSWER_MAX_CHARS] + "\n[…truncated…]"

    # Null-confidence rule: we never synthesize a confidence. Log but don't fail.
    if output.confidence is None:
        warnings.append(
            "Confidence not produced by model; confidence field preserved as null."
        )
    else:
        try:
            c = float(output.confidence)
            if not (0.0 <= c <= 1.0):
                warnings.append(
                    f"Model-reported confidence {c} outside [0,1]; discarding (set to null)."
                )
                output.confidence = None
        except (TypeError, ValueError):
            warnings.append(
                f"Model-reported confidence {output.confidence!r} is not a float; discarding."
            )
            output.confidence = None

    return VQAValidationResult(valid=True, cleaned_answer=cleaned, warnings=warnings)


def _clean_answer_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = s.strip()
    return s


# ---------------------------------------------------------------------------
# Final API-result validation (phase-level)
# ---------------------------------------------------------------------------

@dataclass
class AnalysisValidationReport:
    ok: bool
    warnings: List[str] = field(default_factory=list)
    stripped_fields: List[str] = field(default_factory=list)


def validate_analysis_result_payload(
    result: Dict[str, Any],
    is_mock: bool = False,
) -> AnalysisValidationReport:
    """Sanity-check a serialised AnalysisResult before returning it from the API.

    Rules (non-fabrication policy):
      - For REAL runs: discard/flag bounding-boxes, masks, change maps that
        have no documented source. Confidence must be model-provided or null.
      - For MOCK runs: we keep the existing payload but ensure it's still
        structurally valid so Phase 1 tests keep passing.
    """
    report = AnalysisValidationReport(ok=True)
    try:
        status = result.get("status")
        if status not in {"queued", "processing", "completed", "failed"}:
            report.warnings.append(f"Unknown analysis status: {status!r}")

        images = result.get("images", [])
        if not isinstance(images, list):
            report.warnings.append("images field is not a list")
            report.ok = False

        if status == "completed" and result.get("answerText") is None and result.get("errorReason") is None:
            report.warnings.append("completed analysis has neither answerText nor errorReason")

        # Anti-fabrication checks for real runs
        if not is_mock:
            # Boxes must either (a) not exist, or (b) come from a documented grounding tool.
            selected_tools = set(result.get("selectedTools") or [])
            boxes = result.get("boundingBoxes")
            if boxes and "rs_grounding" not in selected_tools and "optical_sar_analyzer" not in selected_tools:
                report.warnings.append(
                    "boundingBoxes present without a grounding-capable tool selected; stripping to avoid fabricated coordinates."
                )
                result["boundingBoxes"] = None
                report.stripped_fields.append("boundingBoxes")

            change_map = result.get("changeMap")
            if change_map and "change_detector" not in selected_tools:
                report.warnings.append(
                    "changeMap present without change_detector tool; stripping to avoid fabricated mask."
                )
                result["changeMap"] = None
                report.stripped_fields.append("changeMap")

            # Confidence without any tool that produces it → leave as null (don't strip it,
            # the schema already accepts Optional — we just warn if it looks obviously bogus).
            confidence = result.get("confidence")
            if confidence is not None:
                try:
                    c = float(confidence)
                    if not (0.0 <= c <= 1.0):
                        report.warnings.append(
                            f"confidence={confidence} outside [0,1]; setting to null."
                        )
                        result["confidence"] = None
                        report.stripped_fields.append("confidence")
                except (TypeError, ValueError):
                    report.warnings.append(f"confidence={confidence!r} not numeric; setting to null.")
                    result["confidence"] = None
                    report.stripped_fields.append("confidence")

    except Exception as e:
        logger.exception(f"validate_analysis_result_payload raised: {e}")
        report.ok = False
        report.warnings.append(f"validation exception: {e}")
    return report
