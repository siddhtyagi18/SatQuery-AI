"""
Multilingual Query Normalization Layer (Hindi -> English)
=========================================================

Additive-only, minimal service that normalizes Hindi-language remote-sensing
queries into English before they reach the existing keyword-based task
classifier, tool selection, and specialist model routing.

Design goals (per SatQuery-AI rules):
  * Existing English queries are NEVER modified.
  * Hindi queries pass through a two-stage normalization:
      1. LLM-based translation via the existing AIGateway (Gemini / OpenRouter)
         if credentials are configured.
      2. Deterministic keyword dictionary fallback for high-signal RS domain
         terms so that task classification still works even when no AI
         provider is reachable.
  * The caller (routers/analysis.py) stores the ORIGINAL user query in the
    DB record for UI display; only the translated copy is used for internal
    task classification and tool prompting.
  * No new dependencies. Uses requests (already present via ai_provider.py).
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

from ..config import get_settings
from ..logging_setup import logger

settings = get_settings()


# ---------------------------------------------------------------------------
# Stage 2 fallback: high-signal Hindi -> English domain dictionary
# ---------------------------------------------------------------------------
# Only the most discriminative satellite-query terms are listed. The job of
# this table is NOT to produce fluent English; it is to seed the existing
# keyword classifier (task_classifier.py) with enough signal words that the
# correct tasks are selected even without a live LLM provider.

_HI_EN_DOMAIN: dict[str, str] = {
    # --- Core task verbs (highest leverage for classify_task) ---
    "क्या है": "what is",
    "क्या हैं": "what are",
    "कहाँ हैं": "where are",
    "कितने": "how many",
    "कितना": "how much",
    "कौन सा": "which",
    "कौन सी": "which",
    "बताओ": "tell me",
    "पहचानो": "identify",
    "खोजो": "find",
    "ढूंढो": "find",
    "गिनो": "count",
    "गिनिए": "count",
    "दिखाओ": "show",
    "दर्शाओ": "show",
    "वर्णन करो": "describe",
    "वर्णन कीजिए": "describe",
    "सारांश": "summary caption",
    "सारांश दो": "summarize",
    "अंतर": "change difference",
    "अंतर बताओ": "change difference compare",
    "परिवर्तन": "change",
    "बदलाव": "change",
    "वृद्धि": "increase growth",
    "कमी": "decrease reduction",
    "नया": "new",
    "नए": "new",
    "निर्माण": "construction",
    "बाढ़": "flood inundation",
    "जल स्तर": "water level",

    # --- Land cover / geo objects (GROUNDING + VQA signal) ---
    "भवन": "building",
    "इमारतें": "buildings",
    "इमारतें हैं": "buildings",
    "सड़कें": "roads",
    "सड़क": "road",
    "पानी": "water",
    "जल": "water",
    "तालाब": "water reservoir lake",
    "झील": "lake water",
    "नदी": "river water",
    "बांध": "reservoir dam",
    "जंगल": "forest vegetation tree",
    "वन": "forest vegetation",
    "पेड़": "trees vegetation",
    "खेत": "agriculture farm field crop",
    "खेती": "agriculture crop",
    "फसल": "agriculture crop health",
    "हरी भरी": "vegetation",
    "घास": "grass vegetation",
    "शहरी": "urban built",
    "निर्मित संरचना": "built structure",

    # --- Change detection (CHANGE_DETECTION + CHANGE_VQA signal) ---
    "दो तारीखों के बीच": "between dates temporal",
    "दो समयों के बीच": "before after temporal",
    "पहले और बाद में": "before after change",
    "T1 और T2": "T1 T2 bi-temporal",
    "समय के साथ": "over time change",
    "अस्तित्व में परिवर्तन": "change occurred",
    "नई सड़क": "new road construction",
    "नया भवन": "new building construction",
    "नगरीय विस्तार": "urban expansion",
    "वन कटाई": "deforestation vegetation decrease",

    # --- Optical + SAR fusion (OPTICAL_SAR signal) ---
    "सर": "sar",
    "SAR": "sar",
    "रडार": "radar sar backscatter",
    "रडार डेटा": "sar data radar",
    "प्रकाशीय": "optical",
    "ऑप्टिकल": "optical",
    "बैकस्कैटर": "backscatter",
    "दृश्य पुष्टि": "sar confirm",
    "सेंटिनेल-1": "sentinel-1 sar",
    "सेंटिनेल 1": "sentinel-1 sar",
    "रिसैट": "risat sar",
    "क्रॉस मोडल": "cross-modal fusion",
    "संलयन": "fusion",

    # --- General wh-words / helper glue ---
    "क्या": "what",
    "किस": "which",
    "किसमें": "where in",
    "कैसे": "how",
    "क्यों": "why",
    "सबसे": "most all",
    "सभी": "all",
    "हर": "every all",
    "या": "or",
    "और": "and",
    "के साथ": "with",
    "के लिए": "for",
    "में": "in",
    "पर": "on",
    "की": "of",
    "का": "of",
    "को": "to",
    "है": "is",
    "हैं": "are",
    "जो": "which that",
    "वह": "that",
    "यह": "this",
    "छवि": "image scene",
    "तस्वीर": "image picture",
    "दृश्य": "scene",
    "स्थिति": "status condition",
    "दशा": "condition health",
}


def _apply_dict_fallback(hindi_text: str) -> str:
    """Apply the deterministic dictionary translator (Stage 2).

    Returns a best-effort English-ish string seeded with RS domain keywords.
    Fluent output is NOT required — only enough signal for classify_task().
    """
    if not hindi_text:
        return ""

    # Work on a lowercased copy but preserve original ranges for replacement.
    work = hindi_text
    lowered = hindi_text.lower()

    # Sort keys by length DESC so multi-word phrases are replaced before
    # shorter single-word keys they may contain.
    sorted_terms = sorted(_HI_EN_DOMAIN.keys(), key=len, reverse=True)

    for hi_phrase in sorted_terms:
        en_equiv = _HI_EN_DOMAIN[hi_phrase]
        # Case-insensitive replace; use a regex that preserves spacing.
        pattern = re.compile(
            r"\b" + re.escape(hi_phrase) + r"\b",
            flags=re.IGNORECASE,
        )
        new_work, n = pattern.subn(" " + en_equiv + " ", work)
        if n > 0:
            work = new_work

    # Collapse whitespace, remove stray punctuation clusters.
    work = re.sub(r"\s+", " ", work).strip()
    # Ensure at least one classifier-recognised hint suffix for safety.
    # (If user literally typed nothing dictionary-recognised, the empty/near
    #  empty output would force captioning-default behaviour in classify_task
    #  which is acceptable.)
    return work


def _has_devanagari(text: str) -> bool:
    """Return True if the text contains any Devanagari codepoints."""
    if not text:
        return False
    for ch in text:
        cp = ord(ch)
        # Devanagari Unicode block: U+0900 .. U+097F
        if 0x0900 <= cp <= 0x097F:
            return True
    return False


def translate_hindi_to_english(query: str, language: Optional[str] = None) -> Tuple[str, bool]:
    """Normalize a query to English for downstream task classification.

    Args:
        query: The raw user query.
        language: Optional explicit language tag ('en' | 'hi' | None).

    Returns:
        (normalized_query, was_translated)
          - normalized_query: English query to feed into the existing pipeline.
          - was_translated: True if Hindi normalization was attempted.
            The caller can use this flag to decide whether to record the
            original query separately (it already does — analysis.query is
            always the user's text; translated is only used for internal
            pipeline).
    """
    if not query or not query.strip():
        return query, False

    # --- Language detection: explicit tag first, then Devanagari heuristic ---
    explicit_hi = (language or "").lower() == "hi"
    looks_hi = _has_devanagari(query)
    is_hindi = explicit_hi or looks_hi
    if not is_hindi:
        # English or unrecognized — leave untouched.
        return query, False

    # --- Stage 1: LLM-based fluent translation via existing AIGateway -------
    llm_english: Optional[str] = None
    try:
        from .ai_provider import get_ai_gateway, AIProviderError

        gateway = get_ai_gateway()
        # Any available provider? We don't need images for text translation.
        if gateway is not None:
            providers_ok = any(
                p.is_available() for p in gateway._providers.values()
            )
            if providers_ok:
                prompt = (
                    "You are a Hindi-to-English translator for satellite remote-sensing queries.\n"
                    "Rules:\n"
                    "  - Translate the Hindi query below into natural, idiomatic English.\n"
                    "  - PRESERVE all technical remote-sensing / GIS / SAR / optical terms.\n"
                    "  - Do NOT add explanations, notes, or commentary.\n"
                    "  - Return ONLY the translated English text, nothing else.\n\n"
                    f"HINDI QUERY:\n{query}\n\nENGLISH TRANSLATION:"
                )
                try:
                    result = gateway.generate(
                        prompt=prompt,
                        image_path=None,
                        preferred_provider=None,
                        max_tokens=512,
                        temperature=0.1,
                    )
                    text_out = (result or {}).get("answer", "").strip()
                    if text_out:
                        llm_english = text_out
                        logger.info(
                            "[multilingual] LLM Hindi->English OK "
                            "(len=%d->%d)", len(query), len(llm_english)
                        )
                except AIProviderError as ai_err:
                    logger.warning(
                        "[multilingual] LLM translator unavailable (%s); "
                        "falling back to dict translator.", ai_err
                    )
                except Exception as exc:  # noqa: BLE001 — defensive
                    logger.warning(
                        "[multilingual] LLM translator raised (%s: %s); "
                        "falling back to dict translator.",
                        type(exc).__name__, exc,
                    )
    except Exception as exc:  # noqa: BLE001 — never break the pipeline
        logger.warning(
            "[multilingual] Gateway import/init failed (%s: %s); dict fallback.",
            type(exc).__name__, exc,
        )

    # --- Stage 2: deterministic dictionary fallback / enrichment -----------
    dict_english = _apply_dict_fallback(query)

    # Choose output:
    #   * If LLM produced text, use it (dict enrichment is redundant since
    #     classifier works on fluent English anyway).
    #   * Otherwise use the dict-produced English-ish text.
    if llm_english:
        normalized = llm_english
    else:
        normalized = dict_english
        logger.info(
            "[multilingual] Used dictionary fallback; signal tokens=%d",
            len(normalized.split()),
        )

    # Safety net: empty normalized output would break classify_task defaults
    # (captioning fallback for single_image, vqa default for other modes is
    #  already encoded in classify_task, but we still preserve the original
    #  Devanagari text as a last resort so nothing crashes / silently
    #  ignores the user input).
    if not normalized or not normalized.strip():
        normalized = query

    return normalized.strip(), True
