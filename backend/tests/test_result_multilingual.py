import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import pytest
import re
from app.services.result_multilingual import (
    build_multilingual_summaries,
    _dict_translate_en_to_hi,
    _strip_markdown,
)
from app.services.multilingual import _has_devanagari as is_hindi_query, translate_hindi_to_english


def _extract_numeric_entities(text: str):
    return re.findall(r"\b\d+(?:\.\d+)?%?", text)


def test_hindi_query_detection_and_translation():
    """Verify Hindi query detection and English translation for RS specialist pipeline."""
    hindi_q = "क्या 2022 और 2024 के बीच कोई बदलाव आया है?"
    english_q = "What changes occurred between these two dates?"

    assert is_hindi_query(hindi_q) is True
    assert is_hindi_query(english_q) is False

    translated, was_translated = translate_hindi_to_english(hindi_q)
    assert was_translated is True
    assert translated is not None
    assert len(translated) > 0
    # Technical query translation contains change detection intent
    assert any(term in translated.lower() for term in ["change", "difference", "between", "2022", "2024"])


def test_fact_preservation_numbers_and_percentages():
    """Golden rule: English and Hindi must describe the SAME analysis with IDENTICAL numbers."""
    en_text = (
        "Comparing the earlier acquisition (T1) with the later acquisition (T2), "
        "localized structural changes are observable in the scene.\n"
        "- Detected Changed Area: 3.14% (Severity: low)\n"
        "- Confidence: 88%\n"
        "- Total buildings: 127\n"
        "- Flooded parcel: 7.3 ha"
    )

    # Extract all numbers from original English
    original_numbers = _extract_numeric_entities(en_text)
    assert "3.14%" in original_numbers or "3.14" in original_numbers
    assert "88%" in original_numbers or "88" in original_numbers
    assert "127" in original_numbers
    assert "7.3" in original_numbers

    # Generate Hindi via deterministic dictionary fallback
    hi_summary = _dict_translate_en_to_hi(en_text)

    # Verify that every single numeric entity from English is preserved identically in Hindi
    for num in original_numbers:
        assert num in hi_summary, f"Number {num} missing in Hindi summary: {hi_summary}"


def test_clean_for_speech_tts_readiness():
    """Verify text is stripped of markdown artifacts (*, #, `, []) for clean TTS audio."""
    raw_markdown = "### Bi-Temporal Scene Change Interpretation\n\n**Qualitative Visual Interpretation (VLM):**\n- **Detected Changed Area:** `3.14%`"
    cleaned = _strip_markdown(raw_markdown)

    assert "#" not in cleaned
    assert "*" not in cleaned
    assert "`" not in cleaned
    assert "3.14%" in cleaned or "3.14" in cleaned
    assert len(cleaned) > 0


def test_deterministic_summary_generation(monkeypatch=None):
    """Verify build_multilingual_summaries builds both en and hi without touching golden pipeline."""
    import app.services.result_multilingual as rm
    orig_llm = rm._llm_simplify_to_hi
    rm._llm_simplify_to_hi = lambda **kwargs: None  # force fast deterministic dictionary path

    try:
        answer = "Localized structural changes detected. Changed Area: 3.14%. Confidence: 88%."
        summaries = build_multilingual_summaries(
            answer_text=answer,
            detected_tasks=["change_detection"],
            confidence=0.88,
            query="What changes occurred?",
        )

        assert "summary_en" in summaries
        assert "summary_hi" in summaries
        assert len(summaries["summary_en"]) > 0
        assert len(summaries["summary_hi"]) > 0
        assert "3.14" in summaries["summary_hi"] or "3.14%" in summaries["summary_hi"]
        assert "88" in summaries["summary_hi"] or "88%" in summaries["summary_hi"]
    finally:
        rm._llm_simplify_to_hi = orig_llm


if __name__ == "__main__":
    print("Running test_hindi_query_detection_and_translation...")
    test_hindi_query_detection_and_translation()
    print("PASS: test_hindi_query_detection_and_translation")

    print("Running test_fact_preservation_numbers_and_percentages...")
    test_fact_preservation_numbers_and_percentages()
    print("PASS: test_fact_preservation_numbers_and_percentages")

    print("Running test_clean_for_speech_tts_readiness...")
    test_clean_for_speech_tts_readiness()
    print("PASS: test_clean_for_speech_tts_readiness")

    print("Running test_deterministic_summary_generation...")
    test_deterministic_summary_generation()
    print("PASS: test_deterministic_summary_generation")

    print("\n>>> ALL MULTILINGUAL VERIFICATION TESTS PASSED SUCCESSFULLY! <<<")
