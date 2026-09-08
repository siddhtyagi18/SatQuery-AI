"""
Multilingual Result Summary Layer (English Technical + Simple Hindi)
===================================================================

Additive-only, zero-dependency service that generates two side-by-side
views of an already-produced SatQuery-AI analysis result:

  A. summary_en — the existing technical English executive summary
                  (either re-used from the final answer_text as-is, or
                  a cleaned-up 3-bullet version of it)

  B. summary_hi — a simple, non-technical, rural-user-friendly Hindi
                  summary that avoids scientific/engineering jargon and
                  uses short, easy-to-read sentences for browser
                  text-to-speech playback.

Strict design rules (per SatQuery-AI GOLDEN BASELINE):
  * This service NEVER triggers re-analysis. It ONLY reads from the
    already-computed Analysis row (answer_text, detected_tasks,
    bounding_boxes, change_map, confidence, tool_invocations).
  * This service NEVER touches confidence, evidence, change masks,
    bounding-boxes, GIS data, or task classification.
  * Uses the EXISTING AIGateway (ai_provider.py) for Hindi rendering
    IF a provider with credentials is available; otherwise falls back
    to a large deterministic phrase dictionary so rural users still get
    a meaningful Hindi summary even without LLM connectivity.
  * Results are cached ONCE per analysis on the `adaptation` JSON
    column (key: "multilingual_summaries") so subsequent GET calls are
    O(1) and free. The adaptation column was already declared in the
    Analysis model (models.py L68) but never populated, so NO DB
    migration is required and NO teammate data is ever overwritten
    (we only write to a sub-key).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from ..config import get_settings
from ..logging_setup import logger

settings = get_settings()

CACHE_KEY = "multilingual_summaries"

# ---------------------------------------------------------------------------
# Stage 2 — Deterministic dictionary for the no-LLM fallback.
#
# Same philosophy as services/multilingual.py query translator but in
# REVERSE (EN -> HI) and tuned for *simple*, non-technical, spoken-Hindi
# output suitable for a rural listener receiving browser TTS playback.
# ---------------------------------------------------------------------------

_EN_HI_SIMPLE: Dict[str, str] = {
    # --- Top-level task / high-level findings (highest leverage) ---
    "buildings detected": "भवन मिले हैं",
    "building detected": "एक भवन मिला है",
    "new building": "नया भवन बना है",
    "new buildings": "कई नए भवन बने हैं",
    "new construction": "नई निर्माण गतिविधि",
    "construction detected": "निर्माण कार्य देखा गया है",
    "roads detected": "सड़कें मिली हैं",
    "road detected": "एक सड़क मिली है",
    "new road": "नई सड़क बनी है",
    "new roads": "नई सड़कें बनी हैं",
    "water body": "पानी का स्रोत",
    "water bodies": "पानी के स्रोत",
    "water detected": "पानी दिख रहा है",
    "flood detected": "बाढ़ की स्थिति",
    "flooding": "बाढ़ पड़ी है",
    "inundation": "पानी भरा है",
    "standing water": "ठहरा हुआ पानी",
    "agricultural": "खेती से जुड़ा",
    "agriculture": "खेती",
    "crop health": "फसल की सेहत",
    "healthy crops": "फसल अच्छी है",
    "stressed crops": "फसल कमजोर पड़ी है",
    "vegetation": "पेड़-पौधे",
    "forest": "जंगल",
    "trees": "पेड़",
    "deforestation": "पेड़ कटे हैं",
    "urban expansion": "शहर फैला है",
    "urban area": "शहरी इलाका",
    "built area": "बनी हुई संरचनाएँ",
    "increase": "बढ़ता है",
    "increased": "बढ़ गया है",
    "decrease": "कम होता है",
    "decreased": "कम हो गया है",
    "change detected": "फर्क दिख रहा है",
    "no change": "कोई खास फर्क नहीं",
    "significant change": "बड़ा बदलाव है",
    "minor change": "थोड़ा बदलाव है",
    "sar confirms": "रडार भी यही कहता है",
    "sar backscatter": "रडार की रिपोर्ट",
    "radar confirms": "रडार से पुष्टि हुई",
    "optical only": "केवल तस्वीर से पता चला",
    "cross-modal": "तस्वीर और रडार दोनों से",
    "double-bounce": "भवनों की छत और दीवार से",
    "specular reflection": "चमकदार पानी की सतह से",

    # --- Confidence / trust phrases ---
    "high confidence": "पूर्ण भरोसे के साथ",
    "medium confidence": "काफी हद तक पक्का",
    "low confidence": "इसमें कम पक्कापन है",
    "confidence": "विश्वास स्तर",
    "uncertain": "यक़ीनी नहीं",
    "estimated": "अंदाजा लगाया गया",

    # --- Narrative helpers for the bullet splitter ---
    "and": "और",
    "or": "या",
    "is": "है",
    "are": "हैं",
    "has": "है",
    "have": "हैं",
    "was": "था",
    "were": "थे",
    "in": "में",
    "on": "पर",
    "of": "का",
    "to": "को",
    "with": "के साथ",
    "for": "के लिए",
    "from": "से",
    "between": "के बीच",
    "before": "पहले",
    "after": "बाद में",
    "over time": "समय के साथ",
    "this area": "इस जगह पर",
    "the area": "इस इलाके में",
    "scene": "तस्वीर में",
    "image": "तस्वीर",
    "report": "खबर",
    "summary": "संक्षेप",
    "total": "कुल",
    "approximately": "लगभग",
    "about": "लगभग",
    "found": "मिले हैं",
    "observed": "देखे गए हैं",
    "located": "पाए गए हैं",
    "visible": "दिख रहे हैं",
    "detected": "पहचाने गए हैं",
    "identified": "बताए गए हैं",
    "counted": "गिने गए हैं",
}


# ---------------------------------------------------------------------------
# Utility: strip markdown/formatting so we don't TTS-read weird tokens
# ---------------------------------------------------------------------------

_STRIP_RE = re.compile(r"[*_#`]+")
_BULLET_SPLIT_RE = re.compile(r"\n|•|\bor\b|;| - ")


def _strip_markdown(text: str) -> str:
    """Remove markdown markers, clean whitespace, drop provenance banners."""
    if not text:
        return ""
    lines: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith(("[", "##", "###", "**Model details", "**Interpretation", "**Note", "*Analysis performed")):
            continue
        lines.append(_STRIP_RE.sub("", s).strip())
    return "  ".join(lines).strip()


def _build_en_bullets(answer_text: str, detected_tasks: List[str]) -> List[str]:
    """Build 2-4 clean English executive bullets from raw answer_text.

    Guaranteed to return at least one non-empty bullet so downstream
    Hindi generation always has input. Never touches the original
    answer_text.
    """
    cleaned = _strip_markdown(answer_text or "")
    bullets: List[str] = []
    if cleaned:
        # Heuristic: split on period or bullet markers, keep substantive
        # sentences only (>= 15 chars), de-dupe.
        for chunk in re.split(r"(?<=[.!?])\s+", cleaned):
            s = chunk.strip(" .!?")
            if len(s) >= 15 and s not in bullets:
                bullets.append(s[0].upper() + s[1:])
            if len(bullets) >= 4:
                break
    if not bullets:
        # Fallback narrative from detected tasks.
        task_map = {
            "captioning": "This satellite scene has been summarised in plain language.",
            "vqa": "Your question has been answered based on the satellite image(s).",
            "grounding": "The requested objects have been located on the satellite image.",
            "change_detection": "Temporal comparison was performed between the two images.",
            "change_vqa": "Changes visible between the two dates have been described.",
            "change_description": "A before-versus-after description is available.",
        }
        for t in detected_tasks or []:
            line = task_map.get(t)
            if line and line not in bullets:
                bullets.append(line)
    if not bullets:
        bullets.append("Satellite image analysis completed successfully.")
    return bullets[:4]


def _dict_translate_en_to_hi(en_text: str) -> str:
    """Deterministic English -> simple Hindi fallback (Stage 2).

    Produces non-fluent-but-comprehensible Hindi that a rural user can
    understand via browser TTS. Flaw grammar is acceptable here; it is
    strictly better than showing English to a Hindi-only user.
    """
    if not en_text:
        return ""
    work = en_text
    # Longest keys first to avoid partial-word clobber.
    sorted_keys = sorted(_EN_HI_SIMPLE.keys(), key=len, reverse=True)
    for en_phrase in sorted_keys:
        pattern = re.compile(r"\b" + re.escape(en_phrase) + r"\b", flags=re.IGNORECASE)
        hi_equiv = _EN_HI_SIMPLE[en_phrase]
        work = pattern.sub(hi_equiv, work)
    # Collapse doubled spaces, stray punctuation.
    work = re.sub(r"\s+", " ", work).strip(" -;:,.")
    # Ensure sentence ending for smoother TTS.
    if work and work[-1] not in "।!?":
        work = work + "।"
    return work


def _llm_simplify_to_hi(
    *,
    en_bullets: List[str],
    detected_tasks: List[str],
    confidence: Optional[float],
) -> Optional[List[str]]:
    """Stage 1 LLM simplification — uses the existing AIGateway.

    Returns 2-4 simple Hindi sentences as a list, or None on any
    provider error (caller will fall back to dictionary-based output).
    """
    try:
        from .ai_provider import get_ai_gateway, AIProviderError

        gateway = get_ai_gateway()
        if gateway is None:
            return None
        if not any(p.is_available() for p in gateway._providers.values()):
            return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("[result_multi] gateway not available: %s", exc)
        return None

    mode_label = "change analysis" if any("change" in t.lower() for t in detected_tasks) else \
                 "sar-optical analysis" if any(t == "optical_sar" for t in detected_tasks) else \
                 "satellite image"

    conf_line = ""
    if confidence is not None:
        try:
            pct = int(round(float(confidence) * 100))
            if pct >= 80:
                conf_line = "- इस नतीजे पर लगभग " + str(pct) + "% भरोसा किया जा सकता है।"
            elif pct >= 55:
                conf_line = "- इस नतीजे पर सावधानी बरतें, भरोसा सिर्फ़ " + str(pct) + "% है।"
            else:
                conf_line = "- यह नतीजा कम पक्का है, दुबारा जाँच करना चाहिए।"
        except (TypeError, ValueError):
            conf_line = ""

    bullet_block = "\n".join(f"- {b}" for b in en_bullets)
    prompt = (
        "You are a rural-literacy Hindi explainer for satellite-image findings.\n"
        "Rules — strictly follow each:\n"
        "1. Output ONLY simple Hindi. No English words except proper nouns (city names, SAR, etc).\n"
        "2. Sentences must be SHORT — 6 to 10 words each — so they are easy to listen to.\n"
        "3. Avoid ALL technical jargon: no 'spectral signature', 'backscatter', "
        "'IoU', 'classification', 'temporal analysis', 'urbanisation', etc.\n"
        "4. Instead speak to a farmer / village resident in plain language.\n"
        "   Examples of good replacements:\n"
        "   · 'Urban expansion detected'  → 'इस इलाके में नए घर बने हैं।'\n"
        "   · 'Vegetation decrease'      → 'पेड़ और हरियाली कम हो गई है।'\n"
        "   · 'SAR specular water'       → 'पानी का ठंडा इलाका भी रडार से दिखा।'\n"
        "   · 'High confidence 92%'      → 'यह बात लगभग पक्की है (92 प्रतिशत)।'\n"
        "5. Return EXACTLY 2 to 4 numbered lines, each ending in '।' (poorna viraam).\n"
        "6. NO headings, NO bullets, NO markdown, NO explanation.\n"
        "7. If a confidence hint is provided, include exactly 1 line for it.\n\n"
        f"Analysis type: {mode_label}\n"
        f"English findings (translate + simplify):\n{bullet_block}\n"
        f"Confidence hint (include 1 line if non-empty): {conf_line}\n"
        "Your 2-4 numbered Hindi lines now:"
    )

    try:
        result = gateway.generate(
            prompt=prompt,
            image_path=None,
            preferred_provider=None,
            max_tokens=600,
            temperature=0.2,
        )
    except AIProviderError as exc:
        logger.warning("[result_multi] LLM Hindi simplification unavailable: %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("[result_multi] LLM error (%s): %s", type(exc).__name__, exc)
        return None

    text_out = (result or {}).get("answer", "") or ""
    if not text_out.strip():
        return None

    hi_sentences: List[str] = []
    for raw_line in re.split(r"\n+", text_out):
        line = raw_line.strip()
        if not line:
            continue
        # Strip leading digits / markers like "1.", "2)", "- ", "• ".
        line = re.sub(r"^\s*(?:\d+[.\)\-]\s*|[•\-]\s*)", "", line).strip()
        if not line:
            continue
        if line[-1] not in "।!?":
            line = line + "।"
        hi_sentences.append(line)
        if len(hi_sentences) >= 4:
            break

    if len(hi_sentences) < 2:
        return None  # let caller use dict fallback
    return hi_sentences[:4]


def build_multilingual_summaries(
    *,
    answer_text: str,
    detected_tasks: Optional[List[str]],
    confidence: Optional[float],
    query: str,
) -> Dict[str, Any]:
    """Produce { summary_en, summary_hi, bullet_en, bullet_hi, language }

    * Calls LLM at most ONCE per analysis (results cached into the
      adaptation JSON field by the caller).
    * Never raises; always returns a valid structure even if everything
      else fails so the UI can always show both tabs.
    """
    detected = list(detected_tasks or [])
    en_bullets = _build_en_bullets(answer_text or "", detected)

    hi_sentences: List[str] = []
    used_llm = False
    if query and (
        _contains_devanagari(query)
        or (detected and len(detected) >= 1)
    ):
        llm_out = _llm_simplify_to_hi(
            en_bullets=en_bullets,
            detected_tasks=detected,
            confidence=confidence,
        )
        if llm_out:
            hi_sentences = llm_out
            used_llm = True

    if not hi_sentences:
        # Fallback: dictionary-translate each English bullet, join.
        for b in en_bullets:
            translated = _dict_translate_en_to_hi(b)
            if translated:
                hi_sentences.append(translated)
    if not hi_sentences:
        hi_sentences = [
            "सैटलाइट की तस्वीर की जाँच हो गई।",
            "इसके बारे में विस्तृत रिपोर्ट बनी हुई है।",
        ]

    # Final paragraph versions for direct display / TTS.
    summary_en = _capitalize_first(". ".join(b.rstrip(".") for b in en_bullets)) + "."
    summary_hi = " ".join(s.rstrip("।") for s in hi_sentences) + "।"

    language = "hi" if _contains_devanagari(query) else "en"

    return {
        "language": language,          # user's likely intended UI language (query-based)
        "summary_en": summary_en,      # full English paragraph (technical-ish)
        "summary_hi": summary_hi,      # full Hindi paragraph (simple)
        "bullet_en": en_bullets,       # bullets for TTS chunk playback (English)
        "bullet_hi": hi_sentences,     # bullets for TTS chunk playback (Hindi)
        "generated_via_llm": used_llm,  # provenance flag for debugging only
    }


def _contains_devanagari(text: str) -> bool:
    if not text:
        return False
    return any(0x0900 <= ord(ch) <= 0x097F for ch in text)


def _capitalize_first(text: str) -> str:
    if not text:
        return text
    return text[0].upper() + text[1:]


# ---------------------------------------------------------------------------
# Caching helpers (read-from / write-into adaptation dict)
# ---------------------------------------------------------------------------

def cached_summaries(a) -> Optional[Dict[str, Any]]:
    """Return cached summaries from an Analysis ORM object, or None."""
    try:
        adapt = a.adaptation or {}
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(adapt, dict):
        return None
    cached = adapt.get(CACHE_KEY)
    return cached if isinstance(cached, dict) and cached.get("summary_en") else None


def store_cached_summaries(db, a, summaries: Dict[str, Any]) -> None:
    """Persist summaries into Analysis.adaptation JSON.

    Safe additive write — any teammate content already inside
    `adaptation` is preserved under other keys; we only touch
    CACHE_KEY.  Also flushes the session (but does NOT commit — the
    caller already commits or not based on its own logic).
    """
    if not summaries:
        return
    try:
        existing = a.adaptation or {}
        if not isinstance(existing, dict):
            existing = {}
        # Never overwrite a teammate-written key in the adaptation
        # dict.  We only ever modify our own namespaced sub-key.
        existing[CACHE_KEY] = summaries
        a.adaptation = existing
        db.flush()
    except Exception as exc:  # noqa: BLE001
        # Caching failure must NEVER break the read API.
        logger.warning("[result_multi] cache write failed (%s: %s)", type(exc).__name__, exc)
