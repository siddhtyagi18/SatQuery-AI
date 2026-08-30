import re
from typing import Any, Dict, List, Tuple

from ..schemas import AnalysisMode, TaskType
from ..logging_setup import logger


VQA_KEYWORDS = [
    "what", "is", "are", "how", "many", "does", "do", "which", "where", "who",
    "explain", "describe", "identify", "tell me", "?",
]
CAPTION_KEYWORDS = [
    "caption", "summarize", "summary", "describe the image", "image caption",
    "what does this image show", "overview of",
]
GROUNDING_KEYWORDS = [
    "locate", "find", "detect", "where are", "mark", "bounding", "box", "boxes",
    "count.*building", "count.*road", "count.*water", "how many", "all buildings",
    "all roads", "all water", "identify all",
]
CHANGE_KEYWORDS = [
    "change", "changed", "difference", "differences", "before.*after",
    "over time", "between.*dates", "between.*time", "expansion", "increase",
    "decrease", "growth", "new construction", "destroyed", "damaged",
    "compare", "comparison", "temporal",
]
CHANGE_VQA_KEYWORDS = [
    "how much change", "why.*change", "which areas changed", "what areas changed",
    "how many.*new", "how much.*increase", "how much.*decrease",
    "has.*affected", "impact.*change", "effect.*change",
]
OPTICAL_SAR_KEYWORDS = [
    "sar", "sentinel-1", "risat", "radar", "sar data", "backscatter",
    "sar confirm", "sar detect", "cross.modal", "fused", "fusion",
    "optical.*sar", "sar.*optical",
]


def _score(text: str, keywords: List[str]) -> float:
    t = text.lower()
    score = 0.0
    for kw in keywords:
        if ".*" in kw:
            if re.search(kw, t):
                score += 1.0
        else:
            count = len(re.findall(r"\b" + re.escape(kw) + r"\b", t))
            if count == 0 and kw in t:
                count = 1
            score += count * 1.0
    return score


def classify_task(query: str, mode: AnalysisMode) -> Tuple[List[TaskType], Dict[str, float]]:
    """
    Deterministic keyword-based task classifier.
    Returns ordered list of detected tasks and raw score dict (for trace).
    """
    text = query.strip()

    scores: Dict[str, float] = {
        "vqa": 0.0,
        "captioning": 0.0,
        "grounding": 0.0,
        "change_detection": 0.0,
        "change_vqa": 0.0,
        "change_description": 0.0,
    }

    scores["captioning"] = _score(text, CAPTION_KEYWORDS)
    scores["grounding"] = _score(text, GROUNDING_KEYWORDS)
    scores["change_detection"] = _score(text, CHANGE_KEYWORDS)
    scores["change_vqa"] = _score(text, CHANGE_VQA_KEYWORDS)

    vqa_base = _score(text, VQA_KEYWORDS)
    if mode == "optical_sar":
        vqa_base += _score(text, OPTICAL_SAR_KEYWORDS) * 0.8
    scores["vqa"] = vqa_base

    if scores["change_detection"] > 0:
        scores["change_description"] = scores["change_detection"] * 0.7

    tasks: List[TaskType] = []
    threshold = 0.5

    if mode == "bi_temporal":
        if scores["change_detection"] <= threshold:
            scores["change_detection"] = 1.0
            scores["change_description"] = max(scores["change_description"], 0.8)
        if scores["vqa"] > threshold:
            scores["change_vqa"] = max(scores["change_vqa"], scores["vqa"])

    if mode == "optical_sar":
        if scores["vqa"] <= threshold:
            scores["vqa"] = 1.0
        if scores["change_detection"] > threshold:
            pass
        elif "confirm" in text.lower() or "change" in text.lower():
            scores["change_detection"] = 0.8

    if mode == "single_image":
        if scores["vqa"] <= 0 and scores["captioning"] <= 0 and scores["grounding"] <= 0:
            scores["captioning"] = 1.0
            scores["vqa"] = 0.6

    if scores["vqa"] > threshold and "vqa" not in tasks:
        tasks.append("vqa")
    if scores["captioning"] > threshold and "captioning" not in tasks:
        tasks.append("captioning")
    if scores["grounding"] > threshold and "grounding" not in tasks:
        tasks.append("grounding")
    if scores["change_detection"] > threshold and "change_detection" not in tasks:
        tasks.append("change_detection")
    if scores["change_vqa"] > threshold and "change_vqa" not in tasks:
        tasks.append("change_vqa")
    if scores["change_description"] > threshold and "change_description" not in tasks:
        tasks.append("change_description")

    if not tasks:
        tasks = ["captioning"] if mode == "single_image" else ["vqa"]

    logger.info(f"Task classification: query='{text[:60]}...' mode={mode} -> tasks={tasks}")
    return tasks, scores
