from typing import List

from fastapi import APIRouter

from ..schemas import BenchmarkMetric

router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])

MOCK_METRICS: List[BenchmarkMetric] = [
    BenchmarkMetric(taskType="vqa", metricName="Accuracy", value=None, datasetName="RSVQA-HR", evaluatedAt=None),
    BenchmarkMetric(taskType="vqa", metricName="F1 Score", value=None, datasetName="RSVQA-LR", evaluatedAt=None),
    BenchmarkMetric(taskType="captioning", metricName="BLEU-4", value=None, datasetName="RSITMD", evaluatedAt=None),
    BenchmarkMetric(taskType="captioning", metricName="CIDEr", value=None, datasetName="RSITMD", evaluatedAt=None),
    BenchmarkMetric(taskType="captioning", metricName="METEOR", value=None, datasetName="UCM-Captions", evaluatedAt=None),
    BenchmarkMetric(taskType="grounding", metricName="mAP@0.5", value=None, datasetName="DIOR-RSVG", evaluatedAt=None),
    BenchmarkMetric(taskType="grounding", metricName="IoU (mean)", value=None, datasetName="DIOR-RSVG", evaluatedAt=None),
    BenchmarkMetric(taskType="change_detection", metricName="F1 Score", value=None, datasetName="LEVIR-CD", evaluatedAt=None),
    BenchmarkMetric(taskType="change_detection", metricName="IoU", value=None, datasetName="LEVIR-CD", evaluatedAt=None),
    BenchmarkMetric(taskType="change_detection", metricName="Precision", value=None, datasetName="xBD", evaluatedAt=None),
    BenchmarkMetric(taskType="change_detection", metricName="Recall", value=None, datasetName="xBD", evaluatedAt=None),
    BenchmarkMetric(taskType="change_vqa", metricName="Accuracy", value=None, datasetName="LEVIR-CD-QA (custom)", evaluatedAt=None),
]


@router.get("", response_model=List[BenchmarkMetric])
def get_benchmark_metrics():
    return MOCK_METRICS
