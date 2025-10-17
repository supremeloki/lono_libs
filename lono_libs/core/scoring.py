import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Protocol, runtime_checkable
_logger = logging.getLogger(__name__)
@runtime_checkable
class IMetric(Protocol):
    name: str
    is_higher_better: bool
    weight: float
    target_score: Optional[float]
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float: ...
    def calculate_from_proba(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float: ...
class ScoreAggregator:
    def __init__(self, metric_definitions: Dict[str, IMetric]):
        self._metric_definitions = metric_definitions
    def calculate_weighted_average(self, model_metric_results: List[Dict[str, Any]]) -> float:
        weighted_normalized_sum = 0.0
        total_contributing_weight = 0.0
        for result in model_metric_results:
            metric_name = result['metric_name']
            metric_obj = self._metric_definitions.get(metric_name)
            if not metric_obj or metric_obj.weight == 0:
                continue
            score = result['Testing Score']
            normalized_score = score
            if not metric_obj.is_higher_better:
                normalized_score = 1.0 / (1.0 + score) if score >= 0 else 0.0
            weighted_normalized_sum += normalized_score * metric_obj.weight
            total_contributing_weight += metric_obj.weight
        return weighted_normalized_sum / total_contributing_weight if total_contributing_weight > 0 else 0.0
    def calculate_score_difference(self, train_score: float, test_score: float) -> float:
        return train_score - test_score
    def calculate_relative_improvement(self, baseline_score: float, current_score: float, is_higher_better: bool) -> float:
        if baseline_score == 0:
            return float('inf') if current_score != 0 else 0.0
        if is_higher_better:
            return (current_score - baseline_score) / abs(baseline_score)
        else:
            return (baseline_score - current_score) / abs(baseline_score)
