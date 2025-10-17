import numpy as np
from typing import Optional
# Assume IMetric is available from 'test/IMetric_mock.py' or a similar context for this mock
class IMetric: # Redefining IMetric mock here for standalone execution context
    name: str = ""
    is_higher_better: bool = False
    weight: float = 0.0
    target_score: Optional[float] = None
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError
    def calculate_from_proba(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        raise NotImplementedError

class Accuracy(IMetric):
    name: str = "Accuracy"
    is_higher_better: bool = True
    weight: float = 1.0
    target_score: Optional[float] = 1.0

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length.")
        if len(y_true) == 0:
            return 0.0
        correct_predictions = np.sum(y_true == y_pred)
        return correct_predictions / len(y_true)

    def calculate_from_proba(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        raise NotImplementedError("Accuracy typically calculated from discrete predictions, not probabilities directly.")