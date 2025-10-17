import numpy as np
from typing import Optional

class IMetric:
    name: str = ""
    is_higher_better: bool = False
    weight: float = 0.0
    target_score: Optional[float] = None

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError("Subclasses must implement the calculate method.")

    def calculate_from_proba(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        raise NotImplementedError("Subclasses must implement calculate_from_proba if applicable.")