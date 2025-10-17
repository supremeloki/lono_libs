from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, List

class IMetric(ABC):
    name: str = ""
    is_higher_better: bool = False
    weight: float = 0.0
    target_score: Optional[float] = None

    @abstractmethod
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

    @abstractmethod
    def calculate_from_proba(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        pass
    @abstractmethod
    def get_params(self) -> dict:
        pass