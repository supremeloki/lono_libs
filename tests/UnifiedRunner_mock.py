import pandas as pd
import numpy as np
import os
import shutil
import logging
from typing import Optional
# Assume Accuracy is available from 'test/Accuracy_mock.py' or similar context for this mock
class IMetric: # Redefining IMetric mock here for standalone execution context
    name: str = ""
    is_higher_better: bool = False
    weight: float = 0.0
    target_score: Optional[float] = None
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError
    def calculate_from_proba(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
