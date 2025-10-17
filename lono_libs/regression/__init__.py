import numpy as np
from lono_libs.core import IMetric
from typing import Optional, List
from scipy.stats import skew, kurtosis  # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score as sk_r2_score  # type: ignore
class Correlation(IMetric):
    name: str = "Correlation"
    is_higher_better: bool = True
    weight: float = 0.0
    target_score: Optional[float] = 1.0
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_true.size == 0 or y_pred.size == 0:
            return 0.0
        from scipy.stats import pearsonr
        corr, _ = pearsonr(y_true, y_pred)
        return round(float(corr), 4)
    def calculate_from_proba(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        raise NotImplementedError("Correlation is not calculated from probability scores.")
class Kurtosis(IMetric):
    name: str = "Kurtosis"
    is_higher_better: bool = False
    weight: float = 0.0
    target_score: Optional[float] = 0.0
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_true.size == 0 or y_pred.size == 0:
            return np.nan
        residuals = y_true - y_pred
        if residuals.size < 4:
            return np.nan
        return round(float(kurtosis(residuals)), 4)
    def calculate_from_proba(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        raise NotImplementedError("Kurtosis is not calculated from probability scores.")
class MAE(IMetric):
    name: str = "MAE"
    is_higher_better: bool = False
    weight: float = 1.0
    target_score: Optional[float] = 0.0
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_true.size == 0 or y_pred.size == 0:
            return 0.0
        return round(float(mean_absolute_error(y_true, y_pred)), 4)
    def calculate_from_proba(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        raise NotImplementedError("MAE is not calculated from probability scores.")
class MSE(IMetric):
    name: str = "MSE"
    is_higher_better: bool = False
    weight: float = 1.0
    target_score: Optional[float] = 0.0
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_true.size == 0 or y_pred.size == 0:
            return 0.0
        return round(float(mean_squared_error(y_true, y_pred)), 4)
    def calculate_from_proba(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        raise NotImplementedError("MSE is not calculated from probability scores.")
class R2Score(IMetric):
    name: str = "R2Score"
    is_higher_better: bool = True
    weight: float = 1.0
    target_score: Optional[float] = 1.0
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_true.size == 0 or y_pred.size == 0:
            return 0.0
        return round(float(sk_r2_score(y_true, y_pred)), 4)
    def calculate_from_proba(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        raise NotImplementedError("R2Score is not calculated from probability scores.")
class Skewness(IMetric):
    name: str = "Skewness"
    is_higher_better: bool = False
    weight: float = 0.0
    target_score: Optional[float] = 0.0
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_true.size == 0 or y_pred.size == 0:
            return np.nan
        residuals = y_true - y_pred
        if residuals.size < 2:
            return np.nan
        return round(float(skew(residuals)), 4)
    def calculate_from_proba(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        raise NotImplementedError("Skewness is not calculated from probability scores.")
__all__: List[str] = [
    "Correlation",
    "Kurtosis",
    "MAE",
    "MSE",
    "R2Score",
    "Skewness",
]
