import numpy as np
from lono_libs.core import IMetric
from typing import Optional, List, Tuple, Union
import logging
_logger = logging.getLogger(__name__)
class Accuracy(IMetric):
    name: str = "Accuracy"
    is_higher_better: bool = True
    weight: float = 0.0
    target_score: Optional[float] = None
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_true) == 0: return 0.0
        return float(np.mean(y_true == y_pred))
    def calculate_from_proba(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        raise NotImplementedError("Accuracy does not typically use probability scores directly for calculation.")
class BalancedAccuracy(IMetric):
    name: str = "BalancedAccuracy"
    is_higher_better: bool = True
    weight: float = 0.0
    target_score: Optional[float] = None
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_true) == 0: return 0.0
        classes = np.unique(y_true)
        if len(classes) == 0: return 0.0
        per_class_accuracy = []
        for c in classes:
            mask = (y_true == c)
            if np.sum(mask) == 0:
                continue
            per_class_accuracy.append(np.mean(y_pred[mask] == c))
        return float(np.mean(per_class_accuracy))
    def calculate_from_proba(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        raise NotImplementedError("Balanced Accuracy does not typically use probability scores directly.")
class CohensKappa(IMetric):
    name: str = "CohensKappa"
    is_higher_better: bool = True
    weight: float = 0.0
    target_score: Optional[float] = None
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_true) == 0: return 0.0
        unique_labels = np.union1d(np.unique(y_true), np.unique(y_pred))
        k = len(unique_labels)
        if k == 0: return 0.0
        confusion_matrix = np.zeros((k, k), dtype=int)
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        for t, p in zip(y_true, y_pred):
            if t in label_to_idx and p in label_to_idx:
                confusion_matrix[label_to_idx[t], label_to_idx[p]] += 1
        n_samples = np.sum(confusion_matrix)
        if n_samples == 0: return 0.0
        p_o = np.trace(confusion_matrix) / n_samples
        row_sums = np.sum(confusion_matrix, axis=1)
        col_sums = np.sum(confusion_matrix, axis=0)
        p_e = np.sum(row_sums * col_sums) / (n_samples * n_samples)
        if p_e == 1.0:
            return 1.0 if p_o == 1.0 else 0.0
        return (p_o - p_e) / (1 - p_e)
    def calculate_from_proba(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        raise NotImplementedError("Cohen's Kappa does not typically use probability scores directly.")
class ConfusionMatrix(IMetric):
    name: str = "ConfusionMatrix"
    is_higher_better: bool = False
    weight: float = 0.0
    target_score: Optional[float] = None
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        if len(y_true) == 0: return np.array([]).reshape(0, 0)
        unique_labels = np.union1d(np.unique(y_true), np.unique(y_pred))
        k = len(unique_labels)
        if k == 0: return np.array([]).reshape(0, 0)
        confusion_mat = np.zeros((k, k), dtype=int)
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        for t, p in zip(y_true, y_pred):
            if t in label_to_idx and p in label_to_idx:
                confusion_mat[label_to_idx[t], label_to_idx[p]] += 1
        return confusion_mat
    def calculate_from_proba(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> np.ndarray:
        y_pred = np.argmax(y_pred_proba, axis=1)
        return self.calculate(y_true, y_pred)
class UnifiedRunner:
    # Removed UnifiedRunner from __init__.py as it should be in its own file
    pass
class F1Score(IMetric):
    name: str = "F1Score"
    is_higher_better: bool = True
    weight: float = 0.0
    target_score: Optional[float] = None
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary', pos_label: Union[int, str] = 1) -> float:
        if len(y_true) == 0: return 0.0
        if average == 'binary':
            true_positives = np.sum((y_true == pos_label) & (y_pred == pos_label))
            false_positives = np.sum((y_true != pos_label) & (y_pred == pos_label))
            false_negatives = np.sum((y_true == pos_label) & (y_pred != pos_label))
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        elif average == 'macro':
            f1_scores = []
            for c in np.unique(y_true):
                y_true_binary = (y_true == c).astype(int)
                y_pred_binary = (y_pred == c).astype(int)
                f1_scores.append(self.calculate(y_true_binary, y_pred_binary, average='binary', pos_label=1))
            return float(np.mean(f1_scores))
        elif average == 'weighted':
            f1_scores = []
            class_counts = []
            for c in np.unique(y_true):
                y_true_binary = (y_true == c).astype(int)
                y_pred_binary = (y_pred == c).astype(int)
                f1_scores.append(self.calculate(y_true_binary, y_pred_binary, average='binary', pos_label=1))
                class_counts.append(np.sum(y_true == c))
            total_count = np.sum(class_counts)
            if total_count == 0: return 0.0
            return float(np.sum(np.array(f1_scores) * np.array(class_counts)) / total_count)
        else:
            raise ValueError(f"Unsupported average type: {average}")
    def calculate_from_proba(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        return self.calculate(y_true, np.argmax(y_pred_proba, axis=1))
class LogLoss(IMetric):
    name: str = "LogLoss"
    is_higher_better: bool = False
    weight: float = 0.0
    target_score: Optional[float] = None
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError("LogLoss requires probability scores (y_pred_proba). Use calculate_from_proba.")
    def calculate_from_proba(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        if len(y_true) == 0: return 0.0
        epsilon = 1e-15
        y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
        if y_pred_proba.ndim == 1:
            loss = -(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))
        elif y_pred_proba.ndim == 2:
            y_true_one_hot = np.eye(y_pred_proba.shape[1])[y_true]
            loss = -np.sum(y_true_one_hot * np.log(y_pred_proba), axis=1)
        else:
            raise ValueError("Unsupported dimensions for y_pred_proba.")
        return float(np.mean(loss))
class MatthewsCorrelationCoefficient(IMetric):
    name: str = "MatthewsCorrelationCoefficient"
    is_higher_better: bool = True
    weight: float = 0.0
    target_score: Optional[float] = None
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_true) == 0: return 0.0
        if len(np.unique(y_true)) > 2 or len(np.unique(y_pred)) > 2:
            _logger.warning("MCC is typically for binary classification. Converting to binary.")
            # Convert to binary by taking the last unique value as positive class
            unique_true = np.unique(y_true)
            unique_pred = np.unique(y_pred)
            if len(unique_true) > 2:
                pos_true = unique_true[-1]  # Assume last is positive
                y_true = (y_true == pos_true).astype(int)
            else:
                y_true = y_true.astype(int)
            if len(unique_pred) > 2:
                pos_pred = unique_pred[-1]  # Assume last is positive
                y_pred = (y_pred == pos_pred).astype(int)
            else:
                y_pred = y_pred.astype(int)
        true_pos = np.sum((y_true == 1) & (y_pred == 1))
        true_neg = np.sum((y_true == 0) & (y_pred == 0))
        false_pos = np.sum((y_true == 0) & (y_pred == 1))
        false_neg = np.sum((y_true == 1) & (y_pred == 0))
        numerator = (true_pos * true_neg) - (false_pos * false_neg)
        denominator = np.sqrt((true_pos + false_pos) * (true_pos + false_neg) *
                              (true_neg + false_pos) * (true_neg + false_neg))
        if denominator == 0:
            return 0.0
        return float(numerator / denominator)
    def calculate_from_proba(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        return self.calculate(y_true, np.argmax(y_pred_proba, axis=1))
class Precision(IMetric):
    name: str = "Precision"
    is_higher_better: bool = True
    weight: float = 0.0
    target_score: Optional[float] = None
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary', pos_label: Union[int, str] = 1) -> float:
        if len(y_true) == 0: return 0.0
        if average == 'binary':
            true_positives = np.sum((y_true == pos_label) & (y_pred == pos_label))
            false_positives = np.sum((y_true != pos_label) & (y_pred == pos_label))
            return float(true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0)
        elif average == 'macro':
            precisions = []
            for c in np.unique(y_true):
                y_true_binary = (y_true == c).astype(int)
                y_pred_binary = (y_pred == c).astype(int)
                precisions.append(self.calculate(y_true_binary, y_pred_binary, average='binary', pos_label=1))
            return float(np.mean(precisions))
        elif average == 'weighted':
            precisions = []
            class_counts = []
            for c in np.unique(y_true):
                y_true_binary = (y_true == c).astype(int)
                y_pred_binary = (y_pred == c).astype(int)
                precisions.append(self.calculate(y_true_binary, y_pred_binary, average='binary', pos_label=1))
                class_counts.append(np.sum(y_true == c))
            total_count = np.sum(class_counts)
            if total_count == 0: return 0.0
            return float(np.sum(np.array(precisions) * np.array(class_counts)) / total_count)
        else:
            raise ValueError(f"Unsupported average type: {average}")
    def calculate_from_proba(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        return self.calculate(y_true, np.argmax(y_pred_proba, axis=1))
class Recall(IMetric):
    name: str = "Recall"
    is_higher_better: bool = True
    weight: float = 0.0
    target_score: Optional[float] = None
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary', pos_label: Union[int, str] = 1) -> float:
        if len(y_true) == 0: return 0.0
        if average == 'binary':
            true_positives = np.sum((y_true == pos_label) & (y_pred == pos_label))
            false_negatives = np.sum((y_true == pos_label) & (y_pred != pos_label))
            return float(true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0)
        elif average == 'macro':
            recalls = []
            for c in np.unique(y_true):
                y_true_binary = (y_true == c).astype(int)
                y_pred_binary = (y_pred == c).astype(int)
                recalls.append(self.calculate(y_true_binary, y_pred_binary, average='binary', pos_label=1))
            return float(np.mean(recalls))
        elif average == 'weighted':
            recalls = []
            class_counts = []
            for c in np.unique(y_true):
                y_true_binary = (y_true == c).astype(int)
                y_pred_binary = (y_pred == c).astype(int)
                recalls.append(self.calculate(y_true_binary, y_pred_binary, average='binary', pos_label=1))
                class_counts.append(np.sum(y_true == c))
            total_count = np.sum(class_counts)
            if total_count == 0: return 0.0
            return float(np.sum(np.array(recalls) * np.array(class_counts)) / total_count)
        else:
            raise ValueError(f"Unsupported average type: {average}")
    def calculate_from_proba(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        return self.calculate(y_true, np.argmax(y_pred_proba, axis=1))
class ROCAUC(IMetric):
    name: str = "ROCAUC"
    is_higher_better: bool = True
    weight: float = 0.0
    target_score: Optional[float] = None
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError("ROCAUC requires probability scores (y_pred_proba). Use calculate_from_proba.")
    def calculate_from_proba(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        if len(y_true) == 0: return 0.0
        if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] > 2:
            unique_classes = np.unique(y_true)
            class_aucs = []
            for class_id in unique_classes:
                y_true_binary = (y_true == class_id).astype(int)
                class_proba = y_pred_proba[:, class_id if isinstance(class_id, int) else list(unique_classes).index(class_id)]
                if len(np.unique(y_true_binary)) < 2:
                    continue
                fpr, tpr, _ = self._roc_curve(y_true_binary, class_proba)
                class_aucs.append(self._auc_from_roc(fpr, tpr))
            return float(np.mean(class_aucs))
        elif y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
            y_pred_proba_positive = y_pred_proba[:, 1]
        elif y_pred_proba.ndim == 1:
            y_pred_proba_positive = y_pred_proba
        else:
            raise ValueError("Unsupported y_pred_proba dimensions for ROCAUC calculation.")
        if len(np.unique(y_true)) < 2:
            return 0.5
        fpr, tpr, _ = self._roc_curve(y_true, y_pred_proba_positive)
        return float(self._auc_from_roc(fpr, tpr))
    def _roc_curve(self, y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
        y_score = y_score[desc_score_indices]
        y_true = y_true[desc_score_indices]
        tps = np.cumsum(y_true == 1)
        fps = np.cumsum(y_true == 0)
        tps = np.concatenate([[0], tps])
        fps = np.concatenate([[0], fps])
        n_pos = tps[-1]
        n_neg = fps[-1]
        if n_pos == 0:
            tpr = np.zeros_like(tps, dtype=float)
        else:
            tpr = tps / n_pos
        if n_neg == 0:
            fpr = np.zeros_like(fps, dtype=float)
        else:
            fpr = fps / n_neg
        return fpr, tpr, y_score
    def _auc_from_roc(self, fpr: np.ndarray, tpr: np.ndarray) -> float:
        return np.trapz(tpr, fpr)
__all__: List[str] = [
    "Accuracy",
    "BalancedAccuracy",
    "CohensKappa",
    "ConfusionMatrix",
    "F1Score",
    "LogLoss",
    "MatthewsCorrelationCoefficient",
    "Precision",
    "Recall",
    "ROCAUC",
]
