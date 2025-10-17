.. _metrics:

Evaluation Metrics
==================

`lono_libs` provides a comprehensive suite of evaluation metrics for both classification and regression tasks.
All metrics adhere to the :class:`~lono_libs.core._base_metric.IMetric` interface, ensuring consistency across your evaluation pipeline.

Common Metric Attributes
------------------------

All metrics implement the following standard attributes:

*   **`name`**: The display name of the metric (used for identification and reporting).
*   **`is_higher_better`**: Boolean flag indicating whether higher values indicate better performance (``True`` for accuracy, ``False`` for loss metrics).
*   **`weight`**: A weighting factor for combining metrics (default: 0.0; currently not used in aggregation but available for future extensions).
*   **`target_score`**: Optional ideal or desired score for the metric (e.g., 1.0 for perfect classification accuracy).

Classification Metrics
----------------------

These metrics evaluate the performance of classification models by comparing predicted class labels with true labels.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   lono_libs.classification.Accuracy
   lono_libs.classification.BalancedAccuracy
   lono_libs.classification.CohensKappa
   lono_libs.classification.ConfusionMatrix
   lono_libs.classification.F1Score
   lono_libs.classification.LogLoss
   lono_libs.classification.MatthewsCorrelationCoefficient
   lono_libs.classification.Precision
   lono_libs.classification.Recall
   lono_libs.classification.ROCAUC

Regression Metrics
------------------

These metrics assess the performance of regression models by quantifying the difference between predicted and actual continuous values.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   lono_libs.regression.Correlation
   lono_libs.regression.Kurtosis
   lono_libs.regression.MAE
   lono_libs.regression.MSE
   lono_libs.regression.R2Score
   lono_libs.regression.Skewness