.. _examples:

Usage Examples
==============

This section provides a collection of detailed examples demonstrating how to leverage `lono_libs`
for various evaluation scenarios.

Example: Classification with Accuracy (Quick Start)
---------------------------------------------------
This example provides a quick introduction to using `lono_libs` for evaluating a classification model
using the Accuracy metric. It's designed for users who want to get started quickly.

.. literalinclude:: ../examples/quick start.py
   :language: python
   :linenos:

Example: Classification with Cohen's Kappa
------------------------------------------
This example shows how to use `UnifiedRunner` to evaluate a classification model
using Cohen's Kappa, including preprocessing and target score overrides.

.. note::
   This example needs to be created. For now, you can refer to the quick start example
   above and replace "Accuracy" with "CohenKappa" in the metrics_to_evaluate list.

Example: Regression with Correlation
------------------------------------
This example demonstrates evaluating a regression model using the Correlation metric.

.. literalinclude:: ../examples/example_correlation.py
   :language: python
   :linenos:

Example: Multiple Metrics Evaluation
------------------------------------
You can configure the `UnifiedRunner` to evaluate multiple metrics simultaneously.
This example demonstrates running Accuracy, F1Score (classification), and MAE (regression)
in a single pipeline run.

.. literalinclude:: ../examples/example_multiple_metrics.py
   :language: python
   :linenos: