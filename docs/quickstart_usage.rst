.. _quickstart:

Quickstart Guide
================

This guide provides a quick introduction to using `lono_libs` to evaluate your machine learning models.

Installation
------------
First, ensure you have `uv` installed:

.. code-block:: bash

   pip install uv
   # or pipx install uv

Navigate to your project root and install `lono_libs` in a virtual environment:

.. code-block:: bash

   uv venv
   source .venv/bin/activate  # On Linux/macOS
   # .venv\Scripts\activate   # On Windows
   uv pip install -e .

Basic Usage
-----------
Here's a simple example demonstrating how to use the `UnifiedRunner` with a classification metric.

.. code-block:: python

   import pandas as pd
   import numpy as np
   from sklearn.model_selection import train_test_split

   from lono_libs import UnifiedRunner
   from lono_libs.classification import Accuracy

   # Constants for configuration (improves maintainability)
   RANDOM_SEED = 42
   TEST_SIZE = 0.3
   NUM_SAMPLES = 100
   NUM_FEATURES = 5
   TARGET_ACCURACY = 0.95

   # Generate synthetic data for demonstration
   # Note: In production, use real datasets; this is optimized for quickstart
   np.random.seed(RANDOM_SEED)
   X = pd.DataFrame(
       np.random.rand(NUM_SAMPLES, NUM_FEATURES),
       columns=[f'feature_{i}' for i in range(NUM_FEATURES)]
   )
   y_class = pd.Series(
       np.random.choice([0, 1], NUM_SAMPLES),
       name='target_class'
   )
   y_reg = pd.Series(
       np.random.rand(NUM_SAMPLES),
       name='target_reg'
   )

   # Split data with stratification for classification balance
   # Using stratify to maintain class distribution
   (
       X_train, X_test,
       y_train_class, y_test_class,
       y_train_reg, y_test_reg
   ) = train_test_split(
       X, y_class, y_reg,
       test_size=TEST_SIZE,
       random_state=RANDOM_SEED,
       stratify=y_class
   )

   # Configuration dictionaries - centralized for easy modification
   preprocessing_config = {
       "add_polynomial_features": False,
       "apply_scaler": True
   }
   metric_target_overrides = {
       "Accuracy": TARGET_ACCURACY
   }

   # Initialize UnifiedRunner with error handling
   try:
       runner = UnifiedRunner(
           output_base_dir="quickstart_results",
           enable_logging=True,
           enable_visualizations=True,
           target_score_overrides=metric_target_overrides,
       )

       # Run pipeline with comprehensive error handling
       full_results_df, best_models_by_metric = runner.run_pipeline(
           X_train=X_train,
           X_test=X_test,
           y_train_class=y_train_class,
           y_test_class=y_test_class,
           y_train_reg=y_train_reg,
           y_test_reg=y_test_reg,
           preprocessing_config=preprocessing_config,
           metrics_to_evaluate=["Accuracy"]
       )

       # Generate reports and visualizations
       runner.generate_reports_and_visualizations(
           full_results_df,
           best_models_by_metric
       )

       # Display results summary
       print("Pipeline completed successfully!")
       print(f"Results shape: {full_results_df.shape}")
       print(full_results_df.head())

   except Exception as e:
       print(f"An error occurred during pipeline execution: {e}")
       # In production, implement more specific error handling
       raise

Next Steps
----------
Explore the :ref:`metrics` section for detailed information on available evaluation metrics,
and the :ref:`examples` for more complex usage scenarios.