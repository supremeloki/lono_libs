import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from lono_libs import UnifiedRunner, Precision
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.4f}'.format)
np.random.seed(42)
num_samples = 200
num_features = 5
X = pd.DataFrame(np.random.rand(num_samples, num_features) * 100, columns=[f'feature_{i}' for i in range(num_features)])
y_class = pd.Series(np.random.choice(['A', 'B', 'C'], num_samples))
y_reg = pd.Series(np.random.rand(num_samples) * 10 + X['feature_0'] * 0.5 + X['feature_1'] * 0.2 + np.random.randn(num_samples) * 2)
X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
    X, y_class, y_reg, test_size=0.3, random_state=42
)
target_precision_score = 1.0
preprocessing_config = {
    "add_polynomial_features": True,
    "poly_degree": 2,
    "apply_scaler": True
}
metric_target_overrides = {
    "Precision": target_precision_score
}
runner = UnifiedRunner(
    output_base_dir="lono_results_precision_example",
    enable_logging=True,
    enable_visualizations=True,
    target_score_overrides=metric_target_overrides,
)
print("Starting LONO_LIBS UnifiedRunner pipeline execution...")
full_results_df, best_models_by_metric = runner.run_pipeline(
    X_train=X_train,
    X_test=X_test,
    y_train_class=y_train_class,
    y_test_class=y_test_class,
    y_train_reg=y_train_reg,
    y_test_reg=y_test_reg,
    preprocessing_config=preprocessing_config,
    metrics_to_evaluate=["Precision"]
)
print("\n--- LONO_LIBS Pipeline Execution Complete ---")
runner.generate_reports_and_visualizations(
    full_results_df=full_results_df,
    best_models_by_metric=best_models_by_metric
)
print("\n--- Reports and Visualizations Generated (Check 'lono_results_precision_example' directory) ---")
print("\nPrecision Model Performance Comparison (from UnifiedRunner results):")
print("=" * 100)
precision_results_df = full_results_df[full_results_df['metric_name'] == 'Precision']
print(precision_results_df[['Model', 'Type', 'Training Score', 'Testing Score', 'Difference', 'Score_Difference_From_Target']].to_string(index=False))
print("=" * 100)
if 'Precision' in best_models_by_metric and not best_models_by_metric['Precision'].empty:
    best_precision_model = best_models_by_metric['Precision'].iloc[0]
    print(f"\nBest Precision Overall Model (Closest to {best_precision_model['target_score']:.4f}):")
    print(f"Model: {best_precision_model['Model']} ({best_precision_model['Type']})")
    print(f"Testing Score: {best_precision_model['Testing Score']:.4f}")
    if 'Score_Difference_From_Target' in best_precision_model:
        print(f"Difference from target ({best_precision_model['target_score']:.4f}): {best_precision_model['Score_Difference_From_Target']:.4f}")
else:
    print("\nNo best Precision model found in the results.")
print("\nExecution Finished. Check the 'lono_results_precision_example' directory for detailed reports and plots.")
