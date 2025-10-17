import logging
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  # type: ignore

from lono_libs import UnifiedRunner
from lono_libs.regression import Correlation  # type: ignore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
RANDOM_SEED = 42
NUM_SAMPLES = 200
NUM_FEATURES = 5
TEST_SIZE = 0.3
TARGET_CORRELATION_SCORE = 0.9800

# Set pandas display options for better output formatting
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.4f}'.format)


def generate_sample_data(num_samples: int = NUM_SAMPLES,
                        num_features: int = NUM_FEATURES,
                        test_size: float = TEST_SIZE) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Generate sample regression and classification data for demonstration.

    Args:
        num_samples: Number of samples to generate
        num_features: Number of features
        test_size: Proportion of data for testing

    Returns:
        Tuple of (X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg)
    """
    np.random.seed(RANDOM_SEED)

    # Generate features
    X = pd.DataFrame(
        np.random.rand(num_samples, num_features) * 100,
        columns=[f'feature_{i}' for i in range(num_features)]
    )

    # Generate classification target (not used in correlation example but kept for compatibility)
    y_class = pd.Series(np.random.choice(['A', 'B', 'C'], num_samples))

    # Generate regression target with correlation to features
    y_reg = pd.Series(
        np.random.rand(num_samples) * 10 +
        X['feature_0'] * 0.5 + X['feature_1'] * 0.2 +
        np.random.randn(num_samples) * 2
    )

    # Split data
    try:
        X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
            X, y_class, y_reg, test_size=test_size, random_state=RANDOM_SEED
        )
        return X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg
    except ValueError as e:
        logger.error(f"Error splitting data: {e}")
        raise


def setup_evaluation_config() -> Tuple[dict, dict]:
    """
    Set up preprocessing and metric target configurations.

    Returns:
        Tuple of (preprocessing_config, metric_target_overrides)
    """
    preprocessing_config = {
        "add_polynomial_features": True,
        "poly_degree": 2,
        "apply_scaler": True
    }

    metric_target_overrides = {
        "Correlation": TARGET_CORRELATION_SCORE
    }

    return preprocessing_config, metric_target_overrides

def run_correlation_evaluation() -> None:
    """
    Run the correlation evaluation pipeline.
    """
    logger.info("Starting correlation evaluation...")

    # Generate sample data
    X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = generate_sample_data()

    # Setup configurations
    preprocessing_config, metric_target_overrides = setup_evaluation_config()

    # Initialize UnifiedRunner
    runner = UnifiedRunner(
        output_base_dir="correlation_results",
        enable_logging=True,
        enable_visualizations=True,
        target_score_overrides=metric_target_overrides,
    )

    try:
        # Run pipeline
        full_results_df, best_models_by_metric = runner.run_pipeline(
            X_train=X_train,
            X_test=X_test,
            y_train_class=y_train_class,
            y_test_class=y_test_class,
            y_train_reg=y_train_reg,
            y_test_reg=y_test_reg,
            preprocessing_config=preprocessing_config,
            metrics_to_evaluate=["Correlation"]
        )

        logger.info("Pipeline execution completed successfully.")

        # Generate reports and visualizations
        runner.generate_reports_and_visualizations(
            full_results_df=full_results_df,
            best_models_by_metric=best_models_by_metric
        )

        # Display results
        display_correlation_results(full_results_df, best_models_by_metric)

    except Exception as e:
        logger.error(f"Error during pipeline execution: {e}")
        raise

def display_correlation_results(full_results_df: pd.DataFrame,
                              best_models_by_metric: dict) -> None:
    """
    Display correlation evaluation results.

    Args:
        full_results_df: Full results dataframe
        best_models_by_metric: Best models by metric
    """
    print("\nCorrelation Model Performance Comparison:")
    print("=" * 100)

    correlation_results_df = full_results_df[full_results_df['metric_name'] == 'Correlation']

    if correlation_results_df.empty:
        print("No correlation results found.")
        return

    # Display key columns
    columns_to_display = ['Model', 'Type', 'Training Score', 'Testing Score',
                        'Difference', 'Score_Difference_From_Target']
    available_columns = [col for col in columns_to_display if col in correlation_results_df.columns]
    if available_columns:
        print(correlation_results_df[available_columns].to_string(index=False))
    else:
        print("No suitable columns found for display.")
    print("=" * 100)

    # Display best model
    if 'Correlation' in best_models_by_metric and not best_models_by_metric['Correlation'].empty:
        best_corr_model = best_models_by_metric['Correlation'].iloc[0]
        target_score = best_corr_model.get('target_score', None)
        target_str = f"{target_score:.4f}" if target_score is not None else "N/A"
        print(f"\nBest Correlation Model (Target: {target_str}):")
        print(f"  Model: {best_corr_model['Model']} ({best_corr_model['Type']})")
        print(f"  Testing Score: {best_corr_model['Testing Score']:.4f}")
        if 'Score_Difference_From_Target' in best_corr_model:
            print(f"  Difference from Target: {best_corr_model['Score_Difference_From_Target']:.4f}")
    else:
        print("\nNo best correlation model found.")

    print("\nExecution finished. Check 'correlation_results' directory for detailed reports and plots.")


if __name__ == "__main__":
    try:
        run_correlation_evaluation()
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        exit(1)