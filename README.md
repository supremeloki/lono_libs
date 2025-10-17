# Lono Library

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight and modular Python library for comprehensive machine learning model evaluation and comparison.

## 🚀 Overview

`lono_libs` provides a streamlined framework to evaluate various classification and regression models across multiple metrics, allowing for custom metric weighting, performance reporting, and visualization. It emphasizes modularity, extensibility, and ease of use, making it ideal for MLOps workflows and research.

## ✨ Features

- **Modular Metric Calculation:** Easily add and manage custom evaluation metrics for both classification and regression
- **Flexible Model Evaluation:** Evaluate a wide range of models with configurable metrics and data splits
- **Weighted Scoring:** Define custom weights for metrics to generate a composite score for model ranking
- **Automated Reporting & Visualization:** Generate performance reports and insightful plots automatically
- **Dependency Injection:** A robust architecture ensuring low coupling and high maintainability

## 🚧 Structure

```
lono_libs/
├── .flake8
├── .gitattributes
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
├── requirements.txt
├── run_all_tests.py
├── run_evaluation.py
├── docs/
│   ├── __init__.py
│   ├── api.rst
│   ├── conf.py
│   ├── examples.rst
│   ├── index.rst
│   ├── make.bat
│   ├── Makefile
│   ├── metrics.rst
│   └── quickstart_usage.rst
├── examples/
│   ├── __init__.py
│   ├── example_correlation.py
│   ├── example_multiple_metrics.py
│   └── quick start.py
├── lono_libs/
│   ├── __init__.py
│   ├── best_result.py
│   ├── data_prep.py
│   ├── models.py
│   ├── reporting.py
│   ├── summary_reports.py
│   ├── unified_runner.py
│   ├── visualization.py
│   ├── classification/
│   │   ├── __init__.py
│   │   ├── accuracy.py
│   │   ├── balanced_accuracy.py
│   │   ├── cohens_kappa.py
│   │   ├── confusion_matrix.py
│   │   ├── f1_score.py
│   │   ├── LogLoss.py
│   │   ├── matthews_correlation_coefficient.py
│   │   ├── precision.py
│   │   ├── recall.py
│   │   └── roc_auc.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── _base_metric.py
│   │   ├── evaluator.py
│   │   └── scoring.py
│   └── regression/
│       ├── __init__.py
│       ├── correlation.py
│       ├── kurtosis.py
│       ├── mae.py
│       ├── mse.py
│       ├── r2_score.py
│       └── skewness.py
└── tests/
    ├── Accuracy_mock.py
    ├── IMetric_mock.py
    ├── UnifiedRunner_mock.py
    ├── __init__.py
    ├── test_evaluator.py
    ├── test_unified_runner.py
    ├── classification/
    │   ├── __init__.py
    │   ├── test_accuracy.py
    │   ├── test_balanced_accuracy.py
    │   ├── test_cohens_kappa.py
    │   ├── test_confusion_matrix.py
    │   ├── test_f1_score.py
    │   ├── test_log_loss.py
    │   ├── test_matthews_correlation_coefficient.py
    │   ├── test_precision.py
    │   ├── test_recall.py
    │   └── test_roc_auc.py
    └── regression/
        ├── __init__.py
        ├── test_correlation.py
        ├── test_kurtosis.py
        ├── test_mae.py
        ├── test_mse.py
        ├── test_r2_score.py
        └── test_skewness.py
```
## 📦 Installation

### From PyPI
```bash
pip install lono_libs
```

### For Development
```bash
# Clone the repository
git clone https://github.com/supremeloki/lono_libs.git
cd lono_libs

# Create and activate virtual environment
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Unix/Mac
source .venv/bin/activate

# Install with development dependencies
pip install -e ".[dev,test,docs]"
```

## 📋 Requirements

- Python 3.9 or higher
- Required dependencies:
  - numpy >= 1.20
  - pandas >= 1.3
  - scikit-learn >= 1.0
  - matplotlib >= 3.4
  - scipy >= 1.7
  - xgboost >= 1.6
  - lightgbm >= 3.3
  - catboost >= 1.0

## 🏃 Quick Start

Here's a simple example of how to use `lono_libs`:

```python
from lono_libs import Evaluator
from lono_libs.classification import accuracy, f1_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Create sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Create evaluator with metrics
evaluator = Evaluator(metrics=[accuracy, f1_score])

# Evaluate model
results = evaluator.evaluate(model, X_test, y_test)
print(results.summary())
```

## 📊 Available Metrics

### Classification Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Balanced Accuracy
- Cohen's Kappa
- Matthews Correlation Coefficient
- ROC AUC
- Log Loss
- Confusion Matrix

### Regression Metrics
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score
- Correlation
- Kurtosis
- Skewness

## ⚡ Performance Benchmarks

### Evaluation Speed
- **10,000 samples, 50 features**: ~0.15-0.25 seconds for 4 metrics
- **Throughput**: ~40,000-65,000 samples/second
- **Memory Usage**: Scales linearly with dataset size

### Benchmark Example
```python
import time
from lono_libs import Evaluator
from lono_libs.classification import accuracy, f1_score, precision, recall

# 10K samples, 50 features dataset
evaluator = Evaluator(metrics=[accuracy, f1_score, precision, recall])

start_time = time.time()
results = evaluator.evaluate(model, X_test, y_test)
eval_time = time.time() - start_time

print(f"Evaluation completed in {eval_time:.4f} seconds")
# Output: Evaluation completed in 0.1876 seconds
```

### System Requirements
- **Minimum**: Python 3.9+, 4GB RAM, modern CPU
- **Recommended**: Python 3.9+, 8GB+ RAM, multi-core CPU
- **Optimal**: Python 3.11+, 16GB+ RAM, high-core CPU for parallel processing
## 🔧 Usage with UnifiedRunner

For comprehensive model evaluation across multiple algorithms:

```python
from lono_libs import UnifiedRunner

runner = UnifiedRunner(
    output_base_dir="results",
    enable_logging=True,
    enable_visualizations=True
)

# Run evaluation pipeline
results_df, best_models = runner.run_pipeline(
    X_train=X_train,
    X_test=X_test,
    y_train_class=y_train_class,
    y_test_class=y_test_class,
    y_train_reg=None,
    y_test_reg=None,
    preprocessing_config={
        "add_polynomial_features": True,
        "poly_degree": 2,
        "apply_scaler": True
    },
    metrics_to_evaluate=["Accuracy", "F1Score", "Precision", "Recall"]
)
```

## 🧪 Testing

Run the full test suite:
```bash
python run_all_tests.py
```

Or use pytest directly:
```bash
pytest --cov=lono_libs tests/
```

## 📝 Code Quality

We maintain high code quality standards:
- `black` for code formatting
- `ruff` for linting
- `mypy` for type checking

Run quality checks:
```bash
black .
ruff check .
mypy lono_libs
```

## 📚 Documentation

Build documentation locally:
```bash
cd docs
make html
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Kooroush Masoumi** - [kooroushmasoumi@gmail.com](mailto:kooroushmasoumi@gmail.com)

---

⭐ Star this repo if you find it useful!