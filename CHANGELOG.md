# Changelog

All notable changes to lono_libs will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-10-17 - "Ver. 1.00"

### 🎉 **First Stable Release**

#### ✨ Added
- **Core Architecture**: Modular design with dependency injection for extensibility
- **Classification Metrics** (10+ metrics):
  - Accuracy, Precision, Recall, F1-Score
  - Balanced Accuracy, ROC AUC, Log Loss
  - Cohen's Kappa, Matthews Correlation Coefficient
  - Confusion Matrix
- **Regression Metrics** (6+ metrics):
  - Mean Absolute Error (MAE), Mean Squared Error (MSE)
  - R² Score, Correlation, Kurtosis, Skewness
- **UnifiedRunner**: Streamlined pipeline for batch model evaluation
- **Evaluator**: Flexible single-model evaluation with custom metrics
- **Weighted Scoring**: Custom metric weighting for composite scores
- **Automated Reporting**: Performance reports and summaries
- **Visualization Suite**: Charts, spider plots, and performance graphs
- **Data Preprocessing**: Polynomial features and standard scaling
- **ML Framework Support**: scikit-learn, XGBoost, LightGBM, CatBoost
- **Parallel Processing**: Optimized module loading with threading
- **Comprehensive Testing**: 100+ unit tests with pytest
- **Documentation**: Complete API docs with Sphinx and examples
- **Code Quality**: Black formatting, Ruff linting, MyPy type checking
- **CI/CD Pipeline**: GitHub Actions for automated testing and releases

#### 🔧 Technical Features
- **Python 3.9+** compatibility
- **MIT License** for open-source usage
- **Type Hints** throughout codebase
- **Clean API** with intuitive interfaces
- **Extensible Design** for custom metrics and components

#### 📚 Documentation
- Complete README with installation and usage examples
- API documentation with docstrings
- Example scripts and notebooks
- Contributing guidelines and development setup

---

## Types of changes
- `Added` for new features
- `Changed` for changes in existing functionality
- `Deprecated` for soon-to-be removed features
- `Removed` for now removed features
- `Fixed` for any bug fixes
- `Security` in case of vulnerabilities

---

## Development Version History

### [Unreleased]
- Future enhancements and improvements

### [0.x.x] - Development Phase
- Initial development and prototyping
- Core architecture design
- Metric implementations and testing
- Documentation and examples creation

---

**Legend:**
- 🎉 Major release
- ✨ New feature
- 🔧 Technical improvement
- 📚 Documentation
- 🐛 Bug fix
- 💥 Breaking change