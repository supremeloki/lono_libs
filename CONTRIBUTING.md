# Contributing to lono_libs

Thank you for your interest in contributing to lono_libs! We welcome contributions from the community and are grateful for your help in improving this project.

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- Git
- Basic understanding of machine learning concepts

### Development Setup
1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/lono_libs.git
   cd lono_libs
   ```

3. Create a virtual environment:
   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On Unix/Mac
   source .venv/bin/activate
   ```

4. Install development dependencies:
   ```bash
   pip install -e ".[dev,test,docs]"
   ```

## 📋 Contribution Guidelines

### 🐛 Reporting Bugs
- Use the GitHub issue tracker
- Provide detailed steps to reproduce the bug
- Include your environment details (Python version, OS, dependencies)
- Attach relevant code snippets or error messages

### 💡 Feature Requests
- Check existing issues before creating a new one
- Clearly describe the feature and its use case
- Explain how it would benefit the library

### 🔧 Making Changes

#### Code Style
We follow strict code quality standards:
- **Black**: Code formatting (line length: 88 characters)
- **Ruff**: Linting and import sorting
- **MyPy**: Type checking

Run quality checks:
```bash
black .
ruff check .
mypy lono_libs
```

#### Adding New Metrics
1. Create metric class in appropriate module (`lono_libs/classification/` or `lono_libs/regression/`)
2. Inherit from `IMetric` base class
3. Implement required methods: `calculate()`, `name` property
4. Add comprehensive unit tests
5. Update documentation

#### Testing
- Write unit tests for all new functionality
- Ensure all tests pass: `python run_all_tests.py`
- Maintain or improve test coverage
- Test edge cases and error conditions

#### Documentation
- Update docstrings for all public methods
- Add examples for new features
- Update README.md if needed
- Build docs locally: `cd docs && make html`

### 📝 Commit Guidelines
- Use clear, descriptive commit messages
- Start with a verb (Add, Fix, Update, Remove, etc.)
- Reference issue numbers when applicable
- Keep commits focused on single changes

Example:
```
Add support for custom metric weights in Evaluator

- Implement weighted scoring algorithm
- Add validation for weight parameters
- Update documentation with examples
- Fixes #123
```

### 🔄 Pull Request Process
1. Create a feature branch from `main`
2. Make your changes following the guidelines above
3. Run tests and quality checks
4. Update documentation if needed
5. Push your branch to GitHub
6. Create a Pull Request with:
   - Clear title and description
   - Reference to related issues
   - Summary of changes made
   - Screenshots/demo for UI changes (if applicable)

### 📊 Performance Considerations
- Profile code for performance bottlenecks
- Consider memory usage for large datasets
- Optimize algorithms where possible
- Add benchmarks for performance-critical code

### 🔒 Security
- Be aware of potential security implications
- Don't commit sensitive information
- Use secure coding practices
- Report security issues privately to maintainers

## 🎯 Areas for Contribution

### High Priority
- Additional ML metrics (AUC-PR, Cohen's Kappa variants, etc.)
- Performance optimizations
- Better error handling and validation
- Enhanced visualization options

### Medium Priority
- Integration with additional ML frameworks
- CLI interface for batch evaluation
- Export functionality for different formats
- Plugin system for custom metrics

### Nice to Have
- GUI/web interface
- Jupyter notebook integrations
- Docker containerization
- CI/CD pipeline improvements

## 🤝 Code of Conduct
- Be respectful and inclusive
- Provide constructive feedback
- Help newcomers learn and contribute
- Follow the Golden Rule: treat others as you'd like to be treated

## 📞 Getting Help
- Check existing documentation first
- Search GitHub issues for similar problems
- Ask questions in GitHub discussions
- Contact maintainers for guidance

## 🙏 Recognition
Contributors will be:
- Listed in the project's contributors file
- Credited in release notes
- Recognized for their valuable contributions

Thank you for contributing to lono_libs! 🎉