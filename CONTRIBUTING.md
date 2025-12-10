# Contributing to ExaTune

Thank you for your interest in contributing to ExaTune! This document provides guidelines and instructions for contributing.

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue on GitHub with:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Your environment (OS, Python version, ExaTune version)
- Any relevant logs or error messages

### Suggesting Enhancements

Enhancement suggestions are welcome! Please create an issue with:

- A clear, descriptive title
- Detailed description of the proposed enhancement
- Examples of how it would be used
- Why this enhancement would be useful

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following our coding standards
3. **Add tests** for any new functionality
4. **Update documentation** as needed
5. **Ensure all tests pass** and code is formatted correctly
6. **Submit a pull request** with a clear description

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- (Optional) Access to a SLURM HPC cluster for testing

### Setting Up Your Development Environment

1. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/exatune.git
   cd exatune
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Coding Standards

### Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use meaningful variable and function names

### Code Formatting

We use automated tools for code formatting:

```bash
# Format code with black
black exatune

# Sort imports with isort
isort exatune

# Check linting with flake8
flake8 exatune

# Type check with mypy
mypy exatune
```

Pre-commit hooks will automatically run these tools before each commit.

### Documentation

- Add docstrings to all public functions, classes, and modules
- Use Google-style docstrings
- Update README.md and documentation as needed
- Include examples in docstrings where appropriate

Example docstring:
```python
def my_function(arg1: int, arg2: str) -> bool:
    """
    Brief description of function.

    Longer description if needed, explaining what the function does
    in more detail.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ValueError: When arg1 is negative
    """
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=exatune --cov-report=html

# Run specific test file
pytest tests/test_core/test_config.py

# Run tests in parallel
pytest -n auto
```

### Writing Tests

- Write tests for all new functionality
- Aim for >80% code coverage
- Use descriptive test names
- Follow the Arrange-Act-Assert pattern

Example test:
```python
def test_grid_generator_creates_correct_combinations():
    """Test that GridGenerator produces expected number of combinations."""
    # Arrange
    specs = {
        "param1": HyperparameterSpec(values=[1, 2, 3]),
        "param2": HyperparameterSpec(values=["a", "b"]),
    }
    generator = GridGenerator(specs)

    # Act
    grid = generator.generate_grid()

    # Assert
    assert len(grid) == 6  # 3 * 2 = 6 combinations
```

## Project Structure

```
exatune/
├── core/           # Core experiment orchestration
├── models/         # Model wrappers
├── hpc/            # SLURM integration
├── storage/        # Result storage backends
├── visualization/  # Plotting and visualization
├── analysis/       # Landscape analysis
└── cli/            # Command-line interface
```

## Branch Naming Convention

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test additions/updates

## Commit Message Guidelines

Write clear, concise commit messages:

```
Short summary (50 chars or less)

More detailed explanation if needed (wrap at 72 chars).
Explain what changed and why, not how.

- Bullet points are okay
- Use imperative mood ("Add feature" not "Added feature")
```

## Release Process

Releases are managed by the project maintainers:

1. Update version in `exatune/__version__.py`
2. Update CHANGELOG.md
3. Create a git tag: `git tag -a v0.1.0 -m "Release 0.1.0"`
4. Push tag: `git push origin v0.1.0`
5. GitHub Actions will automatically publish to PyPI

## Getting Help

- Create an issue for bugs or questions
- Join discussions in GitHub Discussions
- Email maintainers: kevin.dick@bornontario.ca

## Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to ExaTune!
