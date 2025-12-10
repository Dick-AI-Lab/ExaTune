"""
Pytest configuration and fixtures for ExaTune tests.
"""

import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression

from exatune.core.config import (
    ExaTuneConfig,
    ExperimentConfig,
    ModelConfig,
    DatasetConfig,
    EvaluationConfig,
    SlurmConfig,
    HyperparameterSpec,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_classification_data():
    """Generate sample classification data."""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    return X, y


@pytest.fixture
def sample_regression_data():
    """Generate sample regression data."""
    X, y = make_regression(
        n_samples=100,
        n_features=10,
        n_informative=5,
        random_state=42
    )
    return X, y


@pytest.fixture
def sample_hyperparameter_specs():
    """Sample hyperparameter specifications."""
    return {
        "n_estimators": HyperparameterSpec(values=[10, 50, 100], type="int"),
        "max_depth": HyperparameterSpec(values=[3, 5, 10], type="int"),
        "min_samples_split": HyperparameterSpec(values=[2, 5], type="int"),
    }


@pytest.fixture
def sample_config(temp_dir):
    """Create a sample ExaTune configuration."""
    config = ExaTuneConfig(
        experiment=ExperimentConfig(
            name="test_experiment",
            output_dir=temp_dir / "results",
            random_seed=42
        ),
        model=ModelConfig(
            type="sklearn",
            **{"class": "RandomForestClassifier"},
            task="classification"
        ),
        hyperparameters={
            "n_estimators": [10, 50],
            "max_depth": [3, 5],
        },
        dataset=DatasetConfig(
            name="iris",
            test_size=0.2,
            random_state=42
        ),
        evaluation=EvaluationConfig(
            cv_folds=3,
            scoring="accuracy"
        ),
        slurm=SlurmConfig(
            partition="compute",
            time="00:10:00",
            memory="2G"
        )
    )
    return config


@pytest.fixture
def mock_slurm(monkeypatch):
    """Mock SLURM commands for testing."""
    def mock_run(*args, **kwargs):
        """Mock subprocess.run for SLURM commands."""
        class MockResult:
            stdout = "Submitted batch job 12345\n"
            stderr = ""
            returncode = 0

        return MockResult()

    import subprocess
    monkeypatch.setattr(subprocess, "run", mock_run)
