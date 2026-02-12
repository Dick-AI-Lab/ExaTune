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


@pytest.fixture
def sample_results_dataframe():
    """Create a synthetic results DataFrame for visualization/analysis testing.

    Generates a 4x4x3 = 48 configuration grid with a known performance
    function: score = 0.5 + 0.001*n_estimators + 0.02*max_depth - 0.01*min_samples_split
    """
    import pandas as pd

    np.random.seed(42)
    configs = []
    for ne in [10, 50, 100, 200]:
        for md in [3, 5, 10, 15]:
            for mss in [2, 5, 10]:
                score = 0.5 + 0.001 * ne + 0.02 * md - 0.01 * mss
                train_score = score + 0.05 + np.random.normal(0, 0.003)
                configs.append({
                    "job_id": len(configs),
                    "config_hash": f"hash_{len(configs):04d}",
                    "timestamp": "2025-01-01 00:00:00",
                    "success": True,
                    "mean_score": round(score, 6),
                    "std_score": round(abs(np.random.normal(0, 0.01)), 6),
                    "mean_train_score": round(train_score, 6),
                    "fit_time_mean": round(np.random.uniform(0.1, 2.0), 3),
                    "score_time_mean": round(np.random.uniform(0.01, 0.1), 3),
                    "n_estimators": ne,
                    "max_depth": md,
                    "min_samples_split": mss,
                    "rank": None,
                    "best_score": False,
                })
    df = pd.DataFrame(configs)
    df = df.sort_values("mean_score", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    df.loc[0, "best_score"] = True
    return df


@pytest.fixture
def sample_results_with_failures(sample_results_dataframe):
    """Results DataFrame with some failed jobs."""
    df = sample_results_dataframe.copy()
    fail_indices = [5, 10, 15]
    for idx in fail_indices:
        if idx < len(df):
            df.loc[idx, "success"] = False
            df.loc[idx, "mean_score"] = np.nan
    return df
