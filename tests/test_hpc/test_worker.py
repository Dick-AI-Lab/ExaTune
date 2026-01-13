"""
Tests for worker script (exatune.hpc.worker).
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from exatune.hpc.worker import (
    load_dataset,
    create_model_wrapper,
    perform_cross_validation,
    generate_result,
    save_result,
)
from exatune.core.config import ExaTuneConfig


@pytest.fixture
def minimal_config():
    """Minimal configuration for testing."""
    config_dict = {
        "experiment": {
            "name": "test",
            "output_dir": "/tmp",
            "random_seed": 42
        },
        "dataset": {"name": "iris"},
        "model": {
            "type": "sklearn",
            "class_name": "sklearn.tree.DecisionTreeClassifier",
            "task": "classification"
        },
        "hyperparameters": {"max_depth": {"type": "discrete", "values": [3]}},
        "evaluation": {
            "cv_folds": 3,
            "scoring": "accuracy",
            "additional_metrics": ["f1_weighted"]
        },
        "slurm": {
            "partition": "default",
            "time": "00:30:00",
            "memory": "4G",
            "cpus_per_task": 1
        }
    }
    return ExaTuneConfig.from_dict(config_dict)


class TestLoadDataset:
    """Tests for dataset loading."""

    def test_load_iris(self, minimal_config):
        """Test loading iris dataset."""
        X, y = load_dataset(minimal_config)
        assert X.shape == (150, 4)
        assert y.shape == (150,)

    def test_load_digits(self, minimal_config):
        """Test loading digits dataset."""
        minimal_config.dataset.name = "digits"
        X, y = load_dataset(minimal_config)
        assert X.shape[0] == 1797
        assert len(y) == 1797

    def test_invalid_dataset(self, minimal_config):
        """Test that invalid dataset raises error."""
        minimal_config.dataset.name = "nonexistent_dataset"
        with pytest.raises(ValueError):
            load_dataset(minimal_config)


class TestCreateModelWrapper:
    """Tests for model wrapper creation."""

    def test_create_sklearn_wrapper(self, minimal_config):
        """Test creating sklearn model wrapper."""
        hyperparameters = {"max_depth": 5}
        wrapper = create_model_wrapper(minimal_config, hyperparameters)
        assert wrapper is not None
        assert wrapper.model.max_depth == 5

    def test_create_xgboost_wrapper(self, minimal_config):
        """Test creating XGBoost wrapper."""
        minimal_config.model.type = "xgboost"
        hyperparameters = {"max_depth": 3, "learning_rate": 0.1}
        wrapper = create_model_wrapper(minimal_config, hyperparameters)
        assert wrapper is not None


class TestPerformCrossValidation:
    """Tests for cross-validation."""

    def test_cross_validation(self, minimal_config):
        """Test cross-validation execution."""
        X, y = load_dataset(minimal_config)
        hyperparameters = {"max_depth": 5}
        wrapper = create_model_wrapper(minimal_config, hyperparameters)

        cv_results = perform_cross_validation(wrapper, X, y, minimal_config)

        assert "test_accuracy" in cv_results
        assert len(cv_results["test_accuracy"]) == 3  # 3 folds
        assert "train_accuracy" in cv_results
        assert "fit_time" in cv_results

    def test_cv_with_additional_metrics(self, minimal_config):
        """Test CV with additional metrics."""
        X, y = load_dataset(minimal_config)
        hyperparameters = {"max_depth": 5}
        wrapper = create_model_wrapper(minimal_config, hyperparameters)

        cv_results = perform_cross_validation(wrapper, X, y, minimal_config)

        assert "test_f1_weighted" in cv_results


class TestGenerateResult:
    """Tests for result generation."""

    def test_generate_successful_result(self, minimal_config):
        """Test generating result for successful job."""
        import numpy as np

        hyperparameters = {"max_depth": 5}
        cv_results = {
            "test_accuracy": np.array([0.95, 0.96, 0.94]),
            "train_accuracy": np.array([0.99, 0.98, 0.99]),
            "fit_time": np.array([0.01, 0.01, 0.01]),
            "score_time": np.array([0.001, 0.001, 0.001])
        }

        result = generate_result(
            job_id=0,
            hyperparameters=hyperparameters,
            cv_results=cv_results,
            config=minimal_config,
            success=True
        )

        assert result["job_id"] == 0
        assert result["success"] is True
        assert result["mean_score"] is not None
        assert result["std_score"] is not None
        assert "config_hash" in result
        assert "timestamp" in result

    def test_generate_failed_result(self, minimal_config):
        """Test generating result for failed job."""
        hyperparameters = {"max_depth": 5}

        result = generate_result(
            job_id=1,
            hyperparameters=hyperparameters,
            cv_results={},
            config=minimal_config,
            success=False,
            error_message="Test error"
        )

        assert result["job_id"] == 1
        assert result["success"] is False
        assert result["error_message"] == "Test error"
        assert result["mean_score"] is None


class TestSaveResult:
    """Tests for result saving."""

    def test_save_result_json(self, minimal_config):
        """Test saving result to JSON."""
        result = {
            "job_id": 0,
            "config_hash": "abc123",
            "hyperparameters": {"max_depth": 5},
            "success": True,
            "mean_score": 0.95,
            "std_score": 0.02,
            "timestamp": "2024-01-01 12:00:00"
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            save_result(result, Path(tmpdir))

            json_path = Path(tmpdir) / "result_000000.json"
            assert json_path.exists()

            with open(json_path) as f:
                loaded = json.load(f)

            assert loaded["job_id"] == 0
            assert loaded["mean_score"] == 0.95

    def test_save_result_parquet(self, minimal_config):
        """Test saving result to Parquet."""
        result = {
            "job_id": 0,
            "config_hash": "abc123",
            "hyperparameters": {"max_depth": 5},
            "success": True,
            "mean_score": 0.95,
            "std_score": 0.02,
            "timestamp": "2024-01-01 12:00:00"
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            save_result(result, Path(tmpdir))

            parquet_path = Path(tmpdir) / "results.parquet"
            assert parquet_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
