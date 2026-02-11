"""
Tests for configuration system (exatune.core.config).
"""

import pytest
import tempfile
from pathlib import Path

from exatune.core.config import (
    ExaTuneConfig,
    HyperparameterSpec,
    ModelConfig,
    DatasetConfig,
    EvaluationConfig,
    SlurmConfig,
    ExperimentConfig,
    load_config,
)


class TestHyperparameterSpec:
    """Tests for HyperparameterSpec class."""

    def test_discrete_values(self):
        """Test hyperparameter specification with explicit values."""
        spec = HyperparameterSpec(
            type="int",
            values=[10, 50, 100]
        )
        assert spec.type == "int"
        assert spec.values == [10, 50, 100]
        assert spec.min is None
        assert spec.max is None

    def test_range_values(self):
        """Test hyperparameter specification with min/max range."""
        spec = HyperparameterSpec(
            type="float",
            min=0.01,
            max=1.0,
            step=0.1,
            scale="log"
        )
        assert spec.type == "float"
        assert spec.min == 0.01
        assert spec.max == 1.0
        assert spec.step == 0.1
        assert spec.scale == "log"

    def test_invalid_no_values_or_range(self):
        """Test that spec requires either values or min/max."""
        with pytest.raises(ValueError):
            HyperparameterSpec(type="int")

    def test_invalid_missing_max(self):
        """Test that range spec requires both min and max."""
        with pytest.raises(ValueError):
            HyperparameterSpec(type="float", min=0.01)


class TestModelConfig:
    """Tests for ModelConfig class."""

    def test_sklearn_model(self):
        """Test sklearn model configuration."""
        config = ModelConfig(
            type="sklearn",
            class_name="sklearn.ensemble.RandomForestClassifier",
            task="classification"
        )
        assert config.type == "sklearn"
        assert config.class_name == "sklearn.ensemble.RandomForestClassifier"
        assert config.task == "classification"

    def test_xgboost_model(self):
        """Test XGBoost model configuration."""
        config = ModelConfig(
            type="xgboost",
            class_name="XGBRegressor",
            task="regression"
        )
        assert config.type == "xgboost"
        assert config.task == "regression"
        assert config.class_name == "XGBRegressor"

    def test_invalid_task(self):
        """Test that invalid task raises error."""
        with pytest.raises(ValueError):
            ModelConfig(
                type="sklearn",
                class_name="sklearn.tree.DecisionTreeClassifier",
                task="invalid_task"
            )


class TestDatasetConfig:
    """Tests for DatasetConfig class."""

    def test_builtin_dataset(self):
        """Test built-in dataset configuration."""
        config = DatasetConfig(name="iris")
        assert config.name == "iris"
        assert config.path is None
        assert config.target_column is None

    def test_custom_dataset(self):
        """Test custom dataset configuration."""
        config = DatasetConfig(
            name="custom",
            path="/path/to/data.csv",
            target_column="target"
        )
        assert config.name == "custom"
        assert config.path == Path("/path/to/data.csv")
        assert config.target_column == "target"


class TestEvaluationConfig:
    """Tests for EvaluationConfig class."""

    def test_default_evaluation(self):
        """Test default evaluation configuration."""
        config = EvaluationConfig(
            cv_folds=5,
            scoring="accuracy"
        )
        assert config.cv_folds == 5
        assert config.scoring == "accuracy"
        assert config.additional_metrics == []

    def test_with_additional_metrics(self):
        """Test evaluation with additional metrics."""
        config = EvaluationConfig(
            cv_folds=10,
            scoring="f1",
            additional_metrics=["precision", "recall"]
        )
        assert config.cv_folds == 10
        assert config.scoring == "f1"
        assert config.additional_metrics == ["precision", "recall"]

    def test_invalid_cv_folds(self):
        """Test that CV folds must be >= 2."""
        with pytest.raises(ValueError):
            EvaluationConfig(cv_folds=1, scoring="accuracy")


class TestSlurmConfig:
    """Tests for SlurmConfig class."""

    def test_minimal_slurm_config(self):
        """Test minimal SLURM configuration."""
        config = SlurmConfig(
            partition="default",
            time="01:00:00",
            memory="4G",
            cpus_per_task=1
        )
        assert config.partition == "default"
        assert config.time == "01:00:00"
        assert config.memory == "4G"
        assert config.cpus_per_task == 1
        assert config.nodes == 1  # default

    def test_full_slurm_config(self):
        """Test full SLURM configuration with all options."""
        config = SlurmConfig(
            partition="gpu",
            time="24:00:00",
            memory="32G",
            cpus_per_task=8,
            nodes=1,
            account="my_account",
            qos="high",
            email="user@example.com",
            email_type="ALL",
            additional_directives={"gres": "gpu:1"}
        )
        assert config.partition == "gpu"
        assert config.time == "24:00:00"
        assert config.memory == "32G"
        assert config.cpus_per_task == 8
        assert config.nodes == 1
        assert config.account == "my_account"
        assert config.qos == "high"
        assert config.email == "user@example.com"
        assert config.email_type == "ALL"
        assert config.additional_directives == {"gres": "gpu:1"}


class TestExaTuneConfig:
    """Tests for main ExaTuneConfig class."""

    @pytest.fixture
    def minimal_config_dict(self):
        """Minimal valid configuration dictionary."""
        return {
            "experiment": {
                "name": "test_exp",
                "description": "Test experiment",
                "output_dir": "/tmp/test_output"
            },
            "dataset": {
                "name": "iris"
            },
            "model": {
                "type": "sklearn",
                "class_name": "sklearn.tree.DecisionTreeClassifier",
                "task": "classification"
            },
            "hyperparameters": {
                "max_depth": [3, 5, 7]
            },
            "evaluation": {
                "cv_folds": 5,
                "scoring": "accuracy"
            },
            "slurm": {
                "partition": "default",
                "time": "00:30:00",
                "memory": "4G",
                "cpus_per_task": 1
            }
        }

    def test_from_dict(self, minimal_config_dict):
        """Test creating config from dictionary."""
        config = ExaTuneConfig.from_dict(minimal_config_dict)
        assert config.experiment.name == "test_exp"
        assert config.dataset.name == "iris"
        assert config.model.type == "sklearn"
        assert "max_depth" in config.hyperparameters

    def test_get_hyperparameter_specs(self, minimal_config_dict):
        """Test getting hyperparameter specifications."""
        config = ExaTuneConfig.from_dict(minimal_config_dict)
        specs = config.get_hyperparameter_specs()
        assert "max_depth" in specs
        assert isinstance(specs["max_depth"], HyperparameterSpec)
        assert specs["max_depth"].type == "int"

    def test_yaml_roundtrip(self, minimal_config_dict):
        """Test saving and loading config from YAML."""
        config = ExaTuneConfig.from_dict(minimal_config_dict)

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            config.to_yaml(yaml_path)

            # Load back
            loaded_config = load_config(yaml_path)

            assert loaded_config.experiment.name == config.experiment.name
            assert loaded_config.dataset.name == config.dataset.name
            assert loaded_config.model.type == config.model.type

    def test_dict_roundtrip(self, minimal_config_dict):
        """Test converting config to dict and back."""
        config = ExaTuneConfig.from_dict(minimal_config_dict)

        # Convert to dict and back
        config_dict = config.to_dict()
        loaded_config = ExaTuneConfig.from_dict(config_dict)

        assert loaded_config.experiment.name == config.experiment.name
        assert loaded_config.dataset.name == config.dataset.name

    def test_missing_required_field(self):
        """Test that missing required field raises error."""
        incomplete_dict = {
            "experiment": {"name": "test"},
            # Missing dataset, model, etc.
        }
        with pytest.raises(Exception):  # Pydantic validation error
            ExaTuneConfig.from_dict(incomplete_dict)

    def test_multiple_hyperparameters(self, minimal_config_dict):
        """Test configuration with multiple hyperparameters."""
        minimal_config_dict["hyperparameters"] = {
            "max_depth": [3, 5, 7],
            "min_samples_split": [2, 5, 10],
            "learning_rate": {
                "type": "float",
                "min": 0.01,
                "max": 1.0,
                "step": 0.1,
                "scale": "log"
            }
        }
        config = ExaTuneConfig.from_dict(minimal_config_dict)
        specs = config.get_hyperparameter_specs()
        assert len(specs) == 3
        assert "max_depth" in specs
        assert "min_samples_split" in specs
        assert "learning_rate" in specs

    def test_random_seed(self, minimal_config_dict):
        """Test random seed configuration."""
        minimal_config_dict["experiment"]["random_seed"] = 42
        config = ExaTuneConfig.from_dict(minimal_config_dict)
        assert config.experiment.random_seed == 42

    def test_tags(self, minimal_config_dict):
        """Test experiment tags."""
        minimal_config_dict["experiment"]["tags"] = ["test", "development"]
        config = ExaTuneConfig.from_dict(minimal_config_dict)
        assert config.experiment.tags == ["test", "development"]


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_invalid_yaml_path(self):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_empty_hyperparameters(self):
        """Test that empty hyperparameters results in empty specs."""
        config_dict = {
            "experiment": {"name": "test", "output_dir": "/tmp"},
            "dataset": {"name": "iris"},
            "model": {
                "type": "sklearn",
                "class_name": "sklearn.tree.DecisionTreeClassifier",
                "task": "classification"
            },
            "hyperparameters": {},  # Empty
            "evaluation": {"cv_folds": 5, "scoring": "accuracy"},
            "slurm": {
                "partition": "default",
                "time": "00:30:00",
                "memory": "4G",
                "cpus_per_task": 1
            }
        }
        config = ExaTuneConfig.from_dict(config_dict)
        specs = config.get_hyperparameter_specs()
        assert specs == {}

    def test_sklearn_requires_class_name(self):
        """Test that sklearn models require class_name."""
        config_dict = {
            "experiment": {"name": "test", "output_dir": "/tmp"},
            "dataset": {"name": "iris"},
            "model": {
                "type": "sklearn",
                # Missing class_name
                "task": "classification"
            },
            "hyperparameters": {"max_depth": [3]},
            "evaluation": {"cv_folds": 5, "scoring": "accuracy"},
            "slurm": {
                "partition": "default",
                "time": "00:30:00",
                "memory": "4G",
                "cpus_per_task": 1
            }
        }
        with pytest.raises(Exception):  # Validation error
            ExaTuneConfig.from_dict(config_dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
