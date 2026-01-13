"""
End-to-end integration tests for ExaTune.

These tests verify the complete workflow from configuration to result collection.
"""

import pytest

pytestmark = pytest.mark.integration
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import yaml
import json

from exatune.core.experiment import Experiment
from exatune.core.config import ExaTuneConfig


@pytest.fixture
def minimal_config():
    """Minimal configuration for integration testing."""
    return {
        "experiment": {
            "name": "integration_test",
            "output_dir": None,  # Will be set in tests
            "random_seed": 42
        },
        "dataset": {"name": "iris"},
        "model": {
            "type": "sklearn",
            "class_name": "sklearn.tree.DecisionTreeClassifier",
            "task": "classification"
        },
        "hyperparameters": {
            "max_depth": {"type": "discrete", "values": [3, 5]},
            "min_samples_split": {"type": "discrete", "values": [2]}
        },
        "evaluation": {
            "cv_folds": 3,
            "scoring": "accuracy"
        },
        "slurm": {
            "partition": "default",
            "time": "00:30:00",
            "memory": "4G",
            "cpus_per_task": 1
        }
    }


class TestEndToEndWorkflow:
    """Tests for complete end-to-end workflow."""

    def test_experiment_initialization(self, minimal_config):
        """Test that experiment initializes correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config["experiment"]["output_dir"] = tmpdir

            experiment = Experiment.from_dict(minimal_config)

            # Check directories were created
            assert experiment.output_dir.exists()
            assert experiment.jobs_dir.exists()
            assert experiment.results_dir.exists()
            assert experiment.checkpoints_dir.exists()
            assert experiment.logs_dir.exists()

            # Check config was saved
            assert (experiment.output_dir / "config.yaml").exists()

            # Check metadata
            assert (experiment.output_dir / "metadata.json").exists()

    def test_job_generation(self, minimal_config):
        """Test job script generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config["experiment"]["output_dir"] = tmpdir

            experiment = Experiment.from_dict(minimal_config)
            n_jobs = experiment.generate_jobs()

            # Should generate 2 jobs (2 max_depth values × 1 min_samples_split)
            assert n_jobs == 2

            # Check job scripts exist
            job_scripts = list(experiment.jobs_dir.glob("job_*.sh"))
            assert len(job_scripts) == 2

            # Check job script content
            job_script_content = job_scripts[0].read_text()
            assert "#!/bin/bash" in job_script_content
            assert "worker.py" in job_script_content

    @patch('exatune.hpc.slurm_client.SlurmClient.is_slurm_available')
    @patch('exatune.hpc.slurm_client.SlurmClient.submit_job')
    def test_job_submission_workflow(self, mock_submit, mock_available, minimal_config):
        """Test job submission with mocked SLURM."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config["experiment"]["output_dir"] = tmpdir

            # Setup mocks
            mock_available.return_value = True
            mock_submit.side_effect = ["job_1", "job_2"]

            # Create and run experiment
            experiment = Experiment.from_dict(minimal_config)
            experiment.generate_jobs()
            job_ids = experiment.submit_jobs()

            # Verify
            assert len(job_ids) == 2
            assert job_ids == ["job_1", "job_2"]
            assert mock_submit.call_count == 2

            # Check checkpoint was created
            checkpoint_file = experiment.checkpoints_dir / "after_submission.json"
            assert checkpoint_file.exists()

    def test_result_collection_workflow(self, minimal_config):
        """Test result collection from generated results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config["experiment"]["output_dir"] = tmpdir

            experiment = Experiment.from_dict(minimal_config)
            experiment.generate_jobs()

            # Manually create fake result files
            result1 = {
                "job_id": 0,
                "config_hash": "hash1",
                "hyperparameters": {"max_depth": 3, "min_samples_split": 2},
                "success": True,
                "mean_score": 0.95,
                "std_score": 0.02,
                "timestamp": "2024-01-01 12:00:00"
            }
            result2 = {
                "job_id": 1,
                "config_hash": "hash2",
                "hyperparameters": {"max_depth": 5, "min_samples_split": 2},
                "success": True,
                "mean_score": 0.97,
                "std_score": 0.01,
                "timestamp": "2024-01-01 12:01:00"
            }

            with open(experiment.results_dir / "result_000000.json", "w") as f:
                json.dump(result1, f)
            with open(experiment.results_dir / "result_000001.json", "w") as f:
                json.dump(result2, f)

            # Collect results
            results_df = experiment.collect_results()

            # Verify
            assert len(results_df) == 2
            assert results_df["success"].all()
            assert "mean_score" in results_df.columns
            assert "rank" in results_df.columns
            assert "best_score" in results_df.columns

            # Best score should be marked
            best_rows = results_df[results_df["best_score"]]
            assert len(best_rows) == 1
            assert best_rows.iloc[0]["mean_score"] == 0.97


class TestLocalExecution:
    """Tests for local (non-SLURM) execution mode."""

    def test_local_worker_execution(self, minimal_config):
        """Test worker script can run locally."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config["experiment"]["output_dir"] = tmpdir

            # Create config file
            config_path = Path(tmpdir) / "config.yaml"
            config = ExaTuneConfig.from_dict(minimal_config)
            config.to_yaml(config_path)

            # Create output directory
            results_dir = Path(tmpdir) / "results"
            results_dir.mkdir()

            # Import and run worker functions
            from exatune.hpc.worker import (
                load_dataset,
                create_model_wrapper,
                perform_cross_validation,
                generate_result,
                save_result
            )

            # Load dataset
            X, y = load_dataset(config)
            assert X.shape[0] == 150  # Iris dataset

            # Create model
            hyperparameters = {"max_depth": 3}
            wrapper = create_model_wrapper(config, hyperparameters)
            assert wrapper is not None

            # Perform CV
            cv_results = perform_cross_validation(wrapper, X, y, config)
            assert "test_accuracy" in cv_results

            # Generate result
            result = generate_result(
                job_id=0,
                hyperparameters=hyperparameters,
                cv_results=cv_results,
                config=config,
                success=True
            )
            assert result["success"]
            assert result["mean_score"] is not None

            # Save result
            save_result(result, results_dir)

            # Verify result was saved
            result_file = results_dir / "result_000000.json"
            assert result_file.exists()


class TestCheckpointResume:
    """Tests for checkpoint and resume functionality."""

    @patch('exatune.hpc.slurm_client.SlurmClient.is_slurm_available')
    @patch('exatune.hpc.slurm_client.SlurmClient.submit_job')
    def test_checkpoint_save_load(self, mock_submit, mock_available, minimal_config):
        """Test checkpoint saving and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config["experiment"]["output_dir"] = tmpdir

            # Setup mocks
            mock_available.return_value = True
            mock_submit.side_effect = ["job_1", "job_2"]

            # Create experiment and submit jobs
            experiment = Experiment.from_dict(minimal_config)
            experiment.generate_jobs()
            job_ids = experiment.submit_jobs()

            # Verify checkpoint
            checkpoint_file = experiment.checkpoints_dir / "after_submission.json"
            assert checkpoint_file.exists()

            with open(checkpoint_file) as f:
                checkpoint = json.load(f)

            assert checkpoint["job_ids"] == job_ids

            # Create new experiment and load checkpoint
            experiment2 = Experiment.from_config(experiment.output_dir / "config.yaml")
            experiment2.load_checkpoint("after_submission")

            assert experiment2._job_ids == job_ids


class TestErrorHandling:
    """Tests for error handling in integration scenarios."""

    def test_missing_results_detection(self, minimal_config):
        """Test detection of missing job results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config["experiment"]["output_dir"] = tmpdir

            experiment = Experiment.from_dict(minimal_config)
            experiment.generate_jobs()  # Should create 2 jobs

            # Only create 1 result (missing job 1)
            result = {
                "job_id": 0,
                "config_hash": "hash1",
                "hyperparameters": {"max_depth": 3, "min_samples_split": 2},
                "success": True,
                "mean_score": 0.95,
                "std_score": 0.02,
                "timestamp": "2024-01-01 12:00:00"
            }

            with open(experiment.results_dir / "result_000000.json", "w") as f:
                json.dump(result, f)

            # Collect results
            results_df = experiment.collect_results()

            # Should only have 1 result
            assert len(results_df) == 1

    def test_failed_job_handling(self, minimal_config):
        """Test handling of failed job results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config["experiment"]["output_dir"] = tmpdir

            experiment = Experiment.from_dict(minimal_config)
            experiment.generate_jobs()

            # Create one successful and one failed result
            result1 = {
                "job_id": 0,
                "config_hash": "hash1",
                "hyperparameters": {"max_depth": 3, "min_samples_split": 2},
                "success": True,
                "mean_score": 0.95,
                "std_score": 0.02,
                "timestamp": "2024-01-01 12:00:00"
            }
            result2 = {
                "job_id": 1,
                "config_hash": "hash2",
                "hyperparameters": {"max_depth": 5, "min_samples_split": 2},
                "success": False,
                "error_message": "Model training failed",
                "mean_score": None,
                "std_score": None,
                "timestamp": "2024-01-01 12:01:00"
            }

            with open(experiment.results_dir / "result_000000.json", "w") as f:
                json.dump(result1, f)
            with open(experiment.results_dir / "result_000001.json", "w") as f:
                json.dump(result2, f)

            # Collect results
            results_df = experiment.collect_results()

            # Should have both results
            assert len(results_df) == 2

            # Check success rate
            success_rate = results_df["success"].sum() / len(results_df)
            assert success_rate == 0.5


class TestConfigurationVariations:
    """Tests for different configuration scenarios."""

    def test_xgboost_configuration(self):
        """Test with XGBoost model configuration."""
        config = {
            "experiment": {
                "name": "xgboost_test",
                "output_dir": None,
                "random_seed": 42
            },
            "dataset": {"name": "iris"},
            "model": {
                "type": "xgboost",
                "task": "classification"
            },
            "hyperparameters": {
                "max_depth": {"type": "discrete", "values": [3]},
                "learning_rate": {"type": "discrete", "values": [0.1]}
            },
            "evaluation": {
                "cv_folds": 3,
                "scoring": "accuracy"
            },
            "slurm": {
                "partition": "default",
                "time": "00:30:00",
                "memory": "4G",
                "cpus_per_task": 1
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config["experiment"]["output_dir"] = tmpdir

            experiment = Experiment.from_dict(config)
            n_jobs = experiment.generate_jobs()

            # Should generate 1 job
            assert n_jobs == 1

    def test_custom_metrics_configuration(self):
        """Test with custom evaluation metrics."""
        config = {
            "experiment": {
                "name": "metrics_test",
                "output_dir": None,
                "random_seed": 42
            },
            "dataset": {"name": "iris"},
            "model": {
                "type": "sklearn",
                "class_name": "sklearn.ensemble.RandomForestClassifier",
                "task": "classification"
            },
            "hyperparameters": {
                "n_estimators": {"type": "discrete", "values": [10]}
            },
            "evaluation": {
                "cv_folds": 5,
                "scoring": "f1_weighted",
                "additional_metrics": ["accuracy", "precision_weighted", "recall_weighted"]
            },
            "slurm": {
                "partition": "default",
                "time": "00:30:00",
                "memory": "4G",
                "cpus_per_task": 1
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config["experiment"]["output_dir"] = tmpdir

            experiment = Experiment.from_dict(config)
            assert experiment.config.evaluation.scoring == "f1_weighted"
            assert len(experiment.config.evaluation.additional_metrics) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
