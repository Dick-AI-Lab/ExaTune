"""
Tests for experiment orchestration (exatune.core.experiment).
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from exatune.core.experiment import Experiment
from exatune.core.config import ExaTuneConfig


@pytest.fixture
def minimal_config():
    """Minimal valid configuration for testing."""
    return {
        "experiment": {
            "name": "test_experiment",
            "description": "Test experiment",
            "output_dir": None,  # Will be set to temp dir
            "random_seed": 42
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
            "max_depth": [3, 5],
            "min_samples_split": [2, 4]
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


class TestExperimentInitialization:
    """Tests for Experiment initialization."""

    def test_init_creates_directories(self, minimal_config):
        """Test that initialization creates required directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config["experiment"]["output_dir"] = tmpdir
            config = ExaTuneConfig.from_dict(minimal_config)
            experiment = Experiment(config)

            # Check directories were created
            assert experiment.output_dir.exists()
            assert experiment.jobs_dir.exists()
            assert experiment.results_dir.exists()
            assert experiment.checkpoints_dir.exists()
            assert experiment.logs_dir.exists()

    def test_init_saves_config(self, minimal_config):
        """Test that initialization saves configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config["experiment"]["output_dir"] = tmpdir
            config = ExaTuneConfig.from_dict(minimal_config)
            experiment = Experiment(config)

            config_path = experiment.output_dir / "config.yaml"
            assert config_path.exists()

    def test_init_saves_metadata(self, minimal_config):
        """Test that initialization saves metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config["experiment"]["output_dir"] = tmpdir
            config = ExaTuneConfig.from_dict(minimal_config)
            experiment = Experiment(config)

            metadata_path = experiment.output_dir / "metadata.json"
            assert metadata_path.exists()

            with open(metadata_path) as f:
                metadata = json.load(f)

            assert metadata["experiment_name"] == "test_experiment"
            assert metadata["model_type"] == "sklearn"
            assert metadata["total_combinations"] == 4  # 2 x 2

    def test_from_config_file(self, minimal_config):
        """Test creating experiment from config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config["experiment"]["output_dir"] = tmpdir
            config = ExaTuneConfig.from_dict(minimal_config)

            # Save config to file
            config_path = Path(tmpdir) / "config.yaml"
            config.to_yaml(config_path)

            # Load experiment from file
            experiment = Experiment.from_config(config_path)
            assert experiment.name == "test_experiment"
            assert experiment.grid_size == 4

    def test_from_dict(self, minimal_config):
        """Test creating experiment from dictionary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config["experiment"]["output_dir"] = tmpdir
            experiment = Experiment.from_dict(minimal_config)
            assert experiment.name == "test_experiment"

    def test_grid_size_property(self, minimal_config):
        """Test grid_size property."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config["experiment"]["output_dir"] = tmpdir
            experiment = Experiment.from_dict(minimal_config)
            assert experiment.grid_size == 4  # 2 x 2 combinations


class TestJobGeneration:
    """Tests for job script generation."""

    @patch('exatune.hpc.slurm_client.SlurmClient._check_slurm_available')
    def test_generate_jobs(self, mock_check_slurm, minimal_config):
        """Test job script generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config["experiment"]["output_dir"] = tmpdir
            experiment = Experiment.from_dict(minimal_config)

            n_jobs = experiment.generate_jobs()

            assert n_jobs == 4
            # Check job scripts were created
            job_scripts = list(experiment.jobs_dir.glob("job_*.sh"))
            assert len(job_scripts) == 4

    @patch('exatune.hpc.slurm_client.SlurmClient._check_slurm_available')
    def test_job_script_content(self, mock_check_slurm, minimal_config):
        """Test that job scripts have correct content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config["experiment"]["output_dir"] = tmpdir
            experiment = Experiment.from_dict(minimal_config)
            experiment.generate_jobs()

            # Read first job script
            job_script = experiment.jobs_dir / "job_000000.sh"
            with open(job_script) as f:
                content = f.read()

            # Check basic structure
            assert "#!/bin/bash" in content
            assert "SBATCH" in content
            assert "worker.py" in content
            assert "--config" in content
            assert "--job-id" in content
            assert "--hyperparameters" in content

    @patch('exatune.hpc.slurm_client.SlurmClient._check_slurm_available')
    def test_create_job_script_with_seed(self, mock_check_slurm, minimal_config):
        """Test that job scripts use correct random seeds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config["experiment"]["output_dir"] = tmpdir
            minimal_config["experiment"]["random_seed"] = 100
            experiment = Experiment.from_dict(minimal_config)
            experiment.generate_jobs()

            # Check that seed appears in job script
            job_script = experiment.jobs_dir / "job_000000.sh"
            with open(job_script) as f:
                content = f.read()

            # Job 0 should have seed 100 (base + 0)
            assert "--random-seed 100" in content


class TestJobSubmission:
    """Tests for job submission (mocked)."""

    @patch('exatune.hpc.slurm_client.SlurmClient')
    def test_submit_jobs_checks_slurm(self, mock_slurm_class, minimal_config):
        """Test that submit_jobs checks for SLURM availability."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config["experiment"]["output_dir"] = tmpdir
            experiment = Experiment.from_dict(minimal_config)

            # Create dummy job scripts so generate_jobs isn't needed
            for i in range(4):
                (experiment.jobs_dir / f"job_{i:06d}.sh").write_text("#!/bin/bash\n")

            # Mock SLURM not available
            mock_slurm_class.is_slurm_available.return_value = False

            job_ids = experiment.submit_jobs()

            assert job_ids == []

    @patch('exatune.hpc.slurm_client.SlurmClient')
    def test_submit_individual_jobs(self, mock_slurm_class, minimal_config):
        """Test submitting individual jobs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config["experiment"]["output_dir"] = tmpdir
            experiment = Experiment.from_dict(minimal_config)

            # Create dummy job scripts so generate_jobs isn't needed
            for i in range(4):
                (experiment.jobs_dir / f"job_{i:06d}.sh").write_text("#!/bin/bash\n")

            # Mock SLURM available
            mock_slurm_class.is_slurm_available.return_value = True
            mock_client = MagicMock()
            mock_client.submit_job.side_effect = ["job_1", "job_2", "job_3", "job_4"]
            mock_slurm_class.return_value = mock_client

            job_ids = experiment.submit_jobs()

            assert len(job_ids) == 4
            assert job_ids == ["job_1", "job_2", "job_3", "job_4"]
            assert mock_client.submit_job.call_count == 4

    @patch('exatune.hpc.slurm_client.SlurmClient')
    def test_submit_saves_checkpoint(self, mock_slurm_class, minimal_config):
        """Test that job submission saves checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config["experiment"]["output_dir"] = tmpdir
            experiment = Experiment.from_dict(minimal_config)

            # Create dummy job scripts so generate_jobs isn't needed
            for i in range(4):
                (experiment.jobs_dir / f"job_{i:06d}.sh").write_text("#!/bin/bash\n")

            # Mock SLURM
            mock_slurm_class.is_slurm_available.return_value = True
            mock_client = MagicMock()
            mock_client.submit_job.side_effect = ["job_1", "job_2", "job_3", "job_4"]
            mock_slurm_class.return_value = mock_client

            experiment.submit_jobs()

            # Check checkpoint was created
            checkpoint_path = experiment.checkpoints_dir / "after_submission.json"
            assert checkpoint_path.exists()


class TestJobMonitoring:
    """Tests for job monitoring (mocked)."""

    @patch('exatune.hpc.slurm_client.SlurmClient')
    def test_monitor_no_jobs(self, mock_slurm_class, minimal_config):
        """Test monitoring with no jobs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config["experiment"]["output_dir"] = tmpdir
            experiment = Experiment.from_dict(minimal_config)

            # No jobs submitted - should return without error
            experiment.monitor_progress()  # Should not crash

    @patch('exatune.hpc.slurm_client.SlurmClient')
    def test_monitor_checks_slurm(self, mock_slurm_class, minimal_config):
        """Test that monitor checks SLURM availability."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config["experiment"]["output_dir"] = tmpdir
            experiment = Experiment.from_dict(minimal_config)
            experiment._job_ids = ["job_1", "job_2"]

            # Mock SLURM not available
            mock_slurm_class.is_slurm_available.return_value = False

            experiment.monitor_progress()  # Should not crash


class TestResultCollection:
    """Tests for result collection."""

    def test_collect_results_empty(self, minimal_config):
        """Test collecting results when none exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config["experiment"]["output_dir"] = tmpdir
            experiment = Experiment.from_dict(minimal_config)

            results = experiment.collect_results()

            assert results.empty

    def test_collect_results_with_data(self, minimal_config):
        """Test collecting results with actual result files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config["experiment"]["output_dir"] = tmpdir
            experiment = Experiment.from_dict(minimal_config)

            # Create fake result files
            result1 = {
                "job_id": 0,
                "config_hash": "abc123",
                "hyperparameters": {"max_depth": 3, "min_samples_split": 2},
                "success": True,
                "mean_score": 0.95,
                "std_score": 0.02,
                "timestamp": "2024-01-01 12:00:00"
            }
            result2 = {
                "job_id": 1,
                "config_hash": "def456",
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

            results = experiment.collect_results()

            assert len(results) == 2
            assert results["success"].all()
            assert "mean_score" in results.columns
            assert "best_score" in results.columns

    def test_validate_and_enrich_adds_best_score(self, minimal_config):
        """Test that validation adds best_score flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config["experiment"]["output_dir"] = tmpdir
            experiment = Experiment.from_dict(minimal_config)

            # Create result files with different scores
            for i, score in enumerate([0.90, 0.95, 0.92]):
                result = {
                    "job_id": i,
                    "config_hash": f"hash{i}",
                    "hyperparameters": {"max_depth": i + 3},
                    "success": True,
                    "mean_score": score,
                    "std_score": 0.01,
                    "timestamp": f"2024-01-01 12:0{i}:00"
                }
                with open(experiment.results_dir / f"result_{i:06d}.json", "w") as f:
                    json.dump(result, f)

            results = experiment.collect_results()

            # Best score should be 0.95 (index 1)
            best_rows = results[results["best_score"]]
            assert len(best_rows) == 1
            assert best_rows.iloc[0]["mean_score"] == 0.95

    def test_validate_and_enrich_adds_rank(self, minimal_config):
        """Test that validation adds rank column."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config["experiment"]["output_dir"] = tmpdir
            experiment = Experiment.from_dict(minimal_config)

            # Create result files
            for i, score in enumerate([0.90, 0.95, 0.92]):
                result = {
                    "job_id": i,
                    "config_hash": f"hash{i}",
                    "hyperparameters": {"max_depth": i + 3},
                    "success": True,
                    "mean_score": score,
                    "std_score": 0.01,
                    "timestamp": f"2024-01-01 12:0{i}:00"
                }
                with open(experiment.results_dir / f"result_{i:06d}.json", "w") as f:
                    json.dump(result, f)

            results = experiment.collect_results()

            assert "rank" in results.columns
            # Ranks: 0.95 -> rank 1, 0.92 -> rank 2, 0.90 -> rank 3
            ranks = results.sort_values("job_id")["rank"].tolist()
            assert ranks == [3, 1, 2]


class TestCheckpoints:
    """Tests for checkpoint functionality."""

    def test_save_checkpoint(self, minimal_config):
        """Test saving checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config["experiment"]["output_dir"] = tmpdir
            experiment = Experiment.from_dict(minimal_config)
            experiment._job_ids = ["job_1", "job_2", "job_3"]

            experiment.save_checkpoint("test_checkpoint")

            checkpoint_path = experiment.checkpoints_dir / "test_checkpoint.json"
            assert checkpoint_path.exists()

            with open(checkpoint_path) as f:
                checkpoint = json.load(f)

            assert checkpoint["job_ids"] == ["job_1", "job_2", "job_3"]

    def test_load_checkpoint(self, minimal_config):
        """Test loading checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config["experiment"]["output_dir"] = tmpdir
            experiment = Experiment.from_dict(minimal_config)
            experiment._job_ids = ["job_1", "job_2"]
            experiment.save_checkpoint("test_checkpoint")

            # Clear job IDs and reload
            experiment._job_ids = []
            experiment.load_checkpoint("test_checkpoint")

            assert experiment._job_ids == ["job_1", "job_2"]

    def test_load_nonexistent_checkpoint(self, minimal_config):
        """Test loading non-existent checkpoint raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config["experiment"]["output_dir"] = tmpdir
            experiment = Experiment.from_dict(minimal_config)

            with pytest.raises(FileNotFoundError):
                experiment.load_checkpoint("nonexistent")


class TestResultSaving:
    """Tests for result saving."""

    def test_save_results(self, minimal_config):
        """Test saving results to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config["experiment"]["output_dir"] = tmpdir
            experiment = Experiment.from_dict(minimal_config)

            import pandas as pd
            results = pd.DataFrame([
                {"job_id": 0, "mean_score": 0.95, "success": True},
                {"job_id": 1, "mean_score": 0.92, "success": True}
            ])

            experiment.save_results(results)

            # Check files were created
            parquet_path = experiment.results_dir / "all_results.parquet"
            csv_path = experiment.results_dir / "all_results.csv"

            assert parquet_path.exists()
            assert csv_path.exists()

    def test_load_results(self, minimal_config):
        """Test loading previously saved results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config["experiment"]["output_dir"] = tmpdir
            experiment = Experiment.from_dict(minimal_config)

            import pandas as pd
            results = pd.DataFrame([
                {"job_id": 0, "mean_score": 0.95, "success": True},
                {"job_id": 1, "mean_score": 0.92, "success": True}
            ])

            experiment.save_results(results)

            # Load back
            loaded_results = experiment.load_results()

            assert loaded_results is not None
            assert len(loaded_results) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
