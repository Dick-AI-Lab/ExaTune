"""
Tests for CLI commands (exatune.cli.main).
"""

import pytest
import tempfile
from pathlib import Path
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
import yaml

from exatune.cli.main import cli
from exatune.core.config import ExaTuneConfig


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_config():
    """Create a sample configuration dictionary."""
    return {
        "experiment": {
            "name": "test_experiment",
            "output_dir": None,  # Will be set by test
            "random_seed": 42
        },
        "dataset": {"name": "iris"},
        "model": {
            "type": "sklearn",
            "class_name": "DecisionTreeClassifier",
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


class TestVersionCommand:
    """Tests for the version command."""

    def test_version(self, runner):
        """Test version command."""
        result = runner.invoke(cli, ["version"])

        assert result.exit_code == 0
        assert "ExaTune" in result.output or "version" in result.output.lower()


class TestValidateCommand:
    """Tests for the validate command."""

    def test_validate_valid_config(self, runner, sample_config):
        """Test validating a valid configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_config["experiment"]["output_dir"] = tmpdir
            config_path = Path(tmpdir) / "config.yaml"

            with open(config_path, "w") as f:
                yaml.dump(sample_config, f)

            result = runner.invoke(cli, ["validate", str(config_path)])

            assert result.exit_code == 0
            assert "valid" in result.output.lower() or "\u2713" in result.output

    def test_validate_invalid_config(self, runner):
        """Test validating an invalid configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "invalid_config.yaml"

            # Create invalid config (missing required fields)
            invalid_config = {
                "experiment": {"name": "test"}
                # Missing dataset, model, hyperparameters, etc.
            }

            with open(config_path, "w") as f:
                yaml.dump(invalid_config, f)

            result = runner.invoke(cli, ["validate", str(config_path)])

            assert result.exit_code != 0
            assert "error" in result.output.lower() or "invalid" in result.output.lower()

    def test_validate_nonexistent_file(self, runner):
        """Test validating a non-existent config file."""
        result = runner.invoke(cli, ["validate", "/nonexistent/config.yaml"])

        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()


class TestRunCommand:
    """Tests for the run command."""

    @patch('exatune.hpc.slurm_client.SlurmClient._check_slurm_available')
    def test_run_dry_run(self, mock_check_slurm, runner, sample_config):
        """Test dry-run mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_config["experiment"]["output_dir"] = tmpdir
            config_path = Path(tmpdir) / "config.yaml"

            with open(config_path, "w") as f:
                yaml.dump(sample_config, f)

            result = runner.invoke(cli, ["run", str(config_path), "--dry-run"])

            assert result.exit_code == 0
            assert "dry run" in result.output.lower()

    @patch('exatune.hpc.slurm_client.SlurmClient._check_slurm_available')
    @patch('exatune.hpc.slurm_client.SlurmClient.is_slurm_available')
    @patch('exatune.hpc.slurm_client.SlurmClient.submit_job')
    def test_run_full_workflow(self, mock_submit, mock_available, mock_check, runner, sample_config):
        """Test full run workflow (generate + submit)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_config["experiment"]["output_dir"] = tmpdir
            config_path = Path(tmpdir) / "config.yaml"

            with open(config_path, "w") as f:
                yaml.dump(sample_config, f)

            # Mock SLURM available and successful submission
            mock_available.return_value = True
            mock_submit.side_effect = ["job_1", "job_2", "job_3", "job_4"]

            result = runner.invoke(cli, ["run", str(config_path)])

            assert result.exit_code == 0

    def test_run_invalid_config(self, runner):
        """Test run with invalid config."""
        result = runner.invoke(cli, ["run", "/nonexistent/config.yaml"])

        assert result.exit_code != 0


class TestStatusCommand:
    """Tests for the status command."""

    def test_status_no_experiment(self, runner):
        """Test status when experiment doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, [
                "status", "nonexistent_experiment",
                "--output-dir", tmpdir
            ])

            assert result.exit_code != 0

    @patch('exatune.core.experiment.Experiment.from_config')
    def test_status_with_jobs(self, mock_from_config, runner):
        """Test status command with active jobs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal experiment directory structure
            exp_dir = Path(tmpdir) / "test_experiment"
            exp_dir.mkdir()
            (exp_dir / "config.yaml").write_text("experiment:\n  name: test\n")

            # Mock experiment
            mock_exp = MagicMock()
            mock_exp._job_ids = ["job_1", "job_2"]
            mock_from_config.return_value = mock_exp

            result = runner.invoke(cli, [
                "status", "test_experiment",
                "--output-dir", tmpdir
            ])

            # Should show status information
            assert result.exit_code == 0


class TestCollectCommand:
    """Tests for the collect command."""

    @patch('exatune.core.experiment.Experiment.from_config')
    def test_collect_results(self, mock_from_config, runner):
        """Test collecting results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create experiment directory
            exp_dir = Path(tmpdir) / "test_experiment"
            exp_dir.mkdir()
            (exp_dir / "config.yaml").write_text("experiment:\n  name: test\n")

            # Mock experiment
            mock_exp = MagicMock()
            import pandas as pd
            mock_df = pd.DataFrame([
                {"job_id": 0, "mean_score": 0.95, "success": True},
                {"job_id": 1, "mean_score": 0.92, "success": True}
            ])
            mock_exp.collect_results.return_value = mock_df
            mock_from_config.return_value = mock_exp

            result = runner.invoke(cli, [
                "collect", "test_experiment",
                "--output-dir", tmpdir
            ])

            assert result.exit_code == 0

    @patch('exatune.core.experiment.Experiment.from_config')
    def test_collect_no_results(self, mock_from_config, runner):
        """Test collecting when no results exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir) / "test_experiment"
            exp_dir.mkdir()
            (exp_dir / "config.yaml").write_text("experiment:\n  name: test\n")

            # Mock experiment with empty results
            mock_exp = MagicMock()
            import pandas as pd
            mock_exp.collect_results.return_value = pd.DataFrame()
            mock_from_config.return_value = mock_exp

            result = runner.invoke(cli, [
                "collect", "test_experiment",
                "--output-dir", tmpdir
            ])

            # Should handle gracefully
            assert result.exit_code == 0 or "no results" in result.output.lower()


class TestVisualizeCommand:
    """Tests for the visualize command."""

    def test_visualize_missing_experiment(self, runner):
        """Test that visualize aborts when experiment directory doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, [
                "visualize", "nonexistent_exp",
                "--output-dir", tmpdir,
            ])

            assert result.exit_code != 0
            assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_visualize_no_results_data(self, runner):
        """Test that visualize handles missing results data gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create experiment directory but no results files
            exp_dir = Path(tmpdir) / "test_exp"
            exp_dir.mkdir()

            result = runner.invoke(cli, [
                "visualize", "test_exp",
                "--output-dir", tmpdir,
                "--params", "max_depth"
            ])

            # Should abort with an error about missing/empty results
            assert result.exit_code != 0


class TestAnalyzeCommand:
    """Tests for the analyze command."""

    def test_analyze_missing_experiment(self, runner):
        """Test that analyze aborts when experiment directory doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, [
                "analyze", "nonexistent_exp",
                "--output-dir", tmpdir
            ])

            assert result.exit_code != 0
            assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_analyze_no_results_data(self, runner):
        """Test that analyze handles missing results data gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir) / "test_exp"
            exp_dir.mkdir()

            result = runner.invoke(cli, [
                "analyze", "test_exp",
                "--output-dir", tmpdir
            ])

            # Should abort with an error about missing/empty results
            assert result.exit_code != 0


class TestCLIErrorHandling:
    """Tests for CLI error handling."""

    def test_invalid_command(self, runner):
        """Test invalid command."""
        result = runner.invoke(cli, ["invalid-command"])

        assert result.exit_code != 0

    def test_missing_required_argument(self, runner):
        """Test missing required argument."""
        result = runner.invoke(cli, ["run"])

        assert result.exit_code != 0
        assert "usage" in result.output.lower() or "error" in result.output.lower()

    def test_help_flag(self, runner):
        """Test --help flag."""
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "usage" in result.output.lower() or "commands" in result.output.lower()

    def test_command_help(self, runner):
        """Test command-specific help."""
        result = runner.invoke(cli, ["run", "--help"])

        assert result.exit_code == 0
        assert "usage" in result.output.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
