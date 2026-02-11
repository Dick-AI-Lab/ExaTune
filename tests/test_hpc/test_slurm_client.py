"""
Tests for SLURM client (exatune.hpc.slurm_client).
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import subprocess

from exatune.hpc.slurm_client import SlurmClient
from exatune.core.config import SlurmConfig


@pytest.fixture
def minimal_slurm_config():
    """Minimal SLURM configuration for testing."""
    return SlurmConfig(
        partition="default",
        time="00:30:00",
        memory="4G",
        cpus_per_task=1
    )


@pytest.fixture
def full_slurm_config():
    """Full SLURM configuration with all options."""
    return SlurmConfig(
        partition="compute",
        time="02:00:00",
        memory="16G",
        cpus_per_task=4,
        nodes=1,
        account="def-user",
        email="user@example.com",
        modules=["python/3.9", "scipy-stack"],
        python_environment="/home/user/venv"
    )


class TestSlurmClientInitialization:
    """Tests for SlurmClient initialization."""

    @patch('exatune.hpc.slurm_client.SlurmClient._check_slurm_available')
    def test_init_with_minimal_config(self, mock_check, minimal_slurm_config):
        """Test initialization with minimal configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = SlurmClient(minimal_slurm_config, Path(tmpdir))
            assert client.config == minimal_slurm_config
            assert client.work_dir == Path(tmpdir)

    @patch('exatune.hpc.slurm_client.SlurmClient._check_slurm_available')
    def test_init_with_full_config(self, mock_check, full_slurm_config):
        """Test initialization with full configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = SlurmClient(full_slurm_config, Path(tmpdir))
            assert client.config.partition == "compute"


class TestSlurmAvailability:
    """Tests for SLURM availability checking."""

    @patch('subprocess.run')
    def test_slurm_available(self, mock_run):
        """Test SLURM availability check when available."""
        mock_run.return_value = Mock(returncode=0)

        assert SlurmClient.is_slurm_available() is True
        mock_run.assert_called_once()

    @patch('subprocess.run')
    def test_slurm_not_available(self, mock_run):
        """Test SLURM availability check when not available."""
        mock_run.side_effect = FileNotFoundError()

        assert SlurmClient.is_slurm_available() is False

    @patch('subprocess.run')
    def test_slurm_check_error(self, mock_run):
        """Test SLURM availability check with error."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "sbatch")

        assert SlurmClient.is_slurm_available() is False


class TestJobScriptGeneration:
    """Tests for SLURM job script generation."""

    @patch('exatune.hpc.slurm_client.SlurmClient._check_slurm_available')
    def test_generate_simple_script(self, mock_check, minimal_slurm_config):
        """Test generating a simple job script."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = SlurmClient(minimal_slurm_config, Path(tmpdir))

            script = client.generate_job_script(
                job_id=0,
                job_name="test_job",
                hyperparameters={"max_depth": 5},
                config_path=Path(tmpdir) / "config.yaml",
                script_path=Path("/path/to/worker.py"),
                output_dir=Path(tmpdir) / "results",
                log_dir=Path(tmpdir) / "logs",
                random_seed=42,
                python_env=None
            )

            # Check script contains required SBATCH directives
            assert "#!/bin/bash" in script
            assert "#SBATCH --job-name=test_job" in script
            assert "#SBATCH --partition=default" in script
            assert "#SBATCH --time=00:30:00" in script
            assert "#SBATCH --mem=4G" in script
            assert "#SBATCH --cpus-per-task=1" in script

            # Check worker invocation
            assert "python" in script
            assert "worker.py" in script
            assert "--config" in script
            assert "--job-id 0" in script
            assert "--random-seed 42" in script

    @patch('exatune.hpc.slurm_client.SlurmClient._check_slurm_available')
    def test_generate_script_with_modules(self, mock_check, full_slurm_config):
        """Test generating script with module loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = SlurmClient(full_slurm_config, Path(tmpdir))

            script = client.generate_job_script(
                job_id=0,
                job_name="module_job",
                hyperparameters={},
                config_path=Path(tmpdir) / "config.yaml",
                script_path=Path("/path/to/worker.py"),
                output_dir=Path(tmpdir) / "results",
                log_dir=Path(tmpdir) / "logs",
                random_seed=42,
                python_env=None
            )

            # Check module loading
            assert "module load python/3.9" in script
            assert "module load scipy-stack" in script

    @patch('exatune.hpc.slurm_client.SlurmClient._check_slurm_available')
    def test_generate_script_with_python_env(self, mock_check, full_slurm_config):
        """Test generating script with Python environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = SlurmClient(full_slurm_config, Path(tmpdir))

            script = client.generate_job_script(
                job_id=5,
                job_name="env_job",
                hyperparameters={"learning_rate": 0.01},
                config_path=Path(tmpdir) / "config.yaml",
                script_path=Path("/path/to/worker.py"),
                output_dir=Path(tmpdir) / "results",
                log_dir=Path(tmpdir) / "logs",
                random_seed=None,
                python_env="/home/user/venv"
            )

            # Check Python environment activation
            assert "source /home/user/venv/bin/activate" in script


class TestJobSubmission:
    """Tests for job submission."""

    @patch('subprocess.run')
    @patch('exatune.hpc.slurm_client.SlurmClient._check_slurm_available')
    def test_submit_job_success(self, mock_check, mock_run, minimal_slurm_config):
        """Test successful job submission."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = SlurmClient(minimal_slurm_config, Path(tmpdir))

            # Create a dummy job script
            job_script = Path(tmpdir) / "job.sh"
            job_script.write_text("#!/bin/bash\necho 'test'")

            # Mock sbatch output
            mock_run.return_value = Mock(
                returncode=0,
                stdout="Submitted batch job 12345\n"
            )

            job_id = client.submit_job(job_script)

            assert job_id == "12345"

    @patch('subprocess.run')
    @patch('exatune.hpc.slurm_client.SlurmClient._check_slurm_available')
    def test_submit_job_failure(self, mock_check, mock_run, minimal_slurm_config):
        """Test failed job submission."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = SlurmClient(minimal_slurm_config, Path(tmpdir))

            job_script = Path(tmpdir) / "job.sh"
            job_script.write_text("#!/bin/bash\necho 'test'")

            # Mock sbatch failure (check=True raises CalledProcessError)
            mock_run.side_effect = subprocess.CalledProcessError(
                1, "sbatch", stderr="sbatch: error: invalid partition"
            )

            job_id = client.submit_job(job_script)

            assert job_id is None

    @patch('subprocess.run')
    @patch('exatune.hpc.slurm_client.SlurmClient._check_slurm_available')
    def test_submit_array_job(self, mock_check, mock_run, minimal_slurm_config):
        """Test array job submission."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = SlurmClient(minimal_slurm_config, Path(tmpdir))

            job_script = Path(tmpdir) / "job_array.sh"
            job_script.write_text("#!/bin/bash\necho 'test'")

            # Mock sbatch output for array job
            mock_run.return_value = Mock(
                returncode=0,
                stdout="Submitted batch job 12345\n"
            )

            job_id = client.submit_job_array(job_script, array_size=10)

            assert job_id == "12345"
            # Check that --array was used
            call_args = mock_run.call_args[0][0]
            assert "--array=0-9" in call_args


class TestJobStatus:
    """Tests for job status queries."""

    @patch('subprocess.run')
    @patch('exatune.hpc.slurm_client.SlurmClient._check_slurm_available')
    def test_get_job_status_running(self, mock_check, mock_run, minimal_slurm_config):
        """Test getting status of running job."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = SlurmClient(minimal_slurm_config, Path(tmpdir))

            # Mock squeue output
            mock_run.return_value = Mock(
                returncode=0,
                stdout="RUNNING\n"
            )

            status = client.get_job_status("12345")

            assert status == "RUNNING"

    @patch('subprocess.run')
    @patch('exatune.hpc.slurm_client.SlurmClient._check_slurm_available')
    def test_get_job_status_completed(self, mock_check, mock_run, minimal_slurm_config):
        """Test getting status of completed job."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = SlurmClient(minimal_slurm_config, Path(tmpdir))

            # Mock squeue returns nothing (job not in queue)
            # Then mock sacct for completed job
            mock_run.side_effect = [
                Mock(returncode=0, stdout=""),  # squeue
                Mock(returncode=0, stdout="COMPLETED\n")  # sacct
            ]

            status = client.get_job_status("12345")

            assert status == "COMPLETED"

    @patch('subprocess.run')
    @patch('exatune.hpc.slurm_client.SlurmClient._check_slurm_available')
    def test_get_multiple_job_status(self, mock_check, mock_run, minimal_slurm_config):
        """Test getting status of multiple jobs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = SlurmClient(minimal_slurm_config, Path(tmpdir))

            # Mock squeue output with multiple jobs
            mock_run.return_value = Mock(
                returncode=0,
                stdout="12345 RUNNING\n12346 PENDING\n12347 COMPLETED\n"
            )

            status_dict = client.get_multiple_job_status(["12345", "12346", "12347"])

            assert status_dict["12345"] == "RUNNING"
            assert status_dict["12346"] == "PENDING"
            assert status_dict["12347"] == "COMPLETED"


class TestJobCancellation:
    """Tests for job cancellation."""

    @patch('subprocess.run')
    @patch('exatune.hpc.slurm_client.SlurmClient._check_slurm_available')
    def test_cancel_job_success(self, mock_check, mock_run, minimal_slurm_config):
        """Test successful job cancellation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = SlurmClient(minimal_slurm_config, Path(tmpdir))

            # Mock scancel success
            mock_run.return_value = Mock(returncode=0)

            result = client.cancel_job("12345")

            assert result is True

    @patch('subprocess.run')
    @patch('exatune.hpc.slurm_client.SlurmClient._check_slurm_available')
    def test_cancel_job_failure(self, mock_check, mock_run, minimal_slurm_config):
        """Test failed job cancellation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = SlurmClient(minimal_slurm_config, Path(tmpdir))

            # Mock scancel failure (check=True raises CalledProcessError)
            mock_run.side_effect = subprocess.CalledProcessError(1, "scancel")

            result = client.cancel_job("12345")

            assert result is False

    @patch('subprocess.run')
    @patch('exatune.hpc.slurm_client.SlurmClient._check_slurm_available')
    def test_cancel_multiple_jobs(self, mock_check, mock_run, minimal_slurm_config):
        """Test cancelling multiple jobs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = SlurmClient(minimal_slurm_config, Path(tmpdir))

            # Mock scancel success
            mock_run.return_value = Mock(returncode=0)

            count = client.cancel_multiple_jobs(["12345", "12346", "12347"])

            assert count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
