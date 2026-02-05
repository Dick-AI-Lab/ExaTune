"""
SLURM client for job submission and management.

This module provides functionality to interact with SLURM HPC clusters
for submitting and managing hyperparameter search jobs.
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Template

from exatune.core.config import SlurmConfig


class SlurmClient:
    """
    Client for interacting with SLURM workload manager.

    Handles job submission, status monitoring, and management.
    """

    def __init__(self, config: SlurmConfig, work_dir: Path):
        """
        Initialize SLURM client.

        Args:
            config: SLURM configuration
            work_dir: Working directory for the experiment
        """
        self.config = config
        self.work_dir = Path(work_dir)
        self.template_dir = Path(__file__).parent / "templates"
        self._check_slurm_available()

    def _check_slurm_available(self) -> None:
        """
        Check if SLURM commands are available.

        Raises:
            RuntimeError: If SLURM is not available
        """
        try:
            subprocess.run(["sbatch", "--version"], capture_output=True, check=True, timeout=5)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            raise RuntimeError(
                "SLURM is not available. Ensure you are on a SLURM-enabled system "
                "or use local execution mode."
            )

    def generate_job_script(
        self,
        job_id: int,
        job_name: str,
        hyperparameters: Dict[str, Any],
        config_path: Path,
        script_path: Path,
        output_dir: Path,
        log_dir: Path,
        random_seed: Optional[int] = None,
        python_env: Optional[str] = None,
    ) -> str:
        """
        Generate a SLURM job script from template.

        Args:
            job_id: Unique job identifier
            job_name: Human-readable job name
            hyperparameters: Hyperparameter configuration
            config_path: Path to experiment configuration
            script_path: Path to Python training script
            output_dir: Directory for job outputs
            log_dir: Directory for SLURM logs
            random_seed: Random seed for reproducibility
            python_env: Commands to activate Python environment (e.g., "module load python/3.9")

        Returns:
            Rendered job script content
        """
        template_path = self.template_dir / "job_template.sh"

        with open(template_path, "r") as f:
            template = Template(f.read())

        # Generate configuration hash
        config_hash = self._generate_config_hash(hyperparameters)

        # Serialize hyperparameters to JSON
        hyperparameters_json = json.dumps(hyperparameters)

        # Build python_env from config if not explicitly provided
        if python_env is None and self.config.python_environment:
            python_env = self.config.python_environment

        # Render template
        script_content = template.render(
            job_id=job_id,
            job_name=job_name,
            partition=self.config.partition,
            time=self.config.time,
            memory=self.config.memory,
            cpus_per_task=self.config.cpus_per_task,
            nodes=self.config.nodes,
            account=self.config.account,
            qos=self.config.qos,
            email=self.config.email,
            email_type=self.config.email_type,
            additional_directives=self.config.additional_directives,
            modules=self.config.modules,
            config_hash=config_hash,
            hyperparameters_json=hyperparameters_json,
            config_path=config_path,
            script_path=script_path,
            output_dir=output_dir,
            log_dir=log_dir,
            work_dir=self.work_dir,
            random_seed=random_seed,
            python_env=python_env,
        )

        return script_content

    @staticmethod
    def _generate_config_hash(hyperparameters: Dict[str, Any]) -> str:
        """
        Generate a hash for a hyperparameter configuration.

        Args:
            hyperparameters: Hyperparameter dictionary

        Returns:
            Hash string
        """
        import hashlib

        config_str = json.dumps(hyperparameters, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    def submit_job(self, job_script_path: Path) -> Optional[str]:
        """
        Submit a job script to SLURM.

        Args:
            job_script_path: Path to the job script

        Returns:
            SLURM job ID or None if submission failed
        """
        try:
            result = subprocess.run(
                ["sbatch", str(job_script_path)],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )

            # Parse job ID from output (typically: "Submitted batch job 12345")
            match = re.search(r"Submitted batch job (\d+)", result.stdout)
            if match:
                return match.group(1)
            else:
                print(f"Warning: Could not parse job ID from: {result.stdout}")
                return None

        except subprocess.CalledProcessError as e:
            print(f"Error submitting job: {e.stderr}")
            return None
        except subprocess.TimeoutExpired:
            print("Error: sbatch command timed out")
            return None

    def submit_job_array(
        self, job_script_path: Path, array_size: int, max_concurrent: Optional[int] = None
    ) -> Optional[str]:
        """
        Submit a job array to SLURM.

        Args:
            job_script_path: Path to the job script template
            array_size: Number of array tasks
            max_concurrent: Maximum number of concurrent tasks

        Returns:
            SLURM job array ID or None if submission failed
        """
        array_spec = f"0-{array_size-1}"
        if max_concurrent is not None:
            array_spec += f"%{max_concurrent}"

        try:
            result = subprocess.run(
                ["sbatch", f"--array={array_spec}", str(job_script_path)],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )

            match = re.search(r"Submitted batch job (\d+)", result.stdout)
            if match:
                return match.group(1)
            else:
                print(f"Warning: Could not parse job ID from: {result.stdout}")
                return None

        except subprocess.CalledProcessError as e:
            print(f"Error submitting job array: {e.stderr}")
            return None
        except subprocess.TimeoutExpired:
            print("Error: sbatch command timed out")
            return None

    def get_job_status(self, job_id: str) -> Optional[str]:
        """
        Get the status of a SLURM job.

        Args:
            job_id: SLURM job ID

        Returns:
            Job status string or None if query failed
            Common statuses: PENDING, RUNNING, COMPLETED, FAILED, CANCELLED
        """
        try:
            result = subprocess.run(
                ["squeue", "-j", job_id, "-h", "-o", "%T"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            status = result.stdout.strip()
            if status:
                return status
            else:
                # Job not in queue, check sacct for completed jobs
                return self._get_completed_job_status(job_id)

        except subprocess.TimeoutExpired:
            print(f"Warning: squeue timed out for job {job_id}")
            return None

    def _get_completed_job_status(self, job_id: str) -> Optional[str]:
        """
        Get status of a completed job from sacct.

        Args:
            job_id: SLURM job ID

        Returns:
            Job status or None
        """
        try:
            result = subprocess.run(
                ["sacct", "-j", job_id, "-n", "-o", "State"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            lines = result.stdout.strip().split("\n")
            if lines:
                # Return the first status (job status, not job steps)
                return lines[0].strip()

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass

        return None

    def get_multiple_job_status(self, job_ids: List[str]) -> Dict[str, str]:
        """
        Get status of multiple jobs efficiently.

        Args:
            job_ids: List of SLURM job IDs

        Returns:
            Dictionary mapping job ID to status
        """
        status_dict = {}

        # Query squeue for all jobs at once
        try:
            result = subprocess.run(
                ["squeue", "-h", "-o", "%i %T"], capture_output=True, text=True, timeout=15
            )

            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        job_id, status = parts[0], parts[1]
                        if job_id in job_ids:
                            status_dict[job_id] = status

        except subprocess.TimeoutExpired:
            print("Warning: squeue timed out")

        # For jobs not found in queue, check sacct
        missing_jobs = set(job_ids) - set(status_dict.keys())
        if missing_jobs:
            for job_id in missing_jobs:
                status = self._get_completed_job_status(job_id)
                if status:
                    status_dict[job_id] = status

        return status_dict

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a SLURM job.

        Args:
            job_id: SLURM job ID to cancel

        Returns:
            True if cancellation succeeded
        """
        try:
            subprocess.run(["scancel", job_id], capture_output=True, check=True, timeout=10)
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False

    def cancel_multiple_jobs(self, job_ids: List[str]) -> int:
        """
        Cancel multiple SLURM jobs.

        Args:
            job_ids: List of SLURM job IDs to cancel

        Returns:
            Number of jobs successfully cancelled
        """
        count = 0
        for job_id in job_ids:
            if self.cancel_job(job_id):
                count += 1
        return count

    def get_job_info(self, job_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a job.

        Args:
            job_id: SLURM job ID

        Returns:
            Dictionary with job information
        """
        try:
            result = subprocess.run(
                ["scontrol", "show", "job", job_id], capture_output=True, text=True, timeout=10
            )

            # Parse scontrol output (key=value format)
            info = {}
            for line in result.stdout.split("\n"):
                for item in line.split():
                    if "=" in item:
                        key, value = item.split("=", 1)
                        info[key] = value

            return info

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return {}

    @staticmethod
    def is_slurm_available() -> bool:
        """
        Check if SLURM is available on the system.

        Returns:
            True if SLURM commands are available
        """
        try:
            subprocess.run(["sbatch", "--version"], capture_output=True, check=True, timeout=5)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
