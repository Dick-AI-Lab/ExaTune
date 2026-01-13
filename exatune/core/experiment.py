"""
Main experiment orchestration for ExaTune.

This module provides the Experiment class which coordinates the entire
hyperparameter search workflow including grid generation, job submission,
monitoring, and result collection.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from exatune.core.config import ExaTuneConfig, load_config
from exatune.core.grid_generator import GridGenerator

console = Console()


class Experiment:
    """
    Main experiment orchestrator for ExaTune hyperparameter search.

    This class coordinates the entire workflow from configuration to results.
    """

    def __init__(self, config: ExaTuneConfig):
        """
        Initialize an experiment from configuration.

        Args:
            config: ExaTune configuration object
        """
        self.config = config
        self.output_dir = Path(config.experiment.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.grid_generator = GridGenerator(config.get_hyperparameter_specs())
        self.results: Optional[pd.DataFrame] = None
        self._job_ids: List[str] = []
        self._metadata: Dict[str, Any] = {}

        # Create experiment subdirectories
        self.jobs_dir = self.output_dir / "jobs"
        self.results_dir = self.output_dir / "results"
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.logs_dir = self.output_dir / "logs"

        for directory in [self.jobs_dir, self.results_dir, self.checkpoints_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Save configuration
        self._save_config()
        self._save_metadata()

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> "Experiment":
        """
        Create an experiment from a configuration file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Experiment instance
        """
        config = load_config(config_path)
        return cls(config)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Experiment":
        """
        Create an experiment from a configuration dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Experiment instance
        """
        config = ExaTuneConfig.from_dict(config_dict)
        return cls(config)

    def _save_config(self) -> None:
        """Save the experiment configuration."""
        config_path = self.output_dir / "config.yaml"
        self.config.to_yaml(config_path)
        console.print(f"[green]Configuration saved to: {config_path}[/green]")

    def _save_metadata(self) -> None:
        """Save experiment metadata."""
        grid_summary = self.grid_generator.get_summary()

        self._metadata = {
            "experiment_name": self.config.experiment.name,
            "description": self.config.experiment.description,
            "tags": self.config.experiment.tags,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": self.config.model.type,
            "model_class": self.config.model.class_name,
            "task": self.config.model.task,
            "n_hyperparameters": grid_summary["n_parameters"],
            "total_combinations": grid_summary["total_combinations"],
            "dataset_name": self.config.dataset.name,
            "cv_folds": self.config.evaluation.cv_folds,
            "scoring": self.config.evaluation.scoring,
        }

        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self._metadata, f, indent=2)

    def print_summary(self) -> None:
        """Print a summary of the experiment configuration."""
        console.print("\n[bold cyan]ExaTune Experiment Summary[/bold cyan]")
        console.print("=" * 60)

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Experiment Name", self.config.experiment.name)
        table.add_row("Model Type", self.config.model.type)
        table.add_row("Model Class", self.config.model.class_name)
        table.add_row("Task", self.config.model.task)
        table.add_row("Number of Hyperparameters", str(self._metadata["n_hyperparameters"]))
        table.add_row("Total Combinations", f"{self._metadata['total_combinations']:,}")
        table.add_row("CV Folds", str(self.config.evaluation.cv_folds))
        table.add_row("Scoring Metric", self.config.evaluation.scoring)
        table.add_row("Output Directory", str(self.output_dir))

        console.print(table)
        console.print()

        # Print hyperparameter grid summary
        self.grid_generator.print_summary()

    def generate_jobs(self) -> int:
        """
        Generate job scripts for all hyperparameter configurations.

        Returns:
            Number of jobs generated
        """
        console.print("[bold]Generating job scripts...[/bold]")

        n_jobs = 0
        with Progress() as progress:
            task = progress.add_task(
                "[cyan]Creating jobs...", total=self.grid_generator.estimate_grid_size()
            )

            for idx, config in self.grid_generator.generate_grid_iterator():
                job_script = self._create_job_script(idx, config)
                job_path = self.jobs_dir / f"job_{idx:06d}.sh"

                with open(job_path, "w") as f:
                    f.write(job_script)

                n_jobs += 1
                progress.update(task, advance=1)

        console.print(f"[green]Generated {n_jobs:,} job scripts in {self.jobs_dir}[/green]")
        return n_jobs

    def _create_job_script(self, job_id: int, hyperparameters: Dict[str, Any]) -> str:
        """
        Create a SLURM job script for a single hyperparameter configuration.

        Args:
            job_id: Unique job identifier
            hyperparameters: Hyperparameter configuration

        Returns:
            Job script content
        """
        from exatune.hpc.slurm_client import SlurmClient
        import exatune.hpc

        # Get path to worker script
        worker_path = Path(exatune.hpc.__file__).parent / "worker.py"

        # Initialize SLURM client
        slurm_client = SlurmClient(self.config.slurm, self.output_dir)

        # Get random seed for this job
        random_seed = None
        if self.config.experiment.random_seed is not None:
            random_seed = self.config.experiment.random_seed + job_id

        # Get Python environment setup commands
        python_env = getattr(self.config.slurm, "python_environment", None)

        # Generate job script
        return slurm_client.generate_job_script(
            job_id=job_id,
            job_name=f"{self.config.experiment.name}_j{job_id:06d}",
            hyperparameters=hyperparameters,
            config_path=self.output_dir / "config.yaml",
            script_path=worker_path,
            output_dir=self.results_dir,
            log_dir=self.logs_dir,
            random_seed=random_seed,
            python_env=python_env,
        )

    def submit_jobs(self) -> List[str]:
        """
        Submit all jobs to SLURM.

        Returns:
            List of SLURM job IDs

        Note:
            This method requires SLURM to be available on the system.
        """
        from exatune.hpc.slurm_client import SlurmClient

        # Check if SLURM is available
        if not SlurmClient.is_slurm_available():
            console.print(
                "[red]SLURM is not available on this system.[/red]\n"
                "[yellow]Please run this on a SLURM-enabled HPC cluster.[/yellow]"
            )
            return []

        slurm_client = SlurmClient(self.config.slurm, self.output_dir)
        grid_size = self.grid_generator.estimate_grid_size()

        # Strategy: Use job arrays for grids > 50 jobs
        if grid_size > 50:
            console.print(f"[cyan]Using job array for {grid_size} configurations[/cyan]")
            return self._submit_job_array(slurm_client, grid_size)
        else:
            console.print(f"[cyan]Submitting {grid_size} individual jobs[/cyan]")
            return self._submit_individual_jobs(slurm_client, grid_size)

    def _submit_individual_jobs(self, slurm_client, grid_size: int) -> List[str]:
        """
        Submit jobs individually.

        Args:
            slurm_client: SLURM client instance
            grid_size: Number of jobs to submit

        Returns:
            List of SLURM job IDs
        """
        job_ids = []

        with Progress() as progress:
            task = progress.add_task("[cyan]Submitting jobs...", total=grid_size)

            for idx in range(grid_size):
                job_script_path = self.jobs_dir / f"job_{idx:06d}.sh"

                if not job_script_path.exists():
                    console.print(
                        f"[yellow]Warning: Job script not found: {job_script_path}[/yellow]"
                    )
                    progress.update(task, advance=1)
                    continue

                job_id = slurm_client.submit_job(job_script_path)

                if job_id:
                    job_ids.append(job_id)
                    self._job_ids.append(job_id)
                else:
                    console.print(f"[yellow]Warning: Failed to submit job {idx}[/yellow]")

                progress.update(task, advance=1)

        if job_ids:
            self.save_checkpoint("after_submission")
            console.print(f"[green]Successfully submitted {len(job_ids)} jobs[/green]")

        return job_ids

    def _submit_job_array(self, slurm_client, array_size: int) -> List[str]:
        """
        Submit jobs as a SLURM job array.

        Args:
            slurm_client: SLURM client instance
            array_size: Size of job array

        Returns:
            List with single job array ID
        """
        # For job arrays, we need a master script that uses SLURM_ARRAY_TASK_ID
        # For now, fall back to individual job submission
        # TODO: Implement proper job array support in future version
        console.print(
            "[yellow]Job arrays not yet fully implemented, using individual submission[/yellow]"
        )
        return self._submit_individual_jobs(slurm_client, array_size)

    def monitor_progress(self) -> None:
        """
        Monitor the progress of submitted jobs.

        Note:
            This method requires SLURM to be available on the system.
        """
        from exatune.hpc.slurm_client import SlurmClient
        from rich.live import Live

        # Load job IDs from checkpoint if not in memory
        if not self._job_ids:
            try:
                self.load_checkpoint("after_submission")
            except FileNotFoundError:
                console.print(
                    "[yellow]No checkpoint found. Make sure jobs have been submitted.[/yellow]"
                )
                return

        if not self._job_ids:
            console.print("[yellow]No jobs to monitor[/yellow]")
            return

        # Check if SLURM is available
        if not SlurmClient.is_slurm_available():
            console.print(
                "[red]SLURM is not available on this system.[/red]\n"
                "[yellow]Cannot monitor job status.[/yellow]"
            )
            return

        slurm_client = SlurmClient(self.config.slurm, self.output_dir)

        console.print(f"\n[bold]Monitoring {len(self._job_ids)} jobs...[/bold]")
        console.print("[dim]Press Ctrl+C to stop monitoring[/dim]\n")

        def generate_status_table(status_counts: Dict[str, int]) -> Table:
            """Generate a Rich table showing job status."""
            table = Table(title="Job Status", show_header=True, header_style="bold cyan")
            table.add_column("Status", style="cyan", width=15)
            table.add_column("Count", justify="right", style="white")
            table.add_column("Percentage", justify="right", style="white")

            total = sum(status_counts.values())

            for status in ["PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED", "UNKNOWN"]:
                count = status_counts.get(status, 0)
                percentage = (count / total * 100) if total > 0 else 0

                # Color based on status
                if status == "COMPLETED":
                    style = "green"
                elif status == "FAILED" or status == "CANCELLED":
                    style = "red"
                elif status == "RUNNING":
                    style = "yellow"
                else:
                    style = "white"

                table.add_row(status, str(count), f"{percentage:.1f}%", style=style)

            table.add_row("─" * 15, "─" * 5, "─" * 10, style="dim")
            table.add_row("TOTAL", str(total), "100.0%", style="bold")

            return table

        try:
            # Monitor with live updating display
            all_complete = False
            status_counts = {}

            while not all_complete:
                # Query job status
                status_dict = slurm_client.get_multiple_job_status(self._job_ids)

                # Count by status
                status_counts = {
                    "PENDING": 0,
                    "RUNNING": 0,
                    "COMPLETED": 0,
                    "FAILED": 0,
                    "CANCELLED": 0,
                    "UNKNOWN": 0,
                }

                for job_id in self._job_ids:
                    status = status_dict.get(job_id, "UNKNOWN")
                    # Normalize status names
                    if "COMPLET" in status:
                        status = "COMPLETED"
                    elif "FAIL" in status:
                        status = "FAILED"
                    elif "CANCEL" in status:
                        status = "CANCELLED"
                    elif "RUN" in status:
                        status = "RUNNING"
                    elif "PEND" in status:
                        status = "PENDING"
                    else:
                        status = "UNKNOWN"

                    status_counts[status] = status_counts.get(status, 0) + 1

                # Display table
                console.print(generate_status_table(status_counts))

                # Check if all jobs are complete
                terminal_states = ["COMPLETED", "FAILED", "CANCELLED"]
                all_complete = all(
                    status_dict.get(job_id, "UNKNOWN") in terminal_states
                    or "COMPLET" in status_dict.get(job_id, "")
                    or "FAIL" in status_dict.get(job_id, "")
                    or "CANCEL" in status_dict.get(job_id, "")
                    for job_id in self._job_ids
                )

                if not all_complete:
                    # Wait before next poll
                    time.sleep(30)
                    # Clear previous output
                    console.clear()
                else:
                    break

            # Final summary
            console.print("\n[bold]Job Monitoring Complete[/bold]")
            console.print(f"[green]Completed: {status_counts['COMPLETED']}[/green]")
            console.print(f"[red]Failed: {status_counts['FAILED']}[/red]")
            console.print(f"[yellow]Cancelled: {status_counts['CANCELLED']}[/yellow]")

        except KeyboardInterrupt:
            console.print("\n\n[yellow]Monitoring interrupted by user[/yellow]")
            console.print("Jobs will continue running in the background.")

    def collect_results(self) -> pd.DataFrame:
        """
        Collect and aggregate results from completed jobs.

        Returns:
            DataFrame with all results
        """
        console.print("[bold]Collecting results...[/bold]")

        # Scan for result files
        results_list = self._collect_json_results()

        if not results_list:
            console.print("[yellow]No results found[/yellow]")
            console.print(f"[dim]Searched in: {self.results_dir}[/dim]")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(results_list)

        # Validate and enrich
        df = self._validate_and_enrich_results(df)

        # Store in instance
        self.results = df

        # Report statistics
        total = len(df)
        successful = df["success"].sum()
        failed = total - successful

        console.print(f"\n[green]Collected {total} results[/green]")
        console.print(f"[green]Successful: {successful} ({successful/total*100:.1f}%)[/green]")
        if failed > 0:
            console.print(f"[red]Failed: {failed} ({failed/total*100:.1f}%)[/red]")

        return df

    def _collect_json_results(self) -> List[Dict[str, Any]]:
        """
        Collect results from JSON files.

        Returns:
            List of result dictionaries
        """
        results_list = []

        # Find all result JSON files
        result_files = sorted(self.results_dir.glob("result_*.json"))

        if not result_files:
            return results_list

        with Progress() as progress:
            task = progress.add_task(
                "[cyan]Loading results...", total=len(result_files)
            )

            for result_file in result_files:
                try:
                    with open(result_file, "r") as f:
                        result = json.load(f)
                        results_list.append(result)
                except Exception as e:
                    console.print(
                        f"[yellow]Warning: Failed to load {result_file.name}: {e}[/yellow]"
                    )

                progress.update(task, advance=1)

        return results_list

    def _validate_and_enrich_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and enrich results DataFrame.

        Args:
            df: Raw results DataFrame

        Returns:
            Enriched DataFrame
        """
        # Check for missing jobs
        expected_jobs = self.grid_generator.estimate_grid_size()
        actual_jobs = len(df)

        if actual_jobs < expected_jobs:
            missing = expected_jobs - actual_jobs
            console.print(
                f"[yellow]Warning: {missing} jobs missing "
                f"({missing/expected_jobs*100:.1f}%)[/yellow]"
            )

        # Sort by mean score (descending for successful jobs)
        if "mean_score" in df.columns:
            # Put successful jobs first, sorted by score
            df["sort_key"] = df.apply(
                lambda row: (
                    -row["mean_score"] if row["success"] and row["mean_score"] is not None
                    else float("inf")
                ),
                axis=1
            )
            df = df.sort_values("sort_key").drop(columns=["sort_key"])

        # Add best_score flag
        if "mean_score" in df.columns and df["success"].any():
            successful_df = df[df["success"]]
            if not successful_df.empty:
                best_score = successful_df["mean_score"].max()
                df["best_score"] = (df["mean_score"] == best_score) & df["success"]
            else:
                df["best_score"] = False
        else:
            df["best_score"] = False

        # Add rank column (for successful jobs only)
        if "mean_score" in df.columns:
            df["rank"] = None
            successful_mask = df["success"]
            if successful_mask.any():
                df.loc[successful_mask, "rank"] = (
                    df[successful_mask]["mean_score"]
                    .rank(ascending=False, method="min")
                    .astype(int)
                )

        return df

    def load_results(self) -> Optional[pd.DataFrame]:
        """
        Load previously collected results.

        Returns:
            DataFrame with results or None if no results found
        """
        results_path = self.results_dir / "all_results.parquet"
        if results_path.exists():
            self.results = pd.read_parquet(results_path)
            console.print(f"[green]Loaded {len(self.results)} results from {results_path}[/green]")
            return self.results
        else:
            console.print(f"[yellow]No results found at {results_path}[/yellow]")
            return None

    def save_results(self, results: pd.DataFrame) -> None:
        """
        Save results to disk.

        Args:
            results: DataFrame with experiment results
        """
        # Save as Parquet (efficient)
        parquet_path = self.results_dir / "all_results.parquet"
        results.to_parquet(parquet_path, index=False)
        console.print(f"[green]Results saved to {parquet_path}[/green]")

        # Also save as CSV for human readability
        csv_path = self.results_dir / "all_results.csv"
        results.to_csv(csv_path, index=False)
        console.print(f"[green]Results saved to {csv_path}[/green]")

    def get_best_config(self, metric: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the best hyperparameter configuration.

        Args:
            metric: Metric to optimize (uses default scoring if None)

        Returns:
            Best hyperparameter configuration

        Raises:
            ValueError: If no results are available
        """
        if self.results is None:
            raise ValueError("No results available. Run collect_results() first.")

        if metric is None:
            metric = self.config.evaluation.scoring

        # Assume higher is better (may need to be configurable)
        best_idx = self.results[metric].idxmax()
        best_row = self.results.iloc[best_idx]

        # Extract hyperparameter columns
        hyperparam_cols = [
            col for col in self.results.columns if col in self.config.hyperparameters.keys()
        ]

        best_config = best_row[hyperparam_cols].to_dict()
        best_score = best_row[metric]

        console.print(f"\n[bold green]Best Configuration (metric={metric}):[/bold green]")
        console.print(f"Score: {best_score:.4f}")
        console.print("Hyperparameters:")
        for param, value in best_config.items():
            console.print(f"  {param}: {value}")

        return best_config

    def save_checkpoint(self, checkpoint_name: str = "latest") -> None:
        """
        Save an experiment checkpoint.

        Args:
            checkpoint_name: Name for this checkpoint
        """
        checkpoint_data = {
            "metadata": self._metadata,
            "job_ids": self._job_ids,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        checkpoint_path = self.checkpoints_dir / f"{checkpoint_name}.json"
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        console.print(f"[green]Checkpoint saved to {checkpoint_path}[/green]")

    def load_checkpoint(self, checkpoint_name: str = "latest") -> None:
        """
        Load an experiment checkpoint.

        Args:
            checkpoint_name: Name of checkpoint to load
        """
        checkpoint_path = self.checkpoints_dir / f"{checkpoint_name}.json"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        with open(checkpoint_path, "r") as f:
            checkpoint_data = json.load(f)

        self._job_ids = checkpoint_data.get("job_ids", [])
        console.print(f"[green]Checkpoint loaded from {checkpoint_path}[/green]")

    @property
    def name(self) -> str:
        """Get the experiment name."""
        return self.config.experiment.name

    @property
    def grid_size(self) -> int:
        """Get the total number of hyperparameter combinations."""
        return self.grid_generator.estimate_grid_size()

    def __repr__(self) -> str:
        """String representation of the experiment."""
        return (
            f"Experiment(name='{self.name}', "
            f"model={self.config.model.class_name}, "
            f"grid_size={self.grid_size:,})"
        )
