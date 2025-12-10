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
                "[cyan]Creating jobs...",
                total=self.grid_generator.estimate_grid_size()
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
        # This will be implemented with the SLURM integration
        # For now, return a placeholder
        return f"#!/bin/bash\n# Job {job_id}\n# Hyperparameters: {hyperparameters}\n"

    def submit_jobs(self) -> List[str]:
        """
        Submit all jobs to SLURM.

        Returns:
            List of SLURM job IDs

        Note:
            This method requires SLURM integration to be implemented.
        """
        console.print("[yellow]Job submission requires SLURM integration (not yet implemented)[/yellow]")
        return []

    def monitor_progress(self) -> None:
        """
        Monitor the progress of submitted jobs.

        Note:
            This method requires SLURM integration to be implemented.
        """
        console.print("[yellow]Job monitoring requires SLURM integration (not yet implemented)[/yellow]")

    def collect_results(self) -> pd.DataFrame:
        """
        Collect and aggregate results from completed jobs.

        Returns:
            DataFrame with all results

        Note:
            This method requires result collection implementation.
        """
        console.print("[yellow]Result collection not yet implemented[/yellow]")
        return pd.DataFrame()

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
            col for col in self.results.columns
            if col in self.config.hyperparameters.keys()
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
