"""
Command-line interface for ExaTune.

This module provides the main CLI entry point for ExaTune operations.
"""

from pathlib import Path
from typing import Optional

import click
from rich.console import Console

from exatune import __version__
from exatune.core.experiment import Experiment

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="exatune")
def cli() -> None:
    """
    ExaTune: Exhaustive Hyperparameter Landscape Exploration via HPC.

    A tool for comprehensive hyperparameter search using SLURM-based HPC clusters.
    """
    pass


@cli.command()
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--dry-run",
    is_flag=True,
    help="Generate jobs without submitting them"
)
def run(config_path: Path, dry_run: bool) -> None:
    """
    Run a hyperparameter search experiment.

    CONFIG_PATH: Path to the YAML configuration file
    """
    console.print(f"[bold]Loading configuration from: {config_path}[/bold]\n")

    try:
        experiment = Experiment.from_config(config_path)
        experiment.print_summary()

        # Generate job scripts
        n_jobs = experiment.generate_jobs()

        if dry_run:
            console.print("\n[yellow]Dry run mode: Jobs generated but not submitted[/yellow]")
            console.print(f"Job scripts saved to: {experiment.jobs_dir}")
        else:
            # Submit jobs
            console.print("\n[bold]Submitting jobs to SLURM...[/bold]")
            job_ids = experiment.submit_jobs()

            if job_ids:
                console.print(f"[green]Successfully submitted {len(job_ids)} jobs[/green]")
                experiment.save_checkpoint("after_submission")
            else:
                console.print("[red]Job submission failed or not yet implemented[/red]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@cli.command()
@click.argument("experiment_name")
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("./results"),
    help="Experiment output directory"
)
def status(experiment_name: str, output_dir: Path) -> None:
    """
    Check the status of a running experiment.

    EXPERIMENT_NAME: Name of the experiment to check
    """
    exp_dir = output_dir / experiment_name

    if not exp_dir.exists():
        console.print(f"[red]Experiment not found: {experiment_name}[/red]")
        raise click.Abort()

    console.print(f"[bold]Status for experiment: {experiment_name}[/bold]\n")

    try:
        config_path = exp_dir / "config.yaml"
        experiment = Experiment.from_config(config_path)
        experiment.monitor_progress()

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@cli.command()
@click.argument("experiment_name")
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("./results"),
    help="Experiment output directory"
)
def collect(experiment_name: str, output_dir: Path) -> None:
    """
    Collect results from a completed experiment.

    EXPERIMENT_NAME: Name of the experiment
    """
    exp_dir = output_dir / experiment_name

    if not exp_dir.exists():
        console.print(f"[red]Experiment not found: {experiment_name}[/red]")
        raise click.Abort()

    console.print(f"[bold]Collecting results for: {experiment_name}[/bold]\n")

    try:
        config_path = exp_dir / "config.yaml"
        experiment = Experiment.from_config(config_path)
        results = experiment.collect_results()

        if not results.empty:
            experiment.save_results(results)
            console.print(f"\n[green]Collected {len(results)} results[/green]")

            # Show best configuration
            experiment.get_best_config()
        else:
            console.print("[yellow]No results found[/yellow]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@cli.command()
@click.argument("experiment_name")
@click.option(
    "--params",
    multiple=True,
    help="Parameters to visualize (specify multiple times)"
)
@click.option(
    "--metric",
    default="accuracy",
    help="Metric to visualize"
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("./results"),
    help="Experiment output directory"
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    help="Output file for visualization"
)
def visualize(
    experiment_name: str,
    params: tuple,
    metric: str,
    output_dir: Path,
    output: Optional[Path]
) -> None:
    """
    Generate visualizations for experiment results.

    EXPERIMENT_NAME: Name of the experiment
    """
    exp_dir = output_dir / experiment_name

    if not exp_dir.exists():
        console.print(f"[red]Experiment not found: {experiment_name}[/red]")
        raise click.Abort()

    console.print(f"[bold]Generating visualizations for: {experiment_name}[/bold]\n")
    console.print(f"Parameters: {', '.join(params) if params else 'all'}")
    console.print(f"Metric: {metric}\n")

    console.print("[yellow]Visualization not yet implemented[/yellow]")


@cli.command()
@click.argument("experiment_name")
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("./results"),
    help="Experiment output directory"
)
@click.option(
    "--metrics",
    multiple=True,
    default=["smoothness", "multimodality"],
    help="Landscape metrics to compute"
)
def analyze(
    experiment_name: str,
    output_dir: Path,
    metrics: tuple
) -> None:
    """
    Analyze hyperparameter landscape characteristics.

    EXPERIMENT_NAME: Name of the experiment
    """
    exp_dir = output_dir / experiment_name

    if not exp_dir.exists():
        console.print(f"[red]Experiment not found: {experiment_name}[/red]")
        raise click.Abort()

    console.print(f"[bold]Analyzing landscape for: {experiment_name}[/bold]\n")
    console.print(f"Metrics: {', '.join(metrics)}\n")

    console.print("[yellow]Landscape analysis not yet implemented[/yellow]")


@cli.command()
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
def validate(config_path: Path) -> None:
    """
    Validate an experiment configuration file.

    CONFIG_PATH: Path to the YAML configuration file
    """
    console.print(f"[bold]Validating configuration: {config_path}[/bold]\n")

    try:
        experiment = Experiment.from_config(config_path)
        console.print("[green]✓ Configuration is valid[/green]\n")
        experiment.print_summary()

    except Exception as e:
        console.print(f"[red]✗ Configuration is invalid[/red]")
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@cli.command()
def version() -> None:
    """Show ExaTune version information."""
    console.print(f"[bold]ExaTune version {__version__}[/bold]")
    console.print("Exhaustive Hyperparameter Landscape Exploration via HPC")


if __name__ == "__main__":
    cli()
