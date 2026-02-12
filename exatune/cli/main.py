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
@click.option("--dry-run", is_flag=True, help="Generate jobs without submitting them")
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
        experiment.generate_jobs()

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
    help="Experiment output directory",
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
    help="Experiment output directory",
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
@click.option("--params", multiple=True, help="Parameters to visualize (specify multiple times)")
@click.option("--metric", default="mean_score", help="Metric to visualize")
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("./results"),
    help="Experiment output directory",
)
@click.option("--output", type=click.Path(path_type=Path), help="Output file for visualization")
@click.option(
    "--type",
    "plot_type",
    type=click.Choice(
        [
            "heatmap",
            "surface",
            "contour",
            "slice",
            "slices",
            "importance",
            "histogram",
            "violin",
            "parallel",
            "train-vs-test",
            "dashboard",
        ]
    ),
    default="dashboard",
    help="Type of visualization to generate",
)
@click.option(
    "--agg",
    type=click.Choice(["mean", "max", "min", "std"]),
    default="mean",
    help="Aggregation function for heatmap/surface/contour",
)
def visualize(
    experiment_name: str,
    params: tuple,
    metric: str,
    output_dir: Path,
    output: Optional[Path],
    plot_type: str,
    agg: str,
) -> None:
    """
    Generate visualizations for experiment results.

    EXPERIMENT_NAME: Name of the experiment
    """
    import matplotlib

    matplotlib.use("Agg")

    exp_dir = output_dir / experiment_name

    if not exp_dir.exists():
        console.print(f"[red]Experiment not found: {experiment_name}[/red]")
        raise click.Abort()

    console.print(f"[bold]Generating visualizations for: {experiment_name}[/bold]\n")

    try:
        # Load results
        results = _load_experiment_results(exp_dir)
        param_list = list(params) if params else None

        from exatune.visualization import (
            plot_all_slices,
            plot_contour,
            plot_dashboard,
            plot_heatmap,
            plot_importance,
            plot_parallel_coordinates,
            plot_score_histogram,
            plot_score_violin,
            plot_slice,
            plot_surface,
            plot_train_vs_test,
        )

        # Default output path
        if output is None:
            viz_dir = exp_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            output = viz_dir / f"{plot_type}.png"

        if plot_type == "dashboard":
            plot_dashboard(results, metric=metric, params=param_list, output_path=output)
        elif plot_type == "histogram":
            plot_score_histogram(results, metric=metric, output_path=output)
        elif plot_type == "importance":
            plot_importance(results, metric=metric, params=param_list, output_path=output)
        elif plot_type == "slices":
            plot_all_slices(results, params=param_list, metric=metric, output_path=output)
        elif plot_type == "parallel":
            plot_parallel_coordinates(results, params=param_list, metric=metric, output_path=output)
        elif plot_type == "train-vs-test":
            plot_train_vs_test(results, metric=metric, output_path=output)
        elif plot_type == "slice":
            if not param_list:
                console.print("[red]--params required for slice plot[/red]")
                raise click.Abort()
            plot_slice(results, param=param_list[0], metric=metric, output_path=output)
        elif plot_type == "violin":
            if not param_list:
                console.print("[red]--params required for violin plot[/red]")
                raise click.Abort()
            plot_score_violin(results, group_by=param_list[0], metric=metric, output_path=output)
        elif plot_type in ("heatmap", "surface", "contour"):
            if not param_list or len(param_list) < 2:
                console.print("[red]--params requires 2 parameters for heatmap/surface/contour[/red]")
                raise click.Abort()
            plot_func = {"heatmap": plot_heatmap, "surface": plot_surface, "contour": plot_contour}
            plot_func[plot_type](
                results,
                x_param=param_list[0],
                y_param=param_list[1],
                metric=metric,
                agg_func=agg,
                output_path=output,
            )

        import matplotlib.pyplot as plt

        plt.close("all")

        console.print(f"[green]Visualization saved to: {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@cli.command()
@click.argument("experiment_name")
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("./results"),
    help="Experiment output directory",
)
@click.option(
    "--metrics",
    multiple=True,
    default=["all"],
    help="Landscape metrics to compute (smoothness, multimodality, importance, interactions, all)",
)
@click.option("--metric", default="mean_score", help="Performance metric to analyze")
@click.option(
    "--direction",
    type=click.Choice(["maximize", "minimize"]),
    default="maximize",
    help="Optimization direction",
)
@click.option(
    "--report",
    type=click.Path(path_type=Path),
    help="Generate full report to this directory",
)
def analyze(
    experiment_name: str,
    output_dir: Path,
    metrics: tuple,
    metric: str,
    direction: str,
    report: Optional[Path],
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

    try:
        results = _load_experiment_results(exp_dir)

        from exatune.analysis.summary import compute_experiment_summary, format_summary_text

        metrics_set = set(metrics)
        run_all = "all" in metrics_set

        summary = compute_experiment_summary(results, metric, direction=direction)

        # Print formatted summary to console
        console.print(format_summary_text(summary))

        # Generate full report if requested
        if report is not None:
            from exatune.analysis.report import generate_report

            report_path = generate_report(
                results,
                output_path=report,
                metric=metric,
                direction=direction,
                include_plots=True,
            )
            console.print(f"\n[green]Full report saved to: {report_path}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


def _load_experiment_results(exp_dir: Path) -> "pd.DataFrame":
    """Load experiment results from parquet or CSV."""
    import pandas as pd

    results_dir = exp_dir / "results"

    # Try parquet first, then CSV
    parquet_path = results_dir / "all_results.parquet"
    csv_path = results_dir / "all_results.csv"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    elif csv_path.exists():
        return pd.read_csv(csv_path)
    else:
        # Try to collect from individual JSON files
        from exatune.storage.json_backend import JSONStorage

        storage = JSONStorage(str(results_dir))
        return storage.load_all_results()


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
        console.print("[red]✗ Configuration is invalid[/red]")
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@cli.command()
def version() -> None:
    """Show ExaTune version information."""
    console.print(f"[bold]ExaTune version {__version__}[/bold]")
    console.print("Exhaustive Hyperparameter Landscape Exploration via HPC")


if __name__ == "__main__":
    cli()
