#!/usr/bin/env python3
"""ExaTune visualization — all metrics x all hyperparameter pairs.

Reads hyperparameter names and metrics directly from the experiment YAML.
Produces heatmaps, 3D surfaces (static + interactive), histograms, and
violins for every metric x every param pair combination.

Output structure:
    output_dir/
      f1_macro/
        histogram.png
        violin_<param>.png        (one per hyperparameter)
        <x>_x_<y>/
          heatmap.png
          surface.png
          surface.html
      f1_micro/
        ...
      f1_weighted/
        ...
      precision_macro/
        ...
      recall_macro/
        ...
      roc_auc_ovr/
        ...
      <scoring>/                  (primary metric, e.g. accuracy/)
        ...
      train_vs_test.png           (structural plots at root)
      slice_<param>.png
      ...

Usage:
    python examples/scripts/visualize_results.py \
        --results path/to/results.parquet \
        --config  path/to/experiment.yaml \
        --output-dir ./my_plots
"""

import argparse
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def extract_hyperparameter_names(config: dict) -> list[str]:
    return list(config.get("hyperparameters", {}).keys())


def extract_metric_pairs(config: dict) -> list[tuple[str, str]]:
    """
    Build (parquet_column, folder_name) for every metric.

    Handles both column naming conventions so the script works with
    grid search parquet (dot notation) AND random/Bayesian search parquet
    (underscore notation). The actual column present in the file wins.

    Returns pairs in the same order as your existing visualize_my_results.py:
      additional metrics first, then primary metric last.
    """
    evaluation  = config.get("evaluation", {})
    scoring     = evaluation.get("scoring", "accuracy")
    additional  = evaluation.get("additional_metrics") or []

    # additional metrics — folder name is the metric name itself
    pairs = []
    for m in additional:
        pairs.append((m, f"{m}_mean", f"{m}.mean"))
        # (folder_name, underscore_col, dot_col)

    # primary metric last (matches your existing script ordering)
    pairs.append((scoring, "mean_score", "mean_score"))

    return pairs  # list of (folder_name, underscore_col, dot_col)


def resolve_column(df: pd.DataFrame, underscore_col: str, dot_col: str) -> str | None:
    """Return whichever column name actually exists in df, or None."""
    if underscore_col in df.columns:
        return underscore_col
    if dot_col in df.columns:
        return dot_col
    return None


def safe_plot(label: str, func):
    try:
        print(f"    {label}...")
        func()
        plt.close("all")
    except Exception as e:
        print(f"      ERROR: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ExaTune visualizer — all metrics x all param pairs"
    )
    parser.add_argument("--results",    required=True,
                        help="Path to results.parquet")
    parser.add_argument("--config",     default=None,
                        help="Path to experiment YAML config")
    parser.add_argument("--output-dir", default="./visualization_output",
                        help="Root output directory (default: ./visualization_output)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ config
    if args.config:
        config       = load_yaml_config(Path(args.config))
        param_names  = extract_hyperparameter_names(config)
        metric_triples = extract_metric_pairs(config)
        print(f"Config : {args.config}")
    else:
        print("No --config provided — using fallback params.")
        param_names    = ["n_estimators", "max_depth", "learning_rate"]
        metric_triples = [
            ("f1_macro",       "f1_macro_mean",       "f1_macro.mean"),
            ("f1_micro",       "f1_micro_mean",       "f1_micro.mean"),
            ("f1_weighted",    "f1_weighted_mean",    "f1_weighted.mean"),
            ("precision_macro","precision_macro_mean","precision_macro.mean"),
            ("recall_macro",   "recall_macro_mean",   "recall_macro.mean"),
            ("roc_auc_ovr",    "roc_auc_ovr_mean",    "roc_auc_ovr.mean"),
            ("mean_score",     "mean_score",          "mean_score"),
        ]

    param_pairs = list(itertools.combinations(param_names, 2))

    print(f"  Hyperparameters : {param_names}")
    print(f"  Param pairs     : {len(param_pairs)}")
    print(f"  Metrics         : {[f for f, _, _ in metric_triples]}")

    # ------------------------------------------------------------------ data
    print(f"\nLoading: {args.results}")
    df = pd.read_parquet(args.results)

    # Normalise column names — support both old and new formats
    df.columns = [
        c.replace("hyperparameters.", "")
         .replace("additional_metrics.", "")
        for c in df.columns
    ]
    df = df.drop(columns=["cv_scores", "train_scores"], errors="ignore")

    # Drop failed rows
    if "success" in df.columns:
        n_before = len(df)
        df = df[df["success"] == True].copy()
        dropped = n_before - len(df)
        if dropped:
            print(f"  Dropped {dropped} failed rows")

    print(f"  Rows   : {len(df)}")
    print(f"  Columns: {list(df.columns)}\n")

    # Resolve which column name actually exists for each metric
    available = []  # (folder_name, actual_col)
    for folder_name, underscore_col, dot_col in metric_triples:
        col = resolve_column(df, underscore_col, dot_col)
        if col and df[col].notna().any():
            available.append((folder_name, col))
        else:
            print(f"  SKIP '{folder_name}' — neither '{underscore_col}' nor "
                  f"'{dot_col}' found / all-NaN")

    if not available:
        print("ERROR: no valid metric columns found. Exiting.")
        return

    primary_col = available[-1][1]  # last entry = primary metric
    print(f"  Plotting {len(available)} metrics: {[f for f, _ in available]}\n")

    # ------------------------------------------------------------------ imports
    from exatune.visualization import (
        plot_all_slices,
        plot_conditional_slice,
        plot_contour,
        plot_dashboard,
        plot_heatmap,
        plot_importance,
        plot_interaction_heatmap,
        plot_parallel_coordinates,
        plot_score_histogram,
        plot_score_violin,
        plot_slice,
        plot_surface,
        plot_train_vs_test,
    )
    from exatune.visualization.landscape_3d import plot_surface_plotly

    # ------------------------------------------------------------------ per-metric
    print("=" * 60)
    print("  Per-metric plots")
    print("=" * 60)

    for folder_name, metric_col in available:
        metric_dir = output_dir / folder_name
        metric_dir.mkdir(exist_ok=True)
        print(f"\n  [{folder_name}]  col={metric_col}")

        # Histogram
        safe_plot(
            "histogram",
            lambda m=metric_col, d=metric_dir: plot_score_histogram(
                df, metric=m, output_path=d / "histogram.png"
            ),
        )

        # Violin — one per hyperparameter
        for param in param_names:
            safe_plot(
                f"violin by {param}",
                lambda m=metric_col, d=metric_dir, p=param: plot_score_violin(
                    df, group_by=p, metric=m,
                    output_path=d / f"violin_{p}.png"
                ),
            )

        # Per param-pair subfolder: heatmap + static surface + Plotly surface
        for x_param, y_param in param_pairs:
            pair_dir = metric_dir / f"{x_param}_x_{y_param}"
            pair_dir.mkdir(exist_ok=True)

            safe_plot(
                f"heatmap  {x_param} x {y_param}",
                lambda m=metric_col, d=pair_dir, x=x_param, y=y_param: plot_heatmap(
                    df, x_param=x, y_param=y, metric=m,
                    output_path=d / "heatmap.png"
                ),
            )
            safe_plot(
                f"surface  {x_param} x {y_param}",
                lambda m=metric_col, d=pair_dir, x=x_param, y=y_param: plot_surface(
                    df, x_param=x, y_param=y, metric=m,
                    output_path=d / "surface.png"
                ),
            )
            safe_plot(
                f"surface (plotly)  {x_param} x {y_param}",
                lambda m=metric_col, d=pair_dir, x=x_param, y=y_param: plot_surface_plotly(
                    df, x_param=x, y_param=y, metric=m,
                    output_path=d / "surface.html"
                ),
            )

    # ------------------------------------------------------------------ structural
    print(f"\n{'=' * 60}")
    print(f"  Structural plots  (primary: {primary_col})")
    print("=" * 60)

    safe_plot("train vs test",
              lambda: plot_train_vs_test(df, output_path=output_dir / "train_vs_test.png"))

    for param in param_names:
        safe_plot(f"slice: {param}",
                  lambda p=param: plot_slice(
                      df, param=p, show_individual=True,
                      output_path=output_dir / f"slice_{p}.png"
                  ))

    safe_plot("all slices",
              lambda: plot_all_slices(df, output_path=output_dir / "all_slices.png"))

    if len(param_names) >= 2:
        safe_plot(
            f"conditional slice {param_names[0]} | {param_names[1]}",
            lambda: plot_conditional_slice(
                df, param=param_names[0], condition_param=param_names[1],
                output_path=output_dir / (
                    f"conditional_slice_{param_names[0]}_by_{param_names[1]}.png"
                ),
            ),
        )

    for x_param, y_param in param_pairs:
        safe_plot(
            f"contour {x_param} x {y_param}",
            lambda x=x_param, y=y_param: plot_contour(
                df, x_param=x, y_param=y,
                output_path=output_dir / f"contour_{x}_x_{y}.png"
            ),
        )

    safe_plot("importance (variance)",
              lambda: plot_importance(df, method="variance",
                                      output_path=output_dir / "importance_variance.png"))
    safe_plot("importance (correlation)",
              lambda: plot_importance(df, method="correlation",
                                      output_path=output_dir / "importance_correlation.png"))
    safe_plot("interaction heatmap",
              lambda: plot_interaction_heatmap(df, output_path=output_dir / "interactions.png"))
    safe_plot("parallel coordinates",
              lambda: plot_parallel_coordinates(df, highlight_best=10,
                                                output_path=output_dir / "parallel.png"))
    safe_plot("dashboard",
              lambda: plot_dashboard(df, output_path=output_dir / "dashboard.png"))

    # ------------------------------------------------------------------ summary
    print(f"\n{'=' * 60}")
    print(f"  Done — {output_dir}/")
    print(f"{'=' * 60}")
    print(f"\n  Metric folders ({len(available)}):")
    for folder, _ in available:
        print(f"    {folder}/")
        print(f"      histogram.png  +  {len(param_names)} violin plots")
        print(f"      {len(param_pairs)} param-pair subfolders "
              f"(heatmap.png, surface.png, surface.html each)")
    print(f"\n  Structural plots at root: slices, contours, importance, "
          f"interactions, dashboard")


if __name__ == "__main__":
    main()