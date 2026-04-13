# #!/usr/bin/env python3
# """Demonstrate all ExaTune visualization functions using a synthetic dataset.

# This script generates a synthetic results DataFrame and produces every
# visualization type supported by ExaTune. It can be run without a SLURM
# cluster or any actual experiment data.

# Usage:
#     python examples/scripts/visualize_results.py [--output-dir OUTPUT_DIR]
# """
# #! Copied file to subastute sythetic data with our results

# import argparse
# from pathlib import Path

# import matplotlib

# matplotlib.use("Agg")  # Non-interactive backend

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# # Remove or keep the old one if still used elsewhere:
# # from exatune.visualization import plot_surface
# from exatune.visualization.landscape_3d import plot_surface_plotly

# def main():
#     parser = argparse.ArgumentParser(description="ExaTune Visualization Demo")
#     parser.add_argument(
#         "--output-dir",
#         type=str,
#         default="./visualization_demo",
#         help="Directory for output plots",
#     )
#     parser.add_argument(
#         "--results",
#         type=str,
#         required=True,
#         help="Path to results parquet file",
#     )
#     args = parser.parse_args()

#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     print("Loading results...")
#     results = pd.read_parquet(args.results)
#     results.columns = [c.replace("hyperparameters.", "").replace("additional_metrics.", "") for c in results.columns]
#     results = results.drop(columns=["cv_scores", "train_scores"], errors="ignore")
#     print(f"  Loaded {len(results)} configurations\n")

#     # Import visualization functions
#     from exatune.visualization import (
#         plot_all_slices,
#         plot_conditional_slice,
#         plot_contour,
#         plot_dashboard,
#         plot_heatmap,
#         plot_importance,
#         plot_interaction_heatmap,
#         plot_parallel_coordinates,
#         plot_score_histogram,
#         plot_score_violin,
#         plot_slice,
#         plot_surface,
#         plot_train_vs_test,
       
        
#     )

#     # Columns in the parquet are stored as e.g. "f1_macro.mean" after flattening.
#     # The folder name strips ".mean" so paths stay clean on all OSes.
#     metrics = [
#         "f1_macro.mean",
#         "f1_micro.mean",
#         "f1_weighted.mean",
#         "precision_macro.mean",
#         "recall_macro.mean",
#         "roc_auc_ovr.mean",
#     ]
#     primary = "f1_macro.mean"

#     # --- Per-metric plots ---
#     # These visualize the metric value itself, so running once per metric is meaningful.
#     print("Generating per-metric plots...")
#     for metric in metrics:
#         metric_dir = output_dir / metric.replace(".mean", "")  # e.g. "f1_macro"
#         metric_dir.mkdir(exist_ok=True)

#         per_metric_plots = [
#             (f"Score Histogram ({metric})",
#             lambda m=metric, d=metric_dir: plot_score_histogram(results, metric=m, output_path=d / "histogram.png")),
#             (f"Score Violin by max_depth ({metric})",
#             lambda m=metric, d=metric_dir: plot_score_violin(results, group_by="max_depth", metric=m, output_path=d / "violin.png")),
#             (f"Heatmap n_estimators x max_depth ({metric})",
#             lambda m=metric, d=metric_dir: plot_heatmap(results, x_param="n_estimators", y_param="max_depth", metric=m, output_path=d / "heatmap.png")),
#             (f"3D Surface ({metric})",
#             lambda m=metric, d=metric_dir: plot_surface(results, x_param="n_estimators", y_param="max_depth", metric=m, output_path=d / "surface_n_m.png")),
#             (f"3D Surface ({metric})",
#             lambda m=metric, d=metric_dir: plot_surface(results, x_param="learning_rate", y_param="max_depth", metric=m, output_path=d / "surface_lr_m.png")),
#             (f"3D Surface ({metric})",
#             lambda m=metric, d=metric_dir: plot_surface(results, x_param="n_estimators", y_param="learning_rate", metric=m, output_path=d / "surface_n_lr.png")),
#             (f"3D Surface ({metric})",
#             lambda m=metric, d=metric_dir: plot_surface_plotly(results, x_param="n_estimators", y_param="max_depth", metric=m, output_path=d / "surface_n_estimators_x_max_depth.html")),
#             (f"3D Surface ({metric})",
#             lambda m=metric, d=metric_dir: plot_surface_plotly(results, x_param="learning_rate", y_param="max_depth", metric=m, output_path=d / "surface_learning_rate_x_max_depth.html")),
#             (f"3D Surface ({metric})",
#             lambda m=metric, d=metric_dir: plot_surface_plotly(results, x_param="n_estimators", y_param="learning_rate", metric=m, output_path=d / "surface_n_estimators_x_learning_rate.html")),
#         ]

#         for name, plot_func in per_metric_plots:
#             try:
#                 print(f"  Generating: {name}...")
#                 plot_func()
#                 plt.close("all")
#             except Exception as e:
#                 print(f"    ERROR: {e}")

#     # --- Single plots (structural/relationship plots, primary metric only) ---
#     # These show parameter relationships and landscape structure, which is
#     # stable across metrics. Running once with the primary metric is sufficient.
#     print(f"\nGenerating single plots (primary metric: {primary})...")
#     single_plots = [
#         ("Train vs Test Scatter",
#          lambda: plot_train_vs_test(results, output_path=output_dir / "train_vs_test.png")),
#         ("Slice: n_estimators",
#          lambda: plot_slice(results, param="n_estimators", show_individual=True, output_path=output_dir / "slice_n_estimators.png")),
#         ("All Slices",
#          lambda: plot_all_slices(results, output_path=output_dir / "all_slices.png")),
#         ("Conditional Slice",
#          lambda: plot_conditional_slice(results, param="n_estimators", condition_param="max_depth", output_path=output_dir / "conditional_slice.png")),
#         ("3D Surface",
#          lambda: plot_surface(results, x_param="n_estimators", y_param="max_depth", output_path=output_dir / "surface.png")),
#         ("Contour Plot",
#          lambda: plot_contour(results, x_param="n_estimators", y_param="max_depth", output_path=output_dir / "contour.png")),
#         ("Parameter Importance (variance)",
#          lambda: plot_importance(results, method="variance", output_path=output_dir / "importance_variance.png")),
#         ("Parameter Importance (correlation)",
#          lambda: plot_importance(results, method="correlation", output_path=output_dir / "importance_correlation.png")),
#         ("Interaction Heatmap",
#          lambda: plot_interaction_heatmap(results, output_path=output_dir / "interactions.png")),
#         ("Parallel Coordinates",
#          lambda: plot_parallel_coordinates(results, highlight_best=10, output_path=output_dir / "parallel.png")),
#         ("Dashboard",
#          lambda: plot_dashboard(results, output_path=output_dir / "dashboard.png")),
#     ]

#     for name, plot_func in single_plots:
#         try:
#             print(f"  Generating: {name}...")
#             plot_func()
#             plt.close("all")
#         except Exception as e:
#             print(f"    ERROR: {e}")

#     n_per_metric = len(metrics) * 3  # 3 plot types per metric
#     n_single = len(single_plots)
#     metric_dirs = ", ".join(m.replace(".mean", "") for m in metrics)
#     print(f"\nAll plots saved to: {output_dir}/")
#     print(f"  {n_per_metric} per-metric plots (in subdirs: {metric_dirs})")
#     print(f"  {n_single} single plots (in root output dir)")
#     print(f"  {n_per_metric + n_single} total files generated.")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""Demonstrate all ExaTune visualization functions using actual experiment results.

Hyperparameters and metrics are read directly from the experiment YAML config,
so nothing is hardcoded — the script adapts to whichever experiment you point it at.

Usage:
    python examples/scripts/visualize_results.py \
        --results path/to/results.parquet \
        --config path/to/experiment.yaml \
        [--output-dir OUTPUT_DIR]
"""

import argparse
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml_config(config_path: Path) -> dict:
    """Load raw YAML config (no Pydantic needed)."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def extract_hyperparameter_names(config: dict) -> list[str]:
    """Return the list of hyperparameter names defined in the YAML."""
    return list(config.get("hyperparameters", {}).keys())


def extract_metrics(config: dict) -> tuple[str, list[str]]:
    """
    Return (primary_metric, all_metrics) derived from the evaluation section.

    Parquet columns use the '.mean' suffix after flattening, so we append it here.
    The primary metric is evaluation.scoring; additional_metrics are appended after.
    """
    evaluation = config.get("evaluation", {})
    primary_raw = evaluation.get("scoring", "accuracy")
    additional_raw = evaluation.get("additional_metrics", [])

    # Parquet columns are stored as e.g. "f1_macro.mean" after json_normalize flattening.
    # The primary scoring metric is stored directly as "mean_score", but additional
    # metrics are stored as "<metric>.mean". We handle both.
    primary_col = "mean_score"  # always present regardless of scoring name
    additional_cols = [f"{m}.mean" for m in additional_raw]

    all_metrics = [primary_col] + additional_cols
    return primary_col, all_metrics, primary_raw


def generate_param_pairs(param_names: list[str]) -> list[tuple[str, str]]:
    """Return all unique (x, y) pairs for surface/contour/heatmap plots."""
    return list(itertools.combinations(param_names, 2))


def safe_plot(name: str, func):
    """Run a plot function, catch and report errors without aborting."""
    try:
        print(f"  Generating: {name}...")
        func()
        plt.close("all")
    except Exception as e:
        print(f"    ERROR in '{name}': {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ExaTune Visualization — driven by experiment YAML config"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to results parquet file",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config (used to read hyperparameter names and metrics)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./visualization_output",
        help="Directory for output plots (default: ./visualization_output)",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load config and derive parameters / metrics
    # ------------------------------------------------------------------
    print(f"Loading config from: {config_path}")
    config = load_yaml_config(config_path)

    param_names = extract_hyperparameter_names(config)
    primary_metric, all_metrics, primary_raw = extract_metrics(config)
    param_pairs = generate_param_pairs(param_names)

    print(f"  Hyperparameters ({len(param_names)}): {', '.join(param_names)}")
    print(f"  Primary metric column : {primary_metric}  (scoring = '{primary_raw}')")
    print(f"  Additional metric cols : {', '.join(all_metrics[1:]) if all_metrics[1:] else 'none'}")
    print(f"  Parameter pairs for 2D plots: {len(param_pairs)}\n")

    # ------------------------------------------------------------------
    # 2. Load results and clean column names
    # ------------------------------------------------------------------
    print(f"Loading results from: {results_path}")
    results = pd.read_parquet(results_path)

    # Strip dot-notation prefixes added by json_normalize
    results.columns = [
        c.replace("hyperparameters.", "").replace("additional_metrics.", "")
        for c in results.columns
    ]
    # Drop array-valued columns that break groupby
    results = results.drop(columns=["cv_scores", "train_scores"], errors="ignore")

    print(f"  Loaded {len(results)} configurations")
    print(f"  Columns: {list(results.columns)}\n")

    # Warn about any expected metric columns that are missing
    missing = [m for m in all_metrics if m not in results.columns]
    if missing:
        print(f"  WARNING: These metric columns were not found and will be skipped: {missing}")
        all_metrics = [m for m in all_metrics if m in results.columns]
        if primary_metric not in results.columns:
            # Fall back to the first available metric
            primary_metric = all_metrics[0] if all_metrics else None

    # ------------------------------------------------------------------
    # 3. Import visualization functions
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 4. Per-metric plots (histogram, violin, heatmap, surfaces)
    # ------------------------------------------------------------------
    print("Generating per-metric plots...")
    for metric in all_metrics:
        # Use the part before ".mean" as the folder name (e.g. "f1_macro")
        # primary metric "mean_score" gets its own folder too
        folder_name = metric.replace(".mean", "")
        metric_dir = output_dir / folder_name
        metric_dir.mkdir(exist_ok=True)

        safe_plot(
            f"Score Histogram ({metric})",
            lambda m=metric, d=metric_dir: plot_score_histogram(
                results, metric=m, output_path=d / "histogram.png"
            ),
        )

        # Violin — use first param as grouping axis (usually the most important one)
        if param_names:
            group_param = param_names[0]
            safe_plot(
                f"Score Violin by {group_param} ({metric})",
                lambda m=metric, d=metric_dir, p=group_param: plot_score_violin(
                    results, group_by=p, metric=m, output_path=d / "violin.png"
                ),
            )

        # Heatmap, static surface, and Plotly surface for every param pair
        for x_param, y_param in param_pairs:
            pair_label = f"{x_param}_x_{y_param}"

            safe_plot(
                f"Heatmap {x_param} x {y_param} ({metric})",
                lambda m=metric, d=metric_dir, x=x_param, y=y_param, lbl=pair_label: plot_heatmap(
                    results, x_param=x, y_param=y, metric=m,
                    output_path=d / f"heatmap_{lbl}.png"
                ),
            )

            safe_plot(
                f"3D Surface {x_param} x {y_param} ({metric})",
                lambda m=metric, d=metric_dir, x=x_param, y=y_param, lbl=pair_label: plot_surface(
                    results, x_param=x, y_param=y, metric=m,
                    output_path=d / f"surface_{lbl}.png"
                ),
            )

            safe_plot(
                f"3D Surface Plotly {x_param} x {y_param} ({metric})",
                lambda m=metric, d=metric_dir, x=x_param, y=y_param, lbl=pair_label: plot_surface_plotly(
                    results, x_param=x, y_param=y, metric=m,
                    output_path=d / f"surface_{lbl}.html"
                ),
            )

    # ------------------------------------------------------------------
    # 5. Single structural plots (run once with primary metric)
    # ------------------------------------------------------------------
    print(f"\nGenerating structural plots (primary metric: {primary_metric})...")

    safe_plot(
        "Train vs Test Scatter",
        lambda: plot_train_vs_test(results, output_path=output_dir / "train_vs_test.png"),
    )

    # Per-parameter slice plots
    for param in param_names:
        safe_plot(
            f"Slice: {param}",
            lambda p=param: plot_slice(
                results, param=p, show_individual=True,
                output_path=output_dir / f"slice_{p}.png"
            ),
        )

    safe_plot(
        "All Slices",
        lambda: plot_all_slices(results, output_path=output_dir / "all_slices.png"),
    )

    # Conditional slice — first param conditioned on second param (if ≥2 exist)
    if len(param_names) >= 2:
        safe_plot(
            f"Conditional Slice: {param_names[0]} | {param_names[1]}",
            lambda: plot_conditional_slice(
                results,
                param=param_names[0],
                condition_param=param_names[1],
                output_path=output_dir / f"conditional_slice_{param_names[0]}_by_{param_names[1]}.png",
            ),
        )

    # Contour for every param pair
    for x_param, y_param in param_pairs:
        pair_label = f"{x_param}_x_{y_param}"
        safe_plot(
            f"Contour {x_param} x {y_param}",
            lambda x=x_param, y=y_param, lbl=pair_label: plot_contour(
                results, x_param=x, y_param=y,
                output_path=output_dir / f"contour_{lbl}.png"
            ),
        )

    safe_plot(
        "Parameter Importance (variance)",
        lambda: plot_importance(
            results, method="variance", output_path=output_dir / "importance_variance.png"
        ),
    )
    safe_plot(
        "Parameter Importance (correlation)",
        lambda: plot_importance(
            results, method="correlation", output_path=output_dir / "importance_correlation.png"
        ),
    )
    safe_plot(
        "Interaction Heatmap",
        lambda: plot_interaction_heatmap(results, output_path=output_dir / "interactions.png"),
    )
    safe_plot(
        "Parallel Coordinates",
        lambda: plot_parallel_coordinates(
            results, highlight_best=10, output_path=output_dir / "parallel.png"
        ),
    )
    safe_plot(
        "Dashboard",
        lambda: plot_dashboard(results, output_path=output_dir / "dashboard.png"),
    )

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    metric_folders = ", ".join(m.replace(".mean", "") for m in all_metrics)
    print(f"\nDone! All plots saved to: {output_dir}/")
    print(f"  Metric subdirs : {metric_folders}")
    print(f"  Structural plots in root output dir")


if __name__ == "__main__":
    main()