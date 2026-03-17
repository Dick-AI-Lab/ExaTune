#!/usr/bin/env python3
"""Demonstrate all ExaTune visualization functions using a synthetic dataset.

This script generates a synthetic results DataFrame and produces every
visualization type supported by ExaTune. It can be run without a SLURM
cluster or any actual experiment data.

Usage:
    python examples/scripts/visualize_results.py [--output-dir OUTPUT_DIR]
"""
#! Copied file to subastute sythetic data with our results

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="ExaTune Visualization Demo")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./visualization_demo",
        help="Directory for output plots",
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to results parquet file",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    results = pd.read_parquet(args.results)
    results.columns = [c.replace("hyperparameters.", "").replace("additional_metrics.", "") for c in results.columns]
    results = results.drop(columns=["cv_scores", "train_scores"], errors="ignore")
    print(f"  Loaded {len(results)} configurations\n")

    # Import visualization functions
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

    # Columns in the parquet are stored as e.g. "f1_macro.mean" after flattening.
    # The folder name strips ".mean" so paths stay clean on all OSes.
    metrics = [
        "f1_macro.mean",
        "f1_micro.mean",
        "f1_weighted.mean",
        "precision_macro.mean",
        "recall_macro.mean",
        "roc_auc_ovr.mean",
    ]
    primary = "f1_macro.mean"

    # --- Per-metric plots ---
    # These visualize the metric value itself, so running once per metric is meaningful.
    print("Generating per-metric plots...")
    for metric in metrics:
        metric_dir = output_dir / metric.replace(".mean", "")  # e.g. "f1_macro"
        metric_dir.mkdir(exist_ok=True)

        per_metric_plots = [
            (f"Score Histogram ({metric})",
             lambda m=metric, d=metric_dir: plot_score_histogram(results, metric=m, output_path=d / "histogram.png")),
            (f"Score Violin by max_depth ({metric})",
             lambda m=metric, d=metric_dir: plot_score_violin(results, group_by="max_depth", metric=m, output_path=d / "violin.png")),
            (f"Heatmap n_estimators x max_depth ({metric})",
             lambda m=metric, d=metric_dir: plot_heatmap(results, x_param="n_estimators", y_param="max_depth", metric=m, output_path=d / "heatmap.png")),
        ]

        for name, plot_func in per_metric_plots:
            try:
                print(f"  Generating: {name}...")
                plot_func()
                plt.close("all")
            except Exception as e:
                print(f"    ERROR: {e}")

    # --- Single plots (structural/relationship plots, primary metric only) ---
    # These show parameter relationships and landscape structure, which is
    # stable across metrics. Running once with the primary metric is sufficient.
    print(f"\nGenerating single plots (primary metric: {primary})...")
    single_plots = [
        ("Train vs Test Scatter",
         lambda: plot_train_vs_test(results, output_path=output_dir / "train_vs_test.png")),
        ("Slice: n_estimators",
         lambda: plot_slice(results, param="n_estimators", show_individual=True, output_path=output_dir / "slice_n_estimators.png")),
        ("All Slices",
         lambda: plot_all_slices(results, output_path=output_dir / "all_slices.png")),
        ("Conditional Slice",
         lambda: plot_conditional_slice(results, param="n_estimators", condition_param="max_depth", output_path=output_dir / "conditional_slice.png")),
        ("3D Surface",
         lambda: plot_surface(results, x_param="n_estimators", y_param="max_depth", output_path=output_dir / "surface.png")),
        ("Contour Plot",
         lambda: plot_contour(results, x_param="n_estimators", y_param="max_depth", output_path=output_dir / "contour.png")),
        ("Parameter Importance (variance)",
         lambda: plot_importance(results, method="variance", output_path=output_dir / "importance_variance.png")),
        ("Parameter Importance (correlation)",
         lambda: plot_importance(results, method="correlation", output_path=output_dir / "importance_correlation.png")),
        ("Interaction Heatmap",
         lambda: plot_interaction_heatmap(results, output_path=output_dir / "interactions.png")),
        ("Parallel Coordinates",
         lambda: plot_parallel_coordinates(results, highlight_best=10, output_path=output_dir / "parallel.png")),
        ("Dashboard",
         lambda: plot_dashboard(results, output_path=output_dir / "dashboard.png")),
    ]

    for name, plot_func in single_plots:
        try:
            print(f"  Generating: {name}...")
            plot_func()
            plt.close("all")
        except Exception as e:
            print(f"    ERROR: {e}")

    n_per_metric = len(metrics) * 3  # 3 plot types per metric
    n_single = len(single_plots)
    metric_dirs = ", ".join(m.replace(".mean", "") for m in metrics)
    print(f"\nAll plots saved to: {output_dir}/")
    print(f"  {n_per_metric} per-metric plots (in subdirs: {metric_dirs})")
    print(f"  {n_single} single plots (in root output dir)")
    print(f"  {n_per_metric + n_single} total files generated.")


if __name__ == "__main__":
    main()