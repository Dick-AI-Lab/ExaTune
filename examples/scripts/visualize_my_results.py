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


def create_synthetic_results(n_estimators_vals=None, max_depth_vals=None, mss_vals=None):
    """Create a synthetic results DataFrame for demonstration."""
    if n_estimators_vals is None:
        n_estimators_vals = [10, 50, 100, 200, 300]
    if max_depth_vals is None:
        max_depth_vals = [3, 5, 10, 15, 20]
    if mss_vals is None:
        mss_vals = [2, 5, 10]

    np.random.seed(42)
    configs = []
    for ne in n_estimators_vals:
        for md in max_depth_vals:
            for mss in mss_vals:
                # Score function with non-linear interactions
                score = (
                    0.5
                    + 0.0008 * ne
                    + 0.025 * md
                    - 0.008 * mss
                    - 0.00001 * ne * md  # Interaction term
                    + np.random.normal(0, 0.01)
                )
                score = min(max(score, 0.3), 0.99)
                train_score = min(score + 0.04 + np.random.normal(0, 0.005), 1.0)
                configs.append(
                    {
                        "job_id": len(configs),
                        "config_hash": f"hash_{len(configs):04d}",
                        "timestamp": "2025-01-01 00:00:00",
                        "success": True,
                        "mean_score": round(score, 6),
                        "std_score": round(abs(np.random.normal(0, 0.015)), 6),
                        "mean_train_score": round(train_score, 6),
                        "fit_time_mean": round(np.random.uniform(0.1, 3.0), 3),
                        "score_time_mean": round(np.random.uniform(0.01, 0.15), 3),
                        "n_estimators": ne,
                        "max_depth": md,
                        "min_samples_split": mss,
                    }
                )

    df = pd.DataFrame(configs)
    df = df.sort_values("mean_score", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    df["best_score"] = False
    df.loc[0, "best_score"] = True
    return df


def main():
    parser = argparse.ArgumentParser(description="ExaTune Visualization Demo")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./visualization_demo",
        help="Directory for output plots",
    )
    # adding the reuslts path to the argument 
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to results parquet file",
    )
    args = parser.parse_args()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Creating synthetic results...")
    results = pd.read_parquet(args.results)
    results.columns = [c.replace("hyperparameters.", "").replace("additional_metrics.", "") for c in results.columns]
    results = results.drop(columns=["cv_scores", "train_scores"], errors="ignore")
    print(f"  Generated {len(results)} configurations\n")

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
#TODO look at the different results for all of the differnt metrics 
    # - "f1_macro"
    # - "f1_micro"
    # - "f1_weighted"
    # - "precision_macro"
    # - "recall_macro"
    # - "roc_auc_ovr"

#! loop this correctly 
    plots = [
        ("Score Histogram", lambda: plot_score_histogram(results, output_path=output_dir / "histogram.png")),
        ("Score Violin (by max_depth)", lambda: plot_score_violin(results, group_by="max_depth", output_path=output_dir / "violin.png")),
        ("Train vs Test Scatter", lambda: plot_train_vs_test(results, output_path=output_dir / "train_vs_test.png")),
        ("Slice: n_estimators", lambda: plot_slice(results, param="n_estimators", show_individual=True, output_path=output_dir / "slice_n_estimators.png")),
        ("All Slices", lambda: plot_all_slices(results, output_path=output_dir / "all_slices.png")),
        ("Conditional Slice", lambda: plot_conditional_slice(results, param="n_estimators", condition_param="max_depth", output_path=output_dir / "conditional_slice.png")),
        ("Heatmap (n_estimators x max_depth)", lambda: plot_heatmap(results, x_param="n_estimators", y_param="max_depth", metric="f1_macro", output_path=output_dir / "heatmap_f1_macro.png")),
        ("3D Surface", lambda: plot_surface(results, x_param="n_estimators", y_param="max_depth", output_path=output_dir / "surface.png")),
        ("Contour Plot", lambda: plot_contour(results, x_param="n_estimators", y_param="max_depth", output_path=output_dir / "contour.png")),
        ("Parameter Importance (variance)", lambda: plot_importance(results, method="variance", output_path=output_dir / "importance_variance.png")),
        ("Parameter Importance (correlation)", lambda: plot_importance(results, method="correlation", output_path=output_dir / "importance_correlation.png")),
        ("Interaction Heatmap", lambda: plot_interaction_heatmap(results, output_path=output_dir / "interactions.png")),
        ("Parallel Coordinates", lambda: plot_parallel_coordinates(results, highlight_best=10, output_path=output_dir / "parallel.png")),
        ("Dashboard", lambda: plot_dashboard(results, output_path=output_dir / "dashboard.png")),
    ]

    for name, plot_func in plots:
        try:
            print(f"  Generating: {name}...")
            plot_func()
            plt.close("all")
        except Exception as e:
            print(f"    ERROR: {e}")

    print(f"\nAll plots saved to: {output_dir}/")
    print(f"Generated {len(plots)} visualization files.")


if __name__ == "__main__":
    main()
