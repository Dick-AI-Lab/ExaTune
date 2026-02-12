#!/usr/bin/env python3
"""Demonstrate all ExaTune analysis functions using a synthetic dataset.

This script generates a synthetic results DataFrame and runs every
landscape analysis metric supported by ExaTune. It can be run without
a SLURM cluster or any actual experiment data.

Usage:
    python examples/scripts/analyze_landscape.py [--output-dir OUTPUT_DIR]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def create_synthetic_results():
    """Create a synthetic results DataFrame for demonstration."""
    np.random.seed(42)
    configs = []
    for ne in [10, 50, 100, 200, 300]:
        for md in [3, 5, 10, 15, 20]:
            for mss in [2, 5, 10]:
                score = (
                    0.5
                    + 0.0008 * ne
                    + 0.025 * md
                    - 0.008 * mss
                    - 0.00001 * ne * md
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
    parser = argparse.ArgumentParser(description="ExaTune Landscape Analysis Demo")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./analysis_demo",
        help="Directory for output report",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("Creating synthetic results...")
    results = create_synthetic_results()
    print(f"  Generated {len(results)} configurations\n")

    # --- Individual Analysis Functions ---
    from exatune.analysis import (
        compute_autocorrelation,
        compute_experiment_summary,
        compute_fitness_distance_correlation,
        compute_gradient_statistics,
        compute_importance_summary,
        compute_multimodality_metrics,
        compute_pairwise_interactions,
        compute_ruggedness,
        compute_variance_importance,
        detect_significant_interactions,
        find_local_optima,
        format_summary_text,
        generate_report,
    )

    print("=" * 60)
    print("SMOOTHNESS ANALYSIS")
    print("=" * 60)

    autocorr = compute_autocorrelation(results)
    print(f"  Moran's I:           {autocorr['morans_i']:.4f}")
    print(f"  Interpretation:      {autocorr['interpretation']}")

    gradient = compute_gradient_statistics(results)
    print(f"  Mean gradient:       {gradient['mean_gradient']:.6f}")
    print(f"  Max gradient:        {gradient['max_gradient']:.6f}")
    print(f"  Roughness:           {gradient['roughness']:.4f}")

    ruggedness = compute_ruggedness(results)
    print(f"  Ruggedness index:    {ruggedness['ruggedness_index']:.4f}")
    print(f"  Interpretation:      {ruggedness['interpretation']}")

    print(f"\n{'=' * 60}")
    print("MULTIMODALITY ANALYSIS")
    print("=" * 60)

    optima = find_local_optima(results, direction="maximize")
    print(f"  Local optima found:  {len(optima)}")

    mm = compute_multimodality_metrics(results, direction="maximize")
    print(f"  Optima density:      {mm['optima_density']:.4f}")
    print(f"  Global optimum:      {mm['global_optimum_value']:.6f}")
    print(f"  Funnel index:        {mm['funnel_index']:.4f}")
    print(f"  Interpretation:      {mm['interpretation']}")

    fdc = compute_fitness_distance_correlation(results, direction="maximize")
    print(f"  FDC:                 {fdc['fdc']:.4f}")
    print(f"  Interpretation:      {fdc['interpretation']}")

    print(f"\n{'=' * 60}")
    print("HYPERPARAMETER IMPORTANCE")
    print("=" * 60)

    importance = compute_variance_importance(results)
    for param, imp in importance.items():
        print(f"  {param:25s} {imp:.4f}")

    summary_df = compute_importance_summary(results)
    print(f"\n  Importance Summary:")
    print(summary_df.to_string(index=False))

    print(f"\n{'=' * 60}")
    print("INTERACTION ANALYSIS")
    print("=" * 60)

    interactions = compute_pairwise_interactions(results)
    print(interactions.to_string(index=False))

    significant = detect_significant_interactions(results, threshold=0.001)
    if significant:
        print(f"\n  Significant interactions (threshold=0.001):")
        for p1, p2, strength in significant:
            print(f"    {p1} x {p2}: {strength:.4f}")
    else:
        print(f"\n  No significant interactions detected.")

    # --- Full Summary ---
    print(f"\n{'=' * 60}")
    print("FULL EXPERIMENT SUMMARY")
    print("=" * 60)

    full_summary = compute_experiment_summary(results, direction="maximize")
    print(format_summary_text(full_summary))

    # --- Generate Report ---
    print(f"\nGenerating full report to: {output_dir}/")
    report_path = generate_report(
        results,
        output_path=output_dir,
        direction="maximize",
        include_plots=True,
        format="markdown",
    )
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
