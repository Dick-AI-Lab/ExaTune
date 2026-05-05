#!/usr/bin/env python3
"""
Run all ExaTune landscape analysis metrics on a results parquet file.
Metrics are read from the experiment YAML config file.

Output structure:
    <output-dir>/
        <metric_1>/
            smoothness/
            multimodality/
            importance/
            interactions/
            summary/
        <metric_2>/
            ...

Usage:
    python run_full_analysis.py \\
        --data   path/to/results.parquet \\
        --config path/to/experiment.yaml \\
        --output-dir path/to/output
"""

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------

def load_and_clean(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)

    # Flatten dot-notation columns from json_normalize
    df.columns = [c.replace(".", "_") for c in df.columns]
    df.columns = [c.replace("hyperparameters_", "") for c in df.columns]
    df.columns = [c.replace("additional_metrics_", "") for c in df.columns]

    # Drop array-valued columns (list / dict / ndarray) that break groupby
    def is_array_col(series):
        for val in series.dropna():
            if isinstance(val, (list, dict, np.ndarray)):
                return True
        return False

    array_cols = [c for c in df.columns if is_array_col(df[c])]
    if array_cols:
        print(f"  Dropping array-valued columns: {array_cols}")
        df = df.drop(columns=array_cols)

    return df


# ---------------------------------------------------------------------------
# Metric resolution
# ---------------------------------------------------------------------------

def resolve_metrics(df: pd.DataFrame, config_path: Path | None, explicit_metric: str | None) -> list[str]:
    if explicit_metric:
        if explicit_metric not in df.columns:
            raise ValueError(
                f"--metric '{explicit_metric}' not found in parquet.\n"
                f"Available columns: {df.columns.tolist()}"
            )
        return [explicit_metric]

    if config_path is not None:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        eval_cfg = cfg.get("evaluation", {})
        primary  = eval_cfg.get("scoring", "accuracy")
        extras   = eval_cfg.get("additional_metrics", [])

        metrics: list[str] = []

        def try_add(yaml_name: str) -> None:
            candidates = [
                yaml_name,
                f"{yaml_name}_mean",
                "mean_score",
            ]
            for c in candidates:
                if c in df.columns and c not in metrics:
                    metrics.append(c)
                    return
            print(f"  Warning: no column found for metric '{yaml_name}' — skipping.")

        try_add(primary)
        for m in extras:
            try_add(m)

        if metrics:
            return metrics

    # Auto-detect: find all _mean cols, excluding noise columns
    _NON_METRIC_COLS = {
        "fit_time_mean", "score_time_mean", "mean_train_score",
    }
    auto = [
        c for c in df.columns
        if c.endswith("_mean") and c not in _NON_METRIC_COLS
    ]
    if not auto and "mean_score" in df.columns:
        auto = ["mean_score"]

    if auto:
        print(f"  Auto-selected metrics: {auto}")
        return auto

    raise ValueError(
        "Could not resolve any metric columns. "
        "Pass --config or --metric explicitly."
    )
# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------

def write_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def section(title: str) -> None:
    print(f"\n    ── {title} {'─' * max(0, 48 - len(title))}")


# ---------------------------------------------------------------------------
# Per-category runners
# ---------------------------------------------------------------------------

def run_smoothness(df: pd.DataFrame, metric: str, out: Path) -> None:
    section("Smoothness")
    from exatune.analysis import (
        compute_autocorrelation,
        compute_gradient_statistics,
        compute_ruggedness,
    )

    autocorr   = compute_autocorrelation(df, metric=metric)
    gradient   = compute_gradient_statistics(df, metric=metric)
    ruggedness = compute_ruggedness(df, metric=metric)

    print(f"      Moran's I:        {autocorr['morans_i']:.4f}  — {autocorr['interpretation']}")
    print(f"      Mean gradient:    {gradient['mean_gradient']:.6f}")
    print(f"      Max gradient:     {gradient['max_gradient']:.6f}")
    print(f"      Roughness:        {gradient['roughness']:.4f}")
    print(f"      Ruggedness index: {ruggedness['ruggedness_index']:.4f}  — {ruggedness['interpretation']}")

    write_json(autocorr,   out / "autocorrelation.json")
    write_json(gradient,   out / "gradient_statistics.json")
    write_json(ruggedness, out / "ruggedness.json")
    write_text("\n".join([
        "# Smoothness Analysis",
        "",
        f"**Moran's I:** {autocorr['morans_i']:.4f}",
        f"- {autocorr['interpretation']}",
        "",
        f"**Mean gradient:** {gradient['mean_gradient']:.6f}  ",
        f"**Max gradient:** {gradient['max_gradient']:.6f}  ",
        f"**Roughness:** {gradient['roughness']:.4f}",
        "",
        f"**Ruggedness index:** {ruggedness['ruggedness_index']:.4f}",
        f"- {ruggedness['interpretation']}",
    ]), out / "smoothness_summary.md")


def run_multimodality(df: pd.DataFrame, metric: str, direction: str, out: Path) -> None:
    section("Multimodality")
    from exatune.analysis import (
        compute_fitness_distance_correlation,
        compute_multimodality_metrics,
        find_local_optima,
    )

    optima = find_local_optima(df, direction=direction, metric=metric)
    mm     = compute_multimodality_metrics(df, direction=direction, metric=metric)
    fdc    = compute_fitness_distance_correlation(df, direction=direction, metric=metric)

    print(f"      Local optima found:  {len(optima)}")
    print(f"      Optima density:      {mm['optima_density']:.4f}")
    print(f"      Global optimum:      {mm['global_optimum_value']:.6f}")
    print(f"      Funnel index:        {mm['funnel_index']:.4f}  — {mm['interpretation']}")
    print(f"      FDC:                 {fdc['fdc']:.4f}  — {fdc['interpretation']}")

    out.mkdir(parents=True, exist_ok=True)
    optima.to_csv(out / "local_optima.csv", index=False)
    write_json(mm,  out / "multimodality_metrics.json")
    write_json(fdc, out / "fitness_distance_correlation.json")
    write_text("\n".join([
        "# Multimodality Analysis",
        "",
        f"**Local optima found:** {len(optima)}  ",
        f"**Optima density:** {mm['optima_density']:.4f}  ",
        f"**Global optimum:** {mm['global_optimum_value']:.6f}  ",
        f"**Funnel index:** {mm['funnel_index']:.4f}",
        f"- {mm['interpretation']}",
        "",
        f"**Fitness Distance Correlation (FDC):** {fdc['fdc']:.4f}",
        f"- {fdc['interpretation']}",
        "",
        "Local optima configurations saved to `local_optima.csv`.",
    ]), out / "multimodality_summary.md")


def run_importance(df: pd.DataFrame, metric: str, out: Path) -> None:
    section("Hyperparameter Importance")
    from exatune.analysis import compute_importance_summary, compute_variance_importance

    importance = compute_variance_importance(df, metric=metric)
    summary_df = compute_importance_summary(df, metric=metric)

    print("      Variance importance:")
    for param, imp in importance.items():
        print(f"        {param:30s} {imp:.4f}")

    out.mkdir(parents=True, exist_ok=True)
    write_json(importance, out / "variance_importance.json")
    summary_df.to_csv(out / "importance_summary.csv", index=False)

    md_rows = ["| Parameter | Importance |", "| --- | --- |"]
    for _, row in summary_df.iterrows():
        md_rows.append(f"| {row.iloc[0]} | {row.iloc[1]:.4f} |")
    write_text("\n".join(["# Hyperparameter Importance", "", *md_rows]),
               out / "importance_summary.md")


def run_interactions(df: pd.DataFrame, metric: str, out: Path) -> None:
    section("Interactions")
    from exatune.analysis import (
        compute_pairwise_interactions,
        detect_significant_interactions,
    )

    interactions = compute_pairwise_interactions(df, metric=metric)
    significant  = detect_significant_interactions(df, threshold=0.001, metric=metric)

    print("      Pairwise interactions:")
    print(textwrap.indent(interactions.to_string(index=False), "        "))
    if significant:
        print("      Significant interactions (threshold=0.001):")
        for p1, p2, strength in significant:
            print(f"        {p1} × {p2}: {strength:.4f}")
    else:
        print("      No significant interactions detected.")

    out.mkdir(parents=True, exist_ok=True)
    interactions.to_csv(out / "pairwise_interactions.csv", index=False)
    write_json(
        [{"param1": p1, "param2": p2, "strength": s} for p1, p2, s in significant],
        out / "significant_interactions.json",
    )

    sig_md = "\n".join(f"- **{p1} × {p2}:** {s:.4f}" for p1, p2, s in significant) \
        if significant else "_None detected at threshold=0.001._"

    try:
        interactions_md = interactions.to_markdown(index=False)
    except Exception:
        interactions_md = interactions.to_string(index=False)

    write_text("\n".join([
        "# Interaction Analysis",
        "",
        "## Pairwise Interaction Strengths",
        "",
        interactions_md,
        "",
        "## Significant Interactions (threshold = 0.001)",
        "",
        sig_md,
    ]), out / "interactions_summary.md")


def run_summary(df: pd.DataFrame, metric: str, direction: str, out: Path) -> None:
    section("Experiment Summary + Report")
    from exatune.analysis import (
        compute_experiment_summary,
        format_summary_text,
        generate_report,
    )

    summary = compute_experiment_summary(df, metric=metric, direction=direction)
    text    = format_summary_text(summary)
    print(textwrap.indent(text, "      "))

    out.mkdir(parents=True, exist_ok=True)
    write_text(text, out / "experiment_summary.txt")
    report_path = generate_report(
        df,
        output_path=out,
        metric=metric,
        direction=direction,
        include_plots=True,
        format="markdown",
    )
    print(f"      Full report: {report_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run all ExaTune landscape analyses for every metric "
            "defined in the experiment YAML config."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Output structure:
              <output-dir>/
                <metric>/
                  smoothness/
                  multimodality/
                  importance/
                  interactions/
                  summary/

            Examples:
              # Recommended — metrics pulled from YAML
              python run_full_analysis.py \\
                  --data   results.parquet \\
                  --config experiment.yaml \\
                  --output-dir ./analysis_out

              # Override to a single metric
              python run_full_analysis.py \\
                  --data results.parquet \\
                  --config experiment.yaml \\
                  --output-dir ./analysis_out \\
                  --metric f1_macro_mean

              # Skip slow categories
              python run_full_analysis.py \\
                  --data results.parquet \\
                  --config experiment.yaml \\
                  --output-dir ./analysis_out \\
                  --skip interactions multimodality
        """),
    )
    parser.add_argument("--data",       required=True,
                        help="Path to results parquet file")
    parser.add_argument("--config",     default=None,
                        help="Path to experiment YAML — used to read the metric list")
    parser.add_argument("--output-dir", required=True,
                        help="Root output directory; one subdir is created per metric")
    parser.add_argument("--metric",     default=None,
                        help="Analyse only this metric column (bypasses YAML metric list)")
    parser.add_argument("--direction",  default="maximize",
                        choices=["maximize", "minimize"],
                        help="Optimisation direction (default: maximize)")
    parser.add_argument("--skip",       nargs="*", default=[],
                        choices=["smoothness", "multimodality", "importance",
                                 "interactions", "summary"],
                        help="Analysis categories to skip")
    args = parser.parse_args()

    input_path  = Path(args.data).resolve()
    config_path = Path(args.config).resolve() if args.config else None
    root_out    = Path(args.output_dir).resolve()
    skip        = set(args.skip or [])

    print(f"\nExaTune Full Landscape Analysis")
    print(f"  Input:      {input_path}")
    print(f"  Config:     {config_path or '(none — metric auto-detected)'}")
    print(f"  Output:     {root_out}")
    print(f"  Direction:  {args.direction}")

    print("\nLoading and cleaning data...")
    df = load_and_clean(input_path)
    print(f"  Columns ({len(df.columns)}): {df.columns.tolist()}")
    print(f"  Rows: {len(df)}")

    metrics = resolve_metrics(df, config_path, args.metric)
    print(f"\nMetrics to analyse ({len(metrics)}): {metrics}")

    direction = args.direction
    if "mean_score" in metrics and df["mean_score"].mean() < 0:
        direction = "maximize"
        print("  Negative mean_score detected (neg_* scorer) — using direction=maximize")

    # ── Main loop ────────────────────────────────────────────────────────────
    for metric in metrics:
        print(f"\n{'═' * 60}")
        print(f"  Metric: {metric}")
        print(f"  Range:  {df[metric].min():.4f} – {df[metric].max():.4f}")
        print(f"{'═' * 60}")

        metric_dir = root_out / metric  # e.g. .../analysis_out/f1_macro_mean/

        if "smoothness"    not in skip:
            run_smoothness(df, metric, metric_dir / "smoothness")
        if "multimodality" not in skip:
            run_multimodality(df, metric, direction, metric_dir / "multimodality")
        if "importance"    not in skip:
            run_importance(df, metric, metric_dir / "importance")
        if "interactions"  not in skip:
            run_interactions(df, metric, metric_dir / "interactions")
        if "summary"       not in skip:
            run_summary(df, metric, direction, metric_dir / "summary")

    print(f"\n{'═' * 60}")
    print(f"  All analyses complete.")
    print(f"  Results written to: {root_out}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()