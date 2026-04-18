#!/usr/bin/env python3
"""
Batch landscape analysis runner for ExaTune.

Runs complete_analysis.py for every metric found in each parquet file,
producing a separate report subdirectory per experiment x metric.

Output structure:
  <output-dir>/
    <experiment>/
      accuracy/
        landscape_report.md
        plots/
      f1_macro/
        landscape_report.md
        plots/
      ...

Usage examples:
  # Explicit parquet files (all metrics auto-discovered)
  python run_batch_analysis.py --files results/mnist/results.parquet results/iris/results.parquet

  # Scan a directory recursively
  python run_batch_analysis.py --dir examples/results/benchmarks

  # Only run specific metrics
  python run_batch_analysis.py --files results/mnist/results.parquet --metrics accuracy f1_macro_mean
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

# defaults
DEFAULT_SCRIPT = Path(__file__).parent / "complete_analysis.py"
DEFAULT_OUTPUT_DIR = Path(r"C:\Users\katel\OneDrive\Desktop\ExaTune\exatune\analysis\neurips")

_NON_METRIC_COLS = {
    "job_id", "config_hash", "timestamp", "success",
    "mean_train_score", "fit_time_mean", "score_time_mean",
    "error_message", "error_type", "rank", "best_score",
    "kg_co2", "energy_kwh",
}
_SKIP_SUFFIXES = ("_std", "_rank")


def infer_metrics(parquet_path: Path) -> list:
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"  Warning: could not read {parquet_path} to infer metrics: {e}")
        return ["mean_score"]

    metrics = []
    for col in df.columns:
        if col in _NON_METRIC_COLS:
            continue
        if any(col.endswith(s) for s in _SKIP_SUFFIXES):
            continue
        if col == "mean_score" or col.endswith("_mean"):
            metrics.append(col)

    return metrics if metrics else ["mean_score"]


def get_exp_name(parquet_path: Path) -> str:
    parts = parquet_path.parts
    name_parts = [p for p in parts[:-1] if p.lower() != "results"]
    return name_parts[-1] if name_parts else "unknown"


def run_one(parquet_path, metric, exp_output, script, direction):
    cmd = [
        sys.executable, str(script),
        "--data", str(parquet_path),
        "--metric", metric,
        "--direction", direction,
        "--output-dir", str(exp_output),
    ]

    t0 = time.time()
    result = subprocess.run(cmd, text=True)
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"    v  {metric:<30s} ({elapsed:.1f}s)")
        return True
    else:
        print(f"    x  {metric:<30s} FAILED ({elapsed:.1f}s)")
        return False


def find_parquets(directory):
    return sorted(Path(directory).rglob("results.parquet"))


def main():
    parser = argparse.ArgumentParser(
        description="Run ExaTune landscape analysis for all metrics across multiple parquet files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dir", type=Path, default=None,
                        help="Directory to scan recursively for results.parquet files.")
    parser.add_argument("--files", type=Path, nargs="+", default=[],
                        help="Explicit list of parquet file paths.")
    parser.add_argument("--metrics", type=str, nargs="+", default=None,
                        help="Metrics to run (default: auto-discover all from each file).")
    parser.add_argument("--direction", type=str, default="maximize",
                        choices=["maximize", "minimize"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Root output directory.")
    parser.add_argument("--script", type=Path, default=DEFAULT_SCRIPT,
                        help="Path to complete_analysis.py.")

    args = parser.parse_args()

    parquets = list(args.files)
    if args.dir:
        found = find_parquets(args.dir)
        print(f"Found {len(found)} parquet file(s) under {args.dir}")
        parquets.extend(found)

    if not parquets:
        print("Error: no parquet files specified. Use --dir or --files.")
        sys.exit(1)

    seen = set()
    unique_parquets = []
    for p in parquets:
        r = Path(p).resolve()
        if r not in seen:
            seen.add(r)
            unique_parquets.append(r)

    print(f"\n{'=' * 60}")
    print(f"  ExaTune Batch Analysis - All Metrics")
    print(f"  Files     : {len(unique_parquets)}")
    print(f"  Output    : {args.output_dir}")
    print(f"  Direction : {args.direction}")
    print(f"  Metrics   : {'specified: ' + str(args.metrics) if args.metrics else 'auto-discover per file'}")
    print(f"{'=' * 60}")

    all_results = {}
    total_start = time.time()

    for parquet_path in unique_parquets:
        exp_name = get_exp_name(parquet_path)
        metrics = args.metrics if args.metrics else infer_metrics(parquet_path)

        print(f"\n{'-' * 60}")
        print(f"  Experiment : {exp_name}")
        print(f"  Input      : {parquet_path}")
        print(f"  Metrics    : {metrics}")
        print(f"{'-' * 60}")

        for metric in metrics:
            folder_name = metric.removesuffix("_mean") if metric != "mean_score" else "accuracy"
            exp_output = args.output_dir / exp_name / folder_name

            success = run_one(
                parquet_path=parquet_path,
                metric=metric,
                exp_output=exp_output,
                script=args.script,
                direction=args.direction,
            )
            all_results[(str(parquet_path), metric)] = success

    total_elapsed = time.time() - total_start

    n_ok = sum(all_results.values())
    n_fail = len(all_results) - n_ok

    print(f"\n{'=' * 60}")
    print(f"  Batch Complete - {total_elapsed:.1f}s total")
    print(f"{'=' * 60}")

    by_exp = {}
    for (path, metric), ok in all_results.items():
        by_exp.setdefault(path, []).append((metric, ok))

    for path, metric_results in by_exp.items():
        exp_name = get_exp_name(Path(path))
        print(f"\n  {exp_name}")
        for metric, ok in metric_results:
            status = "v" if ok else "x"
            print(f"    {status}  {metric}")

    print(f"\n  {n_ok}/{len(all_results)} runs succeeded, {n_fail} failed")
    print(f"{'=' * 60}\n")

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()