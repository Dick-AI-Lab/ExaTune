#!/usr/bin/env python3
"""
Run regression benchmarks with ExaTune.

This script demonstrates how to run multiple regression benchmark experiments
using different datasets and models.
"""

import argparse
from pathlib import Path
from exatune.core.experiment import Experiment


# Define benchmark configurations
REGRESSION_BENCHMARKS = {
    "diabetes": {
        "config": "configs/benchmarks/diabetes_regression.yaml",
        "description": "Diabetes dataset (442 samples) - Random Forest Regressor",
        "grid_size_estimate": 2520,  # 6 * 7 * 5 * 5 * 5 * 2
    },
    "california_housing": {
        "config": "configs/benchmarks/california_housing_regression.yaml",
        "description": "California Housing dataset (20640 samples) - XGBoost Regressor",
        "grid_size_estimate": 7200,  # 6 * 5 * 6 * 5 * 5 * 5 * 5 * 4 * 4
    },
}


def run_benchmark(benchmark_name: str, submit: bool = False, dry_run: bool = False):
    """
    Run a single benchmark experiment.

    Args:
        benchmark_name: Name of the benchmark to run
        submit: Whether to submit jobs to SLURM
        dry_run: If True, only generate jobs without submitting
    """
    if benchmark_name not in REGRESSION_BENCHMARKS:
        print(f"Error: Unknown benchmark '{benchmark_name}'")
        print(f"Available benchmarks: {', '.join(REGRESSION_BENCHMARKS.keys())}")
        return False

    benchmark = REGRESSION_BENCHMARKS[benchmark_name]
    config_path = Path(__file__).parent.parent.parent / benchmark["config"]

    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        return False

    print("\n" + "=" * 80)
    print(f"ExaTune Regression Benchmark: {benchmark_name.upper()}")
    print("=" * 80)
    print(f"\nDescription: {benchmark['description']}")
    print(f"Config: {config_path}")
    print(f"Estimated grid size: {benchmark['grid_size_estimate']} configurations")
    print()

    # Load experiment from config
    experiment = Experiment.from_config(config_path)

    # Print summary
    experiment.print_summary()

    # Generate job scripts
    print("\n[1/3] Generating job scripts...")
    n_jobs = experiment.generate_jobs()
    print(f"✓ Generated {n_jobs} job scripts")

    if dry_run:
        print("\n[DRY RUN] Stopping before job submission")
        print(f"\nTo submit jobs manually, run:")
        print(f"  exatune run {experiment.output_dir}/config.yaml")
        return True

    # Submit jobs if requested
    if submit:
        print("\n[2/3] Submitting jobs to SLURM...")
        job_ids = experiment.submit_jobs()

        if job_ids:
            print(f"✓ Submitted {len(job_ids)} jobs")
            print(f"  Job IDs: {', '.join(job_ids[:5])}" +
                  (f"... (+{len(job_ids)-5} more)" if len(job_ids) > 5 else ""))

            print("\n[3/3] Monitoring job progress...")
            print("  (Press Ctrl+C to stop monitoring without cancelling jobs)")
            try:
                experiment.monitor_progress()
            except KeyboardInterrupt:
                print("\n\nMonitoring stopped. Jobs are still running.")
        else:
            print("✗ Job submission failed")
            return False
    else:
        print(f"\n[SKIPPED] Job submission (use --submit to submit)")
        print(f"\nTo submit jobs, run:")
        print(f"  exatune run {experiment.output_dir}/config.yaml")

    print(f"\n{'=' * 80}")
    print(f"Experiment output directory: {experiment.output_dir}")
    print(f"{'=' * 80}\n")

    return True


def run_all_benchmarks(submit: bool = False, dry_run: bool = False):
    """
    Run all regression benchmarks sequentially.

    Args:
        submit: Whether to submit jobs to SLURM
        dry_run: If True, only generate jobs without submitting
    """
    print("\n" + "=" * 80)
    print("Running All Regression Benchmarks")
    print("=" * 80)

    results = {}
    for benchmark_name in REGRESSION_BENCHMARKS:
        success = run_benchmark(benchmark_name, submit=submit, dry_run=dry_run)
        results[benchmark_name] = success

    # Print summary
    print("\n" + "=" * 80)
    print("Benchmark Summary")
    print("=" * 80)
    for benchmark_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {benchmark_name:20s} {status}")
    print("=" * 80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run ExaTune regression benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available benchmarks:
  diabetes           : Diabetes dataset (442 samples) - Random Forest
  california_housing : California Housing (20640 samples) - XGBoost
  all                : Run all benchmarks sequentially

Examples:
  # Generate jobs for diabetes benchmark (no submission)
  python run_regression_benchmarks.py diabetes

  # Generate and submit jobs for California housing
  python run_regression_benchmarks.py california_housing --submit

  # Dry run (generate jobs only)
  python run_regression_benchmarks.py all --dry-run

  # Run all benchmarks and submit to SLURM
  python run_regression_benchmarks.py all --submit
        """
    )

    parser.add_argument(
        "benchmark",
        choices=list(REGRESSION_BENCHMARKS.keys()) + ["all"],
        help="Name of benchmark to run (or 'all' for all benchmarks)"
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit jobs to SLURM after generation"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate jobs but don't submit (useful for testing)"
    )

    args = parser.parse_args()

    if args.benchmark == "all":
        run_all_benchmarks(submit=args.submit, dry_run=args.dry_run)
    else:
        run_benchmark(args.benchmark, submit=args.submit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
