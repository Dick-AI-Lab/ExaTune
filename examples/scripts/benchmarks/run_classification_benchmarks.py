#!/usr/bin/env python3
"""
Run classification benchmarks with ExaTune.

This script demonstrates how to run multiple classification benchmark experiments
using different datasets and models.
"""

import argparse
from pathlib import Path
from exatune.core.experiment import Experiment


# Define benchmark configurations
CLASSIFICATION_BENCHMARKS = {
    "iris": {
        "config": "configs/benchmarks/iris_classification.yaml",
        "description": "Iris dataset (150 samples, 3 classes) - Random Forest",
        "grid_size_estimate": 1200,  # 6 * 8 * 5 * 5 * 5
    },
    "digits": {
        "config": "configs/benchmarks/digits_classification.yaml",
        "description": "Digits dataset (1797 samples, 10 classes) - SVM",
        "grid_size_estimate": 480,  # 5 * 4 * 6 * 4
    },
    "wine": {
        "config": "configs/benchmarks/wine_classification.yaml",
        "description": "Wine dataset (178 samples, 3 classes) - Gradient Boosting",
        "grid_size_estimate": 720,  # 5 * 6 * 6 * 4 * 4 * 5
    },
    "breast_cancer": {
        "config": "configs/benchmarks/breast_cancer_classification.yaml",
        "description": "Breast Cancer dataset (569 samples, 2 classes) - XGBoost",
        "grid_size_estimate": 15120,  # 5 * 6 * 7 * 5 * 6 * 6 * 5 * 5 * 5
    },
"pregnancy_outcome": {
        "config": "configs/born_air/pregnancy_outcome_calssification.yaml",
        "description": "My custom dataset - XGB from born air",
        "grid_size_estimate": 5184,
    },

    "pregnancy_outcome_LS": {
        "config": "configs/born_air/pregnancy_outcome_calssification_large_scale.yaml",
        "description": "My custom dataset - XGB from born air",
        "grid_size_estimate": 4680,
},
    "gestational_diabetes": {
        "config": "configs/born_air/gestational_diabetes_classification_large_scale.yaml",
        "description": "My custom dataset - XGB from born air",
        "grid_size_estimate": 4680,
},
    "maternal_risk": {
        
        "config": "configs/born_air/maternal_risk_classification_ls.yaml",
        "description": "My custom dataset - XGB from born air",
        "grid_size_estimate": 4680,
},
    "titanic": {
        
        "config": "configs/neurips/titanic_classification.yaml",
        "description": "My custom dataset - XGB from born air",
        "grid_size_estimate": 505440,
},
    "diabetes": {
        
        "config": "configs/neurips/diabetes_classification.yaml",
        "description": "My custom dataset - XGB from born air",
        "grid_size_estimate": 8316,
},    
"housing": {
        
        "config": "configs/neurips/housing_classification.yaml",
        "description": "My custom dataset - XGB from born air",
    "grid_size_estimate": 8316,

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
    if benchmark_name not in CLASSIFICATION_BENCHMARKS:
        print(f"Error: Unknown benchmark '{benchmark_name}'")
        print(f"Available benchmarks: {', '.join(CLASSIFICATION_BENCHMARKS.keys())}")
        return False

    benchmark = CLASSIFICATION_BENCHMARKS[benchmark_name]
    config_path = Path(__file__).parent.parent.parent / benchmark["config"]

    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        return False

    print("\n" + "=" * 80)
    print(f"ExaTune Classification Benchmark: {benchmark_name.upper()}")
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
    Run all classification benchmarks sequentially.

    Args:
        submit: Whether to submit jobs to SLURM
        dry_run: If True, only generate jobs without submitting
    """
    print("\n" + "=" * 80)
    print("Running All Classification Benchmarks")
    print("=" * 80)

    results = {}
    for benchmark_name in CLASSIFICATION_BENCHMARKS:
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
        description="Run ExaTune classification benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available benchmarks:
  iris           : Iris dataset (150 samples, 3 classes) - Random Forest
  digits         : Digits dataset (1797 samples, 10 classes) - SVM
  wine           : Wine dataset (178 samples, 3 classes) - Gradient Boosting
  breast_cancer  : Breast Cancer dataset (569 samples, 2 classes) - XGBoost
  all            : Run all benchmarks sequentially

Examples:
  # Generate jobs for iris benchmark (no submission)
  python run_classification_benchmarks.py iris

  # Generate and submit jobs for breast cancer benchmark
  python run_classification_benchmarks.py breast_cancer --submit

  # Dry run (generate jobs only)
  python run_classification_benchmarks.py all --dry-run

  # Run all benchmarks and submit to SLURM
  python run_classification_benchmarks.py all --submit
        """
    )

    parser.add_argument(
        "benchmark",
        choices=list(CLASSIFICATION_BENCHMARKS.keys()) + ["all"],
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
