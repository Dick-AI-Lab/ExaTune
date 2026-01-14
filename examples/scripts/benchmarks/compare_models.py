#!/usr/bin/env python3
"""
Compare multiple models on the same dataset using ExaTune.

This script runs hyperparameter searches for multiple models on the same dataset
and compares their performance.
"""

import argparse
from pathlib import Path
from typing import List, Dict
import pandas as pd
from exatune.core.experiment import Experiment


def create_model_configs(dataset: str = "iris") -> Dict[str, Dict]:
    """
    Create configuration dictionaries for different models.

    Args:
        dataset: Dataset name

    Returns:
        Dictionary mapping model names to their configurations
    """
    base_config = {
        "experiment": {
            "name": None,  # Will be set per model
            "output_dir": None,  # Will be set per model
            "random_seed": 42,
        },
        "dataset": {
            "name": dataset,
            "test_size": 0.2,
            "random_state": 42,
            "stratify": True,
        },
        "evaluation": {
            "cv_folds": 5,
            "cv_strategy": "stratified",
            "scoring": "accuracy",
            "additional_metrics": ["f1_macro", "precision_macro", "recall_macro"],
            "n_jobs": 1,
        },
        "slurm": {
            "partition": "compute",
            "time": "01:00:00",
            "memory": "8G",
            "cpus_per_task": 2,
        },
    }

    models = {
        "random_forest": {
            **base_config,
            "model": {
                "type": "sklearn",
                "class_name": "sklearn.ensemble.RandomForestClassifier",
                "task": "classification",
                "fixed_params": {"random_state": 42},
            },
            "hyperparameters": {
                "n_estimators": {"type": "discrete", "values": [50, 100, 200]},
                "max_depth": {"type": "discrete", "values": [5, 10, 15, None]},
                "min_samples_split": {"type": "discrete", "values": [2, 5, 10]},
            },
        },
        "gradient_boosting": {
            **base_config,
            "model": {
                "type": "sklearn",
                "class_name": "sklearn.ensemble.GradientBoostingClassifier",
                "task": "classification",
                "fixed_params": {"random_state": 42},
            },
            "hyperparameters": {
                "n_estimators": {"type": "discrete", "values": [50, 100, 200]},
                "learning_rate": {"type": "discrete", "values": [0.01, 0.1, 0.2]},
                "max_depth": {"type": "discrete", "values": [3, 5, 7]},
            },
        },
        "svm": {
            **base_config,
            "model": {
                "type": "sklearn",
                "class_name": "sklearn.svm.SVC",
                "task": "classification",
                "fixed_params": {"random_state": 42},
            },
            "hyperparameters": {
                "C": {"type": "discrete", "values": [0.1, 1.0, 10.0]},
                "kernel": {"type": "discrete", "values": ["rbf", "linear", "poly"]},
                "gamma": {"type": "discrete", "values": ["scale", "auto"]},
            },
        },
        "xgboost": {
            **base_config,
            "model": {
                "type": "xgboost",
                "task": "classification",
                "fixed_params": {
                    "objective": "multi:softprob",
                    "random_state": 42,
                },
            },
            "hyperparameters": {
                "n_estimators": {"type": "discrete", "values": [50, 100, 200]},
                "learning_rate": {"type": "discrete", "values": [0.01, 0.1, 0.2]},
                "max_depth": {"type": "discrete", "values": [3, 5, 7]},
                "subsample": {"type": "discrete", "values": [0.7, 0.8, 1.0]},
            },
        },
    }

    # Update experiment names and output dirs
    for model_name in models:
        models[model_name]["experiment"]["name"] = f"{dataset}_{model_name}_comparison"
        models[model_name]["experiment"]["output_dir"] = (
            f"./results/comparison/{dataset}/{model_name}"
        )

    return models


def run_model_comparison(
    dataset: str = "iris",
    models: List[str] = None,
    submit: bool = False,
    dry_run: bool = False,
):
    """
    Run model comparison experiments.

    Args:
        dataset: Dataset name
        models: List of model names to compare
        submit: Whether to submit jobs to SLURM
        dry_run: If True, only generate jobs without submitting
    """
    print("\n" + "=" * 80)
    print(f"ExaTune Model Comparison: {dataset.upper()}")
    print("=" * 80)

    # Get model configurations
    all_model_configs = create_model_configs(dataset)

    if models is None:
        models = list(all_model_configs.keys())

    # Validate model names
    invalid_models = set(models) - set(all_model_configs.keys())
    if invalid_models:
        print(f"Error: Invalid model names: {', '.join(invalid_models)}")
        print(f"Available models: {', '.join(all_model_configs.keys())}")
        return

    print(f"\nComparing {len(models)} models:")
    for model_name in models:
        print(f"  - {model_name}")

    # Run experiments for each model
    results = {}
    for model_name in models:
        print("\n" + "-" * 80)
        print(f"Running: {model_name.upper()}")
        print("-" * 80)

        config = all_model_configs[model_name]

        # Create experiment from config dictionary
        experiment = Experiment.from_dict(config)

        # Generate jobs
        print(f"\n[1/2] Generating job scripts...")
        n_jobs = experiment.generate_jobs()
        print(f"✓ Generated {n_jobs} job scripts")

        results[model_name] = {
            "experiment": experiment,
            "n_jobs": n_jobs,
            "output_dir": experiment.output_dir,
        }

        if dry_run:
            print(f"[DRY RUN] Skipping submission for {model_name}")
            continue

        # Submit jobs if requested
        if submit:
            print(f"\n[2/2] Submitting jobs to SLURM...")
            job_ids = experiment.submit_jobs()

            if job_ids:
                print(f"✓ Submitted {len(job_ids)} jobs")
                results[model_name]["job_ids"] = job_ids
            else:
                print(f"✗ Job submission failed")
                results[model_name]["job_ids"] = []

    # Print summary
    print("\n" + "=" * 80)
    print("Model Comparison Summary")
    print("=" * 80)
    for model_name, info in results.items():
        print(f"\n{model_name}:")
        print(f"  Jobs generated: {info['n_jobs']}")
        print(f"  Output directory: {info['output_dir']}")
        if "job_ids" in info:
            print(f"  Jobs submitted: {len(info['job_ids'])}")

    if not dry_run and submit:
        print("\n" + "=" * 80)
        print("Next Steps:")
        print("=" * 80)
        print("\nMonitor progress with:")
        for model_name in results:
            print(f"  exatune status --experiment-dir {results[model_name]['output_dir']}")

        print("\nCollect results with:")
        for model_name in results:
            print(f"  exatune collect --experiment-dir {results[model_name]['output_dir']}")

        print("\nAfter jobs complete, compare results with:")
        print(f"  python compare_results.py {dataset} {' '.join(models)}")

    print("\n" + "=" * 80 + "\n")


def compare_results(dataset: str, models: List[str]):
    """
    Compare results from completed experiments.

    Args:
        dataset: Dataset name
        models: List of model names
    """
    print("\n" + "=" * 80)
    print(f"Comparing Results: {dataset.upper()}")
    print("=" * 80 + "\n")

    comparison_data = []

    for model_name in models:
        output_dir = Path(f"./results/comparison/{dataset}/{model_name}")
        results_file = output_dir / "results" / "all_results.parquet"

        if not results_file.exists():
            print(f"⚠ Results not found for {model_name}: {results_file}")
            continue

        # Load results
        df = pd.read_parquet(results_file)

        # Get best configuration
        best_idx = df["mean_score"].idxmax()
        best_row = df.loc[best_idx]

        comparison_data.append({
            "Model": model_name,
            "Best Score": best_row["mean_score"],
            "Std Score": best_row["std_score"],
            "Configurations Tested": len(df),
            "Success Rate": (df["success"].sum() / len(df)) * 100,
        })

        print(f"{model_name}:")
        print(f"  Best score: {best_row['mean_score']:.4f} ± {best_row['std_score']:.4f}")
        print(f"  Configurations tested: {len(df)}")
        print(f"  Success rate: {(df['success'].sum() / len(df)) * 100:.1f}%")
        print()

    # Create comparison table
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values("Best Score", ascending=False)

        print("\n" + "=" * 80)
        print("Model Rankings")
        print("=" * 80)
        print(comparison_df.to_string(index=False))

        # Save comparison
        output_path = Path(f"./results/comparison/{dataset}/comparison_summary.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(output_path, index=False)
        print(f"\n✓ Comparison saved to: {output_path}")

    print("\n" + "=" * 80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare multiple models on the same dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available models:
  random_forest      : Random Forest Classifier
  gradient_boosting  : Gradient Boosting Classifier
  svm                : Support Vector Machine
  xgboost            : XGBoost Classifier

Examples:
  # Compare all models on iris (generate jobs only)
  python compare_models.py iris

  # Compare specific models and submit jobs
  python compare_models.py iris --models random_forest xgboost --submit

  # Compare results after experiments complete
  python compare_models.py iris --compare-results --models random_forest xgboost

  # Dry run
  python compare_models.py digits --dry-run
        """
    )

    parser.add_argument(
        "dataset",
        choices=["iris", "digits", "wine", "breast_cancer"],
        help="Dataset to use for comparison"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["random_forest", "gradient_boosting", "svm", "xgboost"],
        help="Models to compare (default: all)"
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit jobs to SLURM after generation"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate jobs but don't submit"
    )
    parser.add_argument(
        "--compare-results",
        action="store_true",
        help="Compare results from completed experiments"
    )

    args = parser.parse_args()

    if args.compare_results:
        models = args.models if args.models else ["random_forest", "gradient_boosting", "svm", "xgboost"]
        compare_results(args.dataset, models)
    else:
        run_model_comparison(
            dataset=args.dataset,
            models=args.models,
            submit=args.submit,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
