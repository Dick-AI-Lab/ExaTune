"""
Worker script for ExaTune hyperparameter search jobs.

This script is executed by SLURM jobs to train models with specific
hyperparameter configurations and save results.
"""

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate

from exatune.core.config import ExaTuneConfig, load_config


def load_dataset(config: ExaTuneConfig) -> tuple:
    """
    Load dataset from configuration.

    Args:
        config: ExaTune configuration

    Returns:
        Tuple of (X, y) arrays

    Raises:
        ValueError: If dataset not found or invalid
    """
    dataset_name = config.dataset.name.lower()

    # Built-in sklearn datasets
    if dataset_name in ["iris", "digits", "wine", "breast_cancer"]:
        from sklearn import datasets

        if dataset_name == "iris":
            data = datasets.load_iris()
        elif dataset_name == "digits":
            data = datasets.load_digits()
        elif dataset_name == "wine":
            data = datasets.load_wine()
        elif dataset_name == "breast_cancer":
            data = datasets.load_breast_cancer()

        return data.data, data.target

    # Custom dataset from file
    elif config.dataset.path:
        dataset_path = Path(config.dataset.path)

        if not dataset_path.exists():
            raise ValueError(f"Dataset file not found: {dataset_path}")

        # Load from CSV
        if dataset_path.suffix == ".csv":
            df = pd.read_csv(dataset_path)
        # Load from Parquet
        elif dataset_path.suffix == ".parquet":
            df = pd.read_parquet(dataset_path)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path.suffix}")

        # Extract features and target
        if config.dataset.target_column:
            if config.dataset.target_column not in df.columns:
                raise ValueError(
                    f"Target column '{config.dataset.target_column}' not found in dataset"
                )
            y = df[config.dataset.target_column].values
            X = df.drop(columns=[config.dataset.target_column]).values
        else:
            raise ValueError("target_column must be specified for custom datasets")

        return X, y

    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            "Specify a built-in dataset (iris, digits, wine, breast_cancer) "
            "or provide a dataset.path"
        )


def create_model_wrapper(config: ExaTuneConfig, hyperparameters: Dict[str, Any]):
    """
    Create a model wrapper from configuration and hyperparameters.

    Args:
        config: ExaTune configuration
        hyperparameters: Hyperparameter dictionary

    Returns:
        Model wrapper instance

    Raises:
        ValueError: If model type not supported
    """
    model_type = config.model.type.lower()

    if model_type == "sklearn":
        from exatune.models.sklearn_wrapper import SklearnModelWrapper

        # Extract just the class name from full path (e.g., 'RandomForestClassifier' from 'sklearn.ensemble.RandomForestClassifier')
        full_class_path = config.model.class_name
        model_class_name = full_class_path.split('.')[-1]

        return SklearnModelWrapper(
            model_class=model_class_name,
            hyperparameters=hyperparameters,
            task=config.model.task,
            random_state=config.experiment.random_seed,
        )

    elif model_type == "xgboost":
        from exatune.models.xgboost_wrapper import XGBoostModelWrapper

        return XGBoostModelWrapper(
            hyperparameters=hyperparameters,
            task=config.model.task,
            random_state=config.experiment.random_seed,
        )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def perform_cross_validation(
    model_wrapper, X: np.ndarray, y: np.ndarray, config: ExaTuneConfig
) -> Dict[str, Any]:
    """
    Perform cross-validation and compute metrics.

    Args:
        model_wrapper: Model wrapper instance
        X: Feature matrix
        y: Target vector
        config: ExaTune configuration

    Returns:
        Dictionary with CV results
    """
    # Determine CV strategy
    cv_folds = config.evaluation.cv_folds

    # Use stratified CV for classification
    if config.model.task == "classification":
        from sklearn.model_selection import StratifiedKFold

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.experiment.random_seed)
    else:
        from sklearn.model_selection import KFold

        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=config.experiment.random_seed)

    # Prepare scoring metrics
    scoring_metrics = [config.evaluation.scoring]
    if config.evaluation.additional_metrics:
        scoring_metrics.extend(config.evaluation.additional_metrics)

    # Create the model instance before cross-validation
    if model_wrapper.model is None:
        model_wrapper.model = model_wrapper._create_model()

    # Perform cross-validation
    cv_results = cross_validate(
        model_wrapper.model,
        X,
        y,
        cv=cv,
        scoring=scoring_metrics,
        return_train_score=True,
        n_jobs=1,  # Use single job per SLURM task
    )

    return cv_results


def generate_result(
    job_id: int,
    hyperparameters: Dict[str, Any],
    cv_results: Dict[str, Any],
    config: ExaTuneConfig,
    success: bool = True,
    error_message: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate result dictionary from CV results.

    Args:
        job_id: Job identifier
        hyperparameters: Hyperparameter configuration
        cv_results: Cross-validation results
        config: ExaTune configuration
        success: Whether training succeeded
        error_message: Error message if failed

    Returns:
        Result dictionary
    """
    # Generate configuration hash
    config_str = json.dumps(hyperparameters, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]

    result = {
        "job_id": job_id,
        "config_hash": config_hash,
        "hyperparameters": hyperparameters,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "success": success,
    }

    if success:
        # Extract primary metric scores
        primary_metric_key = f"test_{config.evaluation.scoring}"
        result["cv_scores"] = cv_results[primary_metric_key].tolist()
        result["mean_score"] = float(np.mean(cv_results[primary_metric_key]))
        result["std_score"] = float(np.std(cv_results[primary_metric_key]))

        # Extract training scores
        train_metric_key = f"train_{config.evaluation.scoring}"
        if train_metric_key in cv_results:
            result["train_scores"] = cv_results[train_metric_key].tolist()
            result["mean_train_score"] = float(np.mean(cv_results[train_metric_key]))

        # Extract additional metrics
        if config.evaluation.additional_metrics:
            result["additional_metrics"] = {}
            for metric in config.evaluation.additional_metrics:
                test_key = f"test_{metric}"
                if test_key in cv_results:
                    result["additional_metrics"][metric] = {
                        "mean": float(np.mean(cv_results[test_key])),
                        "std": float(np.std(cv_results[test_key])),
                    }

        # Add fit time
        result["fit_time_mean"] = float(np.mean(cv_results["fit_time"]))
        result["score_time_mean"] = float(np.mean(cv_results["score_time"]))

    else:
        # Failed job
        result["error_message"] = error_message
        result["mean_score"] = None
        result["std_score"] = None

    return result


def save_result(result: Dict[str, Any], output_dir: Path) -> None:
    """
    Save result to JSON and Parquet.

    Args:
        result: Result dictionary
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    job_id = result["job_id"]

    # Save individual JSON file
    json_path = output_dir / f"result_{job_id:06d}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    # Append to Parquet file (for efficient aggregation)
    try:
        # Flatten result for DataFrame
        flat_result = {
            "job_id": result["job_id"],
            "config_hash": result["config_hash"],
            "timestamp": result["timestamp"],
            "success": result["success"],
            "mean_score": result.get("mean_score"),
            "std_score": result.get("std_score"),
            "mean_train_score": result.get("mean_train_score"),
            "fit_time_mean": result.get("fit_time_mean"),
            "score_time_mean": result.get("score_time_mean"),
        }

        # Add hyperparameters as columns
        for param, value in result["hyperparameters"].items():
            flat_result[param] = value

        # Add additional metrics as columns
        if "additional_metrics" in result:
            for metric, values in result["additional_metrics"].items():
                flat_result[f"{metric}_mean"] = values["mean"]
                flat_result[f"{metric}_std"] = values["std"]

        # Create DataFrame
        df = pd.DataFrame([flat_result])

        # Append to Parquet
        parquet_path = output_dir / "results.parquet"
        if parquet_path.exists():
            # Append to existing file
            existing_df = pd.read_parquet(parquet_path)
            df = pd.concat([existing_df, df], ignore_index=True)

        df.to_parquet(parquet_path, index=False)

    except Exception as e:
        # Non-critical error - JSON is already saved
        print(f"Warning: Failed to save Parquet: {e}")


def main():
    """Main worker function."""
    parser = argparse.ArgumentParser(description="ExaTune worker script")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment configuration")
    parser.add_argument("--job-id", type=int, required=True, help="Job identifier")
    parser.add_argument(
        "--hyperparameters", type=str, required=True, help="Hyperparameters as JSON string"
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--random-seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    print("=" * 60)
    print(f"ExaTune Worker - Job {args.job_id}")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Output: {args.output_dir}")
    print(f"Random seed: {args.random_seed}")

    try:
        # Load configuration
        print("\n[1/5] Loading configuration...")
        config = load_config(args.config)

        # Parse hyperparameters
        print("[2/5] Parsing hyperparameters...")
        hyperparameters = json.loads(args.hyperparameters)
        print(f"Hyperparameters: {hyperparameters}")

        # Load dataset
        print("[3/5] Loading dataset...")
        X, y = load_dataset(config)
        print(f"Dataset shape: X={X.shape}, y={y.shape}")

        # Create model wrapper
        print("[4/5] Training model with cross-validation...")
        model_wrapper = create_model_wrapper(config, hyperparameters)

        # Perform cross-validation
        cv_results = perform_cross_validation(model_wrapper, X, y, config)

        # Generate result
        result = generate_result(args.job_id, hyperparameters, cv_results, config)

        print(f"[5/5] Saving results...")
        print(f"Mean {config.evaluation.scoring}: {result['mean_score']:.4f} "
              f"(±{result['std_score']:.4f})")

        # Save result
        save_result(result, args.output_dir)

        print("=" * 60)
        print("Job completed successfully!")
        print("=" * 60)
        sys.exit(0)

    except Exception as e:
        print("\n" + "=" * 60)
        print("ERROR: Job failed!")
        print("=" * 60)
        print(f"Error: {e}")
        print("\nTraceback:")
        traceback.print_exc()

        # Save error result
        try:
            hyperparameters = json.loads(args.hyperparameters)
            error_result = generate_result(
                args.job_id,
                hyperparameters,
                {},
                config if "config" in locals() else None,
                success=False,
                error_message=str(e),
            )
            save_result(error_result, args.output_dir)
        except Exception as save_error:
            print(f"\nFailed to save error result: {save_error}")

        sys.exit(1)


if __name__ == "__main__":
    main()