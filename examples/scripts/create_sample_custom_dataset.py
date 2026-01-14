#!/usr/bin/env python3
"""
Create sample custom datasets for testing ExaTune.

This script generates synthetic datasets in CSV and Parquet formats
that can be used to test ExaTune with custom data.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression


def create_classification_dataset(
    n_samples: int = 1000,
    n_features: int = 20,
    n_classes: int = 3,
    output_path: Path = None,
    format: str = "csv"
):
    """
    Create a synthetic classification dataset.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes
        output_path: Path to save the dataset
        format: File format ("csv" or "parquet")
    """
    print(f"\nCreating synthetic classification dataset...")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    print(f"  Classes: {n_classes}")

    # Generate synthetic data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.7),
        n_redundant=int(n_features * 0.2),
        n_classes=n_classes,
        random_state=42
    )

    # Create DataFrame
    feature_names = [f"feature_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    # Save to file
    if output_path is None:
        output_path = Path(f"custom_classification_dataset.{format}")

    if format == "csv":
        df.to_csv(output_path, index=False)
    elif format == "parquet":
        df.to_parquet(output_path, index=False)

    print(f"✓ Dataset saved to: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {', '.join(df.columns.tolist())}")

    # Print class distribution
    print(f"\nClass distribution:")
    class_counts = df["target"].value_counts().sort_index()
    for class_label, count in class_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  Class {class_label}: {count} ({percentage:.1f}%)")

    return df


def create_regression_dataset(
    n_samples: int = 1000,
    n_features: int = 20,
    output_path: Path = None,
    format: str = "csv"
):
    """
    Create a synthetic regression dataset.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        output_path: Path to save the dataset
        format: File format ("csv" or "parquet")
    """
    print(f"\nCreating synthetic regression dataset...")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")

    # Generate synthetic data
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.7),
        noise=10.0,
        random_state=42
    )

    # Create DataFrame
    feature_names = [f"feature_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    # Save to file
    if output_path is None:
        output_path = Path(f"custom_regression_dataset.{format}")

    if format == "csv":
        df.to_csv(output_path, index=False)
    elif format == "parquet":
        df.to_parquet(output_path, index=False)

    print(f"✓ Dataset saved to: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {', '.join(df.columns.tolist())}")

    # Print target statistics
    print(f"\nTarget statistics:")
    print(f"  Mean: {df['target'].mean():.2f}")
    print(f"  Std: {df['target'].std():.2f}")
    print(f"  Min: {df['target'].min():.2f}")
    print(f"  Max: {df['target'].max():.2f}")

    return df


def create_example_config(dataset_path: Path, task: str):
    """
    Create an example ExaTune configuration for the dataset.

    Args:
        dataset_path: Path to the dataset file
        task: "classification" or "regression"
    """
    config_path = dataset_path.parent / f"{dataset_path.stem}_config.yaml"

    if task == "classification":
        model_config = """  type: "sklearn"
  class_name: "sklearn.ensemble.RandomForestClassifier"
  task: "classification"
  fixed_params:
    random_state: 42"""

        hyperparameters = """  n_estimators:
    type: "discrete"
    values: [50, 100, 200]
  max_depth:
    type: "discrete"
    values: [5, 10, 15, null]
  min_samples_split:
    type: "discrete"
    values: [2, 5, 10]"""

        evaluation = """  cv_folds: 5
  cv_strategy: "stratified"
  scoring: "accuracy"
  additional_metrics:
    - "f1_weighted"
    - "precision_weighted"
    - "recall_weighted"""

    else:  # regression
        model_config = """  type: "sklearn"
  class_name: "sklearn.ensemble.RandomForestRegressor"
  task: "regression"
  fixed_params:
    random_state: 42"""

        hyperparameters = """  n_estimators:
    type: "discrete"
    values: [50, 100, 200]
  max_depth:
    type: "discrete"
    values: [5, 10, 15, null]
  min_samples_split:
    type: "discrete"
    values: [2, 5, 10]"""

        evaluation = """  cv_folds: 5
  cv_strategy: "kfold"
  scoring: "r2"
  additional_metrics:
    - "neg_mean_squared_error"
    - "neg_mean_absolute_error"""

    config_content = f"""# ExaTune Configuration for {dataset_path.name}
experiment:
  name: "{dataset_path.stem}"
  output_dir: "./results/{dataset_path.stem}"
  description: "Hyperparameter search on {dataset_path.name}"
  random_seed: 42

model:
{model_config}

hyperparameters:
{hyperparameters}

dataset:
  path: "{dataset_path.absolute()}"
  target_column: "target"
  test_size: 0.2
  random_state: 42
  stratify: {"true" if task == "classification" else "false"}

evaluation:
{evaluation}
  n_jobs: 1

slurm:
  partition: "compute"
  time: "01:00:00"
  memory: "8G"
  cpus_per_task: 2

storage:
  backend: "parquet"
"""

    with open(config_path, "w") as f:
        f.write(config_content)

    print(f"\n✓ Example config saved to: {config_path}")
    print(f"\nTo run this experiment:")
    print(f"  exatune run {config_path}")

    return config_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create sample custom datasets for ExaTune",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "task",
        choices=["classification", "regression"],
        help="Type of ML task"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of samples (default: 1000)"
    )
    parser.add_argument(
        "--features",
        type=int,
        default=20,
        help="Number of features (default: 20)"
    )
    parser.add_argument(
        "--classes",
        type=int,
        default=3,
        help="Number of classes for classification (default: 3)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for dataset"
    )
    parser.add_argument(
        "--format",
        choices=["csv", "parquet"],
        default="csv",
        help="Output format (default: csv)"
    )
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Also create an example ExaTune config file"
    )

    args = parser.parse_args()

    # Create dataset
    if args.task == "classification":
        df = create_classification_dataset(
            n_samples=args.samples,
            n_features=args.features,
            n_classes=args.classes,
            output_path=args.output,
            format=args.format
        )
    else:
        df = create_regression_dataset(
            n_samples=args.samples,
            n_features=args.features,
            output_path=args.output,
            format=args.format
        )

    # Get the actual output path
    if args.output is None:
        output_path = Path(f"custom_{args.task}_dataset.{args.format}")
    else:
        output_path = args.output

    # Create config if requested
    if args.create_config:
        create_example_config(output_path, args.task)

    print("\n" + "=" * 80)
    print("Dataset creation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
