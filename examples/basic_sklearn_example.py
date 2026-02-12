"""
Basic example of using ExaTune with scikit-learn.

This example demonstrates how to set up and run a hyperparameter search
experiment using ExaTune with a Random Forest classifier.
"""

from pathlib import Path

from exatune.core.config import (
    ExaTuneConfig,
    ExperimentConfig,
    ModelConfig,
    DatasetConfig,
    EvaluationConfig,
    SlurmConfig,
)
from exatune.core.experiment import Experiment


def main():
    """Run a basic hyperparameter search experiment."""

    # Define experiment configuration
    config = ExaTuneConfig(
        experiment=ExperimentConfig(
            name="rf_iris_example",
            output_dir=Path("./results/rf_iris_example"),
            description="Example Random Forest hyperparameter search on Iris dataset",
            tags=["example", "classification", "random-forest"],
            random_seed=42,
        ),
        model=ModelConfig(
            type="sklearn",
            **{"class": "RandomForestClassifier"},
            task="classification",
            fixed_params={
                "criterion": "gini",
                "bootstrap": True,
            }
        ),
        hyperparameters={
            "n_estimators": [10, 50, 100],
            "max_depth": [3, 5, 10, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        dataset=DatasetConfig(
            name="iris",
            test_size=0.2,
            random_state=42,
            stratify=True,
        ),
        evaluation=EvaluationConfig(
            cv_folds=5,
            cv_strategy="stratified",
            scoring="accuracy",
            additional_metrics=["f1_macro", "precision_macro", "recall_macro"],
        ),
        slurm=SlurmConfig(
            partition="cpubase_bynode_b5",
            time="00:30:00",
            memory="4G",
            cpus_per_task=2,
            email_type="FAIL",
        )
    )

    # Create experiment
    experiment = Experiment(config)

    # Print experiment summary
    print("\n" + "="*60)
    print("ExaTune Example: Random Forest on Iris Dataset")
    print("="*60 + "\n")

    experiment.print_summary()

    # Generate job scripts
    print("\nGenerating job scripts...")
    n_jobs = experiment.generate_jobs()
    print(f"Generated {n_jobs} job scripts")

    # In a real scenario, you would submit jobs here
    # For this example, we'll just save the configuration
    print(f"\nExperiment configuration saved to: {experiment.output_dir}")
    print("\nTo submit jobs to SLURM, run:")
    print(f"  exatune run {experiment.output_dir}/config.yaml")


if __name__ == "__main__":
    main()
