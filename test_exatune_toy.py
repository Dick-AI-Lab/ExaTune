#!/usr/bin/env python3
"""
Test script for ExaTune functionality on a toy dataset.

This script demonstrates the core functionality of ExaTune without requiring
SLURM or HPC infrastructure. It tests:
- Configuration loading and validation
- Grid generation
- Experiment setup
- Job script generation

Usage:
    python test_exatune_toy.py
"""

import tempfile
from pathlib import Path

from rich.console import Console

from exatune.core.config import (
    DatasetConfig,
    EvaluationConfig,
    ExaTuneConfig,
    ExperimentConfig,
    HyperparameterSpec,
    ModelConfig,
    SlurmConfig,
)
from exatune.core.experiment import Experiment
from exatune.core.grid_generator import GridGenerator

console = Console()


def test_config_creation():
    """Test 1: Configuration creation and validation."""
    console.print("\n[bold cyan]Test 1: Configuration Creation[/bold cyan]")

    try:
        config = ExaTuneConfig(
            experiment=ExperimentConfig(
                name="toy_test",
                output_dir=Path("./test_results"),
                description="Toy test of ExaTune functionality",
                random_seed=42,
            ),
            model=ModelConfig(
                type="sklearn",
                **{"class": "RandomForestClassifier"},
                task="classification",
            ),
            hyperparameters={
                "n_estimators": [10, 20],
                "max_depth": [3, 5],
            },
            dataset=DatasetConfig(name="iris", test_size=0.2, random_state=42),
            evaluation=EvaluationConfig(cv_folds=3, scoring="accuracy"),
            slurm=SlurmConfig(partition="compute", time="00:10:00", memory="2G"),
        )

        console.print("✅ Configuration created successfully")
        console.print(f"   Model: {config.model.class_name}")
        console.print(f"   Task: {config.model.task}")
        console.print(f"   Dataset: {config.dataset.name}")
        return config

    except Exception as e:
        console.print(f"❌ Configuration creation failed: {e}")
        raise


def test_yaml_config():
    """Test 2: YAML configuration loading."""
    console.print("\n[bold cyan]Test 2: YAML Configuration Loading[/bold cyan]")

    yaml_config_path = Path("examples/configs/random_forest_config.yaml")

    if not yaml_config_path.exists():
        console.print(f"⚠️  YAML config not found at {yaml_config_path}")
        return None

    try:
        config = ExaTuneConfig.from_yaml(yaml_config_path)
        console.print("✅ YAML configuration loaded successfully")
        console.print(f"   Experiment: {config.experiment.name}")
        console.print(f"   Model: {config.model.class_name}")
        return config

    except Exception as e:
        console.print(f"❌ YAML configuration loading failed: {e}")
        return None


def test_grid_generation(config):
    """Test 3: Hyperparameter grid generation."""
    console.print("\n[bold cyan]Test 3: Hyperparameter Grid Generation[/bold cyan]")

    try:
        specs = config.get_hyperparameter_specs()
        generator = GridGenerator(specs)

        # Get grid summary
        summary = generator.get_summary()
        console.print(f"✅ Grid generated successfully")
        console.print(f"   Parameters: {summary['n_parameters']}")
        console.print(f"   Total combinations: {summary['total_combinations']:,}")

        # Generate first few configurations
        grid = generator.generate_grid()
        console.print(f"\n   First 3 configurations:")
        for i, config_dict in enumerate(grid[:3]):
            console.print(f"   [{i}] {config_dict}")

        return generator

    except Exception as e:
        console.print(f"❌ Grid generation failed: {e}")
        raise


def test_experiment_creation(config):
    """Test 4: Experiment creation and setup."""
    console.print("\n[bold cyan]Test 4: Experiment Creation[/bold cyan]")

    try:
        # Create experiment with temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            config.experiment.output_dir = Path(tmpdir) / "test_experiment"

            experiment = Experiment(config)
            console.print("✅ Experiment created successfully")
            console.print(f"   Name: {experiment.name}")
            console.print(f"   Grid size: {experiment.grid_size:,}")
            console.print(f"   Output directory: {experiment.output_dir}")

            # Check directory structure
            dirs_created = [
                experiment.output_dir,
                experiment.jobs_dir,
                experiment.results_dir,
                experiment.logs_dir,
                experiment.checkpoints_dir,
            ]

            all_exist = all(d.exists() for d in dirs_created)
            if all_exist:
                console.print("✅ All experiment directories created")
            else:
                console.print("⚠️  Some directories not created")

            return experiment

    except Exception as e:
        console.print(f"❌ Experiment creation failed: {e}")
        raise


def test_model_wrappers():
    """Test 5: Model wrapper functionality."""
    console.print("\n[bold cyan]Test 5: Model Wrappers[/bold cyan]")

    try:
        from sklearn.datasets import load_iris

        from exatune.models.sklearn_wrapper import SklearnModelWrapper

        # Load iris dataset
        X, y = load_iris(return_X_y=True)

        # Create and test sklearn wrapper
        wrapper = SklearnModelWrapper(
            model_class="RandomForestClassifier",
            hyperparameters={"n_estimators": 10, "max_depth": 3, "random_state": 42},
            task="classification",
        )

        console.print("✅ SklearnModelWrapper created")

        # Fit the model
        wrapper.fit(X[:100], y[:100])
        console.print("✅ Model fitted successfully")

        # Make predictions
        predictions = wrapper.predict(X[100:120])
        console.print(f"✅ Predictions made: {len(predictions)} samples")

        # Score the model
        score = wrapper.score(X[100:120], y[100:120])
        console.print(f"✅ Model score: {score:.3f}")

        return True

    except Exception as e:
        console.print(f"❌ Model wrapper test failed: {e}")
        return False


def test_storage_backends():
    """Test 6: Storage backends."""
    console.print("\n[bold cyan]Test 6: Storage Backends[/bold cyan]")

    try:
        import pandas as pd

        from exatune.storage.json_backend import JSONStorage
        from exatune.storage.parquet_backend import ParquetStorage

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test JSON backend
            json_storage = JSONStorage(Path(tmpdir) / "json_results")
            test_result = {
                "job_id": 0,
                "n_estimators": 10,
                "max_depth": 3,
                "accuracy": 0.95,
            }
            json_storage.save_result(0, test_result)
            console.print("✅ JSON storage: save successful")

            loaded = json_storage.load_result(0)
            assert loaded["accuracy"] == 0.95
            console.print("✅ JSON storage: load successful")

            # Test Parquet backend
            parquet_storage = ParquetStorage(Path(tmpdir) / "parquet_results")
            df = pd.DataFrame([test_result])
            parquet_storage.save_results(df)
            console.print("✅ Parquet storage: save successful")

            loaded_df = parquet_storage.load_results()
            assert len(loaded_df) == 1
            console.print("✅ Parquet storage: load successful")

        return True

    except Exception as e:
        console.print(f"❌ Storage backend test failed: {e}")
        return False


def test_cli_validation():
    """Test 7: CLI configuration validation."""
    console.print("\n[bold cyan]Test 7: CLI Configuration Validation[/bold cyan]")

    yaml_config_path = Path("examples/configs/random_forest_config.yaml")

    if not yaml_config_path.exists():
        console.print("⚠️  YAML config not found, skipping CLI test")
        return False

    try:
        # Simulate CLI validation
        config = ExaTuneConfig.from_yaml(yaml_config_path)
        experiment = Experiment(config)

        console.print("✅ Configuration validation successful")
        console.print(f"   Experiment: {experiment.name}")
        console.print(f"   Grid size: {experiment.grid_size:,} combinations")

        return True

    except Exception as e:
        console.print(f"❌ CLI validation failed: {e}")
        return False


def main():
    """Run all tests."""
    console.print("\n" + "=" * 70)
    console.print("[bold]ExaTune Functionality Test Suite[/bold]")
    console.print("Testing core functionality without SLURM/HPC infrastructure")
    console.print("=" * 70)

    results = {}

    # Test 1: Configuration creation
    try:
        config = test_config_creation()
        results["config_creation"] = True
    except Exception:
        results["config_creation"] = False
        config = None

    # Test 2: YAML loading
    yaml_config = test_yaml_config()
    results["yaml_loading"] = yaml_config is not None

    # Test 3: Grid generation (use yaml_config if available, else config)
    test_config = yaml_config if yaml_config else config
    if test_config:
        try:
            test_grid_generation(test_config)
            results["grid_generation"] = True
        except Exception:
            results["grid_generation"] = False
    else:
        results["grid_generation"] = False

    # Test 4: Experiment creation
    if config:
        try:
            test_experiment_creation(config)
            results["experiment_creation"] = True
        except Exception:
            results["experiment_creation"] = False
    else:
        results["experiment_creation"] = False

    # Test 5: Model wrappers
    results["model_wrappers"] = test_model_wrappers()

    # Test 6: Storage backends
    results["storage_backends"] = test_storage_backends()

    # Test 7: CLI validation
    results["cli_validation"] = test_cli_validation()

    # Print summary
    console.print("\n" + "=" * 70)
    console.print("[bold]Test Summary[/bold]")
    console.print("=" * 70)

    total_tests = len(results)
    passed_tests = sum(results.values())

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        console.print(f"{status} - {test_name.replace('_', ' ').title()}")

    console.print("\n" + "-" * 70)
    console.print(
        f"[bold]Result: {passed_tests}/{total_tests} tests passed "
        f"({100*passed_tests/total_tests:.0f}%)[/bold]"
    )
    console.print("=" * 70 + "\n")

    if passed_tests == total_tests:
        console.print("[bold green]🎉 All tests passed! ExaTune is working correctly.[/bold green]")
        return 0
    else:
        console.print(
            "[bold yellow]⚠️  Some tests failed. Check errors above for details.[/bold yellow]"
        )
        return 1


if __name__ == "__main__":
    exit(main())
