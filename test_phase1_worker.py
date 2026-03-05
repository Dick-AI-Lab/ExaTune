#!/usr/bin/env python
"""
Quick integration test for Phase 1 implementation.
Tests the worker script locally without SLURM.
"""

import json
import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from exatune.core.config import ExaTuneConfig
from exatune.hpc.worker import (
    load_dataset,
    create_model_wrapper,
    perform_cross_validation,
    generate_result,
    save_result,
)


def test_worker_components():
    """Test individual worker components."""
    print("=" * 60)
    print("Testing Phase 1 Worker Implementation")
    print("=" * 60)

    # Create a minimal config
    config_dict = {
        "experiment": {
            "name": "test_experiment",
            "description": "Test",
            "output_dir": "/tmp/test_output",
            "random_seed": 42,
        },
        "dataset": {
            "name": "iris",
        },
        "model": {
            "type": "sklearn",
            "class_name": "sklearn.ensemble.RandomForestClassifier",
            "task": "classification",
        },
        "hyperparameters": {
            "n_estimators": [10, 50, 100],
            "max_depth": [3, 5, 7],
        },
        "evaluation": {
            "cv_folds": 3,
            "scoring": "accuracy",
            "additional_metrics": ["f1_weighted"],
        },
        "slurm": {
            "partition": "default",
            "time": "00:30:00",
            "memory": "4G",
            "cpus_per_task": 1,
            "nodes": 1,
        },
    }

    config = ExaTuneConfig.from_dict(config_dict)

    # Test 1: Load dataset
    print("\n[1/5] Testing dataset loading...")
    try:
        X, y = load_dataset(config)
        print(f"✓ Loaded dataset: X shape={X.shape}, y shape={y.shape}")
        assert X.shape[0] > 0, "Dataset is empty"
        assert len(y) == X.shape[0], "X and y shape mismatch"
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return False

    # Test 2: Create model wrapper
    print("\n[2/5] Testing model wrapper creation...")
    hyperparameters = {"n_estimators": 50, "max_depth": 5}
    try:
        model_wrapper = create_model_wrapper(config, hyperparameters)
        print(f"✓ Created model wrapper: {type(model_wrapper).__name__}")
        assert model_wrapper is not None
    except Exception as e:
        print(f"✗ Failed to create model wrapper: {e}")
        return False

    # Test 3: Perform cross-validation
    print("\n[3/5] Testing cross-validation...")
    try:
        cv_results = perform_cross_validation(model_wrapper, X, y, config)
        print(f"✓ Cross-validation completed")
        print(f"  Folds: {len(cv_results['test_accuracy'])}")
        print(f"  Mean accuracy: {cv_results['test_accuracy'].mean():.4f}")
        assert "test_accuracy" in cv_results
        assert len(cv_results["test_accuracy"]) == config.evaluation.cv_folds
    except Exception as e:
        print(f"✗ Failed cross-validation: {e}")
        return False

    # Test 4: Generate result
    print("\n[4/5] Testing result generation...")
    try:
        result = generate_result(
            job_id=0,
            hyperparameters=hyperparameters,
            cv_results=cv_results,
            config=config,
            success=True,
        )
        print(f"✓ Generated result")
        print(f"  Job ID: {result['job_id']}")
        print(f"  Config hash: {result['config_hash']}")
        print(f"  Mean score: {result['mean_score']:.4f}")
        print(f"  Std score: {result['std_score']:.4f}")
        assert result["success"] == True
        assert result["mean_score"] is not None
        assert "hyperparameters" in result
    except Exception as e:
        print(f"✗ Failed to generate result: {e}")
        return False

    # Test 5: Save result
    print("\n[5/5] Testing result saving...")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_result(result, output_dir)

            # Check JSON file was created
            json_file = output_dir / "result_000000.json"
            assert json_file.exists(), "JSON file not created"

            # Load and verify JSON
            with open(json_file, "r") as f:
                loaded_result = json.load(f)
            assert loaded_result["job_id"] == result["job_id"]
            assert loaded_result["mean_score"] == result["mean_score"]

            print(f"✓ Saved result to {json_file}")

            # Check Parquet file was created
            parquet_file = output_dir / "results.parquet"
            
            assert parquet_file.exists(), "Parquet file not created"
            print(f"✓ Saved result to {parquet_file}")

    except Exception as e:
        print(f"✗ Failed to save result: {e}")
        return False

    print("\n" + "=" * 60)
    print("✓ All worker component tests passed!")
    print("=" * 60)
    return True


def test_experiment_methods():
    """Test experiment methods (without SLURM)."""
    print("\n" + "=" * 60)
    print("Testing Experiment Methods")
    print("=" * 60)

    from exatune.core.experiment import Experiment

    # Create a minimal config
    config_dict = {
        "experiment": {
            "name": "test_experiment",
            "description": "Test",
            "output_dir": "/tmp/test_exatune_experiment",
            "random_seed": 42,
        },
        "dataset": {
            "name": "iris",
        },
        "model": {
            "type": "sklearn",
            "class_name": "sklearn.ensemble.RandomForestClassifier",
            "task": "classification",
        },
        "hyperparameters": {
            "n_estimators": [10, 50],
            "max_depth": [3, 5],
        },
        "evaluation": {
            "cv_folds": 2,
            "scoring": "accuracy",
        },
        "slurm": {
            "partition": "default",
            "time": "00:30:00",
            "memory": "4G",
            "cpus_per_task": 1,
            "nodes": 1,
        },
    }

    # Test 1: Create experiment
    print("\n[1/3] Testing experiment creation...")
    try:
        experiment = Experiment.from_dict(config_dict)
        print(f"✓ Created experiment: {experiment.name}")
        print(f"  Grid size: {experiment.grid_size}")
        assert experiment.grid_size == 4  # 2 x 2 combinations
    except Exception as e:
        print(f"✗ Failed to create experiment: {e}")
        return False

    # Test 2: Generate job scripts
    print("\n[2/3] Testing job script generation...")
    try:
        n_jobs = experiment.generate_jobs()
        print(f"✓ Generated {n_jobs} job scripts")
        assert n_jobs == experiment.grid_size

        # Verify scripts were created
        job_scripts = list(experiment.jobs_dir.glob("job_*.sh"))
        assert len(job_scripts) == n_jobs
        print(f"  Job scripts in: {experiment.jobs_dir}")

        # Check one script
        with open(job_scripts[0], "r") as f:
            script_content = f.read()
        assert "#!/bin/bash" in script_content
        assert "worker.py" in script_content
        print(f"✓ Job scripts are valid")

    except Exception as e:
        print(f"✗ Failed to generate jobs: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Checkpoint functionality
    print("\n[3/3] Testing checkpoint save/load...")
    try:
        experiment.save_checkpoint("test_checkpoint")
        print(f"✓ Saved checkpoint")

        # Load checkpoint
        experiment.load_checkpoint("test_checkpoint")
        print(f"✓ Loaded checkpoint")

    except Exception as e:
        print(f"✗ Failed checkpoint operations: {e}")
        return False

    print("\n" + "=" * 60)
    print("✓ All experiment method tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = True

    # Test worker components
    if not test_worker_components():
        success = False

    # Test experiment methods
    if not test_experiment_methods():
        success = False

    if success:
        print("\n" + "=" * 60)
        print("✓✓✓ ALL PHASE 1 TESTS PASSED! ✓✓✓")
        print("=" * 60)
        print("\nPhase 1 Implementation Complete:")
        print("  ✓ Worker script functional")
        print("  ✓ Job script generation working")
        print("  ✓ Experiment orchestration ready")
        print("  ✓ Result collection implemented")
        print("\nNext steps:")
        print("  - Deploy to SLURM cluster for full testing")
        print("  - Implement Phase 2 (comprehensive testing)")
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("✗ SOME TESTS FAILED")
        print("=" * 60)
        sys.exit(1)
