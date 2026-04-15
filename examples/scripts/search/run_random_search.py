#!/usr/bin/env python3
"""
ExaTune Random Search Runner
=============================
Runs random hyperparameter search from an ExaTune YAML config and saves
results in the same Parquet format used by ExaTune's grid search, so all
existing visualization tools (landscape 3D surface, heatmaps, violin plots)
work without modification.

Usage
-----
    python run_random_search.py --config iris_classification.yaml --n-samples 200
    python run_random_search.py --config iris_classification.yaml --n-samples 500 --seed 99
    python run_random_search.py --config iris_classification.yaml --n-samples 200 --output-dir ./results/random_search/iris

The output Parquet file can be loaded by:
    - visualize_my_results.py / visualize_results.py
    - dashboard.py / landscape.py
    - exatune/analysis/complete_analysis.py
"""

import argparse
import hashlib
import json
import random
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Hyperparameter sampling
# ---------------------------------------------------------------------------

def sample_hyperparameters(hp_config: dict, rng: random.Random) -> Dict[str, Any]:
    """
    Draw one random sample from the hyperparameter space defined in the YAML.

    Supports:
      type: categorical  → uniform draw from values list
      type: int          → randint(low, high)   (inclusive)
      type: float        → uniform(low, high)
      type: log_float    → log-uniform(low, high)
    """
    sample = {}
    for name, spec in hp_config.items():
        hp_type = spec.get("type", "categorical").lower()

        # If a values list is present, always sample from it (ExaTune standard)
        if "values" in spec:
            sample[name] = rng.choice(spec["values"])

        elif hp_type == "int":
            low, high = int(spec["low"]), int(spec["high"])
            sample[name] = rng.randint(low, high)

        elif hp_type == "float":
            low, high = float(spec["low"]), float(spec["high"])
            sample[name] = rng.uniform(low, high)

        elif hp_type in ("log_float", "loguniform"):
            low, high = float(spec["low"]), float(spec["high"])
            log_val = rng.uniform(np.log(low), np.log(high))
            sample[name] = float(np.exp(log_val))

        else:
            raise ValueError(
                f"Hyperparameter '{name}' has type '{hp_type}' but no 'values', 'low', or 'high' defined."
            )

    return sample


# ---------------------------------------------------------------------------
# Dataset loading  (mirrors worker.py exactly)
# ---------------------------------------------------------------------------

def load_dataset(dataset_cfg: dict) -> Tuple[np.ndarray, np.ndarray]:
    name = dataset_cfg.get("name", "").lower()

    if name in ("iris", "digits", "wine", "breast_cancer"):
        from sklearn import datasets as sk_ds
        loaders = {
            "iris": sk_ds.load_iris,
            "digits": sk_ds.load_digits,
            "wine": sk_ds.load_wine,
            "breast_cancer": sk_ds.load_breast_cancer,
        }
        data = loaders[name]()
        return data.data, data.target

    # Custom file-based dataset
    path = dataset_cfg.get("path")
    if path:
        dataset_path = Path(path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        suffix = dataset_path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(dataset_path)
        elif suffix == ".parquet":
            df = pd.read_parquet(dataset_path)
        elif suffix == ".npz":
            data = np.load(dataset_path)
            X = data["x_train"].reshape(-1, -1).astype("float32")
            y = data["y_train"]
            return X, y
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        target_col = dataset_cfg.get("target_column")
        if not target_col:
            raise ValueError("target_column must be specified for custom datasets")
        y = df[target_col].values
        X = df.drop(columns=[target_col]).values
        return X, y

    raise ValueError(f"Unknown dataset: '{name}'. Provide a built-in name or dataset.path.")


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------

def create_model(model_cfg: dict, hyperparams: Dict[str, Any], random_seed: int):
    """
    Instantiate the model from config, merging fixed_params and hyperparams.
    Mirrors worker.py's create_model_wrapper logic.
    """
    model_type = model_cfg.get("type", "sklearn").lower()
    fixed = model_cfg.get("fixed_params", {}) or {}

    # Merge: hyperparams override fixed_params (intentionally)
    all_params = {**fixed, **hyperparams}

    # Inject random_state where supported
    if model_type == "sklearn":
        class_path = model_cfg.get("class_name") or model_cfg.get("class")
        if not class_path:
            raise ValueError("model.class_name is required for sklearn models")

        parts = class_path.rsplit(".", 1)
        if len(parts) == 2:
            module_path, class_name = parts
            import importlib
            module = importlib.import_module(module_path)
            ModelClass = getattr(module, class_name)
        else:
            raise ValueError(f"Invalid class_name: {class_path!r} — expected 'module.ClassName'")

        # Only inject random_state for models that accept it
        NO_RANDOM_STATE = {"KNeighborsClassifier", "SVC", "SVR", "NearestNeighbors"}
        if class_name not in NO_RANDOM_STATE and "random_state" not in all_params:
            all_params["random_state"] = random_seed

        return ModelClass(**all_params)

    if model_type == "sklearn":
        class_path = model_cfg.get("class_name") or model_cfg.get("class")
        if not class_path:
            raise ValueError("model.class_name is required for sklearn models")

        # Dynamic import
        parts = class_path.rsplit(".", 1)
        if len(parts) == 2:
            module_path, class_name = parts
            import importlib
            module = importlib.import_module(module_path)
            ModelClass = getattr(module, class_name)
        else:
            raise ValueError(f"Invalid class_name: {class_path!r} — expected 'module.ClassName'")

        return ModelClass(**all_params)

    elif model_type == "xgboost":
        import xgboost as xgb
        task = model_cfg.get("task", "classification")
        if task == "classification":
            all_params.setdefault("verbosity", 0)
            all_params.setdefault("use_label_encoder", False)
            return xgb.XGBClassifier(**all_params)
        else:
            return xgb.XGBRegressor(**all_params)

    else:
        raise ValueError(f"Unsupported model type: {model_type!r}")


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def run_cv(
    model,
    X: np.ndarray,
    y: np.ndarray,
    eval_cfg: dict,
    model_cfg: dict,
    random_seed: int,
) -> Dict[str, Any]:
    cv_folds = eval_cfg.get("cv_folds", 5)
    task = model_cfg.get("task", "classification")
    primary = eval_cfg.get("scoring", "accuracy")
    additional = eval_cfg.get("additional_metrics") or []

    scoring = [primary] + additional

    if task == "classification":
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)

    return cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=1,
    )


# ---------------------------------------------------------------------------
# Result assembly  (matches worker.py's save_result flat schema exactly)
# ---------------------------------------------------------------------------

def build_flat_result(
    job_id: int,
    hyperparams: Dict[str, Any],
    cv_results: Dict[str, Any],
    eval_cfg: dict,
    success: bool = True,
    error_message: Optional[str] = None,
) -> Dict[str, Any]:
    config_str = json.dumps(hyperparams, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]

    primary = eval_cfg.get("scoring", "accuracy")
    additional = eval_cfg.get("additional_metrics") or []

    flat = {
        "job_id": job_id,
        "config_hash": config_hash,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "success": success,
        "mean_score": None,
        "std_score": None,
        "mean_train_score": None,
        "fit_time_mean": None,
        "score_time_mean": None,
    }

    # Hyperparameter columns
    for k, v in hyperparams.items():
        flat[k] = v

    if success and cv_results:
        test_key = f"test_{primary}"
        flat["mean_score"] = float(np.mean(cv_results[test_key]))
        flat["std_score"] = float(np.std(cv_results[test_key]))

        train_key = f"train_{primary}"
        if train_key in cv_results:
            flat["mean_train_score"] = float(np.mean(cv_results[train_key]))

        flat["fit_time_mean"] = float(np.mean(cv_results["fit_time"]))
        flat["score_time_mean"] = float(np.mean(cv_results["score_time"]))

        # Additional metrics — stored as metric_mean / metric_std columns
        for metric in additional:
            mk = f"test_{metric}"
            if mk in cv_results:
                flat[f"{metric}_mean"] = float(np.mean(cv_results[mk]))
                flat[f"{metric}_std"] = float(np.std(cv_results[mk]))
    else:
        flat["error_message"] = error_message or "unknown error"

    return flat


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_random_search(
    config_path: Path,
    n_samples: int,
    seed: int,
    output_dir: Optional[Path],
    verbose: bool,
) -> Path:
    print("=" * 65)
    print("  ExaTune Random Search")
    print("=" * 65)

    cfg = load_yaml(config_path)

    exp_cfg   = cfg.get("experiment", {})
    model_cfg = cfg.get("model", {})
    hp_cfg    = cfg.get("hyperparameters", {})
    ds_cfg    = cfg.get("dataset", {})
    eval_cfg  = cfg.get("evaluation", {})

    random_seed = seed if seed is not None else exp_cfg.get("random_seed", 42)
    exp_name    = exp_cfg.get("name", "random_search")

    # Output directory
    if output_dir is None:
        base = Path(exp_cfg.get("output_dir", f"./results/random_search/{exp_name}"))
        output_dir = base
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Config      : {config_path}")
    print(f"  Experiment  : {exp_name}")
    print(f"  Dataset     : {ds_cfg.get('name', ds_cfg.get('path', '?'))}")
    print(f"  Model       : {model_cfg.get('class_name', model_cfg.get('type', '?'))}")
    print(f"  n_samples   : {n_samples}")
    print(f"  seed        : {random_seed}")
    print(f"  Output dir  : {output_dir}")
    print("=" * 65)

    # Load dataset once
    print("\n[1/3] Loading dataset...")
    X, y = load_dataset(ds_cfg)
    print(f"  Shape: X={X.shape}, y={y.shape}")

    # RNG
    rng = random.Random(random_seed)
    np.random.seed(random_seed)

    # Run search
    print(f"\n[2/3] Running {n_samples} random evaluations...")
    rows: List[Dict[str, Any]] = []
    start_total = time.time()

    for i in range(n_samples):
        hyperparams = sample_hyperparameters(hp_cfg, rng)
        t0 = time.time()

        try:
            model = create_model(model_cfg, hyperparams, random_seed)
            cv_results = run_cv(model, X, y, eval_cfg, model_cfg, random_seed)
            flat = build_flat_result(i, hyperparams, cv_results, eval_cfg, success=True)
            elapsed = time.time() - t0

            if verbose:
                print(f"  [{i+1:>{len(str(n_samples))}}/{n_samples}] "
                      f"score={flat['mean_score']:.4f} ± {flat['std_score']:.4f}  "
                      f"params={hyperparams}  ({elapsed:.2f}s)")
            else:
                # Progress bar style
                pct = (i + 1) / n_samples
                bar = int(pct * 40)
                eta = (time.time() - start_total) / (i + 1) * (n_samples - i - 1)
                print(
                    f"\r  [{'█' * bar}{'░' * (40 - bar)}] "
                    f"{i+1}/{n_samples}  "
                    f"best={max((r['mean_score'] for r in rows + [flat] if r['mean_score'] is not None), default=0):.4f}  "
                    f"ETA {eta:.0f}s   ",
                    end="", flush=True,
                )

        except Exception as e:
            flat = build_flat_result(
                i, hyperparams, {}, eval_cfg,
                success=False, error_message=str(e)
            )
            if verbose:
                print(f"  [{i+1}/{n_samples}] FAILED: {e}")

        rows.append(flat)

    if not verbose:
        print()  # newline after progress bar

    elapsed_total = time.time() - start_total
    print(f"\n  Completed {n_samples} evaluations in {elapsed_total:.1f}s "
          f"({elapsed_total/n_samples:.2f}s/eval)")

    # Save results
    print("\n[3/3] Saving results...")
    df = pd.DataFrame(rows)

    parquet_path = output_dir / "results.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"  Parquet : {parquet_path}")

    # Also save a summary CSV for quick inspection
    csv_path = output_dir / "results_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"  CSV     : {csv_path}")

    # Print top-5 results
    successful = df[df["success"] == True].copy()
    if not successful.empty:
        top5 = successful.nlargest(5, "mean_score")
        print("\n  Top 5 configurations:")
        print(f"  {'Rank':<5} {'Score':>8} {'± Std':>8}  Hyperparameters")
        print(f"  {'-'*5} {'-'*8} {'-'*8}  {'-'*40}")
        for rank, (_, row) in enumerate(top5.iterrows(), 1):
            hp_str = ", ".join(
                f"{k}={row[k]}"
                for k in hp_cfg
                if k in row and row[k] is not None
            )
            print(f"  {rank:<5} {row['mean_score']:>8.4f} {row['std_score']:>8.4f}  {hp_str}")

        best = successful.loc[successful["mean_score"].idxmax()]
        print(f"\n  Best score : {best['mean_score']:.4f} ± {best['std_score']:.4f}")

    n_failed = int((df["success"] == False).sum())
    if n_failed:
        print(f"  Failed     : {n_failed}/{n_samples} evaluations")

    print("\n" + "=" * 65)
    print("  Done! Load results in your visualization tools:")
    print(f"  >>> df = pd.read_parquet('{parquet_path}')")
    print("=" * 65)

    return parquet_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ExaTune random hyperparameter search — compatible with all ExaTune visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 200 random samples from the Iris config
  python run_random_search.py --config iris_classification.yaml --n-samples 200

  # 500 samples with a fixed seed and custom output dir
  python run_random_search.py --config iris_classification.yaml --n-samples 500 \\
      --seed 99 --output-dir ./results/random_iris

  # Verbose per-sample output
  python run_random_search.py --config iris_classification.yaml --n-samples 100 --verbose

Visualizing results:
  # In visualize_my_results.py, point at the output Parquet:
  df = pd.read_parquet("./results/random_search/iris_rf_benchmark/results.parquet")

  # Or with the landscape dashboard:
  python -m http.server 5500  # from examples/viz/
        """,
    )
    parser.add_argument(
        "--config", "-c",
        type=Path, required=True,
        help="Path to ExaTune YAML configuration file",
    )
    parser.add_argument(
        "--n-samples", "-n",
        type=int, default=100,
        help="Number of random configurations to evaluate (default: 100)",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int, default=None,
        help="Random seed (overrides YAML experiment.random_seed)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path, default=None,
        help="Output directory for results (default: experiment.output_dir from YAML)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-sample results instead of a progress bar",
    )

    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: config file not found: {args.config}")
        sys.exit(1)
    if args.n_samples < 1:
        print("Error: --n-samples must be >= 1")
        sys.exit(1)

    try:
        run_random_search(
            config_path=args.config,
            n_samples=args.n_samples,
            seed=args.seed,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted — partial results may have been saved.")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()