#!/usr/bin/env python3
"""
ExaTune Bayesian Search Runner
================================
Runs Bayesian hyperparameter optimization from an ExaTune YAML config and saves
results in the same Parquet format used by ExaTune's grid search and random search,
so all existing visualization tools (landscape 3D surface, heatmaps, violin plots)
work without modification.

Uses scikit-optimize (skopt) with Gaussian Process surrogate model by default.
Install dependency:  pip install scikit-optimize

Usage
-----
    python run_bayesian_search.py --config iris_classification.yaml --n-samples 100
    python run_bayesian_search.py --config iris_classification.yaml --n-samples 100 --n-initial 20
    python run_bayesian_search.py --config iris_classification.yaml --n-samples 100 --acq-func EI
    python run_bayesian_search.py --config iris_classification.yaml --n-samples 100 --output-dir ./results/bayes/iris

The output Parquet file is identical in schema to grid search and random search outputs,
and can be loaded by:
    - visualize_my_results.py / visualize_results.py
    - dashboard.py / landscape.py
    - exatune/analysis/complete_analysis.py

How it works
------------
Bayesian optimization builds a surrogate model (Gaussian Process by default) over
the observed (hyperparameters → score) landscape, then uses an acquisition function
to decide which point to evaluate next — trading off exploration vs exploitation.
This means each evaluation is informed by all previous results, unlike random search.

For categorical hyperparameters (all ExaTune standard configs use these), skopt
treats them as discrete nominal dimensions.

Acquisition functions:
  LCB  - Lower Confidence Bound (default, good general purpose)
  EI   - Expected Improvement (aggressive, good when you want fast convergence)
  PI   - Probability of Improvement (very exploitative)
  gp_hedge - Probabilistic mix of LCB/EI/PI (robust, slightly slower)
"""

import argparse
import hashlib
import json
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
# Search space construction
# ---------------------------------------------------------------------------

def build_search_space(hp_config: dict):
    """
    Convert ExaTune YAML hyperparameter config into skopt dimensions.

    Returns:
        dimensions   : list of skopt dimension objects
        param_names  : list of parameter names (same order)
        cat_indices  : set of indices that are categorical (for decoding)
        cat_values   : dict mapping param_name -> list of possible values
    """
    try:
        from skopt.space import Categorical, Integer, Real
    except ImportError:
        print("ERROR: scikit-optimize is required. Install it with:")
        print("  pip install scikit-optimize")
        sys.exit(1)

    dimensions = []
    param_names = []
    cat_values = {}  # param_name -> values list (for categorical lookup)

    for name, spec in hp_config.items():
        hp_type = spec.get("type", "categorical").lower()
        param_names.append(name)

        # If a values list is present — treat as Categorical (ExaTune standard)
        if "values" in spec:
            vals = spec["values"]
            # skopt Categorical needs hashable items; convert None to the string "None"
            # We'll reverse this when decoding
            safe_vals = [str(v) if v is None else v for v in vals]
            dimensions.append(Categorical(safe_vals, name=name))
            cat_values[name] = vals  # keep originals for reverse-lookup

        elif hp_type == "int":
            dimensions.append(Integer(int(spec["low"]), int(spec["high"]), name=name))

        elif hp_type == "float":
            dimensions.append(Real(float(spec["low"]), float(spec["high"]), name=name))

        elif hp_type in ("log_float", "loguniform"):
            dimensions.append(
                Real(float(spec["low"]), float(spec["high"]), prior="log-uniform", name=name)
            )

        else:
            raise ValueError(
                f"Hyperparameter '{name}' has type '{hp_type}' but no 'values', 'low', or 'high'."
            )

    return dimensions, param_names, cat_values


def decode_point(point: list, param_names: list, cat_values: dict) -> Dict[str, Any]:
    """
    Convert a skopt suggestion (list of values) back to a hyperparameter dict,
    restoring None values that were stringified for skopt compatibility.
    """
    hp = {}
    for name, val in zip(param_names, point):
        if name in cat_values:
            # Find the original value (handles None -> "None" reversal)
            originals = cat_values[name]
            str_originals = [str(v) if v is None else v for v in originals]
            try:
                idx = str_originals.index(val)
                val = originals[idx]
            except ValueError:
                pass  # val is already fine
        hp[name] = val
    return hp


# ---------------------------------------------------------------------------
# Dataset loading  (identical to run_random_search.py)
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
            X = data["x_train"].reshape(len(data["x_train"]), -1).astype("float32")
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
# Model creation  (identical to run_random_search.py)
# ---------------------------------------------------------------------------

def create_model(model_cfg: dict, hyperparams: Dict[str, Any], random_seed: int):
    model_type = model_cfg.get("type", "sklearn").lower()
    fixed = model_cfg.get("fixed_params", {}) or {}
    all_params = {**fixed, **hyperparams}

    if model_type == "sklearn":
        class_path = model_cfg.get("class_name") or model_cfg.get("class")
        if not class_path:
            raise ValueError("model.class_name is required for sklearn models")
        parts = class_path.rsplit(".", 1)
        if len(parts) == 2:
            import importlib
            module = importlib.import_module(parts[0])
            ModelClass = getattr(module, parts[1])
        else:
            raise ValueError(f"Invalid class_name: {class_path!r}")

        NO_RANDOM_STATE = {"KNeighborsClassifier", "SVC", "SVR", "NearestNeighbors"}
        if parts[1] not in NO_RANDOM_STATE and "random_state" not in all_params:
            all_params["random_state"] = random_seed

        return ModelClass(**all_params)

    elif model_type == "xgboost":
        import xgboost as xgb
        task = model_cfg.get("task", "classification")
        all_params.setdefault("verbosity", 0)
        all_params.setdefault("use_label_encoder", False)
        if task == "classification":
            return xgb.XGBClassifier(**all_params)
        else:
            return xgb.XGBRegressor(**all_params)

    else:
        raise ValueError(f"Unsupported model type: {model_type!r}")
# ---------------------------------------------------------------------------
# Cross-validation  (identical to run_random_search.py)
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
# Result assembly  (identical schema to random search and grid search)
# ---------------------------------------------------------------------------

def build_flat_result(
    job_id: int,
    hyperparams: Dict[str, Any],
    cv_results: Dict[str, Any],
    eval_cfg: dict,
    success: bool = True,
    error_message: Optional[str] = None,
) -> Dict[str, Any]:
    config_str = json.dumps(hyperparams, sort_keys=True, default=str)
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

        for metric in additional:
            mk = f"test_{metric}"
            if mk in cv_results:
                flat[f"{metric}_mean"] = float(np.mean(cv_results[mk]))
                flat[f"{metric}_std"] = float(np.std(cv_results[mk]))
    else:
        flat["error_message"] = error_message or "unknown error"

    return flat


# ---------------------------------------------------------------------------
# Main Bayesian search loop
# ---------------------------------------------------------------------------

def run_bayesian_search(
    config_path: Path,
    n_samples: int,
    n_initial: int,
    acq_func: str,
    seed: int,
    output_dir: Optional[Path],
    verbose: bool,
) -> Path:
    try:
        from skopt import Optimizer
    except ImportError:
        print("ERROR: scikit-optimize is required. Install it with:")
        print("  pip install scikit-optimize")
        sys.exit(1)

    print("=" * 65)
    print("  ExaTune Bayesian Search")
    print("=" * 65)

    cfg = load_yaml(config_path)

    exp_cfg   = cfg.get("experiment", {})
    model_cfg = cfg.get("model", {})
    hp_cfg    = cfg.get("hyperparameters", {})
    ds_cfg    = cfg.get("dataset", {})
    eval_cfg  = cfg.get("evaluation", {})

    random_seed = seed if seed is not None else exp_cfg.get("random_seed", 42)
    exp_name    = exp_cfg.get("name", "bayesian_search")

    if output_dir is None:
        base = Path(exp_cfg.get("output_dir", f"./results/bayesian_search/{exp_name}"))
        output_dir = base
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cap n_initial to n_samples
    n_initial = min(n_initial, n_samples)

    print(f"  Config       : {config_path}")
    print(f"  Experiment   : {exp_name}")
    print(f"  Dataset      : {ds_cfg.get('name', ds_cfg.get('path', '?'))}")
    print(f"  Model        : {model_cfg.get('class_name', model_cfg.get('type', '?'))}")
    print(f"  n_samples    : {n_samples}  (initial random: {n_initial}, Bayesian: {n_samples - n_initial})")
    print(f"  acq_func     : {acq_func}")
    print(f"  seed         : {random_seed}")
    print(f"  Output dir   : {output_dir}")
    print("=" * 65)

    # Load dataset
    print("\n[1/3] Loading dataset...")
    X, y = load_dataset(ds_cfg)
    print(f"  Shape: X={X.shape}, y={y.shape}")

    # Build search space
    print("\n[2/3] Building search space and optimizer...")
    dimensions, param_names, cat_values = build_search_space(hp_cfg)
    print(f"  Dimensions   : {len(dimensions)}")
    for name, dim in zip(param_names, dimensions):
        print(f"    {name}: {dim}")

    # Initialise skopt optimizer
    optimizer = Optimizer(
        dimensions=dimensions,
        base_estimator="GP",       # Gaussian Process surrogate
        acq_func=acq_func,
        acq_optimizer="auto",
        n_initial_points=n_initial,
        random_state=random_seed,
    )

    # Run evaluations
    print(f"\n[3/3] Running {n_samples} evaluations...")
    rows: List[Dict[str, Any]] = []
    best_score = -np.inf
    start_total = time.time()

    for i in range(n_samples):
        # Ask optimizer for next point
        suggestion = optimizer.ask()
        hyperparams = decode_point(suggestion, param_names, cat_values)

        phase = "init" if i < n_initial else "bayes"
        t0 = time.time()

        try:
            model = create_model(model_cfg, hyperparams, random_seed)
            cv_results = run_cv(model, X, y, eval_cfg, model_cfg, random_seed)
            flat = build_flat_result(i, hyperparams, cv_results, eval_cfg, success=True)
            score = flat["mean_score"]
            elapsed = time.time() - t0

            # Tell optimizer the result (skopt minimises, so negate score)
            optimizer.tell(suggestion, -score)

            if score > best_score:
                best_score = score

            if verbose:
                print(f"  [{i+1:>{len(str(n_samples))}}/{n_samples}] [{phase}] "
                      f"score={score:.4f} ± {flat['std_score']:.4f}  "
                      f"best={best_score:.4f}  params={hyperparams}  ({elapsed:.2f}s)")
            else:
                pct = (i + 1) / n_samples
                bar = int(pct * 38)
                eta = (time.time() - start_total) / (i + 1) * (n_samples - i - 1)
                print(
                    f"\r  [{phase}] [{'█' * bar}{'░' * (38 - bar)}] "
                    f"{i+1}/{n_samples}  best={best_score:.4f}  ETA {eta:.0f}s   ",
                    end="", flush=True,
                )

        except Exception as e:
            flat = build_flat_result(
                i, hyperparams, {}, eval_cfg,
                success=False, error_message=str(e)
            )
            # Tell optimizer a very bad score so it avoids this region
            optimizer.tell(suggestion, 0.0)

            if verbose:
                print(f"  [{i+1}/{n_samples}] [{phase}] FAILED: {e}")

        rows.append(flat)

    if not verbose:
        print()

    elapsed_total = time.time() - start_total
    print(f"\n  Completed {n_samples} evaluations in {elapsed_total:.1f}s "
          f"({elapsed_total/n_samples:.2f}s/eval)")

    # Save results
    print("\n  Saving results...")
    df = pd.DataFrame(rows)

    parquet_path = output_dir / "results.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"  Parquet : {parquet_path}")

    csv_path = output_dir / "results_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"  CSV     : {csv_path}")

    # Print convergence summary
    successful = df[df["success"] == True].copy()
    if not successful.empty:
        # Show how score improved over iterations
        successful = successful.reset_index(drop=True)
        successful["best_so_far"] = successful["mean_score"].cummax()

        print("\n  Convergence (score improvement over evaluations):")
        checkpoints = [n_initial - 1] + [
            int(n_initial + (n_samples - n_initial) * f) - 1
            for f in [0.25, 0.5, 0.75, 1.0]
            if int(n_initial + (n_samples - n_initial) * f) - 1 < len(successful)
        ]
        checkpoints = sorted(set(max(0, c) for c in checkpoints))
        print(f"  {'Eval':>6}  {'Phase':<6}  {'Best score':>10}")
        print(f"  {'-'*6}  {'-'*6}  {'-'*10}")
        for c in checkpoints:
            phase = "init" if c < n_initial else "bayes"
            print(f"  {c+1:>6}  {phase:<6}  {successful.loc[c, 'best_so_far']:>10.4f}")

        # Top 5
        top5 = successful.nlargest(5, "mean_score")
        print(f"\n  Top 5 configurations:")
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
        description="ExaTune Bayesian hyperparameter search — compatible with all ExaTune visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Requires:  pip install scikit-optimize

Examples:
  # 100 evaluations (20 random init + 80 Bayesian)
  python run_bayesian_search.py --config iris_classification.yaml --n-samples 100

  # More initial random points before Bayesian kicks in
  python run_bayesian_search.py --config iris_classification.yaml --n-samples 150 --n-initial 30

  # Use Expected Improvement acquisition function
  python run_bayesian_search.py --config iris_classification.yaml --n-samples 100 --acq-func EI

  # Custom output dir, verbose per-sample output
  python run_bayesian_search.py --config iris_classification.yaml --n-samples 100 \\
      --output-dir ./results/bayes/iris --verbose

Acquisition functions:
  LCB      Lower Confidence Bound (default, balanced exploration/exploitation)
  EI       Expected Improvement (faster convergence, more exploitative)
  PI       Probability of Improvement (very exploitative)
  gp_hedge Probabilistic mix — robust but slower

Visualizing results:
  The output results.parquet is identical in schema to grid/random search outputs.
  Load it directly in visualize_my_results.py, dashboard.py, or landscape.py:
    df = pd.read_parquet("./results/bayesian_search/.../results.parquet")
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
        help="Total number of configurations to evaluate (default: 100)",
    )
    parser.add_argument(
        "--n-initial",
        type=int, default=10,
        help="Number of initial random evaluations before Bayesian kicks in (default: 10)",
    )
    parser.add_argument(
        "--acq-func",
        type=str, default="LCB",
        choices=["LCB", "EI", "PI", "gp_hedge"],
        help="Acquisition function (default: LCB)",
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
    if args.n_initial < 1:
        print("Error: --n-initial must be >= 1")
        sys.exit(1)

    try:
        run_bayesian_search(
            config_path=args.config,
            n_samples=args.n_samples,
            n_initial=args.n_initial,
            acq_func=args.acq_func,
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