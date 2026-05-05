#!/usr/bin/env python3
"""
ExaTune Optuna Search Runner
================================
Runs Optuna hyperparameter optimization from an ExaTune YAML config and saves
results in the same Parquet format used by ExaTune's grid search and random search,
so all existing visualization tools (landscape 3D surface, heatmaps, violin plots)
work without modification.

Install dependency:  pip install optuna

Usage
-----
    python run_optuna_local.py --config iris_classification.yaml --n-trials 100
    python run_optuna_local.py --config iris_classification.yaml --n-trials 100 --sampler TPE
    python run_optuna_local.py --config iris_classification.yaml --n-trials 100 --sampler Random
    python run_optuna_local.py --config iris_classification.yaml --n-trials 100 --output-dir ./results/optuna/iris

The output Parquet file is identical in schema to grid search and random search outputs,
and can be loaded by:
    - visualize_my_results.py / visualize_results.py
    - dashboard.py / landscape.py
    - exatune/analysis/complete_analysis.py

How it works
------------
Optuna builds a surrogate model over the observed (hyperparameters → score) landscape,
then uses an acquisition function to decide which point to evaluate next — trading off
exploration vs exploitation.

Samplers:
  TPE      - Tree-structured Parzen Estimator (default, good general purpose)
  CmaEs    - CMA-ES evolution strategy (good for continuous spaces)
  Random   - Random search (useful as a baseline comparison)
  GP       - Gaussian Process (similar to Bayesian/skopt approach)

Carbon tracking
---------------
Per-evaluation kg_co2 and energy_kwh are recorded via codecarbon and saved as
columns in the output Parquet. Install the dependency with:
    pip install codecarbon
If codecarbon is not installed, both columns default to None.
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
# Repo root on sys.path so worker utilities are importable
# ---------------------------------------------------------------------------

sys.path.insert(0, r"C:\Users\katel\OneDrive\Desktop\ExaTune\exatune\hpc")

# ---------------------------------------------------------------------------
# Optional codecarbon import
# ---------------------------------------------------------------------------

try:
    from codecarbon import EmissionsTracker
    _CODECARBON_AVAILABLE = True
except ImportError:
    _CODECARBON_AVAILABLE = False


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Dataset loading
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
            X = data["x_train"].reshape(-1, 784).astype("float32") / 255.0
            y = data["y_train"]
            return X, y
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        target_col = dataset_cfg.get("target_column")
        if not target_col:
            raise ValueError("target_column must be specified for custom datasets")
        y = df[target_col].values
        X = df.drop(columns=[target_col]).values

        if y.dtype == object:
            from sklearn.preprocessing import LabelEncoder
            y = LabelEncoder().fit_transform(y)

        return X, y

    raise ValueError(f"Unknown dataset: '{name}'. Provide a built-in name or dataset.path.")


# ---------------------------------------------------------------------------
# Model creation
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
# Result assembly
# ---------------------------------------------------------------------------

def build_flat_result(
    job_id: int,
    hyperparams: Dict[str, Any],
    cv_results: Dict[str, Any],
    eval_cfg: dict,
    success: bool = True,
    error_message: Optional[str] = None,
    kg_co2: Optional[float] = None,
    energy_kwh: Optional[float] = None,
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
        "kg_co2": kg_co2,
        "energy_kwh": energy_kwh,
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
# Carbon tracking helpers
# ---------------------------------------------------------------------------

def _make_tracker(output_dir: Path) -> Optional[object]:
    if not _CODECARBON_AVAILABLE:
        return None
    return EmissionsTracker(
        output_dir=str(output_dir),
        output_file="codecarbon_log.csv",
        log_level="error",
        save_to_file=True,
        tracking_mode="process",
    )


def _track_evaluation(tracker) -> Tuple[Optional[float], Optional[float]]:
    if tracker is None:
        return None, None
    try:
        emissions = tracker.stop()
        energy_kwh = tracker._total_energy.kWh if hasattr(tracker, "_total_energy") else None
        return (
            float(emissions) if emissions is not None else None,
            float(energy_kwh) if energy_kwh is not None else None,
        )
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Optuna search space construction
# ---------------------------------------------------------------------------

def build_trial_params(trial, hp_cfg: dict) -> Dict[str, Any]:
    """
    Translate ExaTune YAML hyperparameter config into Optuna trial suggestions.
    All ExaTune params carry a 'values' list regardless of declared type,
    so we always use suggest_categorical to sample from the explicit grid.
    """
    params = {}
    for name, spec in hp_cfg.items():
        if isinstance(spec, dict):
            values = spec.get("values")
        else:
            # Pydantic HyperparameterSpec object
            values = spec.values if hasattr(spec, "values") else None

        if values is not None:
            # Stringify None entries so Optuna can handle them; restore after
            safe_values = [str(v) if v is None else v for v in values]
            chosen = trial.suggest_categorical(name, safe_values)
            # Restore original None if it was stringified
            if chosen == "None":
                chosen = None
            params[name] = chosen
        else:
            raise ValueError(
                f"Hyperparameter '{name}' has no 'values' list. "
                "ExaTune Optuna runner requires explicit values for all parameters."
            )
    return params


# ---------------------------------------------------------------------------
# Sampler factory
# ---------------------------------------------------------------------------

def make_sampler(sampler_name: str, seed: int):
    import optuna
    name = sampler_name.lower()
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    elif name == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed)
    elif name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    elif name == "gp":
        return optuna.samplers.GPSampler(seed=seed)
    else:
        raise ValueError(f"Unknown sampler: {sampler_name!r}. Choose from: TPE, CmaEs, Random, GP")


# ---------------------------------------------------------------------------
# Main Optuna search loop
# ---------------------------------------------------------------------------

def run_optuna_search(
    config_path: Path,
    n_trials: int,
    sampler_name: str,
    seed: int,
    output_dir: Optional[Path],
    verbose: bool,
) -> Path:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("=" * 65)
    print("  ExaTune Optuna Search")
    print("=" * 65)

    cfg = load_yaml(config_path)

    exp_cfg   = cfg.get("experiment", {})
    model_cfg = cfg.get("model", {})
    hp_cfg    = cfg.get("hyperparameters", {})
    ds_cfg    = cfg.get("dataset", {})
    eval_cfg  = cfg.get("evaluation", {})

    random_seed = seed if seed is not None else exp_cfg.get("random_seed", 42)
    exp_name    = exp_cfg.get("name", "optuna_search")

    if output_dir is None:
        base = Path(exp_cfg.get("output_dir", f"./results/optuna/{exp_name}"))
        output_dir = base
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    carbon_status = "enabled" if _CODECARBON_AVAILABLE else "disabled (pip install codecarbon)"
    print(f"  Config       : {config_path}")
    print(f"  Experiment   : {exp_name}")
    print(f"  Dataset      : {ds_cfg.get('name', ds_cfg.get('path', '?'))}")
    print(f"  Model        : {model_cfg.get('class_name', model_cfg.get('type', '?'))}")
    print(f"  n_trials     : {n_trials}")
    print(f"  Sampler      : {sampler_name}")
    print(f"  seed         : {random_seed}")
    print(f"  Output dir   : {output_dir}")
    print(f"  Carbon       : {carbon_status}")
    print("=" * 65)

    print("\n[1/3] Loading dataset...")
    X, y = load_dataset(ds_cfg)
    print(f"  Shape: X={X.shape}, y={y.shape}")

    print("\n[2/3] Setting up Optuna study...")
    sampler = make_sampler(sampler_name, random_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    print(f"  Sampler      : {type(sampler).__name__}")
    print(f"  Direction    : maximize")

    print(f"\n[3/3] Running {n_trials} trials...")
    rows: List[Dict[str, Any]] = []
    best_score = -np.inf
    start_total = time.time()

    def objective(trial):
        nonlocal best_score

        hyperparams = build_trial_params(trial, hp_cfg)
        t0 = time.time()

        tracker = _make_tracker(output_dir)
        if tracker is not None:
            tracker.start()

        try:
            model = create_model(model_cfg, hyperparams, random_seed)
            cv_results = run_cv(model, X, y, eval_cfg, model_cfg, random_seed)
            kg_co2, energy_kwh = _track_evaluation(tracker)

            flat = build_flat_result(
                trial.number, hyperparams, cv_results, eval_cfg,
                success=True,
                kg_co2=kg_co2,
                energy_kwh=energy_kwh,
            )
            score = flat["mean_score"]
            elapsed = time.time() - t0

            if score > best_score:
                best_score = score

            if verbose:
                co2_str = f"  co2={kg_co2*1e6:.2f}µg" if kg_co2 is not None else ""
                print(f"  [{trial.number+1:>{len(str(n_trials))}}/{n_trials}] "
                      f"score={score:.4f} ± {flat['std_score']:.4f}  "
                      f"best={best_score:.4f}  params={hyperparams}  ({elapsed:.2f}s){co2_str}")
            else:
                pct = (trial.number + 1) / n_trials
                bar = int(pct * 38)
                eta = (time.time() - start_total) / (trial.number + 1) * (n_trials - trial.number - 1)
                print(
                    f"\r  [{'█' * bar}{'░' * (38 - bar)}] "
                    f"{trial.number+1}/{n_trials}  best={best_score:.4f}  ETA {eta:.0f}s   ",
                    end="", flush=True,
                )

            rows.append(flat)
            return score

        except Exception as e:
            _track_evaluation(tracker)
            flat = build_flat_result(
                trial.number, hyperparams, {}, eval_cfg,
                success=False, error_message=str(e),
                kg_co2=None, energy_kwh=None,
            )
            rows.append(flat)
            if verbose:
                print(f"  [{trial.number+1}/{n_trials}] FAILED: {e}")
            raise optuna.exceptions.TrialPruned()

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    if not verbose:
        print()

    elapsed_total = time.time() - start_total
    print(f"\n  Completed {len(rows)} evaluations in {elapsed_total:.1f}s "
          f"({elapsed_total/max(len(rows),1):.2f}s/eval)")

    print("\n  Saving results...")
    df = pd.DataFrame(rows)

    parquet_path = output_dir / "results.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"  Parquet : {parquet_path}")

    csv_path = output_dir / "results_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"  CSV     : {csv_path}")

    # Convergence summary
    successful = df[df["success"] == True].copy()
    if not successful.empty:
        successful = successful.reset_index(drop=True)
        successful["best_so_far"] = successful["mean_score"].cummax()

        print("\n  Convergence (score improvement over trials):")
        checkpoints = [int(n_trials * f) - 1 for f in [0.1, 0.25, 0.5, 0.75, 1.0]]
        checkpoints = sorted(set(max(0, c) for c in checkpoints if c < len(successful)))
        print(f"  {'Trial':>6}  {'Best score':>10}")
        print(f"  {'-'*6}  {'-'*10}")
        for c in checkpoints:
            print(f"  {c+1:>6}  {successful.loc[c, 'best_so_far']:>10.4f}")

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

    # Carbon summary
    if _CODECARBON_AVAILABLE and "kg_co2" in df.columns:
        total_co2 = df["kg_co2"].sum(skipna=True)
        total_kwh = df["energy_kwh"].sum(skipna=True)
        print(f"\n  Carbon summary (all evaluations):")
        print(f"    Total CO₂   : {total_co2*1000:.4f} g CO₂eq")
        print(f"    Total energy: {total_kwh*1000:.4f} Wh")
        print(f"    Per eval avg: {total_co2/n_trials*1e6:.2f} µg CO₂eq")

    n_failed = int((df["success"] == False).sum())
    if n_failed:
        print(f"  Failed     : {n_failed}/{n_trials} trials")

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
        description="ExaTune Optuna hyperparameter search — compatible with all ExaTune visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Requires:  pip install optuna

Examples:
  # 100 trials with default TPE sampler
  python run_optuna_local.py --config iris_classification.yaml --n-trials 100

  # Use Random sampler (useful as a baseline)
  python run_optuna_local.py --config iris_classification.yaml --n-trials 100 --sampler Random

  # Use CMA-ES sampler
  python run_optuna_local.py --config iris_classification.yaml --n-trials 100 --sampler CmaEs

  # Verbose output (per-trial results)
  python run_optuna_local.py --config iris_classification.yaml --n-trials 100 --verbose

  # Custom output directory
  python run_optuna_local.py --config iris_classification.yaml --n-trials 100 --output-dir ./results/optuna/iris

Carbon emissions:
  Install codecarbon to enable per-trial CO2 tracking:
    pip install codecarbon

Samplers:
  TPE      Tree-structured Parzen Estimator (default, balanced exploration/exploitation)
  CmaEs    CMA-ES evolution strategy (good for continuous spaces)
  Random   Pure random search (useful as a baseline)
  GP       Gaussian Process (similar to Bayesian/skopt approach)

Visualizing results:
  df = pd.read_parquet("./results/optuna/.../results.parquet")
        """,
    )
    parser.add_argument("--config", "-c", type=Path, required=True,
                        help="Path to ExaTune YAML configuration file")
    parser.add_argument("--n-trials", "-n", type=int, default=100,
                        help="Number of trials to run (default: 100)")
    parser.add_argument("--sampler", "-s", type=str, default="TPE",
                        choices=["TPE", "CmaEs", "Random", "GP"],
                        help="Optuna sampler to use (default: TPE)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides YAML experiment.random_seed)")
    parser.add_argument("--output-dir", "-o", type=Path, default=None,
                        help="Output directory for results (default: experiment.output_dir from YAML)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-trial results instead of a progress bar")

    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: config file not found: {args.config}")
        sys.exit(1)
    if args.n_trials < 1:
        print("Error: --n-trials must be >= 1")
        sys.exit(1)

    try:
        run_optuna_search(
            config_path=args.config,
            n_trials=args.n_trials,
            sampler_name=args.sampler,
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