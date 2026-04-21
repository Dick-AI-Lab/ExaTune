#!/usr/bin/env python3
"""
ExaTune Multi-Search Orchestrator
===================================
Run multiple instances of random search and/or Bayesian search across one or
more ExaTune YAML configs, saving all results under a shared output root
organised by dataset and sample count.

Output structure
----------------
    results/{dataset}_{n_samples}/{search_type}/
        results.parquet
        results_summary.csv
        codecarbon_log.csv   (if codecarbon installed)

    e.g.
    results/
      iris_100/
        random/
          results.parquet
        bayesian/
          results.parquet
      iris_200/
        random/
          results.parquet
        bayesian/
          results.parquet

Usage examples
--------------
    # 3 runs of random search on iris, 100 samples each
    python run_multi_search.py --config iris_classification.yaml \\
        --search random --n-samples 100 --runs 3

    # Random + Bayesian on iris, 100 and 200 samples
    python run_multi_search.py --config iris_classification.yaml \\
        --search random bayesian --n-samples 100 200

    # Multiple configs, multiple sample counts, both search types
    python run_multi_search.py \\
        --config iris_classification.yaml digits_classification.yaml \\
        --search random bayesian --n-samples 100 200 --runs 2

    # Custom output root and Bayesian settings
    python run_multi_search.py --config iris_classification.yaml \\
        --search bayesian --n-samples 150 --n-initial 20 --acq-func EI \\
        --output-root ./my_results --verbose
"""

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional

import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dataset_name_from_config(config_path: Path) -> str:
    """Extract dataset name from YAML config for use in directory names."""
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        ds = cfg.get("dataset", {})
        name = ds.get("name") or ""
        if name:
            return name.lower()
        # Fall back to filename stem
        path = ds.get("path", "")
        if path:
            return Path(path).stem.lower()
    except Exception:
        pass
    return config_path.stem.lower()


def _separator(char: str = "=", width: int = 70) -> str:
    return char * width


def _print_run_header(run_num: int, total_runs: int, search_type: str,
                      dataset: str, n_samples: int, config_path: Path) -> None:
    print()
    print(_separator())
    print(f"  Run {run_num}/{total_runs}  |  {search_type.upper()}  |  "
          f"dataset={dataset}  |  n_samples={n_samples}")
    print(f"  Config: {config_path}")
    print(_separator())


# ---------------------------------------------------------------------------
# Single-run dispatchers
# ---------------------------------------------------------------------------

def _run_random(
    config_path: Path,
    n_samples: int,
    seed: int,
    output_dir: Path,
    verbose: bool,
) -> bool:
    """Import and call run_random_search directly."""
    try:
        from run_random_search import run_random_search
        run_random_search(
            config_path=config_path,
            n_samples=n_samples,
            seed=seed,
            output_dir=output_dir,
            verbose=verbose,
        )
        return True
    except ImportError:
        print("ERROR: run_random_search.py not found on sys.path. "
              "Ensure it is in the same directory as this script.")
        return False
    except Exception as e:
        print(f"\nERROR in random search: {e}")
        traceback.print_exc()
        return False


def _run_bayesian(
    config_path: Path,
    n_samples: int,
    n_initial: int,
    acq_func: str,
    seed: int,
    output_dir: Path,
    verbose: bool,
) -> bool:
    """Import and call run_bayesian_search directly."""
    try:
        from run_bayesian_search import run_bayesian_search
        run_bayesian_search(
            config_path=config_path,
            n_samples=n_samples,
            n_initial=n_initial,
            acq_func=acq_func,
            seed=seed,
            output_dir=output_dir,
            verbose=verbose,
        )
        return True
    except ImportError:
        print("ERROR: run_bayesian_search.py not found on sys.path. "
              "Ensure it is in the same directory as this script.")
        return False
    except Exception as e:
        print(f"\nERROR in Bayesian search: {e}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_multi_search(
    configs: List[Path],
    search_types: List[str],
    n_samples_list: List[int],
    runs: int,
    base_seed: int,
    output_root: Path,
    n_initial: int,
    acq_func: str,
    verbose: bool,
) -> None:

    # Build the full job list up front so we can show a clear plan
    jobs = []
    for config_path in configs:
        dataset = _dataset_name_from_config(config_path)
        for n_samples in n_samples_list:
            for search_type in search_types:
                for run_idx in range(1, runs + 1):
                    # Each run gets a deterministic but distinct seed
                    seed = base_seed + (run_idx - 1) * 1000
                    out_dir = output_root / f"{dataset}_{n_samples}" / search_type
                    jobs.append({
                        "config_path": config_path,
                        "dataset": dataset,
                        "n_samples": n_samples,
                        "search_type": search_type,
                        "run_idx": run_idx,
                        "seed": seed,
                        "output_dir": out_dir,
                    })

    total = len(jobs)
    print()
    print(_separator("="))
    print("  ExaTune Multi-Search Orchestrator")
    print(_separator("="))
    print(f"  Total runs     : {total}")
    print(f"  Configs        : {[str(c) for c in configs]}")
    print(f"  Search types   : {search_types}")
    print(f"  Sample counts  : {n_samples_list}")
    print(f"  Runs per combo : {runs}")
    print(f"  Base seed      : {base_seed}")
    print(f"  Output root    : {output_root}")
    if "bayesian" in search_types:
        print(f"  Bayes n_initial: {n_initial}  acq_func: {acq_func}")
    print(_separator("="))

    # Confirm plan
    print("\n  Planned jobs:")
    for j_num, job in enumerate(jobs, 1):
        print(f"    [{j_num:>{len(str(total))}}/{total}]  "
              f"{job['dataset']:<20} "
              f"{job['search_type']:<10} "
              f"n={job['n_samples']:<6} "
              f"run={job['run_idx']}  "
              f"seed={job['seed']}  "
              f"→ {job['output_dir']}")

    print()

    results = []
    wall_start = time.time()

    for j_num, job in enumerate(jobs, 1):
        _print_run_header(
            j_num, total,
            job["search_type"], job["dataset"],
            job["n_samples"], job["config_path"],
        )

        t0 = time.time()

        if job["search_type"] == "random":
            ok = _run_random(
                config_path=job["config_path"],
                n_samples=job["n_samples"],
                seed=job["seed"],
                output_dir=job["output_dir"],
                verbose=verbose,
            )
        else:  # bayesian
            ok = _run_bayesian(
                config_path=job["config_path"],
                n_samples=job["n_samples"],
                n_initial=n_initial,
                acq_func=acq_func,
                seed=job["seed"],
                output_dir=job["output_dir"],
                verbose=verbose,
            )

        elapsed = time.time() - t0
        status = "✓ OK" if ok else "✗ FAILED"
        results.append({**job, "ok": ok, "elapsed": elapsed})
        print(f"\n  {status}  ({elapsed:.1f}s)")

    # Final summary
    wall_elapsed = time.time() - wall_start
    n_ok = sum(r["ok"] for r in results)
    n_fail = total - n_ok

    print()
    print(_separator("="))
    print("  SUMMARY")
    print(_separator("="))
    print(f"  {'Job':<5} {'Dataset':<20} {'Type':<10} {'Samples':<8} "
          f"{'Run':<5} {'Status':<10} {'Time':>8}  Output")
    print(f"  {'-'*5} {'-'*20} {'-'*10} {'-'*8} {'-'*5} {'-'*10} {'-'*8}  {'-'*30}")
    for j_num, r in enumerate(results, 1):
        status = "✓ OK" if r["ok"] else "✗ FAILED"
        print(f"  {j_num:<5} {r['dataset']:<20} {r['search_type']:<10} "
              f"{r['n_samples']:<8} {r['run_idx']:<5} {status:<10} "
              f"{r['elapsed']:>7.1f}s  {r['output_dir']}")

    print(_separator("-"))
    print(f"  Passed: {n_ok}/{total}    Failed: {n_fail}/{total}    "
          f"Total wall time: {wall_elapsed:.1f}s")
    print(_separator("="))

    if n_fail:
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ExaTune multi-search orchestrator — run random/Bayesian searches "
                    "across datasets and sample counts, saving to a shared output tree.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output structure:
  results/{dataset}_{n_samples}/{search_type}/
    results.parquet
    results_summary.csv

Examples:
  # 3 sequential random-search runs on iris, 100 samples each
  python run_multi_search.py --config iris_classification.yaml \\
      --search random --n-samples 100 --runs 3

  # Random + Bayesian on iris, two sample sizes
  python run_multi_search.py --config iris_classification.yaml \\
      --search random bayesian --n-samples 100 200

  # Two configs, both search types, 2 runs each at 150 samples
  python run_multi_search.py \\
      --config iris_classification.yaml digits_classification.yaml \\
      --search random bayesian --n-samples 150 --runs 2

  # Bayesian with custom acquisition function and more initial points
  python run_multi_search.py --config iris_classification.yaml \\
      --search bayesian --n-samples 200 --n-initial 30 --acq-func EI

  # Verbose per-sample output
  python run_multi_search.py --config iris_classification.yaml \\
      --search random --n-samples 100 --verbose
        """,
    )

    parser.add_argument(
        "--config", "-c",
        type=Path, nargs="+", required=True,
        metavar="YAML",
        help="One or more ExaTune YAML config files",
    )
    parser.add_argument(
        "--search",
        nargs="+", required=True,
        choices=["random", "bayesian"],
        metavar="TYPE",
        help="Search type(s) to run: random, bayesian (or both)",
    )
    parser.add_argument(
        "--n-samples", "-n",
        type=int, nargs="+", required=True,
        metavar="N",
        help="Number of samples per run (one or more values)",
    )
    parser.add_argument(
        "--runs", "-r",
        type=int, default=1,
        help="Number of sequential runs per (config × search_type × n_samples) combo (default: 1)",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int, default=42,
        help="Base random seed; each run increments by 1000 (default: 42)",
    )
    parser.add_argument(
        "--output-root", "-o",
        type=Path, default=Path("./results"),
        help="Root directory for all outputs (default: ./results)",
    )
    # Bayesian-specific
    parser.add_argument(
        "--n-initial",
        type=int, default=10,
        help="[Bayesian] Initial random evaluations before GP kicks in (default: 10)",
    )
    parser.add_argument(
        "--acq-func",
        type=str, default="LCB",
        choices=["LCB", "EI", "PI", "gp_hedge"],
        help="[Bayesian] Acquisition function (default: LCB)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Pass --verbose to each search run (per-sample output)",
    )

    args = parser.parse_args()

    # Validate configs exist
    missing = [c for c in args.config if not c.exists()]
    if missing:
        for m in missing:
            print(f"Error: config not found: {m}")
        sys.exit(1)

    # Deduplicate search types while preserving order
    seen = set()
    search_types = []
    for s in args.search:
        if s not in seen:
            search_types.append(s)
            seen.add(s)

    # Add the script directory to sys.path so imports of
    # run_random_search / run_bayesian_search work regardless of cwd
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    run_multi_search(
        configs=args.config,
        search_types=search_types,
        n_samples_list=args.n_samples,
        runs=args.runs,
        base_seed=args.seed,
        output_root=args.output_root,
        n_initial=args.n_initial,
        acq_func=args.acq_func,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()