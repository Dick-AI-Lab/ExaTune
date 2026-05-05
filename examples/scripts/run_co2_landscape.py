#!/usr/bin/env python3
"""
ExaTune Multi-Dataset CO2 Landscape Runner
============================================
Runs visualize_co2_landscape.py sequentially across multiple datasets,
auto-resolving Parquet paths from a shared results root organised as:

    results_root/{dataset}_{n_samples}/{search_type}/results.parquet

This matches the output structure produced by run_multi_search.py.

Usage
-----
    # All datasets found under results root, one sample count
    python run_co2_landscape_batch.py \\
        --results-root ./results/search/neurips/CO2_search \\
        --n-samples 100 \\
        --output-root ./co2_plots/neurips

    # Specific datasets only
    python run_co2_landscape_batch.py \\
        --results-root ./results/search/neurips/CO2_search \\
        --datasets adult diabetes wine \\
        --n-samples 100 \\
        --output-root ./co2_plots/neurips

    # Multiple sample counts
    python run_co2_landscape_batch.py \\
        --results-root ./results/search/neurips/CO2_search \\
        --n-samples 10 100 \\
        --output-root ./co2_plots/neurips

    # Also pass a config directory so HP names are read from YAML
    python run_co2_landscape_batch.py \\
        --results-root ./results/search/neurips/CO2_search \\
        --config-dir   ./configs/neurips \\
        --n-samples 100 \\
        --output-root  ./co2_plots/neurips

    # All pairs, skip Plotly (faster)
    python run_co2_landscape_batch.py \\
        --results-root ./results/search/neurips/CO2_search \\
        --n-samples 100 --all-pairs --no-plotly \\
        --output-root ./co2_plots/neurips

Windows example (PowerShell):
    python run_co2_landscape_batch.py `
        --results-root "C:\\...\\CO2_search" `
        --config-dir   "C:\\...\\configs\\neurips" `
        --n-samples 10 100 `
        --output-root  "C:\\...\\co2_plots\\neurips"
"""

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------

SEARCH_TYPES = ("grid", "random", "bayesian")


def find_datasets(results_root: Path, n_samples: int) -> List[str]:
    """
    Scan results_root for directories matching {dataset}_{n_samples}.
    Returns sorted list of dataset names.
    """
    suffix = f"_{n_samples}"
    datasets = []
    if not results_root.exists():
        return datasets
    for p in sorted(results_root.iterdir()):
        if p.is_dir() and p.name.endswith(suffix):
            datasets.append(p.name[: -len(suffix)])
    return datasets


def resolve_parquet(
    results_root: Path, dataset: str, n_samples: int, search_type: str
) -> Optional[Path]:
    """Return path to results.parquet if it exists, else None."""
    p = results_root / f"{dataset}_{n_samples}" / search_type / "results.parquet"
    return p if p.exists() else None


def find_config(config_dir: Optional[Path], dataset: str) -> Optional[Path]:
    """
    Look for a YAML config whose stem contains the dataset name
    (case-insensitive, partial match).
    """
    if config_dir is None or not config_dir.exists():
        return None
    for yaml_file in sorted(config_dir.glob("*.yaml")):
        if dataset.lower() in yaml_file.stem.lower():
            return yaml_file
    return None


# ---------------------------------------------------------------------------
# Single-dataset dispatch
# ---------------------------------------------------------------------------

def visualize_dataset(
    dataset: str,
    n_samples: int,
    results_root: Path,
    output_root: Path,
    config_dir: Optional[Path],
    extra_flags: List[str],
) -> bool:
    """
    Resolve parquet paths and call visualize_co2_landscape.run() directly.
    Returns True on success.
    """
    print()
    print("=" * 70)
    print(f"  Dataset: {dataset}   n_samples: {n_samples}")
    print("=" * 70)

    # Resolve parquet files
    parquets = {}
    for stype in SEARCH_TYPES:
        p = resolve_parquet(results_root, dataset, n_samples, stype)
        if p:
            parquets[stype] = p
            print(f"  Found {stype:<10}: {p}")
        else:
            print(f"  Missing {stype:<8}: "
                  f"{results_root / f'{dataset}_{n_samples}' / stype / 'results.parquet'}")

    if not parquets:
        print(f"  SKIP: No result files found for {dataset} n={n_samples}")
        return False

    # Resolve config
    config_path = find_config(config_dir, dataset)
    if config_path:
        print(f"  Config: {config_path}")
    else:
        print("  Config: not found (HP names will be inferred from data)")

    output_dir = output_root / f"{dataset}_{n_samples}"

    # Build args namespace and call directly (no subprocess overhead)
    try:
        import visualize_co2_landscape as viz

        class _Args:
            pass

        args = _Args()
        args.grid     = parquets.get("grid")
        args.random   = parquets.get("random")
        args.bayesian = parquets.get("bayesian")
        args.config   = config_path
        args.output_dir = str(output_dir)

        # Parse extra flags
        args.x_param        = None
        args.y_param        = None
        args.all_pairs      = "--all-pairs"      in extra_flags
        args.no_heatmap     = "--no-heatmap"     in extra_flags
        args.no_surface     = "--no-surface"     in extra_flags
        args.no_plotly      = "--no-plotly"      in extra_flags
        args.no_distribution= "--no-distribution" in extra_flags

        # Handle --x-param / --y-param if passed
        if "--x-param" in extra_flags:
            idx = extra_flags.index("--x-param")
            args.x_param = extra_flags[idx + 1]
        if "--y-param" in extra_flags:
            idx = extra_flags.index("--y-param")
            args.y_param = extra_flags[idx + 1]

        viz.run(args)
        return True

    except ImportError:
        print("ERROR: visualize_co2_landscape.py not found. "
              "Ensure it is in the same directory as this script.")
        return False
    except Exception as e:
        print(f"\nERROR: {e}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run CO₂ landscape visualizations across multiple datasets sequentially.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Directory structure expected (produced by run_multi_search.py):
  results_root/
    {dataset}_{n_samples}/
      random/results.parquet
      bayesian/results.parquet
      grid/results.parquet        (optional)

Examples:
  # Auto-discover all datasets at n=100
  python run_co2_landscape_batch.py \\
      --results-root ./results/CO2_search \\
      --n-samples 100 --output-root ./co2_plots

  # Specific datasets, two sample counts
  python run_co2_landscape_batch.py \\
      --results-root ./results/CO2_search \\
      --datasets adult wine diabetes \\
      --n-samples 10 100 --output-root ./co2_plots

  # All pairs, no Plotly
  python run_co2_landscape_batch.py \\
      --results-root ./results/CO2_search \\
      --n-samples 100 --all-pairs --no-plotly --output-root ./co2_plots

  # With config directory for HP names
  python run_co2_landscape_batch.py \\
      --results-root ./results/CO2_search \\
      --config-dir   ./configs/neurips \\
      --n-samples 100 --output-root ./co2_plots
        """,
    )

    parser.add_argument(
        "--results-root", "-r",
        type=Path, required=True,
        help="Root directory containing {dataset}_{n_samples}/{search_type}/ folders",
    )
    parser.add_argument(
        "--n-samples", "-n",
        type=int, nargs="+", required=True,
        metavar="N",
        help="Sample count(s) to visualize (e.g. 10 100)",
    )
    parser.add_argument(
        "--datasets", "-d",
        type=str, nargs="*", default=None,
        metavar="NAME",
        help="Dataset names to process (default: auto-discover all)",
    )
    parser.add_argument(
        "--config-dir", "-c",
        type=Path, default=None,
        metavar="DIR",
        help="Directory containing YAML configs (matched by dataset name)",
    )
    parser.add_argument(
        "--output-root", "-o",
        type=Path, default=Path("./co2_plots"),
        help="Root output directory (default: ./co2_plots)",
    )

    # Pass-through flags for visualize_co2_landscape
    parser.add_argument("--all-pairs",       action="store_true")
    parser.add_argument("--no-heatmap",      action="store_true")
    parser.add_argument("--no-surface",      action="store_true")
    parser.add_argument("--no-plotly",       action="store_true")
    parser.add_argument("--no-distribution", action="store_true")
    parser.add_argument("--x-param", type=str, default=None)
    parser.add_argument("--y-param",  type=str, default=None)

    args = parser.parse_args()

    if not args.results_root.exists():
        print(f"Error: results-root not found: {args.results_root}")
        sys.exit(1)

    # Add script dir to path so imports work
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    # Build extra-flags list to forward
    extra_flags = []
    if args.all_pairs:       extra_flags.append("--all-pairs")
    if args.no_heatmap:      extra_flags.append("--no-heatmap")
    if args.no_surface:      extra_flags.append("--no-surface")
    if args.no_plotly:       extra_flags.append("--no-plotly")
    if args.no_distribution: extra_flags.append("--no-distribution")
    if args.x_param:
        extra_flags += ["--x-param", args.x_param]
    if args.y_param:
        extra_flags += ["--y-param", args.y_param]

    # Build job list
    jobs = []
    for n_samples in args.n_samples:
        if args.datasets:
            datasets = args.datasets
        else:
            datasets = find_datasets(args.results_root, n_samples)
            if not datasets:
                print(f"WARNING: No dataset folders found for n_samples={n_samples} "
                      f"under {args.results_root}")
                continue
        for dataset in datasets:
            jobs.append((dataset, n_samples))

    if not jobs:
        print("No jobs to run. Check --results-root and --n-samples.")
        sys.exit(1)

    # Print plan
    print()
    print("=" * 70)
    print("  ExaTune CO₂ Landscape Batch Runner")
    print("=" * 70)
    print(f"  Results root : {args.results_root}")
    print(f"  Output root  : {args.output_root}")
    print(f"  Config dir   : {args.config_dir or 'not provided'}")
    print(f"  Total jobs   : {len(jobs)}")
    print(f"  Flags        : {extra_flags or 'none'}")
    print()
    print("  Planned jobs:")
    for j_num, (dataset, n_samples) in enumerate(jobs, 1):
        print(f"    [{j_num:>{len(str(len(jobs)))}}/{len(jobs)}]  "
              f"{dataset:<25}  n={n_samples}")
    print()

    results = []
    wall_start = time.time()

    for j_num, (dataset, n_samples) in enumerate(jobs, 1):
        print(f"\n[{j_num}/{len(jobs)}] {dataset}  n={n_samples}")
        t0 = time.time()
        ok = visualize_dataset(
            dataset=dataset,
            n_samples=n_samples,
            results_root=args.results_root,
            output_root=args.output_root,
            config_dir=args.config_dir,
            extra_flags=extra_flags,
        )
        elapsed = time.time() - t0
        results.append({"dataset": dataset, "n": n_samples,
                        "ok": ok, "elapsed": elapsed})
        print(f"\n  {'✓ OK' if ok else '✗ FAILED'}  ({elapsed:.1f}s)")

    # Final summary
    wall_elapsed = time.time() - wall_start
    n_ok   = sum(r["ok"] for r in results)
    n_fail = len(results) - n_ok

    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'#':<4} {'Dataset':<25} {'N':>6}  {'Status':<10} {'Time':>8}  Output")
    print(f"  {'-'*4} {'-'*25} {'-'*6}  {'-'*10} {'-'*8}  {'-'*30}")
    for j_num, r in enumerate(results, 1):
        status = "✓ OK" if r["ok"] else "✗ FAILED"
        out = args.output_root / f"{r['dataset']}_{r['n']}"
        print(f"  {j_num:<4} {r['dataset']:<25} {r['n']:>6}  "
              f"{status:<10} {r['elapsed']:>7.1f}s  {out}")
    print("-" * 70)
    print(f"  Passed: {n_ok}/{len(results)}   Failed: {n_fail}/{len(results)}   "
          f"Wall time: {wall_elapsed:.1f}s")
    print("=" * 70)

    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()

# # Auto-discover all datasets from your neurips CO2 run
# python run_co2_landscape_batch.py `
#     --results-root "C:\Users\katel\OneDrive\Desktop\ExaTune\examples\results\search\neurips\CO2_search" `
#     --config-dir   "C:\Users\katel\OneDrive\Desktop\ExaTune\examples\configs\neurips" `
#     --n-samples 10 100 `
#     --output-root  "C:\Users\katel\OneDrive\Desktop\ExaTune\examples\results\co2_plots\neurips"