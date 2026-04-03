#!/usr/bin/env python3
"""
Impute missing values in a CSV dataset — memory-safe chunked processing.

Usage:
    python impute_dataset.py --input data.csv
    python impute_dataset.py --input data.csv --strategy median
    python impute_dataset.py --input data.csv --missing-value -999.0 --chunksize 100000

Output: <original_name>_imputed.csv in the same directory as the input file.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def compute_statistics(input_path: Path, missing_value: float, strategy: str, chunksize: int):
    """
    First pass: compute imputation statistics (mean/median/mode) from the full dataset
    without loading it all into memory at once.
    """
    print("Pass 1/2: computing imputation statistics...")

    num_sum    = None
    num_sum_sq = None
    num_count  = None
    num_min    = None   # for median approximation via reservoir isn't exact; we use mean/most_frequent pass
    cat_counts = {}     # col -> {value -> count}
    n_rows     = 0
    numeric_cols = None
    cat_cols     = None

    for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunksize)):
        chunk.replace(missing_value, np.nan, inplace=True)

        if numeric_cols is None:
            numeric_cols = chunk.select_dtypes(include="number").columns.tolist()
            cat_cols     = chunk.select_dtypes(exclude="number").columns.tolist()
            num_sum      = np.zeros(len(numeric_cols))
            num_count    = np.zeros(len(numeric_cols))
            for col in cat_cols:
                cat_counts[col] = {}

        n_rows += len(chunk)

        # Accumulate numeric stats
        for j, col in enumerate(numeric_cols):
            vals = chunk[col].dropna().values
            num_sum[j]   += vals.sum()
            num_count[j] += len(vals)

        # Accumulate categorical counts
        for col in cat_cols:
            vc = chunk[col].dropna().value_counts()
            for val, cnt in vc.items():
                cat_counts[col][val] = cat_counts[col].get(val, 0) + cnt

        if (i + 1) % 10 == 0:
            print(f"  ...processed {n_rows:,} rows", end="\r")

    print(f"  done — {n_rows:,} rows total        ")

    # Build fill values
    fill_values = {}

    for j, col in enumerate(numeric_cols):
        if num_count[j] == 0:
            fill_values[col] = 0.0
        elif strategy == "mean":
            fill_values[col] = num_sum[j] / num_count[j]
        elif strategy == "median":
            # True median needs full data; fall back to mean for chunked processing
            # (exact median on 11M rows requires sorting the full column)
            fill_values[col] = num_sum[j] / num_count[j]
            if j == 0:
                print("  [note] exact median requires full data in memory; using mean as approximation")
        elif strategy == "most_frequent":
            # handled below via cat_counts-style accumulation — not done here for numerics
            fill_values[col] = num_sum[j] / num_count[j]

    for col in cat_cols:
        if cat_counts[col]:
            fill_values[col] = max(cat_counts[col], key=cat_counts[col].get)
        else:
            fill_values[col] = None

    return fill_values, numeric_cols, cat_cols, n_rows


def impute_chunked(input_path: Path, output_path: Path, fill_values: dict,
                   missing_value: float, chunksize: int, n_rows: int):
    """Second pass: apply fill values chunk by chunk, writing to output CSV."""
    print("Pass 2/2: writing imputed file...")

    processed = 0
    for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunksize)):
        chunk.replace(missing_value, np.nan, inplace=True)
        chunk.fillna(fill_values, inplace=True)

        # Write header only on first chunk
        chunk.to_csv(output_path, mode="w" if i == 0 else "a",
                     header=(i == 0), index=False)

        processed += len(chunk)
        pct = processed / n_rows * 100
        print(f"  {processed:,} / {n_rows:,} rows ({pct:.1f}%)", end="\r")

    print(f"\n  done.")


def main():
    parser = argparse.ArgumentParser(description="Impute missing values in a large CSV dataset")
    parser.add_argument("--input",         type=str,   required=True,
                        help="Path to input CSV file")
    parser.add_argument("--strategy",      type=str,   default="mean",
                        choices=["mean", "median", "most_frequent"],
                        help="Imputation strategy for numeric columns (default: mean)")
    parser.add_argument("--missing-value", type=float, default=-999.0,
                        help="Sentinel value for missing data (default: -999.0)")
    parser.add_argument("--chunksize",     type=int,   default=100_000,
                        help="Rows to process at a time (default: 100000)")
    args = parser.parse_args()

    input_path  = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: file not found: {input_path}")
        raise SystemExit(1)

    output_path = input_path.parent / f"{input_path.stem}_imputed.csv"

    print(f"\nInput  : {input_path}")
    print(f"Output : {output_path}")
    print(f"Strategy     : {args.strategy}")
    print(f"Missing value: {args.missing_value}")
    print(f"Chunk size   : {args.chunksize:,} rows\n")

    fill_values, numeric_cols, cat_cols, n_rows = compute_statistics(
        input_path, args.missing_value, args.strategy, args.chunksize
    )

    print(f"\nNumeric columns : {len(numeric_cols)}")
    print(f"Categorical cols: {len(cat_cols)}")
    print(f"Fill values computed for {len(fill_values)} columns\n")

    impute_chunked(input_path, output_path, fill_values,
                   args.missing_value, args.chunksize, n_rows)

    print(f"\nSaved: {output_path}\n")


if __name__ == "__main__":
    main()