#!/usr/bin/env python3
"""Compute the total time and CO2/energy emissions for an ExaTune experiment.

Usage:
    python compute_time.py --results PATH_TO_PARQUET
"""

import argparse
from pathlib import Path
import pandas as pd


def format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m {s:.1f}s"
    elif m > 0:
        return f"{m}m {s:.1f}s"
    return f"{s:.2f}s"


def print_section(title: str):
    print(f"\n{'─' * 45}")
    print(f"  {title}")
    print(f"{'─' * 45}")


def main():
    parser = argparse.ArgumentParser(description="Compute ExaTune experiment duration and emissions")
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to the results parquet or CSV file",
    )
    args = parser.parse_args()

    path = Path(args.results)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    print(f"\nLoaded {len(df)} configurations from {path.name}")

    # ── Wall clock duration ──────────────────────────────────────────────────
    print_section("Wall clock time")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    start = df["timestamp"].min()
    end   = df["timestamp"].max()
    wall_seconds = (end - start).total_seconds()

    print(f"  Start:    {start}")
    print(f"  End:      {end}")
    print(f"  Duration: {format_duration(wall_seconds)}")

    # ── Model fit time ───────────────────────────────────────────────────────
    if "fit_time_mean" in df.columns:
        print_section("Total model fit time")
        total_fit = df["fit_time_mean"].sum()
        print(f"  {format_duration(total_fit)}")
        print(f"  ({total_fit:.2f} s  /  {total_fit/60:.2f} min  /  {total_fit/3600:.4f} hr)")

    # ── Score time ───────────────────────────────────────────────────────────
    if "score_time_mean" in df.columns:
        print_section("Total model score time")
        total_score = df["score_time_mean"].sum()
        print(f"  {format_duration(total_score)}")
        print(f"  ({total_score:.2f} s  /  {total_score/60:.2f} min  /  {total_score/3600:.4f} hr)")

    # ── CO2 emissions ────────────────────────────────────────────────────────
    if "emissions_kg_co2" in df.columns:
        print_section("CO2 emissions")
        total_kg  = df["emissions_kg_co2"].sum()
        total_g   = total_kg * 1000
        total_mg  = total_kg * 1_000_000

        success_kg = df.loc[df["success"] == True, "emissions_kg_co2"].sum() if "success" in df.columns else None
        failed_kg  = df.loc[df["success"] == False, "emissions_kg_co2"].sum() if "success" in df.columns else None

        print(f"  Total:    {total_kg:.6e} kg  ({total_g:.6e} g  /  {total_mg:.4f} mg)")

        if success_kg is not None:
            print(f"  Successful jobs: {success_kg:.6e} kg")
            print(f"  Failed jobs:     {failed_kg:.6e} kg")

        # Equivalents
        km_driven     = total_kg / 0.21          # ~0.21 kg CO2/km average car
        phone_charges = total_kg / 0.00884       # ~8.84 g CO2 per full charge
        tree_hours    = total_g / (21_000 / 8760) # avg tree absorbs ~21 kg/year

        print(f"\n  Equivalents:")
        print(f"    ≈ {km_driven:.6f} km driven (avg car)")
        print(f"    ≈ {phone_charges:.6f} smartphone charges")
        print(f"    ≈ {tree_hours:.6f} tree-hours of CO2 absorption")

    # ── Energy consumption ───────────────────────────────────────────────────
    if "energy_consumed_kwh" in df.columns:
        print_section("Energy consumption")
        total_kwh = df["energy_consumed_kwh"].sum()
        total_wh  = total_kwh * 1000
        total_j   = total_kwh * 3_600_000

        print(f"  Total:    {total_kwh:.6e} kWh  ({total_wh:.6e} Wh  /  {total_j:.4e} J)")

        if "success" in df.columns:
            ok_kwh   = df.loc[df["success"] == True,  "energy_consumed_kwh"].sum()
            fail_kwh = df.loc[df["success"] == False, "energy_consumed_kwh"].sum()
            print(f"  Successful jobs: {ok_kwh:.6e} kWh")
            print(f"  Failed jobs:     {fail_kwh:.6e} kWh")

        # Per-job average
        avg_kwh = df["energy_consumed_kwh"].mean()
        print(f"  Avg per job: {avg_kwh:.6e} kWh")



    # ── Summary ──────────────────────────────────────────────────────────────
    print_section("Summary")
    total = len(df)
    if "success" in df.columns:
        n_ok   = df["success"].sum()
        n_fail = total - n_ok
        print(f"  Jobs: {total} total  |  {n_ok} succeeded  |  {n_fail} failed")
    else:
        print(f"  Jobs: {total} total")

    if "mean_score" in df.columns:
        best = df.loc[df["mean_score"].idxmax()]
        print(f"  Best score: {best['mean_score']:.4f}  (job {best.get('job_id', '?')}, config {best.get('config_hash', '?')})")

    print()


if __name__ == "__main__":
    main()