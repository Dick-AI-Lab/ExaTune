import pandas as pd
from pathlib import Path
from exatune.analysis import compute_experiment_summary, format_summary_text, generate_report
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--metric", type=str, default=None,
                    help="Metric to analyze (auto-detected if not specified)")
parser.add_argument("--direction", type=str, default="maximize",
                    choices=["maximize", "minimize"])
parser.add_argument("--output-dir", type=str, default=None,
                    help="Output directory for report (overrides hardcoded default)")
args = parser.parse_args()

input_path = Path(args.data).resolve()
df = pd.read_parquet(input_path)

# Flatten dot-notation columns from json_normalize
df.columns = [c.replace(".", "_") for c in df.columns]
df.columns = [c.replace("hyperparameters_", "") for c in df.columns]
df.columns = [c.replace("additional_metrics_", "") for c in df.columns]

# Drop array-valued columns (list, dict, or numpy.ndarray) that break groupby
def is_array_col(series):
    for val in series.dropna():
        if isinstance(val, (list, dict, np.ndarray)):
            return True
    return False

array_cols = [c for c in df.columns if is_array_col(df[c])]
if array_cols:
    print(f"Dropping array-valued columns: {array_cols}")
    df = df.drop(columns=array_cols)

print("Detected columns:", df.columns.tolist())
print(f"mean_score range: {df['mean_score'].min():.4f} to {df['mean_score'].max():.4f}")

# Auto-detect metric if not provided
if args.metric:
    metric = args.metric
else:
    if "f1_macro_mean" in df.columns:
        metric = "f1_macro_mean"
    elif "f1_macro" in df.columns:
        metric = "f1_macro"
    else:
        metric = "mean_score"
    print(f"Auto-selected metric: {metric}")

# Auto-detect direction for negative scorers
direction = args.direction
if metric == "mean_score" and df["mean_score"].mean() < 0:
    direction = "maximize"
    print("Negative mean_score detected (neg_* scorer) — using direction=maximize")

summary = compute_experiment_summary(df, metric=metric, direction=direction)

output_dir = Path(args.output_dir) if args.output_dir else Path(r"C:\Users\katel\OneDrive\Desktop\ExaTune\exatune\analysis\neurips")
report_path = generate_report(
    df,
    output_path=output_dir,
    metric=metric,
    direction=direction,
    include_plots=True,
    format="markdown",
)
print(f"\nReport saved to: {report_path}")
print(format_summary_text(summary))