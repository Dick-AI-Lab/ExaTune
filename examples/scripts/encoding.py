#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
args = parser.parse_args()

input_path = Path(args.input).resolve()
output_path = input_path.parent / f"{input_path.stem}_encoded.csv"

print(f"Loading {input_path}...")
df = pd.read_csv(input_path)

print(f"Label values before: {df['Label'].unique()}")
df['Label'] = df['Label'].map({'s': 1, 'b': 0})
print(f"Label values after:  {df['Label'].unique()}")

df.to_csv(output_path, index=False)
print(f"Saved: {output_path}")