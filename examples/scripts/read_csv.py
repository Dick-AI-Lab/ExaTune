import pandas as pd
from pathlib import Path

input_path = r"C:\Users\katel\OneDrive\Desktop\ExaTune\examples\parquet\parquet_neurips\wine_clean50k.csv"

df = pd.read_csv(input_path)
print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head())

output_path = Path(input_path).with_suffix(".parquet")
df.to_parquet(output_path, index=False)
print(f"Saved to: {output_path}")