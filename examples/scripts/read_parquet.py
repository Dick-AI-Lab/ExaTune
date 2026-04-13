import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--path', required=True, help='Path to the parquet file')
parser.add_argument('--new_path', required=False, help='path to new file')
args = parser.parse_args()

df = pd.read_parquet(args.path)

if(args.new_path):
    to_csv = df.to_csv(args.new_path)

print(df.head())
print(df['config_hash'].head())
print(df.columns.tolist())
print( df.dtypes)