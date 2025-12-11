"""Parquet storage backend for ExaTune results."""

from pathlib import Path
from typing import Any, Dict

import pandas as pd


class ParquetStorage:
    """Storage backend using Parquet format for efficient storage."""

    def __init__(self, output_dir: Path):
        """
        Initialize Parquet storage backend.

        Args:
            output_dir: Directory for storing results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.output_dir / "results.parquet"

    def save_results(self, df: pd.DataFrame) -> None:
        """
        Save results DataFrame to Parquet.

        Args:
            df: Results DataFrame
        """
        df.to_parquet(self.results_file, index=False, compression="snappy")

    def load_results(self) -> pd.DataFrame:
        """
        Load results from Parquet file.

        Returns:
            Results DataFrame
        """
        if self.results_file.exists():
            return pd.read_parquet(self.results_file)
        return pd.DataFrame()

    def append_result(self, result: Dict[str, Any]) -> None:
        """
        Append a single result to the Parquet file.

        Args:
            result: Result dictionary
        """
        new_df = pd.DataFrame([result])

        if self.results_file.exists():
            existing_df = pd.read_parquet(self.results_file)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df

        combined_df.to_parquet(self.results_file, index=False, compression="snappy")

    def file_exists(self) -> bool:
        """
        Check if results file exists.

        Returns:
            True if file exists
        """
        return self.results_file.exists()
