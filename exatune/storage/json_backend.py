"""JSON storage backend for ExaTune results."""

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


class JSONStorage:
    """Storage backend using JSON format."""

    def __init__(self, output_dir: Path):
        """
        Initialize JSON storage backend.

        Args:
            output_dir: Directory for storing results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_result(self, job_id: int, result: Dict[str, Any]) -> None:
        """
        Save a single job result.

        Args:
            job_id: Job identifier
            result: Result dictionary
        """
        result_path = self.output_dir / f"result_{job_id:06d}.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

    def load_result(self, job_id: int) -> Dict[str, Any]:
        """
        Load a single job result.

        Args:
            job_id: Job identifier

        Returns:
            Result dictionary
        """
        result_path = self.output_dir / f"result_{job_id:06d}.json"
        with open(result_path, "r") as f:
            return json.load(f)

    def load_all_results(self) -> pd.DataFrame:
        """
        Load all results into a DataFrame.

        Returns:
            DataFrame with all results
        """
        results = []
        for result_file in sorted(self.output_dir.glob("result_*.json")):
            with open(result_file, "r") as f:
                results.append(json.load(f))

        return pd.DataFrame(results) if results else pd.DataFrame()

    def result_exists(self, job_id: int) -> bool:
        """
        Check if a result exists.

        Args:
            job_id: Job identifier

        Returns:
            True if result exists
        """
        result_path = self.output_dir / f"result_{job_id:06d}.json"
        return result_path.exists()
