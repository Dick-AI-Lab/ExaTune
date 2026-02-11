"""
Tests for storage backends (exatune.storage).
"""

import pytest
import json
import tempfile
from pathlib import Path
import pandas as pd

from exatune.storage.json_backend import JSONStorage
from exatune.storage.parquet_backend import ParquetStorage


class TestJSONStorage:
    """Tests for JSON storage backend."""

    def test_save_single_result(self):
        """Test saving a single result to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = JSONStorage(Path(tmpdir))

            result = {
                "job_id": 0,
                "config_hash": "abc123",
                "hyperparameters": {"max_depth": 5},
                "success": True,
                "mean_score": 0.95,
                "std_score": 0.02
            }

            backend.save_result(job_id=0, result=result)

            # Check file was created
            result_file = Path(tmpdir) / "result_000000.json"
            assert result_file.exists()

            # Check content
            with open(result_file) as f:
                loaded = json.load(f)

            assert loaded["job_id"] == 0
            assert loaded["mean_score"] == 0.95

    def test_load_single_result(self):
        """Test loading a single result from JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = JSONStorage(Path(tmpdir))

            # Create a JSON file
            result = {
                "job_id": 0,
                "mean_score": 0.95,
                "success": True
            }

            result_file = Path(tmpdir) / "result_000000.json"
            with open(result_file, "w") as f:
                json.dump(result, f)

            # Load it
            loaded = backend.load_result(job_id=0)

            assert loaded["job_id"] == 0
            assert loaded["mean_score"] == 0.95

    def test_load_nonexistent_result(self):
        """Test loading non-existent result raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = JSONStorage(Path(tmpdir))

            with pytest.raises(FileNotFoundError):
                backend.load_result(job_id=999)

    def test_load_all_results(self):
        """Test loading all results from directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = JSONStorage(Path(tmpdir))

            # Create multiple results
            for i in range(5):
                result = {
                    "job_id": i,
                    "mean_score": 0.9 + i * 0.01,
                    "success": True
                }
                backend.save_result(job_id=i, result=result)

            # Load all - returns DataFrame
            df = backend.load_all_results()

            assert len(df) == 5
            assert df["success"].all()

    def test_result_exists(self):
        """Test checking if result exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = JSONStorage(Path(tmpdir))

            assert not backend.result_exists(job_id=0)

            result = {"job_id": 0, "mean_score": 0.95}
            backend.save_result(job_id=0, result=result)

            assert backend.result_exists(job_id=0)

    def test_load_all_empty_directory(self):
        """Test loading from empty directory returns empty DataFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = JSONStorage(Path(tmpdir))

            df = backend.load_all_results()
            assert df.empty


class TestParquetStorage:
    """Tests for Parquet storage backend."""

    def test_append_single_result(self):
        """Test appending a single result to Parquet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ParquetStorage(Path(tmpdir))

            result = {
                "job_id": 0,
                "config_hash": "abc123",
                "mean_score": 0.95,
                "std_score": 0.02,
                "success": True
            }

            backend.append_result(result)

            # Check file was created
            assert backend.file_exists()

    def test_append_multiple_results(self):
        """Test appending multiple results to Parquet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ParquetStorage(Path(tmpdir))

            # Append results one by one
            result1 = {"job_id": 0, "mean_score": 0.95, "success": True}
            backend.append_result(result1)

            result2 = {"job_id": 1, "mean_score": 0.92, "success": True}
            backend.append_result(result2)

            # Load and check
            df = backend.load_results()

            assert len(df) == 2
            assert df["job_id"].tolist() == [0, 1]

    def test_save_results_dataframe(self):
        """Test saving a full DataFrame to Parquet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ParquetStorage(Path(tmpdir))

            df = pd.DataFrame([
                {"job_id": i, "mean_score": 0.9 + i * 0.01, "success": True}
                for i in range(5)
            ])

            backend.save_results(df)

            # Load and check
            loaded_df = backend.load_results()
            assert len(loaded_df) == 5
            assert loaded_df["success"].all()

    def test_load_empty_parquet(self):
        """Test loading from directory with no parquet file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ParquetStorage(Path(tmpdir))

            df = backend.load_results()
            assert df.empty

    def test_save_with_string_hyperparameters(self):
        """Test saving results with stringified hyperparameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ParquetStorage(Path(tmpdir))

            result = {
                "job_id": 0,
                "hyperparameters": str({
                    "max_depth": 5,
                    "learning_rate": 0.1,
                    "subsample": 0.8
                }),
                "mean_score": 0.95,
                "success": True
            }

            backend.append_result(result)

            # Load and check
            df = backend.load_results()

            assert len(df) == 1
            assert df.iloc[0]["job_id"] == 0

    def test_data_integrity(self):
        """Test that data integrity is maintained."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ParquetStorage(Path(tmpdir))

            result = {
                "job_id": 0,
                "mean_score": 0.95123456789,
                "std_score": 0.02,
                "success": True,
                "config_hash": "abc123",
            }

            backend.append_result(result)

            # Load and verify
            df = backend.load_results()

            assert df.iloc[0]["mean_score"] == pytest.approx(0.95123456789, rel=1e-9)
            assert df.iloc[0]["success"] == True
            assert df.iloc[0]["config_hash"] == "abc123"

    def test_file_exists(self):
        """Test file_exists method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ParquetStorage(Path(tmpdir))

            assert not backend.file_exists()

            backend.append_result({"job_id": 0, "mean_score": 0.95})

            assert backend.file_exists()


class TestBackendComparison:
    """Tests comparing JSON and Parquet backends."""

    def test_same_data_both_backends(self):
        """Test that both backends can store and retrieve same data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_backend = JSONStorage(Path(tmpdir) / "json")
            parquet_backend = ParquetStorage(Path(tmpdir) / "parquet")

            # Create test data
            results = []
            for i in range(5):
                result = {
                    "job_id": i,
                    "mean_score": 0.9 + i * 0.01,
                    "std_score": 0.02,
                    "success": True,
                }
                results.append(result)

            # Save to both backends
            for result in results:
                json_backend.save_result(job_id=result["job_id"], result=result)
                parquet_backend.append_result(result)

            # Load from both
            json_df = json_backend.load_all_results()
            parquet_df = parquet_backend.load_results()

            # Compare counts
            assert len(json_df) == len(parquet_df)
            assert len(json_df) == 5

            # Compare specific values
            for i in range(5):
                json_row = json_df[json_df["job_id"] == i].iloc[0]
                parquet_row = parquet_df[parquet_df["job_id"] == i].iloc[0]

                assert json_row["job_id"] == parquet_row["job_id"]
                assert json_row["mean_score"] == pytest.approx(parquet_row["mean_score"])
                assert json_row["success"] == parquet_row["success"]


class TestErrorHandling:
    """Tests for error handling in backends."""

    def test_json_corrupt_file(self):
        """Test handling of corrupted JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = JSONStorage(Path(tmpdir))

            # Create corrupt JSON file
            corrupt_file = Path(tmpdir) / "result_000000.json"
            corrupt_file.write_text("{ this is not valid JSON }")

            with pytest.raises(json.JSONDecodeError):
                backend.load_result(job_id=0)

    def test_json_directory_creation(self):
        """Test that JSONStorage creates its output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = Path(tmpdir) / "subdir" / "results"
            backend = JSONStorage(subdir)

            assert subdir.exists()

    def test_parquet_directory_creation(self):
        """Test that ParquetStorage creates its output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = Path(tmpdir) / "subdir" / "results"
            backend = ParquetStorage(subdir)

            assert subdir.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
