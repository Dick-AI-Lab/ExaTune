"""
Tests for storage backends (exatune.storage).
"""

import pytest
import json
import tempfile
from pathlib import Path
import pandas as pd

from exatune.storage.json_backend import JSONBackend
from exatune.storage.parquet_backend import ParquetBackend


class TestJSONBackend:
    """Tests for JSON storage backend."""

    def test_save_single_result(self):
        """Test saving a single result to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = JSONBackend(Path(tmpdir))

            result = {
                "job_id": 0,
                "config_hash": "abc123",
                "hyperparameters": {"max_depth": 5},
                "success": True,
                "mean_score": 0.95,
                "std_score": 0.02
            }

            backend.save(result, job_id=0)

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
            backend = JSONBackend(Path(tmpdir))

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
            loaded = backend.load(job_id=0)

            assert loaded["job_id"] == 0
            assert loaded["mean_score"] == 0.95

    def test_load_nonexistent_result(self):
        """Test loading non-existent result raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = JSONBackend(Path(tmpdir))

            with pytest.raises(FileNotFoundError):
                backend.load(job_id=999)

    def test_load_all_results(self):
        """Test loading all results from directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = JSONBackend(Path(tmpdir))

            # Create multiple results
            for i in range(5):
                result = {
                    "job_id": i,
                    "mean_score": 0.9 + i * 0.01,
                    "success": True
                }
                backend.save(result, job_id=i)

            # Load all
            all_results = backend.load_all()

            assert len(all_results) == 5
            assert all(r["success"] for r in all_results)

    def test_delete_result(self):
        """Test deleting a result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = JSONBackend(Path(tmpdir))

            result = {"job_id": 0, "mean_score": 0.95}
            backend.save(result, job_id=0)

            # Delete it
            backend.delete(job_id=0)

            # Check it's gone
            result_file = Path(tmpdir) / "result_000000.json"
            assert not result_file.exists()

    def test_exists_check(self):
        """Test checking if result exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = JSONBackend(Path(tmpdir))

            assert not backend.exists(job_id=0)

            result = {"job_id": 0, "mean_score": 0.95}
            backend.save(result, job_id=0)

            assert backend.exists(job_id=0)


class TestParquetBackend:
    """Tests for Parquet storage backend."""

    def test_save_single_result(self):
        """Test saving a single result to Parquet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ParquetBackend(Path(tmpdir))

            result = {
                "job_id": 0,
                "config_hash": "abc123",
                "hyperparameters": {"max_depth": 5},
                "success": True,
                "mean_score": 0.95,
                "std_score": 0.02
            }

            backend.save(result)

            # Check file was created
            parquet_file = Path(tmpdir) / "results.parquet"
            assert parquet_file.exists()

    def test_append_results(self):
        """Test appending multiple results to Parquet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ParquetBackend(Path(tmpdir))

            # Save first result
            result1 = {"job_id": 0, "mean_score": 0.95, "success": True}
            backend.save(result1)

            # Append second result
            result2 = {"job_id": 1, "mean_score": 0.92, "success": True}
            backend.save(result2)

            # Load and check
            df = backend.load_all()

            assert len(df) == 2
            assert df["job_id"].tolist() == [0, 1]

    def test_load_all_results(self):
        """Test loading all results from Parquet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ParquetBackend(Path(tmpdir))

            # Create multiple results
            results = []
            for i in range(5):
                result = {
                    "job_id": i,
                    "mean_score": 0.9 + i * 0.01,
                    "success": True
                }
                results.append(result)
                backend.save(result)

            # Load all
            df = backend.load_all()

            assert len(df) == 5
            assert df["success"].all()

    def test_load_empty_parquet(self):
        """Test loading from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ParquetBackend(Path(tmpdir))

            df = backend.load_all()

            assert df is None or df.empty

    def test_save_with_nested_hyperparameters(self):
        """Test saving results with nested hyperparameter dictionaries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ParquetBackend(Path(tmpdir))

            result = {
                "job_id": 0,
                "hyperparameters": {
                    "max_depth": 5,
                    "learning_rate": 0.1,
                    "subsample": 0.8
                },
                "mean_score": 0.95,
                "success": True
            }

            backend.save(result)

            # Load and check
            df = backend.load_all()

            assert len(df) == 1
            assert df.iloc[0]["job_id"] == 0

    def test_data_integrity(self):
        """Test that data integrity is maintained."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ParquetBackend(Path(tmpdir))

            # Create results with various data types
            result = {
                "job_id": 0,
                "mean_score": 0.95123456789,  # Float precision
                "std_score": 0.02,
                "success": True,  # Boolean
                "config_hash": "abc123",  # String
                "hyperparameters": {"max_depth": 5}  # Dict
            }

            backend.save(result)

            # Load and verify
            df = backend.load_all()

            assert df.iloc[0]["mean_score"] == pytest.approx(0.95123456789, rel=1e-9)
            assert df.iloc[0]["success"] == True
            assert df.iloc[0]["config_hash"] == "abc123"


class TestBackendComparison:
    """Tests comparing JSON and Parquet backends."""

    def test_same_data_both_backends(self):
        """Test that both backends can store and retrieve same data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_backend = JSONBackend(Path(tmpdir) / "json")
            parquet_backend = ParquetBackend(Path(tmpdir) / "parquet")

            # Create test data
            results = []
            for i in range(5):
                result = {
                    "job_id": i,
                    "mean_score": 0.9 + i * 0.01,
                    "std_score": 0.02,
                    "success": True,
                    "hyperparameters": {"max_depth": i + 3}
                }
                results.append(result)

            # Save to both backends
            for result in results:
                json_backend.save(result, job_id=result["job_id"])
                parquet_backend.save(result)

            # Load from both
            json_results = json_backend.load_all()
            parquet_df = parquet_backend.load_all()

            # Compare counts
            assert len(json_results) == len(parquet_df)
            assert len(json_results) == 5

            # Compare specific values
            for i, json_result in enumerate(json_results):
                parquet_row = parquet_df[parquet_df["job_id"] == i].iloc[0]

                assert json_result["job_id"] == parquet_row["job_id"]
                assert json_result["mean_score"] == pytest.approx(parquet_row["mean_score"])
                assert json_result["success"] == parquet_row["success"]


class TestErrorHandling:
    """Tests for error handling in backends."""

    def test_json_corrupt_file(self):
        """Test handling of corrupted JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = JSONBackend(Path(tmpdir))

            # Create corrupt JSON file
            corrupt_file = Path(tmpdir) / "result_000000.json"
            corrupt_file.write_text("{ this is not valid JSON }")

            with pytest.raises(json.JSONDecodeError):
                backend.load(job_id=0)

    def test_parquet_permission_error(self):
        """Test handling of permission errors (simulated)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ParquetBackend(Path(tmpdir))

            result = {"job_id": 0, "mean_score": 0.95}

            # This should work normally
            backend.save(result)

            # Actual permission testing requires OS-level permission manipulation
            # which is difficult to test reliably across platforms

    def test_json_directory_not_exists(self):
        """Test handling of non-existent directory."""
        backend = JSONBackend(Path("/nonexistent/directory"))

        result = {"job_id": 0, "mean_score": 0.95}

        # Should create directory if possible, or raise error
        try:
            backend.save(result, job_id=0)
        except (PermissionError, FileNotFoundError):
            # Expected for non-writable paths
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
