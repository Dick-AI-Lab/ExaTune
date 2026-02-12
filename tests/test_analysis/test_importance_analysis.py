"""Tests for hyperparameter importance analysis (exatune.analysis.importance)."""

import numpy as np
import pandas as pd
import pytest

from exatune.analysis.importance import (
    compute_correlation_importance,
    compute_importance_summary,
    compute_range_importance,
    compute_variance_importance,
)


class TestComputeVarianceImportance:
    """Tests for compute_variance_importance."""

    def test_returns_dict_with_all_params(self, sample_results_dataframe):
        """Result should map every hyperparameter to an importance score."""
        result = compute_variance_importance(sample_results_dataframe)
        assert isinstance(result, dict)
        assert "n_estimators" in result
        assert "max_depth" in result
        assert "min_samples_split" in result

    def test_max_depth_most_important(self, sample_results_dataframe):
        """max_depth (coefficient 0.02) should have the highest variance importance."""
        result = compute_variance_importance(sample_results_dataframe)
        assert result["max_depth"] > result["n_estimators"], (
            "max_depth should be more important than n_estimators"
        )
        assert result["max_depth"] > result["min_samples_split"], (
            "max_depth should be more important than min_samples_split"
        )

    def test_importance_values_non_negative(self, sample_results_dataframe):
        """All importance values should be non-negative."""
        result = compute_variance_importance(sample_results_dataframe)
        for param, val in result.items():
            assert val >= 0.0, f"Importance for {param} should be >= 0, got {val}"


class TestComputeCorrelationImportance:
    """Tests for compute_correlation_importance."""

    def test_returns_dict(self, sample_results_dataframe):
        """Result should be a dict mapping params to correlations."""
        result = compute_correlation_importance(sample_results_dataframe)
        assert isinstance(result, dict)
        assert len(result) == 3

    def test_values_between_0_and_1(self, sample_results_dataframe):
        """Absolute correlation values should lie in [0, 1]."""
        result = compute_correlation_importance(sample_results_dataframe)
        for param, val in result.items():
            assert 0.0 <= val <= 1.0, (
                f"Correlation importance for {param} should be in [0,1], got {val}"
            )


class TestComputeRangeImportance:
    """Tests for compute_range_importance."""

    def test_returns_dict(self, sample_results_dataframe):
        """Result should be a dict mapping params to range importance."""
        result = compute_range_importance(sample_results_dataframe)
        assert isinstance(result, dict)
        assert len(result) == 3

    def test_max_depth_most_important_by_range(self, sample_results_dataframe):
        """max_depth should dominate range importance due to its coefficient."""
        result = compute_range_importance(sample_results_dataframe)
        assert result["max_depth"] > result["n_estimators"]
        assert result["max_depth"] > result["min_samples_split"]


class TestComputeImportanceSummary:
    """Tests for compute_importance_summary."""

    def test_returns_dataframe_with_expected_columns(self, sample_results_dataframe):
        """Summary DataFrame should contain expected columns."""
        result = compute_importance_summary(sample_results_dataframe)
        assert isinstance(result, pd.DataFrame)
        expected_cols = {
            "parameter",
            "variance_importance",
            "correlation_importance",
            "range_importance",
            "overall_rank",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_rows_match_number_of_params(self, sample_results_dataframe):
        """Summary should have one row per hyperparameter."""
        result = compute_importance_summary(sample_results_dataframe)
        assert len(result) == 3

    def test_max_depth_ranked_first(self, sample_results_dataframe):
        """max_depth should have the best (lowest) overall rank."""
        result = compute_importance_summary(sample_results_dataframe)
        top_param = result.loc[result["overall_rank"] == 1, "parameter"].values[0]
        assert top_param == "max_depth", (
            f"Expected max_depth as rank 1, got {top_param}"
        )
