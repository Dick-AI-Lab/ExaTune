"""Tests for landscape smoothness analysis (exatune.analysis.smoothness)."""

import numpy as np
import pandas as pd
import pytest

from exatune.analysis.smoothness import (
    compute_autocorrelation,
    compute_gradient_statistics,
    compute_ruggedness,
)


class TestComputeAutocorrelation:
    """Tests for compute_autocorrelation."""

    def test_returns_expected_keys(self, sample_results_dataframe):
        """Autocorrelation result contains all required keys."""
        result = compute_autocorrelation(sample_results_dataframe)
        assert "morans_i" in result
        assert "expected_i" in result
        assert "lag_correlations" in result
        assert "interpretation" in result

    def test_morans_i_positive_for_smooth_landscape(self, sample_results_dataframe):
        """Linear deterministic function should produce positive Moran's I."""
        result = compute_autocorrelation(sample_results_dataframe)
        assert result["morans_i"] > 0, (
            f"Expected positive Moran's I for smooth linear landscape, "
            f"got {result['morans_i']}"
        )

    def test_lag_correlations_length(self, sample_results_dataframe):
        """Lag correlations list has length equal to max_lag."""
        result = compute_autocorrelation(sample_results_dataframe, max_lag=5)
        assert len(result["lag_correlations"]) == 5

    def test_interpretation_is_string(self, sample_results_dataframe):
        """Interpretation should be a non-empty string."""
        result = compute_autocorrelation(sample_results_dataframe)
        assert isinstance(result["interpretation"], str)
        assert len(result["interpretation"]) > 0


class TestComputeGradientStatistics:
    """Tests for compute_gradient_statistics."""

    def test_returns_expected_keys(self, sample_results_dataframe):
        """Gradient statistics result contains all expected keys."""
        result = compute_gradient_statistics(sample_results_dataframe)
        expected_keys = {
            "mean_gradient",
            "max_gradient",
            "std_gradient",
            "gradient_cv",
            "roughness",
            "per_param_gradients",
        }
        assert expected_keys.issubset(result.keys())

    def test_per_param_gradients_has_all_params(self, sample_results_dataframe):
        """Per-parameter gradients include all hyperparameter columns."""
        result = compute_gradient_statistics(sample_results_dataframe)
        per_param = result["per_param_gradients"]
        assert "n_estimators" in per_param
        assert "max_depth" in per_param
        assert "min_samples_split" in per_param

    def test_gradients_non_negative(self, sample_results_dataframe):
        """Gradient magnitudes should be non-negative."""
        result = compute_gradient_statistics(sample_results_dataframe)
        assert result["mean_gradient"] >= 0
        assert result["max_gradient"] >= 0
        assert result["std_gradient"] >= 0


class TestComputeRuggedness:
    """Tests for compute_ruggedness."""

    def test_ruggedness_between_0_and_1(self, sample_results_dataframe):
        """Ruggedness index should lie in [0, 1]."""
        result = compute_ruggedness(sample_results_dataframe)
        assert 0.0 <= result["ruggedness_index"] <= 1.0

    def test_smooth_landscape_low_ruggedness(self, sample_results_dataframe):
        """A linear deterministic function should have low ruggedness."""
        result = compute_ruggedness(sample_results_dataframe)
        assert result["ruggedness_index"] < 0.5, (
            f"Expected low ruggedness for smooth linear landscape, "
            f"got {result['ruggedness_index']}"
        )

    def test_identical_values_ruggedness_zero(self, sample_results_dataframe):
        """All identical scores should produce ruggedness 0."""
        df = sample_results_dataframe.copy()
        df["mean_score"] = 0.75
        result = compute_ruggedness(df)
        assert result["ruggedness_index"] == 0.0

    def test_returns_expected_keys(self, sample_results_dataframe):
        """Ruggedness result contains all expected keys."""
        result = compute_ruggedness(sample_results_dataframe)
        expected_keys = {
            "ruggedness_index",
            "local_variance",
            "global_variance",
            "interpretation",
        }
        assert expected_keys.issubset(result.keys())
