"""Tests for multimodality detection (exatune.analysis.multimodality)."""

import numpy as np
import pandas as pd
import pytest

from exatune.analysis.multimodality import (
    compute_fitness_distance_correlation,
    compute_multimodality_metrics,
    find_local_optima,
)


class TestFindLocalOptima:
    """Tests for find_local_optima."""

    def test_returns_dataframe_with_optimality_gap(self, sample_results_dataframe):
        """Result should be a DataFrame with an optimality_gap column."""
        optima = find_local_optima(sample_results_dataframe)
        assert isinstance(optima, pd.DataFrame)
        assert "optimality_gap" in optima.columns

    def test_linear_function_has_single_optimum(self, sample_results_dataframe):
        """A monotonic linear function on a grid has exactly 1 local maximum."""
        optima = find_local_optima(sample_results_dataframe, direction="maximize")
        assert len(optima) == 1, (
            f"Expected exactly 1 local optimum for a linear function, "
            f"got {len(optima)}"
        )

    def test_global_optimum_at_expected_config(self, sample_results_dataframe):
        """Global optimum should be at max n_estimators, max max_depth, min min_samples_split."""
        optima = find_local_optima(sample_results_dataframe, direction="maximize")
        best = optima.iloc[0]
        assert best["n_estimators"] == 200
        assert best["max_depth"] == 15
        assert best["min_samples_split"] == 2

    def test_optimality_gap_is_zero_for_global(self, sample_results_dataframe):
        """The single local optimum IS the global optimum, so gap should be 0."""
        optima = find_local_optima(sample_results_dataframe, direction="maximize")
        assert optima.iloc[0]["optimality_gap"] == pytest.approx(0.0)


class TestComputeMultimodalityMetrics:
    """Tests for compute_multimodality_metrics."""

    def test_returns_expected_keys(self, sample_results_dataframe):
        """Result dict should contain all expected keys."""
        result = compute_multimodality_metrics(sample_results_dataframe)
        expected_keys = {
            "n_local_optima",
            "n_configurations",
            "optima_density",
            "global_optimum_value",
            "second_best_optimum_value",
            "gap_global_to_second",
            "mean_optimum_value",
            "std_optimum_values",
            "funnel_index",
            "interpretation",
        }
        assert expected_keys.issubset(result.keys())

    def test_unimodal_interpretation(self, sample_results_dataframe):
        """Linear function should be identified as unimodal."""
        result = compute_multimodality_metrics(sample_results_dataframe)
        assert result["n_local_optima"] == 1
        assert "unimodal" in result["interpretation"].lower()

    def test_n_configurations_matches_input(self, sample_results_dataframe):
        """Number of configurations should match the input DataFrame length."""
        result = compute_multimodality_metrics(sample_results_dataframe)
        assert result["n_configurations"] == len(sample_results_dataframe)


class TestComputeFitnessDistanceCorrelation:
    """Tests for compute_fitness_distance_correlation."""

    def test_fdc_between_minus1_and_1(self, sample_results_dataframe):
        """FDC should be in [-1, 1]."""
        result = compute_fitness_distance_correlation(sample_results_dataframe)
        assert -1.0 <= result["fdc"] <= 1.0

    def test_fdc_returns_interpretation(self, sample_results_dataframe):
        """Result should include an interpretation string."""
        result = compute_fitness_distance_correlation(sample_results_dataframe)
        assert "interpretation" in result
        assert isinstance(result["interpretation"], str)
        assert len(result["interpretation"]) > 0

    def test_fdc_positive_for_smooth_landscape(self, sample_results_dataframe):
        """For a smooth, well-structured landscape FDC should be positive."""
        result = compute_fitness_distance_correlation(sample_results_dataframe)
        # For maximize direction, the code negates values and correlates with distance.
        # A smooth big-valley landscape should yield positive FDC.
        assert result["fdc"] > 0, (
            f"Expected positive FDC for smooth linear landscape, got {result['fdc']}"
        )
