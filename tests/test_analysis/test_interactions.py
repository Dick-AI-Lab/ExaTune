"""Tests for pairwise hyperparameter interaction detection (exatune.analysis.interactions)."""

import numpy as np
import pandas as pd
import pytest

from exatune.analysis.interactions import (
    compute_pairwise_interactions,
    detect_significant_interactions,
)


class TestComputePairwiseInteractions:
    """Tests for compute_pairwise_interactions."""

    def test_returns_dataframe_with_expected_columns(self, sample_results_dataframe):
        """Result should be a DataFrame with the required columns."""
        result = compute_pairwise_interactions(sample_results_dataframe)
        assert isinstance(result, pd.DataFrame)
        expected_cols = {
            "param_1",
            "param_2",
            "main_effect_1",
            "main_effect_2",
            "joint_effect",
            "interaction_strength",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_number_of_pairs(self, sample_results_dataframe):
        """With 3 params, there should be C(3,2)=3 pairwise rows."""
        result = compute_pairwise_interactions(sample_results_dataframe)
        assert len(result) == 3

    def test_low_interaction_for_additive_model(self, sample_results_dataframe):
        """A purely additive (linear) model should have negligible interactions."""
        result = compute_pairwise_interactions(sample_results_dataframe)
        for _, row in result.iterrows():
            assert row["interaction_strength"] < 0.01, (
                f"Interaction between {row['param_1']} and {row['param_2']} "
                f"should be near zero for additive model, "
                f"got {row['interaction_strength']}"
            )

    def test_interaction_strengths_non_negative(self, sample_results_dataframe):
        """Interaction strengths should be non-negative."""
        result = compute_pairwise_interactions(sample_results_dataframe)
        assert (result["interaction_strength"] >= 0).all()


class TestDetectSignificantInteractions:
    """Tests for detect_significant_interactions."""

    def test_high_threshold_returns_empty(self, sample_results_dataframe):
        """With a high threshold, no interactions should be significant."""
        result = detect_significant_interactions(
            sample_results_dataframe, threshold=0.1
        )
        assert result == [], (
            f"Expected no significant interactions at threshold=0.1, got {result}"
        )

    def test_returns_list_of_tuples(self, sample_results_dataframe):
        """Result should be a list of (param1, param2, strength) tuples."""
        result = detect_significant_interactions(
            sample_results_dataframe, threshold=0.0
        )
        assert isinstance(result, list)
        for item in result:
            assert len(item) == 3
            assert isinstance(item[0], str)
            assert isinstance(item[1], str)
            assert isinstance(item[2], float)
