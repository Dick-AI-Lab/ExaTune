"""Tests for experiment summary (exatune.analysis.summary)."""

import numpy as np
import pandas as pd
import pytest

from exatune.analysis.summary import (
    compute_experiment_summary,
    format_summary_text,
)


class TestComputeExperimentSummary:
    """Tests for compute_experiment_summary."""

    def test_returns_all_sections(self, sample_results_dataframe):
        """Summary dict should contain all expected top-level sections."""
        result = compute_experiment_summary(sample_results_dataframe)
        expected_sections = {
            "overview",
            "best_config",
            "importance",
            "smoothness",
            "multimodality",
            "interactions",
            "timing",
        }
        assert expected_sections.issubset(result.keys())

    def test_overview_has_metric_stats(self, sample_results_dataframe):
        """Overview section should include metric statistics."""
        result = compute_experiment_summary(sample_results_dataframe)
        ov = result["overview"]
        assert "metric_mean" in ov
        assert "metric_std" in ov
        assert "metric_min" in ov
        assert "metric_max" in ov
        assert "n_successful" in ov

    def test_best_config_contains_params(self, sample_results_dataframe):
        """Best config should contain hyperparameter values and score."""
        result = compute_experiment_summary(sample_results_dataframe)
        bc = result["best_config"]
        assert "n_estimators" in bc
        assert "max_depth" in bc
        assert "min_samples_split" in bc
        assert "_score" in bc

    def test_importance_is_list(self, sample_results_dataframe):
        """Importance section should be a list of dicts (one per param)."""
        result = compute_experiment_summary(sample_results_dataframe)
        assert isinstance(result["importance"], list)
        assert len(result["importance"]) == 3

    def test_smoothness_has_subsections(self, sample_results_dataframe):
        """Smoothness section should contain autocorrelation, gradient, ruggedness."""
        result = compute_experiment_summary(sample_results_dataframe)
        sm = result["smoothness"]
        assert "autocorrelation" in sm
        assert "gradient_statistics" in sm
        assert "ruggedness" in sm


class TestFormatSummaryText:
    """Tests for format_summary_text."""

    def test_returns_non_empty_string(self, sample_results_dataframe):
        """Formatted text should be a non-empty string."""
        summary = compute_experiment_summary(sample_results_dataframe)
        text = format_summary_text(summary)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_contains_key_sections(self, sample_results_dataframe):
        """Formatted text should contain all major section headings."""
        summary = compute_experiment_summary(sample_results_dataframe)
        text = format_summary_text(summary)
        assert "OVERVIEW" in text
        assert "BEST CONFIGURATION" in text
        assert "HYPERPARAMETER IMPORTANCE" in text
        assert "LANDSCAPE SMOOTHNESS" in text
        assert "MULTIMODALITY" in text
        assert "SIGNIFICANT INTERACTIONS" in text

    def test_contains_report_header(self, sample_results_dataframe):
        """Formatted text should begin with the report title."""
        summary = compute_experiment_summary(sample_results_dataframe)
        text = format_summary_text(summary)
        assert "EXATUNE LANDSCAPE ANALYSIS REPORT" in text
