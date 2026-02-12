"""Tests for score distribution visualization (exatune.visualization.distributions)."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from exatune.visualization.distributions import (
    plot_score_histogram,
    plot_score_violin,
    plot_train_vs_test,
)


class TestPlotScoreHistogram:
    """Tests for the score histogram plot."""

    def test_returns_figure(self, sample_results_dataframe):
        """plot_score_histogram should return a matplotlib Figure."""
        fig = plot_score_histogram(sample_results_dataframe)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_without_kde(self, sample_results_dataframe):
        """Setting show_kde=False should still produce a valid Figure."""
        fig = plot_score_histogram(sample_results_dataframe, show_kde=False)
        assert isinstance(fig, matplotlib.figure.Figure)
        # Y-axis label should say "Count" when KDE is off
        ax = fig.axes[0]
        assert ax.get_ylabel() == "Count"

    def test_with_kde(self, sample_results_dataframe):
        """With show_kde=True, the y-axis should say 'Density'."""
        fig = plot_score_histogram(sample_results_dataframe, show_kde=True)
        ax = fig.axes[0]
        assert ax.get_ylabel() == "Density"

    def test_saves_to_file(self, sample_results_dataframe, tmp_path):
        """Histogram should be saved when output_path is given."""
        output = tmp_path / "hist.png"
        plot_score_histogram(sample_results_dataframe, output_path=output)
        assert output.exists()
        assert output.stat().st_size > 0

    def teardown_method(self):
        plt.close("all")


class TestPlotScoreViolin:
    """Tests for the violin plot grouped by a hyperparameter."""

    def test_returns_figure(self, sample_results_dataframe):
        """plot_score_violin should return a matplotlib Figure."""
        fig = plot_score_violin(sample_results_dataframe, group_by="n_estimators")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_grouped_by_different_param(self, sample_results_dataframe):
        """Grouping by max_depth should also work."""
        fig = plot_score_violin(sample_results_dataframe, group_by="max_depth")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_axis_labels(self, sample_results_dataframe):
        """X-axis should match the group_by parameter."""
        fig = plot_score_violin(sample_results_dataframe, group_by="n_estimators")
        ax = fig.axes[0]
        assert ax.get_xlabel() == "n_estimators"

    def teardown_method(self):
        plt.close("all")


class TestPlotTrainVsTest:
    """Tests for the train-vs-test scatter plot."""

    def test_returns_figure(self, sample_results_dataframe):
        """plot_train_vs_test should return a matplotlib Figure."""
        fig = plot_train_vs_test(sample_results_dataframe)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_axis_labels(self, sample_results_dataframe):
        """Axes should reference train and test scores."""
        fig = plot_train_vs_test(sample_results_dataframe)
        ax = fig.axes[0]
        assert "Train" in ax.get_ylabel()
        assert "Test" in ax.get_xlabel()

    def test_saves_to_file(self, sample_results_dataframe, tmp_path):
        """Train-vs-test plot should be saved when output_path is given."""
        output = tmp_path / "tvt.png"
        plot_train_vs_test(sample_results_dataframe, output_path=output)
        assert output.exists()

    def teardown_method(self):
        plt.close("all")
