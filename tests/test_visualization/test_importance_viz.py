"""Tests for parameter importance visualization (exatune.visualization.importance)."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from exatune.visualization.importance import plot_importance, plot_interaction_heatmap


class TestPlotImportance:
    """Tests for the hyperparameter importance bar chart."""

    def test_variance_method(self, sample_results_dataframe):
        """Variance-based importance should return a Figure."""
        fig = plot_importance(sample_results_dataframe, method="variance")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_correlation_method(self, sample_results_dataframe):
        """Correlation-based importance should return a Figure."""
        fig = plot_importance(sample_results_dataframe, method="correlation")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_range_method(self, sample_results_dataframe):
        """Range-based importance should return a Figure."""
        fig = plot_importance(sample_results_dataframe, method="range")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_invalid_method_raises(self, sample_results_dataframe):
        """An unknown importance method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            plot_importance(sample_results_dataframe, method="nonexistent")

    def test_specific_params(self, sample_results_dataframe):
        """Passing explicit params should limit which bars appear."""
        fig = plot_importance(
            sample_results_dataframe, params=["n_estimators", "max_depth"]
        )
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert len(labels) == 2

    def test_saves_to_file(self, sample_results_dataframe, tmp_path):
        """Importance plot should be saved when output_path is given."""
        output = tmp_path / "importance.png"
        plot_importance(sample_results_dataframe, output_path=output)
        assert output.exists()

    def teardown_method(self):
        plt.close("all")


class TestPlotInteractionHeatmap:
    """Tests for the pairwise interaction heatmap."""

    def test_returns_figure(self, sample_results_dataframe):
        """plot_interaction_heatmap should return a Figure."""
        fig = plot_interaction_heatmap(sample_results_dataframe)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_raises_with_fewer_than_two_params(self, sample_results_dataframe):
        """Fewer than 2 parameters should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            plot_interaction_heatmap(
                sample_results_dataframe, params=["n_estimators"]
            )

    def test_specific_params(self, sample_results_dataframe):
        """Passing two specific params should produce a 2x2 interaction matrix."""
        fig = plot_interaction_heatmap(
            sample_results_dataframe, params=["n_estimators", "max_depth"]
        )
        assert isinstance(fig, matplotlib.figure.Figure)
        # The image should have a 2x2 array
        ax = fig.axes[0]
        images = ax.get_images()
        assert len(images) > 0
        assert images[0].get_array().shape == (2, 2)

    def teardown_method(self):
        plt.close("all")
