"""Tests for slice plot visualization (exatune.visualization.slices)."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from exatune.visualization.slices import (
    plot_all_slices,
    plot_conditional_slice,
    plot_slice,
)


class TestPlotSlice:
    """Tests for single-parameter slice plots."""

    def test_returns_figure(self, sample_results_dataframe):
        """plot_slice should return a matplotlib Figure."""
        fig = plot_slice(sample_results_dataframe, param="n_estimators")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_show_individual_points(self, sample_results_dataframe):
        """Enabling show_individual should not error and should return a Figure."""
        fig = plot_slice(
            sample_results_dataframe, param="n_estimators", show_individual=True
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_saves_to_file(self, sample_results_dataframe, tmp_path):
        """Slice plot should be saved when output_path is provided."""
        output = tmp_path / "slice.png"
        fig = plot_slice(
            sample_results_dataframe, param="max_depth", output_path=output
        )
        assert output.exists()
        assert output.stat().st_size > 0

    def test_no_std_band(self, sample_results_dataframe):
        """Setting show_std=False should still produce a valid Figure."""
        fig = plot_slice(
            sample_results_dataframe, param="n_estimators", show_std=False
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def teardown_method(self):
        plt.close("all")


class TestPlotAllSlices:
    """Tests for the multi-panel all-slices grid."""

    def test_returns_figure(self, sample_results_dataframe):
        """plot_all_slices should return a matplotlib Figure."""
        fig = plot_all_slices(sample_results_dataframe)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_correct_subplot_count(self, sample_results_dataframe):
        """There should be one visible subplot per hyperparameter."""
        fig = plot_all_slices(sample_results_dataframe, ncols=3)
        visible_axes = [ax for ax in fig.axes if ax.get_visible()]
        # 3 hyperparameters -> 3 visible subplots
        assert len(visible_axes) == 3

    def test_specific_params(self, sample_results_dataframe):
        """Passing explicit params should limit the subplot count."""
        fig = plot_all_slices(
            sample_results_dataframe, params=["n_estimators", "max_depth"]
        )
        visible_axes = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible_axes) == 2

    def teardown_method(self):
        plt.close("all")


class TestPlotConditionalSlice:
    """Tests for conditional slice plots (interaction views)."""

    def test_returns_figure(self, sample_results_dataframe):
        """plot_conditional_slice should return a matplotlib Figure."""
        fig = plot_conditional_slice(
            sample_results_dataframe,
            param="n_estimators",
            condition_param="max_depth",
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_has_legend(self, sample_results_dataframe):
        """The plot should contain a legend showing condition values."""
        fig = plot_conditional_slice(
            sample_results_dataframe,
            param="n_estimators",
            condition_param="max_depth",
        )
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) > 0

    def teardown_method(self):
        plt.close("all")
