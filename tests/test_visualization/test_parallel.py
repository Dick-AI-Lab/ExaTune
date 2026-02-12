"""Tests for parallel coordinates visualization (exatune.visualization.parallel)."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from exatune.visualization.parallel import plot_parallel_coordinates


class TestPlotParallelCoordinates:
    """Tests for the parallel coordinates plot."""

    def test_returns_figure(self, sample_results_dataframe):
        """plot_parallel_coordinates should return a matplotlib Figure."""
        fig = plot_parallel_coordinates(sample_results_dataframe)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_top_n(self, sample_results_dataframe):
        """Limiting to top_n configurations should still produce a Figure."""
        fig = plot_parallel_coordinates(sample_results_dataframe, top_n=10)
        assert isinstance(fig, matplotlib.figure.Figure)
        # The title should reflect the limited count
        ax = fig.axes[0]
        assert "10" in ax.get_title()

    def test_with_specific_params(self, sample_results_dataframe):
        """Passing a subset of params should restrict axes shown."""
        params = ["n_estimators", "max_depth"]
        fig = plot_parallel_coordinates(sample_results_dataframe, params=params)
        ax = fig.axes[0]
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert tick_labels == params

    def test_saves_to_file(self, sample_results_dataframe, tmp_path):
        """Parallel coordinates should be saved when output_path is given."""
        output = tmp_path / "parallel.png"
        plot_parallel_coordinates(sample_results_dataframe, output_path=output)
        assert output.exists()
        assert output.stat().st_size > 0

    def teardown_method(self):
        plt.close("all")
