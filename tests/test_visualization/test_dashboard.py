"""Tests for the summary dashboard (exatune.visualization.dashboard)."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from exatune.visualization.dashboard import plot_dashboard


class TestPlotDashboard:
    """Tests for the multi-panel landscape dashboard."""

    def test_returns_figure(self, sample_results_dataframe):
        """plot_dashboard should return a matplotlib Figure."""
        fig = plot_dashboard(sample_results_dataframe)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_saves_to_file(self, sample_results_dataframe, tmp_path):
        """Dashboard should be saved when output_path is given."""
        output = tmp_path / "dashboard.png"
        fig = plot_dashboard(sample_results_dataframe, output_path=output)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_has_multiple_axes(self, sample_results_dataframe):
        """The dashboard figure should contain multiple subplots."""
        fig = plot_dashboard(sample_results_dataframe)
        # The dashboard creates at least 4 panels (hist, importance, slices, heatmap)
        assert len(fig.axes) >= 4

    def test_has_suptitle(self, sample_results_dataframe):
        """The figure should have a super-title."""
        fig = plot_dashboard(sample_results_dataframe)
        assert fig._suptitle is not None
        assert "Dashboard" in fig._suptitle.get_text()

    def test_custom_params(self, sample_results_dataframe):
        """Passing explicit params should not raise."""
        fig = plot_dashboard(
            sample_results_dataframe,
            params=["n_estimators", "max_depth", "min_samples_split"],
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def teardown_method(self):
        plt.close("all")
