"""Tests for landscape visualization (exatune.visualization.landscape)."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from exatune.visualization.landscape import plot_contour, plot_heatmap, plot_surface


class TestPlotHeatmap:
    """Tests for the 2D heatmap plot."""

    def test_returns_figure(self, sample_results_dataframe):
        """plot_heatmap should return a matplotlib Figure."""
        fig = plot_heatmap(
            sample_results_dataframe, x_param="n_estimators", y_param="max_depth"
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_saves_to_file(self, sample_results_dataframe, tmp_path):
        """Heatmap should be saved when output_path is given."""
        output = tmp_path / "heatmap.png"
        fig = plot_heatmap(
            sample_results_dataframe,
            x_param="n_estimators",
            y_param="max_depth",
            output_path=output,
        )
        assert output.exists()
        assert output.stat().st_size > 0

    def test_different_agg_func(self, sample_results_dataframe):
        """Using agg_func='max' should still produce a valid Figure."""
        fig = plot_heatmap(
            sample_results_dataframe,
            x_param="n_estimators",
            y_param="max_depth",
            agg_func="max",
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_custom_title(self, sample_results_dataframe):
        """A custom title should appear on the axes."""
        fig = plot_heatmap(
            sample_results_dataframe,
            x_param="n_estimators",
            y_param="max_depth",
            title="Custom Title",
        )
        ax = fig.axes[0]
        assert ax.get_title() == "Custom Title"

    def test_invalid_param_raises(self, sample_results_dataframe):
        """An invalid parameter name should raise ValueError."""
        with pytest.raises(ValueError):
            plot_heatmap(
                sample_results_dataframe,
                x_param="nonexistent",
                y_param="max_depth",
            )

    def teardown_method(self):
        plt.close("all")


class TestPlotSurface:
    """Tests for the 3D surface plot."""

    def test_returns_figure(self, sample_results_dataframe):
        """plot_surface should return a matplotlib Figure."""
        fig = plot_surface(
            sample_results_dataframe, x_param="n_estimators", y_param="max_depth"
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_saves_to_file(self, sample_results_dataframe, tmp_path):
        """Surface plot should be saved when output_path is given."""
        output = tmp_path / "surface.png"
        fig = plot_surface(
            sample_results_dataframe,
            x_param="n_estimators",
            y_param="max_depth",
            output_path=output,
        )
        assert output.exists()

    def test_invalid_param_raises(self, sample_results_dataframe):
        """An invalid parameter name should raise ValueError."""
        with pytest.raises(ValueError):
            plot_surface(
                sample_results_dataframe,
                x_param="n_estimators",
                y_param="nonexistent",
            )

    def teardown_method(self):
        plt.close("all")


class TestPlotContour:
    """Tests for the contour plot."""

    def test_returns_figure(self, sample_results_dataframe):
        """plot_contour should return a matplotlib Figure."""
        fig = plot_contour(
            sample_results_dataframe, x_param="n_estimators", y_param="max_depth"
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_unfilled_contours(self, sample_results_dataframe):
        """Unfilled contours (filled=False) should still produce a Figure."""
        fig = plot_contour(
            sample_results_dataframe,
            x_param="n_estimators",
            y_param="max_depth",
            filled=False,
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_invalid_param_raises(self, sample_results_dataframe):
        """An invalid parameter name should raise ValueError."""
        with pytest.raises(ValueError):
            plot_contour(
                sample_results_dataframe,
                x_param="fake",
                y_param="max_depth",
            )

    def teardown_method(self):
        plt.close("all")
