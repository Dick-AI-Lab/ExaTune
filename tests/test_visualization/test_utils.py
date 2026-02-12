"""Tests for visualization utility functions (exatune.visualization._utils)."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from exatune.visualization._utils import (
    get_figure_and_axes,
    get_metric_column,
    infer_hyperparameter_columns,
    pivot_for_heatmap,
    save_figure,
    validate_results_dataframe,
)


class TestInferHyperparameterColumns:
    """Tests for inferring hyperparameter columns from a results DataFrame."""

    def test_infers_correct_columns(self, sample_results_dataframe):
        """Standard fixture should yield exactly the three HP columns."""
        hp_cols = infer_hyperparameter_columns(sample_results_dataframe)
        assert hp_cols == ["n_estimators", "max_depth", "min_samples_split"]

    def test_excludes_metadata_columns(self, sample_results_dataframe):
        """Metadata columns must never appear in the inferred list."""
        hp_cols = infer_hyperparameter_columns(sample_results_dataframe)
        for col in ("job_id", "config_hash", "success", "mean_score", "rank"):
            assert col not in hp_cols

    def test_excludes_metric_suffix_columns(self, sample_results_dataframe):
        """Columns ending with _mean, _std, etc. should be excluded."""
        df = sample_results_dataframe.copy()
        df["custom_mean"] = 0.5
        hp_cols = infer_hyperparameter_columns(df)
        assert "custom_mean" not in hp_cols


class TestGetMetricColumn:
    """Tests for resolving metric names to DataFrame column names."""

    def test_direct_match(self, sample_results_dataframe):
        """A column name present in the DataFrame should be returned as-is."""
        result = get_metric_column(sample_results_dataframe, "mean_score")
        assert result == "mean_score"

    def test_alias_accuracy(self, sample_results_dataframe):
        """The alias 'accuracy' should resolve to 'mean_score'."""
        result = get_metric_column(sample_results_dataframe, "accuracy")
        assert result == "mean_score"

    def test_suffix_resolution(self, sample_results_dataframe):
        """A metric like 'f1_macro' should resolve to 'f1_macro_mean' if present."""
        df = sample_results_dataframe.copy()
        df["f1_macro_mean"] = 0.9
        result = get_metric_column(df, "f1_macro")
        assert result == "f1_macro_mean"

    def test_unknown_metric_raises(self, sample_results_dataframe):
        """An unresolvable metric name must raise ValueError."""
        with pytest.raises(ValueError, match="Cannot resolve metric"):
            get_metric_column(sample_results_dataframe, "nonexistent_metric")


class TestValidateResultsDataframe:
    """Tests for DataFrame validation and filtering."""

    def test_filters_success_rows(self, sample_results_with_failures):
        """Only rows with success=True should remain after filtering."""
        filtered = validate_results_dataframe(sample_results_with_failures)
        assert filtered["success"].all()
        assert len(filtered) < len(sample_results_with_failures)

    def test_raises_on_empty_dataframe(self):
        """An empty DataFrame must raise ValueError."""
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            validate_results_dataframe(df)

    def test_raises_on_all_failed(self, sample_results_dataframe):
        """If all rows are unsuccessful, filtering should raise ValueError."""
        df = sample_results_dataframe.copy()
        df["success"] = False
        df["mean_score"] = np.nan
        with pytest.raises(ValueError, match="No valid results"):
            validate_results_dataframe(df)

    def test_missing_required_columns(self, sample_results_dataframe):
        """Missing required columns must raise ValueError."""
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_results_dataframe(
                sample_results_dataframe, required_columns=["nonexistent"]
            )

    def test_missing_param_columns(self, sample_results_dataframe):
        """Missing hyperparameter columns must raise ValueError."""
        with pytest.raises(ValueError, match="Missing hyperparameter columns"):
            validate_results_dataframe(
                sample_results_dataframe, param_columns=["fake_param"]
            )


class TestPivotForHeatmap:
    """Tests for pivoting a results DataFrame into a 2D grid."""

    def test_correct_shape(self, sample_results_dataframe):
        """Pivot of n_estimators (4 values) x max_depth (4 values) should be 4x4."""
        pivot = pivot_for_heatmap(
            sample_results_dataframe, "n_estimators", "max_depth", "mean_score"
        )
        assert pivot.shape == (4, 4)

    def test_invalid_param_raises(self, sample_results_dataframe):
        """A parameter not in the DataFrame should raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            pivot_for_heatmap(
                sample_results_dataframe, "fake_param", "max_depth", "mean_score"
            )

    def test_aggregation_max(self, sample_results_dataframe):
        """Using agg_func='max' should yield values >= 'mean' aggregation."""
        pivot_mean = pivot_for_heatmap(
            sample_results_dataframe, "n_estimators", "max_depth", "mean_score", agg_func="mean"
        )
        pivot_max = pivot_for_heatmap(
            sample_results_dataframe, "n_estimators", "max_depth", "mean_score", agg_func="max"
        )
        assert (pivot_max.values >= pivot_mean.values - 1e-9).all()


class TestSaveFigure:
    """Tests for saving a matplotlib figure to disk."""

    def test_saves_to_file(self, tmp_path):
        """A figure should be written to the specified path."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        output = tmp_path / "test_fig.png"
        result = save_figure(fig, output)
        assert result == output
        assert output.exists()
        assert output.stat().st_size > 0
        plt.close(fig)

    def test_none_path_returns_none(self):
        """Passing None should skip saving and return None."""
        fig, _ = plt.subplots()
        result = save_figure(fig, None)
        assert result is None
        plt.close(fig)

    def test_creates_parent_directories(self, tmp_path):
        """Parent directories should be created automatically."""
        fig, _ = plt.subplots()
        output = tmp_path / "sub" / "dir" / "fig.png"
        save_figure(fig, output)
        assert output.exists()
        plt.close(fig)


class TestGetFigureAndAxes:
    """Tests for the figure/axes factory function."""

    def test_creates_new_figure(self):
        """Without an existing axes, a new figure should be created."""
        fig, ax = get_figure_and_axes()
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close(fig)

    def test_reuses_existing_axes(self):
        """An existing axes should be returned with its parent figure."""
        existing_fig, existing_ax = plt.subplots()
        fig, ax = get_figure_and_axes(ax=existing_ax)
        assert fig is existing_fig
        assert ax is existing_ax
        plt.close(existing_fig)

    def test_3d_projection(self):
        """Requesting '3d' projection should create a 3D axes."""
        fig, ax = get_figure_and_axes(projection="3d")
        assert isinstance(fig, matplotlib.figure.Figure)
        # The axes type name should contain '3D'
        assert "3D" in type(ax).__name__ or "3d" in type(ax).__name__.lower()
        plt.close(fig)

    def teardown_method(self):
        plt.close("all")
