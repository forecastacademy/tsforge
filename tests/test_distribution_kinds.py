"""
Unit tests for plot_distribution kind parameter in core/distribution.py.

Tests that the kind parameter (histogram, density, box, violin) works correctly
in metric mode (columns parameter) and time series mode.
"""

import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from tsforge.plots.core.distribution import plot_distribution


@pytest.fixture
def metric_df():
    """Create a DataFrame with metric columns for metric mode testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "trend": np.random.beta(8, 2, n),  # Strong trend
        "seasonal_strength": np.random.beta(7, 3, n),
        "entropy": np.random.beta(3, 8, n),  # Low entropy (predictable)
        "adi": np.random.exponential(2, n),
    })


@pytest.fixture
def timeseries_df():
    """Create a DataFrame for time series mode testing."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=365, freq="D")

    data = []
    for series_id in range(1, 4):
        for date in dates:
            value = 100 + 10 * np.sin(2 * np.pi * series_id / 5) + np.random.normal(0, 5)
            data.append({
                "unique_id": f"series_{series_id}",
                "ds": date,
                "y": max(0, value),
            })

    return pd.DataFrame(data)


class TestMetricModeKinds:
    """Test plot_distribution with different kinds in metric mode (columns parameter)."""

    def test_metric_kind_histogram(self, metric_df):
        """Test metric mode with kind='histogram'."""
        fig = plot_distribution(
            metric_df,
            columns=["trend", "entropy"],
            kind="histogram",
            mode="facet",
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        # Histograms should be in the data
        has_histogram = any(isinstance(trace, go.Histogram) for trace in fig.data)
        assert has_histogram, "No Histogram traces found"

    def test_metric_kind_density(self, metric_df):
        """Test metric mode with kind='density'."""
        fig = plot_distribution(
            metric_df,
            columns=["trend"],
            kind="density",
            mode="facet",
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        # Density should use Scatter traces with fill
        has_scatter = any(isinstance(trace, go.Scatter) for trace in fig.data)
        assert has_scatter, "No Scatter traces found for density"

    def test_metric_kind_box(self, metric_df):
        """Test metric mode with kind='box'."""
        fig = plot_distribution(
            metric_df,
            columns=["trend", "entropy"],
            kind="box",
            mode="facet",
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        # Box plots should be in the data
        has_box = any(isinstance(trace, go.Box) for trace in fig.data)
        assert has_box, "No Box traces found"

    def test_metric_kind_violin(self, metric_df):
        """Test metric mode with kind='violin'."""
        fig = plot_distribution(
            metric_df,
            columns=["seasonal_strength"],
            kind="violin",
            mode="facet",
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        # Violin plots should be in the data
        has_violin = any(isinstance(trace, go.Violin) for trace in fig.data)
        assert has_violin, "No Violin traces found"

    def test_metric_kind_overlay_mode(self, metric_df):
        """Test that kind works with overlay mode."""
        fig = plot_distribution(
            metric_df,
            columns=["trend"],
            kind="box",
            mode="overlay",
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_metric_kind_with_color_col(self, metric_df):
        """Test kind parameter with color_col grouping."""
        # Create a grouped version
        grouped_df = pd.concat([metric_df, metric_df], keys=["group_a", "group_b"])
        grouped_df["group"] = grouped_df.index.get_level_values(0)
        grouped_df = grouped_df.reset_index(drop=True)

        fig = plot_distribution(
            grouped_df,
            columns=["trend"],
            color_col="group",
            kind="density",
            mode="overlay",
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_metric_kind_invalid(self, metric_df):
        """Test that invalid kind raises error."""
        with pytest.raises(ValueError, match="Invalid kind"):
            plot_distribution(
                metric_df,
                columns=["trend"],
                kind="invalid_kind",
                mode="facet",
            )

    def test_metric_multiple_columns_different_kinds(self, metric_df):
        """Test multiple columns with different kinds."""
        for kind in ["histogram", "density", "box", "violin"]:
            fig = plot_distribution(
                metric_df,
                columns=["trend", "entropy", "adi"],
                kind=kind,
                mode="facet",
            )
            assert isinstance(fig, go.Figure)
            assert len(fig.data) > 0


class TestTimeseriesModeKinds:
    """Test plot_distribution with different kinds in time series mode."""

    def test_timeseries_kind_histogram(self, timeseries_df):
        """Test time series mode with kind='histogram'."""
        fig = plot_distribution(
            timeseries_df,
            id_col="unique_id",
            date_col="ds",
            value_col="y",
            ids=2,
            kind="histogram",
            mode="overlay",
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        has_histogram = any(isinstance(trace, go.Histogram) for trace in fig.data)
        assert has_histogram

    def test_timeseries_kind_density(self, timeseries_df):
        """Test time series mode with kind='density'."""
        fig = plot_distribution(
            timeseries_df,
            id_col="unique_id",
            date_col="ds",
            value_col="y",
            ids=1,
            kind="density",
            mode="overlay",
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_timeseries_kind_box(self, timeseries_df):
        """Test time series mode with kind='box'."""
        fig = plot_distribution(
            timeseries_df,
            id_col="unique_id",
            date_col="ds",
            value_col="y",
            ids=3,
            kind="box",
            mode="facet",
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        has_box = any(isinstance(trace, go.Box) for trace in fig.data)
        assert has_box

    def test_timeseries_kind_violin(self, timeseries_df):
        """Test time series mode with kind='violin'."""
        fig = plot_distribution(
            timeseries_df,
            id_col="unique_id",
            date_col="ds",
            value_col="y",
            max_ids=2,
            kind="violin",
            mode="overlay",
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        has_violin = any(isinstance(trace, go.Violin) for trace in fig.data)
        assert has_violin

    def test_timeseries_kind_invalid(self, timeseries_df):
        """Test that invalid kind raises error in time series mode."""
        with pytest.raises(ValueError, match="Invalid kind"):
            plot_distribution(
                timeseries_df,
                id_col="unique_id",
                date_col="ds",
                value_col="y",
                kind="invalid",
            )


class TestAggregatedModeKinds:
    """Test plot_distribution with different kinds in aggregated mode."""

    def test_aggregated_kind_histogram(self, timeseries_df):
        """Test aggregated mode with kind='histogram'."""
        fig = plot_distribution(
            timeseries_df,
            id_col="unique_id",
            value_col="ds",
            agg="nunique",
            kind="histogram",
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_aggregated_kind_density(self, timeseries_df):
        """Test aggregated mode with kind='density'."""
        fig = plot_distribution(
            timeseries_df,
            id_col="unique_id",
            value_col="y",
            agg="mean",
            kind="density",
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_aggregated_kind_box(self, timeseries_df):
        """Test aggregated mode with kind='box'."""
        fig = plot_distribution(
            timeseries_df,
            id_col="unique_id",
            value_col="y",
            agg="std",
            kind="box",
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_aggregated_kind_violin(self, timeseries_df):
        """Test aggregated mode with kind='violin'."""
        fig = plot_distribution(
            timeseries_df,
            id_col="unique_id",
            value_col="y",
            agg="sum",
            kind="violin",
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


class TestKindParameterWithOptions:
    """Test kind parameter combined with other options."""

    def test_box_with_show_median(self, metric_df):
        """Test box plot with show_median option."""
        fig = plot_distribution(
            metric_df,
            columns=["trend"],
            kind="box",
            mode="facet",
            show_median=True,
        )

        assert isinstance(fig, go.Figure)

    def test_density_with_show_mean(self, metric_df):
        """Test density with show_mean option."""
        fig = plot_distribution(
            metric_df,
            columns=["entropy"],
            kind="density",
            mode="facet",
            show_mean=True,
        )

        assert isinstance(fig, go.Figure)

    def test_histogram_with_kde(self, metric_df):
        """Test histogram with KDE overlay."""
        fig = plot_distribution(
            metric_df,
            columns=["trend"],
            kind="histogram",
            mode="facet",
            show_kde=True,
        )

        assert isinstance(fig, go.Figure)

    def test_density_kde_not_applied(self, metric_df):
        """Test that KDE is not applied for non-histogram kinds."""
        # This should not error, just be ignored
        fig = plot_distribution(
            metric_df,
            columns=["trend"],
            kind="violin",
            mode="facet",
            show_kde=True,  # Should be ignored for violin
        )

        assert isinstance(fig, go.Figure)

    def test_box_with_thresholds(self, metric_df):
        """Test box plot with threshold lines."""
        fig = plot_distribution(
            metric_df,
            columns=["trend"],
            kind="box",
            mode="facet",
            thresholds={"trend": 0.5},
        )

        assert isinstance(fig, go.Figure)

    def test_violin_with_bins(self, metric_df):
        """Test that bins parameter is handled for non-histogram kinds."""
        fig = plot_distribution(
            metric_df,
            columns=["entropy"],
            kind="violin",
            mode="facet",
            bins=50,  # Should be ignored for violin
        )

        assert isinstance(fig, go.Figure)


class TestKindTraceProperties:
    """Test that different kinds produce the expected trace properties."""

    def test_histogram_has_nbinsx(self, metric_df):
        """Test that histogram traces have nbinsx."""
        fig = plot_distribution(
            metric_df,
            columns=["trend"],
            kind="histogram",
            mode="facet",
            bins=25,
        )

        for trace in fig.data:
            if isinstance(trace, go.Histogram):
                assert trace.nbinsx == 25

    def test_box_has_boxpoints(self, metric_df):
        """Test that box traces have boxpoints set."""
        fig = plot_distribution(
            metric_df,
            columns=["trend"],
            kind="box",
            mode="facet",
        )

        for trace in fig.data:
            if isinstance(trace, go.Box):
                assert hasattr(trace, "boxpoints")

    def test_violin_has_box_visible(self, metric_df):
        """Test that violin traces have box_visible."""
        fig = plot_distribution(
            metric_df,
            columns=["trend"],
            kind="violin",
            mode="facet",
        )

        for trace in fig.data:
            if isinstance(trace, go.Violin):
                assert trace.box_visible is True

    def test_density_has_fill(self, metric_df):
        """Test that density traces have fill set."""
        fig = plot_distribution(
            metric_df,
            columns=["trend"],
            kind="density",
            mode="facet",
        )

        for trace in fig.data:
            if isinstance(trace, go.Scatter) and hasattr(trace, "fill"):
                if trace.fill:
                    assert trace.fill == "tozeroy"
