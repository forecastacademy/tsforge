"""
Unit tests for plot_distribution function.
"""

import sys
import importlib.util
from pathlib import Path

# Import directly from the module file to bypass tsforge/__init__.py
spec = importlib.util.spec_from_file_location(
    "plot_metric_distribution",
    Path(__file__).parent.parent / "src" / "tsforge" / "plots" / "plot_metric_distribution.py"
)
plot_module = importlib.util.module_from_spec(spec)

import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Mock the apply_theme import
import plotly.graph_objects as go_mock

def mock_apply_theme(fig):
    return fig

sys.modules['tsforge.plots._styling'] = type(sys)('tsforge.plots._styling')
sys.modules['tsforge.plots._styling'].apply_theme = mock_apply_theme

spec.loader.exec_module(plot_module)

plot_distribution = plot_module.plot_distribution
_plot_boxplot = plot_module._plot_boxplot
_plot_histogram = plot_module._plot_histogram


@pytest.fixture
def simple_df():
    """Create a simple DataFrame for boxplot testing."""
    np.random.seed(42)
    data = {
        "model": ["Model_A"] * 50 + ["Model_B"] * 50 + ["Model_C"] * 50,
        "wmape": np.concatenate([
            np.random.normal(0.15, 0.02, 50),  # Model A: mean 0.15
            np.random.normal(0.18, 0.03, 50),  # Model B: mean 0.18
            np.random.normal(0.12, 0.02, 50),  # Model C: mean 0.12
        ]),
    }
    return pd.DataFrame(data)


@pytest.fixture
def hierarchical_df():
    """Create a hierarchical evaluation DataFrame for histogram testing."""
    np.random.seed(42)
    data = {
        "level": ["total"] * 100,
        "metric": ["wmape"] * 100,
        "Model_A": np.random.normal(0.15, 0.02, 100),
        "Model_B": np.random.normal(0.18, 0.03, 100),
        "Model_C": np.random.normal(0.12, 0.02, 100),
    }
    return pd.DataFrame(data)


class TestPlotDistributionBoxplot:
    """Test boxplot functionality."""

    def test_boxplot_basic(self, simple_df):
        """Test basic boxplot creation."""
        fig = plot_distribution(
            simple_df,
            metric="wmape",
            anchor_model="Model_A",
            plot_type="boxplot"
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.layout.title.text is not None

    def test_boxplot_with_clipping(self, simple_df):
        """Test boxplot with outlier clipping."""
        fig = plot_distribution(
            simple_df,
            metric="wmape",
            anchor_model="Model_A",
            plot_type="boxplot",
            clip_quantiles=(0.1, 0.9)
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_boxplot_no_clipping(self, simple_df):
        """Test boxplot without clipping."""
        fig = plot_distribution(
            simple_df,
            metric="wmape",
            anchor_model="Model_A",
            plot_type="boxplot",
            clip_quantiles=(0, 1)
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_boxplot_custom_figsize(self, simple_df):
        """Test boxplot with custom figure size."""
        custom_size = (1200, 800)
        fig = plot_distribution(
            simple_df,
            metric="wmape",
            anchor_model="Model_A",
            plot_type="boxplot",
            figsize=custom_size
        )

        assert fig.layout.width == custom_size[0]
        assert fig.layout.height == custom_size[1]

    def test_boxplot_custom_title(self, simple_df):
        """Test boxplot with custom title."""
        custom_title = "Custom Model Comparison"
        fig = plot_distribution(
            simple_df,
            metric="wmape",
            anchor_model="Model_A",
            plot_type="boxplot",
            title=custom_title
        )

        assert fig.layout.title.text == custom_title

    def test_boxplot_with_points(self, simple_df):
        """Test boxplot with data points shown."""
        fig = plot_distribution(
            simple_df,
            metric="wmape",
            anchor_model="Model_A",
            plot_type="boxplot",
            show_points=True
        )

        assert isinstance(fig, go.Figure)
        # Check that box plot traces are configured with points
        for trace in fig.data:
            if isinstance(trace, go.Box):
                assert trace.boxpoints is not None

    def test_boxplot_without_points(self, simple_df):
        """Test boxplot without data points."""
        fig = plot_distribution(
            simple_df,
            metric="wmape",
            anchor_model="Model_A",
            plot_type="boxplot",
            show_points=False
        )

        assert isinstance(fig, go.Figure)
        for trace in fig.data:
            if isinstance(trace, go.Box):
                assert trace.boxpoints is False

    def test_boxplot_color_scheme_default(self, simple_df):
        """Test boxplot with default color scheme."""
        fig = plot_distribution(
            simple_df,
            metric="wmape",
            anchor_model="Model_A",
            plot_type="boxplot",
            color_scheme="default"
        )

        assert isinstance(fig, go.Figure)

    def test_boxplot_color_scheme_viridis(self, simple_df):
        """Test boxplot with viridis color scheme."""
        fig = plot_distribution(
            simple_df,
            metric="wmape",
            anchor_model="Model_A",
            plot_type="boxplot",
            color_scheme="viridis"
        )

        assert isinstance(fig, go.Figure)

    def test_boxplot_color_scheme_red_blue(self, simple_df):
        """Test boxplot with red_blue color scheme."""
        fig = plot_distribution(
            simple_df,
            metric="wmape",
            anchor_model="Model_A",
            plot_type="boxplot",
            color_scheme="red_blue"
        )

        assert isinstance(fig, go.Figure)

    def test_boxplot_requires_anchor(self, simple_df):
        """Test that boxplot requires anchor_model."""
        with pytest.raises(ValueError, match="anchor_model is required"):
            plot_distribution(
                simple_df,
                metric="wmape",
                plot_type="boxplot"
            )

    def test_boxplot_shows_stats(self, simple_df, capsys):
        """Test that boxplot prints statistics when requested."""
        plot_distribution(
            simple_df,
            metric="wmape",
            anchor_model="Model_A",
            plot_type="boxplot",
            show_stats=True
        )

        captured = capsys.readouterr()
        assert "SUMMARY" in captured.out
        assert "Model_A" in captured.out

    def test_boxplot_hides_stats(self, simple_df, capsys):
        """Test that boxplot doesn't print when show_stats=False."""
        plot_distribution(
            simple_df,
            metric="wmape",
            anchor_model="Model_A",
            plot_type="boxplot",
            show_stats=False
        )

        captured = capsys.readouterr()
        assert "SUMMARY" not in captured.out


class TestPlotDistributionHistogram:
    """Test histogram functionality."""

    def test_histogram_with_models(self, hierarchical_df):
        """Test histogram with two specific models."""
        fig = plot_distribution(
            hierarchical_df,
            metric="wmape",
            models=["Model_A", "Model_B"],
            plot_type="histogram"
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2

    def test_histogram_with_anchor(self, hierarchical_df):
        """Test histogram with anchor model."""
        fig = plot_distribution(
            hierarchical_df,
            metric="wmape",
            anchor_model="Model_A",
            plot_type="histogram"
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2

    def test_histogram_with_custom_bins(self, hierarchical_df):
        """Test histogram with custom bin count."""
        fig = plot_distribution(
            hierarchical_df,
            metric="wmape",
            anchor_model="Model_A",
            plot_type="histogram",
            bins=50
        )

        assert isinstance(fig, go.Figure)

    def test_histogram_with_clipping(self, hierarchical_df):
        """Test histogram with outlier clipping."""
        fig = plot_distribution(
            hierarchical_df,
            metric="wmape",
            anchor_model="Model_A",
            plot_type="histogram",
            clip_quantiles=(0.1, 0.9)
        )

        assert isinstance(fig, go.Figure)

    def test_histogram_custom_figsize(self, hierarchical_df):
        """Test histogram with custom figure size."""
        custom_size = (1000, 700)
        fig = plot_distribution(
            hierarchical_df,
            metric="wmape",
            anchor_model="Model_A",
            plot_type="histogram",
            figsize=custom_size
        )

        assert fig.layout.width == custom_size[0]
        assert fig.layout.height == custom_size[1]

    def test_histogram_custom_title(self, hierarchical_df):
        """Test histogram with custom title."""
        custom_title = "Model Distribution Comparison"
        fig = plot_distribution(
            hierarchical_df,
            metric="wmape",
            anchor_model="Model_A",
            plot_type="histogram",
            title=custom_title
        )

        assert fig.layout.title.text == custom_title

    def test_histogram_requires_models_or_anchor(self, hierarchical_df):
        """Test that histogram requires models or anchor_model."""
        with pytest.raises(ValueError, match="Must provide either"):
            plot_distribution(
                hierarchical_df,
                metric="wmape",
                plot_type="histogram"
            )

    def test_histogram_models_count_validation(self, hierarchical_df):
        """Test that histogram requires exactly 2 models."""
        with pytest.raises(ValueError, match="Exactly 2 models"):
            plot_distribution(
                hierarchical_df,
                metric="wmape",
                models=["Model_A", "Model_B", "Model_C"],
                plot_type="histogram"
            )

    def test_histogram_shows_stats(self, hierarchical_df, capsys):
        """Test that histogram prints statistics when requested."""
        plot_distribution(
            hierarchical_df,
            metric="wmape",
            anchor_model="Model_A",
            plot_type="histogram",
            show_stats=True
        )

        captured = capsys.readouterr()
        assert "WMAPE Distribution" in captured.out or "distribution comparison" in captured.out.lower()

    def test_histogram_hides_stats(self, hierarchical_df, capsys):
        """Test that histogram doesn't print when show_stats=False."""
        plot_distribution(
            hierarchical_df,
            metric="wmape",
            anchor_model="Model_A",
            plot_type="histogram",
            show_stats=False
        )

        captured = capsys.readouterr()
        # Should not have detailed statistics output
        assert "Clipping" not in captured.out or len(captured.out) < 50


class TestPlotDistributionValidation:
    """Test input validation and error handling."""

    def test_invalid_plot_type(self, simple_df):
        """Test that invalid plot_type raises error."""
        with pytest.raises(ValueError, match="Invalid plot_type"):
            plot_distribution(
                simple_df,
                metric="wmape",
                anchor_model="Model_A",
                plot_type="invalid"
            )

    def test_histogram_invalid_metric(self, hierarchical_df):
        """Test that histogram with invalid metric raises error."""
        with pytest.raises(ValueError, match="No data found for metric"):
            plot_distribution(
                hierarchical_df,
                metric="invalid_metric",
                anchor_model="Model_A",
                plot_type="histogram"
            )

    def test_histogram_models_not_in_data(self, hierarchical_df):
        """Test that missing models raise error."""
        with pytest.raises(ValueError):
            plot_distribution(
                hierarchical_df,
                metric="wmape",
                models=["Invalid_A", "Invalid_B"],
                plot_type="histogram"
            )


class TestInternalFunctions:
    """Test internal helper functions."""

    def test_plot_boxplot_direct(self, simple_df):
        """Test _plot_boxplot directly."""
        fig = _plot_boxplot(
            df=simple_df,
            metric="wmape",
            anchor_model="Model_A",
            clip_quantiles=(0.05, 0.95),
            show_points=True,
            show_stats=False,
            figsize=(900, 600),
            title=None,
            color_scheme="default"
        )

        assert isinstance(fig, go.Figure)

    def test_plot_histogram_direct(self, hierarchical_df):
        """Test _plot_histogram directly."""
        fig = _plot_histogram(
            df=hierarchical_df,
            metric="wmape",
            models=["Model_A", "Model_B"],
            anchor_model=None,
            clip_quantiles=(0.05, 0.95),
            show_stats=False,
            bins=30,
            figsize=(900, 600),
            title=None,
            color_scheme="default"
        )

        assert isinstance(fig, go.Figure)
