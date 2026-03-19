"""Tests for ima_plotter.plotter."""

import pandas as pd
import pytest

from ima_plotter.plotter import plot_magnetic_vs_time


def _make_df():
    """Build a minimal tidy DataFrame with two IDs and two frequencies."""
    rows = []
    for id_ in ["EX1A", "EX1B"]:
        for freq in [2, 4]:
            for t in [1, 2, 3]:
                rows.append(
                    {
                        "id": id_,
                        "frequency": freq,
                        "group": "Cal0",
                        "time_index": t,
                        "avg_time": float(t),
                        "avg_v2abs_magnetic": float(t * 10),
                        "std_v2abs_magnetic": 1.0,
                    }
                )
    return pd.DataFrame(rows)


class TestFacetByNoneSentinel:
    """Tests for the None-as-positional-sentinel feature in facet_by."""

    def test_vertical_layout_rows_cols(self):
        """[col, None] should produce rows=n_ids, cols=1 (vertical stacking)."""
        df = _make_df()
        fig = plot_magnetic_vs_time(df, facet_by=["id", None], filter_frequency=2)
        # Two IDs → 2 rows, 1 column
        assert len(fig._grid_ref) == 2
        # Each entry in the grid ref should be a single column
        assert all(len(row) == 1 for row in fig._grid_ref)

    def test_vertical_layout_subplot_count(self):
        """Vertical layout should create exactly n_ids subplot traces."""
        df = _make_df()
        fig = plot_magnetic_vs_time(df, facet_by=["id", None], filter_frequency=2)
        # Should have 2 subplots (one per ID)
        assert len(fig._grid_ref) == 2

    def test_horizontal_explicit_layout(self):
        """[None, col] should produce rows=1, cols=n (horizontal, same as bare string)."""
        df = _make_df()
        fig_explicit = plot_magnetic_vs_time(df, facet_by=[None, "id"], filter_frequency=2)
        fig_string = plot_magnetic_vs_time(df, facet_by="id", filter_frequency=2)
        # Both should have 1 row and 2 columns
        assert len(fig_explicit._grid_ref) == 1
        assert len(fig_explicit._grid_ref[0]) == 2
        assert len(fig_string._grid_ref) == 1
        assert len(fig_string._grid_ref[0]) == 2

    def test_none_none_raises(self):
        """[None, None] should raise a clear ValueError."""
        df = _make_df()
        with pytest.raises(ValueError, match="invalid"):
            plot_magnetic_vs_time(df, facet_by=[None, None])

    def test_backward_compat_string(self):
        """facet_by='id' (string) should still produce horizontal layout."""
        df = _make_df()
        fig = plot_magnetic_vs_time(df, facet_by="id", filter_frequency=2)
        # 1 row, 2 columns
        assert len(fig._grid_ref) == 1
        assert len(fig._grid_ref[0]) == 2

    def test_backward_compat_2d_list(self):
        """facet_by=['id', 'frequency'] should still produce 2D grid (no None)."""
        df = _make_df()
        fig = plot_magnetic_vs_time(df, facet_by=["id", "frequency"])
        # 2 ids × 2 frequencies = 2 rows × 2 cols
        assert len(fig._grid_ref) == 2
        assert len(fig._grid_ref[0]) == 2

    def test_legend_appears_once_vertical(self):
        """Legend entries should not be duplicated in vertical layout."""
        df = _make_df()
        fig = plot_magnetic_vs_time(df, facet_by=["id", None], filter_frequency=2)
        legend_names = [t.name for t in fig.data if t.showlegend]
        # Each group name should appear at most once
        assert len(legend_names) == len(set(legend_names))

    def test_no_facet_still_works(self):
        """facet_by=None should still produce a single plot."""
        df = _make_df()
        fig = plot_magnetic_vs_time(df, filter_frequency=2)
        # Single figure, no subplots grid
        assert fig._grid_ref is None
