"""Tests for ima_plotter.plotter."""

import pandas as pd
import pytest

from ima_plotter.plotter import (
    plot_magnetic_vs_time,
    _symbol_map,
    _size_map,
    _line_style_map,
    _line_width_map,
    _SYMBOLS,
    _LINE_STYLES,
    _SIZES,
    _WIDTHS,
)


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


def _make_multi_group_df():
    """Build a DataFrame with multiple groups to test category-based styling."""
    rows = []
    for group in ["Cal0", "Cal1", "Cal2", "Cal3"]:
        for t in [1, 2, 3]:
            rows.append(
                {
                    "id": "EX1A",
                    "frequency": 2,
                    "group": group,
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


class TestHelperMappingFunctions:
    """Tests for the internal helper mapping functions."""

    def test_symbol_map_auto_generation(self):
        """Auto-generated symbol map should cycle through _SYMBOLS."""
        values = ["Cal0", "Cal1", "Cal2"]
        result = _symbol_map(values)
        sorted_vals = sorted(values)
        for i, v in enumerate(sorted_vals):
            assert result[v] == _SYMBOLS[i % len(_SYMBOLS)]

    def test_symbol_map_custom_override(self):
        """Custom map values should be used; others auto-generated."""
        values = ["Cal0", "Cal1", "Cal2"]
        custom = {"Cal0": "star", "Cal2": "cross"}
        result = _symbol_map(values, custom)
        assert result["Cal0"] == "star"
        assert result["Cal2"] == "cross"
        # Cal1 is the only value not in custom_map, so it is the sole auto-assigned
        # value and receives index 0 → _SYMBOLS[0]
        assert result["Cal1"] == _SYMBOLS[0]

    def test_size_map_auto_generation(self):
        """Auto-generated size map should cycle through _SIZES."""
        values = ["a", "b", "c"]
        result = _size_map(values)
        sorted_vals = sorted(values)
        for i, v in enumerate(sorted_vals):
            assert result[v] == _SIZES[i % len(_SIZES)]

    def test_size_map_custom_override(self):
        """Custom size values should be used; others auto-generated."""
        values = ["Cal0", "Cal1"]
        custom = {"Cal0": 20}
        result = _size_map(values, custom)
        assert result["Cal0"] == 20
        assert result["Cal1"] == _SIZES[0]

    def test_line_style_map_auto_generation(self):
        """Auto-generated line style map should cycle through _LINE_STYLES."""
        values = ["x", "y", "z"]
        result = _line_style_map(values)
        sorted_vals = sorted(values)
        for i, v in enumerate(sorted_vals):
            assert result[v] == _LINE_STYLES[i % len(_LINE_STYLES)]

    def test_line_width_map_auto_generation(self):
        """Auto-generated line width map should cycle through _WIDTHS."""
        values = ["a", "b"]
        result = _line_width_map(values)
        sorted_vals = sorted(values)
        for i, v in enumerate(sorted_vals):
            assert result[v] == _WIDTHS[i % len(_WIDTHS)]

    def test_partial_custom_map_fills_remainder(self):
        """Partial custom_map: specified values use custom, rest are auto."""
        values = ["Cal0", "Cal1", "Cal2", "Cal3"]
        custom = {"Cal1": "diamond"}
        result = _symbol_map(values, custom)
        assert result["Cal1"] == "diamond"
        # The remaining 3 values auto-assigned in sorted order
        auto_vals = [v for v in sorted(values) if v != "Cal1"]
        for i, v in enumerate(auto_vals):
            assert result[v] == _SYMBOLS[i % len(_SYMBOLS)]


class TestGlobalStylingParameters:
    """Tests for global styling parameters (apply to all traces)."""

    def test_global_marker_size(self):
        """marker_size should set the same size on all traces."""
        df = _make_multi_group_df()
        fig = plot_magnetic_vs_time(df, marker_size=12)
        for trace in fig.data:
            assert trace.marker.size == 12

    def test_global_marker_symbol(self):
        """marker_symbol should set the same symbol on all traces."""
        df = _make_multi_group_df()
        fig = plot_magnetic_vs_time(df, marker_symbol="diamond")
        for trace in fig.data:
            assert trace.marker.symbol == "diamond"

    def test_global_line_style(self):
        """line_style should set the same dash on all traces."""
        df = _make_multi_group_df()
        fig = plot_magnetic_vs_time(df, line_style="dash")
        for trace in fig.data:
            assert trace.line.dash == "dash"

    def test_global_line_width(self):
        """line_width should set the same width on all traces."""
        df = _make_multi_group_df()
        fig = plot_magnetic_vs_time(df, line_width=4)
        for trace in fig.data:
            assert trace.line.width == 4

    def test_default_marker_size_is_5(self):
        """Without any marker_size param, default size should be 5."""
        df = _make_multi_group_df()
        fig = plot_magnetic_vs_time(df)
        for trace in fig.data:
            assert trace.marker.size == 5

    def test_backward_compat_no_styling(self):
        """Omitting all new parameters should produce the same figure as before."""
        df = _make_df()
        fig = plot_magnetic_vs_time(df, filter_frequency=2)
        assert len(fig.data) >= 1
        for trace in fig.data:
            assert trace.marker.size == 5


class TestCategoryBasedStylingParameters:
    """Tests for category-based styling parameters."""

    def test_marker_symbol_by_auto(self):
        """marker_symbol_by without a map should auto-assign different symbols."""
        df = _make_multi_group_df()
        fig = plot_magnetic_vs_time(df, marker_symbol_by="group")
        symbols = [trace.marker.symbol for trace in fig.data]
        # Should have different symbols for at least some traces
        assert len(set(symbols)) > 1

    def test_marker_symbol_by_custom_map(self):
        """marker_symbol_by with custom map should use specified symbols."""
        df = _make_multi_group_df()
        custom = {"Cal0": "circle", "Cal1": "square", "Cal2": "diamond", "Cal3": "star"}
        fig = plot_magnetic_vs_time(df, marker_symbol_by="group", marker_symbol_map=custom)
        symbol_by_name = {trace.name: trace.marker.symbol for trace in fig.data}
        for group, expected_symbol in custom.items():
            assert symbol_by_name[group] == expected_symbol

    def test_marker_size_by_auto(self):
        """marker_size_by without a map should auto-assign graduated sizes."""
        df = _make_multi_group_df()
        fig = plot_magnetic_vs_time(df, marker_size_by="group")
        sizes = [trace.marker.size for trace in fig.data]
        assert len(set(sizes)) > 1

    def test_marker_size_by_custom_map(self):
        """marker_size_by with custom map should use specified sizes."""
        df = _make_multi_group_df()
        custom = {"Cal0": 5, "Cal1": 7, "Cal2": 9, "Cal3": 11}
        fig = plot_magnetic_vs_time(df, marker_size_by="group", marker_size_map=custom)
        size_by_name = {trace.name: trace.marker.size for trace in fig.data}
        for group, expected_size in custom.items():
            assert size_by_name[group] == expected_size

    def test_line_style_by_auto(self):
        """line_style_by without a map should auto-assign different dash styles."""
        df = _make_multi_group_df()
        fig = plot_magnetic_vs_time(df, line_style_by="group")
        styles = [trace.line.dash for trace in fig.data]
        assert len(set(styles)) > 1

    def test_line_style_by_custom_map(self):
        """line_style_by with custom map should use specified dash styles."""
        df = _make_multi_group_df()
        custom = {"Cal0": "solid", "Cal1": "dash", "Cal2": "dot", "Cal3": "dashdot"}
        fig = plot_magnetic_vs_time(df, line_style_by="group", line_style_map=custom)
        style_by_name = {trace.name: trace.line.dash for trace in fig.data}
        for group, expected_style in custom.items():
            assert style_by_name[group] == expected_style

    def test_line_width_by_auto(self):
        """line_width_by without a map should auto-assign different widths."""
        df = _make_multi_group_df()
        fig = plot_magnetic_vs_time(df, line_width_by="group")
        widths = [trace.line.width for trace in fig.data]
        assert len(set(widths)) > 1

    def test_line_width_by_custom_map(self):
        """line_width_by with custom map should use specified widths."""
        df = _make_multi_group_df()
        custom = {"Cal0": 1, "Cal1": 2, "Cal2": 3, "Cal3": 4}
        fig = plot_magnetic_vs_time(df, line_width_by="group", line_width_map=custom)
        width_by_name = {trace.name: trace.line.width for trace in fig.data}
        for group, expected_width in custom.items():
            assert width_by_name[group] == expected_width

    def test_category_overrides_global_marker_size(self):
        """Category-based marker_size_by should override global marker_size."""
        df = _make_multi_group_df()
        custom = {"Cal0": 20, "Cal1": 20, "Cal2": 20, "Cal3": 20}
        fig = plot_magnetic_vs_time(
            df, marker_size=5, marker_size_by="group", marker_size_map=custom
        )
        # category-based wins; all should be 20 from custom map
        for trace in fig.data:
            assert trace.marker.size == 20

    def test_category_overrides_global_line_style(self):
        """Category-based line_style_by should override global line_style."""
        df = _make_multi_group_df()
        custom = {"Cal0": "dot", "Cal1": "dot", "Cal2": "dot", "Cal3": "dot"}
        fig = plot_magnetic_vs_time(
            df, line_style="dash", line_style_by="group", line_style_map=custom
        )
        for trace in fig.data:
            assert trace.line.dash == "dot"


class TestYMultiplier:
    """Tests for the y_multiplier parameter."""

    def test_y_multiplier_scales_values(self):
        """y_multiplier should scale all y-values."""
        df = _make_df()
        fig_normal = plot_magnetic_vs_time(df, filter_frequency=2)
        fig_scaled = plot_magnetic_vs_time(df, filter_frequency=2, y_multiplier=1000.0)

        for normal_trace, scaled_trace in zip(fig_normal.data, fig_scaled.data):
            for yn, ys in zip(normal_trace.y, scaled_trace.y):
                assert abs(ys - yn * 1000.0) < 1e-9

    def test_y_multiplier_scales_error_bars(self):
        """y_multiplier should scale error bars by |y_multiplier|."""
        df = _make_df()
        fig_normal = plot_magnetic_vs_time(df, filter_frequency=2, show_error_bars=True)
        fig_scaled = plot_magnetic_vs_time(
            df, filter_frequency=2, show_error_bars=True, y_multiplier=10.0
        )
        for normal_trace, scaled_trace in zip(fig_normal.data, fig_scaled.data):
            if normal_trace.error_y and normal_trace.error_y.array:
                for en, es in zip(
                    normal_trace.error_y.array, scaled_trace.error_y.array
                ):
                    assert abs(es - en * 10.0) < 1e-9

    def test_y_multiplier_negative_scales_error_bars_by_abs(self):
        """Negative y_multiplier should scale error bars by absolute value."""
        df = _make_df()
        fig_pos = plot_magnetic_vs_time(
            df, filter_frequency=2, show_error_bars=True, y_multiplier=10.0
        )
        fig_neg = plot_magnetic_vs_time(
            df, filter_frequency=2, show_error_bars=True, y_multiplier=-10.0
        )
        for pos_trace, neg_trace in zip(fig_pos.data, fig_neg.data):
            if pos_trace.error_y and pos_trace.error_y.array:
                for ep, en_val in zip(
                    pos_trace.error_y.array, neg_trace.error_y.array
                ):
                    assert abs(ep - en_val) < 1e-9

    def test_y_multiplier_default_no_change(self):
        """Default y_multiplier=1.0 should leave values unchanged."""
        df = _make_df()
        fig_default = plot_magnetic_vs_time(df, filter_frequency=2)
        fig_explicit = plot_magnetic_vs_time(df, filter_frequency=2, y_multiplier=1.0)
        for t1, t2 in zip(fig_default.data, fig_explicit.data):
            for y1, y2 in zip(t1.y, t2.y):
                assert abs(y1 - y2) < 1e-9
