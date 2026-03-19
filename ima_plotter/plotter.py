"""Plotting functions for IMA magnetic measurement data."""

from __future__ import annotations

from typing import Optional, Union

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_magnetic_vs_time(
    df: pd.DataFrame,
    facet_by: Union[str, list[Optional[str]], None] = None,
    filter_frequency: Union[float, list[float], None] = None,
    filter_id: Union[str, list[str], None] = None,
    filter_group: Union[str, list[str], None] = None,
    show_error_bars: bool = True,
    use_baseline_subtracted: bool = False,
    color_by: str = "group",
    # Global styling (apply to all traces)
    marker_size: Union[int, float, None] = None,
    marker_symbol: Union[str, None] = None,
    line_style: Union[str, None] = None,
    line_width: Union[int, float, None] = None,
    # Category-based styling (vary by column values)
    marker_size_by: Union[str, None] = None,
    marker_size_map: Union[dict, None] = None,
    marker_symbol_by: Union[str, None] = None,
    marker_symbol_map: Union[dict, None] = None,
    line_style_by: Union[str, None] = None,
    line_style_map: Union[dict, None] = None,
    line_width_by: Union[str, None] = None,
    line_width_map: Union[dict, None] = None,
    # Y-value scaling
    y_multiplier: float = 1.0,
) -> go.Figure:
    """Create an interactive Plotly figure of magnetic signal vs time.

    Parameters
    ----------
    df:
        Tidy DataFrame as returned by
        :func:`~ima_plotter.loader.load_excel_files` (optionally processed
        by :func:`~ima_plotter.transformer.subtract_baseline`).
    facet_by:
        Column name(s) to use for faceting (subplots).  Accepts:

        * ``None`` – single plot, no faceting
        * ``"id"`` or ``"frequency"`` – one subplot per unique value,
          arranged horizontally (equivalent to ``[None, "id"]``)
        * ``["id", "frequency"]`` – grid of subplots (rows × columns)
        * ``["id", None]`` – vertical layout: one subplot per ID, stacked
          in rows (subplots share the x-axis)
        * ``[None, "frequency"]`` – horizontal layout: one subplot per
          frequency, arranged in columns (subplots share the y-axis)
    filter_frequency:
        Keep only rows where ``frequency`` is in this value / list of values.
        ``None`` means no filtering.
    filter_id:
        Keep only rows where ``id`` is in this value / list of values.
        ``None`` means no filtering.
    filter_group:
        Keep only rows where ``group`` is in this value / list of values.
        ``None`` means no filtering.
    show_error_bars:
        Whether to draw error bars.
    use_baseline_subtracted:
        If ``True`` use ``delta_avg_v2abs_magnetic`` and
        ``delta_std_v2abs_magnetic`` for the y-axis.  The DataFrame must
        have been processed with :func:`~ima_plotter.transformer.subtract_baseline`
        first.
    color_by:
        Column whose unique values determine trace colours (default:
        ``"group"``).
    marker_size:
        Global marker size applied to all traces (default: ``5``).
        Overridden per-trace when ``marker_size_by`` is also provided.
    marker_symbol:
        Global marker symbol applied to all traces (default: ``"circle"``).
        See Plotly documentation for valid symbol names.
        Overridden per-trace when ``marker_symbol_by`` is also provided.
    line_style:
        Global line dash style applied to all traces.  Valid values:
        ``"solid"``, ``"dash"``, ``"dot"``, ``"dashdot"``.
        Overridden per-trace when ``line_style_by`` is also provided.
    line_width:
        Global line width applied to all traces (default: Plotly default).
        Overridden per-trace when ``line_width_by`` is also provided.
    marker_size_by:
        Column whose unique values determine per-category marker sizes.
        If ``marker_size_map`` is not provided, sizes are auto-generated
        from :data:`_SIZES`.
    marker_size_map:
        Dict mapping category values to marker sizes.  Values not present
        in the map receive auto-assigned sizes from :data:`_SIZES`.
    marker_symbol_by:
        Column whose unique values determine per-category marker symbols.
        If ``marker_symbol_map`` is not provided, symbols are auto-generated
        from :data:`_SYMBOLS`.
    marker_symbol_map:
        Dict mapping category values to Plotly marker symbol names.  Values
        not present in the map receive auto-assigned symbols from
        :data:`_SYMBOLS`.
    line_style_by:
        Column whose unique values determine per-category line dash styles.
        If ``line_style_map`` is not provided, styles are auto-generated
        from :data:`_LINE_STYLES`.
    line_style_map:
        Dict mapping category values to line dash styles.  Values not
        present in the map receive auto-assigned styles from
        :data:`_LINE_STYLES`.
    line_width_by:
        Column whose unique values determine per-category line widths.
        If ``line_width_map`` is not provided, widths are auto-generated
        from :data:`_WIDTHS`.
    line_width_map:
        Dict mapping category values to line widths.  Values not present
        in the map receive auto-assigned widths from :data:`_WIDTHS`.
    y_multiplier:
        Scalar multiplied against all y-values (and error bars) before
        plotting.  Useful for unit conversions (default: ``1.0``).

    Returns
    -------
    plotly.graph_objects.Figure

    Notes
    -----
    **Priority rules** – when both global and category-based parameters are
    provided for the same property, the category-based value takes
    precedence for each trace.
    """
    data = df.copy()

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------
    if filter_frequency is not None:
        freqs = (
            [filter_frequency]
            if not isinstance(filter_frequency, list)
            else filter_frequency
        )
        data = data[data["frequency"].isin(freqs)]

    if filter_id is not None:
        ids = [filter_id] if not isinstance(filter_id, list) else filter_id
        data = data[data["id"].isin(ids)]

    if filter_group is not None:
        groups = [filter_group] if not isinstance(filter_group, list) else filter_group
        data = data[data["group"].isin(groups)]

    if data.empty:
        raise ValueError("No data remains after applying the requested filters.")

    # ------------------------------------------------------------------
    # Choose y columns
    # ------------------------------------------------------------------
    if use_baseline_subtracted:
        required = ["delta_avg_v2abs_magnetic"]
        if show_error_bars:
            required.append("delta_std_v2abs_magnetic")
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise ValueError(
                f"Baseline-subtracted column(s) not found: {missing}. "
                "Run subtract_baseline() first."
            )
        y_col = "delta_avg_v2abs_magnetic"
        err_col = "delta_std_v2abs_magnetic"
        y_label = "Δ Avg V2abs Magnetic"
    else:
        y_col = "avg_v2abs_magnetic"
        err_col = "std_v2abs_magnetic"
        y_label = "Avg V2abs Magnetic"

    # ------------------------------------------------------------------
    # Apply y-value multiplier
    # ------------------------------------------------------------------
    if y_multiplier != 1.0:
        data[y_col] = data[y_col] * y_multiplier
        if show_error_bars and err_col in data.columns:
            data[err_col] = data[err_col] * abs(y_multiplier)

    # ------------------------------------------------------------------
    # Build a shared dict of styling kwargs to pass to _add_traces
    # ------------------------------------------------------------------
    _style_kwargs: dict = dict(
        marker_size=marker_size,
        marker_symbol=marker_symbol,
        line_style=line_style,
        line_width=line_width,
        marker_size_by=marker_size_by,
        marker_size_map=marker_size_map,
        marker_symbol_by=marker_symbol_by,
        marker_symbol_map=marker_symbol_map,
        line_style_by=line_style_by,
        line_style_map=line_style_map,
        line_width_by=line_width_by,
        line_width_map=line_width_map,
    )

    # ------------------------------------------------------------------
    # Normalise facet_by to a list
    # ------------------------------------------------------------------
    if facet_by is None:
        facet_cols: list[Optional[str]] = []
    elif isinstance(facet_by, str):
        facet_cols = [facet_by]
    else:
        facet_cols = list(facet_by)

    if len(facet_cols) > 2:
        raise ValueError(
            f"facet_by supports at most 2 columns, got {len(facet_cols)}: {facet_cols}"
        )

    # Determine layout direction when None is used as a positional sentinel.
    # [col, None]  → vertical (rows=n, cols=1)
    # [None, col]  → horizontal (rows=1, cols=n)  – same as a bare string
    # [None, None] → error
    vertical_single = False  # True when [col, None]
    if len(facet_cols) == 2 and (facet_cols[0] is None or facet_cols[1] is None):
        if facet_cols[0] is None and facet_cols[1] is None:
            raise ValueError(
                "facet_by=[None, None] is invalid – at least one element must be a column name."
            )
        vertical_single = facet_cols[1] is None
        # Collapse to a single real column name
        facet_cols = [c for c in facet_cols if c is not None]

    # ------------------------------------------------------------------
    # Build subplot grid
    # ------------------------------------------------------------------
    if len(facet_cols) == 0:
        fig = go.Figure()
        _add_traces(
            fig,
            data,
            row=None,
            col=None,
            y_col=y_col,
            err_col=err_col,
            show_error_bars=show_error_bars,
            color_by=color_by,
            show_legend=True,
            **_style_kwargs,
        )
        fig.update_layout(
            xaxis_title="Average Time (s)",
            yaxis_title=y_label,
        )

    elif len(facet_cols) == 1:
        facet_col = facet_cols[0]
        facet_values = sorted(data[facet_col].unique())
        n_plots = len(facet_values)

        if vertical_single:
            # Vertical layout: one row per value, single column
            fig = make_subplots(
                rows=n_plots,
                cols=1,
                shared_xaxes=True,
                subplot_titles=[f"{facet_col}={v}" for v in facet_values],
            )
            seen_legends: set[str] = set()
            for row_idx, fv in enumerate(facet_values, start=1):
                subset = data[data[facet_col] == fv]
                show_legend = row_idx == 1
                _add_traces(
                    fig,
                    subset,
                    row=row_idx,
                    col=1,
                    y_col=y_col,
                    err_col=err_col,
                    show_error_bars=show_error_bars,
                    color_by=color_by,
                    show_legend=show_legend,
                    seen_legends=seen_legends,
                    **_style_kwargs,
                )
        else:
            # Horizontal layout: single row, one column per value (default)
            fig = make_subplots(
                rows=1,
                cols=n_plots,
                shared_yaxes=True,
                subplot_titles=[f"{facet_col}={v}" for v in facet_values],
            )
            seen_legends: set[str] = set()
            for col_idx, fv in enumerate(facet_values, start=1):
                subset = data[data[facet_col] == fv]
                show_legend = col_idx == 1
                _add_traces(
                    fig,
                    subset,
                    row=1,
                    col=col_idx,
                    y_col=y_col,
                    err_col=err_col,
                    show_error_bars=show_error_bars,
                    color_by=color_by,
                    show_legend=show_legend,
                    seen_legends=seen_legends,
                    **_style_kwargs,
                )
        fig.update_layout(
            xaxis_title="Average Time (s)",
            yaxis_title=y_label,
        )

    else:
        # Two-dimensional faceting: rows = facet_cols[0], cols = facet_cols[1]
        row_col, col_col = facet_cols[0], facet_cols[1]
        row_values = sorted(data[row_col].unique())
        col_values = sorted(data[col_col].unique())
        n_rows = len(row_values)
        n_cols = len(col_values)

        subplot_titles = [
            f"{row_col}={rv}, {col_col}={cv}"
            for rv in row_values
            for cv in col_values
        ]
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            shared_yaxes=True,
            shared_xaxes=True,
            subplot_titles=subplot_titles,
        )
        seen_legends: set[str] = set()
        for row_idx, rv in enumerate(row_values, start=1):
            for col_idx, cv in enumerate(col_values, start=1):
                subset = data[
                    (data[row_col] == rv) & (data[col_col] == cv)
                ]
                show_legend = (row_idx == 1) and (col_idx == 1)
                _add_traces(
                    fig,
                    subset,
                    row=row_idx,
                    col=col_idx,
                    y_col=y_col,
                    err_col=err_col,
                    show_error_bars=show_error_bars,
                    color_by=color_by,
                    show_legend=show_legend,
                    seen_legends=seen_legends,
                    **_style_kwargs,
                )

        fig.update_layout(
            xaxis_title="Average Time (s)",
            yaxis_title=y_label,
        )

    # ------------------------------------------------------------------
    # Common layout
    # ------------------------------------------------------------------
    title_parts = ["IMA Magnetic Measurement"]
    if filter_frequency is not None:
        title_parts.append(f"frequency={filter_frequency}")
    if filter_id is not None:
        title_parts.append(f"id={filter_id}")
    if use_baseline_subtracted:
        title_parts.append("(baseline subtracted)")

    fig.update_layout(
        title=" | ".join(title_parts),
        template="plotly_white",
        legend_title=color_by,
        hovermode="x unified",
    )

    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Simple colour palette that works well on white backgrounds
_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

# Default palettes for auto-generation of category-based styling
_SYMBOLS = [
    "circle", "square", "diamond", "cross", "x",
    "triangle-up", "triangle-down", "star", "hexagon", "pentagon",
]
_LINE_STYLES = ["solid", "dash", "dot", "dashdot"]
_SIZES = [5, 7, 9, 11, 13]
_WIDTHS = [1, 2, 3, 4]


def _color_map(values: list) -> dict:
    """Map unique values to hex colour strings."""
    return {v: _PALETTE[i % len(_PALETTE)] for i, v in enumerate(sorted(values))}


def _symbol_map(values: list, custom_map: Union[dict, None] = None) -> dict:
    """Map unique values to Plotly marker symbol names.

    Values present in *custom_map* use the custom symbol; remaining values
    are assigned symbols from :data:`_SYMBOLS` cycling round-robin.
    """
    custom = custom_map or {}
    auto_values = [v for v in sorted(values) if v not in custom]
    auto = {v: _SYMBOLS[i % len(_SYMBOLS)] for i, v in enumerate(auto_values)}
    return {**auto, **custom}


def _size_map(values: list, custom_map: Union[dict, None] = None) -> dict:
    """Map unique values to marker sizes.

    Values present in *custom_map* use the custom size; remaining values
    are assigned sizes from :data:`_SIZES` cycling round-robin.
    """
    custom = custom_map or {}
    auto_values = [v for v in sorted(values) if v not in custom]
    auto = {v: _SIZES[i % len(_SIZES)] for i, v in enumerate(auto_values)}
    return {**auto, **custom}


def _line_style_map(values: list, custom_map: Union[dict, None] = None) -> dict:
    """Map unique values to Plotly line dash styles.

    Values present in *custom_map* use the custom style; remaining values
    are assigned styles from :data:`_LINE_STYLES` cycling round-robin.
    """
    custom = custom_map or {}
    auto_values = [v for v in sorted(values) if v not in custom]
    auto = {v: _LINE_STYLES[i % len(_LINE_STYLES)] for i, v in enumerate(auto_values)}
    return {**auto, **custom}


def _line_width_map(values: list, custom_map: Union[dict, None] = None) -> dict:
    """Map unique values to line widths.

    Values present in *custom_map* use the custom width; remaining values
    are assigned widths from :data:`_WIDTHS` cycling round-robin.
    """
    custom = custom_map or {}
    auto_values = [v for v in sorted(values) if v not in custom]
    auto = {v: _WIDTHS[i % len(_WIDTHS)] for i, v in enumerate(auto_values)}
    return {**auto, **custom}


def _add_traces(
    fig: go.Figure,
    data: pd.DataFrame,
    row,
    col,
    y_col: str,
    err_col: str,
    show_error_bars: bool,
    color_by: str,
    show_legend: bool,
    seen_legends: set | None = None,
    # Global styling
    marker_size: Union[int, float, None] = None,
    marker_symbol: Union[str, None] = None,
    line_style: Union[str, None] = None,
    line_width: Union[int, float, None] = None,
    # Category-based styling
    marker_size_by: Union[str, None] = None,
    marker_size_map: Union[dict, None] = None,
    marker_symbol_by: Union[str, None] = None,
    marker_symbol_map: Union[dict, None] = None,
    line_style_by: Union[str, None] = None,
    line_style_map: Union[dict, None] = None,
    line_width_by: Union[str, None] = None,
    line_width_map: Union[dict, None] = None,
) -> None:
    """Add scatter traces to *fig* for each unique value in *color_by*."""
    if data.empty:
        return

    if seen_legends is None:
        seen_legends = set()

    colors = _color_map(data[color_by].unique().tolist())

    # Pre-compute category-based style maps (only when the column is available)
    symbols = (
        _symbol_map(data[marker_symbol_by].unique().tolist(), marker_symbol_map)
        if marker_symbol_by and marker_symbol_by in data.columns
        else None
    )
    sizes = (
        _size_map(data[marker_size_by].unique().tolist(), marker_size_map)
        if marker_size_by and marker_size_by in data.columns
        else None
    )
    styles = (
        _line_style_map(data[line_style_by].unique().tolist(), line_style_map)
        if line_style_by and line_style_by in data.columns
        else None
    )
    widths = (
        _line_width_map(data[line_width_by].unique().tolist(), line_width_map)
        if line_width_by and line_width_by in data.columns
        else None
    )

    for value, group_df in data.groupby(color_by, sort=True):
        group_df = group_df.sort_values("avg_time")

        error_y = None
        if show_error_bars and err_col in group_df.columns:
            error_y = dict(
                type="data",
                array=group_df[err_col].tolist(),
                visible=True,
            )

        legend_key = str(value)
        show_this_legend = show_legend and (legend_key not in seen_legends)
        if show_this_legend:
            seen_legends.add(legend_key)

        # ---- Determine marker properties (priority: category > global > default) ----
        # For category-based styling, look up the category value for each *_by column.
        # If the styling column equals color_by, reuse the current group value;
        # otherwise read the first row's value from that column.
        def _cat_value(by_col):
            """Return the category value for styling lookups."""
            if by_col == color_by:
                return value
            first_row = group_df[by_col].iloc[0] if by_col in group_df.columns else None
            return first_row

        marker_dict: dict = {"color": colors[value]}

        if sizes is not None:
            cat_val = _cat_value(marker_size_by)
            if cat_val in sizes:
                marker_dict["size"] = sizes[cat_val]
            else:
                marker_dict["size"] = marker_size if marker_size is not None else 5
        elif marker_size is not None:
            marker_dict["size"] = marker_size
        else:
            marker_dict["size"] = 5  # default

        if symbols is not None:
            cat_val = _cat_value(marker_symbol_by)
            if cat_val in symbols:
                marker_dict["symbol"] = symbols[cat_val]
            elif marker_symbol is not None:
                marker_dict["symbol"] = marker_symbol
        elif marker_symbol is not None:
            marker_dict["symbol"] = marker_symbol

        # ---- Determine line properties ----
        line_dict: dict = {"color": colors[value]}

        if styles is not None:
            cat_val = _cat_value(line_style_by)
            if cat_val in styles:
                line_dict["dash"] = styles[cat_val]
            elif line_style is not None:
                line_dict["dash"] = line_style
        elif line_style is not None:
            line_dict["dash"] = line_style

        if widths is not None:
            cat_val = _cat_value(line_width_by)
            if cat_val in widths:
                line_dict["width"] = widths[cat_val]
            elif line_width is not None:
                line_dict["width"] = line_width
        elif line_width is not None:
            line_dict["width"] = line_width

        trace = go.Scatter(
            x=group_df["avg_time"],
            y=group_df[y_col],
            error_y=error_y,
            mode="lines+markers",
            name=str(value),
            legendgroup=str(value),
            showlegend=show_this_legend,
            line=line_dict,
            marker=marker_dict,
        )

        if row is not None and col is not None:
            fig.add_trace(trace, row=row, col=col)
        else:
            fig.add_trace(trace)
