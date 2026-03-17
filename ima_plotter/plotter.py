"""Plotting functions for IMA magnetic measurement data."""

from __future__ import annotations

from typing import Union

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_magnetic_vs_time(
    df: pd.DataFrame,
    facet_by: Union[str, list[str], None] = None,
    filter_frequency: Union[float, list[float], None] = None,
    filter_id: Union[str, list[str], None] = None,
    show_error_bars: bool = True,
    use_baseline_subtracted: bool = False,
    color_by: str = "group",
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
        * ``"id"`` or ``"frequency"`` – one subplot per unique value
        * ``["id", "frequency"]`` – grid of subplots (rows × columns)
    filter_frequency:
        Keep only rows where ``frequency`` is in this value / list of values.
        ``None`` means no filtering.
    filter_id:
        Keep only rows where ``id`` is in this value / list of values.
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

    Returns
    -------
    plotly.graph_objects.Figure
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
    # Normalise facet_by to a list
    # ------------------------------------------------------------------
    if facet_by is None:
        facet_cols: list[str] = []
    elif isinstance(facet_by, str):
        facet_cols = [facet_by]
    else:
        facet_cols = list(facet_by)

    if len(facet_cols) > 2:
        raise ValueError(
            f"facet_by supports at most 2 columns, got {len(facet_cols)}: {facet_cols}"
        )

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
        )
        fig.update_layout(
            xaxis_title="Average Time (s)",
            yaxis_title=y_label,
        )

    elif len(facet_cols) == 1:
        facet_col = facet_cols[0]
        facet_values = sorted(data[facet_col].unique())
        n_plots = len(facet_values)

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


def _color_map(values: list) -> dict:
    """Map unique values to hex colour strings."""
    return {v: _PALETTE[i % len(_PALETTE)] for i, v in enumerate(sorted(values))}


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
) -> None:
    """Add scatter traces to *fig* for each unique value in *color_by*."""
    if data.empty:
        return

    if seen_legends is None:
        seen_legends = set()

    colors = _color_map(data[color_by].unique().tolist())

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

        trace = go.Scatter(
            x=group_df["avg_time"],
            y=group_df[y_col],
            error_y=error_y,
            mode="lines+markers",
            name=str(value),
            legendgroup=str(value),
            showlegend=show_this_legend,
            line=dict(color=colors[value]),
            marker=dict(color=colors[value], size=5),
        )

        if row is not None and col is not None:
            fig.add_trace(trace, row=row, col=col)
        else:
            fig.add_trace(trace)
