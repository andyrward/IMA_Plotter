"""Data transformation functions for IMA Plotter."""

import warnings

import numpy as np
import pandas as pd


def subtract_baseline(
    df: pd.DataFrame,
    baseline_group: str = "Cal0",
    by: list[str] | None = None,
    time_col: str = "time_index",
) -> pd.DataFrame:
    """Subtract a baseline group from all other groups using time-matched subtraction.

    For each combination of columns in *by* (default: ``["id",
    "frequency"]``) **and** each value of *time_col* (default:
    ``"time_index"``), the baseline row (where ``group == baseline_group``)
    at that specific time point is identified and its
    ``avg_v2abs_magnetic`` value is subtracted from every other row at
    the same time point.  If no baseline row exists for a particular time
    point the delta columns are left as ``NaN`` for that time point.
    Combined uncertainty is propagated as:

    .. math::

        \\delta_{combined} = \\sqrt{\\sigma_{original}^2 + \\sigma_{baseline}^2}

    The original columns are preserved.  Two new columns are added:

    * ``delta_avg_v2abs_magnetic``
    * ``delta_std_v2abs_magnetic``

    Parameters
    ----------
    df:
        Tidy DataFrame as returned by :func:`~ima_plotter.loader.load_excel_files`.
    baseline_group:
        Value of the ``group`` column that identifies the baseline.
        Defaults to ``"Cal0"``.
    by:
        List of column names that define the sub-groups within which the
        baseline is applied.  Defaults to ``["id", "frequency"]``.
    time_col:
        Column name that identifies the time point for matching.
        Defaults to ``"time_index"``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with the two additional delta columns.
    """
    if by is None:
        by = ["id", "frequency"]

    result = df.copy()
    result["delta_avg_v2abs_magnetic"] = np.nan
    result["delta_std_v2abs_magnetic"] = np.nan

    groupby_cols = by + [time_col]

    for keys, group_df in result.groupby(groupby_cols, sort=False):
        baseline_rows = group_df[group_df["group"] == baseline_group]

        if baseline_rows.empty:
            # No baseline at this time point – leave delta as NaN
            continue

        if len(baseline_rows) > 1:
            label = dict(zip(groupby_cols, keys if isinstance(keys, tuple) else (keys,)))
            warnings.warn(
                f"Multiple baseline rows found for {label}. "
                "Using the first one.",
                stacklevel=2,
            )

        # Use the baseline value at this specific time point
        baseline_avg = baseline_rows["avg_v2abs_magnetic"].iloc[0]
        baseline_std = baseline_rows["std_v2abs_magnetic"].iloc[0]

        mask = group_df.index
        result.loc[mask, "delta_avg_v2abs_magnetic"] = (
            result.loc[mask, "avg_v2abs_magnetic"] - baseline_avg
        )
        result.loc[mask, "delta_std_v2abs_magnetic"] = np.sqrt(
            result.loc[mask, "std_v2abs_magnetic"] ** 2 + baseline_std ** 2
        )

    return result
