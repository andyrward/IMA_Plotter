"""Data transformation functions for IMA Plotter."""

import warnings

import numpy as np
import pandas as pd


def subtract_baseline(
    df: pd.DataFrame,
    baseline_group: str = "Cal0",
    by: list[str] | None = None,
) -> pd.DataFrame:
    """Subtract a baseline group from all other groups in each sub-group.

    For each combination of columns in *by* (default: ``["id",
    "frequency"]``), the baseline row (where ``group == baseline_group``)
    is identified and its ``avg_v2abs_magnetic`` value is subtracted from
    every other row in that sub-group.  Combined uncertainty is propagated
    as:

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

    for keys, group_df in result.groupby(by, sort=False):
        baseline_rows = group_df[group_df["group"] == baseline_group]

        if baseline_rows.empty:
            label = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
            warnings.warn(
                f"Baseline group '{baseline_group}' not found for {label}. "
                "Skipping this sub-group.",
                stacklevel=2,
            )
            continue

        # Use the mean of baseline rows in case there are multiple
        baseline_avg = baseline_rows["avg_v2abs_magnetic"].mean()
        baseline_std = baseline_rows["std_v2abs_magnetic"].mean()

        mask = group_df.index
        result.loc[mask, "delta_avg_v2abs_magnetic"] = (
            result.loc[mask, "avg_v2abs_magnetic"] - baseline_avg
        )
        result.loc[mask, "delta_std_v2abs_magnetic"] = np.sqrt(
            result.loc[mask, "std_v2abs_magnetic"] ** 2 + baseline_std ** 2
        )

    return result
