"""Utility functions for IMA Plotter."""

import os
import re


def parse_filename(filename: str) -> dict:
    """Extract experiment name and ID from a filename.

    Supports the format ``ExperimentName_[ID].xlsx``.

    Parameters
    ----------
    filename:
        The basename or full path of the Excel file, e.g.
        ``Exp1-on-off-field-summary_EX1A.xlsx``.

    Returns
    -------
    dict
        A dictionary with keys ``experiment`` and ``id``.
        If the expected format is not found both values default to the
        stem of the filename.
    """
    basename = os.path.basename(filename)
    stem = os.path.splitext(basename)[0]

    # Pattern: everything before the last underscore is the experiment name;
    # everything after is the ID.
    match = re.match(r"^(.+)_([^_]+)$", stem)
    if match:
        return {"experiment": match.group(1), "id": match.group(2)}

    # Fallback: use the full stem for both fields
    return {"experiment": stem, "id": stem}
