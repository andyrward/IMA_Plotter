"""Functions for loading IMA magnetic measurement data from Excel files."""

import glob
import os

import pandas as pd

from .utils import parse_filename

# Mapping from raw Excel column names to standardised tidy column names.
_COLUMN_RENAME = {
    "Group": "group",
    "Frequency": "frequency",
    "TimeIndex": "time_index",
    "Average V2abs MAGNETIC": "avg_v2abs_magnetic",
    "Std V2abs MAGNETIC": "std_v2abs_magnetic",
    "N": "n",
    "Average Time": "avg_time",
    "CV V2abs MAGNETIC (%)": "cv_v2abs_magnetic",
}


def load_excel_files(
    directory: str,
    pattern: str = "*.xlsx",
    sheet_name: str = "summary_average",
) -> pd.DataFrame:
    """Load all matching Excel files from a directory into a tidy DataFrame.

    Each file must contain a sheet named *sheet_name* (default
    ``summary_average``) with the following columns:

    * ``Group``
    * ``Frequency``
    * ``TimeIndex``
    * ``Average V2abs MAGNETIC``
    * ``Std V2abs MAGNETIC``
    * ``N``
    * ``Average Time``
    * ``CV V2abs MAGNETIC (%)``

    The filename is expected to follow the convention
    ``ExperimentName_[ID].xlsx``.  Two extra columns are added to the
    returned DataFrame:

    * ``id``         – identifier extracted from the filename
    * ``experiment`` – experiment name extracted from the filename

    Parameters
    ----------
    directory:
        Path to the directory that contains the Excel files.
    pattern:
        Glob pattern used to match files inside *directory*.
        Defaults to ``"*.xlsx"``.
    sheet_name:
        Name of the worksheet to read from each file.
        Defaults to ``"summary_average"``.

    Returns
    -------
    pd.DataFrame
        Combined tidy DataFrame with standardised lowercase column names.

    Raises
    ------
    FileNotFoundError
        If *directory* does not exist.
    ValueError
        If no files matching *pattern* are found in *directory*.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    file_paths = sorted(glob.glob(os.path.join(directory, pattern)))
    if not file_paths:
        raise ValueError(
            f"No files matching '{pattern}' found in '{directory}'"
        )

    frames: list[pd.DataFrame] = []
    for path in file_paths:
        try:
            df = pd.read_excel(path, sheet_name=sheet_name)
        except Exception as exc:
            print(f"Warning: could not read '{path}': {exc}")
            continue

        meta = parse_filename(path)
        df["id"] = meta["id"]
        df["experiment"] = meta["experiment"]

        frames.append(df)

    if not frames:
        raise ValueError(
            f"No valid data could be loaded from '{directory}' "
            f"using sheet '{sheet_name}'"
        )

    combined = pd.concat(frames, ignore_index=True)

    # Rename to standardised column names (keep any unexpected columns as-is)
    combined = combined.rename(
        columns={k: v for k, v in _COLUMN_RENAME.items() if k in combined.columns}
    )

    # Put id and experiment first, then the rest in a logical order
    ordered = ["id", "experiment", "group", "frequency", "time_index",
               "avg_v2abs_magnetic", "std_v2abs_magnetic", "n",
               "avg_time", "cv_v2abs_magnetic"]
    remaining = [c for c in combined.columns if c not in ordered]
    combined = combined[[c for c in ordered if c in combined.columns] + remaining]

    return combined
