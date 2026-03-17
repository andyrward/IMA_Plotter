"""Tests for ima_plotter.loader and ima_plotter.utils."""

import os
import tempfile

import pandas as pd
import pytest

from ima_plotter.loader import load_excel_files
from ima_plotter.utils import parse_filename

# Path to the sample data shipped with the repository
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_data")


# ---------------------------------------------------------------------------
# parse_filename tests
# ---------------------------------------------------------------------------


class TestParseFilename:
    def test_standard_format(self):
        result = parse_filename("Exp1-on-off-field-summary_EX1A.xlsx")
        assert result["id"] == "EX1A"
        assert result["experiment"] == "Exp1-on-off-field-summary"

    def test_full_path(self):
        result = parse_filename("/some/dir/MyExp_REF.xlsx")
        assert result["id"] == "REF"
        assert result["experiment"] == "MyExp"

    def test_no_underscore_fallback(self):
        result = parse_filename("nodash.xlsx")
        assert result["id"] == "nodash"
        assert result["experiment"] == "nodash"

    def test_multiple_underscores_uses_last(self):
        # Only the last underscore is used as the separator
        result = parse_filename("Exp_part1_ID42.xlsx")
        assert result["id"] == "ID42"
        assert result["experiment"] == "Exp_part1"


# ---------------------------------------------------------------------------
# load_excel_files tests
# ---------------------------------------------------------------------------


class TestLoadExcelFiles:
    def test_loads_all_sample_files(self):
        df = load_excel_files(SAMPLE_DIR)
        # Five sample files with varying row counts (EX1A=240, EX1B=240,
        # EX1C=16, EX1D=16, REF=240) → 752 rows total
        assert len(df) == 752

    def test_id_column_extracted(self):
        df = load_excel_files(SAMPLE_DIR)
        ids = set(df["id"].unique())
        assert ids == {"EX1A", "EX1B", "EX1C", "EX1D", "REF"}

    def test_standardised_column_names(self):
        df = load_excel_files(SAMPLE_DIR)
        expected = {
            "id", "experiment", "group", "frequency", "time_index",
            "avg_v2abs_magnetic", "std_v2abs_magnetic", "n",
            "avg_time", "cv_v2abs_magnetic",
        }
        assert expected.issubset(set(df.columns))

    def test_no_raw_column_names_remain(self):
        df = load_excel_files(SAMPLE_DIR)
        raw_names = [
            "Group", "Frequency", "TimeIndex",
            "Average V2abs MAGNETIC", "Std V2abs MAGNETIC",
            "Average Time", "CV V2abs MAGNETIC (%)",
        ]
        for name in raw_names:
            assert name not in df.columns

    def test_missing_directory_raises(self):
        with pytest.raises(FileNotFoundError):
            load_excel_files("/nonexistent/path")

    def test_no_matching_files_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError):
                load_excel_files(tmpdir, pattern="*.xlsx")

    def test_experiment_column_present(self):
        df = load_excel_files(SAMPLE_DIR)
        assert "experiment" in df.columns
        # All rows from the same experiment family
        assert df["experiment"].nunique() == 1
        assert df["experiment"].iloc[0] == "Exp1-on-off-field-summary"
