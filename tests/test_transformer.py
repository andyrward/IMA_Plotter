"""Tests for ima_plotter.transformer."""

import warnings

import numpy as np
import pandas as pd
import pytest

from ima_plotter.transformer import subtract_baseline


def _make_df(rows):
    """Build a minimal tidy DataFrame for testing."""
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# subtract_baseline – time-matched subtraction
# ---------------------------------------------------------------------------


class TestSubtractBaseline:
    @staticmethod
    def _base_df():
        """Two groups (Cal0, Cal1) with three time points each."""
        return _make_df(
            [
                # Cal0 at time_index 1, 2, 3
                {"id": "A", "frequency": 2, "group": "Cal0", "time_index": 1,
                 "avg_v2abs_magnetic": 10.0, "std_v2abs_magnetic": 1.0},
                {"id": "A", "frequency": 2, "group": "Cal0", "time_index": 2,
                 "avg_v2abs_magnetic": 20.0, "std_v2abs_magnetic": 2.0},
                {"id": "A", "frequency": 2, "group": "Cal0", "time_index": 3,
                 "avg_v2abs_magnetic": 30.0, "std_v2abs_magnetic": 3.0},
                # Cal1 at time_index 1, 2, 3
                {"id": "A", "frequency": 2, "group": "Cal1", "time_index": 1,
                 "avg_v2abs_magnetic": 15.0, "std_v2abs_magnetic": 1.5},
                {"id": "A", "frequency": 2, "group": "Cal1", "time_index": 2,
                 "avg_v2abs_magnetic": 25.0, "std_v2abs_magnetic": 2.5},
                {"id": "A", "frequency": 2, "group": "Cal1", "time_index": 3,
                 "avg_v2abs_magnetic": 35.0, "std_v2abs_magnetic": 3.5},
            ]
        )

    def test_time_matched_delta_values(self):
        """Cal1 delta should equal Cal1 - Cal0 at each matching time_index."""
        df = self._base_df()
        result = subtract_baseline(df, baseline_group="Cal0")

        cal1 = result[result["group"] == "Cal1"].sort_values("time_index")
        expected_deltas = [
            15.0 - 10.0,  # time_index=1
            25.0 - 20.0,  # time_index=2
            35.0 - 30.0,  # time_index=3
        ]
        assert list(cal1["delta_avg_v2abs_magnetic"]) == expected_deltas

    def test_cal0_delta_is_zero(self):
        """Cal0 subtracted from itself should give delta=0 at every time point."""
        df = self._base_df()
        result = subtract_baseline(df, baseline_group="Cal0")

        cal0 = result[result["group"] == "Cal0"]
        assert (cal0["delta_avg_v2abs_magnetic"] == 0.0).all()

    def test_uncertainty_propagation(self):
        """delta_std should equal sqrt(std_original^2 + std_baseline^2)."""
        df = self._base_df()
        result = subtract_baseline(df, baseline_group="Cal0")

        cal1 = result[result["group"] == "Cal1"].sort_values("time_index")
        # time_index=1: sqrt(1.5^2 + 1.0^2)
        expected = np.sqrt(1.5**2 + 1.0**2)
        assert pytest.approx(cal1["delta_std_v2abs_magnetic"].iloc[0]) == expected

    def test_original_columns_preserved(self):
        """avg_v2abs_magnetic and std_v2abs_magnetic must remain unchanged."""
        df = self._base_df()
        result = subtract_baseline(df, baseline_group="Cal0")

        pd.testing.assert_series_equal(
            result["avg_v2abs_magnetic"].reset_index(drop=True),
            df["avg_v2abs_magnetic"].reset_index(drop=True),
        )

    def test_returns_copy(self):
        """The original DataFrame must not be mutated."""
        df = self._base_df()
        assert "delta_avg_v2abs_magnetic" not in df.columns
        subtract_baseline(df)
        assert "delta_avg_v2abs_magnetic" not in df.columns

    def test_missing_baseline_at_time_point_leaves_nan(self):
        """If Cal0 is absent at a specific time_index, delta should be NaN."""
        df = _make_df(
            [
                # Cal0 only at time_index 1
                {"id": "A", "frequency": 2, "group": "Cal0", "time_index": 1,
                 "avg_v2abs_magnetic": 10.0, "std_v2abs_magnetic": 1.0},
                # Cal1 at time_index 1 and 2
                {"id": "A", "frequency": 2, "group": "Cal1", "time_index": 1,
                 "avg_v2abs_magnetic": 15.0, "std_v2abs_magnetic": 1.5},
                {"id": "A", "frequency": 2, "group": "Cal1", "time_index": 2,
                 "avg_v2abs_magnetic": 25.0, "std_v2abs_magnetic": 2.5},
            ]
        )
        result = subtract_baseline(df, baseline_group="Cal0")

        cal1 = result[result["group"] == "Cal1"].sort_values("time_index")
        assert pytest.approx(cal1["delta_avg_v2abs_magnetic"].iloc[0]) == 5.0
        assert np.isnan(cal1["delta_avg_v2abs_magnetic"].iloc[1])

    def test_custom_time_col(self):
        """time_col parameter should allow using a different column name."""
        df = _make_df(
            [
                {"id": "A", "frequency": 2, "group": "Cal0", "t": 0,
                 "avg_v2abs_magnetic": 100.0, "std_v2abs_magnetic": 5.0},
                {"id": "A", "frequency": 2, "group": "Cal1", "t": 0,
                 "avg_v2abs_magnetic": 110.0, "std_v2abs_magnetic": 5.0},
            ]
        )
        result = subtract_baseline(df, baseline_group="Cal0", time_col="t")
        cal1 = result[result["group"] == "Cal1"]
        assert pytest.approx(cal1["delta_avg_v2abs_magnetic"].iloc[0]) == 10.0

    def test_multiple_ids_independent(self):
        """Each id+frequency combination should be subtracted independently."""
        df = _make_df(
            [
                {"id": "A", "frequency": 2, "group": "Cal0", "time_index": 1,
                 "avg_v2abs_magnetic": 10.0, "std_v2abs_magnetic": 1.0},
                {"id": "A", "frequency": 2, "group": "Cal1", "time_index": 1,
                 "avg_v2abs_magnetic": 15.0, "std_v2abs_magnetic": 1.0},
                {"id": "B", "frequency": 2, "group": "Cal0", "time_index": 1,
                 "avg_v2abs_magnetic": 50.0, "std_v2abs_magnetic": 1.0},
                {"id": "B", "frequency": 2, "group": "Cal1", "time_index": 1,
                 "avg_v2abs_magnetic": 60.0, "std_v2abs_magnetic": 1.0},
            ]
        )
        result = subtract_baseline(df, baseline_group="Cal0")

        delta_a = result[(result["id"] == "A") & (result["group"] == "Cal1")][
            "delta_avg_v2abs_magnetic"
        ].iloc[0]
        delta_b = result[(result["id"] == "B") & (result["group"] == "Cal1")][
            "delta_avg_v2abs_magnetic"
        ].iloc[0]

        assert pytest.approx(delta_a) == 5.0
        assert pytest.approx(delta_b) == 10.0

    def test_warns_on_duplicate_baseline(self):
        """A warning should be emitted when multiple baseline rows exist for one time point."""
        df = _make_df(
            [
                # Two Cal0 rows at the same time_index (duplicate)
                {"id": "A", "frequency": 2, "group": "Cal0", "time_index": 1,
                 "avg_v2abs_magnetic": 10.0, "std_v2abs_magnetic": 1.0},
                {"id": "A", "frequency": 2, "group": "Cal0", "time_index": 1,
                 "avg_v2abs_magnetic": 12.0, "std_v2abs_magnetic": 1.0},
                {"id": "A", "frequency": 2, "group": "Cal1", "time_index": 1,
                 "avg_v2abs_magnetic": 20.0, "std_v2abs_magnetic": 1.0},
            ]
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            subtract_baseline(df, baseline_group="Cal0")

        assert any("Multiple baseline rows" in str(w.message) for w in caught)

    def test_integration_with_sample_data(self):
        """Time-matched subtraction should work on the real sample data."""
        import os
        from ima_plotter.loader import load_excel_files

        sample_dir = os.path.join(os.path.dirname(__file__), "..", "sample_data")
        df = load_excel_files(sample_dir)
        result = subtract_baseline(df, baseline_group="Cal0")

        # Cal0 delta should be exactly 0 everywhere it has a match
        cal0 = result[result["group"] == "Cal0"]
        assert (cal0["delta_avg_v2abs_magnetic"] == 0.0).all()

        # Non-Cal0 groups should have non-NaN deltas where Cal0 is present
        non_cal0 = result[result["group"] != "Cal0"]
        assert non_cal0["delta_avg_v2abs_magnetic"].notna().any()
