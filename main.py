"""IMA Plotter – command-line example.

Demonstrates loading sample data, applying baseline subtraction, and
creating a basic interactive plot.
"""

import os

from ima_plotter import load_excel_files, subtract_baseline, plot_magnetic_vs_time

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_data")


def main() -> None:
    # 1. Load all Excel files from the sample_data directory
    print("Loading data …")
    df = load_excel_files(SAMPLE_DIR)
    print(f"  Loaded {len(df)} rows from {df['id'].nunique()} files.")
    print(f"  IDs      : {sorted(df['id'].unique())}")
    print(f"  Groups   : {sorted(df['group'].unique())}")
    print(f"  Freqs    : {sorted(df['frequency'].unique())}")

    # 2. Apply baseline subtraction (Cal0 is the baseline)
    print("\nApplying baseline subtraction (Cal0) …")
    df_sub = subtract_baseline(df, baseline_group="Cal0")
    print("  Done.  New columns: delta_avg_v2abs_magnetic, delta_std_v2abs_magnetic")

    # 3. Create a basic interactive plot (single frequency, all IDs)
    print("\nCreating plot for frequency=2 Hz, faceted by ID …")
    fig = plot_magnetic_vs_time(
        df_sub,
        facet_by="id",
        filter_frequency=2,
        use_baseline_subtracted=True,
        show_error_bars=True,
    )
    fig.show()
    print("  Plot displayed in your browser.")


if __name__ == "__main__":
    main()
