# IMA Plotter

A Python utility for loading, transforming, and interactively plotting IMA magnetic measurement data from Excel files.

## Features

- Load multiple `.xlsx` files from a directory into a single tidy DataFrame
- Automatic ID and experiment name extraction from filenames
- Baseline subtraction (e.g. subtract Cal0 from Cal1, Cal2, …) with combined uncertainty propagation
- Interactive Plotly figures with flexible faceting and filtering

## Installation

```bash
pip install -e .                        # core dependencies
pip install -e ".[notebook]"            # + Jupyter support
pip install -e ".[dev]"                 # + pytest
```

## Quick Start

```python
from ima_plotter import load_excel_files, subtract_baseline, plot_magnetic_vs_time

# 1. Load all Excel files from a directory
df = load_excel_files("sample_data/")

# 2. Subtract the Cal0 baseline within each id + frequency combination
df_sub = subtract_baseline(df, baseline_group="Cal0")

# 3. Plot – facet by ID, filter to 2 Hz, show error bars
fig = plot_magnetic_vs_time(
    df_sub,
    facet_by="id",
    filter_frequency=2,
    filter_group=["Cal1", "Cal2", "Cal3"],  # exclude Cal0 (baseline)
    use_baseline_subtracted=True,
    show_error_bars=True,
)
fig.show()
```

Run the bundled CLI example:

```bash
python main.py
```

## Expected Excel Structure

| Column | Description |
|---|---|
| `Group` | Measurement group (e.g. `Cal0`, `Cal1`, `Cal2`) |
| `Frequency` | Measurement frequency (Hz) |
| `TimeIndex` | Time-point index |
| `Average V2abs MAGNETIC` | Mean magnetic signal |
| `Std V2abs MAGNETIC` | Standard deviation |
| `N` | Sample count |
| `Average Time` | Average elapsed time (s) |
| `CV V2abs MAGNETIC (%)` | Coefficient of variation |

Filenames must follow the pattern `ExperimentName_[ID].xlsx`.

## Demo Notebook

Open `notebooks/demo.ipynb` for a fully worked example covering:

1. Data loading
2. Data exploration
3. Baseline subtraction
4. Basic plot (single frequency)
5. Faceting by ID
6. Faceting by frequency
7. Combined faceting (ID × frequency)
8. Exporting processed data to CSV

```bash
jupyter notebook notebooks/demo.ipynb
```

## Running Tests

```bash
python -m pytest tests/ -v
```
