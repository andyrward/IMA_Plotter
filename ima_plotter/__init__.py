"""IMA Plotter – utilities for loading and analysing IMA magnetic measurement data."""

__version__ = "0.1.0"

from .loader import load_excel_files
from .transformer import subtract_baseline
from .plotter import plot_magnetic_vs_time
from .utils import parse_filename

__all__ = [
    "load_excel_files",
    "subtract_baseline",
    "plot_magnetic_vs_time",
    "parse_filename",
    "__version__",
]
