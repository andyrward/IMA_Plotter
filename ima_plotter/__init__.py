"""IMA Plotter – utilities for loading and analysing IMA magnetic measurement data."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ima-plotter")
except PackageNotFoundError:  # package not installed (e.g. during development)
    __version__ = "unknown"

from .loader import load_excel_files
from .transformer import subtract_baseline
from .plotter import plot_magnetic_vs_time
from .utils import parse_filename
from .widgets import DataManager, PlotWidgets, create_interactive_plotter

__all__ = [
    "load_excel_files",
    "subtract_baseline",
    "plot_magnetic_vs_time",
    "parse_filename",
    "DataManager",
    "PlotWidgets",
    "create_interactive_plotter",
    "__version__",
]
