"""Interactive ipywidgets interface for the IMA Plotter toolbox.

Provides:
    DataManager   – loads data and manages baseline subtraction.
    PlotWidgets   – builds and manages all ipywidgets for plot configuration.
    create_interactive_plotter – convenience factory for a complete interface.
"""

from __future__ import annotations

import os

import ipywidgets as widgets
from IPython.display import display
from ipyfilechooser import FileChooser

from .loader import load_excel_files
from .plotter import plot_magnetic_vs_time
from .transformer import subtract_baseline


# ---------------------------------------------------------------------------
# DataManager
# ---------------------------------------------------------------------------


class DataManager:
    """Manages loaded IMA data and optional baseline-subtracted variant.

    Parameters
    ----------
    None – use :meth:`load_data` to populate.
    """

    def __init__(self) -> None:
        self._data = None
        self._data_baseline_subtracted = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def data(self):
        """Raw loaded DataFrame, or ``None`` if not yet loaded."""
        return self._data

    @property
    def data_baseline_subtracted(self):
        """Baseline-subtracted DataFrame, or ``None`` if not yet computed."""
        return self._data_baseline_subtracted

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def load_data(self, directory: str, pattern: str = "*.xlsx") -> None:
        """Load Excel files from *directory* matching *pattern*.

        Parameters
        ----------
        directory:
            Path to the directory containing ``.xlsx`` files.
        pattern:
            Glob pattern used to select files (default ``"*.xlsx"``).
        """
        self._data = load_excel_files(directory, pattern=pattern)
        # Reset previously computed baseline subtraction
        self._data_baseline_subtracted = None

    def subtract_baseline(self, baseline_group: str = "Cal0") -> None:
        """Compute baseline-subtracted data in-place.

        Parameters
        ----------
        baseline_group:
            The group label to use as the baseline (default ``"Cal0"``).

        Raises
        ------
        RuntimeError
            If :meth:`load_data` has not been called yet.
        """
        if self._data is None:
            raise RuntimeError("No data loaded. Call load_data() first.")
        self._data_baseline_subtracted = subtract_baseline(
            self._data, baseline_group=baseline_group
        )

    def get_data_info(self) -> dict:
        """Return a summary dict about the currently loaded data.

        Returns
        -------
        dict with keys:
            ``rows``, ``unique_ids``, ``frequencies``, ``groups``.
        """
        if self._data is None:
            return {}
        df = self._data
        return {
            "rows": len(df),
            "unique_ids": self.get_unique_values("id"),
            "frequencies": self.get_unique_values("frequency"),
            "groups": self.get_unique_values("group"),
        }

    def get_unique_values(self, column: str) -> list:
        """Return sorted unique values for *column*, or ``[]`` if unavailable.

        Parameters
        ----------
        column:
            Column name in the loaded DataFrame.
        """
        if self._data is None or column not in self._data.columns:
            return []
        return sorted(self._data[column].dropna().unique().tolist())

    def get_column_names(self) -> list[str]:
        """Return list of column names in the loaded data."""
        if self._data is None:
            return []
        return list(self._data.columns)


# ---------------------------------------------------------------------------
# PlotWidgets
# ---------------------------------------------------------------------------

_FACET_ROW_COL_OPTIONS = [None, "id", "frequency"]

_MAX_FACET_UNIQUE_VALUES = 20

_MARKER_SIZE_OPTIONS = [None, 5, 7, 9, 11, 13, 15, 20]

_MARKER_SYMBOL_OPTIONS = [
    None,
    "circle",
    "square",
    "diamond",
    "cross",
    "x",
    "triangle-up",
    "triangle-down",
    "star",
    "hexagon",
    "pentagon",
]

_LINE_STYLE_OPTIONS = [None, "solid", "dash", "dot", "dashdot"]

_LINE_WIDTH_OPTIONS = [None, 1, 2, 3, 4, 5, 6]

_DESCRIPTION_STYLE = {"description_width": "160px"}
_WIDGET_LAYOUT = widgets.Layout(width="340px")


def _make_dropdown(description: str, options: list, value=None) -> widgets.Dropdown:
    """Create a :class:`~ipywidgets.Dropdown` with consistent styling."""
    return widgets.Dropdown(
        description=description,
        options=options,
        value=value,
        style=_DESCRIPTION_STYLE,
        layout=_WIDGET_LAYOUT,
    )


def _make_select_multiple(description: str, options: list) -> widgets.SelectMultiple:
    """Create a :class:`~ipywidgets.SelectMultiple` with consistent styling."""
    return widgets.SelectMultiple(
        description=description,
        options=options,
        value=[],
        style=_DESCRIPTION_STYLE,
        layout=widgets.Layout(width="340px", height="90px"),
    )


class PlotWidgets:
    """Creates and manages all ipywidgets for configuring a plot.

    Parameters
    ----------
    data_manager:
        A :class:`DataManager` instance used to populate filter options and
        column-based dropdowns.
    """

    def __init__(self, data_manager: DataManager) -> None:
        self._dm = data_manager
        self.layout: widgets.Widget | None = None

        # Placeholders – populated by create_widgets()
        self.filter_frequency: widgets.SelectMultiple | None = None
        self.filter_id: widgets.SelectMultiple | None = None
        self.filter_group: widgets.SelectMultiple | None = None

        self.facet_row: widgets.Dropdown | None = None
        self.facet_col: widgets.Dropdown | None = None
        self.color_by: widgets.Dropdown | None = None
        self.show_error_bars: widgets.Checkbox | None = None
        self.use_baseline_subtracted: widgets.Checkbox | None = None

        self.marker_size: widgets.Dropdown | None = None
        self.marker_symbol: widgets.Dropdown | None = None
        self.line_style: widgets.Dropdown | None = None
        self.line_width: widgets.Dropdown | None = None
        self.marker_size_by: widgets.Dropdown | None = None
        self.marker_symbol_by: widgets.Dropdown | None = None
        self.line_style_by: widgets.Dropdown | None = None
        self.line_width_by: widgets.Dropdown | None = None

        self.y_multiplier: widgets.FloatText | None = None
        self.fig_width: widgets.IntText | None = None
        self.fig_height: widgets.IntText | None = None

        # Action buttons
        self.btn_load: widgets.Button | None = None
        self.btn_plot: widgets.Button | None = None
        self.btn_baseline: widgets.Button | None = None
        self.btn_reset: widgets.Button | None = None

        # Directory chooser
        self.directory_chooser: FileChooser | None = None

        self.create_widgets()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def create_widgets(self) -> None:
        """Initialise all widgets and assemble :attr:`layout`."""
        self._build_widgets()
        self._assemble_layout()

    def get_plot_params(self) -> dict:
        """Return current widget values mapped to :func:`plot_magnetic_vs_time` kwargs.

        Returns
        -------
        dict
            Ready to be unpacked as ``**kwargs`` into
            :func:`~ima_plotter.plotter.plot_magnetic_vs_time`.
        """
        params: dict = {}

        # --- Facet by ---
        facet_row = self.facet_row.value
        facet_col = self.facet_col.value

        if facet_row is None and facet_col is None:
            params["facet_by"] = None
        elif facet_row is not None and facet_col is None:
            params["facet_by"] = [facet_row, None]
        elif facet_row is None and facet_col is not None:
            params["facet_by"] = [None, facet_col]
        else:
            params["facet_by"] = [facet_row, facet_col]

        # --- Filters ---
        freq_sel = list(self.filter_frequency.value)
        params["filter_frequency"] = freq_sel if freq_sel else None

        id_sel = list(self.filter_id.value)
        params["filter_id"] = id_sel if id_sel else None

        grp_sel = list(self.filter_group.value)
        params["filter_group"] = grp_sel if grp_sel else None

        # --- Display options ---
        params["color_by"] = self.color_by.value or "group"
        params["show_error_bars"] = self.show_error_bars.value
        params["use_baseline_subtracted"] = self.use_baseline_subtracted.value

        # --- Global styling ---
        params["marker_size"] = self.marker_size.value
        params["marker_symbol"] = self.marker_symbol.value
        params["line_style"] = self.line_style.value
        params["line_width"] = self.line_width.value

        # --- Category-based styling ---
        params["marker_size_by"] = self.marker_size_by.value
        params["marker_symbol_by"] = self.marker_symbol_by.value
        params["line_style_by"] = self.line_style_by.value
        params["line_width_by"] = self.line_width_by.value

        # --- Advanced ---
        params["y_multiplier"] = self.y_multiplier.value
        params["fig_width"] = self.fig_width.value
        params["fig_height"] = self.fig_height.value

        return params

    def reset_to_defaults(self) -> None:
        """Reset all widgets to their default values."""
        self.facet_row.value = None
        self.facet_col.value = None
        self.color_by.value = "group"
        self.show_error_bars.value = True
        self.use_baseline_subtracted.value = False

        self.marker_size.value = None
        self.marker_symbol.value = None
        self.line_style.value = None
        self.line_width.value = None
        self.marker_size_by.value = None
        self.marker_symbol_by.value = None
        self.line_style_by.value = None
        self.line_width_by.value = None

        self.y_multiplier.value = 1.0
        self.fig_width.value = 1000
        self.fig_height.value = 600

        self.filter_frequency.value = []
        self.filter_id.value = []
        self.filter_group.value = []

    def update_filter_options(self) -> None:
        """Refresh filter dropdowns and column-based selectors from loaded data."""
        self.filter_frequency.options = self._dm.get_unique_values("frequency")
        self.filter_frequency.value = []

        self.filter_id.options = self._dm.get_unique_values("id")
        self.filter_id.value = []

        self.filter_group.options = self._dm.get_unique_values("group")
        self.filter_group.value = []

        # Update column-based dropdowns
        col_opts = [None] + self._dm.get_column_names()
        self.color_by.options = self._dm.get_column_names() or ["group"]
        if "group" in (self._dm.get_column_names() or []):
            self.color_by.value = "group"

        for w in (
            self.marker_size_by,
            self.marker_symbol_by,
            self.line_style_by,
            self.line_width_by,
        ):
            w.options = col_opts
            w.value = None

        # Update facet dropdowns with dynamic column options
        facet_opts = self._get_facet_columns()
        self.facet_row.options = facet_opts
        self.facet_row.value = None
        self.facet_col.options = facet_opts
        self.facet_col.value = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_facet_columns(self) -> list:
        """Return suitable columns for faceting from the loaded DataFrame.

        Always includes ``None`` as the first entry (no faceting).  Columns
        with between 2 and 20 unique values are considered good candidates;
        purely numeric measurement columns are excluded.
        """
        if self._dm.data is None:
            return list(_FACET_ROW_COL_OPTIONS)  # fallback defaults

        _SKIP = {"time_index", "avg_time", "magnetic", "magnetic_err"}
        facet_candidates: list = [None]
        for col in self._dm.data.columns:
            if col in _SKIP:
                continue
            n_unique = self._dm.data[col].nunique()
            if 1 < n_unique <= _MAX_FACET_UNIQUE_VALUES:
                facet_candidates.append(col)
        return facet_candidates

    def _build_widgets(self) -> None:
        """Instantiate all individual widgets."""
        # --- Data loader ---
        self.directory_chooser = FileChooser(
            path=os.getcwd(),
            select_default=True,
            show_only_dirs=True,
            title="<b>Select Data Directory</b>",
            show_hidden=False,
            layout=widgets.Layout(width="600px"),
        )
        self.btn_load = widgets.Button(
            description="Load Data",
            button_style="info",
            icon="folder-open",
            layout=widgets.Layout(width="130px"),
        )

        # --- Filters ---
        self.filter_frequency = _make_select_multiple("Frequency:", [])
        self.filter_id = _make_select_multiple("ID:", [])
        self.filter_group = _make_select_multiple("Group:", [])

        # --- Display options ---
        facet_opts = self._get_facet_columns()
        self.facet_row = _make_dropdown("Facet Row:", facet_opts, None)
        self.facet_col = _make_dropdown("Facet Column:", facet_opts, None)
        self.color_by = _make_dropdown("Color By:", ["group"], "group")
        self.show_error_bars = widgets.Checkbox(
            description="Show Error Bars",
            value=True,
            style=_DESCRIPTION_STYLE,
        )
        self.use_baseline_subtracted = widgets.Checkbox(
            description="Use Baseline Subtracted",
            value=False,
            style=_DESCRIPTION_STYLE,
        )

        # --- Global styling ---
        self.marker_size = _make_dropdown("Marker Size:", _MARKER_SIZE_OPTIONS, None)
        self.marker_symbol = _make_dropdown(
            "Marker Symbol:", _MARKER_SYMBOL_OPTIONS, None
        )
        self.line_style = _make_dropdown("Line Style:", _LINE_STYLE_OPTIONS, None)
        self.line_width = _make_dropdown("Line Width:", _LINE_WIDTH_OPTIONS, None)

        # --- Category-based styling (populated after data load) ---
        col_opts: list = [None]
        self.marker_size_by = _make_dropdown("Marker Size By:", col_opts, None)
        self.marker_symbol_by = _make_dropdown("Marker Symbol By:", col_opts, None)
        self.line_style_by = _make_dropdown("Line Style By:", col_opts, None)
        self.line_width_by = _make_dropdown("Line Width By:", col_opts, None)

        # --- Advanced ---
        self.y_multiplier = widgets.FloatText(
            description="Y Multiplier:",
            value=1.0,
            step=0.1,
            style=_DESCRIPTION_STYLE,
            layout=_WIDGET_LAYOUT,
        )
        self.fig_width = widgets.IntText(
            description="Figure Width:",
            value=1000,
            step=50,
            style=_DESCRIPTION_STYLE,
            layout=_WIDGET_LAYOUT,
        )
        self.fig_height = widgets.IntText(
            description="Figure Height:",
            value=600,
            step=50,
            style=_DESCRIPTION_STYLE,
            layout=_WIDGET_LAYOUT,
        )

        # --- Action buttons ---
        self.btn_plot = widgets.Button(
            description="Generate Plot",
            button_style="success",
            icon="bar-chart",
            disabled=True,
            layout=widgets.Layout(width="150px"),
        )
        self.btn_baseline = widgets.Button(
            description="Subtract Baseline",
            button_style="warning",
            icon="minus-circle",
            disabled=True,
            layout=widgets.Layout(width="160px"),
        )
        self.btn_reset = widgets.Button(
            description="Reset Defaults",
            button_style="",
            icon="refresh",
            layout=widgets.Layout(width="150px"),
        )

    def _assemble_layout(self) -> None:
        """Assemble widgets into a tabbed layout."""
        # ----- Tab 1: Filters -----
        tab_filters = widgets.VBox(
            [
                widgets.Label("Select values to include (hold Ctrl/⌘ for multiple):"),
                self.filter_frequency,
                self.filter_id,
                self.filter_group,
            ],
            layout=widgets.Layout(padding="10px"),
        )

        # ----- Tab 2: Display Options -----
        tab_display = widgets.VBox(
            [
                self.facet_row,
                self.facet_col,
                self.color_by,
                self.show_error_bars,
                self.use_baseline_subtracted,
                widgets.HTML("<b>Figure Size</b>"),
                self.fig_width,
                self.fig_height,
            ],
            layout=widgets.Layout(padding="10px"),
        )

        # ----- Tab 3: Styling -----
        global_styling = widgets.VBox(
            [
                widgets.HTML("<b>Global Styling</b>"),
                self.marker_size,
                self.marker_symbol,
                self.line_style,
                self.line_width,
            ]
        )
        category_styling = widgets.VBox(
            [
                widgets.HTML("<b>Category-Based Styling</b>"),
                self.marker_size_by,
                self.marker_symbol_by,
                self.line_style_by,
                self.line_width_by,
            ]
        )
        tab_styling = widgets.HBox(
            [global_styling, category_styling],
            layout=widgets.Layout(padding="10px"),
        )

        # ----- Tab 4: Advanced -----
        tab_advanced = widgets.VBox(
            [self.y_multiplier],
            layout=widgets.Layout(padding="10px"),
        )

        # ----- Tab container -----
        tab = widgets.Tab()
        tab.children = [tab_filters, tab_display, tab_styling, tab_advanced]
        tab.set_title(0, "Filters")
        tab.set_title(1, "Display Options")
        tab.set_title(2, "Styling")
        tab.set_title(3, "Advanced")

        # ----- Data loader section -----
        loader_section = widgets.VBox(
            [
                widgets.HTML("<h3>📁 Data Loader</h3>"),
                self.directory_chooser,
                widgets.HBox(
                    [self.btn_load],
                    layout=widgets.Layout(padding="6px 0px"),
                ),
            ]
        )

        # ----- Action buttons row -----
        btn_row = widgets.HBox(
            [self.btn_reset, self.btn_plot, self.btn_baseline],
            layout=widgets.Layout(padding="6px 0px"),
        )

        self.layout = widgets.VBox(
            [
                loader_section,
                widgets.HTML("<h3>🎛️ Plot Controls</h3>"),
                tab,
                btn_row,
            ],
            layout=widgets.Layout(border="1px solid #ccc", padding="12px"),
        )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_interactive_plotter() -> tuple[DataManager, PlotWidgets, widgets.Output]:
    """Create a complete interactive plotter interface.

    Returns
    -------
    data_manager : DataManager
        The data manager instance.
    plot_widgets : PlotWidgets
        The widget container (display ``plot_widgets.layout``).
    output : widgets.Output
        The output widget where plots are rendered.

    Examples
    --------
    >>> data_manager, plot_widgets, output = create_interactive_plotter()
    >>> display(plot_widgets.layout)
    >>> display(output)
    """
    data_manager = DataManager()
    plot_widgets = PlotWidgets(data_manager)
    output = widgets.Output()

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------

    def _on_load(_btn) -> None:
        directory = plot_widgets.directory_chooser.selected_path
        with output:
            output.clear_output(wait=True)
            if not directory:
                print("⚠️  Please select a data directory using the file browser.")
                return
            if not os.path.isdir(directory):
                print(f"❌ Selected path is not a valid directory: '{directory}'")
                return
            print(f"⏳ Loading data from '{directory}' …")
            try:
                data_manager.load_data(directory)
            except FileNotFoundError:
                print(f"❌ Directory not found: '{directory}'")
                return
            except (ValueError, OSError) as exc:
                print(f"❌ Error loading data: {exc}")
                return

            plot_widgets.update_filter_options()
            plot_widgets.btn_plot.disabled = False
            plot_widgets.btn_baseline.disabled = False

            info = data_manager.get_data_info()
            print(
                f"✅ Loaded {info['rows']} rows.\n"
                f"   IDs:         {info['unique_ids']}\n"
                f"   Frequencies: {info['frequencies']}\n"
                f"   Groups:      {info['groups']}"
            )

    def _on_plot(_btn) -> None:
        with output:
            output.clear_output(wait=True)
            if data_manager.data is None:
                print("⚠️  No data loaded. Click 'Load Data' first.")
                return

            params = plot_widgets.get_plot_params()
            use_sub = params.get("use_baseline_subtracted", False)

            if use_sub and data_manager.data_baseline_subtracted is None:
                print(
                    "⚠️  Baseline-subtracted data not available.\n"
                    "   Click 'Subtract Baseline' first."
                )
                return

            df = (
                data_manager.data_baseline_subtracted
                if use_sub
                else data_manager.data
            )

            print("⏳ Generating plot …")
            try:
                fig_width = params.pop("fig_width", 1000)
                fig_height = params.pop("fig_height", 600)
                fig = plot_magnetic_vs_time(df, **params)
                fig.update_layout(width=fig_width, height=fig_height)
                fig.show()
            except (ValueError, KeyError, TypeError) as exc:
                print(f"❌ Error generating plot: {exc}")

    def _on_baseline(_btn) -> None:
        with output:
            output.clear_output(wait=True)
            if data_manager.data is None:
                print("⚠️  No data loaded. Click 'Load Data' first.")
                return
            print("⏳ Subtracting baseline …")
            try:
                data_manager.subtract_baseline()
                print("✅ Baseline subtracted successfully.")
            except (ValueError, KeyError) as exc:
                print(f"❌ Error during baseline subtraction: {exc}")

    def _on_reset(_btn) -> None:
        plot_widgets.reset_to_defaults()

    plot_widgets.btn_load.on_click(_on_load)
    plot_widgets.btn_plot.on_click(_on_plot)
    plot_widgets.btn_baseline.on_click(_on_baseline)
    plot_widgets.btn_reset.on_click(_on_reset)

    return data_manager, plot_widgets, output
