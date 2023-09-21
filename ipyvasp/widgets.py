__all__ = [
    "load_results",
    "summarize",
    "FilesWidget",
    "PropsPicker",
    "BandsWidget",
    "KpathWidget",
]


import inspect
from time import time
from pathlib import Path
from collections.abc import Iterable
from functools import partial

# Widgets Imports
from IPython.display import display
import ipywidgets as ipw
from ipywidgets import (
    Layout,
    Button,
    Box,
    HBox,
    VBox,
    Dropdown,
    Text,
    Checkbox,
    Stack,
    SelectMultiple,
)

# More imports
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Internal imports
from . import utils as gu
from . import lattice as lat
from .core import serializer, parser as vp, plot_toolkit as ptk
from .utils import _sig_kwargs, _sub_doc
from ._enplots import _fmt_labels


def summarize(files, func, **kwargs):
    """
    Apply given func to each file in files and return a dataframe of the results.

    Parameters
    ----------
    files: Iterable, must be an iterable of PathLike objects, a dictionary of {name: PathLike} pairs also works and name appears in the dataframe.
    func: callable with a single arguemnt path.  Must return a dictionary.

    kwargs: passed to func itself.
    """
    if not callable(func):
        raise TypeError("Argument `func` must be a function.")

    if not isinstance(files, Iterable):
        raise TypeError("Argument `files` must be an iterable of PathLike objects.")

    if not isinstance(files, dict):
        files = {str(path): path for path in files}  # make a dictionary of paths

    outputs = []
    for name, path in files.items():
        output = func(path, **kwargs)
        if not isinstance(output, dict):
            raise TypeError("Function must return a dictionary.")

        if "FILE" in output:
            raise KeyError(
                "FILE is a reserved key to store the file name for reference."
            )

        outputs.append(
            {**output, "FILE": name}
        )  # add the file name to the output at the end

    unique_keys = (
        []
    )  # get all unique keys, there would be missing or extra keys, handle all
    for key in [key for out in outputs for key in out.keys()]:
        if key not in unique_keys:
            unique_keys.append(key)

    return pd.DataFrame(
        {ukey: [out.get(ukey, None) for out in outputs] for ukey in unique_keys}
    )


_progress_svg = """<svg xmlns="http://www.w3.org/2000/svg" height="5em" viewBox="0 0 50 50">
    <path fill="skyblue" d="M25,5A20.14,20.14,0,0,1,45,22.88a2.51,2.51,0,0,0,2.49,2.26h0A2.52,2.52,0,0,0,50,22.33a25.14,25.14,0,0,0-50,0,2.52,2.52,0,0,0,2.5,2.81h0A2.51,2.51,0,0,0,5,22.88,20.14,20.14,0,0,1,25,5Z">
        <animateTransform attributeName="transform" type="rotate" from="0 25 25" to="360 25 25" dur="0.2s" repeatCount="indefinite"/>
    </path>
</svg>"""


def fix_signature(cls):
    # VBox ruins signature of subclass, let's fix it
    cls.__signature__ = inspect.signature(cls.__init__)
    return cls


@fix_signature
class FilesWidget(VBox):
    """A widget for selecting files from a directory and its subdirectories.

    Parameters
    ----------
    path : str, default is '.'. The path to the directory to search.
    glob : str, default is '*'. The glob pattern to match files against. See https://docs.python.org/3/library/glob.html
    exclude : str, default is None. A regex pattern to exclude files from the selection.
    on_file_changed : callable, default is None.
        A function that takes path as an argument and is called when the selected file changes. Output is displayed below the widget.
        To add extra controls and widgets, use `.interactive`/`.interact` methods instead.

    Returns
    -------
    A FilesWidget object where you can filter files by typing in the text box, and select files from the dropdown.
    """

    def __init__(
        self,
        path: str = ".",
        glob: str = "*",
        exclude: str = None,
        on_file_changed=None,
    ) -> None:
        for prop in (path, glob, exclude):
            if prop and not isinstance(prop, str):
                raise ValueError(f"Expected string, got {type(prop)}")

        super().__init__(_dom_classes=["FilesWidget"])  # This makes it truely a widget
        self._files = []  # Selections stored as Path objects
        self._widgets = {
            "input": Text(
                value=path,
                description="Path:",
                tooltip="The path to the directory to search.",
            ),
            "glob": Text(
                value=glob,
                description="Glob:",
                tooltip="The glob pattern to match files against. See https://docs.python.org/3/library/glob.html",
            ),
            "exclude": Text(
                value=exclude or "",
                description="Exclude:",
                tooltip="A regex pattern to exclude files from the selection.",
            ),
            "lock": Checkbox(
                value=False,
                description="Lock selection",
                tooltip="Lock the current selection and prevent changes.",
            ),
            "files": Dropdown(
                options=[], description="File:", tooltip="Select a file from the list."
            ),
        }
        self.children = [
            self._widgets["input"],
            self._widgets["glob"],
            self._widgets["exclude"],
        ]
        self._widgets["lock"].observe(self._lock_selection, names=["value"])

        for key, value in self._widgets.items():
            if key not in ["files", "lock"]:
                value.on_submit(self._process)

        if on_file_changed:
            if not callable(on_file_changed):
                raise TypeError(
                    "Argument `on_file_changed` must be a function that takes path as an argument."
                )

            out = ipw.Output()
            self.layout.max_height = "90vh"  # Only if output is present
            self._widgets["output"] = out

            @out.capture(clear_output=True, wait=True)
            def on_change(change):
                on_file_changed(self.selected)

            self._widgets["files"].observe(on_change, names=["value"])

        self._process(None)  # Initial processing based on given values

    def _lock_selection(self, change):
        for key, value in self._widgets.items():
            if key not in ["files", "lock"]:
                value.disabled = self._widgets[
                    "lock"
                ].value  # Don't allow changes even programatically

        if self._widgets["lock"].value:
            self._widgets["lock"].description = f"{len(self._files)} files selected"
            self.children = [self._widgets["lock"], self._widgets["files"]]
        else:
            self.children = [
                self._widgets["input"],
                self._widgets["glob"],
                self._widgets["exclude"],
            ]

        if self._widgets.get("output"):
            self.children = [*self.children, self._widgets["output"]]

    def _process(self, change):
        self._widgets["lock"].description = "Processing..."
        files = gu.list_files(
            self._widgets["input"].value,
            glob=self._widgets["glob"].value,
            exclude=self._widgets["exclude"].value,
        )

        self._widgets["files"].options = [
            str(p) for p in files
        ]  # shows only relative path
        self._files = [path.absolute() for path in files]  # Store as full path
        self.children = list(self._widgets.values())  #  show all widgets
        self._widgets[
            "lock"
        ].description = f"{len(self._files)} files found. Lock selection?"

    @property
    def paths(self):
        "Returns all availble paths."
        return tuple(self._files)

    @property
    def dropdown(self):
        "Returns the dropdown widget to select files."
        return self._widgets["files"]

    @property
    def selected(self) -> Path:
        "Returns selected item in the dropdown as Path object."
        if self._widgets["files"].value:  # if not empty, otherwise it throws error
            return Path(self._widgets["files"].value).absolute()  # return full path

    @property
    def path(self):  # This is in consistent with other widgets too
        "Return currently selected path."
        return self.selected

    def interactive(
        self,
        func,
        other_widgets=None,
        other_controls=None,
        options={"manual": False},
        height="90vh",
        **kwargs,
    ):
        """
        Interact with a function that takes selected Path as first argument. Returns a widget that saves attributes of the function call such as .f, .args, .kwargs.
        See docs of self.interact for more details on the parameters. kwargs are passed to ipywidgets.interactive to create controls.


        >>> fw = FilesWidget()
        >>> out = fw.interactive(lambda path: print(path.read_text())) # prints contents of selected file on output widget
        >>> out.f # function
        >>> out.args # arguments
        >>> out.kwargs # keyword arguments
        >>> out.result # result of function call which is same as out.f(*out.args, **out.kwargs)
        >>> out.files_widget # reference to FilesWidget created, not the same as fw because it is a new instance


        .. note::
            If you don't need to interpret the result of the function call, you can use the @self.interact decorator instead.

        .. note::
            Each time an underlying new FilesWidget instance is created which picks input from previous one but stays separate. You can access it with `.files_widget` attribute of interactive.
        """
        # Make new FilesWidget with same parameters, to allow multiple interact calls
        new_fw = self.__class__(
            path=self._widgets["input"].value,
            glob=self._widgets["glob"].value,
            exclude=self._widgets["exclude"].value,
        )
        info = ipw.HTML().add_class("FW-Progess")

        def interact_func(fname, **kwargs):
            if fname:  # This would be None if no file is selected
                info.value = _progress_svg
                try:
                    start = time()
                    print(
                        f"Running {func.__name__}({fname!r}, {kwargs})"
                    )  # it also serves as removing the output errors
                    func(
                        Path(fname).absolute(), **kwargs
                    )  # Have Path object absolue if user changes directory
                    print(f"Finished in {time() - start:.3f} seconds.")
                finally:
                    info.value = ""

        out = ipw.interactive(
            interact_func, options, fname=new_fw._widgets["files"], **kwargs
        )

        out.files_widget = new_fw  # save reference to FilesWidget
        out.output_widget = out.children[
            -1
        ]  # save reference to output widget for other widgets to use

        if options.get("manual", False):
            out.interact_button = out.children[
                -2
            ]  # save reference to interact button for other widgets to use

        output = out.children[-1]  # get output widget
        output.clear_output(
            wait=True
        )  # clear output by waiting to avoid flickering, this is important
        output.layout = Layout(
            overflow="auto", max_height="100%", width="100%"
        )  # make output scrollable and avoid overflow

        others = out.children[1:-1]  # exclude files_dd and Output widget
        _style = """<style>
        .FilesWidget-Interact {
            --jp-widgets-inline-label-width: 4em;
            --jp-widgets-inline-width: 18em;
            --jp-widgets-inline-width-short: 9em;
        }
        .FilesWidget-Interact {max-height:90vh;width:100%;}
        .FilesWidget-Interact > div {overflow:auto;max-height:100%;padding:8px;}
        .FilesWidget-Interact > div:first-child {width:20em}
        .FilesWidget-Interact > div:last-child {width:calc(100% - 20em)}
        .FilesWidget-Interact .FW-Progess {position:absolute !important; left:50%; top:50%; transform:translate(-50%,-50%); z-index:1}
        </style>"""
        if others:
            others = [ipw.HTML(f"<hr/>{_style}"), *others]
        else:
            others = [ipw.HTML(_style)]

        if other_controls and not isinstance(other_controls, (list, tuple)):
            raise TypeError("other_controls must be a list or tuple of widgets.")

        if other_widgets and not isinstance(other_widgets, (list, tuple)):
            raise TypeError("other_widgets must be a list or tuple of widgets.")

        if other_widgets:
            output.layout.max_height = "200px"
            out_collapser = Checkbox(description="Hide output widget", value=False)

            def toggle_output(change):
                if out_collapser.value:
                    output.layout.height = "0px"  # dont use display = 'none' as it will clear widgets and wont show again
                else:
                    output.layout.height = "auto"

            out_collapser.observe(toggle_output, "value")
            others.append(out_collapser)

        # This should be below output collapser
        others = [
            *others,
            ipw.HTML(f"<hr/>"),
            *(other_controls or []),
        ]  # add hr to separate other controls

        out.children = [
            HBox(
                [  # reset children to include new widgets
                    VBox(
                        children=[new_fw, VBox(others)]
                    ),  # other widgets in box to make scrollable independent file selection
                    VBox(
                        children=[Box([output]), *(other_widgets or []), info]
                    ),  # output in box to make scrollable,
                ],
                layout=Layout(height=height, max_height=height),
            ).add_class("FilesWidget-Interact")
        ]  # important for every widget separately
        return out

    def interact(
        self,
        other_widgets=None,
        other_controls=None,
        options={"manual": False},
        height="90vh",
        **kwargs,
    ):
        """Interact with a function that takes a selected Path as first argument.
        A CSS class 'FilesWidget-Interact' is added to the final widget to let you style it.

        Parameters
        ----------
        other_widgets : list/tuple
            Any displayable widget can be passed. These are placed below the output widget of interact.
            For example you can add plotly's FigureWidget that updates based on the selection, but is not part of the function, so it is displayed only once.
        other_controls : list/tuple
            Default is None. If not None, these are assumed to be ipywidgets and are placed below the widgets created by kwargs. These are not passed to the decorated function.
        options : dict
            Default is {'manua':False}. If True, the decorated function is not called automatically, and you have to call it manually on button press. You can pass button name as 'manual_name' in options.
        height : str
            Default is '90vh'. height of the final widget. This is important to avoid very long widgets.


        kwargs are passed to ipywidgets.interactive and decorated function. Resulting widgets are placed below the file selection widget.

        `other_widgets` can be controlled by `other_controls` externally. For example, you can add a button to update a plotly's FigureWidget.

        The decorated function can be called later separately as well, and has .args and .kwargs attributes to access the latest arguments
        and .result method to access latest. For a function `f`, `f.result` is same as `f(*f.args, **f.kwargs)`.


        >>> fw = FilesWidget()
        >>> @fw.interact(x = False)
        >>> def f(path,x):
        >>>     print('path:',path)
        >>>     print('Path Type: ', type(path))
        >>>     print('x: ',x)


        .. note::
            Use self.interactive to get a widget that stores the argements and can be called later in a notebook cell.
        """

        def inner(func):
            display(
                self.interactive(
                    func,
                    other_widgets=other_widgets,
                    other_controls=other_controls,
                    options=options,
                    height=height,
                    **kwargs,
                )
            )
            return func

        return inner

    def summarize(self, func, **kwargs):
        """Summarize the results from all selected files using a function that takes a Path object as first arguement.
        kwargs are passed to function. Returns a dataframe."""
        return summarize(
            {
                key: value
                for key, value in zip(self._widgets["files"].options, self._files)
            },
            func,
            **kwargs,
        )


@fix_signature
class _PropPicker(VBox):
    def __init__(self, system_summary=None):
        super().__init__()
        self._widgets = {
            "atoms": Dropdown(description="Atoms"),
            "orbs": Dropdown(description="Orbs"),
        }
        self._html = ipw.HTML()  # to observe

        def observe_change(change):
            self._html.value = change.new  # is a string

        self._widgets["atoms"].observe(observe_change, "value")
        self._widgets["orbs"].observe(observe_change, "value")

        self._atoms = {}
        self._orbs = {}
        self._process(system_summary)

    def _process(self, system_summary):
        if not hasattr(system_summary, "orbs"):
            self.children = [
                ipw.HTML(f"❌ No projection data found from given summary!")
            ]
            return None

        self.children = [self._widgets["atoms"], self._widgets["orbs"]]
        sorbs = system_summary.orbs

        orbs = {"-": [], "All": range(len(sorbs)), "s": [0]}
        if set(["px", "py", "pz"]).issubset(sorbs):
            orbs["p"] = range(1, 4)
            orbs["px+py"] = [
                idx for idx, key in enumerate(sorbs) if key in ("px", "py")
            ]
            orbs.update({k: [v] for k, v in zip(sorbs[1:], range(1, 4))})
        if set(["dxy", "dyz"]).issubset(sorbs):
            orbs["d"] = range(4, 9)
            orbs.update({k: [v] for k, v in zip(sorbs[4:], range(4, 9))})
        if len(sorbs) == 16:
            orbs["f"] = range(9, 16)
            orbs.update({k: [v] for k, v in zip(sorbs[9:], range(9, 16))})
        if len(sorbs) > 16:  # What the hell here
            orbs.update({k: [idx] for idx, k in enumerate(sorbs[16:], start=16)})

        self._orbs = orbs
        old_orb = self._widgets["orbs"].value
        self._widgets["orbs"].options = list(orbs.keys())
        if old_orb in self._widgets["orbs"].options:
            self._widgets["orbs"].value = old_orb

        atoms = {"-": [], "All": range(system_summary.NIONS)}
        for key, tp in system_summary.types.to_dict().items():
            atoms[key] = tp
            for n, v in enumerate(tp, start=1):
                atoms[f"{key}{n}"] = [v]

        self._atoms = atoms
        old_atom = self._widgets["atoms"].value
        self._widgets["atoms"].options = list(atoms.keys())
        if old_atom in self._widgets["atoms"].options:
            self._widgets["atoms"].value = old_atom

    def update(self, system_summary):
        return self._process(system_summary)

    @property
    def props(self):
        items = {k: w.value for k, w in self._widgets.items()}
        items["atoms"] = self._atoms.get(items["atoms"], [])
        items["orbs"] = self._orbs.get(items["orbs"], [])
        items[
            "label"
        ] = f"{self._widgets['atoms'].value or ''}-{self._widgets['orbs'].value or ''}"
        return items


@fix_signature
class PropsPicker(VBox):
    """
    A widget to pick atoms and orbitals for plotting.

    Parameters
    ----------
    system_summary : (Vasprun,Vaspout).summary
    N : int, default is 3, number of projections to pick.
    on_button_click : callable, takes button as arguemnet. Default is None, a function to call when button is clicked.
    on_selection_changed : callable, takes change as argument. Default is None, a function to call when selection is changed.
    """

    def __init__(
        self, system_summary=None, N=3, on_button_click=None, on_selection_changed=None
    ):
        super().__init__()
        self._linked = Dropdown(
            options=[str(i + 1) for i in range(N)]
            if N != 3
            else ("Red", "Green", "Blue"),
            description="Projection" if N != 3 else "Color",
        )
        self._stacked = Stack(
            children=tuple(_PropPicker(system_summary) for _ in range(N)),
            selected_index=0,
        )
        self._button = Button(description="Run Function")

        if callable(on_button_click):
            self._button.on_click(on_button_click)

        for w in [self._button, self._linked]:
            w.layout.width = "max-content"

        ipw.link((self._linked, "index"), (self._stacked, "selected_index"))
        self.children = [HBox([self._linked, self._button]), self._stacked]

        if callable(on_selection_changed):
            for child in self._stacked.children:
                child._html.observe(on_selection_changed, names="value")

    def update(self, system_summary):
        for child in self._stacked.children:
            child.update(system_summary)

    @property
    def button(self):
        return self._button

    @property
    def projections(self):
        out = {}
        for child in self._stacked.children:
            props = child.props
            if props["atoms"] and props["orbs"]:  # discard empty
                out[props["label"]] = (props["atoms"], props["orbs"])

        return out


def __store_figclick_data(fig, store_dict, callback=None, selection=False):
    "Store clicked data in a dict. callback takes trace as argument and is called after storing data."
    if not isinstance(fig, go.FigureWidget):
        raise TypeError("fig must be a FigureWidget")
    if not isinstance(store_dict, dict):
        raise TypeError("store_dict must be a dict")
    if callback and not callable(callback):
        raise TypeError("callback must be callable if given")

    def handle_click(trace, points, state):
        store_dict["data"] = points
        if callback:
            callback(trace)

    for trace in fig.data:
        if selection:
            trace.on_selection(handle_click)
        else:
            trace.on_click(handle_click)


def store_clicked_data(fig, store_dict, callback=None):
    "Store clicked point data to a store_dict. callback takes trace being clicked as argument."
    return __store_figclick_data(fig, store_dict, callback, selection=False)


def store_selected_data(fig, store_dict, callback=None):
    "Store multipoints selected data to a store_dict. callback takes trace being clicked as argument."
    return __store_figclick_data(fig, store_dict, callback, selection=True)


def load_results(paths_list):
    "Loads result.json from paths_list and returns a dataframe."
    paths_list = [Path(p) for p in paths_list]
    result_paths = []
    if paths_list:
        for path in paths_list:
            if path and path.is_dir():
                result_paths.append(path / "result.json")
            elif path and path.is_file():
                result_paths.append(path.parent / "result.json")

    def load_data(path):
        try:
            return serializer.load(str(path.absolute()))
        except:
            return {}  # If not found, return empty dictionary

    return summarize(result_paths, load_data)


@fix_signature
class BandsWidget(VBox):
    """Visualize band structure from VASP calculation. You can click on the graph to get the data such as VBM, CBM, etc.
    Two attributes are important:
    self.clicked_data returns the last clicked point, that can also be stored as VBM, CBM etc, using Click dropdown.
    self.selected_data returns the last selection of points within a box or lasso. You can plot that output separately as plt.plot(data.xs, data.ys) after a selection.
    """

    def __init__(self, use_vaspout=False, height="90vh", **file_widget_kwargs):
        super().__init__(_dom_classes=["BandsWidget"])
        self._bands = None
        self._use_vaspout = use_vaspout
        self._fig = go.FigureWidget()
        self._tsd = Dropdown(
            description="Style", options=["plotly_white", "plotly_dark"]
        )
        self._click = Dropdown(description="Click", options=["None", "VBM", "CBM"])
        self._ktcicks = Text(description="kticks")
        self._elim = Text(description="elim", value="-10, 10")
        self._ppicks = PropsPicker(
            on_button_click=self._update_graph, on_selection_changed=self._warn_update
        )
        self._ppicks.button.description = "Update Graph"
        self._result = {}  # store and save output results
        self._click_dict = {}  # store clicked data
        self._select_dict = {}  # store selection data
        self._kwargs = {}
        file_widget_kwargs = {
            "glob": "vapout.h5" if use_vaspout else "vasprun.xml",
            **file_widget_kwargs,
        }
        self._interact = FilesWidget(**file_widget_kwargs).interactive(
            self._load_data,
            other_widgets=[self._fig],
            other_controls=[
                self._tsd,
                self._elim,
                self._ktcicks,
                ipw.HTML("<hr/>"),
                self._ppicks,
                ipw.HTML("<hr/>Click on graph to read selected option."),
                self._click,
            ],
            height=height,
        )
        self.files_widget = self._interact.files_widget
        self.children = self._interact.children
        self._tsd.observe(self._change_theme, "value")
        self._click.observe(self._click_save_data, "value")
        self._ktcicks.observe(self._warn_update, "value")
        self._elim.observe(self._warn_update, "value")

    @property
    def path(self):
        "Returns currently selected path."
        return self.files_widget.path

    def _load_data(self, path):  # Automatically redirectes to output widget
        self._interact.output_widget.clear_output(wait=True)  # Why need again?
        with self._interact.output_widget:
            self._bands = (
                vp.Vaspout(path) if self._use_vaspout else vp.Vasprun(path)
            ).bands
            self._ppicks.update(self.bands.source.summary)
            self._ktcicks.value = ", ".join(
                f"{k}:{v}" for k, v in self.bands.get_kticks()
            )
            if self.bands.source.summary.LSORBIT:
                self._click.options = ["None", "VBM", "CBM", "so_max", "so_min"]
            else:
                self._click.options = ["None", "VBM", "CBM"]

            if (file := self.files_widget.selected.parent / "result.json").is_file():
                self._result = serializer.load(str(file.absolute()))  # Old data loaded

            pdata = self.bands.source.poscar.data
            self._result.update(
                {
                    "v": round(pdata.volume, 4),
                    **{k: round(v, 4) for k, v in zip("abc", pdata.norms)},
                    **{k: round(v, 4) for k, v in zip("αβγ", pdata.angles)},
                }
            )
            self._click_save_data(None)  # Load into view
            self._warn_update(None)

    @property
    def bands(self):
        "Bands class initialized"
        if not self._bands:
            raise ValueError("No data loaded by BandsWidget yet!")
        return self._bands

    @property
    def kwargs(self):
        "Selected kwargs from GUI"
        return self._kwargs

    @property
    def clicked_data(self):
        "Clicked data from graph"
        return self._click_dict.get("data", None)

    @property
    def selected_data(self):
        "Data selected by box or lasso selection from graph"
        return self._select_dict.get("data", None)

    def _update_graph(self, btn):
        self._interact.output_widget.clear_output(wait=True)  # Why need again?
        with self._interact.output_widget:
            hsk = [
                [v.strip() for v in vs.split(":")]
                for vs in self._ktcicks.value.split(",")
            ]
            kticks = [(int(vs[0]), vs[1]) for vs in hsk if len(vs) == 2] or None
            elim = [float(v) for v in self._elim.value.split(",") if v.strip()] or None
            self._kwargs = {"elim": elim, "kticks": kticks}

            if self._ppicks.projections:
                self._kwargs = {"projections": self._ppicks.projections, **self._kwargs}
                fig = self.bands.iplot_rgb_lines(**self._kwargs, name="Up")
                if self.bands.source.summary.ISPIN == 2:
                    self.bands.iplot_rgb_lines(**self._kwargs, spin=1, name="Down", fig=fig)

                self.iplot = partial(self.bands.iplot_rgb_lines, **self._kwargs)
                self.splot = partial(self.bands.splot_rgb_lines, **self._kwargs)
            else:
                fig = self.bands.iplot_bands(**self._kwargs, name="Up")
                if self.bands.source.summary.ISPIN == 2:
                    self.bands.iplot_bands(**self._kwargs, spin=1, name="Down", fig=fig)

                self.iplot = partial(self.bands.iplot_bands, **self._kwargs)
                self.splot = partial(self.bands.splot_bands, **self._kwargs)

            ptk.iplot2widget(fig, self._fig, template=self._tsd.value)
            self._click_dict.clear()  # avoid data from previous figure
            self._select_dict.clear()  # avoid data from previous figure
            store_clicked_data(
                self._fig,
                self._click_dict,
                callback=lambda trace: self._click_save_data("CLICK"),
            )  # 'CLICK' is needed to inntercept in a function
            store_selected_data(self._fig, self._select_dict, callback=None)
            self._ppicks.button.description = "Update Graph"

    def _change_theme(self, change):
        self._fig.layout.template = self._tsd.value

    def _click_save_data(self, change=None):
        def _show_and_save(data_dict):
            self._interact.output_widget.clear_output(wait=True)  # Why need again?
            with self._interact.output_widget:
                print(
                    ", ".join(
                        f"{key} = {value}"
                        for key, value in data_dict.items()
                        if key not in ("so_max", "so_min")
                    )
                )

            serializer.dump(
                data_dict,
                format="json",
                outfile=self.files_widget.selected.parent / "result.json",
            )

        if (
            change is None
        ):  # called from other functions but not from store_clicked_data
            return _show_and_save(self._result)
        # Should be after checking chnage
        if self._click.value and self._click.value == "None":
            return  # No need to act on None

        data_dict = self._result.copy()  # Copy old data

        if data := self.clicked_data:  # No need to make empty dict
            x = round(data.xs[0], 6)
            y = round(float(data.ys[0]) + self.bands.data.ezero, 6)  # Add ezero

            if key := self._click.value:
                data_dict[key] = y  # Assign value back
                if not key.startswith("so"):
                    data_dict[key + "_k"] = round(
                        x, 6
                    )  # Save x to test direct/indirect

            if data_dict.get("VBM", None) and data_dict.get("CBM", None):
                data_dict["E_gap"] = np.round(data_dict["CBM"] - data_dict["VBM"], 6)

            if data_dict.get("so_max", None) and data_dict.get("so_min", None):
                data_dict["Δ_SO"] = np.round(
                    data_dict["so_max"] - data_dict["so_min"], 6
                )

            self._result.update(data_dict)  # store new data
            _show_and_save(self._result)

        if change == "CLICK":  # Called from store_clicked_data
            self._click.value = "None"  # Reset to None to avoid accidental click

    def _warn_update(self, change):
        self._ppicks.button.description = "🔴 Update Graph"

    @property
    def results(self):
        "Generate a data frame form result.json file in each folder."
        return load_results(self.files_widget.paths)


@fix_signature
class KpathWidget(VBox):
    """
    Interactively bulid a kpath for bandstructure calculation.

    After initialization and disply:

    - Select a POSCAR file from "File:" dropdown menu. It will update the figure.
    - Add points to select box on left by clicking on plot points. When done with points click on Lock to avoid adding more points.
    - To update point(s), select point(s) from the select box and click on a scatter point in figure or use KPOINT input to update it manually, e.g. if a point is not available on plot.
    - Add labels to the points by typing in the "Labels" box such as "Γ,X" or "Γ 5,X" that will add 5 points in interval.
    - To break the path between two points "Γ" and "X" type "Γ 0,X" in the "Labels" box, zero means no points in interval.
    """

    def __init__(self, height="90vh", **files_widget_kwargs):
        super().__init__(_dom_classes=["KpathWidget"])
        self._fig = go.FigureWidget()
        self._sm = SelectMultiple(options=[], layout=Layout(width="auto"))
        self._lab = Text(description="Labels", continuous_update=True)
        self._kpt = Text(description="KPOINT", continuous_update=False)
        self._add = Button(description="Lock", tooltip="Lock/Unlock adding more points")
        self._del = Button(description="❌ Point", tooltip="Delete Selected Points")
        self._tsb = Button(description="Dark Plot", tooltip="Toggle Plot Theme")
        self._poscar = None
        self._clicktime = None
        self._kpoints = {}

        other_controls = [
            HBox([self._add, self._del, self._tsb], layout=Layout(min_height="24px")),
            ipw.HTML(
                "<style>.KpathWidget .widget-select-multiple { min-height: 180px; }\n .widget-select-multiple > select {height: 100%;}</style>"
            ),
            self._sm,
            self._lab,
            self._kpt,
        ]
        files_widget_kwargs = {"glob": "POSCAR", **files_widget_kwargs}
        self._interact = FilesWidget(**files_widget_kwargs).interactive(
            self._update_fig, [self._fig], other_controls, height=height
        )
        self.files_widget = self._interact.files_widget  # sometimes useful
        self.children = self._interact.children

        self._tsb.on_click(self._update_theme)
        self._add.on_click(self._toggle_lock)
        self._del.on_click(self._del_point)
        self._kpt.observe(self._take_kpt, "value")
        self._lab.observe(self._add_label)

    @property
    def path(self):
        "Returns currently selected path."
        return self._interact.files_widget.path

    @property
    def poscar(self):
        "POSCAR class associated to current selection."
        return self._poscar

    def _update_fig(self, path):
        from .misc import POSCAR  # to avoid circular import

        with self._interact.output_widget:
            template = (
                "plotly_dark" if "Light" in self._tsb.description else "plotly_white"
            )
            self._poscar = POSCAR(path)
            ptk.iplot2widget(
                self._poscar.iplot_bz(fill=False, color="red"), self._fig, template
            )
            with self._fig.batch_animate():
                self._fig.add_trace(
                    go.Scatter3d(
                        x=[],
                        y=[],
                        z=[],
                        mode="lines+text",
                        name="path",
                        text=[],
                        hoverinfo="none",  # dont let it block other points
                        textfont_size=18,
                    )
                )  # add path that will be updated later
            self._click()  # handle events
            print("Click points on plot to store for kpath.")

    def _click(self):
        def handle_click(trace, points, state):
            if self._clicktime and (time() - self._clicktime < 1):
                return  # Avoid double clicks

            self._clicktime = time()  # register this click's time

            if points.ys != []:
                index = points.point_inds[0]
                kp = trace.hovertext[index]
                kp = [float(k) for k in kp.split("[")[1].split("]")[0].split()]

                if self._sm.value:
                    self._take_kpt(kp)  # this updates plot back as well
                elif self._add.description == "Lock":  # only add when open
                    self._add_point(kp)

        for trace in self._fig.data:
            if "HSK" in trace.name:
                trace.on_click(handle_click)

    def _update_selection(self):
        with self._interact.output_widget:
            coords, labels = self.get_coords_labels()
            with self._fig.batch_animate():
                for trace in self._fig.data:
                    if "path" in trace.name and coords.any():
                        trace.x = coords[:, 0]
                        trace.y = coords[:, 1]
                        trace.z = coords[:, 2]
                        trace.text = _fmt_labels(
                            labels
                        )  # convert latex to html equivalent

    def get_coords_labels(self):
        "Returns tuple of (coordinates, labels) to directly plot."
        with self._interact.output_widget:
            points = self.get_kpoints()

            coords = (
                self.poscar.bz.to_cartesian([p[:3] for p in points]).tolist()
                if points
                else []
            )
            labels = [
                p[3] if (len(p) >= 4 and isinstance(p[3], str)) else "" for p in points
            ]
            numbers = [
                p[4]
                if len(p) == 5
                else p[3]
                if (len(p) == 4 and isinstance(p[3], int))
                else ""
                for p in points
            ]

            j = 0
            for i, n in enumerate(numbers, start=1):
                if isinstance(n, int) and n == 0:
                    labels.insert(i + j, "")
                    coords.insert(i + j, [np.nan, np.nan, np.nan])
                    j += 1

            return np.array(coords), labels

    def get_kpoints(self):
        "Returns kpoints list including labels and numbers in intervals if given."
        keys = [
            idx for (_, idx) in self._sm.options if idx in self._kpoints
        ]  # order and existence is important
        kpts = [self._kpoints[k] for k in keys]
        LN = [
            lab.split("⋮")[1].strip().split()
            for (lab, idx) in self._sm.options
            if idx in keys
        ]

        for idx, ln in enumerate(LN):
            if len(ln) == 2:
                kpts[idx] = tuple([*kpts[idx], ln[0], int(ln[1])])  # label, number
            elif len(ln) == 1:
                try:
                    kpts[idx] = tuple([*kpts[idx], int(ln[0])])  # number
                except:
                    kpts[idx] = tuple([*kpts[idx], ln[0]])  # label
            elif len(ln) == 0:
                kpts[idx] = tuple(kpts[idx])
            else:
                raise ValueError(
                    "Label and number should be separated by space or only one of them should be present"
                )
        return kpts

    def _update_theme(self, btn):
        if "Dark" in btn.description:
            self._fig.layout.template = "plotly_dark"
            btn.description = "Light Plot"
        else:
            self._fig.layout.template = "plotly_white"
            btn.description = "Dark Plot"

    def _add_point(self, kpt):
        with self._interact.output_widget:
            self._sm.options = [*self._sm.options, ("⋮", len(self._sm.options))]
            self._sm.value = (
                self._sm.options[-1][1],
            )  # select to receive point as well
            self._take_kpt(kpt)  # add point, label and plot back

    def _toggle_lock(self, btn):
        if self._add.description == "Lock":
            self._add.description = "Unlock"
        else:
            self._add.description = "Lock"

    def _del_point(self, btn):
        with self._interact.output_widget:
            for (
                v
            ) in (
                self._sm.value
            ):  # for loop here is important to update selection properly
                self._sm.options = [opt for opt in self._sm.options if opt[1] != v]
                self._update_selection()  # update plot as well

    def _take_kpt(self, change_or_kpt):
        with self._interact.output_widget:
            if isinstance(change_or_kpt, (list, tuple)):
                point = change_or_kpt
            else:
                point = [float(v) for v in self._kpt.value.split(",")]

            if len(point) != 3:
                raise ValueError("Expects KPOINT of 3 floats")

            self._kpoints.update({v: point for v in self._sm.value})
            label = "{:>8.4f} {:>8.4f} {:>8.4f}".format(*point)
            self._sm.options = [
                (label, value) if value in self._sm.value else (lab, value)
                for (lab, value) in self._sm.options
            ]
            self._add_label(None)  # Re-adjust labels and update plot as well

    def _add_label(self, change):
        with self._interact.output_widget:
            labs = [" ⋮ " for _ in self._sm.options]  # as much as options
            for idx, (_, lab) in enumerate(
                zip(self._sm.options, self._lab.value.split(","))
            ):
                labs[idx] = labs[idx] + lab  # don't leave empty anyhow

            self._sm.options = [
                (v.split("⋮")[0].strip() + lab, idx)
                for (v, idx), lab in zip(self._sm.options, labs)
            ]

            self._update_selection()  # Update plot in both cases, by click or manual input

    @_sub_doc(lat.get_kpath, {"kpoints :.*n :": "n :", "rec_basis :.*\n\n": "\n\n"})
    @_sig_kwargs(lat.get_kpath, ("kpoints", "rec_basis"))
    def get_kpath(self, n=5, **kwargs):
        return self.poscar.get_kpath(self.get_kpoints(), n=n, **kwargs)

    def splot(self, plane=None, fmt_label=lambda x: x, plot_kws={}, **kwargs):
        """
        Same as `ipyvasp.lattice.POSCAR.splot_bz` except it also plots path on BZ.

        Parameters
        ----------
        plane : str of plane like 'xy' to plot in 2D or None to plot in 3D
        fmt_label : function, should take a string and return a string or (str, dict) of which dict is passed to `plt.text`.
        plot_kws : dict of keyword arguments for `plt.plot` for kpath.

        kwargs are passed to `ipyvasp.lattice.POSCAR.splot_bz`.
        """
        if not isinstance(plot_kws, dict):
            raise TypeError("plot_ks should be a dict")

        ax = self.poscar.splot_bz(plane=plane, **kwargs)
        kpoints = self.poscar.to_basis(coords, reciprocal=True)
        coords, labels = self.get_coords_labels()
        self.poscar.splot_kpath(
            kpoints, labels=labels, fmt_label=fmt_label, **plot_kws
        )  # plots on ax automatically
        return ax

    def iplot(self):
        "Returns disconnected current plotly figure"
        return go.Figure(data=self._fig.data, layout=self._fig.layout)


# Should be at end
del fix_signature  # no more need
