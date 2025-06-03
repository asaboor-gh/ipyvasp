__all__ = [
    "load_results",
    "summarize",
    "Files",
    "PropsPicker",
    "BandsWidget",
    "KPathWidget",
]


import inspect, re
from pathlib import Path
from collections.abc import Iterable
from functools import partial

# Widgets Imports
from IPython.display import display
from ipywidgets import (
    Layout,
    Button,
    HBox,
    VBox,
    Dropdown,
    Text,
    Stack,
    SelectMultiple,
    TagsInput,
)

# More imports
import numpy as np
import pandas as pd
import ipywidgets as ipw
import traitlets
import plotly.graph_objects as go
import einteract as ei 

# Internal imports
from . import utils as gu
from . import lattice as lat
from .core import serializer, parser as vp, plot_toolkit as ptk
from .utils import _sig_kwargs, _sub_doc, get_file_size
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

    if not isinstance(files, Iterable): # Files is instance of Iterable due to __iter__ method
        raise TypeError("Argument `files` must be an iterable of PathLike objects")

    if not isinstance(files, dict):
        files = {str(path): path for path in files}  # make a dictionary of paths

    outputs = []
    for name, path in files.items():
        output = func(path, **kwargs)
        if not isinstance(output, dict):
            raise TypeError("Function must return a dictionary to create DataFrame.")

        if "FILE" in output:
            raise KeyError(
                "FILE is a reserved key to store the file name for reference."
            )

        outputs.append(
            {**output, "FILE": name}
        )  # add the file name to the output at the end

    unique_keys = {} # handle missing keys with types
    for key,value in [item for out in outputs for item in out.items()]:
        unique_keys[key] = '' if isinstance(value, str) else None

    return pd.DataFrame(
        {key: [out.get(key, ph) for out in outputs] for key,ph in unique_keys.items()}
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
class Files:
    """Creates a Batch of files in a directory recursively based on glob pattern or given list of files.
    This is a boilerplate abstraction to do analysis in multiple calculations simultaneously.

    Parameters
    ----------
    path_or_files : str, current directory by default or list of files or an instance of Files.
    glob : str, glob pattern, '*' by default. '**' is used for recursive glob. Not used if files supplied above.
    exclude : str, regular expression pattern to exclude files.
    files_only : bool, if True, returns only files.
    dirs_only : bool, if True, returns only directories.

    Use methods on return such as `summarize`, `with_name`, `filtered`, `interact` and others.

    >>> Files(root_1, glob_1,...).add(root_2, glob_2,...) # Fully flexible to chain

    WARNING: Don't use write operations on paths in files in batch mode, it can cause unrecoverable data loss.
    """
    def __init__(self, path_or_files = '.', glob = '*', exclude = None,files_only = False, dirs_only=False):
        if isinstance(path_or_files, Files):
            self._files = path_or_files._files
            return # Do nothing
        
        if files_only and dirs_only:
            raise ValueError("files_only and dirs_only cannot be both True")

        files = []
        if isinstance(path_or_files,(str, Path)):
            files = Path(path_or_files).glob(glob)
        else:
            others = []
            for item in path_or_files:
                if isinstance(item, str):
                    item = Path(item)
                elif not isinstance(item, Path):
                    raise TypeError(f"Expected str or Path in sequence, got {type(item)}")
                
                if item.exists():
                    files.append(item)
                else:
                    others.append(str(item))
                    
            if others:
                print(f"Skipping paths that do not exist: {list(set(others))}")
                
        if exclude:
            files = (p for p in files if not re.search(exclude, str(p)))
        if files_only:
            files = (p for p in files if p.is_file())
        if dirs_only:
            files = (p for p in files if p.is_dir())
            
        self._files =  tuple(sorted(files))

    def __str__(self):
        return '\n'.join(str(f) for f in self._files)

    def __repr__(self):
        if not self: return "Files()"
        show = ',\n'.join(f'  {f!r}' for f in self._files)
        return f"Files(\n{show}\n) {len(self._files)} items"

    def __getitem__(self, index): return self._files[index]
    def __iter__(self): return self._files.__iter__()
    def __len__(self): return len(self._files)
    def __bool__(self): return bool(self._files)

    def __add__(self, other):
        raise NotImplementedError("Use self.add method instead!")

    def with_name(self, name):
        "Change name of all files. Only keeps existing files."
        return self.__class__([f.with_name(name) for f in self._files])

    def filtered(self, include=None, exclude=None, files_only = False, dirs_only=False):
        "Filter all files. Only keeps existing file."
        files = [p for p in self._files if re.search(include, str(p))] if include else self._files
        return self.__class__(files, exclude=exclude,dirs_only=dirs_only,files_only=files_only)

    def summarize(self, func, **kwargs):
        "Apply a func(path) -> dict and create a dataframe."
        return summarize(self._files,func, **kwargs)
    
    def load_results(self,exclude_keys=None):
        "Load result.json files from these paths into a dataframe, with optionally excluding keys."
        return load_results(self._files,exclude_keys=exclude_keys)
    
    def input_info(self, *tags):
        "Grab input information into a dataframe from POSCAR and INCAR. Provide INCAR tags (case-insinsitive) to select only few of them."
        from .lattice import POSCAR

        def info(path, tags):
            p = POSCAR(path).data 
            lines = [[v.strip() for v in line.split('=')] 
                     for line in path.with_name('INCAR').read_text().splitlines() 
                     if '=' in line]
            if tags:
                tags = [t.upper() for t in tags] # can send lowercase tag
                lines = [(k,v) for k,v in lines if k in tags]
            d = {k:v for k,v in lines if not k.startswith('#')}
            d.update({k:len(v) for k,v in p.types.items()})
            d.update(zip(['a','b','c','v','alpha','beta','gamma'], [*p.norms,p.volume,*p.angles]))
            return d
        
        return self.with_name('POSCAR').summarize(info, tags=tags)

    def update(self, path_or_files, glob = '*', cleanup = True, exclude=None,**kwargs):
        """Update files inplace with similar parameters as initialization. If `cleanup=False`, older files are kept too.
        Useful for widgets such as BandsWidget to preserve their state while using `widget.files.update`."""
        old = () if cleanup else self._files
        self._files = self._unique(old, self.__class__(path_or_files, glob = glob, exclude=exclude,**kwargs)._files)
        
        if (dd := getattr(self, '_dd', None)): # update dropdown
            old = dd.value
            dd.options = self._files
            if old in dd.options:
                dd.value = old
    
    def to_dropdown(self,description='File'):
        """
        Convert this instance to Dropdown. If there is only one file, adds an 
        empty option to make that file switchable. 
        Options of this dropdown are update on calling `Files.update` method."""
        if hasattr(self,'_dd'): 
            return self._dd # already created
        
        options = self._files if len(self._files) != 1 else ['', *self._files] # make single file work
        self._dd = Dropdown(description=description, options=options)
        return self._dd
    
    def add(self, path_or_files, glob = '*', exclude=None, **kwargs):
        """Add more files or with a diffrent glob on top of exitsing files. Returns same instance.
        Useful to add multiple globbed files into a single chained call.

        >>> Files(root_1, glob_1,...).add(root_2, glob_2,...) # Fully flexible 
        """
        self._files = self._unique(self._files, self.__class__(path_or_files, glob = glob, exclude=exclude,**kwargs)._files)
        return self
    
    def _unique(self, *files_tuples):
        return tuple(np.unique(np.hstack(files_tuples)))
    
    @_sub_doc(ei.interactive)
    def interactive(self, *funcs, auto_update=True, app_layout=None, grid_css={},**kwargs):
        if 'file' in kwargs:
            raise KeyError("file is a reserved keyword argument to select path to file!")
        
        has_file_param = False
        for func in funcs:
            if not callable(func):
                raise TypeError(f"Each item in *funcs should be callable, got {type(func)}")
            params = [k for k,v in inspect.signature(func).parameters.items()]
            for key in params:
                if key == 'file':
                    has_file_param = True 
                    break
        
        if funcs and not has_file_param: # may be no func yet, that is test below
            raise KeyError("At least one of funcs should take 'file' as parameter, none got it!")
        
        return ei.interactive(*funcs,auto_update=auto_update, app_layout = app_layout, grid_css=grid_css, file = self.to_dropdown(), **kwargs)
    
    @_sub_doc(ei.interact)
    def interact(self, *funcs, auto_update=True, app_layout=None, grid_css={},**kwargs):
        def inner(func):
            display(self.interactive(func, *funcs,
                auto_update=auto_update, app_layout = app_layout, grid_css=grid_css,
                **kwargs)
            )
            return func
        return inner
    
    def kpath_widget(self, height='400px'):
        "Get KPathWidget instance with these files."
        return KPathWidget(files = self.with_name('POSCAR'), height = height)

    def bands_widget(self, height='450px'):
        "Get BandsWidget instance with these files."
        return BandsWidget(files=self._files, height=height)
    
    def map(self,func, to_df=False):
        """Map files to a function that takes path as argument. 
        If `to_df=True`, func may return a dict to create named columns, or just two columns will be created.
        Otherwise returns generator of elemnets `(path, func(path))`.
        If you need to operate on opened file pointer, use `.mapf` instead.
        
        >>> import ipyvasp as ipv
        >>> files = ipv.Files(...)
        >>> files.map(lambda path: ipv.read(path, '<pattern>',apply = lambda line: float(line.split()[0])))
        >>> files.map(lambda path: ipv.load(path), to_df=True) 
        """
        if to_df:
            return self._try_return_df(func)
        return ((path, func(path)) for path in self._files) # generator must
    
    def _try_return_df(self, func):
        try: return summarize(self._files,func)
        except: return pd.DataFrame(((path, func(path)) for path in self._files))
    
    def mapf(self, func, to_df=False,mode='r', encoding=None):
        """Map files to a function that takes opened file pointer as argument. Opened files are automatically closed and should be in readonly mode.
        Load files content into a generator sequence of  tuples like `(path, func(open(path)))` or DataFrame if `to_df=True`.
        If `to_df=True`, func may return a dict to create named columns, or just two columns will be created.
        If you need to operate on just path, use `.map` instead.
        
        >>> import json
        >>> import ipyvasp as ipv
        >>> files = ipv.Files(...)
        >>> files.mapf(lambda fp: json.load(fp,cls=ipv.DecodeToNumpy),to_df=True) # or use ipv.load(path) in map
        >>> files.mapf(lambda fp: ipv.take(fp, range(5)) # read first five lines
        >>> files.mapf(lambda fp: ipv.take(fp, range(-5,0)) # read last five lines
        >>> files.mapf(lambda fp: ipv.take(fp, -1, 1, float) # read last line, second column as float
        """
        if not mode in 'rb':
            raise ValueError("Only 'r'/'rb' mode is allowed in this context!")
        
        def loader(path):
            with open(path, mode=mode,encoding=encoding) as f:
                return func(f)

        if to_df:
            return self._try_return_df(loader)
        return ((path, loader(path)) for path in self._files) # generator must
        
    def stat(self):
        "Get files stat as DataFrame. Currently only size is supported."
        return self.summarize(lambda path: {"size": get_file_size(path)})


@fix_signature
class _PropPicker(VBox):
    """Single projection picker with atoms and orbitals selection"""
    props = traitlets.Dict({})
    
    def __init__(self, system_summary=None):
        super().__init__()
        self._atoms = TagsInput(description="Atoms", allowed_tags=[], 
            placeholder="Select atoms", allow_duplicates = False).add_class('props-tags')
        self._orbs = TagsInput(description="Orbs", allowed_tags=[], 
            placeholder="Select orbitals", allow_duplicates = False).add_class('props-tags')
        self.children = [self._atoms, self._orbs]
        self.layout.width = '100%' # avoid horizontal collapse
        self._atoms_map = {}
        self._orbs_map = {}
        
        # Link changes
        self._atoms.observe(self._update_props, 'value')
        self._orbs.observe(self._update_props, 'value')
        self._process(system_summary)

    def _update_props(self, change):
        """Update props trait when selections change"""
        _atoms = [self._atoms_map.get(tag, None) for tag in self._atoms.value]
        _orbs = [self._orbs_map.get(tag, None) for tag in self._orbs.value]
        
        # Filter out None values, and flatten
        # Flatten and filter atoms
        atoms = []
        for ats in _atoms:
            atoms.extend(ats if ats is not None else [])

        # Flatten and filter orbitals
        orbs = []
        for ors in _orbs:
            orbs.extend(ors if ors is not None else [])

        if atoms and orbs:
            self.props = { 
                'atoms': atoms, 'orbs': orbs, 
                'label': f"{'+'.join(self._atoms.value)} | {'+'.join(self._orbs.value)}"
            }
        else:
            self.props = {}

    def _process(self, system_summary):
        """Process system data and setup widget options"""
        if system_summary is None or not hasattr(system_summary, "orbs"):
            return 

        sorbs = system_summary.orbs
        self._orbs_map = {"All": range(len(sorbs)), "s": [0]}

        # p-orbitals
        if set(["px", "py", "pz"]).issubset(sorbs):
            self._orbs_map.update({
                "p": range(1, 4),
                "px+py": [idx for idx, key in enumerate(sorbs) if key in ("px", "py")],
                **{k: [v] for k, v in zip(sorbs[1:4], range(1, 4))}
            })

        # d-orbitals    
        if set(["dxy", "dyz"]).issubset(sorbs):
            self._orbs_map.update({
                "d": range(4, 9),
                **{k: [v] for k, v in zip(sorbs[4:9], range(4, 9))}
            })

        # f-orbitals
        if len(sorbs) == 16:
            self._orbs_map.update({
                "f": range(9, 16),
                **{k: [v] for k, v in zip(sorbs[9:16], range(9, 16))}
            })

        # Extra orbitals beyond f
        if len(sorbs) > 16:
            self._orbs_map.update({
                k: [idx] for idx, k in enumerate(sorbs[16:], start=16)
            })

        self._orbs.allowed_tags = list(self._orbs_map.keys())
        
        # Process atoms
        self._atoms_map = {
            "All": range(system_summary.NIONS),
            **{k: v for k,v in system_summary.types.to_dict().items()},
            **{f"{k}{n}": [v] for k,tp in system_summary.types.to_dict().items() 
               for n,v in enumerate(tp, 1)}
        }
        self._atoms.allowed_tags = list(self._atoms_map.keys())
        self._update_props(None)  # Trigger props update

    def update(self, system_summary):
        """Update widget with new system data while preserving selections"""
        old_atoms = self._atoms.value
        old_orbs = self._orbs.value
        self._process(system_summary)
        
        # Restore previous selections if still valid
        self._atoms.value = [tag for tag in old_atoms if tag in self._atoms.allowed_tags]
        self._orbs.value = [tag for tag in old_orbs if tag in self._orbs.allowed_tags]

@fix_signature
class PropsPicker(VBox): # NOTE: remove New Later
    """
    A widget to pick atoms and orbitals for plotting.

    Parameters
    ----------
    system_summary : (Vasprun,Vaspout).summary
    N : int, default is 3, number of projections to pick.
    
    You can observe `projections` trait.
    """
    projections = traitlets.Dict({})
    
    def __init__(self, system_summary=None, N=3):
        super().__init__()
        self._N = N
        self._pickers = [_PropPicker(system_summary) for _ in range(N)]
        self.add_class("props-picker")
        
        # Create widgets with consistent width
        self._picker = Dropdown(
            description="Color" if N == 3 else "Projection",
            options=["Red", "Green", "Blue"] if N == 3 else [str(i+1) for i in range(N)],
        )
        self._stack = Stack(children=self._pickers, selected_index=0)
        # Link picker dropdown to stack
        ipw.link((self._picker, 'index'), (self._stack, 'selected_index'))
        
        # Setup layout
        self.children = [self._picker, self._stack]
        
        # Observe pickers for props changes and button click
        for picker in self._pickers:
            picker.observe(self._update_projections, names=['props'])
            
    def _update_projections(self, change):
        """Update combined projections when any picker changes"""
        projs = {}
        for picker in self._pickers:
            if picker.props:  # Only add non-empty selections
                projs[picker.props['label']] = (
                    picker.props['atoms'],
                    picker.props['orbs']
                )
        self.projections = projs
            
    def update(self, system_summary):
        """Update all pickers with new system data"""
        for picker in self._pickers:
            picker.update(system_summary)

def _clean_legacy_data(path):
    "clean old style keys like VBM to vbm"
    data = serializer.load(path.absolute())  # Old data loaded
    if not any(key in data for key in ['VBM', 'α','vbm_k']):
        return data # already clean
    
    keys_map = {
        "SYSTEM": "sys",
        "VBM": "vbm",      # Old: New
        "CBM": "cbm", 
        "VBM_k": "kvbm", "vbm_k": "kvbm",
        "CBM_k": "kcbm", "cbm_k": "kcbm",
        "E_gap": "gap", 
        "\u0394_SO": "soc", 
        "α": "alpha", 
        "β": "beta", 
        "γ": "gamma",
    }
    new_data = {k:v for k,v in data.items() if k not in (*keys_map.keys(),*keys_map.values())} # keep other data
    for old, new in keys_map.items():
        if old in data:
            new_data[new] = data[old]  # Transfer value from old key to new key
        elif new in data:
            new_data[new] = data[new]  # Keep existing new style keys
        
    # save cleaned data
    serializer.dump(new_data,format="json",outfile=path)
    return new_data


def load_results(paths_list, exclude_keys=None):
    "Loads result.json from paths_list and returns a dataframe. Use exclude_keys to get subset of data."
    if exclude_keys is not None:
        if not isinstance(exclude_keys, (list,tuple)):
            raise TypeError(f"exclude_keys should be list of keys, got {type(exclude_keys)}")
        if not all([isinstance(key,str) for key in exclude_keys]):
            raise TypeError(f"all keys in exclude_keys should be str!")
    
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
            data = _clean_legacy_data(path)
            return {k:v for k,v in data.items() if k not in (exclude_keys or [])}
        except:
            return {}  # If not found, return empty dictionary

    return summarize(result_paths, load_data)

def _get_css(mode):
    return {
        '--jp-widgets-color':                 'white' if mode == 'dark' else 'black',
        '--jp-widgets-label-color':           'white' if mode == 'dark' else 'black',
        '--jp-widgets-readout-color':         'white' if mode == 'dark' else 'black',
        '--jp-widgets-input-color':           'white' if mode == 'dark' else 'black',
        '--jp-widgets-input-background-color': '#222' if mode == 'dark' else '#f7f7f7',
        '--jp-widgets-input-border-color':    '#8988' if mode == 'dark' else '#ccc',
        '--jp-layout-color2':                  '#555' if mode == 'dark' else '#ddd', # buttons
        '--jp-ui-font-color1':           'whitesmoke' if mode == 'dark' else 'black', # buttons
        '--jp-content-font-color1':           'white' if mode == 'dark' else 'black', # main text
        '--jp-layout-color1':                  '#111' if mode == 'dark' else '#fff', # background
        ':fullscreen': {'min-height':'100vh'},
        'background': 'var(--jp-widgets-input-background-color)', 'border-radius': '4px', 'padding':'4px 4px 0 4px',
        '> *': {
            'box-sizing': 'border-box',
            'background': 'var(--jp-layout-color1)',
            'border-radius': '4px', 'grid-gap': '8px', 'padding': '8px',
        },
        '.left-sidebar .sm': {
            'flex-grow': 1,
            'select': {'height': '100%',},
        },
        '.footer': {'overflow': 'auto','padding':0},
        '.widget-vslider, .jupyter-widget-vslider': {'width': 'auto'}, # otherwise it spans too much area
        'table': { # dataframe display sucks
            'color':'var(--jp-content-font-color1)',
            'background':'var(--jp-layout-color1)',
            'tr': {
                    '^:nth-child(odd)': {'background':'var(--jp-widgets-input-background-color)',},
                    '^:nth-child(even)': {'background':'var(--jp-layout-color1)',},
                },
        },
        '.props-picker': {
            'background': 'var(--jp-widgets-input-background-color)', # make feels like single widget
            'overflow-x': 'hidden', 'border-radius': '4px', 'padding': '4px',
        },
        '.props-tags': { 
            'background':'var(--jp-layout-color1)', 'border-radius': '4px', 'padding': '4px',
            '> input': {'width': '100%'},
            '> input::placeholder': {'color': 'var(--jp-ui-font-color1)'},
        },
    }

class _ThemedFigureInteract(ei.InteractBase):
    "Keeps self._fig anf self._theme button attributes for subclasses to use."
    def __init__(self, *args, **kwargs):
        self._fig = ei.patched_plotly(go.FigureWidget())
        self._theme = Button(icon='sun', description=' ', tooltip="Toggle Theme")
        super().__init__(*args, **kwargs)
        
        if not all([hasattr(self.params, 'fig'), hasattr(self.params, 'theme')]):
            raise AttributeError("subclass must include already initialized "
                "{'fig': self._fig,'theme':self._theme} in returned dict of _interactive_params() method.")
        self._update_theme(self._fig,self._theme) # fix theme in starts
        self.observe(self._autosize_figs, names = 'isfullscreen') # fix figurewidget problem

    def _autosize_figs(self, change):
        for w in self._all_widgets.values():
            # don't know yet about these without importing
            if re.search('plotly.*FigureWidget', str(type(w).__mro__)):
                w.layout.autosize = False # Double trigger is important
                w.layout.autosize = True
    
    def _interactive_params(self): return {} 

    def __init_subclass__(cls):
        if (not '_update_theme' in cls.__dict__) or (not hasattr(cls._update_theme,'_is_interactive_callback')):
            raise AttributeError("implement _update_theme(self, fig, theme) decorated by @callback in subclass, "
                "which should only call super()._update_theme(fig, theme) in its body.")
        super().__init_subclass__()
    
    @ei.callback
    def _update_theme(self, fig, theme):
        require_dark = (theme.icon == 'sun')
        theme.icon = 'moon' if require_dark else 'sun' # we are not observing icon, so we can do this
        fig.layout.template = "plotly_dark" if require_dark else "plotly_white"
        self.set_css() # automatically sets dark/light, ensure after icon set
        fig.layout.autosize = True # must
    
    @_sub_doc(ei.InteractBase.set_css) # overriding to alway be able to set_css
    def set_css(self, main=None, center=None):
        # This is after setting icon above, so logic is fliipped
        style = _get_css("light" if self._theme.icon == 'sun' else 'dark') # infer from icon to match
        if isinstance(main, dict):
            style = {**style, **main} # main should allow override
        elif main is not None:
            raise TypeError("main must be a dict or None, got: {}".format(type(main)))
        super().set_css(style, center)
    
    @property
    def files(self): 
        "Use self.files.update(...) to keep state of widget preserved with new files."
        if not hasattr(self, '_files'): # subclasses must set this, although no check unless user dots it
            raise AttributeError("self._files = Files(...) was never set!")
        return self._files


@fix_signature
class BandsWidget(_ThemedFigureInteract):
    """Visualize band structure from VASP calculation. You can click on the graph to get the data such as VBM, CBM, etc.
    
    You can observe three traits:

    - file: Currently selected file
    - clicked_data:  Last clicked point data, which can be directly passed to a dataframe.
    - selected_data: Last selection of points within a box or lasso, which can be directly passed to a dataframe and plotted accordingly.
    
    - You can use `self.files.update` method to change source files without effecting state of widget.
    - You can also use `self.iplot`, `self.splot` with `self.kws` to get static plts of current state, and self.results to get a dataframe.
    - You can use store_clicks to provide extra names of points you want to click and save data, besides default ones.
    """
    file = traitlets.Any(allow_none=True)
    clicked_data = traitlets.Dict(allow_none=True)
    selected_data = traitlets.Dict(allow_none=True)

    def __init__(self, files, height="600px", store_clicks=None):
        self.add_class("BandsWidget")
        self._kb_fig = go.FigureWidget() # for extra stuff
        self._kb_fig.update_layout(margin=dict(l=40, r=0, b=40, t=40, pad=0)) # show compact
        self._files = Files(files)
        self._bands = None
        self._kws = {}
        self._result = {}
        self._extra_clicks = ()

        if store_clicks is not None:
            if not isinstance(store_clicks, (list,tuple)):
                raise TypeError("store_clicks should be list of names " 
                    f"of point to be stored from click on figure, got {type(store_clicks)}")
        
            for name in store_clicks:
                if not isinstance(name, str) or not name.isidentifier():
                    raise ValueError(f"items in store_clicks should be a valid python variable name, got {name!r}")
                if name in ["vbm", "cbm", "so_max", "so_min"]:
                    raise ValueError(f"{name!r} already exists in default click points!")
                reserved = "gap soc v a b c alpha beta gamma direct".split()
                if name in reserved:
                    raise ValueError(f"{name!r} conflicts with reserved keys {reserved}")

            self._extra_clicks += tuple(store_clicks)
        
        super().__init__() # after extra clicks

        traitlets.dlink((self.params.file,'value'),(self, 'file'))
        traitlets.dlink((self.params.fig,'clicked'),(self, 'clicked_data'))
        traitlets.dlink((self.params.fig,'selected'),(self, 'selected_data'))
        
        self.relayout(
            left_sidebar=[
                'head','file','krange','kticks','brange', 'ppicks',
                [HBox(),('theme','button')], 'kb_fig',
            ],
            center=['hdata','fig','cpoint'],  footer = self.groups.outputs,
            right_sidebar = ['showft'],
            pane_widths=['25em',1,'2em'], pane_heights=[0,1,0], # footer only has uselessoutputs
            height=height
        )

    @traitlets.validate('selected_data','clicked_data')
    def _flatten_dict(self, proposal):
        data = proposal['value']
        if data is None: return None # allow None stuff
    
        if not isinstance(data, dict):
            raise traitlets.TraitError(f"Expected a dict for selected_data, got {type(data)}")
        
        _data = {k:v for k,v in data.items() if k != 'customdata' and 'indexes' not in k}
        _data.update(pd.DataFrame(data.get('customdata',{})).to_dict(orient='list'))
        return _data # since we know customdata, we can flatten dict


    @ei.callback
    def _update_theme(self, fig, theme):
        super()._update_theme(fig, theme)
        self._kb_fig.layout.template = fig.layout.template
        self._kb_fig.layout.autosize = True
    
    def _interactive_params(self):
        return dict(
            fig    = self._fig, theme = self._theme,  # include theme and fig
            kb_fig = self._kb_fig, # show selected data
            head   = ipw.HTML("<b>Band Structure Visualizer</b>"),
            file   = self.files.to_dropdown(),
            ppicks = PropsPicker(),
            button = Button(description="Update Graph", icon= 'update'),
            krange = ipw.IntRangeSlider(description="kpoints",min=0, max=1,value=[0,1], tooltip="Includes non-zero weight kpoints"),
            kticks = Text(description="kticks", tooltip="0 index maps to minimum value of kpoints slider."),
            brange = ipw.IntRangeSlider(description="bands",min=1, max=1), # number, not index
            cpoint = ipw.ToggleButtons(description="Select from options and click on figure to store data points", 
                        value=None, options=["vbm", "cbm", *self._extra_clicks]).add_class('content-width-button'), # the point where clicked
            showft = ipw.IntSlider(description = 'h', orientation='vertical',min=0,max=50, value=0,tooltip="outputs area's height ratio"),
            cdata  = 'fig.clicked', 
            projs  = 'ppicks.projections', # for visual feedback on button
            sdata  = '.selected_data',
            hdata  = ipw.HTML(), # to show data in one place
        )
    
    @ei.callback('out-selected')
    def _plot_data(self, kb_fig, sdata):
        kb_fig.data = [] # clear in any case to avoid confusion
        if not sdata: return # no change

        df = pd.DataFrame(sdata)
        if 'r' in sdata:
            arr = df[['r','g','b']].to_numpy()
            arr[arr == ''] = 0
            arr, fmt = arr / (arr.max() or 1), lambda v : int(v*255) # color norms
            df['color'] = [f"rgb({fmt(r)},{fmt(g)},{fmt(b)})" for r,g,b in arr]
        else:
            df['color'] = sdata['occ']

        df['msize'] = df['occ']*7 + 10
        cdata = (df[["ys","occ","r","g","b"]] if 'r' in sdata else df[['ys','occ']]).to_numpy()
        rgb_temp = '<br>orbs: (%{customdata[2]},%{customdata[3]},%{customdata[4]})' if 'r' in sdata else ''

        kb_fig.add_trace(go.Scatter(x=df.nk, y = df.nb, mode = 'markers', marker = dict(size=df.msize,color=df.color), customdata=cdata))
        kb_fig.update_traces(hovertemplate=f"nk: %{{x}}, nb: %{{y}})<br>en: %{{customdata[0]:.4f}}<br>occ: %{{customdata[1]:.4f}}{rgb_temp}<extra></extra>")
        kb_fig.update_layout(template = self._fig.layout.template, autosize=True,
            title = "Selected Data", showlegend=False,coloraxis_showscale=False,
            margin=dict(l=40, r=0, b=40, t=40, pad=0),font=dict(family="stix, serif", size=14)
        )
    
    @ei.callback('out-data')
    def _load_data(self, file):
        if not file: return  # First time not available
        self._bands = (
            vp.Vasprun(file) if file.parts[-1].endswith('xml') else vp.Vaspout(file)
        ).bands
        self.params.ppicks.update(self.bands.source.summary)
        self.params.krange.max = self.bands.source.summary.NKPTS - 1
        self.params.krange.tooltip = f"Includes {self.bands.source.get_skipk()} non-zero weight kpoints"
        self.bands.source.set_skipk(0) # full range to view for slider flexibility after fix above
        self._kws['kpairs'] = [self.params.krange.value,]
        if (ticks := ", ".join(
            f"{k}:{v}" for k, v in self.bands.get_kticks()
        )): # Do not overwrite if empty
            self.params.kticks.value = ticks
        
        self.params.brange.max = self.bands.source.summary.NBANDS
        if self.bands.source.summary.LSORBIT:
            self.params.cpoint.options = ["vbm", "cbm", "so_max", "so_min", *self._extra_clicks]
        else:
            self.params.cpoint.options = ["vbm", "cbm",*self._extra_clicks]
        if (path := file.parent / "result.json").is_file():
            self._result = _clean_legacy_data(path)

        pdata = self.bands.source.poscar.data
        self._result.update(
            {
                "sys": pdata.SYSTEM, "v": round(pdata.volume, 4),
                **{k: round(v, 4) for k, v in zip("abc", pdata.norms)},
                **{k: round(v, 4) for k, v in zip(["alpha","beta","gamma"], pdata.angles)},
            }
        )
        self._show_data(self._result)  # Load into view
    
    @ei.callback
    def _toggle_footer(self, showft):
        self._app.pane_heights = [0,100 - showft, showft]
    
    @ei.callback
    def _set_krange(self, krange):
        self._kws["kpairs"] = [krange,]

    @ei.callback
    def _warn_update(self, file, kticks, brange, krange, projs):
        self.params.button.description = "🔴 Update Graph"

    @ei.callback('out-graph')
    def _update_graph(self, fig, button):
        if not self.bands: return  # First time not available
        fig.layout.autosize = True # must
        hsk = [
            [v.strip() for v in vs.split(":")]
            for vs in self.params.kticks.value.split(",")
        ]
        kmin, kmax = self.params.krange.value or [0,0]
        kticks = [(int(vs[0]), vs[1]) 
            for vs in hsk  # We are going to pick kticks silently in given range
            if len(vs) == 2 and abs(int(vs[0])) < (kmax - kmin) # handle negative indices too
        ] or None
        
        _bands = None
        if self.params.brange.value:
            l, h = self.params.brange.value
            _bands = range(l-1, h) # from number to index

        self._kws = {**self._kws, "kticks": kticks, "bands": _bands}
        ISPIN = self.bands.source.summary.ISPIN
        if self.params.ppicks.projections:
            self._kws["projections"] = self.params.ppicks.projections
            _fig = self.bands.iplot_rgb_lines(**self._kws, name="Up" if ISPIN == 2 else "")
            if ISPIN == 2:
                self.bands.iplot_rgb_lines(**self._kws, spin=1, name="Down", fig=fig)

            self.iplot = partial(self.bands.iplot_rgb_lines, **self._kws)
            self.splot = partial(self.bands.splot_rgb_lines, **self._kws)
        else:
            self._kws.pop("projections",None) # may be previous one
            _fig = self.bands.iplot_bands(**self._kws, name="Up" if ISPIN == 2 else "")
            if self.bands.source.summary.ISPIN == 2:
                self.bands.iplot_bands(**self._kws, spin=1, name="Down", fig=fig)

            self.iplot = partial(self.bands.iplot_bands, **self._kws)
            self.splot = partial(self.bands.splot_bands, **self._kws)

        ptk.iplot2widget(_fig, fig, template=fig.layout.template)
        fig.clicked = {}  # avoid data from previous figure
        fig.selected = {}  # avoid data from previous figure
        button.description = "Update Graph" # clear trigger

    @ei.callback('out-click')
    def _click_save_data(self, cdata):
        if self.params.cpoint.value is None: return # at reset-
        data_dict = self._result.copy()  # Copy old data

        if cdata:  # No need to make empty dict
            key = self.params.cpoint.value
            if key:
                y = round(float(*cdata['ys']) + self.bands.data.ezero, 6)  # Add ezero
                if not key in self._extra_clicks:
                    data_dict[key] = y  # Assign value back
                
                if not key.startswith("so_"): # not spin-orbit points
                    cst, = cdata.get('customdata',[{}]) # single item
                    kp = [cst.get(f"k{n}", None) for n in 'xyz']
                    kp = tuple([round(k,6) if k else k for k in kp])
                    
                    if key in ("vbm","cbm"):
                        data_dict[f"k{key}"] =  kp 
                    else: # user points, stor both for reference
                        data_dict[key] = {"k":kp,"e":y}   


            if data_dict.get("vbm", None) and data_dict.get("cbm", None):
                data_dict["gap"] = np.round(data_dict["cbm"] - data_dict["vbm"], 6)

            if data_dict.get("so_max", None) and data_dict.get("so_min", None):
                data_dict["soc"] = np.round(
                    data_dict["so_max"] - data_dict["so_min"], 6
                )

            self._result.update(data_dict)  # store new data
            self._show_and_save(self._result, f"{key} = {data_dict[key]}")
        self.params.cpoint.value = None  # Reset to None to avoid accidental click at end

    def _show_data(self, data, last_click=None):
        "Show data in html widget, no matter where it was called."
        keys = "sys vbm cbm gap direct soc v a b c alpha beta gamma".split()
        data = {key:data[key] for key in keys if key in data} # show only standard data
        kv, kc = [self._result.get(k,[None]*3) for k in ('kvbm','kcbm')]
        data['direct'] = (kv == kc) if None not in kv else False

        # Add a caption to the table
        caption = f"<caption style='caption-side:bottom; opacity:0.7;'><code>{last_click or 'clicked data is shown here'}</code></caption>"
    
        headers = "".join(f"<th>{key}</th>" for key in data.keys())
        values = "".join(f"<td>{format(value, '.4f') if isinstance(value, float) else value}</td>" for value in data.values())
        self.params.hdata.value = f"""<table border='1' style='width:100%;max-width:100% !important;border-collapse:collapse;'>
            {caption}<tr>{headers}</tr>\n<tr>{values}</tr></table>"""
    
    def _show_and_save(self, data_dict, last_click=None):
        self._show_data(data_dict,last_click=last_click)
        if self.file:
            serializer.dump(data_dict,format="json",
            outfile=self.file.parent / "result.json")
    
    def results(self, exclude_keys=None):
        "Generate a dataframe form result.json file in each folder, with optionally excluding keys."
        return load_results(self.params.file.options, exclude_keys=exclude_keys)
    
    @property
    def source(self):
        "Returns data source object such as Vasprun or Vaspout."
        return self.bands.source

    @property
    def bands(self):
        "Bands class initialized"
        if not self._bands:
            raise ValueError("No data loaded by BandsWidget yet!")
        return self._bands

    @property
    def kws(self):
        "Selected keyword arguments from GUI"
        return self._kws



@fix_signature
class KPathWidget(_ThemedFigureInteract):
    """
    Interactively bulid a kpath for bandstructure calculation.

    After initialization and disply:

    - Select a POSCAR file from "File:" dropdown menu. It will update the figure.
    - Add points to select box on left by clicking on plot points. When done with points click on Lock to avoid adding more points.
    - To update point(s), select point(s) from the select box and click on a scatter point in figure or use KPOINT input to update it manually, e.g. if a point is not available on plot.
    - Add labels to the points by typing in the "Labels" box such as "Γ,X" or "Γ 5,X" that will add 5 points in interval.
    - To break the path between two points "Γ" and "X" type "Γ 0,X" in the "Labels" box, zero means no points in interval.

    - You can use `self.files.update` method to change source files without effecting state of widget.
    - You can observe `self.file` trait to get current file selected and plot something, e.g. lattice structure.
    """
    file = traitlets.Any(None, allow_none=True)

    @property
    def poscar(self): return self._poscar
        
    def __init__(self, files, height="450px"):
        self.add_class("KPathWidget")
        self._poscar = None
        self._oldclick = None
        self._kpoints = {}
        self._files = Files(files) # set name _files to ensure access to files
        super().__init__()
        traitlets.dlink((self.params.file,'value'),(self, 'file')) # update file trait

        btns = [HBox(layout=Layout(min_height="24px")),('lock','delp', 'theme')]
        self.relayout(
            left_sidebar=['head','file',btns, 'info', 'sm','out-kpt','kpt', 'out-lab', 'lab'],
            center=['fig'],  footer = [c for c in self.groups.outputs if not c in ('out-lab','out-kpt')],
            pane_widths=['25em',1,0], pane_heights=[0,1,0], # footer only has uselessoutputs
            height=height
        )

    def _show_info(self, text, color='skyblue'):
        self.params.info.value = f'<span style="color:{color}">{text}</span>'

    def _interactive_params(self):
        return dict(
            fig = self._fig, theme = self._theme,  # include theme and fig
            head = ipw.HTML("<b>K-Path Builder</b>"),
            file = self.files.to_dropdown(), # auto updatable on files.update
            sm  = SelectMultiple(description="KPOINTS", options=[], layout=Layout(width="auto")),
            lab = Text(description="Labels", continuous_update=True),
            kpt = Text(description="KPOINT", continuous_update=False),
            delp = Button(description=" ", icon='trash', tooltip="Delete Selected Points"),
            click = 'fig.clicked',
            lock = Button(description=" ", icon='unlock', tooltip="Lock/Unlock adding more points"),
            info = ipw.HTML(), # consise information in one place
        )

    @ei.callback('out-fig')
    def _update_fig(self, file, fig):
        if not file: return # empty one

        from ipyvasp.lattice import POSCAR  # to avoid circular import
        self._poscar = POSCAR(file)
        ptk.iplot2widget(
            self._poscar.iplot_bz(fill=False, color="red"), fig, self.params.fig.layout.template
        )
        fig.layout.autosize = True # must
            
        with fig.batch_animate():
            fig.add_trace(
                go.Scatter3d(x=[], y=[], z=[],
                    mode="lines+text",
                    name="path",
                    text=[],
                    hoverinfo="none",  # dont let it block other points
                    textfont_size=18,
                )
            )  # add path that will be updated later
        self._show_info("Click points on plot to store for kpath.")

    @ei.callback('out-click')
    def _click(self, click):
        # We are setting value on select multiple to get it done in one click conveniently
        # But that triggers infinite loop, so we need to check if click is different next time
        if click != self._oldclick and (tidx := click.get('trace_indexes',[])):
            self._oldclick = click # for next time
            data = self.params.fig.data # click depends on fig, so accessing here
            if not [data[i] for i in tidx if 'HSK' in data[i].name]: return
            
            if cp := [*click.get('xs', []),*click.get('ys', []),*click.get('zs', [])]:
                kp = self._poscar.bz.to_fractional(cp) # reciprocal space
                
                if self.params.sm.value:
                    self._set_kpt(kp)  # this updates plot back as well
                elif self.params.lock.icon == "unlock":  # only add when open
                    self._add_point(kp)
                    
    @ei.callback('out-kpt')
    def _take_kpt(self, kpt):
        print("Add kpoint e.g. 0,1,3 at selection(s)")
        self._set_kpt(kpt)

    @ei.callback('out-lab')
    def _set_lab(self, lab):
        print("Add label[:number] e.g. X:5,Y,L:9")
        self._add_label(lab)

    @ei.callback
    def _update_theme(self, fig, theme):
        super()._update_theme(fig, theme)  # call parent method, but important


    @ei.callback
    def _toggle_lock(self, lock):
        self.params.lock.icon = 'lock' if self.params.lock.icon == 'unlock' else 'unlock'
        self._show_info(f"{self.params.lock.icon}ed adding/deleting kpoints!")

    @ei.callback
    def _del_point(self, delp):
        if self.params.lock.icon == 'unlock': # Do not delete locked
            sm = self.params.sm
            for v in sm.value:  # for loop here is important to update selection properly
                sm.options = [opt for opt in sm.options if opt[1] != v]
                self._update_selection()  # update plot as well
            else:
                self._show_info("Select point(s) to delete")
        else:
            self._show_info("cannot delete point when locked!", 'red')
    
    def _add_point(self, kpt):
        sm = self.params.sm
        sm.options = [*sm.options, ("⋮", len(sm.options))]
        # select to receive point as well, this somehow makes infinit loop issues, 
        # but need to work, so self._oldclick is used to check in _click callback
        sm.value = (sm.options[-1][1],) 
        self._set_kpt(kpt)  # add point, label and plot back

    def _set_kpt(self,kpt):
        point = kpt
        if isinstance(kpt, str) and kpt:
            if len(kpt.split(",")) != 3: return # Enter at incomplete input
            point = [float(v) for v in kpt.split(",")] # kpt is value widget

        if not isinstance(point,(list, tuple,np.ndarray)): return # None etc
        
        if len(point) != 3:
            raise ValueError("Expects KPOINT of 3 floats")
        self._kpoints.update({v: point for v in self.params.sm.value})
        label = "{:>8.4f} {:>8.4f} {:>8.4f}".format(*point)
        self.params.sm.options = [
            (label, value) if value in self.params.sm.value else (lab, value)
            for (lab, value) in self.params.sm.options
        ]
        self._add_label(self.params.lab.value)  # Re-adjust labels and update plot as well
        
    def _add_label(self, lab):
        labs = [" ⋮ " for _ in self.params.sm.options]  # as much as options
        for idx, (_, lb) in enumerate(zip(self.params.sm.options, (lab or "").split(","))):
            labs[idx] = labs[idx] + lb  # don't leave empty anyhow
        
        self.params.sm.options = [
            (v.split("⋮")[0].strip() + lb, idx)
            for (v, idx), lb in zip(self.params.sm.options, labs)
        ]
        self._update_selection()  # Update plot in both cases, by click or manual input

    def get_kpoints(self):
        "Returns kpoints list including labels and numbers in intervals if given."
        keys = [idx for (_, idx) in self.params.sm.options if idx in self._kpoints]  # order and existence is important
        kpts = [self._kpoints[k] for k in keys]
        LN = [
            lab.split("⋮")[1].strip().split()
            for (lab, idx) in self.params.sm.options
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

    def get_coords_labels(self):
        "Returns tuple of (coordinates, labels) to directly plot."
        points = self.get_kpoints()
        coords = self.poscar.bz.to_cartesian([p[:3] for p in points]).tolist() if points else []
        labels = [p[3] if (len(p) >= 4 and isinstance(p[3], str)) else "" for p in points]
        numbers = [
            p[4] if len(p) == 5
            else p[3] if (len(p) == 4 and isinstance(p[3], int))
            else "" for p in points]

        j = 0
        for i, n in enumerate(numbers, start=1):
            if isinstance(n, int) and n == 0:
                labels.insert(i + j, "")
                coords.insert(i + j, [np.nan, np.nan, np.nan])
                j += 1
        return np.array(coords), labels

    def _update_selection(self):
        coords, labels = self.get_coords_labels()
        with self.params.fig.batch_animate():
            for trace in self.params.fig.data:
                if "path" in trace.name and coords.any():
                    trace.x = coords[:, 0]
                    trace.y = coords[:, 1]
                    trace.z = coords[:, 2]
                    trace.text = _fmt_labels(labels)  # convert latex to html equivalent

    @_sub_doc(lat.get_kpath, {"kpoints :.*n :": "n :", "rec_basis :.*\n\n": "\n\n"})
    @_sig_kwargs(lat.get_kpath, ("kpoints", "rec_basis"))
    def get_kpath(self, n=5, **kwargs):
        return self.poscar.get_kpath(self.get_kpoints(), n=n, **kwargs)

    def iplot(self):
        "Returns disconnected current plotly figure"
        return go.Figure(data=self.params.fig.data, layout=self.params.fig.layout)

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

# Should be at end
del fix_signature  # no more need
