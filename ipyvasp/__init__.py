"""ipyvasp is a processing tool for VASP DFT input/output processing.

It is designed to primarily be used in Jupyter Notebook because it offers
widgets for interactive visualization and bulk analysis.
"""


__all__ = [  # For documentation purpose
    "get_axes",
    "plt2text",
    "plt2html",
    "iplot2html",
    "iplot2widget",
    "webshow",
    "list_files",
    "parse_text",
    "summarize",
    "OUTCAR",
]

from ._version import __version__
from .core.parser import *
from .core.serializer import *
from .misc import *
from .lattice import *
from .bsdos import *
from .potential import *
from .evals_dataframe import *
from .utils import *
from .widgets import summarize, BandsWidget, KpathWidget, FilesWidget
from .core import plot_toolkit, spatial_toolkit
from .core.spatial_toolkit import to_basis, to_R3, get_TM, get_bz, rotation
from .core.plot_toolkit import (
    get_axes,  # other options are available as attributes of get_axes
    global_matplotlib_settings,
    plt2text,
    plt2html,
    iplot2html,
    iplot2widget,
    webshow,
)

show = plot_toolkit.plt.show  # for convenience (and already imported)
savefig = plot_toolkit.plt.savefig  # for convenience

version = __version__
# Set global matplotlib settings for notebook.
global_matplotlib_settings()
