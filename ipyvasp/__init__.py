"""ipyvasp is a processing tool for VASP DFT input/output processing.
Author: Abdul Saboor
Licence: Apache License Version 2.0, January 2004 #See file
    
Links
-----
    [github](https://github.com/massgh/ipyvasp)            
    [docs](https://massgh.github.io/ipyvasp/)     
"""

links = """[github](https://github.com/massgh/ipyvasp)             
[docs](https://massgh.github.io/ipyvasp/)"""

__version__ = "0.1.0"

__all__ = []

from .api import __all__ as api_all
from .parser import Vaspout, Vasprun, minify_vasprun

__all__.extend(api_all)



# Access all functions through root modile ipyvasp
from .api import *
    
from matplotlib.pyplot import show as _show,savefig as _savefig

mpl_imported=['_show','_savefig']
__all__.extend(mpl_imported)


# Edit rcParams here
import matplotlib as __mpl
from cycler import cycler as __cycler
__mpl.rcParams.update(
    {
        'figure.dpi': 144, #Better to See
        'figure.figsize': [4,2.8],
        'axes.prop_cycle': __cycler(color=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']),
        'axes.linewidth': 0.4, #set the value globally
        'font.serif': "STIXGeneral",
        'font.family': "serif",
        'mathtext.fontset': "stix"
    }
)

def docs():
    from IPython.display import display, Markdown
    return display(Markdown('[ipyvasp-docs](https://massgh.github.io/ipyvasp/)'))


__all__ = ['docs',*__all__]
    
    