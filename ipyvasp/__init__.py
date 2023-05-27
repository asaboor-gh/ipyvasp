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
from .widgets import FilesWidget

__all__.extend(api_all)



# Access all functions through root modile ipyvasp
from .api import *
    
from matplotlib.pyplot import show as _show,savefig as _savefig

mpl_imported=['_show','_savefig']
__all__.extend(mpl_imported)

# Register 'RGB' colormap in current session
from matplotlib.colors import LinearSegmentedColormap as __LSC
import matplotlib.pyplot as __plt, numpy as __np
RGB = __LSC.from_list('RGB',[(0.9,0,0),(0.9,0.9,0),(0,0.9,0),(0,0.9,0.9),(0,0,0.9)])
CMY = __LSC.from_list('CMY',[(0,0.9,0.9),(0,0,0.9),(0.9,0,0.9),(0.9,0,0),(0.9,0.9,0)])
__plt.register_cmap('RGB',RGB)
__plt.register_cmap('CMY',CMY)

def create_colormap(name='RB',colors=[(0.9,0,0),(0,0,0.9)]):
    """
    Create and register a custom colormap from a list of RGB colors. and then use it's name in plottoing functions to get required colors.
    - name: str, name of the colormap
    - colors: list of RGB colors, e.g. [(0.9,0,0),(0,0,0.9)] or named colors, e.g. ['red','blue'], add as many colors as you want.
    
    **Returns**: Colormap object which you can use to get colors from. like cm = create_colormap(); cm(0.5) which will return a color at center of map
    """
    __RGB = __LSC.from_list(name,colors)
    __plt.register_cmap(name,__RGB)
    return __RGB

# color_marices for quick_rgb_lines
color_matrix = __np.array([[0.5,0,0.5,1],[0.5,0.5,0,1],[0,0.5,0.5,0.2],[1,1,0.2,0]]) # lights up to see colors a little bit
rbg_matrix= __np.array([[1,0,0],[0,0,1],[0,1,0]]) # Red, Blue, Green
cmy_matrix = __np.array([[0,0.5,0.5,1],[0.5,0,0.5,1],[0.5,0.5,0,0.2],[1,1,0.2,0]]) # Generates CMYK color palette


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

import webbrowser as __wb
def docs():
    __wb.open('https://massgh.github.io/ipyvasp/',new=1)

def example():
    __wb.open('https://massgh.github.io/ipyvasp/Example.html',new=1)
    
def example_notebook():
    __wb.open('https://colab.research.google.com/github/massgh/ipyvasp/blob/master/test.ipynb',new=1)
    
__all__ = ['docs','example','example_notebook', 'create_colormap' ,*__all__]
    
    