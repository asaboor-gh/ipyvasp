
__all__ = ['modify_axes', 'get_axes', 'append_axes', 'join_axes', 'break_spines', 'add_text', 'splot_bands',
           'add_legend', 'add_colorbar', 'color_wheel', 'color_cube', 'splot_rgb_lines', 'splot_color_lines',
           'splot_dos_lines', 'plt2html', 'show', 'savefig', 'plt2text', 'plot_potential']


from io import BytesIO
import PIL #For text image.

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap as LSC
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import tri
from matplotlib.collections import PolyCollection

from IPython import get_ipython
from IPython.display import HTML, set_matplotlib_formats #HTML for plt2html
from plotly.io._base_renderers import open_html_in_browser

# Inside packages import 
from . import parser as vp
from . import utils as gu
from . import serializer



# print SVG in ipython
try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell' or shell == 'Shell': # Shell for colab.
        set_matplotlib_formats('svg')
except: pass # Not in terminal

# Gloabal settings matplotlib
mpl.rcParams['axes.linewidth'] = 0.4 #set the value globally
mpl.rcParams['font.serif'] = "STIXGeneral"
mpl.rcParams['font.family'] = "serif"
mpl.rcParams['mathtext.fontset'] = "stix"


def modify_axes(ax = None,xticks = [],xt_labels = [],xlim = [],\
            yticks = [],yt_labels = [],ylim = [],xlabel = None,ylabel = None,\
            vlines = False,zeroline = True, **kwargs):
    """
    - Returns None, applies given settings on axes. Prefered to use before other plotting.
    Args:
        - ax  : Matplotlib axes object.
        - (x,y)ticks : List of positions on (x,y axes).
        - (xt,yt)_labels : List of labels on (x,y) ticks points.
        - (x,y)lim : [min, max] of (x,y) axes.
        - (x,y)label : axes labels.
        - vlines : If True, draw vertical lines at points of xticks.
        - zeroline : If True, drawn when `xlim` is not empty.
    
    kwargs are passed to `ax.tick_params`
    """
    if ax is None:
        raise ValueError("Matplotlib axes (ax) is not given.")
    else:
        if xticks:
            ax.set_xticks(xticks)
            ax.set_xticklabels(xt_labels if xt_labels else list(map(str,xticks)))
        if yticks:
            ax.set_yticks(yticks)
            ax.set_yticklabels(yt_labels if yt_labels else list(map(str,yticks)))
        if xlim:
            ax.set_xlim(xlim)
            if zeroline:
                ax.hlines(0,min(xlim),max(xlim),color=(0,0,0,0.6), linestyle='dashed',lw=0.3)
        if ylim:
            ax.set_ylim(ylim)
            
        if vlines:
            ax.xaxis.grid(color=(0,0,0,0.6), linestyle='dashed',lw=0.3)
            
        if xlabel!=None:
            ax.set_xlabel(xlabel)
        if ylabel!=None:
            ax.set_ylabel(ylabel)
        kwargs = {**dict(direction='in', bottom=True,left = True,length=4, width=0.3, grid_alpha=0.8),**kwargs} # Default kwargs
        ax.tick_params(**kwargs)
        ax.set_axisbelow(True) # Aoid grid lines on top of plot.
    return None

def get_axes(figsize  = (3.4,2.6),
            nrows     = 1,
            ncols     = 1,
            widths    = [],
            heights   = [],
            axes_off  = [],
            axes_3d   = [],
            sharex    = False,
            sharey    = False,
            azim      = 45,
            elev      = 15,
            ortho3d   = True,
            **subplots_adjust_kwargs
            ):
    """
    - Returns flatten axes of initialized figure, based on plt.subplots(). If you want to access parent figure, use ax.get_figure() or current figure as plt.gcf().
    Args:
        - figsize   : Tuple (width, height). Default is (3.4,2.6).
        - nrows     : Default 1.
        - ncols     : Default 1.
        - widths    : List with len(widths)==nrows, to set width ratios of subplots.
        - heights   : List with len(heights)==ncols, to set height ratios of subplots.
        - share(x,y): Share axes between plots, this removes shared ticks automatically.
        - axes_off  : Turn off axes visibility, If `nrows = ncols = 1, set True/False`, If anyone of `nrows or ncols > 1`, provide list of axes indices to turn off. If both `nrows and ncols > 1`, provide list of tuples (x_index,y_index) of axes.
        - axes_3d   : Change axes to 3D. If `nrows = ncols = 1, set True/False`, If anyone of `nrows or ncols > 1`, provide list of axes indices to turn off. If both `nrows and ncols > 1`, provide list of tuples (x_index,y_index) of axes.
        - azim,elev   : Matplotlib's 3D angles, defualt are 45,15.
        - ortho3d     : Only works for 3D axes. If True, x,y,z are orthogonal, otherwise perspective.
        - **subplots_adjust_kwargs : These are same as `plt.subplots_adjust()`'s arguements.
    """
    # SVG and rcParams are must in get_axes to bring to other files, not just here.
    # print SVG in ipython
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell' or shell == 'Shell': # Shell for colab.
            set_matplotlib_formats('svg')
    except: pass # Not in terminal

    # Gloabal settings matplotlib, inside figure as well
    mpl.rcParams['axes.linewidth'] = 0.4 #set the value globally
    mpl.rcParams['font.serif'] = "STIXGeneral"
    mpl.rcParams['font.family'] = "serif"
    mpl.rcParams['mathtext.fontset'] = "stix"

    if figsize[0] <= 2.38:
        mpl.rc('font', size=8)
    gs_kw=dict({}) # Define Empty Dictionary.
    if widths!=[] and len(widths)==ncols:
        gs_kw=dict({**gs_kw,'width_ratios':widths})
    if heights!=[] and len(heights)==nrows:
        gs_kw = dict({**gs_kw,'height_ratios':heights})
    fig, axs = plt.subplots(nrows,ncols,figsize=figsize,gridspec_kw=gs_kw,sharex=sharex,sharey=sharey)
    proj = {'proj_type':'ortho'} if ortho3d else {} # For 3D only
    if nrows*ncols==1:
        modify_axes(ax=axs)
        if axes_off == True:
            axs.set_axis_off()
        if axes_3d == True:
            pos = axs.get_position()
            axs.remove()
            axs = fig.add_axes(pos,projection='3d',azim=azim,elev=elev,**proj)
            setattr(axs,add_legend.__name__,add_legend.__get__(axs,type(axs)))

    else:
        _ = [modify_axes(ax=ax) for ax in axs.ravel()]
        _ = [axs[inds].set_axis_off() for inds in axes_off if axes_off!=[]]
        if axes_3d != []:
            for inds in axes_3d:
                pos = axs[inds].get_position()
                axs[inds].remove()
                axs[inds] = fig.add_axes(pos,projection='3d',azim=azim,elev=elev,**proj)
    try:
        for ax in np.array([axs]).flatten():
            for f in [add_text,add_legend,add_colorbar,color_wheel,color_cube, break_spines,modify_axes,append_axes]:
                if ax.name != '3d':
                    setattr(ax,f.__name__,f.__get__(ax,type(ax)))
    except: pass

    plt.subplots_adjust(**subplots_adjust_kwargs)

    return axs

def append_axes(ax,position='right',size=0.2,pad=0.1,sharex=False,sharey=False,**kwargs):
    """
    Append an axes to the given `ax` at given `position` |top|right|left|bottom|. Useful for adding custom colorbar.
    kwargs are passed to `mpl_toolkits.axes_grid1.make_axes_locatable.append_axes`.
    Returns appended axes.
    """
    extra_args = {}
    if sharex:
        extra_args['sharex'] = ax
    if sharey:
        extra_args['sharey'] = ax
    divider = make_axes_locatable(ax)
    added_ax = divider.append_axes(position=position, size=size, pad=pad, **extra_args, **kwargs)
    _ = modify_axes(ax=added_ax) # tweaks of styles
    return added_ax

def join_axes(ax1,ax2, **kwargs):
    """Join two axes together. Useful for adding custom colorbar on a long left axes of whole figure.
    Apply tight_layout() before calling this function.
    kwargs are passed to `fig.add_axes`.
    """
    p1 = ax1.get_position()
    p2 = ax2.get_position()
    x0 = min(p1.x0,p2.x0)
    y0 = min(p1.y0,p2.y0)
    x1 = max(p1.x1,p2.x1)
    y1 = max(p1.y1,p2.y1)
    new_bbox = [x0,y0, x1-x0,y1-y0]
    fig = ax1.get_figure()
    ax1.remove()
    ax2.remove()
    new_ax = fig.add_axes(new_bbox, **kwargs)
    _ = modify_axes(new_ax)
    for f in [add_text,add_legend,add_colorbar,color_wheel,break_spines,modify_axes,append_axes]:
        if new_ax.name != '3d':
            setattr(new_ax,f.__name__,f.__get__(new_ax,type(new_ax)))
    return new_ax

def break_spines(ax,spines,symbol=u'\u2571',**kwargs):
    """Simulates broken axes using subplots. Need to fix heights according to given data for real aspect. Also plot the same data on each axes and set axes limits.
    Args:
        - ax : Axes who's spine(s) to edit.
        - spines: str,list, str/list of any of ['top','bottom','left','right'].
        - symbol: Defult is u'\u2571'. Its at 60 degrees. so you can apply rotation to make it any angle.
    kwargs are passed to plt.text.
    """
    kwargs.update(transform=ax.transAxes, ha='center',va = 'center')
    _spines = [spines] if isinstance(spines,str) else spines
    _ = [ax.spines[s].set_visible(False) for s in _spines]
    ax.tick_params(**{sp:False for sp in _spines})
    if 'top' in spines:
        ax.text(0,1,symbol,**kwargs)
        ax.text(1,1,symbol,**kwargs)
    if 'bottom' in spines:
        ax.set_xticks([])
        ax.text(0,0,symbol,**kwargs)
        ax.text(1,0,symbol,**kwargs)
    if 'left' in spines:
        ax.set_yticks([])
        ax.text(0,0,symbol,**kwargs)
        ax.text(0,1,symbol,**kwargs)
    if 'right' in spines:
        ax.text(1,1,symbol,**kwargs)
        ax.text(1,0,symbol,**kwargs)


def add_text(ax=None,xs=0.25,ys=0.9,txts='[List]',colors='r',transform=True,**kwargs):
    """
    - Adds text entries on axes, given single string or list.
    Args:
        - xs    : List of x coordinates relative to axes or single coordinate.
        - ys    : List of y coordinates relative to axes or single coordinate.
        - txts  : List of strings or one string.
        - colors: List of x colors of txts or one color.
        - transform: Dafault is True and positions are relative to axes, If False, positions are in data coordinates.

    kwargs are passed to plt.text.
    """
    if ax==None:
        raise ValueError("Matplotlib axes (ax) is not given.")
    else:
        bbox = kwargs.get('bbox',dict(edgecolor='white',facecolor='white', alpha=0.4))
        ha, va = kwargs.get('ha','center'), kwargs.get('va','center')
        args_dict = dict(bbox=bbox,ha=ha,va=va)
        if transform:
            args_dict.update({'transform':ax.transAxes})

        if isinstance(txts,str):
            ax.text(xs,ys,txts,color=colors, **args_dict, **kwargs)
        elif isinstance(txts,(list,np.ndarray)):
            for x,y,(i,txt) in zip(xs,ys,enumerate(txts)):
                try:
                    ax.text(x,y,txt,color=colors[i],**args_dict,**kwargs)
                except:
                    ax.text(x,y,txt,color=colors,**args_dict,**kwargs)
                    
def join_ksegments(kpath,*pairs):
    """Joins a broken kpath's next segment to previous. `pairs` should provide the adjacent indices of the kpoints to be joined."""
    path_arr = np.array(kpath)
    path_max = path_arr.max()
    if pairs:
        for pair in pairs:
            if len(pair) != 2:
                raise ValueError(f"{pair} should have exactly two indices.")
            for idx in pair:
                if not isinstance(idx, (int, np.integer)):
                    raise ValueError(f"{pair} should have integers, got {idx!r}.")
            
            idx_1, idx_2 = pair
            if idx_2 - idx_1 != 1:
                raise ValueError(f"Indices in pair ({idx_1}, {idx_2}) are not adjacent.")
            path_arr[idx_2:] -= path_arr[idx_2] - path_arr[idx_1]
        path_arr = path_max * path_arr/path_arr[-1] # Normalize to max value back
    return list(path_arr)
               
# This is to verify things together and make sure they are working as expected.
def _validate_data(K, E,  elim, kticks, interp):
    if np.ndim(E) != 2:
        raise ValueError("E must be a 2D array.")
    
    if np.shape(E)[0] != len(K):
        raise ValueError("Length of first dimension of E must be equal to length of K.")
    
    if kticks is None:
        kticks = []
    
    if not isinstance(kticks, (list, tuple, zip)):
        raise ValueError("kticks must be a list, tuple or zip consisting of (index, label) pairs. index must be an int or tuple of (i, i+1) to join broken path.")
    
    if isinstance(kticks, zip):
        kticks = list(kticks) # otherwise it will be empty after first use

    for k, v in kticks:
        if not isinstance(k, (np.integer, int)):
            raise ValueError("First item of pairs in kticks must be int")
        if not isinstance(v, str):
            raise ValueError("Second item of pairs in kticks must be str.")
    
    pairs = [(k - 1, k) for k, v in kticks if v.startswith('<=')] # Join broken path at these indices
    K = join_ksegments(K, *pairs)
    inds  = [k for k, _ in kticks]
    
    xticks = [K[i] for i in inds] if inds else None # Avoid turning off xticks if no kticks given
    xticklabels = [v.replace('<=','') for _, v in kticks] if kticks else None # clean up labels
    
    if elim and len(elim) != 2:
        raise ValueError("elim must be a list or tuple of length 2.")
    
    if interp and not isinstance(interp, (int, np.integer,list, tuple)):
        raise ValueError("interp must be an integer or a list/tuple of (n,k).")
    
    if isinstance(interp, (list,tuple)) and len(interp) != 2:
        raise ValueError("interp must be an integer or a list/tuple of (n,k).")
    
    return K, E, xticks, xticklabels

                    
def splot_bands(K, E, ax = None, elim = None, kticks = None, interp = None, **kwargs):
    """
    Plot band structure for a single spin channel and return the matplotlib axes which can be used to add other channel if spin polarized.
    
    Parameters
    ----------
    K : array-like of shape (nkpts,)
    E : array-like of shape (nkpts, nbands)
    ax : matplotlib axes 
    elim : list of length 2 
    kticks : [(int, str),...] for indices of high symmetry k-points. To join a broken path, use '<=' before symbol, e.g.  [(0, 'G'),(40, '<=K|M'), ...] will join 40 back to 39. You can also use shortcut like zip([0,10,20],'GMK').
    interp : int or list/tuple of (n,k) for interpolation. If int, n is number of points to interpolate. If list/tuple, n is number of points and k is the order of spline.
    
    kwargs are passed to matplotlib's command `ax.plot`.
    
    Returns
    -------
    ax : matplotlib axes
    """
    K, E, xticks, xticklabels = _validate_data(K, E, elim, kticks, interp)
    
    if interp:
        nk = interp if isinstance(interp, (list, tuple)) else (interp, 3)
        K,E = gu.interpolate_data(K,E,*nk)
    
    # handle broken paths
    breaks = [i for i in range(0,len(K)) if K[i-1] == K[i]]
    K = np.insert(K,breaks,np.nan)
    E = np.insert(E,breaks,np.nan,axis=0)

    ax = get_axes() if ax is None else ax
    if 'color' not in kwargs and 'c' not in kwargs:
        kwargs['color'] = 'C0' # default color from cycler to accommodate themes
    
    if 'linewidth' not in kwargs and 'lw' not in kwargs:
        kwargs['linewidth'] = 0.9 # default linewidth to make it look good
    
    lines = ax.plot(K,E,**kwargs)
    _ = [line.set_label(None) for line in lines[1:]]
    
    modify_axes(ax = ax,ylabel = 'Energy (eV)', xticks = xticks, xt_labels = xticklabels,
                xlim = [min(K),max(K)], ylim = elim, vlines = True,top=True, right=True)
    return ax

def add_legend(ax=None,colors=[],labels=[],styles='solid',\
                widths=0.7,anchor=(0,1), ncol=3,loc='lower left',fontsize='small',frameon=False,**kwargs):
    """
    - Adds custom legeneds on a given axes,returns None.
    Args:
        - ax       : Matplotlib axes.
        - colors   : List of colors.
        - labels   : List of labels.
        - styles   : str or list of line styles.
        - widths   : str or list of line widths.

    kwargs are passed to plt.legend. Given arguments like anchor,ncol etc are preferred.
    """
    kwargs.update(dict(bbox_to_anchor=anchor,ncol=ncol,loc=loc,fontsize=fontsize,frameon=frameon))
    if ax==None:
        raise ValueError("Matplotlib axes (ax) is not given.")
    else:
        if type(widths)==float or type(widths)==int:
            if(type(styles)==str):
                for color,label in zip(colors,labels):
                    ax.plot([],[],color=color,lw=widths,linestyle=styles,label=label)
            else:
                for color,label,style in zip(colors,labels,styles):
                    ax.plot([],[],color=color,lw=widths,linestyle=style,label=label)
        else:
            if(type(styles)==str):
                for color,label,width in zip(colors,labels,widths):
                    ax.plot([],[],color=color,lw=width,linestyle=styles,label=label)
            else:
                for color,label,width,style in zip(colors,labels,widths,styles):
                    ax.plot([],[],color=color,lw=width,linestyle=style,label=label)
        ax.legend(**kwargs)
    return None


def add_colorbar(
    ax,
    cmap_or_clist = None,
    N             = 256,
    ticks         = None,
    ticklabels    = None,
    vmin          = None,
    vmax          = None,
    cax           = None,
    tickloc       = 'right',
    vertical      = True,
    digits        = 2,
    fontsize      = 8
    ):
    """
    - Plots colorbar on a given axes. This axes should be only for colorbar. Returns None or throws ValueError for given colors.
    Args:
        - ax         : Matplotlib axes for which colorbar will be added.
        - cmap_or_clist: List/array of colors in or colormap's name. If None (default), matplotlib's default colormap is plotted.
        - N          : int, number of color points Default 256.
        - ticks      : List of tick values to show on colorbar. To turn off, give [].
        - ticklabels : List of labels for ticks.
        - vmin,vmax  : Minimum and maximum values. Only work if ticks are given.
        - cax        : Matplotlib axes for colorbar. If not given, one is created.
        - tickloc    : Default 'right'. Any of ['right','left','top','bottom'].
        - digits     : Number of digits to show in tick if ticklabels are not given.
        - vertical   : Boolean, default is Fasle.
        - fontsize   : Default 8. Adjustable according to plot space.

    - **Returns**
        - cax : Matplotlib axes for colorbar, you can customize further.
    """
    if cax is None:
        position = 'right' if vertical == True else 'top'
        cax = append_axes(ax,position= position,size='5%',pad=0.05)
    if cmap_or_clist is None:
        colors=np.array([[1,0,1],[1,0,0],[1,1,0],[0,1,0],[0,1,1],[0,0,1],[1,0,1]])
        _hsv_ = LSC.from_list('_hsv_',colors,N=N)
    elif isinstance(cmap_or_clist,(list,np.ndarray)):
        try:
            _hsv_ = LSC.from_list('_hsv_',cmap_or_clist,N=N)
        except Exception as e:
            print(e,"\nFalling back to default color map!")
            _hsv_ = None # fallback
    elif isinstance(cmap_or_clist,str):
        _hsv_ = cmap_or_clist #colormap name
    else:
        _hsv_ = None # default fallback

    if ticks != []:
        if ticks is None: # should be before labels
            ticks = np.linspace(1/6,5/6,3, endpoint=True)
            if ticklabels is None:
                ticklabels = ticks.round(digits).astype(str)

        elif isinstance(ticks,(list,tuple, np.ndarray)):
            ticks = np.array(ticks)
            _vmin = vmin if vmin is not None else np.min(ticks)
            _vmax = vmax if vmax is not None else np.max(ticks)
            if _vmin > _vmax:
                raise ValueError("vmin > vmax is not valid!")

            if ticklabels is None:
                ticklabels = ticks.round(digits).astype(str)
            # Renormalize ticks after assigning ticklabels
            ticks = (ticks - _vmin)/(_vmax - _vmin)
    else:
        ticks = []
        ticklabels = []

    c_vals = np.linspace(0,1,N, endpoint = True).reshape((1,N)) # make 2D array

    ticks_param = dict(direction='out',pad= 2,length=2,width=0.3,top=False,left=False,
                        grid_color=(1,1,1,0), grid_alpha=0)
    ticks_param.update({tickloc:True}) # Only show on given side
    cax.tick_params(**ticks_param)
    if vertical == False:
        cax.imshow(c_vals,aspect='auto',cmap=_hsv_,origin='lower', extent=[0,1,0,1])
        cax.set_yticks([])
        cax.xaxis.tick_top() # to show ticks on top by default
        if tickloc == 'bottom':
            cax.xaxis.tick_bottom() # top is by default
        cax.set_xticks(ticks)
        cax.set_xticklabels(ticklabels,rotation=0,ha='center')
        cax.set_xlim([0,1]) #enforce limit

    if vertical == True:
        c_vals = c_vals.transpose()
        cax.imshow(c_vals,aspect='auto',cmap=_hsv_,origin='lower',extent=[0,1,0,1])
        cax.set_xticks([])
        cax.yaxis.tick_right() # Show right by default
        if tickloc == 'left':
            cax.yaxis.tick_left() # right is by default
        cax.set_yticks(ticks)
        cax.set_yticklabels(ticklabels,rotation=90,va='center')
        cax.set_ylim([0,1]) # enforce limit

    for tick in cax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
    for child in cax.get_children():
        if isinstance(child, mpl.spines.Spine):
            child.set_color((1,1,1,0.4))
    return cax # Return colorbar axes to perform further customization


def color_wheel(
    ax=None,
    xy=(1,1),
    scale = 0.12,
    rlim=(0.2,1),
    N=256,
    colormap=None,
    ticks=[1/6,1/2,5/6],
    labels=['s','p','d'],
    showlegend=True
    ):
    """
    - Returns cax i.e. color wheel axes.
    Args:
        - ax        : Axes on which color wheel will be drawn. Auto created if not given.
        - xy        : (x,y) of center of wheel.
        - scale     : Scale of the cax internally generated by color wheel.
        - rlim      : Values in [0,1] interval, to make donut like shape.
        - N         : Number of segments in color wheel.
        - colormap  : Matplotlib's color map name. fallbacks to `hsv`.
        - ticks     : Ticks in fractions in interval [0,1].
        - labels    : Ticks labels.
        - showlegend: True or False.
    """
    if ax is None:
        ax = get_axes()
    if colormap is None:
        try: colormap = plt.cm.get_cmap('hsv')
        except: colormap = 'viridis'
    pos = ax.get_position()
    ratio = pos.height/pos.width
    cpos = [pos.x0+pos.width*xy[0]-scale/2,pos.y0+pos.height*xy[1]-scale/2,scale,scale]
    cax = ax.get_figure().add_axes(cpos,projection='polar')
    norm = mpl.colors.Normalize(0.0,2*np.pi)
    t = np.linspace(0,2*np.pi,N)
    r = np.linspace(*rlim,2)
    rg,tg = np.meshgrid(r,t)
    cax.pcolormesh(t,r,tg.T,norm=norm,cmap=colormap,edgecolor='face',shading='gouraud')
    cax.set_yticklabels([])
    cax.spines['polar'].set_visible(False)
    ##########
    if showlegend == True:
        colors = plt.cm.get_cmap(colormap)(ticks) # Get colors values.
        labels = ["◾ "+l for l in labels]
        labels[0] = labels[0]+'\n' #hack to write labels correctly on a single point.
        labels[2] = '\n'+ labels[2]
        for l,p,c in zip(labels,['bottom','center','top'],colors):
            cax.text(rlim[1]+0.1,0.5,l,va=p,ha='left',color=c,transform=cax.transAxes,fontsize=9)
        cax.set_xticklabels([])
    else:
        cax.set_xticks([t*2*np.pi for t in ticks])
        cax.set_xticklabels(labels)
    return cax


def _make_line_collection(maxwidth   = 3,
                         colors_list = None,
                         rgb         = False,
                         shadow      = False,
                         uniwidth    = False,
                         **pros_data):
    """
    - Returns a tuple of line collections for each given projection data.
    - **Parametrs**
        - **pros_data: Output dictionary from `_fix_data` containing kpath, evals, colors arrays.
        - maxwidth  : Default is 3. Max linewidth is scaled to maxwidth if an int of float is given.
        - uniwidth  : Default is False. If True, linewidth is set to maxwidth/2 for all lines. Only works for rgb_lines.
        - colors_list: List of colors for multiple lines, length equal to 3rd axis length of colors.
        - rgb        : Default is False. If True and np.shape(colors)[-1] == 3, RGB line collection is returned in a tuple of length 1. Tuple is just to support iteration.
    """
    if not isinstance(maxwidth,(int,np.integer,float)):
        raise ValueError("maxwidth must be an int or float.")
    
    if not pros_data:
        raise ValueError("No pros_data given.")
    else:
        kpath  = pros_data.get('kpath')
        evals  = pros_data.get('evals')
        pros   = pros_data.get('pros')

    for a,t in zip([kpath, evals, pros],['kpath', 'evals', 'pros']):
        if not np.any(a):
            raise ValueError("Missing {!r} from output of `_fix_data()`".format(t))

    # Average pros on two consecutive KPOINTS to get that patch color.
    colors = pros[1:,:,:]/2 + pros[:-1,:,:]/2 # Near kpoints avearge
    colors = colors.transpose((1,0,2)).reshape((-1,np.shape(colors)[-1])) # Must before lws

    if rgb: # Single channel line widths
        lws = np.sum(colors,axis=1) # Sum over RGB
    else: # For separate lines
        lws = colors.T # .T to access in for loop.

    lws = 0.1 + maxwidth*lws/(float(np.max(lws)) or 1) # Rescale to maxwidth, with a residual with 0.1 as must be visible.

    if np.any(colors_list):
        lc_colors = colors_list
    else:
        cmap = plt.cm.get_cmap('viridis')
        lc_colors = cmap(np.linspace(0,1,np.shape(colors)[-1]))
        lc_colors = lc_colors[:,:3] # Skip Alpha

    # Reshaping data same as colors reshaped above, nut making line patches too.
    kgrid = np.repeat(kpath,np.shape(evals)[1],axis=0).reshape((-1,np.shape(evals)[1]))
    narr  = np.concatenate((kgrid,evals),axis=0).reshape((2,-1,np.shape(evals)[1]))
    marr  = np.concatenate((narr[:,:-1,:],narr[:,1:,:]),axis=0).transpose().reshape((-1,2,2))

    # Make Line collection
    path_shadow = None
    if shadow:
        path_shadow = [path_effects.SimpleLineShadow(offset=(0,-0.8),rho=0.2),path_effects.Normal()]
    if rgb and np.shape(colors)[-1] == 3:
        return (LineCollection(marr,colors=colors,linewidths = (maxwidth/2,) if uniwidth else lws, path_effects = path_shadow),)
    else:
        lcs = [LineCollection(marr,colors=_cl,linewidths=lw, path_effects = path_shadow) for _cl,lw in zip(lc_colors,lws)]
        return tuple(lcs)

def color_cube(ax, colormap = 'brg', loc = (1,0.4), size = 0.2,
              N = 7, labels=['s','p','d'],color='k',fontsize=10):
    "Color-mapped hexagon that serves as a legend for `splot_rgb_lines`"
    if N < 3:
        raise ValueError("N must be >= 3 to map colors correctly.")

    X, Y = np.mgrid[0:1:N*1j,0:1:N*1j]
    x = X.flatten()
    y = Y.flatten()
    points_z = np.array([x,y,np.ones_like(x)]).T
    points_y = np.array([x,np.ones_like(x),y]).T
    points_x = np.array([np.ones_like(x), x, y]).T

    from .sio import to_R3, rotation # Avoids circular import

    all_points = []
    ps = to_R3([[1,0,0],[-0.5,-np.sqrt(3)/2,0],[0,0,1]],points_z)
    all_points.extend(ps)
    ps1 = rotation(angle_deg=-120,axis_vec=[0,0,1]).apply(ps+[0,np.sqrt(3),0])
    all_points.extend(ps1)
    ps2 = rotation(angle_deg=180,axis_vec=[0,0,1]).apply(ps*[-1,1,0]+[0,np.sqrt(3),0])
    all_points.extend(ps2)

    all_points = rotation(-30,axis_vec=[0,0,1]).apply(all_points)

    pts = np.asarray(all_points)[:,:2]
    pts = pts - pts.mean(axis=0) # center
    C = np.array([*points_z, *points_x, *points_y])

    fig = ax.get_figure()
    pos = ax.get_position()
    x0 = pos.x0 + loc[0]*pos.width
    y0 = pos.y0 + loc[1]*pos.height
    size = size*pos.width
    cax = fig.add_axes([x0,y0, size, size])

    tr1 = tri.Triangulation(*pts.T)

    # Have same color for traingles sharing hypotenuse to see box
    colors = []
    for t in tr1.triangles:
        a, b, c = C[t] # a,b,c are the 3 points of the triangle
        mid_point = (a + b)/2 # Right angle at c
        if np.dot(a - b, a - c) == 0: # Right angle at a
            mid_point = (b + c)/2
        elif np.dot(b - a, b - c) == 0: # Right angle at b
            mid_point = (a + c)/2

        colors.append(mid_point)

    colors = np.array(colors)

    A,B,_C = plt.cm.get_cmap(colormap)([0,0.5,1])[:,:3]
    _colors = np.array([(r*A + g*B + b*_C)/((r+g+b) or 1) for r,g,b in colors])
    _max = _colors.max(axis=1,keepdims=True) # Should be normalized after matching to colobar as well
    _max[_max == 0] = 1
    _colors = _colors/_max

    col = PolyCollection([pts[t] for t in tr1.triangles],color=_colors,linewidth=0.1,edgecolor='face',alpha=1)
    cax.add_collection(col)
    cax.autoscale_view()
    cax.set_aspect('equal')
    cax.set_facecolor([1,1,1,0])
    cax.set_axis_off()


    cax.text(9*np.sqrt(3)/16, -9/16, '→', fontsize=fontsize, zorder = -10, color=color,rotation = -30, ha='center', va='center')
    cax.text(-9*np.sqrt(3)/16, -9/16, '→', fontsize=fontsize, zorder = -10, color=color,rotation = 210, ha='center', va='center')
    cax.text(0, 9/8, '→', fontsize=fontsize, zorder = -10, color=color,rotation = 90, ha='center', va='center')

    cax.text(np.sqrt(3)/2,-5/8,f' {labels[0]}' ,color=color,fontsize=fontsize,va='top',ha='center', rotation=-90)
    cax.text(-np.sqrt(3)/2,-5/8,f' {labels[1]}' ,color=color,fontsize=fontsize,va='top',ha='center',rotation=-90)
    cax.text(0, 9/8,f'{labels[2]}  ',color=color,fontsize=fontsize,va='bottom',ha='center',rotation=-90)

    return cax

# Further fix data for all cases which have projections
def _fix_data(K, E, pros, labels, interp, rgb = False, **others):
    "Input pros must be [m,nk,nb], output is [nk,nb, m]. `others` must have shape [nk,nb] for occupancies or [nk,3] for kpoints"
    
    if np.shape(pros)[-2:] != np.shape(E):
        raise ValueError("last two dimensions of `pros` must have same shape as `E`")
    
    if np.ndim(pros) == 2:
        pros = np.expand_dims(pros,0) # still as [m,nk,nb]
        
    
    if others:
        for k,v in others.items():
            if np.shape(v)[0] != len(K):
                raise ValueError(f"{k} must have same length as K")
        
    if rgb and len(pros) > 3:
        raise ValueError("In RGB lines mode, pros.shape[-1] <= 3 should hold")
    
    # Should be after exapnding dims but before transposing
    if labels and len(labels) != len(pros):
        raise ValueError("labels must be same length as pros")
    
    pros = np.transpose(pros,(1,2,0)) # [nk,nb,m] now
    
    # Normalize overall data because colors are normalized to 0-1
    min_max_pros = (np.min(pros), np.max(pros)) # For data scales to use later
    c_max = np.ptp(pros)
    if c_max > 0.0000001: # Avoid division error
        pros = (pros - np.min(pros))/c_max
            
    data = {'kpath':K, 'evals':E, 'pros':pros, **others, 'ptp': min_max_pros}
    if interp:
        nk = interp if isinstance(interp, (list, tuple)) else (interp, 3)
        min_d, max_d = np.min(pros),np.max(pros) # For cliping
        _K, E = gu.interpolate_data(K,E,*nk)
        pros = gu.interpolate_data(K,pros,*nk)[1].clip(min=min_d,max=max_d) 
        data.update({'kpath':_K, 'evals':E, 'pros':pros})
        for k,v in others.items():
            data[k] = gu.interpolate_data(K,v,*nk)[1]
    
    # Handle kpath discontinuities
    X = data['kpath']
    breaks = [i for i in range(0,len(X)) if X[i-1] == X[i]]
    if breaks:
        data['kpath'] = np.insert(data['kpath'],breaks,np.nan)
        data['evals'] = np.insert(data['evals'],breaks,np.nan,axis=0)
        data['pros']  = np.insert(data['pros'],breaks,data['pros'][breaks],axis=0) # Repeat the same data to keep color consistent
        for key in others: # don't use items here, as interpolation may have changed the shape
            data[key] = np.insert(data[key],breaks,data[key][breaks],axis=0) # Repeat here too
    
    return data

def splot_rgb_lines(K, E, pros, labels, 
    ax         = None, 
    elim       = None, 
    kticks     = None, 
    interp     = None, 
    maxwidth   = 3,
    uniwidth   = False,
    colormap   = None,
    colorbar   = True,
    N          = 9,
    shadow     = False
    ):
    """
    Plot projected band structure for a given projections.
    
    Parameters
    ----------
    K : array-like, shape (nk,)
    E : array-like, shape (nk,nb)
    pros : array-like, shape (m,nk,nb), m is the number of projections <= 3 in rgb case.
    labels : list of str, length m
    ax : matplotlib.axes.Axes 
    elim : tuple of min and max values    
    kticks : [(int, str),...] for indices of high symmetry k-points. To join a broken path, use '<=' before symbol, e.g.  [(0, 'G'),(40, '<=K|M'), ...] will join 40 back to 39. You can also use shortcut like zip([0,10,20],'GMK').
    interp : int or list/tuple of (n,k) for interpolation. If int, n is number of points to interpolate. If list/tuple, n is number of points and k is the order of spline.
    maxwidth : float, maximum linewidth, 3 by default
    uniwidth : bool, if True, use same linewidth for all patches to maxwidth/2. Otherwise, use linewidth proportional to projection value.
    colormap : str, name of a matplotlib colormap
    colorbar : bool, if True, add colorbar, otherwise add attribute to ax to add colorbar or color cube later
    N : int, number of colors in colormap
    shadow : bool, if True, add shadow to lines
    
    Returns
    ------- 
    ax : matplotlib.axes.Axes which has additional attributes:
        .add_colorbar() : Add colorbar that represents most recent plot
        .color_cube()   : Add color cube that represents most recent plot if `pros` is 3 components
    """
    K, E, xticks, xticklabels = _validate_data(K,E,elim,kticks, interp)
    
    ax  = get_axes() if ax is None else ax
    
    #=====================================================
    pros_data = _fix_data(K, E, pros, labels, interp, rgb = True) # (nk,), (nk, nb), (nk, nb, m) at this point
    colors = pros_data['pros']
    how_many = np.shape(colors)[-1]

    if how_many == 1:
        percent_colors = colors[:,:,0]
        percent_colors = percent_colors/np.max(percent_colors)
        pros_data['pros'] = plt.cm.get_cmap(colormap or 'copper',N)(percent_colors)[:,:,:3] # Get colors in RGB space.

    elif how_many == 2:
        _sum = np.sum(colors,axis=2)
        _sum[_sum == 0] = 1 # Avoid division error
        percent_colors = colors[:,:,1]/_sum # second one is on top
        _colors = plt.cm.get_cmap(colormap or 'coolwarm',N)(percent_colors)[:,:,:3] # Get colors in RGB space.
        _colors[np.sum(colors,axis=2) == 0] = [0,0,0] # Set color to black if no total projection
        pros_data['pros'] = _colors

    else:
        # Normalize color at each point only for 3 projections.
        c_max = np.max(colors,axis=2, keepdims =True)
        c_max[c_max == 0] = 1 #Avoid division error:
        colors = colors/c_max # Weights to be used for color interpolation.

        nsegs = np.linspace(0,1,N,endpoint = True)
        for low,high in zip(nsegs[:-1],nsegs[1:]):
            colors[(colors >= low) & (colors < high)] = (low + high)/2 # Center of squre is taken in color_cube

        A, B, C = plt.cm.get_cmap(colormap or 'brg',N)([0,0.5,1])[:,:3]
        pros_data['pros'] = np.array([
            [(r*A + g*B + b*C)/((r + g + b) or 1) for r,g,b in _cols]
            for _cols in colors
        ])

        # Normalize after picking colors from colormap as well to match the color_cube.
        c_max = np.max(pros_data['pros'],axis=2, keepdims= True)
        c_max[c_max == 0] = 1 #Avoid division error:

        pros_data['pros'] = pros_data['pros']/c_max

    line_coll, = _make_line_collection(**pros_data,rgb=True,colors_list= None, maxwidth = maxwidth, shadow = shadow, uniwidth = uniwidth)
    ax.add_collection(line_coll)
    ax.autoscale_view()
    modify_axes(ax,xticks = xticks,xt_labels = xticklabels,xlim = [min(K), max(K)], ylim = elim, vlines = True, top=True, right=True)
    #====================================================
    
    # Add colorbar/legend etc.
    cmap = colormap or ('copper' if how_many == 1 else 'brg' if how_many == 3 else 'coolwarm')
    ticks = np.linspace(*pros_data['ptp'],5, endpoint=True) if how_many == 1 else None if how_many == 3 else [0,1]
    ticklabels = [f'{t:4.2f}' for t in ticks] if how_many == 1 else labels
    
    if colorbar:
        if how_many < 3:
            cax = add_colorbar(ax, N = N, vertical=True,ticklabels = ticklabels, ticks=ticks,cmap_or_clist = cmap)
            if how_many == 1:
                cax.set_title(labels[0])
        else:
            color_cube(ax,colormap = colormap or 'brg', labels = labels, N = N)
    else:
        # MAKE PARTIAL COLOR CUBE AND COLORBAR HERE FOR LATER USE.
        def recent_colorbar(cax=None,tickloc='right',vertical=True,digits=2,fontsize=8):
            return add_colorbar(ax = ax, cax=cax, cmap_or_clist = cmap, N = N,
                    ticks = ticks, ticklabels = ticklabels, tickloc = tickloc,
                    vertical=vertical,digits=digits,fontsize=fontsize)

        ax.add_colorbar = recent_colorbar

        def recent_color_cube(loc = (0.67,0.67), size=0.3 ,color='k',fontsize = 10):
            return color_cube(ax = ax,colormap = cmap, labels = labels, N = N,
                    loc =loc, size = size, color = color, fontsize = fontsize)

        ax.color_cube = recent_color_cube

    return ax


def splot_color_lines(K, E, pros, labels, 
    axes       = None, 
    elim       = None, 
    kticks     = None, 
    interp     = None, 
    maxwidth   = 3,
    colormap   = None,
    shadow     = False,
    showlegend = True,
    xyc_label   = [0.2, 0.85, 'black'], # x, y, color only if showlegend = False
    **kwargs
    ):
    """
    Plot projected band structure for a given projections.
    
    Parameters
    ----------
    K : array-like, shape (nk,)
    E : array-like, shape (nk,nb)
    pros : array-like, shape (m,nk,nb), m is the number of projections
    labels : list of str, length m
    axes : matplotlib.axes.Axes or list of Axes equal to the number of projections to plot separately. If None, create new axes.
    elim : tuple of min and max values    
    kticks : [(int, str),...] for indices of high symmetry k-points. To join a broken path, use '<=' before symbol, e.g.  [(0, 'G'),(40, '<=K|M'), ...] will join 40 back to 39. You can also use shortcut like zip([0,10,20],'GMK').
    interp : int or list/tuple of (n,k) for interpolation. If int, n is number of points to interpolate. If list/tuple, n is number of points and k is the order of spline.
    maxwidth : float, maximum linewidth, 3 by default
    colormap : str, name of a matplotlib colormap
    shadow : bool, if True, add shadow to lines
    showlegend : bool, if True, add legend, otherwise adds a label to the plot.
    xyc_label : list of x, y, color for the label. Used only if showlegend = False
    
    kwargs are passed to matplotlib's command `ax.legend`.
    
    Returns
    ------- 
    ax : matplotlib.axes.Axes. Use ax.subplots_adjust to adjust the plot if needed.
    """
    K, E, xticks, xticklabels = _validate_data(K, E, elim, kticks, interp)
    pros_data = _fix_data(K, E, pros, labels, interp, rgb = False) 
    
    if colormap not in plt.colormaps():
        c_map = plt.cm.get_cmap('viridis')
        print("colormap = {!r} not exists, falling back to default color map.".format(colormap))
    else:
        c_map = plt.cm.get_cmap(colormap)
    c_vals = np.linspace(0,1,pros_data['pros'].shape[-1]) # Output pros data has shape (nk, nb, projections)
    colors  = c_map(c_vals)
    
    if not np.any([axes]):
        axes = get_axes()
    axes = np.array([axes]).ravel() # Safe list any axes size
    if len(axes) == 1:
        axes = [axes[0] for _ in range(pros_data['pros'].shape[-1])] # Make a list of axes for each projection
    elif len(axes) != pros_data['pros'].shape[-1]:
        raise ValueError("Number of axes should be 1 or same as number of projections")
    
    lcs = _make_line_collection(maxwidth = maxwidth, colors_list=colors, rgb = False, shadow=shadow, **pros_data)
    _ = [ax.add_collection(lc) for ax, lc in zip(axes,lcs)]
    _ = [ax.autoscale_view() for ax in axes]
    
    if showlegend:
        # Default values for legend_kwargs are overwritten by **kwargs
        legend_kwargs = {'ncol': 4, 'anchor': (0, 1.05), 'handletextpad': 0.5, 'handlelength': 1,'fontsize': 'small', 'frameon': False, **kwargs}
        add_legend(ax=axes[0],colors = colors,labels = labels,widths = maxwidth, **legend_kwargs)
        
    else:
        xs, ys, colors = xyc_label
        _ = [add_text(ax, xs = xs, ys = ys, colors = colors, txts = lab) for ax, lab in zip(axes, labels)]
    
    _ = [modify_axes(ax=ax,xticks=xticks,xt_labels=xticklabels,xlim=[min(K),max(K)],ylim = elim,vlines = True,top=True, right=True) for ax in axes]
    return axes
    

def _fix_dos_data(energy, dos_arrays, labels, colors, interp):
    if colors is not None:
        if len(colors) != len(labels):
            raise ValueError(
                "If colors is given,they must have same length as labels.")
    if len(dos_arrays) != len(labels):
        raise ValueError("dos_arrays and labels must have same length.")
    
    for i, arr in enumerate(dos_arrays):
        if len(energy) != len(arr):
            raise ValueError(f"array {i+1} in dos_arrays must have same length as energy.")
    if len(dos_arrays) < 1:
        raise ValueError("dos_arrays must have at least one array.")
        
    if interp and not isinstance(interp, (int, np.integer,list, tuple)):
        raise ValueError("interp must be an integer or a list/tuple of (n,k).")
    
    if isinstance(interp, (list,tuple)) and len(interp) != 2:
        raise ValueError("interp must be an integer or a list/tuple of (n,k).")
    
    
    if interp:
        nk = interp if isinstance(interp,(list,tuple)) else (interp,3) # default spline order is 3.
        en, arr1 = gu.interpolate_data(energy, dos_arrays[0], nk)
        arrays = [arr1]
        for a in dos_arrays[1:]:
            arrays.append(gu.interpolate_data(energy, a, nk)[1])

        return en, arrays, labels, colors
        
    return energy, dos_arrays, labels, colors
    

def splot_dos_lines(energy, dos_arrays, labels,
    ax = None,
    elim = None,
    colormap = 'tab10',
    colors = None,
    fill = True,
    vertical = False,
    stack = False,
    interp = None,
    showlegend = True,
    legend_kwargs = {
        'ncol': 4, 'anchor': (0, 1.0),
        'handletextpad' : 0.5,'handlelength' : 1,
        'fontsize' : 'small','frameon' : False
    },
    **kwargs):
    """
    Plot density of states (DOS) lines.
    
    Parameters
    ----------
    energy : array-like, shape (n,)
    dos_arrays : list of array_like, each of shape (n,) or array-like (m,n)
    labels : list of str, length = len(dos_arrays) should hold.
    ax : matplotlib.axes.Axes
    elim : list of length 2, (emin, emax), if None, (min(energy), max(energy)) is used.
    colormap : str, default 'tab10', any valid matplotlib colormap name.
    colors : list of str, length = len(dos_arrays) should hold if given, and will override colormap.
    fill : bool, default True, if True, fill the area under the DOS lines.
    vertical : bool, default False, if True, plot DOS lines vertically.
    stack : bool, default False, if True, stack the DOS lines. Only works for horizontal plots.
    interp : int or list/tuple of (n,k), default None, if given, interpolate the DOS lines using spline.
    showlegend : bool, default True, if True, show legend.
    legend_kwargs : dict, default {'ncol': 4, 'anchor': (0, 1.0), 'handletextpad' : 0.5,'handlelength' : 1,'fontsize' : 'small','frameon' : False}, only used if showlegend is True.
    
    keyword arguments are passed to matplotlib.axes.Axes.plot or matplotlib.axes.Axes.fill_between or matplotlib.axes.Axes.fill_betweenx.
    
    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    energy, dos_arrays, labels, colors = _fix_dos_data(energy, dos_arrays, labels, colors, interp) # validate data brfore plotting.
    
    if colors is None:
        colors = plt.cm.get_cmap(colormap)(np.linspace(0,1,len(labels)))
    
    if ax is None:
        ax = get_axes()
        
    if 'c' in kwargs:
        kwargs.pop('c')
    if 'color' in kwargs:
        kwargs.pop('color')
          
    if stack:
        if vertical:
            raise NotImplementedError("stack is not supported for vertical plots.")
        else:
            ax.stackplot(energy, *dos_arrays, labels = labels, colors = colors, **kwargs)
    else:
        for arr, label, color in zip(dos_arrays, labels, colors):
            if fill:
                fill_func = ax.fill_betweenx if vertical else ax.fill_between
                fill_func(energy, arr,color = mpl.colors.to_rgba(color,0.4))
            if vertical:
                ax.plot(arr, energy, label = label, color = color, **kwargs) 
            else:
                ax.plot(energy, arr, label = label, color = color, **kwargs)
        
    if showlegend:
        add_legend(ax, **legend_kwargs) # Labels are picked from plot
    
    args = dict(ylim = elim or []) if vertical else dict(xlim = elim or [])
    xlabel, ylabel = 'Energy (eV)', 'DOS'
    if vertical:
        xlabel, ylabel = ylabel, xlabel
    modify_axes(ax, xlabel = xlabel, ylabel = ylabel, zeroline = False, **args)
    return ax
        
def plt2html(plt_fig = None,transparent = True):
    """
    - Returns base64 encoded Image to display in notebook or HTML <svg> or plotly's dash_html_components.Img object.
    Args:
        - plt_fig    : Matplotlib's figure instance, auto picks as well.
        - transparent: True of False for fig background.
    """
    if plt_fig is None:
        plt_fig = plt.gcf()
    plot_bytes = BytesIO()
    plt.savefig(plot_bytes,format='svg',transparent = transparent)

    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell' or shell=='Shell': #Shell for Colab. Don't know why Google ...
            _ = plt.clf() # Clear other display
            return HTML('<svg' + plot_bytes.getvalue().decode('utf-8').split('<svg')[1])
    except:
        _ = plt.clf() # Clear image
        return '<svg' + plot_bytes.getvalue().decode('utf-8').split('<svg')[1]
    else:
        return plt.show()


def show(transparent = False):
    """Displays all available figures in browser without blocking terminal"""
    for i in plt.get_fignums():
        svg = plt2html(plt.figure(i),transparent = transparent,dash_html=False)
        html_str= """\
<!DOCTYPE html>
<head></head>
<body>
    <div>
    {}
    </div>
</body>
""".format(svg)
        open_html_in_browser(html_str)
        del svg, html_str

def savefig(filename, dpi=600,**kwargs):
    """Save matplotlib's figure while handling existing files. `kwargs` are passed to `plt.savefig`"""
    #Avoids File Overwrite
    plt.savefig(gu.prevent_overwrite(filename),dpi=dpi,**kwargs)

def plt2text(plt_fig=None,width=144,vscale=0.96,colorful=True,invert=False,crop=False,outfile=None):
    """Displays matplotlib figure in terminal as text. You should use a monospcae font like `Cascadia Code PL` to display image correctly. Use before plt.show().
    Args:
        - plt_fig: Matplotlib's figure instance. Auto picks if not given.
        - width  : Character width in terminal, default is 144. Decrease font size when width increased.
        - vscale : Useful to tweek aspect ratio. Default is 0.96 and prints actual aspect in `Cascadia Code PL`. It is approximately `2*width/height` when you select a single space in terminal.
        - colorful: Default is False, prints colored picture if terminal supports it, e.g Windows Terminal.
        - invert  : Defult is False, could be useful for grayscale image.
        - crop    : Default is False. Crops extra background, can change image color if top left pixel is not in background, in that case set this to False.
        - outfile: If None, prints to screen. Writes on a file.
    """
    if plt_fig==None:
        plt_fig = plt.gcf()
    plot_bytes = BytesIO()
    plt.savefig(plot_bytes,format='png',dpi=600)
    img = PIL.Image.open(plot_bytes)
    # crop
    if crop:
        bg   = PIL.Image.new(img.mode, img.size, img.getpixel((0,0)))
        diff = PIL.ImageChops.difference(img, bg)
        diff = PIL.ImageChops.add(diff, diff, 2.0, -100) # No idea how it works
        bbox = diff.getbbox()
        img  = img.crop(bbox)

    w, h   = img.size
    aspect = h/w
    height = np.ceil(aspect * width * vscale).astype(int) # Integer
    height = height if height % 2 == 0 else height + 1 #Make even. important

    if colorful:
        img  = img.resize((width, height)).convert('RGB')
        data = np.reshape(img.getdata(),(height,width,-1))[...,:3]
        data = 225 - data if invert else data #Inversion
        fd   = data[:-1:2,...] #Foreground
        bd   = data[1::2,...]  # Background
        # Upper half block is forground and lower part is background, so one spot make two pixels.
        d_str  = "\033[48;2;{};{};{}m\033[38;2;{};{};{}m\u2580\033[00m" #Upper half block
        pixels = [[d_str.format(*v1,*v2) for v1,v2 in zip(b,f)] for b,f in zip(bd,fd)]

    else:
        height = int(height/2) #
        chars  = ['.',':',';','+','*','?','%','S','#','@']
        chars  = chars[::-1] if invert else chars #Inversion
        img    = img.resize((width, height)).convert('L') # grayscale
        pixels = [chars[int(v*len(chars)/255) -1] for v in img.getdata()]
        pixels = np.reshape(pixels,(height,-1)) #Make row/columns

    out_str = '\n'.join([''.join([p for p in ps]) for ps in pixels])

    if outfile:
        with open(outfile,'w', encoding='utf-8') as f: # unicode
            f.write(out_str)
    else:
        # For loop is important for printing lines, otherwise breaks appear.
        for line in out_str.splitlines():
            print(line)

def plot_potential(
    basis = None,
    values = None,
    operation = 'mean_c',
    ax=None,
    period = None,
    period_right = None,
    interface = None,
    lr_pos=(0.25,0.75),
    smoothness = 2,
    labels=(r'$V(z)$',r'$\langle V \rangle _{roll}(z)$',r'$\langle V \rangle $'),
    colors = ((0,0.2,0.7),'b','r'),
    annotate = True
    ):
    """
    - Returns tuple(ax,Data) where Data contains resultatnt parameters of averaged potential of LOCPOT.
    Args:
        - values : `epxort_potential().values` is 3D grid data. As `epxort_potential` is slow, so compute it once and then plot the output data.
        - operation: Default is 'mean_c'. What to do with provided volumetric potential data. Anyone of these 'mean_a','min_a','max_a','mean_b','min_b','max_b','mean_c','min_c','max_c'.
        - ax: Matplotlib axes, if not given auto picks.
        - period: Periodicity of potential in fraction between 0 and 1. For example if a slab is made of 4 super cells in z-direction, period=0.25.
        - period_right: Periodicity of potential in fraction between 0 and 1 if right half of slab has different periodicity.
        - lr_pos: Locations around which averages are taken.Default (0.25,0.75). Provide in fraction between 0 and 1. Center of period is located at these given fractions. Work only if period is given.
        - interface: Default is 0.5 if not given, you may have slabs which have different lengths on left and right side. Provide in fraction between 0 and 1 where slab is divided in left and right halves.
        - smoothness: Default is 3. Large value will smooth the curve of potential. Only works if period is given.
        - labels: List of three labels for legend. Use plt.legend() or pp.add_legend() for labels to appear. First entry is data plot, second is its convolution and third is complete average.
        - colors: List of three colors for lines.
        - annotate: True by default, writes difference of right and left averages on plot.
    """
    check = ['mean_a','min_a','max_a','mean_b','min_b','max_b','mean_c','min_c','max_c']
    if operation not in check:
        raise ValueError("`operation` excepts any of {}, got {}".format(check,operation))
    if ax is None:
        ax = get_axes()
    if values is None or basis is None:
        print('`values` or `basis` not given, trying to autopick LOCPOT...')
        try:
            ep = vp.export_locpot()
            basis = ep.poscar.basis
            values = ep.values
        except:
            raise Exception('Could not auto fix. Make sure `basis` and `v` are provided.')
    x_ind = 'abc'.index(operation.split('_')[1])
    other_inds = tuple([i for i in [0,1,2] if i != x_ind])
    _func_ = np.min if 'min' in operation else np.max if 'max' in operation else np.mean
    pot = _func_(values, axis= other_inds)

    # Direction axis
    x = np.linalg.norm(basis[x_ind])*np.linspace(0,1,len(pot),endpoint = False) # VASP does not include last point, it is same as firts one
    ax.plot(x,pot,lw=0.8,c=colors[0],label=labels[0]) #Potential plot
    ret_dict = {'direction':operation.split('_')[1]}
    # Only go below if periodicity is given
    if period == None:
        return (ax,serializer.Dict2Data(ret_dict)) # Simple Return
    if period != None:
        arr_con = gu.rolling_mean(pot,period,period_right = period_right,interface = interface, mode= 'wrap',smoothness=smoothness)
        x_con =  np.linspace(0,x[-1],len(arr_con),endpoint = False)
        ax.plot(x_con,arr_con,linestyle='dashed',lw=0.7,label=labels[1],c=colors[1]) # Convolved plot
        # Find Averages
        left,right = lr_pos
        ind_1 = int(left*len(pot))
        ind_2 = int(right*len(pot))
        x_1, v_1 = x_con[ind_1], arr_con[ind_1]
        x_2, v_2 = x_con[ind_2], arr_con[ind_2]

        ret_dict.update({'left':{'y':float(v_1),'x':float(x_1)}})
        ret_dict.update({'right':{'y':float(v_2),'x':float(x_2)}})
        ret_dict.update({'deltav':float(v_2 - v_1)})
        #Level plot
        ax.step([x_1, x_2],[v_1, v_2],lw = 0.7,where='mid',marker='.',markersize=5, color=colors[2],label=labels[2])
        # Annotate
        if annotate == True:
            ax.text(0.5,0.07,r'$\Delta _{R,L} = %9.6f$'%(np.round(v_2-v_1,6)),ha="center", va="center",
                            bbox=dict(edgecolor='white',facecolor='white', alpha=0.5),transform=ax.transAxes)
        ax.set_xlabel('$'+ret_dict['direction']+' ('+u'\u212B'+')$')
        ax.set_xlim([x[0],x[-1]])
        return (ax,serializer.Dict2Data(ret_dict))