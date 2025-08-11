from uuid import uuid1
from io import BytesIO

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mplc

from matplotlib import tri
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap as LSC
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from cycler import cycler

from IPython import get_ipython
from IPython.display import HTML
import PIL  # For text image.

import plotly.graph_objects as go
from plotly.io._base_renderers import open_html_in_browser
from einteract import patched_plotly

from .spatial_toolkit import to_R3, rotation
from ..utils import _sig_kwargs


def global_matplotlib_settings(rcParams={}, display_format="svg"): 
    "Set global matplotlib settings for notebook."
    if ip := get_ipython():
        ip.run_line_magic("config", f"InlineBackend.figure_formats = ['{display_format}', 'svg', 'retina','png','jpeg']")
        
    # Gloabal settings matplotlib with some defaults
    rcParams = {
        "axes.linewidth": 0.4,
        "axes.axisbelow": True,
        "font.serif": "STIXGeneral",
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "figure.dpi": 144,  # Better to See
        "figure.figsize": [3.4, 2.8],
        "axes.prop_cycle": cycler(
            color=[
                "#636EFA",
                "#EF553B",
                "#00CC96",
                "#AB63FA",
                "#FFA15A",
                "#19D3F3",
                "#FF6692",
                "#B6E880",
                "#FF97FF",
                "#FECB52",
            ]
        ),
        **rcParams,
    }
    mpl.rcParams.update(rcParams)  # Update rcParams


class _Arrow3D(FancyArrowPatch):
    """Draw 3D fancy arrow."""

    def __init__(self, x, y, z, u, v, w, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = [x, x + u], [y, y + v], [z, z + w]

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(
            xs3d, ys3d, zs3d, self.axes.M
        )  # renderer>M for < 3.4 but we don't need it
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):  # For matplotlib >= 3.5
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return min(zs)

    def on(self, ax):
        ax.add_artist(self)


def quiver3d(X, Y, Z, U, V, W, ax=None, C="r", L=0.7, mutation_scale=10, **kwargs):
    """Plots 3D arrows on a given ax.
    See `FancyArrowPatch <https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.patches.FancyArrowPatch.html>`_ for more details.

    ``X, Y, Z`` should be 1D arrays of coordinates of arrows tail point.

    ``U, V, W`` should be 1D arrays of dx,dy,dz of arrows.

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        3D axes, if not given, auto created.
    C : array_like
        1D colors array mapping for arrows. Could be one color.
    L : array_like
        1D linwidths array mapping for arrows. Could be one linewidth.
    mutation_scale: float
        Arrow head width/size scale. Default is 10.


    kwargs are passed to ``FancyArrowPatch`` excluding `positions`, `color`, `lw`, `mutation_scale`,
    `shrinkA`, `shrinkB` which are already used. An important keyword argument is `arrowstyle`
    which could be any style valid in matplotlib.
    """
    if not ax:
        ax = get_axes(figsize=(3.4, 3.4), axes_3d=True)  # Same aspect ratio.
    if not isinstance(C, (list, tuple, np.ndarray)):
        C = [[*mplc.to_rgb(C)] for x in X]
    C = np.array(C)  # Safe for list

    if not isinstance(L, (list, tuple, np.ndarray)):
        L = [L for x in X]
    args_dict = dict(mutation_scale=mutation_scale, shrinkA=0, shrinkB=0)
    for x, y, z, u, v, w, c, l in zip(X, Y, Z, U, V, W, C, L):
        _Arrow3D(x, y, z, u, v, w, color=c, lw=l, **args_dict, **kwargs).on(ax)

    return ax


def get_axes(
    figsize=(3.4, 2.6),
    nrows=1,
    ncols=1,
    widths=[],
    heights=[],
    axes_off=[],
    axes_3d=[],
    sharex=False,
    sharey=False,
    azim=45,
    elev=15,
    ortho3d=True,
    **subplots_adjust_kwargs,
):
    """Returns axes of initialized figure, based on plt.subplots().
    If you want to access parent figure, use ax.get_figure() or current figure as plt.gcf().

    Parameters
    ----------
    figsize : tuple
        Tuple (width, height). Default is (3.4,2.6).
    nrows : int
        Default 1.
    ncols : int
        Default 1.
    widths : list
        List with len(widths)==nrows, to set width ratios of subplots.
    heights : list
        List with len(heights)==ncols, to set height ratios of subplots.
    sharex : bool
        Share x-axis between plots, this removes shared ticks automatically.
    sharey : bool
        Share y-axis between plots, this removes shared ticks automatically.
    axes_off : bool or list
        Turn off axes visibility, If `nrows = ncols = 1, set True/False`.
        If anyone of `nrows or ncols > 1`, provide list of axes indices to turn off.
        If both `nrows and ncols > 1`, provide list of tuples (x_index,y_index) of axes.
    axes_3d : bool or list
        Change axes to 3D. If `nrows = ncols = 1, set True/False`.
        If anyone of `nrows or ncols > 1`, provide list of axes indices to turn off.
        If both `nrows and ncols > 1`, provide list of tuples (x_index,y_index) of axes.
    ortho3d : bool
        Only works for 3D axes. If True, x,y,z are orthogonal, otherwise perspective.


    `azim, elev` are passed to `ax.view_init`. Defualt values are 45,15 respectively.

    `subplots_adjust_kwargs` are passed to `plt.subplots_adjust`.

    .. note::
        There are extra methods added to each axes (only 2D) object.
        ``add_text``, ``add_legend``, ``add_colorbar``, ``color_wheel``,
        ``break_spines``, ``adjust_axes``, ``append_axes``, ``join_axes``.
    """
    if figsize[0] <= 2.38:
        mpl.rc("font", size=8)
    gs_kw = dict({})  # Define Empty Dictionary.
    if widths != [] and len(widths) == ncols:
        gs_kw = dict({**gs_kw, "width_ratios": widths})
    if heights != [] and len(heights) == nrows:
        gs_kw = dict({**gs_kw, "height_ratios": heights})
    fig, axs = plt.subplots(
        nrows, ncols, figsize=figsize, gridspec_kw=gs_kw, sharex=sharex, sharey=sharey
    )
    proj = {"proj_type": "ortho"} if ortho3d else {}  # For 3D only
    if nrows * ncols == 1:
        adjust_axes(ax=axs)
        if axes_off == True:
            axs.set_axis_off()
        if axes_3d == True:
            pos = axs.get_position()
            axs.remove()
            axs = fig.add_axes(pos, projection="3d", azim=azim, elev=elev, **proj)
            setattr(axs, add_legend.__name__, add_legend.__get__(axs, type(axs)))

    else:
        _ = [adjust_axes(ax=ax) for ax in axs.ravel()]
        _ = [axs[inds].set_axis_off() for inds in axes_off if axes_off != []]
        if axes_3d != []:
            for inds in axes_3d:
                pos = axs[inds].get_position()
                axs[inds].remove()
                axs[inds] = fig.add_axes(
                    pos, projection="3d", azim=azim, elev=elev, **proj
                )
    try:
        for ax in np.array([axs]).flatten():
            for f in [
                add_text,
                add_legend,
                add_colorbar,
                color_wheel,
                color_cube,
                break_spines,
                adjust_axes,
                append_axes,
            ]:
                if ax.name != "3d":
                    setattr(ax, f.__name__, f.__get__(ax, type(ax)))
    except:
        pass

    plt.subplots_adjust(**subplots_adjust_kwargs)
    return axs


def adjust_axes(
    ax=None,
    xticks=[],
    xticklabels=[],
    xlim=[],
    yticks=[],
    yticklabels=[],
    ylim=[],
    xlabel=None,
    ylabel=None,
    vlines=False,
    **kwargs,
):
    """
    Applies given settings on axes obect and returns None.

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        Matplotlib axes object on which settings are applied.
    vlines : bool
        If True, draw vertical lines at points of xticks.


    Other parameters are well known matplotlib parameters.

    kwargs are passed to `ax.tick_params`
    """
    if ax is None:
        raise ValueError("Matplotlib axes (ax) is not given.")
    else:
        if xticks:
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels if xticklabels else list(map(str, xticks)))
        if yticks:
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels if yticklabels else list(map(str, yticks)))
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        if vlines:
            ax.xaxis.grid(color=(0, 0, 0, 0.6), linestyle="dashed", lw=0.3)

        if xlabel != None:
            ax.set_xlabel(xlabel)
        if ylabel != None:
            ax.set_ylabel(ylabel)
        kwargs = {
            **dict(
                direction="in",
                bottom=True,
                left=True,
                length=4,
                width=0.3,
                grid_alpha=0.8,
            ),
            **kwargs,
        }  # Default kwargs
        ax.tick_params(**kwargs)
        ax.set_axisbelow(True)  # Aoid grid lines on top of plot.
    return None


def append_axes(
    ax, position="right", size=0.2, pad=0.1, sharex=False, sharey=False, **kwargs
):
    """Append an axes to the given `ax` at given `position` top,right,left,bottom. Useful for adding custom colorbar.
    kwargs are passed to `mpl_toolkits.axes_grid1.make_axes_locatable.append_axes`.

    Returns appended axes.
    """
    extra_args = {}
    if sharex:
        extra_args["sharex"] = ax
    if sharey:
        extra_args["sharey"] = ax
    divider = make_axes_locatable(ax)
    added_ax = divider.append_axes(
        position=position, size=size, pad=pad, **extra_args, **kwargs
    )
    _ = adjust_axes(ax=added_ax)  # tweaks of styles
    return added_ax


def join_axes(ax1, ax2, **kwargs):
    """Join two axes together. Useful for adding custom colorbar on a long left axes of whole figure.
    Apply tight_layout() before calling this function.
    kwargs are passed to `fig.add_axes`.
    """
    p1 = ax1.get_position()
    p2 = ax2.get_position()
    x0 = min(p1.x0, p2.x0)
    y0 = min(p1.y0, p2.y0)
    x1 = max(p1.x1, p2.x1)
    y1 = max(p1.y1, p2.y1)
    new_bbox = [x0, y0, x1 - x0, y1 - y0]
    fig = ax1.get_figure()
    ax1.remove()
    ax2.remove()
    new_ax = fig.add_axes(new_bbox, **kwargs)
    _ = adjust_axes(new_ax)
    for f in [
        add_text,
        add_legend,
        add_colorbar,
        color_wheel,
        break_spines,
        adjust_axes,
        append_axes,
    ]:
        if new_ax.name != "3d":
            setattr(new_ax, f.__name__, f.__get__(new_ax, type(new_ax)))
    return new_ax


def break_spines(ax, spines, symbol="\u2571", **kwargs):
    """Simulates broken axes using subplots. Need to fix heights according
    to given data for real aspect. Also plot the same data on each axes and set axes limits.

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        Axes who's spine(s) to edit.
    spines : str or list
        str/list of any of ['top','bottom','left','right'].
    symbol: str
        Defult is u'\u2571'. Its at 60 degrees. so you can apply rotation to make it any angle.


    kwargs are passed to plt.text.
    """
    kwargs.update(transform=ax.transAxes, ha="center", va="center")
    _spines = [spines] if isinstance(spines, str) else spines
    _ = [ax.spines[s].set_visible(False) for s in _spines]
    ax.tick_params(**{sp: False for sp in _spines})
    if "top" in spines:
        ax.text(0, 1, symbol, **kwargs)
        ax.text(1, 1, symbol, **kwargs)
    if "bottom" in spines:
        ax.set_xticks([])
        ax.text(0, 0, symbol, **kwargs)
        ax.text(1, 0, symbol, **kwargs)
    if "left" in spines:
        ax.set_yticks([])
        ax.text(0, 0, symbol, **kwargs)
        ax.text(0, 1, symbol, **kwargs)
    if "right" in spines:
        ax.text(1, 1, symbol, **kwargs)
        ax.text(1, 0, symbol, **kwargs)


def add_text(
    ax=None, xs=0.25, ys=0.9, txts="[List]", colors="r", transform=True, **kwargs
):
    """Adds text entries on axes, given single string or list.

    Parameters
    ----------
    xs : float or array_like
        List of x coordinates relative to axes or single coordinate.
    ys : float or array_like
        List of y coordinates relative to axes or single coordinate.
    txts : str or array_like
        List of strings or one string.
    colors : array_like or Any
        List of colors of txts or one color.
    transform : bool
        Dafault is True and positions are relative to axes, If False, positions are in data coordinates.


    kwargs are passed to plt.text.
    """
    if ax == None:
        raise ValueError("Matplotlib axes (ax) is not given.")
    else:
        bbox = kwargs.get("bbox", dict(edgecolor="white", facecolor="white", alpha=0.4))
        ha, va = kwargs.get("ha", "center"), kwargs.get("va", "center")
        args_dict = dict(bbox=bbox, ha=ha, va=va)
        if transform:
            args_dict.update({"transform": ax.transAxes})

        if isinstance(txts, str):
            ax.text(xs, ys, txts, color=colors, **args_dict, **kwargs)
        elif isinstance(txts, (list, np.ndarray)):
            for x, y, (i, txt) in zip(xs, ys, enumerate(txts)):
                try:
                    ax.text(x, y, txt, color=colors[i], **args_dict, **kwargs)
                except:
                    ax.text(x, y, txt, color=colors, **args_dict, **kwargs)


def add_legend(
    ax=None,
    colors=[],
    labels=[],
    styles="solid",
    widths=0.7,
    anchor=(0, 1),
    ncol=3,
    loc="lower left",
    fontsize="small",
    frameon=False,
    **kwargs,
):
    """
    Adds custom legeneds on a given axes, returns None.

    Parameters
    ----------
    ax : Matplotlib axes.
    colors : List of colors.
    labels : List of labels.
    styles : str or list of line styles.
    widths : str or list of line widths.

    kwargs are passed to plt.legend. Given arguments like anchor,ncol etc are preferred.
    """
    kwargs.update(
        dict(
            bbox_to_anchor=anchor,
            ncol=ncol,
            loc=loc,
            fontsize=fontsize,
            frameon=frameon,
        )
    )
    if ax == None:
        raise ValueError("Matplotlib axes (ax) is not given.")
    else:
        if type(widths) == float or type(widths) == int:
            if type(styles) == str:
                for color, label in zip(colors, labels):
                    ax.plot(
                        [], [], color=color, lw=widths, linestyle=styles, label=label
                    )
            else:
                for color, label, style in zip(colors, labels, styles):
                    ax.plot(
                        [], [], color=color, lw=widths, linestyle=style, label=label
                    )
        else:
            if type(styles) == str:
                for color, label, width in zip(colors, labels, widths):
                    ax.plot(
                        [], [], color=color, lw=width, linestyle=styles, label=label
                    )
            else:
                for color, label, width, style in zip(colors, labels, widths, styles):
                    ax.plot([], [], color=color, lw=width, linestyle=style, label=label)
        ax.legend(**kwargs)
    return None


def add_colorbar(
    ax,
    cmap_or_clist=None,
    N=256,
    ticks=None,
    ticklabels=None,
    vmin=None,
    vmax=None,
    cax=None,
    tickloc="right",
    vertical=True,
    digits=2,
    fontsize=8,
):
    """
    Plots colorbar on a given axes. This axes should be only for colorbar.

    Parameters
    ----------
    ax : Matplotlib axes for which colorbar will be added.
    cmap_or_clist : List/array of colors in or colormap's name. If None (default), matplotlib's default colormap is plotted.
    N : int, number of color points Default 256.
    ticks : List of tick values to show on colorbar. To turn off, give [].
    ticklabels : List of labels for ticks.
    vmin,vmax : Minimum and maximum values. Only work if ticks are given.
    cax : Matplotlib axes for colorbar. If not given, one is created.
    tickloc : Default 'right'. Any of ['right','left','top','bottom'].
    digits : Number of digits to show in tick if ticklabels are not given.
    vertical : Boolean, default is Fasle.
    fontsize : Default 8. Adjustable according to plot space.

    Returns
    -------
    cax : Matplotlib axes for colorbar, you can customize further.
    """
    if cax is None:
        position = "right" if vertical == True else "top"
        cax = append_axes(ax, position=position, size="5%", pad=0.05)
    if cmap_or_clist is None:
        colors = np.array(
            [
                [1, 0, 1],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 1, 1],
                [0, 0, 1],
                [1, 0, 1],
            ]
        )
        _hsv_ = LSC.from_list("_hsv_", colors, N=N)
    elif isinstance(cmap_or_clist, (list, np.ndarray)):
        try:
            _hsv_ = LSC.from_list("_hsv_", cmap_or_clist, N=N)
        except Exception as e:
            print(e, "\nFalling back to default color map!")
            _hsv_ = None  # fallback
    elif isinstance(cmap_or_clist, str):
        _hsv_ = cmap_or_clist  # colormap name
    else:
        _hsv_ = None  # default fallback

    if ticks != []:
        if ticks is None:  # should be before labels
            ticks = np.linspace(1 / 6, 5 / 6, 3, endpoint=True)
            if ticklabels is None:
                ticklabels = ticks.round(digits).astype(str)

        elif isinstance(ticks, (list, tuple, np.ndarray)):
            ticks = np.array(ticks)
            _vmin = vmin if vmin is not None else np.min(ticks)
            _vmax = vmax if vmax is not None else np.max(ticks)
            if _vmin > _vmax:
                raise ValueError("vmin > vmax is not valid!")

            if ticklabels is None:
                ticklabels = ticks.round(digits).astype(str)
            # Renormalize ticks after assigning ticklabels
            ticks = (ticks - _vmin) / (_vmax - _vmin)
    else:
        ticks = []
        ticklabels = []

    c_vals = np.linspace(0, 1, N, endpoint=True).reshape((1, N))  # make 2D array

    ticks_param = dict(
        direction="out",
        pad=2,
        length=2,
        width=0.3,
        top=False,
        left=False,
        grid_color=(1, 1, 1, 0),
        grid_alpha=0,
    )
    ticks_param.update({tickloc: True})  # Only show on given side
    cax.tick_params(**ticks_param)
    if vertical == False:
        cax.imshow(
            c_vals, aspect="auto", cmap=_hsv_, origin="lower", extent=[0, 1, 0, 1]
        )
        cax.set_yticks([])
        cax.xaxis.tick_top()  # to show ticks on top by default
        if tickloc == "bottom":
            cax.xaxis.tick_bottom()  # top is by default
        cax.set_xticks(ticks)
        cax.set_xticklabels(ticklabels, rotation=0, ha="center")
        cax.set_xlim([0, 1])  # enforce limit

    if vertical == True:
        c_vals = c_vals.transpose()
        cax.imshow(
            c_vals, aspect="auto", cmap=_hsv_, origin="lower", extent=[0, 1, 0, 1]
        )
        cax.set_xticks([])
        cax.yaxis.tick_right()  # Show right by default
        if tickloc == "left":
            cax.yaxis.tick_left()  # right is by default
        cax.set_yticks(ticks)
        cax.set_yticklabels(ticklabels, rotation=90, va="center")
        cax.set_ylim([0, 1])  # enforce limit

    for tick in cax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
    for child in cax.get_children():
        if isinstance(child, mpl.spines.Spine):
            child.set_color((1, 1, 1, 0.4))
    return cax  # Return colorbar axes to perform further customization


def color_wheel(
    ax=None,
    xy=(1, 1),
    scale=0.12,
    rlim=(0.2, 1),
    N=256,
    colormap=None,
    ticks=[1 / 6, 1 / 2, 5 / 6],
    labels=["s", "p", "d"],
    showlegend=True,
):
    """
    Returns cax i.e. color wheel axes.

    Parameters
    ----------
    ax : Axes on which color wheel will be drawn. Auto created if not given.
    xy : (x,y) of center of wheel.
    scale : Scale of the cax internally generated by color wheel.
    rlim : Values in [0,1] interval, to make donut like shape.
    N : Number of segments in color wheel.
    colormap : Matplotlib's color map name. fallbacks to `hsv`.
    ticks : Ticks in fractions in interval [0,1].
    labels : Ticks labels.
    showlegend : bool, default is True.
    """
    if ax is None:
        ax = get_axes()
    if colormap is None:
        try:
            colormap = plt.cm.get_cmap("hsv")
        except:
            colormap = "viridis"
    pos = ax.get_position()
    ratio = pos.height / pos.width
    cpos = [
        pos.x0 + pos.width * xy[0] - scale / 2,
        pos.y0 + pos.height * xy[1] - scale / 2,
        scale,
        scale,
    ]
    cax = ax.get_figure().add_axes(cpos, projection="polar")
    norm = mpl.colors.Normalize(0.0, 2 * np.pi)
    t = np.linspace(0, 2 * np.pi, N)
    r = np.linspace(*rlim, 2)
    rg, tg = np.meshgrid(r, t)
    cax.pcolormesh(
        t, r, tg.T, norm=norm, cmap=colormap, edgecolor="face", shading="gouraud"
    )
    cax.set_yticklabels([])
    cax.spines["polar"].set_visible(False)
    ##########
    if showlegend == True:
        colors = plt.cm.get_cmap(colormap)(ticks)  # Get colors values.
        labels = ["◾ " + l for l in labels]
        labels[0] = (
            labels[0] + "\n"
        )  # hack to write labels correctly on a single point.
        labels[2] = "\n" + labels[2]
        for l, p, c in zip(labels, ["bottom", "center", "top"], colors):
            cax.text(
                rlim[1] + 0.1,
                0.5,
                l,
                va=p,
                ha="left",
                color=c,
                transform=cax.transAxes,
                fontsize=9,
            )
        cax.set_xticklabels([])
    else:
        cax.set_xticks([t * 2 * np.pi for t in ticks])
        cax.set_xticklabels(labels)
    return cax


def color_cube(
    ax,
    colormap="brg",
    loc=(1, 0.4),
    size=0.2,
    N=7,
    labels=["s", "p", "d"],
    color="k",
    fontsize=10,
):
    "Color-mapped hexagon that serves as a legend for `splot_rgb_lines`"
    if N < 3:
        raise ValueError("N must be >= 3 to map colors correctly.")

    X, Y = np.mgrid[0 : 1 : N * 1j, 0 : 1 : N * 1j]
    x = X.flatten()
    y = Y.flatten()
    points_z = np.array([x, y, np.ones_like(x)]).T
    points_y = np.array([x, np.ones_like(x), y]).T
    points_x = np.array([np.ones_like(x), x, y]).T

    all_points = []
    ps = to_R3([[1, 0, 0], [-0.5, -np.sqrt(3) / 2, 0], [0, 0, 1]], points_z)
    all_points.extend(ps)
    ps1 = rotation(angle_deg=-120, axis_vec=[0, 0, 1]).apply(ps + [0, np.sqrt(3), 0])
    all_points.extend(ps1)
    ps2 = rotation(angle_deg=180, axis_vec=[0, 0, 1]).apply(
        ps * [-1, 1, 0] + [0, np.sqrt(3), 0]
    )
    all_points.extend(ps2)

    all_points = rotation(-30, axis_vec=[0, 0, 1]).apply(all_points)

    pts = np.asarray(all_points)[:, :2]
    pts = pts - pts.mean(axis=0)  # center
    C = np.array([*points_z, *points_x, *points_y])

    fig = ax.get_figure()
    pos = ax.get_position()
    x0 = pos.x0 + loc[0] * pos.width
    y0 = pos.y0 + loc[1] * pos.height
    size = size * pos.width
    cax = fig.add_axes([x0, y0, size, size])

    tr1 = tri.Triangulation(*pts.T)

    # Have same color for traingles sharing hypotenuse to see box
    colors = []
    for t in tr1.triangles:
        a, b, c = C[t]  # a,b,c are the 3 points of the triangle
        mid_point = (a + b) / 2  # Right angle at c
        if np.dot(a - b, a - c) == 0:  # Right angle at a
            mid_point = (b + c) / 2
        elif np.dot(b - a, b - c) == 0:  # Right angle at b
            mid_point = (a + c) / 2

        colors.append(mid_point)

    colors = np.array(colors)

    A, B, _C = plt.cm.get_cmap(colormap)([0, 0.5, 1])[:, :3]
    _colors = np.array(
        [(r * A + g * B + b * _C) / ((r + g + b) or 1) for r, g, b in colors]
    )
    _max = _colors.max(
        axis=1, keepdims=True
    )  # Should be normalized after matching to colobar as well
    _max[_max == 0] = 1
    _colors = _colors / _max

    col = PolyCollection(
        [pts[t] for t in tr1.triangles],
        color=_colors,
        linewidth=0.1,
        edgecolor="face",
        alpha=1,
    )
    cax.add_collection(col)
    cax.autoscale_view()
    cax.set_aspect("equal")
    cax.set_facecolor([1, 1, 1, 0])
    cax.set_axis_off()

    cax.text(
        9 * np.sqrt(3) / 16,
        -9 / 16,
        "→",
        fontsize=fontsize,
        zorder=-10,
        color=color,
        rotation=-30,
        ha="center",
        va="center",
    )
    cax.text(
        -9 * np.sqrt(3) / 16,
        -9 / 16,
        "→",
        fontsize=fontsize,
        zorder=-10,
        color=color,
        rotation=210,
        ha="center",
        va="center",
    )
    cax.text(
        0,
        9 / 8,
        "→",
        fontsize=fontsize,
        zorder=-10,
        color=color,
        rotation=90,
        ha="center",
        va="center",
    )

    cax.text(
        np.sqrt(3) / 2,
        -5 / 8,
        f" {labels[0]}",
        color=color,
        fontsize=fontsize,
        va="top",
        ha="center",
        rotation=-90,
    )
    cax.text(
        -np.sqrt(3) / 2,
        -5 / 8,
        f" {labels[1]}",
        color=color,
        fontsize=fontsize,
        va="top",
        ha="center",
        rotation=-90,
    )
    cax.text(
        0,
        9 / 8,
        f"{labels[2]}  ",
        color=color,
        fontsize=fontsize,
        va="bottom",
        ha="center",
        rotation=-90,
    )

    return cax


def webshow(transparent=False):
    """Displays all available figures in browser without blocking terminal"""
    for i in plt.get_fignums():
        svg = plt2html(plt.figure(i), transparent=transparent)
        html_str = """\
<!DOCTYPE html>
<head></head>
<body>
    <div>
    {}
    </div>
</body>
""".format(
            svg
        )
        open_html_in_browser(html_str)
        del svg, html_str


def plt2text(
    plt_fig=None,
    width=144,
    vscale=0.96,
    colorful=True,
    invert=False,
    crop=False,
    outfile=None,
):
    """
    Displays matplotlib figure in terminal as text. You should use a monospcae font like `Cascadia Code PL` to display image correctly. Use before plt.show().

    Parameters
    ----------
    plt_fig : Matplotlib's figure instance. Auto picks if not given.
    width : Character width in terminal, default is 144. Decrease font size when width increased.
    vscale : Useful to tweek aspect ratio. Default is 0.96 and prints actual aspect in `Cascadia Code PL`. It is approximately `2*width/height` when you select a single space in terminal.
    colorful : Default is False, prints colored picture if terminal supports it, e.g Windows Terminal.
    invert : Defult is False, could be useful for grayscale image.
    crop : Default is False. Crops extra background, can change image color if top left pixel is not in background, in that case set this to False.
    outfile : If None, prints to screen. Writes on a file.
    """
    if plt_fig == None:
        plt_fig = plt.gcf()
    plot_bytes = BytesIO()
    plt_fig.savefig(plot_bytes, format="png", dpi=600)
    img = PIL.Image.open(plot_bytes)
    # crop
    if crop:
        bg = PIL.Image.new(img.mode, img.size, img.getpixel((0, 0)))
        diff = PIL.ImageChops.difference(img, bg)
        diff = PIL.ImageChops.add(diff, diff, 2.0, -100)  # No idea how it works
        bbox = diff.getbbox()
        img = img.crop(bbox)

    w, h = img.size
    aspect = h / w
    height = np.ceil(aspect * width * vscale).astype(int)  # Integer
    height = height if height % 2 == 0 else height + 1  # Make even. important

    if colorful:
        img = img.resize((width, height)).convert("RGB")
        data = np.reshape(img.getdata(), (height, width, -1))[..., :3]
        data = 225 - data if invert else data  # Inversion
        fd = data[:-1:2, ...]  # Foreground
        bd = data[1::2, ...]  # Background
        # Upper half block is forground and lower part is background, so one spot make two pixels.
        d_str = (
            "\033[48;2;{};{};{}m\033[38;2;{};{};{}m\u2580\033[00m"  # Upper half block
        )
        pixels = [
            [d_str.format(*v1, *v2) for v1, v2 in zip(b, f)] for b, f in zip(bd, fd)
        ]

    else:
        height = int(height / 2)  #
        chars = [".", ":", ";", "+", "*", "?", "%", "S", "#", "@"]
        chars = chars[::-1] if invert else chars  # Inversion
        img = img.resize((width, height)).convert("L")  # grayscale
        pixels = [chars[int(v * len(chars) / 255) - 1] for v in img.getdata()]
        pixels = np.reshape(pixels, (height, -1))  # Make row/columns

    out_str = " " + "\n ".join(["".join([p for p in ps]) for ps in pixels])

    if outfile:
        with open(outfile, "w", encoding="utf-8") as f:  # unicode
            f.write(out_str)
    else:
        if colorful:
            for line in out_str.splitlines():
                print(line)  # for loop to give time to termail to adjust
        else:
            print(out_str)


def plt2html(plt_fig=None, transparent=True):
    """Returns ``ipython.display.HTML(<svg of figure>)``. It clears figure after use. So ``plt.show()`` will not work after this.

    Parameters
    ----------
    plt_fig : Matplotlib's figure instance, auto picks as well.
    transparent : True of False for fig background.
    """
    if plt_fig is None:
        plt_fig = plt.gcf()
    plot_bytes = BytesIO()
    plt_fig.savefig(plot_bytes, format="svg", transparent=transparent)

    plt.close(plt_fig)  # Close to avoid auto display in notebook
    return HTML("<svg" + plot_bytes.getvalue().decode("utf-8").split("<svg")[1])


def iplot2html(fig, outfile=None, modebar=True):
    """Writes plotly's figure as HTML file or display in IPython which is accessible when online.
    It is different than plotly's `fig.to_html` as it is minimal in memory. If you need to have
    offline working file, just use `fig.write_html('file.html')` which will be larger in size.

    Parameters
    ----------
    fig : A plotly's figure object.
    outfile : Name of file to save fig. Defualt is None and show plot in Notebook.
    modebar : If True, shows modebar in graph. Default is True. Not used if saving to file.
    """
    div_id = "graph-{}".format(uuid1())
    fig_json = fig.to_json()
    # a simple HTML template
    if outfile:
        template = """<html>
        <head>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <div id='{}'></div>
            <script>
                var fig_data = {}
                Plotly.react('{}', fig_data.data, fig_data.layout);
            </script>
        </body>
        </html>"""

        # write the JSON to the HTML template
        with open(outfile, "w") as f:
            f.write(template.format(div_id, fig_json, div_id))

    else:
        if modebar == True:  # Only for docs issue
            config = "{displayModeBar: true,scrollZoom: true}"
        else:
            config = "{displayModeBar: false,scrollZoom: true}"
        template = """<div>
        <script src='https://cdn.plot.ly/plotly-latest.min.js'></script>
            <div id='{}'><!-- Plotly chart DIV --></div>
            <script>
                var data = {};
                var config = {};
                Plotly.newPlot('{}', data.data,data.layout,config);
            </script>
        </div>""".format(
            div_id, fig_json, config, div_id
        )
        return HTML(template)


def iplot2widget(fig, fig_widget=None, template=None):
    "Converts plotly's figure to FigureWidget by copying attributes and data. If fig_widget is provided, it will update it. Adds template if provided. If fig is FigureWidget, it is just returned"
    if isinstance(fig, go.FigureWidget):
        return patched_plotly(fig) # add attributes selected and clicked
    
    if not isinstance(fig, go.Figure):
        raise ValueError("fig must be instance of plotly.graph_objects.Figure")

    if fig_widget is None:
        fig_widget = go.FigureWidget()
        scene = fig.layout.scene # need to copy scane form given fig
    elif not isinstance(fig_widget, go.FigureWidget):
        raise ValueError("fig_widget must be FigureWidget")
    else:
        scene = fig_widget.layout.scene if fig_widget.data else fig.layout.scene# keep scene from widget, but if looks a new fig, keep from previous

    fig_widget.data = []  # Clear previous data
    if template is not None:
        fig.layout.template = template  # will make white flash if not done before 
    
    fig_widget.layout = fig.layout
    fig_widget.layout.scene = scene # reset scene back

    with fig_widget.batch_update():
        for data in fig.data:
            fig_widget.add_trace(data)

    return patched_plotly(fig_widget) # add attributes selected and clicked

@_sig_kwargs(plt.imshow, ('ax','X'))
def image2plt(image_or_fname, ax = None, crop = None, **kwargs):
    """Plot PIL image, numpy array or image file on given matploltib axes. 
    `crop` is list or tuple of [x0,y0,x1,y1] in [0,1] interval.
    kwargs are passed to plt.imshow."""
    if ax is None:
        ax = get_axes()
    if isinstance(image_or_fname, str):
        im_array = plt.imread(image_or_fname)
    else:
        try:
            im_array = np.asarray(image_or_fname) # PIL image to array
        except:
            raise ValueError("Not a valid PIL image or filename.")
    
    if crop:
        if not isinstance(crop, (list, tuple)):
            raise ValueError("crop must be list or tuple of [x0,y0,x1,y1]")
        if max(crop) > 1 and min(crop) < 0:
            raise ValueError("crop values must be in [0,1] interval.")
        if len(crop) != 4:
            raise ValueError("crop must be list or tuple of [x0,y0,x1,y1]")
        (x0,y0),(x1,y1) = [int(im_array.shape[0]*v) for v in crop[:2]], [int(im_array.shape[1]*v) for v in crop[2:]]
        
        im_array = im_array[y0:y1+1,x0:x1+1] # image origin is top left, so y0 is first
    
    # Some kwargs are very important to be default. User can override them.
    aspect = im_array.shape[0]/im_array.shape[1]
    kwargs = {'interpolation':'none', 'extent': [0,1,0,1], 'aspect': aspect, **kwargs}
    ax.imshow(im_array, **kwargs)
    return ax