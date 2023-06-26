import re
from collections import Iterable

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.collections import LineCollection
import plotly.graph_objects as go


# Inside packages import
from . import utils as gu
from .core.plot_toolkit import (
    adjust_axes,
    get_axes,
    add_text,
    add_legend,
    add_colorbar,
    color_cube,
)


def join_ksegments(kpath, *pairs):
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
                raise ValueError(
                    f"Indices in pair ({idx_1}, {idx_2}) are not adjacent."
                )
            path_arr[idx_2:] -= path_arr[idx_2] - path_arr[idx_1]
        path_arr = path_max * path_arr / path_arr[-1]  # Normalize to max value back
    return list(path_arr)


# This is to verify things together and make sure they are working as expected.
def _validate_data(K, E, elim, kticks, interp):
    if np.ndim(E) != 2:
        raise ValueError("E must be a 2D array.")

    if np.shape(E)[0] != len(K):
        raise ValueError("Length of first dimension of E must be equal to length of K.")

    if kticks is None:
        kticks = []

    if not isinstance(kticks, (list, tuple, zip)):
        raise ValueError(
            "kticks must be a list, tuple or zip consisting of (index, label) pairs. index must be an int or tuple of (i, i+1) to join broken path."
        )

    if isinstance(kticks, zip):
        kticks = list(kticks)  # otherwise it will be empty after first use

    for k, v in kticks:
        if not isinstance(k, (np.integer, int)):
            raise ValueError("First item of pairs in kticks must be int")
        if not isinstance(v, str):
            raise ValueError("Second item of pairs in kticks must be str.")

    pairs = [
        (k - 1, k) for k, v in kticks if v.startswith("<=")
    ]  # Join broken path at these indices
    K = join_ksegments(K, *pairs)
    inds = [k for k, _ in kticks]

    xticks = (
        [K[i] for i in inds] if inds else None
    )  # Avoid turning off xticks if no kticks given
    xticklabels = (
        [v.replace("<=", "") for _, v in kticks] if kticks else None
    )  # clean up labels

    if elim and len(elim) != 2:
        raise ValueError("elim must be a list or tuple of length 2.")

    if interp and not isinstance(interp, (int, np.integer, list, tuple)):
        raise ValueError("interp must be an integer or a list/tuple of (n,k).")

    if isinstance(interp, (list, tuple)) and len(interp) != 2:
        raise ValueError("interp must be an integer or a list/tuple of (n,k).")

    return K, E, xticks, xticklabels


_docs = dict(
    params="""
    Parameters
    ----------""",
    kticks="""kticks : list
        List of pairs [(int, str),...] for indices of high symmetry k-points. 
        To join a broken path, use '<=' before symbol, e.g.  [(0, 'G'),(40, '<=K|M'), ...] 
        will join 40 back to 39. You can also use shortcut like zip([0,10,20],'GMK').""",
    interp="""interp : int or list/tuple
        If int, n is number of points to interpolate. If list/tuple, n is number of points and k is the order of spline.""",
    return_ax="""
    Returns
    -------
    matplotlib.pyplot.Axes""",
    return_fig="""
    Returns
    -------
    plotly.graph_objects.Figure""",
    K="""K : array-like
        Array of kpoints with shape (nkpts,)""",
    E="""E : array-like
        Array of eigenvalues with shape (nkpts, nbands)""",
    ax="""ax : matplotlib.pyplot.Axes
        Matplotlib axes to plot on. If None, a new figure and axes will be created.""",
    elim="""elim : list or tuple
        A list or tuple of length 2 for energy limits.""",
    pros="""pros : array-like
        Projections of shape (m,nk,nb), m is the number of projections. m <= 3 in rgb case.""",
    labels="""labels : list
        As many labels for as projections.""",
    colormap="""colormap : str
        A valid matplotlib colormap name.""",
    maxwidth="""maxwidth : float
        Maximum linewidth to which the projections line width will be scaled. Default is 3.""",
)


@gu._fmt_doc(_docs)
def splot_bands(K, E, ax=None, elim=None, kticks=None, interp=None, **kwargs):
    """Plot band structure for a single spin channel and return the matplotlib axes which can be used to add other channel if spin polarized.
    {params}\n    {K}\n    {E}\n    {ax}\n    {elim}\n    {kticks}\n    {interp}


    kwargs are passed to matplotlib's command `plt.plot`.
    {return_ax}
    """
    K, E, xticks, xticklabels = _validate_data(K, E, elim, kticks, interp)

    if interp:
        nk = interp if isinstance(interp, (list, tuple)) else (interp, 3)
        K, E = gu.interpolate_data(K, E, *nk)

    # handle broken paths
    breaks = [i for i in range(0, len(K)) if K[i - 1] == K[i]]
    K = np.insert(K, breaks, np.nan)
    E = np.insert(E, breaks, np.nan, axis=0)

    ax = get_axes() if ax is None else ax
    if "color" not in kwargs and "c" not in kwargs:
        kwargs["color"] = "C0"  # default color from cycler to accommodate themes

    if "linewidth" not in kwargs and "lw" not in kwargs:
        kwargs["linewidth"] = 0.9  # default linewidth to make it look good

    lines = ax.plot(K, E, **kwargs)
    _ = [line.set_label(None) for line in lines[1:]]

    adjust_axes(
        ax=ax,
        ylabel="Energy (eV)",
        xticks=xticks,
        xticklabels=xticklabels,
        xlim=[min(K), max(K)],
        ylim=elim,
        vlines=True,
        top=True,
        right=True,
    )
    return ax


def _make_line_collection(
    maxwidth=3, colors_list=None, rgb=False, shadow=False, uniwidth=False, **pros_data
):
    """
    Returns a tuple of line collections for each given projection data.

    Parametrs
    ---------
    maxwidth  : Default is 3. Max linewidth is scaled to maxwidth if an int of float is given.
    uniwidth  : Default is False. If True, linewidth is set to maxwidth/2 for all lines. Only works for rgb_lines.
    colors_list: List of colors for multiple lines, length equal to 3rd axis length of colors.
    rgb        : Default is False. If True and np.shape(colors)[-1] == 3, RGB line collection is returned in a tuple of length 1. Tuple is just to support iteration.
    **pros_data: Output dictionary from `_fix_data` containing kpath, evals, colors arrays.
    """
    if not isinstance(maxwidth, (int, np.integer, float)):
        raise ValueError("maxwidth must be an int or float.")

    if not pros_data:
        raise ValueError("No pros_data given.")
    else:
        kpath = pros_data.get("kpath")
        evals = pros_data.get("evals")
        pros = pros_data.get("pros")

    for a, t in zip([kpath, evals, pros], ["kpath", "evals", "pros"]):
        if not np.any(a):
            raise ValueError("Missing {!r} from output of `_fix_data()`".format(t))

    # Average pros on two consecutive KPOINTS to get that patch color.
    colors = pros[1:, :, :] / 2 + pros[:-1, :, :] / 2  # Near kpoints avearge
    colors = colors.transpose((1, 0, 2)).reshape(
        (-1, np.shape(colors)[-1])
    )  # Must before lws

    if rgb:  # Single channel line widths
        lws = np.sum(colors, axis=1)  # Sum over RGB
    else:  # For separate lines
        lws = colors.T  # .T to access in for loop.

    lws = 0.1 + maxwidth * lws / (
        float(np.max(lws)) or 1
    )  # Rescale to maxwidth, with a residual with 0.1 as must be visible.

    if np.any(colors_list):
        lc_colors = colors_list
    else:
        cmap = plt.cm.get_cmap("viridis")
        lc_colors = cmap(np.linspace(0, 1, np.shape(colors)[-1]))
        lc_colors = lc_colors[:, :3]  # Skip Alpha

    # Reshaping data same as colors reshaped above, nut making line patches too.
    kgrid = np.repeat(kpath, np.shape(evals)[1], axis=0).reshape(
        (-1, np.shape(evals)[1])
    )
    narr = np.concatenate((kgrid, evals), axis=0).reshape((2, -1, np.shape(evals)[1]))
    marr = (
        np.concatenate((narr[:, :-1, :], narr[:, 1:, :]), axis=0)
        .transpose()
        .reshape((-1, 2, 2))
    )

    # Make Line collection
    path_shadow = None
    if shadow:
        path_shadow = [
            path_effects.SimpleLineShadow(offset=(0, -0.8), rho=0.2),
            path_effects.Normal(),
        ]
    if rgb and np.shape(colors)[-1] == 3:
        return (
            LineCollection(
                marr,
                colors=colors,
                linewidths=(maxwidth / 2,) if uniwidth else lws,
                path_effects=path_shadow,
            ),
        )
    else:
        lcs = [
            LineCollection(marr, colors=_cl, linewidths=lw, path_effects=path_shadow)
            for _cl, lw in zip(lc_colors, lws)
        ]
        return tuple(lcs)


# Further fix data for all cases which have projections
def _fix_data(K, E, pros, labels, interp, rgb=False, **others):
    "Input pros must be [m,nk,nb], output is [nk,nb, m]. `others` must have shape [nk,nb] for occupancies or [nk,3] for kpoints"

    if np.shape(pros)[-2:] != np.shape(E):
        raise ValueError("last two dimensions of `pros` must have same shape as `E`")

    if np.ndim(pros) == 2:
        pros = np.expand_dims(pros, 0)  # still as [m,nk,nb]

    if others:
        for k, v in others.items():
            if np.shape(v)[0] != len(K):
                raise ValueError(f"{k} must have same length as K")

    if rgb and len(pros) > 3:
        raise ValueError("In RGB lines mode, pros.shape[-1] <= 3 should hold")

    # Should be after exapnding dims but before transposing
    if labels and len(labels) != len(pros):
        raise ValueError("labels must be same length as pros")

    pros = np.transpose(pros, (1, 2, 0))  # [nk,nb,m] now

    # Normalize overall data because colors are normalized to 0-1
    min_max_pros = (np.min(pros), np.max(pros))  # For data scales to use later
    c_max = np.ptp(pros)
    if c_max > 0.0000001:  # Avoid division error
        pros = (pros - np.min(pros)) / c_max

    data = {"kpath": K, "evals": E, "pros": pros, **others, "ptp": min_max_pros}
    if interp:
        nk = interp if isinstance(interp, (list, tuple)) else (interp, 3)
        min_d, max_d = np.min(pros), np.max(pros)  # For cliping
        _K, E = gu.interpolate_data(K, E, *nk)
        pros = gu.interpolate_data(K, pros, *nk)[1].clip(min=min_d, max=max_d)
        data.update({"kpath": _K, "evals": E, "pros": pros})
        for k, v in others.items():
            data[k] = gu.interpolate_data(K, v, *nk)[1]

    # Handle kpath discontinuities
    X = data["kpath"]
    breaks = [i for i in range(0, len(X)) if X[i - 1] == X[i]]
    if breaks:
        data["kpath"] = np.insert(data["kpath"], breaks, np.nan)
        data["evals"] = np.insert(data["evals"], breaks, np.nan, axis=0)
        data["pros"] = np.insert(
            data["pros"], breaks, data["pros"][breaks], axis=0
        )  # Repeat the same data to keep color consistent
        for (
            key
        ) in (
            others
        ):  # don't use items here, as interpolation may have changed the shape
            data[key] = np.insert(
                data[key], breaks, data[key][breaks], axis=0
            )  # Repeat here too

    return data


@gu._fmt_doc(_docs)
def splot_rgb_lines(
    K,
    E,
    pros,
    labels,
    ax=None,
    elim=None,
    kticks=None,
    interp=None,
    maxwidth=3,
    uniwidth=False,
    colormap=None,
    colorbar=True,
    N=9,
    shadow=False,
):
    """Plot projected band structure for a given projections.
    {params}\n    {K}\n    {E}\n    {pros}\n    {labels}\n    {ax}\n    {elim}\n    {kticks}\n    {interp}\n    {maxwidth}
    uniwidth : bool
        If True, use same linewidth for all patches to maxwidth/2. Otherwise, use linewidth proportional to projection value.
    {colormap}
    colorbar : bool
        If True, add colorbar, otherwise add attribute to ax to add colorbar or color cube later
    N : int
        Number of colors in colormap
    shadow : bool
        If True, add shadow to lines

    {return_ax}
        Returned ax has additional attributes:
        .add_colorbar() : Add colorbar that represents most recent plot
        .color_cube()   : Add color cube that represents most recent plot if `pros` is 3 components
    """
    K, E, xticks, xticklabels = _validate_data(K, E, elim, kticks, interp)

    ax = get_axes() if ax is None else ax

    # =====================================================
    pros_data = _fix_data(
        K, E, pros, labels, interp, rgb=True
    )  # (nk,), (nk, nb), (nk, nb, m) at this point
    colors = pros_data["pros"]
    how_many = np.shape(colors)[-1]

    if how_many == 1:
        percent_colors = colors[:, :, 0]
        percent_colors = percent_colors / np.max(percent_colors)
        pros_data["pros"] = plt.cm.get_cmap(colormap or "copper", N)(percent_colors)[
            :, :, :3
        ]  # Get colors in RGB space.

    elif how_many == 2:
        _sum = np.sum(colors, axis=2)
        _sum[_sum == 0] = 1  # Avoid division error
        percent_colors = colors[:, :, 1] / _sum  # second one is on top
        _colors = plt.cm.get_cmap(colormap or "coolwarm", N)(percent_colors)[
            :, :, :3
        ]  # Get colors in RGB space.
        _colors[np.sum(colors, axis=2) == 0] = [
            0,
            0,
            0,
        ]  # Set color to black if no total projection
        pros_data["pros"] = _colors

    else:
        # Normalize color at each point only for 3 projections.
        c_max = np.max(colors, axis=2, keepdims=True)
        c_max[c_max == 0] = 1  # Avoid division error:
        colors = colors / c_max  # Weights to be used for color interpolation.

        nsegs = np.linspace(0, 1, N, endpoint=True)
        for low, high in zip(nsegs[:-1], nsegs[1:]):
            colors[(colors >= low) & (colors < high)] = (
                low + high
            ) / 2  # Center of squre is taken in color_cube

        A, B, C = plt.cm.get_cmap(colormap or "brg", N)([0, 0.5, 1])[:, :3]
        pros_data["pros"] = np.array(
            [
                [(r * A + g * B + b * C) / ((r + g + b) or 1) for r, g, b in _cols]
                for _cols in colors
            ]
        )

        # Normalize after picking colors from colormap as well to match the color_cube.
        c_max = np.max(pros_data["pros"], axis=2, keepdims=True)
        c_max[c_max == 0] = 1  # Avoid division error:

        pros_data["pros"] = pros_data["pros"] / c_max

    (line_coll,) = _make_line_collection(
        **pros_data,
        rgb=True,
        colors_list=None,
        maxwidth=maxwidth,
        shadow=shadow,
        uniwidth=uniwidth,
    )
    ax.add_collection(line_coll)
    ax.autoscale_view()
    adjust_axes(
        ax,
        xticks=xticks,
        xticklabels=xticklabels,
        xlim=[min(K), max(K)],
        ylim=elim,
        vlines=True,
        top=True,
        right=True,
    )
    # ====================================================

    # Add colorbar/legend etc.
    cmap = colormap or (
        "copper" if how_many == 1 else "brg" if how_many == 3 else "coolwarm"
    )
    ticks = (
        np.linspace(*pros_data["ptp"], 5, endpoint=True)
        if how_many == 1
        else None
        if how_many == 3
        else [0, 1]
    )
    ticklabels = [f"{t:4.2f}" for t in ticks] if how_many == 1 else labels

    if colorbar:
        if how_many < 3:
            cax = add_colorbar(
                ax,
                N=N,
                vertical=True,
                ticklabels=ticklabels,
                ticks=ticks,
                cmap_or_clist=cmap,
            )
            if how_many == 1:
                cax.set_title(labels[0])
        else:
            color_cube(ax, colormap=colormap or "brg", labels=labels, N=N)
    else:
        # MAKE PARTIAL COLOR CUBE AND COLORBAR HERE FOR LATER USE.
        def recent_colorbar(
            cax=None, tickloc="right", vertical=True, digits=2, fontsize=8
        ):
            return add_colorbar(
                ax=ax,
                cax=cax,
                cmap_or_clist=cmap,
                N=N,
                ticks=ticks,
                ticklabels=ticklabels,
                tickloc=tickloc,
                vertical=vertical,
                digits=digits,
                fontsize=fontsize,
            )

        ax.add_colorbar = recent_colorbar

        def recent_color_cube(loc=(0.67, 0.67), size=0.3, color="k", fontsize=10):
            return color_cube(
                ax=ax,
                colormap=cmap,
                labels=labels,
                N=N,
                loc=loc,
                size=size,
                color=color,
                fontsize=fontsize,
            )

        ax.color_cube = recent_color_cube

    return ax


@gu._fmt_doc(_docs)
def splot_color_lines(
    K,
    E,
    pros,
    labels,
    axes=None,
    elim=None,
    kticks=None,
    interp=None,
    maxwidth=3,
    colormap=None,
    shadow=False,
    showlegend=True,
    xyc_label=[0.2, 0.85, "black"],  # x, y, color only if showlegend = False
    **kwargs,
):
    """Plot projected band structure for a given projections.
    {params}\n    {K}\n    {E}\n    {pros}\n    {labels}
    axes : matplotlib.axes.Axes or list of Axes
        Number of axes should be 1 or equal to the number of projections to plot separately. If None, creates new axes.
    {elim}\n    {kticks}\n    {interp}\n    {maxwidth}\n    {colormap}
    shadow : bool
        If True, add shadow to lines
    showlegend : bool
        If True, add legend, otherwise adds a label to the plot.
    xyc_label : list or tuple
        List of (x, y, color) for the label. Used only if showlegend = False


    kwargs are passed to matplotlib's command `ax.legend`.

    Returns
    -------
    One or as many matplotlib.axes.Axes as given by `axes` parameter.
    """
    K, E, xticks, xticklabels = _validate_data(K, E, elim, kticks, interp)
    pros_data = _fix_data(K, E, pros, labels, interp, rgb=False)

    if colormap not in plt.colormaps():
        c_map = plt.cm.get_cmap("viridis")
        print(
            "colormap = {!r} not exists, falling back to default color map.".format(
                colormap
            )
        )
    else:
        c_map = plt.cm.get_cmap(colormap)
    c_vals = np.linspace(
        0, 1, pros_data["pros"].shape[-1]
    )  # Output pros data has shape (nk, nb, projections)
    colors = c_map(c_vals)

    if not np.any([axes]):
        axes = get_axes()
    axes = np.array([axes]).ravel()  # Safe list any axes size
    if len(axes) == 1:
        axes = [
            axes[0] for _ in range(pros_data["pros"].shape[-1])
        ]  # Make a list of axes for each projection
    elif len(axes) != pros_data["pros"].shape[-1]:
        raise ValueError("Number of axes should be 1 or same as number of projections")

    lcs = _make_line_collection(
        maxwidth=maxwidth, colors_list=colors, rgb=False, shadow=shadow, **pros_data
    )
    _ = [ax.add_collection(lc) for ax, lc in zip(axes, lcs)]
    _ = [ax.autoscale_view() for ax in axes]

    if showlegend:
        # Default values for legend_kwargs are overwritten by **kwargs
        legend_kws = {
            "ncol": 4,
            "anchor": (0, 1.05),
            "handletextpad": 0.5,
            "handlelength": 1,
            "fontsize": "small",
            "frameon": False,
            **kwargs,
        }
        add_legend(
            ax=axes[0], colors=colors, labels=labels, widths=maxwidth, **legend_kws
        )

    else:
        xs, ys, colors = xyc_label
        _ = [
            add_text(ax, xs=xs, ys=ys, colors=colors, txts=lab)
            for ax, lab in zip(axes, labels)
        ]

    _ = [
        adjust_axes(
            ax=ax,
            xticks=xticks,
            xticklabels=xticklabels,
            xlim=[min(K), max(K)],
            ylim=elim,
            vlines=True,
            top=True,
            right=True,
        )
        for ax in axes
    ]
    return axes


def _fix_dos_data(energy, dos_arrays, labels, colors, interp):
    if colors is not None:
        if len(colors) != len(labels):
            raise ValueError("If colors is given,they must have same length as labels.")
    if len(dos_arrays) != len(labels):
        raise ValueError("dos_arrays and labels must have same length.")

    for i, arr in enumerate(dos_arrays):
        if len(energy) != len(arr):
            raise ValueError(
                f"array {i+1} in dos_arrays must have same length as energy."
            )
    if len(dos_arrays) < 1:
        raise ValueError("dos_arrays must have at least one array.")

    if interp and not isinstance(interp, (int, np.integer, list, tuple)):
        raise ValueError("interp must be an integer or a list/tuple of (n,k).")

    if isinstance(interp, (list, tuple)) and len(interp) != 2:
        raise ValueError("interp must be an integer or a list/tuple of (n,k).")

    if interp:
        nk = (
            interp if isinstance(interp, (list, tuple)) else (interp, 3)
        )  # default spline order is 3.
        en, arr1 = gu.interpolate_data(energy, dos_arrays[0], nk)
        arrays = [arr1]
        for a in dos_arrays[1:]:
            arrays.append(gu.interpolate_data(energy, a, nk)[1])

        return en, arrays, labels, colors

    return energy, dos_arrays, labels, colors


@gu._fmt_doc(_docs)
def splot_dos_lines(
    energy,
    dos_arrays,
    labels,
    ax=None,
    elim=None,
    colormap="tab10",
    colors=None,
    fill=True,
    vertical=False,
    stack=False,
    interp=None,
    showlegend=True,
    legend_kws={
        "ncol": 4,
        "anchor": (0, 1.0),
    },
    **kwargs,
):
    """Plot density of states (DOS) lines.
    {params}
    energy : array-like, shape (n,)
    dos_arrays : list of array_like, each of shape (n,) or array-like (m,n)
    labels : list of str, length = len(dos_arrays) should hold.
    {ax}\n    {elim}\n    {colormap}
    colors : list of str, length = len(dos_arrays) should hold if given, and will override colormap.
    fill : bool, default True, if True, fill the area under the DOS lines.
    vertical : bool, default False, if True, plot DOS lines vertically.
    stack : bool, default False, if True, stack the DOS lines. Only works for horizontal plots.
    {interp}
    showlegend : bool, default True, if True, show legend.
    legend_kws : dict, default is just hint, anything that `ipyvasp.add_legend` accepts can be passed, only used if showlegend is True.

    keyword arguments are passed to matplotlib.axes.Axes.plot or matplotlib.axes.Axes.fill_between or matplotlib.axes.Axes.fill_betweenx.

    {return_ax}"""
    energy, dos_arrays, labels, colors = _fix_dos_data(
        energy, dos_arrays, labels, colors, interp
    )  # validate data brfore plotting.

    if colors is None:
        colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, len(labels)))

    if ax is None:
        ax = get_axes()

    if "c" in kwargs:
        kwargs.pop("c")
    if "color" in kwargs:
        kwargs.pop("color")

    if stack:
        if vertical:
            raise NotImplementedError("stack is not supported for vertical plots.")
        else:
            ax.stackplot(energy, *dos_arrays, labels=labels, colors=colors, **kwargs)
    else:
        for arr, label, color in zip(dos_arrays, labels, colors):
            if fill:
                fill_func = ax.fill_betweenx if vertical else ax.fill_between
                fill_func(energy, arr, color=mpl.colors.to_rgba(color, 0.4))
            if vertical:
                ax.plot(arr, energy, label=label, color=color, **kwargs)
            else:
                ax.plot(energy, arr, label=label, color=color, **kwargs)

    if showlegend:
        kwargs = {
            "ncol": 4,
            "anchor": (0, 1.0),
            "handletextpad": 0.5,
            "handlelength": 1,
            "fontsize": "small",
            "frameon": False,
            **legend_kws,
        }
        add_legend(ax, **kwargs)  # Labels are picked from plot

    kws = dict(ylim=elim or []) if vertical else dict(xlim=elim or [])
    xlabel, ylabel = "Energy (eV)", "DOS"
    if vertical:
        xlabel, ylabel = ylabel, xlabel
    adjust_axes(ax, xlabel=xlabel, ylabel=ylabel, zeroline=False, **kws)
    return ax


# PLOTLY PLOTS


def _format_rgb_data(
    K, E, pros, labels, interp, occs, kpoints, maxwidth=10, indices=None
):
    "Transform data to 1D for rgb lines to plot effectently. Output is a dictionary."
    data = _fix_data(K, E, pros, labels, interp, rgb=True, occs=occs, kpoints=kpoints)
    # Note that data['pros'] is normalized to 0-1
    rgb = np.zeros(
        (*np.shape(data["evals"]), 3)
    )  # Initialize rgb array, because there could be less than three channels
    if data["pros"].shape[2] == 3:
        rgb = data["pros"]
    elif data["pros"].shape[2] == 2:
        rgb[:, :, :2] = data["pros"]  # Normalized overall color data
        labels = [*labels, ""]
    elif data["pros"].shape[2] == 1:
        rgb[:, :, :1] = data["pros"]  # Normalized overall color data
        labels = [*labels, "", ""]

    # Since normalized data is Y = (X - X_min)/(X_max - X_min), so X = Y*(X_max - X_min) + X_min is the actual data.
    low, high = data["ptp"]
    data["norms"] = np.round(
        rgb * (high - low) + low, 3
    )  # Read actual data back from normalized data.
    if data["pros"].shape[2] == 2:
        data["norms"][:, :, 2] = np.nan  # Avoid wrong info here
    elif data["pros"].shape[2] == 1:
        data["pros"][:, :, 1:] = np.nan

    lws = np.sum(rgb, axis=2)  # Sum of all colors
    lws = maxwidth * lws / (float(np.max(lws)) or 1)  # Normalize to maxwidth
    data["widths"] = (
        0.0001 + lws
    )  # should be before scale colors, almost zero size of a data point with no contribution.

    # Now scale colors to 1 at each point.
    cl_max = np.max(data["pros"], axis=2)
    cl_max[cl_max == 0.0] = 1  # avoid divide by zero. Contributions are 4 digits only.
    data["pros"] = (rgb / cl_max[:, :, np.newaxis] * 255).astype(
        int
    )  # Normalized per point and set rgb data back to data.

    if indices is None:  # make sure indices are in range
        indices = range(np.shape(data["evals"])[1])

    # Now process data to make single data for faster plotting.
    txt = "Projection: [{}]</br>Value:".format(", ".join(labels))
    K, E, C, S, PT, OT, KT, ET = [], [], [], [], [], [], [], []
    for i, b in enumerate(indices):
        K = [*K, *data["kpath"], np.nan]
        E = [*E, *data["evals"][:, i], np.nan]
        C = [
            *C,
            *[f"rgb({r},{g},{b})" for (r, g, b) in data["pros"][:, i, :]],
            "rgb(0,0,0)",
        ]
        S = [*S, *data["widths"][:, i], data["widths"][-1, i]]
        PT = [*PT, *[f"{txt} [{s}, {p}, {d}]" for (s, p, d) in data["norms"][:, i]], ""]
        OT = [*OT, *[f"Occ: {t:>7.4f}" for t in data["occs"][:, i]], ""]
        KT = [
            *KT,
            *[
                f"K<sub>{j+1}</sub>: {x:>7.3f}{y:>7.3f}{z:>7.3f}"
                for j, (x, y, z) in enumerate(data["kpoints"])
            ],
            "",
        ]
        ET = [
            *ET,
            *["{}".format(b + 1) for _ in data["kpath"]],
            "",
        ]  # Add bands subscripts to labels.

    T = [
        f"</br>{p} </br></br>Band: {e}  {o}</br>{k}"
        for (p, e, o, k) in zip(PT, ET, OT, KT)
    ]
    return {
        "K": K,
        "E": E,
        "C": C,
        "S": S,
        "T": T,
        "labels": labels,
    }  # K, energy, marker color, marker size, text, labels that get changed


def _fmt_labels(ticklabels):
    if isinstance(ticklabels, Iterable):
        labels = [
            re.sub(
                r"\$\_\{(.*)\}\$|\$\_(.*)\$", r"<sub>\1\2</sub>", lab, flags=re.DOTALL
            )
            for lab in ticklabels
        ]  # will match _{x} or _x not both at the same time.
        return [
            re.sub(
                r"\$\^\{(.*)\}\$|\$\^(.*)\$", r"<sup>\1\2</sup>", lab, flags=re.DOTALL
            )
            for lab in labels
        ]
    return ticklabels


@gu._fmt_doc(_docs)
def iplot_bands(
    K, E, fig=None, elim=None, kticks=None, interp=None, title=None, **kwargs
):
    """Plot band structure using plotly.
    {params}\n    {K}\n    {E}
    fig : plotly.graph_objects.Figure
        If not given, create a new figure.
    {elim}\n    {kticks}\n    {interp}
    title : str, title of plot


    kwargs are passed to plotly.graph_objects.Scatter
    {return_fig}"""
    if isinstance(K, dict):  # Provided by Bands class, don't do is yourself
        K, indices = K["K"], K["indices"]
    else:
        K, indices = K, range(np.shape(E)[1])  # Assume K is provided by user

    K, E, xticks, xticklabels = _validate_data(K, E, elim, kticks, interp)
    data = _format_rgb_data(
        K,
        E,
        [E],
        ["X"],
        interp,
        E,
        np.array([K, K, K]).reshape((-1, 3)),
        maxwidth=1,
        indices=indices,
    )  # moking other arrays, we need only
    K, E, T = data["K"], data["E"], data["T"]  # Fixed K and E as single line data
    T = [
        "Band" + t.split("Band")[1].split("Occ")[0] for t in T
    ]  # Just Band number here

    if fig is None:
        fig = go.Figure()

    kwargs = {
        "mode": "markers + lines",
        "marker": dict(size=0.1),
        **kwargs,
    }  # marker so that it is selectable by box, otherwise it does not
    fig.add_trace(go.Scatter(x=K, y=E, hovertext=T, **kwargs))

    fig.update_layout(
        template="plotly_white",
        title=(
            title or ""
        ),  # Do not set autosize = False, need to be responsive in widgets boxes
        margin=go.layout.Margin(l=60, r=50, b=40, t=75, pad=0),
        yaxis=go.layout.YAxis(title_text="Energy (eV)", range=elim or [min(E), max(E)]),
        xaxis=go.layout.XAxis(
            ticktext=_fmt_labels(xticklabels),
            tickvals=xticks,
            tickmode="array",
            range=[min(K), max(K)],
        ),
        font=dict(family="stix, serif", size=14),
    )
    return fig


def iplot_rgb_lines(
    K,
    E,
    pros,
    labels,
    occs,
    kpoints,
    fig=None,
    elim=None,
    kticks=None,
    interp=None,
    maxwidth=10,
    mode="markers + lines",
    title=None,
    **kwargs,
):
    """
    Interactive plot of band structure with rgb data points using plotly.

    Parameters
    ----------
    K : array-like, shape (nk,)
    E : array-like, shape (nk,nb)
    pros : array-like, shape (m,nk,nb), m is the number of projections
    labels : list of str, length m
    occs : array-like, shape (nk,nb)
    kpoints : array-like, shape (nk,3)
    fig : plotly.graph_objects.Figure, if not provided, a new figure will be created
    elim : tuple, (emin,emax), energy range to plot
    kticks : [(int, str),...] for indices of high symmetry k-points. To join a broken path, use '<=' before symbol, e.g.  [(0, 'G'),(40, '<=K|M'), ...] will join 40 back to 39. You can also use shortcut like zip([0,10,20],'GMK').
    interp : int or list/tuple of (n,k) for interpolation. If int, n is number of points to interpolate. If list/tuple, n is number of points and k is the order of spline.
    maxwidth : float, maximum linewidth, 10 by default
    mode : str, plotly mode, 'markers + lines' by default, see modes in `plotly.graph_objects.Scatter`.
    title : str, title of the figure, labels are added to the end of the title.

    kwargs are passed to `plotly.graph_objects.Scatter`.

    Returns
    -------
    fig : plotly.graph_objects.Figure that can be displayed in Jupyter notebook or saved as html using `ipyvasp.iplot2html`.
    """
    if isinstance(K, dict):  # Provided by Bands class, don't do is yourself
        K, indices = K["K"], K["indices"]
    else:
        K, indices = K, range(np.shape(E)[1])  # Assume K is provided by user

    K, E, xticks, xticklabels = _validate_data(K, E, elim, kticks, interp)
    data = _format_rgb_data(
        K, E, pros, labels, interp, occs, kpoints, maxwidth=maxwidth, indices=indices
    )
    K, E, C, S, T, labels = (
        data["K"],
        data["E"],
        data["C"],
        data["S"],
        data["T"],
        data["labels"],
    )

    if fig is None:
        fig = go.Figure()

    kwargs.pop("marker_color", None)  # Provided by C
    kwargs.pop("marker_size", None)  # Provided by S
    kwargs.update(
        {
            "hovertext": T,
            "marker": {
                "line_color": "rgba(0,0,0,0)",
                **kwargs.get("marker", {}),
                "color": C,
                "size": S,
            },
        }
    )  # marker edge should be free

    fig.add_trace(go.Scatter(x=K, y=E, mode=mode, **kwargs))

    fig.update_layout(
        template="plotly_white",
        title=(title or "")
        + "["
        + ", ".join(labels)
        + "]",  # Do not set autosize = False, need to be responsive in widgets boxes
        margin=go.layout.Margin(l=60, r=50, b=40, t=75, pad=0),
        yaxis=go.layout.YAxis(title_text="Energy (eV)", range=elim or [min(E), max(E)]),
        xaxis=go.layout.XAxis(
            ticktext=_fmt_labels(xticklabels),
            tickvals=xticks,
            tickmode="array",
            range=[min(K), max(K)],
        ),
        font=dict(family="stix, serif", size=14),
    )
    return fig


def iplot_dos_lines(
    energy,
    dos_arrays,
    labels,
    fig=None,
    elim=None,
    colormap="tab10",
    colors=None,
    fill=True,
    vertical=False,
    stack=False,
    mode="lines",
    interp=None,
    **kwargs,
):
    """
    Plot density of states (DOS) lines.

    Parameters
    ----------
    energy : array-like, shape (n,)
    dos_arrays : list of array_like, each of shape (n,) or array-like (m,n)
    labels : list of str, length = len(dos_arrays) should hold.
    fig : plotly.graph_objects.Figure, if not provided, a new figure will be created
    elim : list of length 2, (emin, emax), if None, (min(energy), max(energy)) is used.
    colormap : str, default 'tab10', any valid matplotlib colormap name. Note that colormap is take from matplotlib, not plotly.
    colors : list of str, length = len(dos_arrays) should hold if given, and will override colormap. Should be valid CSS colors.
    fill : bool, default True, if True, fill the area under the DOS lines.
    vertical : bool, default False, if True, plot DOS lines vertically.
    mode : str, default 'lines', plotly mode, see modes in `plotly.graph_objects.Scatter`.
    stack : bool, default False, if True, stack the DOS lines. Only works for horizontal plots.
    interp : int or list/tuple of (n,k), default None, if given, interpolate the DOS lines using spline.

    keyword arguments are passed to `plotly.graph_objects.Scatter`.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    energy, dos_arrays, labels, colors = _fix_dos_data(
        energy, dos_arrays, labels, colors, interp
    )
    if fig is None:
        fig = go.Figure()
        fig.update_layout(
            margin=go.layout.Margin(l=60, r=50, b=40, t=75, pad=0),
            font=dict(family="stix, serif", size=14),
        )  # Do not set autosize = False, need to be responsive in widgets boxes
    if elim:
        ylim = [min(elim), max(elim)]
    else:
        ylim = [min(energy), max(energy)]

    if colors is None:
        from matplotlib.pyplot import cm

        _colors = cm.get_cmap(colormap)(np.linspace(0, 1, 2 * len(labels)))
        colors = [
            "rgb({},{},{})".format(*[int(255 * x) for x in c[:3]]) for c in _colors
        ]
    if vertical:
        if stack:
            raise NotImplementedError("stack is not supported for vertical plots")

        _fill = "tozerox" if fill else None
        fig.update_yaxes(range=ylim, title="Energy (eV)")
        fig.update_xaxes(title="DOS")
        for arr, label, color in zip(dos_arrays, labels, colors):
            fig.add_trace(
                go.Scatter(
                    y=energy,
                    x=arr,
                    line_color=color,
                    fill=_fill,
                    mode=mode,
                    name=label,
                    **kwargs,
                )
            )
    else:
        extra_args = {"stackgroup": "one"} if stack else {}
        _fill = "tozeroy" if fill else None
        fig.update_xaxes(range=ylim, title="Energy (eV)")
        fig.update_yaxes(title="DOS")
        for arr, label, color in zip(dos_arrays, labels, colors):
            fig.add_trace(
                go.Scatter(
                    x=energy,
                    y=arr,
                    line_color=color,
                    fill=_fill,
                    mode=mode,
                    name=label,
                    **kwargs,
                    **extra_args,
                )
            )

    return fig
