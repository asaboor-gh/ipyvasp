__all__ = ["splot_potential", "LOCPOT", "CHG", "ELFCAR", "PARCHG"]

import numpy as np
import matplotlib.pyplot as plt

import ipywidgets as ipw

from . import utils as gu
from .utils import _sig_kwargs, _sub_doc
from .core import serializer, parser as vp
from .core.plot_toolkit import get_axes


def splot_potential(
    basis=None,
    values=None,
    operation="mean_c",
    ax=None,
    period=None,
    period_right=None,
    interface=None,
    lr_pos=(0.25, 0.75),
    smoothness=2,
    labels=(r"$V(z)$", r"$\langle V \rangle _{roll}(z)$", r"$\langle V \rangle $"),
    colors=((0, 0.2, 0.7), "b", "r"),
    annotate=True,
):
    """
    Returns tuple(ax,Data) where Data contains resultatnt parameters of averaged potential of LOCPOT.

    Parameters
    ----------
    values : `epxort_potential().values` is 3D grid data. As `epxort_potential` is slow, so compute it once and then plot the output data.
    operation : Default is 'mean_c'. What to do with provided volumetric potential data. Anyone of these 'mean_a','min_a','max_a','mean_b','min_b','max_b','mean_c','min_c','max_c'.
    ax : Matplotlib axes, if not given auto picks.
    period : Periodicity of potential in fraction between 0 and 1. For example if a slab is made of 4 super cells in z-direction, period=0.25.
    period_right : Periodicity of potential in fraction between 0 and 1 if right half of slab has different periodicity.
    lr_pos : Locations around which averages are taken.Default (0.25,0.75). Provide in fraction between 0 and 1. Center of period is located at these given fractions. Work only if period is given.
    interface : Default is 0.5 if not given, you may have slabs which have different lengths on left and right side. Provide in fraction between 0 and 1 where slab is divided in left and right halves.
    smoothness : Default is 3. Large value will smooth the curve of potential. Only works if period is given.
    labels : List of three labels for legend. Use plt.legend() or ipv.add_legend() for labels to appear. First entry is data plot, second is its convolution and third is complete average.
    colors : List of three colors for lines.
    annotate : True by default, writes difference of right and left averages on plot.
    """
    check = [
        "mean_a",
        "min_a",
        "max_a",
        "mean_b",
        "min_b",
        "max_b",
        "mean_c",
        "min_c",
        "max_c",
    ]
    if operation not in check:
        raise ValueError(
            "`operation` excepts any of {}, got {}".format(check, operation)
        )
    if ax is None:
        ax = get_axes()
    if values is None or basis is None:
        print("`values` or `basis` not given, trying to autopick LOCPOT...")
        try:
            ep = vp.export_locpot()
            basis = ep.poscar.basis
            values = ep.values
        except:
            raise Exception(
                "Could not auto fix. Make sure `basis` and `v` are provided."
            )
    x_ind = "abc".index(operation.split("_")[1])
    other_inds = tuple([i for i in [0, 1, 2] if i != x_ind])
    _func_ = np.min if "min" in operation else np.max if "max" in operation else np.mean
    pot = _func_(values, axis=other_inds)

    # Direction axis
    x = np.linalg.norm(basis[x_ind]) * np.linspace(
        0, 1, len(pot), endpoint=False
    )  # VASP does not include last point, it is same as firts one
    ax.plot(x, pot, lw=0.8, c=colors[0], label=labels[0])  # Potential plot
    ret_dict = {"direction": operation.split("_")[1]}
    # Only go below if periodicity is given
    if period == None:
        return (ax, serializer.Dict2Data(ret_dict))  # Simple Return
    if period != None:
        arr_con = gu.rolling_mean(
            pot,
            period,
            period_right=period_right,
            interface=interface,
            mode="wrap",
            smoothness=smoothness,
        )
        x_con = np.linspace(0, x[-1], len(arr_con), endpoint=False)
        ax.plot(
            x_con, arr_con, linestyle="dashed", lw=0.7, label=labels[1], c=colors[1]
        )  # Convolved plot
        # Find Averages
        left, right = lr_pos
        ind_1 = int(left * len(pot))
        ind_2 = int(right * len(pot))
        x_1, v_1 = x_con[ind_1], arr_con[ind_1]
        x_2, v_2 = x_con[ind_2], arr_con[ind_2]

        ret_dict.update({"left": {"y": float(v_1), "x": float(x_1)}})
        ret_dict.update({"right": {"y": float(v_2), "x": float(x_2)}})
        ret_dict.update({"deltav": float(v_2 - v_1)})
        # Level plot
        ax.step(
            [x_1, x_2],
            [v_1, v_2],
            lw=0.7,
            where="mid",
            marker=".",
            markersize=5,
            color=colors[2],
            label=labels[2],
        )
        # Annotate
        if annotate == True:
            ax.text(
                0.5,
                0.07,
                r"$\Delta _{R,L} = %9.6f$" % (np.round(v_2 - v_1, 6)),
                ha="center",
                va="center",
                bbox=dict(edgecolor="white", facecolor="white", alpha=0.5),
                transform=ax.transAxes,
            )
        ax.set_xlabel("$" + ret_dict["direction"] + " (" + "\u212B" + ")$")
        ax.set_xlim([x[0], x[-1]])
        return (ax, serializer.Dict2Data(ret_dict))


class LOCPOT:
    """
    Class for LOCPOT file. Loads only single set out of 2/4 magnetization data to avoid performance/memory cost while can load electrostatic and one set of magnetization together.

    Parameters
    ----------
    path : path/to/LOCPOT. LOCPOT is auto picked in CWD.
    data_set : 0 for electrostatic data, 1 for magnetization data if ISPIN = 2. If non-colinear calculations, 1,2,3 will pick Mx,My,Mz data sets respectively. Only one data set is loaded, so you should know what you are loading.


    .. note::
        To avoid memory issues while loading multiple LOCPOT files, use this class as a context manager which cleans up the memory after use.


    >>> with LOCPOT('path/to/LOCPOT') as tmp:
    >>>     tmp.splot()
    The object tmp is destroyed here and memory is freed.
    """

    def __init__(self, path=None, data_set=0):
        self._path = path  # Must be
        self._data = vp.export_locpot(path=path, data_set=data_set)

        self.rolling_mean = (
            gu.rolling_mean
        )  # For quick access to rolling mean function.

    def __enter__(self):
        import weakref

        return weakref.proxy(self)

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    @property
    def poscar(self):
        "POSCAR class object"
        from .lattice import POSCAR

        return POSCAR(data=self._data.poscar)

    @property
    def data(self):
        return self._data

    @_sub_doc(splot_potential, {"values :.*operation :": "operation :"})
    @_sig_kwargs(splot_potential, ("values",))
    def splot(self, operation="mean_c", **kwargs):
        return splot_potential(
            basis=self._data.poscar.basis,
            values=self._data.values,
            operation=operation,
            **kwargs,
        )

    def view_period(
        self,
        operation: str = "mean_c",
        interface=0.5,
        lr_pos=(0.25, 0.75),
        smoothness=2,
        figsize=(5, 3),
        **kwargs,
    ):
        """
        Check periodicity using ipywidgets interactive plot.

        Parameters
        ----------
        operation : What to do, such as 'mean_c' or 'mean_a' etc.
        interface : Interface in range [0,1] to divide left and right halves.
        lr_pos : Tuple of (left,right) positions in range [0,1] to get ΔV of right relative to left.
        smoothness : int. Default is 2. Smoothing parameter for rolling mean. Larger is better.
        figsize : Tuple of (width,height) of figure. Since each time a figure is created, we can't reuse it, so we need to specify the size.

        kwargs are passed to the plt.Axes.set(kwargs) method to handle the plot styling.


        .. note::
            You can use return value to retrieve information, like output.f(*output.args, **output.kwargs) in a cell to plot the current state and save it.
        """
        check = [
            "mean_a",
            "min_a",
            "max_a",
            "mean_b",
            "min_b",
            "max_b",
            "mean_c",
            "min_c",
            "max_c",
        ]
        if operation not in check:
            raise ValueError(
                "operation expects any of {!r}, got {}".format(check, operation)
            )

        opr, _dir = operation.split("_")
        x_ind = "abc".index(_dir)
        other_inds = tuple([i for i in [0, 1, 2] if i != x_ind])
        _func_ = getattr(np, opr)
        X_1 = _func_(self._data.values, axis=other_inds)

        _step = round(1 / X_1.size, 4)
        _min = round(4 * _step, 4)  # At least 4 steps per period

        def checker(period, period_right):
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.plot(X_1, label=operation, lw=1)
            X_2 = self.rolling_mean(
                X_1,
                period,
                period_right=period_right,
                interface=interface,
                smoothness=smoothness,
            )
            ax.plot(X_2, label="rolling_mean", ls="dashed", lw=1)

            x = [int(X_2.size * p) for p in lr_pos]
            y = X_2[x]
            ax.step(x, y, where="mid", marker=".", lw=0.7)
            ax.text(
                0,
                y.mean(),
                f"$V_{{R}} - V_{{L}}$ : {y[1]-y[0]:.6f}",
                backgroundcolor=[1, 1, 1, 0.5],
            )
            plt.legend(bbox_to_anchor=(0, 1), loc="lower left", ncol=2, frameon=False)
            ax.set(**kwargs)
            return ax

        return ipw.interactive(
            checker,
            period=ipw.FloatSlider(
                min=_min,
                max=0.5,
                value=0.125,
                step=_step,
                readout_format=".4f",
                continuous_update=False,
            ),
            period_right=ipw.FloatSlider(
                min=_min,
                max=0.5,
                value=0.125,
                step=_step,
                readout_format=".4f",
                continuous_update=False,
            ),
        )

    def view_slice(self, *argse, **kwargs):
        # Use interactive here to select the slice, digonal slices and so on..., tell user to get output results back
        raise NotImplementedError("Coming soon...")


class CHG(LOCPOT):
    __doc__ = LOCPOT.__doc__.replace("LOCPOT", "CHG")

    def __init__(self, path=None, data_set=0):
        super().__init__(path or "CHG", data_set=data_set)


class ELFCAR(LOCPOT):
    __doc__ = LOCPOT.__doc__.replace("LOCPOT", "ELFCAR")

    def __init__(self, path=None, data_set=0):
        super().__init__(path or "ELFCAR", data_set=data_set)


class PARCHG(LOCPOT):
    __doc__ = LOCPOT.__doc__.replace("LOCPOT", "PARCHG")

    def __init__(self, path=None, data_set=0):
        super().__init__(path or "PARCHG", data_set=data_set)
