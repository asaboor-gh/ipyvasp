__all__ = ["visualize_df","EvalsDataFrame"]

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# Inside packages import
from .core.parser import DataSource
from .core import plot_toolkit as ptk


def visualize_df(df, **kwargs):
    "Visualize dataframe using pygwalker interactively in notebook. kwargs are passed to `pygwalker.walk`"
    try:
        import pygwalker
    except ImportError:
        raise ImportError("pygwalker is required to visualize dataframe. Install it using `pip install pygwalker`")

    return pygwalker.walk(df, **kwargs)


def _collect_data(source, spins=None, bands=None, atoms=None, orbs=None):
    if not isinstance(source, DataSource):
        raise TypeError("`source` must be a valid `DataSource` object.")

    kpoints = source.get_kpoints()  # data object, not single array
    evals = source.get_evals(spins=spins, bands=bands, atoms=atoms, orbs=orbs)
    NS, NK, NB = evals.evals.shape

    df_dict = {}
    for arr, name in zip(kpoints.kpoints.T, "xyz"):
        df_dict[f"k{name}"] = np.array([arr for _ in range(NB)]).flatten()

    for arr, name in zip(kpoints.coords.T, "xyz"):
        df_dict[f"{name}"] = np.array([arr for _ in range(NB)]).flatten()

    df_dict["kpt"] = np.array([[i for i in range(NK)] for _ in evals.bands]).flatten()
    df_dict["band"] = np.array([[i for _ in range(NK)] for i in evals.bands]).flatten()

    for s in range(NS):
        key = "e" + "ud"[s] if NS == 2 else "e"
        df_dict[key] = evals.evals[s].T.flatten()

    if hasattr(evals, "orbs"):  # If projections are given
        _orbs = range(evals.pros.shape[-1]) if evals.orbs == -1 else evals.orbs
        _atoms = range(evals.pros.shape[-2]) if evals.atoms == -1 else evals.atoms

        for a in _atoms:
            for o in _orbs:
                for s in range(NS):  # looking spin together in columns is better
                    key = "ud"[s] if NS == 2 else "sxyz"[s]
                    df_dict[f"{key}{a}_{o}"] = evals.pros[s, a, o, ...].T.flatten()

    return df_dict, source.summary, source.poscar, evals.ezero


class EvalsDataFrame(pd.DataFrame):
    """Format eigenvalues from a data source to dataframe.

    Parameters
    ----------
    source : DataSource
        Data source to collect data from. Could be ``ipyvasp.Vasprun`` or ``ipyvasp.Vaspout``.  Alternatively you can use ``DataSource.get_dataframe`` method to get dataframe directly.
    spins : list

    bands : list
        of band indices [zero based here], In output data frame you will see corresponding band number based on full data.
    atoms : list
        of atoms to plot. inner list contains ions indices. Can leave empty to discard projection data.
    orbs : list
        of orbitals to plot. inner list contains orbitals indices. Can leave empty to discard projection data


    kwargs are not used during initialization, they are internally used by dataframe operations.

    Returns
    -------
    EvalsDataFrame
        A subclass of DataFrame with extra methods and attributes.

    Attributes
    ----------
    poscar : ipyvasp.POSCAR
    summary : ipyvasp.DataSource.summary

    Methods
    -------
    sliced : Slice data in a plane orthogonal to given `column` at given `value`.
    masked : Mask data over a constant value in a given column. Useful for plotting fermi level/surface.
    splot : plot data in a 2D plot.
    splot3d : plot data in a 3D plot.
    send_metadata : Copy metadata from this to another EvalsDataFrame, some methods may not carry over metadata, useful in that case.

    All other methods are inherited from pd.DataFrame. If you apply some method that do not pass metadat, then use `send_metadata` to copy metadata to traget EvalsDataFrame.
    """

    current_attrs = {}
    _metadata = ["current_attrs", "source", "ezero", "summary", "poscar"]

    def __init__(self, *source, spins=[0], bands=[0], atoms=None, orbs=None, **kwargs):
        if len(source) == 1 and isinstance(source[0], DataSource):
            df_dict, self.summary, self.poscar, self.ezero = _collect_data(
                source[0], spins=spins, bands=bands, atoms=atoms, orbs=orbs
            )
            super().__init__(df_dict)
            self.source = source[0]  # Need to be there for other types of operations
        elif len(source) > 1:
            for s in source:
                if isinstance(s, DataSource):
                    raise TypeError("source must be a single DataSource object!")
            raise ValueError("source must be a single DataSource object!")
        else:
            super().__init__(*source, **kwargs)  # For other opertaions on dataframe

    @property
    def _constructor(self):
        "That's main hero of this class. This is called when you apply some method of slice it."
        return EvalsDataFrame
    
    def visualize(self, **kwargs):
        "Visualize dataframe using pygwalker interactively in notebook. kwargs are passed to `pygwalker.walk`"
        return visualize_df(self, **kwargs)

    def masked(self, column, value, tol=1e-2, n=None, band=None, method="linear"):
        """Mask dataframe with a given value, using a tolerance.
        If n is given, (band should also be given to avoid multivalued interpolation error) data values are interpolated to grid of size (l,m,n) where n is longest side.
        n could be arbirarily large as mask will filter out data outside the tolerance.
        """
        if n and not isinstance(n, (int, np.integer)):
            raise TypeError(
                "`n` must be an integer to be applied to short side of grid."
            )
        if isinstance(n, (int, np.integer)) and not isinstance(band, (int, np.integer)):
            raise ValueError(
                "A single `band`(int) from dataframe must be given to mask data."
                "Interpolation does not give correct results for multiple values over a point (x,y,z)."
            )

        column_vals = self[column].to_numpy()
        cmin, cmax = column_vals.min(), column_vals.max()
        if (value < cmin) or (value > cmax):
            raise ValueError("value is outside of column data range!")

        df = self.copy()  # To avoid changing original dataframe
        if n and band:
            bands_exist = np.unique(self.band)
            if band not in bands_exist:
                raise ValueError(
                    "Band {} is not in dataframe! Available: {}".format(
                        band, bands_exist
                    )
                )

            _self_ = df[df["band"] == band]  # Single band dataframe
            df.drop(df.index, inplace=True)  # only columns names there and metadata
            kxyz = _self_[["kx", "ky", "kz"]].to_numpy()
            lx, *_, hx = _self_["kx"].sort_values(inplace=False)
            ly, *_, hy = _self_["ky"].sort_values(inplace=False)
            lz, *_, hz = _self_["kz"].sort_values(inplace=False)
            vs = np.array([hx - lx, hy - ly, hz - lz])
            nx, ny, nz = nxyz = (vs / vs.max() * n).astype(int)
            nijk = [i for i, n in enumerate(nxyz) if n > 0]

            if len(nijk) < 2:
                raise ValueError(
                    "At least two of kx,ky,kz must have non-coplanar points."
                )

            if len(nijk) == 3:
                xyz = kxyz.T
                XYZ = [
                    a.flatten()
                    for a in np.mgrid[
                        lx : hx : nx * 1j, ly : hy : ny * 1j, lz : hz : nz * 1j
                    ]
                ]
                for name, index in zip("xyz", range(3)):
                    df[f"k{name}"] = XYZ[index]
            else:
                [l1, l2], [h1, h2], [n1, n2] = np.array(
                    [[lx, ly, lz], [hx, hy, hz], [nx, ny, nz]]
                )[:, nijk]
                xyz = kxyz.T[nijk]
                XYZ = [
                    a.flatten() for a in np.mgrid[l1 : h1 : n1 * 1j, l2 : h2 : n2 * 1j]
                ]
                for name, index in zip("xyz", range(3)):
                    if index in nijk:
                        df[f"k{name}"] = XYZ[index]
                    else:
                        df[f"k{name}"] = np.zeros_like(XYZ[0])

            for c in [_c for _c in _self_.columns if _c not in "kxkykz"]:
                df[c] = griddata(
                    tuple(xyz), _self_[c].to_numpy(), tuple(XYZ), method=method
                )

            del _self_  # To free memory

            df = df.round(6).dropna()

            # SCALE DATA NOTE: See if negative peak to peak scaling required
            _max = []
            for k in [c for c in df.columns if c.startswith("s")]:
                _max.append(np.abs(df[k]).max())

            _max = max(_max)
            for k in [c for c in df.columns if c.startswith("s")]:
                df[k] = df[k] / (_max if _max != 0 else 1)

        # Make sure to keep metadata, it doesn't work otherwise.
        self.send_metadata(df)

        # Sort based on distance, so arrows are drawn in correct order
        out_df = df[
            np.logical_and((df[column] < (value + tol)), (df[column] > (value - tol)))
        ]
        return out_df.sort_values(by=["kx", "ky", "kz"])

    def send_metadata(self, target_spin_dataframe):
        "Copy metadata from this to another EvalsDataFrame."
        for k in self._metadata:
            setattr(target_spin_dataframe, k, getattr(self, k))

    def sliced(self, column="kz", value=0):
        "Slice data in a plane orthogonal to given `column` at given `value`"
        return self[self[column] == value]

    def _collect_arrows_data(self, arrows):
        arrows_data = []
        for arr in arrows:
            if arr not in ["", *self.columns]:
                raise ValueError(f"{arr!r} is not a column in the dataframe")
            arrows_data.append(
                self[arr] if arr else np.zeros_like(self["kx"].to_numpy())
            )

        return np.array(arrows_data).T

    def _collect_kxyz(self, *xyz, shift=0):
        "Return tuple(kxyz, k_order)"
        _kxyz = ["kx", "ky", "kz"]
        kij = [_kxyz.index(a) for a in xyz if a in _kxyz]
        kxyz = self[["kx", "ky", "kz"]].to_numpy()
        kxyz = self.poscar.bz.translate_inside(kxyz, shift=shift)

        # Handle third axis as energy as well
        if len(xyz) == 3 and xyz[2].startswith("e"):
            kxyz[:, 2] = self[xyz[2]].to_numpy()
            kij = [*kij, 2]  # Add energy to kij, it must be of size 2 before

        return kxyz, kij

    def _validate_columns(self, *args):
        for arg in args:
            if (
                isinstance(arg, str) and arg not in self.columns
            ):  # Allow other data as well
                raise ValueError(f"{arg!r} is not a column in the dataframe")

    def splot(
        self,
        *args,
        arrows=[],
        every=1,
        norm=1,
        marker="H",
        ax=None,
        quiver_kws={},
        shift=0,
        **kwargs,
    ):
        """
        Plot energy in 2D with/without arrows.

        Parameters
        ----------
        *args : 3 or 4 names of columns, representing [X,Y,Energy,[Anything]], from given args, last one is colormapped. If kwargs has color, that takes precedence.
        arrows : 2 or 3 names of columns, representing [U,V,[color]]. If quiver_kws has color, that takes precedence.
        every : every nth point is plotted as arrow.
        norm : normalization factor for size of arrows.
        marker : marker to use for scatter, use s as another argument to change size.
        ax : matplotlib axes to plot on (defaults to auto create one).
        quiver_kws : these are passed to matplotlib.pyplot.quiver.
        shift : A number or a list of three numbers that will be added to kpoints before any other operation.

        **kwargs are passed to matplotlib.pyplot.scatter.

        Returns
        --------
        ax : matplotlib axes. It has additinal method `colorbar` to plot colorbar from most recent plot.
        """
        if arrows and len(arrows) not in [2, 3]:
            raise ValueError(
                '`arrows ` requires 2 or 3 items form spin data [s1,s2,[color]], one of s1,s2 could be "".'
            )
        if len(args) not in [3, 4]:
            raise ValueError(
                "splot takes 3 or 4 positional arguments [X,Y,E,[Anything]], last one is colormapped if kwargs don't have color."
            )

        self._validate_columns(*args)
        kxyz, kij = self._collect_kxyz(*args[:2], shift=shift)
        ax = ax or ptk.get_axes()
        minmax_c = [0, 1]
        cmap = kwargs.get("cmap", self.current_attrs["cmap"])

        if arrows:
            arrows_data = self._collect_arrows_data(arrows)
            cmap = quiver_kws.get("cmap", cmap)
            if "color" in quiver_kws:
                cmap = None  # No colorbar for color only
                arrows_data = arrows_data[:, :2]  # color takes precedence
            ax.quiver(
                *kxyz[::every].T[kij], *(norm * arrows_data[::every].T), **quiver_kws
            )

            if len(arrows) == 3 and "color" not in quiver_kws:
                cmap = quiver_kws.get("cmap", "viridis")  # Fallback to default
                minmax_c = [arrows_data[:, 2].min(), arrows_data[:, 2].max()]
        else:
            _C = self[args[-1]]  # Most right arg is color mapped
            kwargs["marker"] = marker  # Avoid double marker
            if "color" in kwargs:
                kwargs["c"] = kwargs["color"]
                del kwargs["color"]  # keep one
                cmap = None  # No colorbar

            kwargs["c"] = kwargs.get("c", _C)
            ax.scatter(*kxyz.T[kij], **kwargs)
            minmax_c = [min(_C), max(_C)]

        self.__class__.current_attrs = {"ax": ax, "minmax_c": minmax_c, "cmap": cmap}
        return ax

    def splot3d(
        self,
        *args,
        arrows=[],
        every=1,
        norm=1,
        marker="H",
        ax=None,
        quiver_kws={"arrowstyle": "-|>", "size": 1},
        shift=0,
        **kwargs,
    ):
        """
        Plot energy in 3D with/without arrows.

        Parameters
        ----------
        *args : 3, 4 or 5 names of columns, representing [X,Y,[Z or Energy],Energy, [Anything]], out of given args, last one is color mapped. if kwargs has color, that takes precedence.
        arrows : 3 or 4 names of columns, representing [U,V,W,[color]]. If color is not given, magnitude of arrows is color mapped. If quiver_kws has color, that takes precedence.
        every : every nth point is plotted as arrow.
        norm : normalization factor for size of arrows.
        marker : marker to use for scatter, use s as another argument to change size.
        ax : matplotlib 3d axes to plot on (defaults to auto create one).
        quiver_kws : these are passed to ipyvasp.quiver3d.
        shift : A number or a list of three numbers that will be added to kpoints before any other operation.

        **kwargs are passed to matplotlib.pyplot.scatter.

        Returns
        -------
        ax : matplotlib 3d axes. It has additinal method `colorbar` to plot colorbar from most recent plot.
        """
        if arrows and len(arrows) not in [3, 4]:
            raise ValueError(
                '`arrows ` requires 3 or 4 items form spin data [s1,s2, s2, [color]], one of s1,s2,s3 could be "".'
            )
        if len(args) not in [3, 4, 5]:
            raise ValueError(
                "splot3d takes 3, 4 or 5 positional arguments [X,Y,E] or [X,Y,Z,E,[Anything]], right-most is color mapped if kwargs don't have color."
            )

        if not args[2][0] in "ek":
            raise ValueError("Z axis must be in [kx,ky,kz, energy]!")

        self._validate_columns(*args)
        kxyz, kij = self._collect_kxyz(*args[:3], shift=shift)
        ax = ax or ptk.get_axes(axes_3d=True)
        minmax_c = [0, 1]
        cmap = kwargs.get("cmap", self.current_attrs["cmap"])

        if arrows:
            arrows_data = self._collect_arrows_data(arrows)
            cmap = quiver_kws.get("cmap", cmap)
            if "color" in quiver_kws:  # color takes precedence
                quiver_kws["C"] = quiver_kws["color"]
                quiver_kws.pop("color")  # It is not in FancyArrowPatch
                cmap = None  # No colorbar
            elif len(arrows) == 4:
                array = arrows_data[::every, 3]
                array = (array - array.min()) / np.ptp(array)
                quiver_kws["C"] = plt.get_cmap(cmap)(array)
                minmax_c = [arrows_data[:, 3].min(), arrows_data[:, 3].max()]
            elif len(arrows) == 3:
                array = np.linalg.norm(arrows_data[::every, :3], axis=1)
                minmax_c = [array.min(), array.max()]  # Fist set then normalize
                array = (array - array.min()) / np.ptp(array)
                quiver_kws["C"] = plt.get_cmap(cmap)(array)

            if "cmap" in quiver_kws:
                quiver_kws.pop("cmap")  # It is not in quiver3d

            ptk.quiver3d(
                *kxyz[::every].T[kij],
                *(norm * arrows_data[::every].T[:3]),
                **quiver_kws,
                ax=ax,
            )

        else:
            _C = self[args[-1]]  # Most righht arg is color mapped
            kwargs["marker"] = marker  # Avoid double marker
            if "color" in kwargs:
                kwargs["c"] = kwargs["color"]
                del kwargs["color"]  # keep one
                cmap = None  # No colorbar

            kwargs["c"] = kwargs.get("c", _C)
            ax.scatter(*kxyz.T[kij], **kwargs)
            minmax_c = [min(_C), max(_C)]

        self.__class__.current_attrs = {"ax": ax, "minmax_c": minmax_c, "cmap": cmap}
        return ax

    def colorbar(self, cax=None, nticks=6, digits=2, **kwargs):
        "Add colobar to most recent plot. kwargs are passed to ipyvasp.splots.add_colorbar"
        if not self.current_attrs["ax"]:
            raise ValueError(
                "No plot has been made yet by using `splot, splot3d` or already consumed by `colorbar`"
            )
        if not self.current_attrs["cmap"]:
            raise ValueError("No Mappable for colorbar found!")

        ax = self.current_attrs["ax"]
        cmap = self.current_attrs["cmap"]
        minmax_c = self.current_attrs["minmax_c"]
        self.current_attrs["ax"] = None  # Reset
        self.current_attrs["cmap"] = None  # Reset
        if ax.name == "3d":
            cax = cax or plt.gcf().add_axes([0.85, 0.15, 0.03, 0.7])

        return ptk.add_colorbar(
            ax,
            cmap,
            cax=cax,
            ticks=np.linspace(*minmax_c, nticks, endpoint=True),
            digits=digits,
            **kwargs,
        )

    @property
    def kpoints(self):
        "Returns kpoints as numpy array"
        return self["kx ky kz".split()].to_numpy()

    @property
    def coords(self):
        "Returns cartesian coodinates of kpoints as numpy array"
        return self["x y z".split()].to_numpy()


# from ipywidgets import interact
# import matplotlib.pyplot as plt
# import numpy as np

# @interact(x = df.columns, y = df.columns, color=df.columns, band = [str(i) for i in np.unique(df.band.to_numpy())])
# def plot(x,y,color, band):
#     df[df.band == int(band)].plot(x,y, c=color, kind='scatter',cmap='turbo',s=200, marker='s', figsize=(3,2.6))
