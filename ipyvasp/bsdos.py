__all__ = [
    "splot_bands",
    "iplot_bands",
    "splot_rgb_lines",
    "iplot_rgb_lines",
    "splot_color_lines",
    "splot_dos_lines",
    "iplot_dos_lines",
    "Bands",
    "DOS",
]

from pathlib import Path
from itertools import product

import numpy as np

from . import widgets as wdg
from . import utils as gu
from .core import parser as vp
from .core import serializer

from .utils import _sig_kwargs, _sub_doc
from ._enplots import (
    splot_bands,
    iplot_bands,
    splot_rgb_lines,
    iplot_rgb_lines,
    splot_color_lines,
    splot_dos_lines,
    iplot_dos_lines,
)


def _format_input(projections, sys_info):
    """
    Format input spins, atoms, orbs and labels according to selected `projections`.
    For example: {'Ga-s':(0,[0]),'Ga-px+py':(0,[2,3]),'Ga-all':(0,'all')} #for Ga in GaAs, to pick Ga-1, use [0] instead of 0 at first place
    or {'Ga-s':('Ga','s'),'Ga-px+py':(0,'px+py'),'all-d':('all','d')}
    In case of 3 items in tuple, the first item is spin index, the second is atoms, the third is orbs.
    """
    if not isinstance(projections, dict):
        raise TypeError(
            "`projections` must be a dictionary, with keys as labels and values from picked projection indices."
        )

    if not hasattr(sys_info, "orbs"):
        raise ValueError("No orbitals found to pick from given data.")

    types = list(sys_info.types.values())
    names = list(sys_info.types.keys())
    max_ind = np.max(
        [t for tt in types for t in tt]
    )  # will be error if two ranges there to compare for max
    norbs = len(sys_info.orbs)

    orbs_map = {} # total if components given
    if 'px' in sys_info.orbs: orbs_map['p'] = range(1,4)
    if 'dxy' in sys_info.orbs: orbs_map['d'] = range(4,9)
    if 'f0' in sys_info.orbs: orbs_map['f'] = range(9,16)

    # Set default values for different situations
    spins, atoms, orbs, labels = [], [], [], []

    for i, (k, v) in enumerate(projections.items()):
        if len(v) not in [2, 3]:
            raise ValueError(
                f"{k!r}: {v} expects 2 items (atoms, orbs), or 3 items (spin, atoms, orbs), got {len(v)}."
            )

        if not isinstance(k, str):
            raise TypeError(f"{k!r} is not a string. Use string as key for labels.")

        labels.append(k)

        if len(v) == 2:
            A, B = v  # A is atom, B is orbs only two cases: (1) int (2) list of int
        else:
            S, A, B = v  #

            if not isinstance(S, (int, np.integer)):
                raise TypeError(
                    f"First itme in a squence of size 3 should be integer to pick spin."
                )

            spins.append(S)  # Only add spins if given
        
        if isinstance(A,str):
            if A.lower() == 'all':
                A = range(max_ind + 1)
            else:
                if not A in sys_info.types.keys():
                    raise KeyError(f"type {A!r} not found. Available are {list(sys_info.types.keys())}, 'all', or indexing with integeres/list of intergers.")
                A = sys_info.types[A]
        
        if isinstance(B,str):
            if B.lower() == 'all':
                B = range(norbs)
            else:
                B = {b:[sys_info.orbs.index(b)] if b in sys_info.orbs else orbs_map.get(b,[]) for b in (a.strip() for a in B.split('+'))}
                for key, value in B.items():
                    if not value:
                        raise KeyError(f"orbital {key!r} not found. Available are {sys_info.orbs}, 'all' or indexing with integers/list of intergers")
                B = list(sorted(set([i for b in B.values() for i in b]))) # flatten

        if not isinstance(A, (int, np.integer, list, tuple, range)):
            raise TypeError(f"{A!r} is not an integer or list/tuple/range of integers.")

        if not isinstance(B, (int, np.integer, list, tuple, range)):
            raise TypeError(f"{B!r} is not an integer or list/tuple/range of integers.")

        # Fix orbs
        B = [B] if isinstance(B, (int, np.integer)) else B

        if np.max(B) >= norbs:
            raise IndexError(
                "index {} is out of bound for {} orbs".format(np.max(B), norbs)
            )
        if np.min(B) < 0:
            raise IndexError(
                "Only positive integers are allowed for selection of orbitals."
            )

        orbs.append(np.unique(B).tolist())

        # Fix atoms
        if isinstance(A, (int, np.integer)):
            if A < 0:
                raise IndexError(
                    "Only positive integers are allowed for selection of atoms."
                )

            if A < len(types):
                atoms.append(types[A])
                info = f"Given {A} at position {i+1} of sequence => {names[A]!r}: {atoms[i]}. "
                print(gu.color.g(info + f"To just pick one ion, write it as [{A}]."))
            else:
                raise IndexError(
                    f"index {A}  at is out of bound for {len(types)} types of ions. Wrap {A} in [] to pick single ion if that was what you meant."
                )
        else:
            if np.max(A) > max_ind:
                raise IndexError(
                    f"index {np.max(A)} is out of bound for {max_ind+1} ions"
                )

            if np.min(A) < 0:
                raise IndexError(
                    "Only positive integers are allowed for selection of atoms."
                )

            atoms.append(np.unique(A).tolist())

    if spins and len(atoms) != len(spins):
        raise ValueError(
            "You should provide spin for each projection or none at all. If not provided, spin is picked from corresponding eigenvalues (up/down) for all projections using 'spin' parameter explicity."
        )

    uatoms = np.unique(
        [a for aa in atoms for a in aa]
    )  # don't use set, need asceding order
    uorbs = np.unique([o for oo in orbs for o in oo])
    uorbs = tuple(uorbs) if len(uorbs) < norbs else -1  # -1 means all orbitals
    uatoms = tuple(uatoms) if len(uatoms) == (max_ind + 1) else -1  # -1 means all atoms
    uspins = tuple(spins)
    
    return (spins, uspins), (atoms, uatoms), (orbs, uorbs), labels


_spin_doc = """spin : int
    0 by default. Use 0 for spin up and 1 for spin down for spin polarized calculations. 
    Data for both channel is loaded by default, so when you plot one spin channel, 
    plotting other with same parameters will use the same data."""
_kind_doc = """kpairs : list/tuple
    List of pair of indices to rearrange a computed path. For example, if you computed
    0:L, 15:G, 25:X, 34:M path and want to plot it as X-G|M-X, use [(25,15), (34,25)] as kpairs.  
bands : list/tuple
    List of indices of bands. If given, this ovverides elim."""
_proj_doc = """projections : dict
    Mapping from str -> [atoms, orbs]. Use dict to select specific projections, 
    e.g. {'Ga-s':(0,[0]),'Ga-px+py':(0,[2,3]),'Ga-all':(0,'all')} or {'Ga-s':('Ga','s'),'Ga-px+py':(0,'px+py'),'all-d':('all','d')}. 
    If values of the dict are callable, they must accept two arguments evals/tdos, occs/idos of from data and 
    should return array of shape[1:] (all but spin dimension)."""


class _BandsDosBase:
    def __init__(self, source):
        if not isinstance(source, vp.DataSource):
            raise TypeError(
                "`source` must be a subclass of `ipyvasp.core.parser.DataSource`."
            )
        self._source = source  # source is instance of DataSource
        self._data = None  # will be updated on demand

    @property
    def source(self):
        return self._source

    @property
    def data(self):
        "Returns a dictionary of information about the picked data after a plotting function called."
        return self._data
    
    def get_skipk(self):
        "Returns number of first few skipped kpoints in bandstructure plot in case of HSE calculations."
        return self.source.get_skipk()
    
    def set_skipk(self, skipk):
        "Set/reset to skip first kpoints in bandstructure plot in case of HSE calculations."
        self.source.set_skipk(skipk)

    def _fix_projections(self, projections):
        labels, spins, atoms, orbs, uspins, uatoms, uorbs = (
            [],
            None,
            None,
            None,
            None,
            None,
            None,
        )

        funcs = []
        if isinstance(projections, dict):
            if not projections:
                raise ValueError(
                    "`projections` dictionary should have at least one item."
                )

            _funcs = [callable(value) for _, value in projections.items()]
            if any(_funcs) and not all(
                _funcs
            ):  # Do not allow mixing of callable and non-callable values, as comparison will not make sense
                raise TypeError(
                    "Either all or none of the values of `projections` must be callable with two arguments evals, occs and return array of same shape as evals."
                )
            elif all(_funcs):
                funcs = [value for _, value in projections.items()]
                labels = list(projections.keys())
            else:
                (spins, uspins), (atoms, uatoms), (orbs, uorbs), labels = _format_input(
                    projections, self.source.summary
                )
        elif projections is not None:
            raise TypeError("`projections` must be a dictionary or None.")

        return (spins, uspins), (atoms, uatoms), (orbs, uorbs), (funcs, labels)


def _read_kticks(kpoints_path):
    "Reads ticks values and labels in header of kpoint file. Returns dictionary of `kticks` that can be used in plotting functions. If not exist in header, returns empty values(still valid)."
    kticks = []
    if (path := Path(kpoints_path)).is_file():
        with path.open("r", encoding="utf-8") as f:  # allow unicode greek letters
            top_line = f.readline()
        if "HSK-PATH" in top_line:
            head = top_line.split("HSK-PATH")[
                1
            ].strip()  # Only update head if HSK-PATH is found.

            hsk = [[v.strip() for v in vs.split(":")] for vs in head.split(",")]
            for k, v in hsk:
                kticks.append((int(k), v))

    return kticks


class Bands(_BandsDosBase):
    """Class to handle and plot bandstructure data.

    Parameters
    ----------
    source : instance of `ipyvasp.DataSource` such as `ipyvasp.Vasprun` or a user defined subclass. 
    You can define your own class to parse data with same attributes and methods by subclassing `ipyvasp.DataSource`.
    """

    def __init__(self, source):
        super().__init__(source)
        self._data_args = ()  # will be updated on demand

    def get_kticks(self, rel_path="KPOINTS"):
        "Reads associated KPOINTS file form a relative path of calculations and returns kticks. If KPOINTS file does not exist or was not created by this module, returns empty dict."
        path = Path(self.source.path).parent / rel_path
        if path.is_file():
            return _read_kticks(path)
        return []

    def get_plot_coords(self, kindices, eindices):
        """Returns coordinates of shape (len(zip(kindices, eindices)), 2) from most recent bandstructure plot.
        Use in a plot command as `plt.plot(*get_plot_coords(kindices, eindices).T)`.
        Enegy values are shifted by `ezero` from a plot command or data. Use coords + [0, ezero] to get data values.
        """
        for inds in (kindices, eindices):
            if not isinstance(inds, (list, tuple, range, np.ndarray)):
                raise TypeError(
                    "`kindices` and `eindices` must be list, tuple, range or numpy array."
                )

        if not hasattr(
            self, "_breaks"
        ):  # There will be data when a plotting function is called, and set _breaks attribute
            raise ValueError(
                "You must call a plotting function first to get band gap coordinates from plot."
            )

        kpath = self.data.kpath
        if self._breaks:
            for i in self._breaks:
                kpath[i:] -= kpath[i] - kpath[i - 1]  # remove distance

            kpath = kpath / np.max(kpath)  # normalize to in this case again

        kvs = [
            kpath[k] for k, e in zip(kindices, eindices)
        ]  # need same size as eindices
        evs = [
            self.data.evals[self._spin][k, e] - self.data.ezero
            for k, e in zip(kindices, eindices)
        ]
        return np.array([kvs, evs]).T  # shape (len(kindices), 2)

    @property
    def gap(
        self,
    ):  # no need for get_here, useful as .gap.coords immediately after a plot
        """
        Retruns band gap data with following attributes:

        coords : array of shape (2, 2) -> (K,E) band gap in coordinates of most recent plot if exists, otherwise in data coordinates. `X,Y = coords.T` for plotting purpose.
        value : band gap value in eV given by E_gap = cbm - vbm.
        vbm, cbm : valence and conduction band energies in data coordinates. No shift is applied.
        kvbm, kcbm : kpoints of vbm and cbm in fractional coordinates. Useful to know which kpoint is at the band gap.

        These attributes will be None if band gap cannot be found. coords will be empty array of size (0,2) in that case.
        """
        if not self.data:
            self.get_data()  # This assigns back to self._data

        if not hasattr(self, "_breaks"):  # even if no plot
            vs = np.array(
                [self.data.kpath[self.data.kvc,], self.data.evc]
            ).T  # shape (K,E) -> (2,2)
        else:
            vs = self.get_plot_coords(self.data.kvc, [0, 0])

        out = {
            "coords": vs,
            "vbm": None,
            "cbm": None,
            "value": None,
            "kvbm": None,
            "kcbm": None,
        }  # will be updated below if vs.size > 0
        if vs.size:  # May not exist, but still same shape in plotting with empty arrays
            if hasattr(self, "_breaks"):  # only shift by ezero if plot exists
                vs[:, 1] = [v - self.data.ezero for v in self.data.evc]
            vbm, cbm = self.data.evc
            kvbm, kcbm = self.data.kpoints[self.data.kvc,]
            out.update(
                {
                    "coords": vs,
                    "vbm": vbm,
                    "cbm": cbm,
                    "value": cbm - vbm,
                    "kvbm": tuple(kvbm),
                    "kcbm": tuple(kcbm),
                }
            )

        return serializer.Dict2Data(out)

    def get_data(self, elim=None, ezero=None, projections: dict = None, kpairs=None, bands=None):
        """
        Selects bands and projections to use in plotting functions. If input arguments are same as previous call, returns cached data.

        Parameters
        ----------
        elim : list, tuple of two floats to pick bands in this energy range. If None, picks all bands.
        ezero : float, None by default. If not None, elim is applied around this energy.
        projections : dict, str -> [atoms, orbs]. Use dict to select specific projections, e.g. {'Ga-s': (0,[0]), 'Ga1-p': ([0],[1,2,3])} in case of GaAs. If values of the dict are callable, they must accept two arguments evals, occs of shape (spin,kpoints, bands) and return array of shape (kpoints, bands).
        kpairs : list, tuple of integers, None by default to select all kpoints in given order. Use this to select specific kpoints intervals in specific order.
        bands : list,tuple of integers, this ovverides elim if given.

        Returns
        -------
        data : Selected bands and projections data to be used in bandstructure plotting functions under this class as `data` argument.
        """
        if self.data and self._data_args == (elim, ezero, projections, kpairs, bands):
            return self.data

        if kpairs and not isinstance(kpairs, (list, tuple)):
            raise TypeError(
                "`kpairs` must be a list/tuple of pair of indices of edge kpoints of intervals."
            )

        kinds = []  # plain indices of kpoints to select
        if kpairs:
            if np.ndim(kpairs) != 2:
                raise ValueError(
                    "`kpairs` must be a list/tuple of pairs indices to select intervals."
                )

            for inds in kpairs:
                if len(inds) != 2:
                    raise ValueError(
                        "`kpairs` must be a list/tuple of pairs indices to select intervals."
                    )

            all_inds = [range(k1, k2, -1 if k2 < k1 else 1) for k1, k2 in kpairs]
            kinds = [
                *[ind for inds in all_inds for ind in inds],
                kpairs[-1][-1],
            ]  # flatten and add last index

        self._data_args = (elim, ezero, projections, bands)

        (
            (spins, uspins),
            (atoms, uatoms),
            (orbs, uorbs),
            (funcs, labels),
        ) = self._fix_projections(projections)

        kpts = self.source.get_kpoints()
        data = self.source.get_evals(
            elim=elim,
            ezero=ezero,
            atoms=uatoms,
            orbs=uorbs,
            spins=uspins or None,
            bands=bands,
        )  # picks available spins if uspins is None

        if not spins:
            spins = [data.spins[0] for _ in labels]
            
        output = {
            "kpath": kpts.kpath,
            "kpoints": kpts.kpoints,
            "coords": kpts.coords,
            **data.to_dict(),
        }

        if kinds:
            coords = output["coords"][kinds]
            kpath = np.cumsum([0, *np.linalg.norm(coords[1:] - coords[:-1], axis=1)])
            output["kpath"] = kpath / np.max(kpath)  # normalize to 1
            output["kpoints"] = output["kpoints"][kinds]
            output["coords"] = coords
            output["evals"] = output["evals"][:, kinds, :]
            output["occs"] = output["occs"][:, kinds, :]
            if "pros" in output:
                output["pros"] = output["pros"][..., kinds, :]

            # Have to see kvc, but it makes no sense when there is some break in kpath
            vbm, cbm = output["evc"]  # evc is a tuple of two tuples
            kvbm = [
                k for k in np.where(output["evals"] == vbm)[1]
            ]  # keep as indices here, we don't know the cartesian coordinates of kpoints here
            kcbm = [k for k in np.where(output["evals"] == cbm)[1]]
            kvc = tuple(
                sorted(product(kvbm, kcbm), key=lambda K: np.ptp(K))
            )  # bring closer points first by sorting
            output["kvc"] = (
                kvc[0] if kvc else (0, 0)
            )  # Only relative minimum indices matter, set to (0,0) if no kvc found

        output["labels"] = labels  # works for both functions and picks

        data = serializer.Dict2Data(output)  # replacing data with updated one
        if funcs:
            pros = []
            for func in funcs:
                out = func(data.evals, data.occs)
                if (
                    np.shape(out) != data.evals.shape[1:]
                ):  # evals shape is (spin, kpoints, bands), but we need single spin
                    raise ValueError(
                        f"Projections returned by {func} must be of same shape as last two dimensions of input evals."
                    )
                pros.append(out)

            output["pros"] = np.array(pros)
            output["info"] = '"Custom projections by user"'

        elif hasattr(
            data, "pros"
        ):  # Data still could be there, but prefer if user provides projections as functions
            arrays = []
            for sp, atom, orb in zip(spins, atoms, orbs):
                if uatoms != -1:
                    atom = [
                        i for i, a in enumerate(data.atoms) if a in atom
                    ]  # indices for partial data loaded
                if uorbs != -1:
                    orb = [i for i, o in enumerate(data.orbs) if o in orb]
                sp = list(data.spins).index(sp)  # index for spin is single
                _pros = np.take(data.pros[sp], atom, axis=0).sum(
                    axis=0
                )  # take dimension of spin and then sum over atoms leaves 3D array
                _pros = np.take(_pros, orb, axis=0).sum(
                    axis=0
                )  # Sum over orbitals leaves 2D array
                arrays.append(_pros)

            output["pros"] = np.array(arrays)
            output.pop("atoms", None)  # No more needed
            output.pop("orbs", None)
            output.pop("spins", None)  # No more needed

        output["shape"] = "(spin[evals,occs]/selection[pros], kpoints, bands)"

        self._data = serializer.Dict2Data(output)  # Assign for later use
        return self._data

    def _handle_kwargs(self, **kwargs):
        "Returns fixed kwargs and new elim relative to fermi energy for gettig data."
        if kwargs.get("spin", None) not in [0, 1]:
            raise ValueError("spin must be 0 or 1")

        self._spin = kwargs.pop(
            "spin", None
        )  # remove from kwargs as plots don't need it, but get_plot_coords need it
        kpairs = kwargs.pop("kpairs", None)  # not needed for plots but need here

        if kpairs is None:
            kwargs["kticks"] = (
                kwargs.get("kticks", None) or self.get_kticks()
            )  # User can provide kticks, but if not, use default
        if isinstance(kwargs["kticks"], zip):
            kwargs["kticks"] = list(kwargs["kticks"]) # otherwise it will consumed below

        # Need to fetch data for gap and plot later
        self._breaks = [
            tick[0]
            for tick in (kwargs["kticks"] or [])
            if tick[1].lstrip().startswith("<=")
        ]
        return kwargs, kwargs.get("elim", None)  # need in plots

    @_sub_doc(
        splot_bands,
        {"K :.*ax :": f"{_spin_doc}\n{_kind_doc}\nax :"},
    )
    @_sig_kwargs(splot_bands, ("K", "E"))
    def splot_bands(self, spin=0, kpairs=None, ezero=None, bands=None, **kwargs):
        kwargs, elim = self._handle_kwargs(spin=spin, **kwargs)
        data = self.get_data(elim=elim, ezero=ezero, kpairs=kpairs, bands=bands)
        return splot_bands(data.kpath, data.evals[spin] - data.ezero, **kwargs)

    @_sub_doc(
        splot_rgb_lines,
        {"K :.*ax :": f"{_proj_doc}\n{_spin_doc}\n{_kind_doc}\nax :"},
    )
    @_sig_kwargs(splot_rgb_lines, ("K", "E", "pros", "labels"))
    def splot_rgb_lines(self, projections, spin=0, kpairs=None, ezero=None, bands=None, **kwargs):
        kwargs, elim = self._handle_kwargs(spin=spin, **kwargs)
        data = self.get_data(elim, ezero, projections, kpairs=kpairs, bands=bands)
        return splot_rgb_lines(
            data.kpath, data.evals[spin] - data.ezero, data.pros, data.labels, **kwargs
        )

    @_sub_doc(
        splot_color_lines,
        {"K :.*axes :": f"{_proj_doc}\n{_spin_doc}\n{_kind_doc}\naxes :"},
    )
    @_sig_kwargs(splot_color_lines, ("K", "E", "pros", "labels"))
    def splot_color_lines(self, projections, spin=0, kpairs=None, ezero=None, bands=None, **kwargs):
        kwargs, elim = self._handle_kwargs(spin=spin, **kwargs)
        data = self.get_data(
            elim, ezero, projections, kpairs=kpairs, bands=bands
        )  # picked relative limit
        return splot_color_lines(
            data.kpath, data.evals[spin] - data.ezero, data.pros, data.labels, **kwargs
        )

    @_sub_doc(
        iplot_rgb_lines,
        {"K :.*fig :": f"{_proj_doc}\n{_spin_doc}\n{_kind_doc}\nfig :"},
    )
    @_sig_kwargs(iplot_rgb_lines, ("K", "E", "pros", "labels", "occs", "kpoints"))
    def iplot_rgb_lines(self, projections, spin=0, kpairs=None, ezero=None, bands=None, **kwargs):
        kwargs, elim = self._handle_kwargs(spin=spin, **kwargs)
        data = self.get_data(elim, ezero, projections, kpairs=kpairs, bands=bands)
        # Send K and bands in place of K for use in iplot_rgb_lines to depict correct band number
        return iplot_rgb_lines(
            {"K": data.kpath, "indices": data.bands},
            data.evals[spin] - data.ezero,
            data.pros,
            data.labels,
            data.occs[spin],
            data.kpoints,
            **kwargs,
        )

    @_sub_doc(
        iplot_bands,
        {"K :.*fig :": f"{_spin_doc}\n{_kind_doc}\nfig :"},
    )
    @_sig_kwargs(iplot_bands, ("K", "E"))
    def iplot_bands(self, spin=0, kpairs=None, ezero=None, bands=None, **kwargs):
        kwargs, elim = self._handle_kwargs(spin=spin, **kwargs)
        data = self.get_data(elim, ezero, kpairs=kpairs, bands=bands)
        # Send K and bands in place of K for use in iplot_rgb_lines to depict correct band number
        return iplot_bands(
            {"K": data.kpath, "indices": data.bands},
            data.evals[spin] - data.ezero,
            occs = data.occs[spin],
            kpoints = data.kpoints,
            **kwargs,
        )

    def view_bands(self, height="450px"):
        "Initialize and return `ipyvasp.widgets.BandsWidget` to view bandstructure interactively."
        return wdg.BandsWidget([self.source.path,],height=height)


_multiply_doc = """multiply : float
    A float number to be multiplied by total dos and its sign is multiplied 
    by partial dos to flip plot in case of spin down."""
_total_doc = """total : bool
    True by default. If False, total dos is not plotted, but sign of multiply 
    parameter is still used for partial dos"""


class DOS(_BandsDosBase):
    """Class to handle and plot density of states data.

    Parameters
    ----------
    source : instance of `ipyvasp.DataSource` such as `ipyvasp.Vasprun` or a user defined subclass. 
    You can define your own class to parse data with same attributes and methods by subclassing `ipyvasp.DataSource`.
    """

    def __init__(self, source):
        super().__init__(source)
        self._data_args = ()  # updated on demand

    def get_data(self, elim=None, ezero=None, projections: dict = None):
        if self.data and self._data_args == (elim, ezero, projections):
            return self.data

        self._data_args = (elim, ezero, projections)

        (
            (spins, uspins),
            (atoms, uatoms),
            (orbs, uorbs),
            (funcs, labels),
        ) = self._fix_projections(projections)
        dos = self.source.get_dos(
            elim=elim, ezero=ezero, atoms=uatoms, orbs=uorbs, spins=uspins or None
        )

        if not spins:
            spins = [dos.spins[0] for _ in labels]
            
        out = dos.to_dict()
        out["labels"] = labels

        if funcs:
            pdos = []
            for func in funcs:
                p = func(dos.tdos, dos.idos)
                if (
                    np.shape(p) != dos.energy.shape[1:]
                ):  # energy shape is (spin, grid), but we need single spin
                    raise ValueError(
                        f"Projections returned by {func} must be of same shape as last dimension of input energy."
                    )
                pdos.append(p)

            out["pdos"] = np.array(pdos)
            out["info"] = '"Custom projections by user"'

        elif hasattr(
            dos, "pdos"
        ):  # Data still could be there, but prefer if user provides projections as functions
            arrays = []
            for sp, atom, orb in zip(spins, atoms, orbs):
                if uatoms != -1:
                    atom = [
                        i for i, a in enumerate(dos.atoms) if a in atom
                    ]  # indices for partial data loaded
                if uorbs != -1:
                    orb = [i for i, o in enumerate(dos.orbs) if o in orb]
                sp = list(dos.spins).index(sp)  # index for spin is single
                _pdos = np.take(dos.pdos[sp], atom, axis=0).sum(
                    axis=0
                )  # take dimension of spin and then sum over atoms leaves 2D array
                _pdos = np.take(_pdos, orb, axis=0).sum(
                    axis=0
                )  # Sum over orbitals leaves 1D array
                arrays.append(_pdos)

            out["pdos"] = np.array(arrays)
            out.pop("atoms", None)  # No more needed
            out.pop("orbs", None)
            out.pop("spins", None)  # No more needed

        out["shape"] = "(spin[energy,tdos,idos]/selection[pdos], NEDOS)"

        self._data = serializer.Dict2Data(out)  # Assign for later use
        return self._data

    def _handle_kwargs(self, **kwargs):
        "Returns fixed kwargs and new elim relative to fermi energy for gettig data."
        if kwargs.get("spin", None) not in [0, 1, 2, 3]:
            raise ValueError("spin must be 0,1,2,3 for dos")

        kwargs.pop("spin", None)  # remove from kwargs as plots don't need it
        kwargs.pop("bands", None) # not required internally
        return kwargs, kwargs.get("elim", None)

    @_sub_doc(
        splot_dos_lines,
        {
            "energy :.*ax :": f"{_proj_doc}\n{_spin_doc}\n{_multiply_doc}\n{_total_doc}\nax :"
        },
    )
    @_sig_kwargs(splot_dos_lines, ("energy", "dos_arrays", "labels"))
    def splot_dos_lines(
        self,
        projections=None,  # dos should allow only total dos as well
        spin=0,
        multiply=1,
        total=True,
        ezero=None,
        **kwargs,
    ):
        kwargs, elim = self._handle_kwargs(spin=spin, **kwargs)
        data = self.get_data(elim, ezero, projections)
        energy, labels = data["energy"][spin], data["labels"]

        dos_arrays = []
        if projections is not None:
            dos_arrays = np.sign(multiply) * data["pdos"]  # filp if asked,

        tlab = kwargs.pop("label", None)  # pop in any case
        if total:
            dos_arrays = [data["tdos"][spin] * multiply, *dos_arrays]
            labels = [tlab or "Total", *labels]
        elif len(dos_arrays) == 0:
            raise ValueError("Either total should be True or projections given!")

        return splot_dos_lines(energy - data.ezero, dos_arrays, labels, **kwargs)

    @_sub_doc(
        iplot_dos_lines,
        {
            "energy :.*fig :": f"{_proj_doc}\n{_spin_doc}\n{_multiply_doc}\n{_total_doc}\nfig :"
        },
    )
    @_sig_kwargs(iplot_dos_lines, ("energy", "dos_arrays", "labels"))
    def iplot_dos_lines(
        self,
        projections=None,  # dos should allow only total dos as well
        spin=0,
        multiply=1,
        total=True,
        ezero=None,
        **kwargs,
    ):
        kwargs, elim = self._handle_kwargs(spin=spin, **kwargs)
        data = self.get_data(elim, ezero, projections)
        energy, labels = data["energy"][spin], data["labels"]

        dos_arrays = []
        if projections is not None:
            dos_arrays = np.sign(multiply) * data["pdos"]  # filp if asked

        tname = kwargs.pop("name", None)  # pop in any case
        if total:
            dos_arrays = [data["tdos"][spin] * multiply, *dos_arrays]
            labels = [tname or "Total", *labels]
        elif len(dos_arrays) == 0:
            raise ValueError("Either total should be True or projections given!")

        return iplot_dos_lines(energy - data.ezero, dos_arrays, labels, **kwargs)
