__all__ = ["POSCAR", "download_structure", "periodic_table", "get_kpath", "get_kmesh"]

from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
from pandas.io.clipboard import clipboard_get, clipboard_set


from .core import serializer
from .core import spatial_toolkit as stk
from .utils import _sig_kwargs, _sub_doc
from . import _lattice as plat
from ._lattice import (
    periodic_table,
    get_kpath,
    get_kmesh,
)  # need these as direct access


def download_structure(
    formula, mp_id=None, max_sites=None, min_sites=None, api_key=None, save_key=False
):
    """
    Download structure data from Materials project website.

    Parameters
    ----------
    formula : chemical formula of the material.
    mp_id : Materials project id of material.
    max_sites : maximum number of sites in structure to download.
    min_sites : minimum number of sites in structure to download.

    Other Parameters
    ----------------
    api_key : API key from Materials project websit, if you use save_key=True, never required again.
    save_key : Save API key to file. You can save any time of key or device changed.

    Return
    ------
    list : List of Structure data containing attribute/method `cif`/`export_poscar, write_cif` etc.


    .. note::
        max_sites and min_sites are used to filter the number of sites in structure, or use mp_id to download a specific structure.
    """
    mp = plat.InvokeMaterialsProject(api_key=api_key)
    output = mp.request(
        formula=formula, mp_id=mp_id, max_sites=max_sites, min_sites=min_sites
    )  # make a request
    if save_key and isinstance(api_key, str):
        mp.save_api_key(api_key)
    if mp.success:
        return output
    else:
        raise ConnectionError("Connection was not sccessful. Try again!")


class POSCAR:
    _cb_instance = {}  # Loads last clipboard data if not changed
    _mp_instance = {}  # Loads last mp data if not changed

    def __init__(self, path=None, content=None, data=None):
        """
        POSCAR class to contain data and related methods, data is PoscarData, json/tuple file/string.
        Do not use `data` yourself, it's for operations on poscar.

        Parameters
        ----------
        path : path to file
        content : string of POSCAR content
        data : PoscarData object. This assumes positions are in fractional coordinates.

        Prefrence order: data > content > path
        """
        self._path = Path(path or "POSCAR")  # Path to file
        self._content = content

        if data:
            self._data = serializer.PoscarData.validated(data)
        else:
            self._data = plat.export_poscar(path=str(self.path), content=content)
        # These after data to work with data
        self._bz = self.get_bz(primitive=False)  # Get defualt regular BZ from sio
        self._cell = self.get_cell()  # Get defualt cell
        self._plane = None  # Get defualt plane, changed with splot_bz
        self._ax = None  # Get defualt axis, changed with splot_bz

    def __repr__(self):
        atoms = ", ".join([f"{k}={len(v)}" for k, v in self._data.types.items()])
        lat = ", ".join(
            [
                f"{k}={v}"
                for k, v in zip(
                    "abcαβγ", (*self._data.norms.round(3), *self._data.angles.round(3))
                )
            ]
        )
        return f"{self.__class__.__name__}({atoms}, {lat})"

    def __str__(self):
        return self.content

    @property
    def path(self):
        return self._path

    def to_ase(self):
        """Convert to ase.Atoms format. You need to have ase installed.
        You can apply all ase methods on this object after conversion.

        Example
        -------
        >>> from ase.visualize import view
        >>> structure = poscar.to_ase()
        >>> view(structure) # POSCAR.view() also uses this method if viewer is given.
        >>> reciprocal_lattice = structure.cell.get_bravais_lattice()
        >>> reciprocal_lattice.plt_bz() # Plot BZ usinn ase, it also plots suggested band path.
        """
        from ase import Atoms

        symbols = [
            lab.split()[0] for lab in self.data.labels
        ]  # Remove numbers from labels
        return Atoms(
            symbols=symbols, positions=self.data.positions, cell=self.data.basis
        )

    def view(self, viewer=None, **kwargs):
        """View POSCAR in notebook. If viewer is given it will be passed ase.visualize.view. You need to have ase installed.

        kwargs are passed to self.splot_lattice if viewer is None, otherwise a  single keyword argument `data` is passed to ase viewer.
        data should be volumetric data for ase.visualize.view, such as charge density, spin density, etc.
        """
        if viewer is None:
            return plat.view_poscar(self.data, **kwargs)
        else:
            from ase.visualize import view

            return view(self.to_ase(), viewer=viewer, data=kwargs.get("data", None))

    def view_kpath(self):
        "Initialize a KpathWidget instance to view kpath for current POSCAR, and you can select others too."
        from .widgets import KpathWidget

        return KpathWidget(path=str(self.path.parent), glob=self.path.name)

    @classmethod
    def from_file(cls, path):
        "Load data from POSCAR file"
        return cls(path=path)

    @classmethod
    def from_string(cls, content):
        "content should be a valid POSCAR string"
        try:
            return cls(content=content)
        except:
            raise ValueError(f"Invalid POSCAR string!!!!!\n{content}")

    @classmethod
    def from_materials_project(cls, formula, mp_id, api_key=None, save_key=False):
        """Downloads POSCAR from materials project. `mp_id` should be string associated with a material on their website. `api_key` is optional if not saved.
        Get your API key from https://legacy.materialsproject.org/open and save it using `save_key` to avoid entering it everytime.
        """
        if cls._mp_instance and cls._mp_instance["kwargs"] == {
            "formula": formula,
            "mp_id": mp_id,
        }:
            if (
                api_key and save_key
            ):  # If user wants to save key even if data is loaded from cache
                plat._save_mp_API(api_key)

            return cls._mp_instance["instance"]

        instance = cls(
            data=download_structure(
                formula=formula, mp_id=mp_id, api_key=api_key, save_key=save_key
            )[0].export_poscar()
        )
        cls._mp_instance = {
            "instance": instance,
            "kwargs": {"formula": formula, "mp_id": mp_id},
        }
        return instance

    @classmethod
    def from_cif(cls, path):
        "Load data from cif file"
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"File {path!r} does not exists")

        with p.open("r", encoding="utf-8") as f:
            cif_str = f.read()
            poscar_content = plat._cif_str_to_poscar_str(
                cif_str, "Exported from cif file using ipyvasp"
            )

        return cls.from_string(poscar_content)

    @classmethod
    def new(cls, basis, sites, scale=1, name=None):
        """
        Crate a new POSCAR instance from basis and sites.

        Parameters
        ----------
        basis : array_like of shape (3, 3)
        sites : dict, key is element and value is array_like of shape (n, 3)
            For example, {'Mg': [[0, 0, 0]], 'Cl': [[1/3, 2/3,0.214],[2/3,1/3,0.786]]} for MgCl2.
        scale : int or float used to scale basis and kept in metadata to use when writing to file.
        name : str, name of the structure, if None, will be inferred from sites.
        """
        if np.ndim(basis) != 2 and np.shape(basis) != (3, 3):
            raise ValueError("basis should be a 3x3 array")
        if not isinstance(scale, (int, float, np.integer)):
            raise ValueError("scale should be a number")

        basis = np.array(basis) * scale  # scale basis, we add scale to metdata later

        if not isinstance(sites, dict):
            raise ValueError(
                "sites should be a dictionary of elements and their positions"
            )
        type_dict, positions, start = {}, [], 0
        for key, value in sites.items():
            if not isinstance(key, str):
                raise ValueError("sites key should be a string like 'Al'")
            if np.ndim(value) != 2 and np.shape(value)[1] != 3:
                raise ValueError("sites value should be an array of shape (n, 3)")
            if np.max(value) > 1.01 or np.min(value) < -0.01:
                raise ValueError(
                    "sites value should be in fractional coordinates between 0 and 1"
                )
            type_dict[key] = range(start, start + len(value))
            positions.extend(value)  # direct stacking of positions
            start += len(value)

        positions = np.array(positions)

        out_dict = {
            "SYSTEM": name if name else "".join(type_dict.keys()),
            "basis": basis,
            "metadata": {
                "scale": scale,
                "cartesian": False,
                "comment": "Created using ipyvasp.lattice.POSCAR.new method",
            },
            "positions": positions,
            "types": type_dict,
        }
        return cls(data=out_dict)

    @classmethod
    def from_clipborad(cls):
        "Read POSCAR from clipboard (based on clipboard reader impelemented by pandas library) It picks the latest from clipboard."
        try:
            instance = cls.from_string(
                content=clipboard_get()
            )  # read from clipboard while allowing error for bad data
            if isinstance(instance, cls):  # if valid POSCAR string
                cls._cb_instance = {"instance": instance}  # cache instance
                return instance
        except:
            if cls._cb_instance:
                print(
                    "Loading from previously cached clipboard data, as current data is not valid POSCAR string."
                )
                return cls._cb_instance["instance"]
            else:
                raise ValueError(
                    "Clipboard does not contain valid POSCAR string and no previous data is cached."
                )

    def to_clipboard(self):
        "Writes POSCAR to clipboard (as implemented by pandas library) for copy in other programs such as vim."
        clipboard_set(self.content)  # write to clipboard

    @property
    def data(self):
        "Data object in POSCAR."
        return self._data

    def copy(self):
        "Copy POSCAR object. It avoids accidental changes to numpy arrays in original object."
        return self.__class__(data=self._data.copy())

    @property
    def content(self):
        "POSCAR content."
        with redirect_stdout(StringIO()) as f:
            self.write(outfile=None)  # print to stdout
            return f.getvalue()

    @property
    def bz(self):
        return self._bz

    @property
    def cell(self):
        return self._cell

    @_sub_doc(stk.get_bz, {"basis :.*loop :": "loop :"})
    @_sig_kwargs(stk.get_bz, ("basis",))
    def get_bz(self, **kwargs):
        self._bz = stk.get_bz(self._data.rec_basis, **kwargs)
        return self._bz

    def get_cell(self, loop=True):
        "See docs of `get_bz`, same except space is inverted and no factor of 2pi."
        self._cell = serializer.CellData(
            stk.get_bz(  # data.basis makes prmitive cell in direct space
                basis=self.data.basis, loop=loop, primitive=True
            ).to_dict()
        )  # cell must be primitive
        return self._cell

    @_sub_doc(plat.splot_bz, {"bz_data :.*plane :": "plane :"})
    @_sig_kwargs(plat.splot_bz, ("bz_data",))
    def splot_bz(self, plane=None, **kwargs):
        self._plane = plane  # Set plane for splot_kpath
        new_ax = plat.splot_bz(bz_data=self._bz, plane=plane, **kwargs)
        self._ax = new_ax  # Set ax for splot_kpath
        self._zoffset = kwargs.get("zoffset", 0)  # Set zoffset for splot_kpath
        return new_ax

    def splot_kpath(
        self, kpoints, labels=None, fmt_label=lambda x: (x, {"color": "blue"}), **kwargs
    ):
        """Plot k-path over existing BZ.

        Parameters
        ----------
        kpoints : array_like
            List of k-points in fractional coordinates. e.g. [(0,0,0),(0.5,0.5,0.5),(1,1,1)] in order of path.
        labels : list
            List of labels for each k-point in same order as kpoints.
        fmt_label : callable
            Function that takes a label from labels and should return a string or (str, dict) of which dict is passed to ``plt.text``.


        kwargs are passed to ``plt.plot`` with some defaults.

        You can get ``kpoints = POSCAR.get_bz().specials.masked(lambda x,y,z : (-0.1 < z 0.1) & (x >= 0) & (y >= 0))`` to get k-points in positive xy plane.
        Then you can reorder them by an indexer like ``kpoints = kpoints[[0,1,2,0,7,6]]``, note double brackets, and also that point at zero index is taken twice.

        .. tip::
            You can use this function multiple times to plot multiple/broken paths over same BZ.
        """
        if not self._bz or not self._ax:
            raise ValueError("BZ not found, use `splot_bz` first")

        if not np.ndim(kpoints) == 2 and np.shape(kpoints)[-1] == 3:
            raise ValueError("kpoints must be 2D array of shape (N,3)")

        ijk = [0, 1, 2]
        _mapping = {
            "xy": [0, 1],
            "xz": [0, 2],
            "yz": [1, 2],
            "zx": [2, 0],
            "zy": [2, 1],
            "yx": [1, 0],
        }
        _zoffset = [0, 0, 0]
        if self._plane:
            _zoffset = (
                [0, 0, self._zoffset]
                if self._plane in "xyx"
                else [0, self._zoffset, 0]
                if self._plane in "xzx"
                else [self._zoffset, 0, 0]
            )

        if isinstance(self._plane, str) and self._plane in _mapping:
            if getattr(self._ax, "name", None) != "3d":
                ijk = _mapping[
                    self._plane
                ]  # only change indices if axes is not 3d, even if plane is given

        if not labels:
            labels = [
                "[{0:5.2f}, {1:5.2f}, {2:5.2f}]".format(x, y, z) for x, y, z in kpoints
            ]

        plat._validate_label_func(fmt_label, labels[0])

        coords = self.bz.to_cartesian(kpoints)
        if _zoffset and self._plane:
            normal = (
                [0, 0, 1]
                if self._plane in "xyx"
                else [0, 1, 0]
                if self._plane in "xzx"
                else [1, 0, 0]
            )
            coords = plat.to_plane(normal, coords) + _zoffset

        coords = coords[:, ijk]  # select only required indices
        kwargs = {
            **dict(color="blue", linewidth=0.8, marker=".", markersize=10),
            **kwargs,
        }  # need some defaults
        self._ax.plot(*coords.T, **kwargs)

        for c, text in zip(coords, labels):
            lab, textkws = fmt_label(text), {}
            if isinstance(lab, (list, tuple)):
                lab, textkws = lab
            self._ax.text(*c, lab, **textkws)

        return self._ax

    @_sig_kwargs(plat.splot_bz, ("bz_data",))
    def splot_cell(self, plane=None, **kwargs):
        "See docs of `splot_bz`, everything is same except space is inverted."
        return plat.splot_bz(bz_data=self._cell, plane=plane, **kwargs)

    @_sub_doc(plat.iplot_bz, {"bz_data :.*fill :": "fill :"})
    @_sig_kwargs(plat.iplot_bz, ("bz_data",))
    def iplot_bz(self, **kwargs):
        return plat.iplot_bz(bz_data=self._bz, **kwargs)

    @_sig_kwargs(plat.iplot_bz, ("bz_data", "special_kpoints"))
    def iplot_cell(self, **kwargs):
        "See docs of `iplot_bz`, everything is same except space is iverted."
        return plat.iplot_bz(bz_data=self._cell, special_kpoints=False, **kwargs)

    @_sub_doc(plat.splot_lattice)
    @_sig_kwargs(plat.splot_lattice, ("poscar_data", "plane"))
    def splot_lattice(self, plane=None, **kwargs):
        return plat.splot_lattice(self._data, plane=plane, **kwargs)

    @_sub_doc(plat.iplot_lattice)
    @_sig_kwargs(plat.iplot_lattice, ("poscar_data",))
    def iplot_lattice(self, **kwargs):
        return plat.iplot_lattice(self._data, **kwargs)

    @_sub_doc(plat.write_poscar)
    @_sig_kwargs(plat.write_poscar, ("poscar_data",))
    def write(self, outfile=None, **kwargs):
        return plat.write_poscar(self._data, outfile=outfile, **kwargs)

    @_sub_doc(plat.join_poscars)
    @_sig_kwargs(plat.join_poscars, ("poscar_data", "other"))
    def join(self, other, direction="c", **kwargs):
        return self.__class__(
            data=plat.join_poscars(
                poscar_data=self._data, other=other.data, direction=direction, **kwargs
            )
        )

    @_sub_doc(plat.scale_poscar)
    @_sig_kwargs(plat.scale_poscar, ("poscar_data",))
    def scale(self, scale=(1, 1, 1), **kwargs):
        return self.__class__(data=plat.scale_poscar(self._data, scale, **kwargs))

    @_sub_doc(plat.rotate_poscar)
    def rotate(self, angle_deg, axis_vec):
        return self.__class__(
            data=plat.rotate_poscar(self._data, angle_deg=angle_deg, axis_vec=axis_vec)
        )

    @_sub_doc(plat.set_zdir)
    def set_zdir(self, hkl, phi=0):
        return self.__class__(data=plat.set_zdir(self._data, hkl, phi=phi))

    @_sub_doc(plat.translate_poscar)
    def translate(self, offset):
        return self.__class__(data=plat.translate_poscar(self._data, offset=offset))

    @_sub_doc(plat.repeat_poscar)
    def repeat(self, n, direction):
        return self.__class__(
            data=plat.repeat_poscar(self._data, n=n, direction=direction)
        )

    @_sub_doc(plat.mirror_poscar)
    def mirror(self, direction):
        return self.__class__(data=plat.mirror_poscar(self._data, direction=direction))

    @_sub_doc(stk.get_TM, replace={"basis1": "self.basis"})
    def get_TM(self, target_basis):
        return stk.get_TM(self._data.basis, target_basis)

    @_sub_doc(plat.transform_poscar)
    def transform(self, transformation, zoom=2, tol=1e-2):
        return self.__class__(
            data=plat.transform_poscar(self._data, transformation, zoom=zoom, tol=tol)
        )

    @_sub_doc(plat.transpose_poscar)
    def transpose(self, axes=[1, 0, 2]):
        return self.__class__(data=plat.transpose_poscar(self._data, axes=axes))

    @_sub_doc(plat.add_vaccum)
    def add_vaccum(self, thickness, direction, left=False):
        return self.__class__(
            data=plat.add_vaccum(
                self._data, thickness=thickness, direction=direction, left=left
            )
        )

    @_sub_doc(plat.add_atoms)
    def add_atoms(self, name, positions):
        return self.__class__(
            data=plat.add_atoms(self._data, name=name, positions=positions)
        )

    @_sub_doc(plat.convert_poscar)
    def convert(self, atoms_mapping, basis_factor):
        return self.__class__(
            data=plat.convert_poscar(
                self._data, atoms_mapping=atoms_mapping, basis_factor=basis_factor
            )
        )

    @_sub_doc(plat.strain_poscar)
    def strain(self, strain_matrix):
        return self.__class__(
            data=plat.strain_poscar(self._data, strain_matrix=strain_matrix)
        )

    @_sub_doc(get_kmesh, {"poscar_data :.*\*args :": "*args :"})
    @_sig_kwargs(get_kmesh, ("poscar_data",))
    def get_kmesh(self, *args, **kwargs):
        return get_kmesh(self.data, *args, **kwargs)

    @_sub_doc(get_kpath, {"rec_basis :.*\n\n": "\n\n"})
    @_sig_kwargs(get_kpath, ("rec_basis",))
    def get_kpath(self, kpoints, n=5, **kwargs):
        return get_kpath(kpoints, n=n, **kwargs, rec_basis=self.data.rec_basis)
