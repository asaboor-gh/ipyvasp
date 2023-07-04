__all__ = [
    "dump",
    "load",
    "dict2tuple",
    "Dict2Data",
    "PoscarData",
    "BrZoneData",
    "CellData",
]

import json, re
import pickle
import inspect
from collections import namedtuple
from copy import deepcopy
from pathlib import Path

import numpy as np

from .spatial_toolkit import (
    angle_deg,
    to_basis,
    to_R3,
    kpoints2bz,
    order,
    ConvexHull,
)
from ..utils import _sub_doc


def dict2tuple(name: str, d: dict):
    """
    Converts a dictionary (nested as well) to namedtuple, accessible via index and dot notation as well as by unpacking.

    Parameters
    ----------
    name : str, Name of the tuple.
    d : dict, Dictionary, nested works as well.
    """
    return namedtuple(name, d.keys())(
        *(dict2tuple(k.upper(), v) if isinstance(v, dict) else v for k, v in d.items())
    )


class Dict2Data:
    """Creates a ``Data`` object with dictionary keys as attributes of Data accessible by dot notation or by key.
    Once an attribute is created, it can not be changed from outside.

    Parameters
    ----------
    d : dict
        Python dictionary (nested as well) containing any python data types.


    >>> x = Dict2Data({'A':1,'B':{'C':2}})
    >>> x
    Data(
        A = 1
        B = Data(
            C = 2
            )
        )
    >>> x.B.to_dict()
    {'C': 2}
    """

    _req_keys = ()
    _subclasses = ()

    def __init__(self, d):
        if not hasattr(self.__class__, "_req_keys"):
            raise AttributeError(
                "Derived class of `Dict2Data` should have attribute '_req_keys'"
            )
        if isinstance(d, (self.__class__, Dict2Data)):
            d = d.to_dict()  # if nested Dict2Data , must expand
        # Check if all required keys are present in main level of subclasses
        for key in self.__class__._req_keys:
            if key not in d:
                raise ValueError(f"Invalid input for {self.__class__.__name__}")
        # ===================
        for a, b in d.items():
            if isinstance(b, (self.__class__, Dict2Data)):
                b = b.to_dict()  # expands self instance !must here.

            if a == "poscar" and "metadata" in b:
                setattr(self, a, PoscarData(b))  # Enables custom methods for PoscarData
            elif isinstance(b, (list, tuple, set)):
                setattr(
                    self,
                    a,
                    tuple(Dict2Data(x) if isinstance(x, dict) else x for x in b),
                )
            else:
                setattr(self, a, Dict2Data(b) if isinstance(b, dict) else b)

    @classmethod
    def validated(cls, data):
        "Validate data like it's own or from json/pickle file/string."
        if type(data) is cls:  # if same type, return as is
            return data

        if isinstance(data, (str, bytes)):
            new_data = load(data)
            if not isinstance(new_data, cls):
                raise TypeError(f"Data is not of type {cls}.")
            return new_data

        if (
            isinstance(data, Dict2Data) and cls is not Dict2Data
        ):  # Check for other classes strictly
            data_keys = data.keys()
            for key in cls._req_keys:
                if key not in data_keys:
                    raise KeyError(f"Invalid data for {cls.__name__}")

        return cls(data)  # make of that type at end

    def to_dict(self):
        """Converts a `Dict2Data` object (root or nested level) to a dictionary."""
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, (self.__class__, Dict2Data)):
                result.update({k: Dict2Data.to_dict(v)})
            else:
                result.update({k: v})
        return deepcopy(result)  # prevent accidental changes in numpy arrays

    def copy(self):
        "Copy of self to avoid changes during inplace operations on numpy arrays."
        return self.__class__(
            self.to_dict()
        )  # make a copy of self through dictionary, otherwise it does not work

    def to_json(self, outfile: str = None, indent: int = 1):
        """
        Dumps a `Dict2Data` object (root or nested level) to json.

        Parameters
        ----------
        outfile : str, Default is None and returns string. If given, writes to file.
        indent : int, JSON indent. Default is 1.
        """
        return dump(self, format="json", outfile=outfile, indent=indent)

    def to_pickle(self, outfile: str = None):
        """
        Dumps a `Dict2Data` or subclass object (root or nested level) to pickle.

        Parameters
        ---------
        outfile : str, Default is None and returns string. If given, writes to file.
        """
        return dump(self, format="pickle", outfile=outfile)

    def to_tuple(self):
        """Creates a namedtuple."""
        return dict2tuple("Data", self.to_dict())

    def __repr__(self):
        items = []
        for k, v in self.__dict__.items():
            if type(v) not in (
                str,
                float,
                int,
                range,
                bool,
                None,
                True,
                False,
            ) and not isinstance(v, Dict2Data):
                if isinstance(v, np.ndarray):
                    v = "<{}:shape={}>".format(v.__class__.__name__, np.shape(v))
                elif type(v) in (list, tuple):
                    v = (
                        "<{}:len={}>".format(v.__class__.__name__, len(v))
                        if len(v) > 10
                        else v
                    )
                else:
                    v = v.__class__
            if isinstance(v, Dict2Data):
                v = repr(v).replace("\n", "\n    ")
            items.append(f"    {k} = {v}")
        name = (
            self.__class__.__name__ if self.__class__ is not Dict2Data else "Data"
        )  # auto handle derived classes
        return "{}(\n{}\n)".format(name, "\n".join(items))

    def __getstate__(self):
        pass  # This is for pickling

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise AttributeError(
                f"Outside assignment is restricted for already present attribute."
            )
        else:
            self.__dict__[name] = value

    # Dictionary-wise access
    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def __getitem__(self, key):
        return self.__dict__[key]

    def items(self):
        return self.__dict__.items()


class SpinData(Dict2Data):
    _req_keys = ("kpoints", "spins", "poscar")

    def __init__(self, d):
        super().__init__(d)
        self.sys_info.fermi = self.fermi

    def get_fermi(self, tol=1e-3):
        "Fermi energy based on occupancy. Returns `self.Fermi` if occupancies cannot be resolved. `tol` is the value of occupnacy to ignore as filled."
        try:
            return float(self.evals.e[self.evals.occs > tol].max())
        except:
            return self.Fermi

    @property
    def fermi(self):
        "Fermi energy based on occupancy. Use .get_fermi() if you want to limit the occupancy tolerance."
        return self.get_fermi(tol=1e-3)

    @property
    def Fermi(self):
        "Fermi energy given in vasprun.xml."
        return self.evals.Fermi


class PoscarData(Dict2Data):
    _req_keys = ("basis", "types", "metadata")

    def __init__(self, d):
        super().__init__(d)

    @property
    def coords(self):
        """Returns the lattice coordinates in cartesian space of the atoms in the poscar data."""
        return to_R3(self.basis, self.positions)

    @property
    def rec_basis(self):
        "Returns the reciprocal lattice basis of the atoms in the poscar data. Gets automatically updated when the lattice is changed."
        return np.linalg.inv(self.basis).T

    @property
    def norms(self):
        "Returns the norm of the lattice basis of the atoms in the poscar data. Gets automatically updated when the lattice is changed."
        return np.linalg.norm(self.basis, axis=1)

    @property
    def rec_norms(self):
        "Returns the norm of the reciprocal lattice basis of the atoms in the poscar data. Gets automatically updated when the lattice is changed."
        return np.linalg.norm(self.rec_basis, axis=1)

    @property
    def angles(self):
        "Returns the angles of the lattice basis of the atoms in the poscar data. Gets automatically updated when the lattice is changed."
        return np.array(
            [
                angle_deg(self.basis[2], self.basis[1]),
                angle_deg(self.basis[2], self.basis[0]),
                angle_deg(self.basis[1], self.basis[0]),
            ]
        )

    @property
    def rec_angles(self):
        "Returns the angles of reciprocal lattice basis of the atoms in the poscar data. Gets automatically updated when the lattice is changed."
        return np.array(
            [
                angle_deg(self.rec_basis[2], self.rec_basis[1]),
                angle_deg(self.rec_basis[2], self.rec_basis[0]),
                angle_deg(self.rec_basis[1], self.rec_basis[0]),
            ]
        )

    @property
    def volume(self):
        "Returns the volume of the lattice."
        return np.abs(
            np.linalg.det(self.basis)
        )  # Didn't think much if negative or positive

    @property
    def rec_volume(self):
        "Returns the volume of the reciprocal lattice."
        return np.abs(np.linalg.det(self.rec_basis))

    @property
    def labels(self):
        "Returns the labels of the atoms in the poscar data."
        if hasattr(
            self.metadata, "eqv_labels"
        ):  # If eqv_labels are present, return them
            return self.metadata.eqv_labels
        return np.array(
            [f"{k} {v - vs.start + 1}" for k, vs in self.types.items() for v in vs]
        )

    def get_distance(self, atom1, atom2):
        """
        Returns the distance between two atoms.
        Provide atom1 and atom2 as strings such as get_distance('Ga', 'As') to get a mimimal distance between two types
        or as a dict with a single key as get_distance({'Ga':0}, {'As':0}) to get distance between specific atoms,
        or mixed as get_distance('Ga', {'As':0}) to get minimum distance between a type and a specific atom.
        """
        idx1, idx2 = [], []
        if isinstance(atom1, str):
            idx1 = self.types[atom1]  # list or range
        elif isinstance(atom1, dict) and len(atom1) == 1:
            idx1 = [
                self.types[key][value] for key, value in atom1.items()
            ]  # keep as list
        else:
            raise ValueError(
                "atom1 must be a string such as 'Ga' or a dict with a single key as {'Ga':0}"
            )

        if isinstance(atom2, str):
            idx2 = self.types[atom2]
        elif isinstance(atom2, dict) and len(atom2) == 1:
            idx2 = [self.types[key][value] for key, value in atom2.items()]
        else:
            raise ValueError(
                "atom2 must be a string such as 'As' or a dict with a single key as {'As':0}"
            )

        dists = []
        for idx in idx1:
            dists = [
                *dists,
                *np.linalg.norm(
                    self.coords[tuple(idx2),] - self.coords[idx, :], axis=1
                ),
            ]  # Get the second closest distance, first is itself

        dists = np.array(dists)
        dists = dists[dists > 0]  # Remove distance with itself
        return np.min(dists)

    def get_selective_dynamics(self, func):
        """
        Returns a dictionary of {'Ga 1': 'T T T', 'As 1': 'T F F',...} for each atom in the poscar data.

        `func` should be a callable like `f(index,x,y,z) -> (bool, bool, bool)` which turns on/off selective dynamics for each atom based in each dimension.

        You can visualize selective dynamics sites by their labels as follows:


        >>> poscar = POSCAR.from_file('POSCAR')
        >>> sd = poscar.data.get_selective_dynamics(lambda i,x,y,z: (True, False, True) if i % 2 == 0 else (False, True, False)) # Just an example
        >>> poscar.splot_lattice(..., fmt_label = lambda lab: sd[lab]) # This will label sites as T T T, F F F, ... and so so on
        """
        if not callable(func):
            raise TypeError(
                "`func` should be a callable function(index, [point in fractional coordinates])!"
            )

        if len(inspect.signature(func).parameters) != 4:
            raise ValueError(
                "`func` should be a callable function with four paramters (index, x,y,z) in fractional coordinates."
            )

        test_output = func(0, 0, 0, 0)
        if (
            not isinstance(test_output, (list, tuple, np.ndarray))
            or len(test_output) != 3
        ):
            raise ValueError(
                "`func` should return a list/tuple/array of three booleans for each direction like (True, False, True)"
            )

        for out in test_output:
            if not isinstance(out, bool):
                raise ValueError(
                    "`func` should return boolean values in list/tuple/array like (True, False, True)"
                )

        sd_list = [
            "  ".join("T" if s else "F" for s in func(i, *p))
            for i, p in enumerate(self.positions)
        ]
        labels = np.array(
            [f"{k} {v - vs.start + 1}" for k, vs in self.types.items() for v in vs]
        )
        return {
            k: v for k, v in zip(labels, sd_list)
        }  # We can't use self.labels here because it can be for equivalent sites as well


class SpecialPoints(Dict2Data):
    _req_keys = ("coords", "kpoints")

    def __init__(self, d):
        super().__init__(d)

    def masked(self, func):
        "Returns a new SpecialPoints object with the mask applied. Example: func = lambda x,y,z: x > 0 where x,y,z are fractional kpoints coordinates, not cartesian."
        if not callable(func):
            raise ValueError(
                "func must be a callable function that returns a boolean and act on x,y,z fractional coordinates"
            )

        if len(inspect.signature(func).parameters) != 3:
            raise ValueError(
                "func takes exactly 3 arguments as x,y,z in fractional coordinates"
            )

        if not isinstance(func(0, 0, 0), bool):
            raise ValueError("func must return a boolean")

        mask = [func(x, y, z) for x, y, z in self.kpoints]
        return SpecialPoints(
            {"coords": self.coords[mask], "kpoints": self.kpoints[mask]}
        )


def _methods_imported():
    # These imports work as methods of the class
    from .._lattice import splot_bz as splot, iplot_bz as iplot  # Avoid circular import

    splot.__doc__ = re.sub(  # This replaces orginal docstring too. That is strange
        "bz_data :.*plane :", "plane :", splot.__doc__, flags=re.DOTALL
    )
    iplot.__doc__ = re.sub(
        "bz_data :.*fill :", "fill :", iplot.__doc__, flags=re.DOTALL
    )
    return splot, iplot


class BrZoneData(Dict2Data):
    splot, iplot = _methods_imported()
    from .._lattice import splot_kpath  # no change in this

    _req_keys = ("basis", "faces", "vertices", "primitive")

    def __init__(self, d):
        super().__init__(d)

    def get_special_points(self, orderby=(1, 1, 1)):
        """Returns the special points in the brillouin zone in the order relative
        to a given point in cartesian coordinates. Gamma is always first."""
        if self.primitive:
            return SpecialPoints(
                {"coords": np.empty((0, 3)), "kpoints": np.empty((0, 3))}
            )  # Primitive Zone has no meaning for special points

        if hasattr(self, "_specials"):  # Transformed BZ
            return self._specials

        mid_faces = np.array(
            [np.mean(np.unique(face, axis=0), axis=0) for face in self.faces_coords]
        )
        mid_edges = []
        for face in self.faces_coords:
            for f, g in zip(face[:-1], face[1:]):
                # NOTE: Do not insert point between unique vertices, is it necessary?
                if np.isclose(np.linalg.norm(f), np.linalg.norm(g)):
                    mid_edges.append(np.mean([f, g], axis=0))

        if mid_edges != []:
            mid_edges = np.unique(mid_edges, axis=0)  # because faces share edges
            mid_faces = np.concatenate([mid_faces, mid_edges])

        # Bring all high symmetry points together.
        sp_carts = np.concatenate(
            [mid_faces, self.vertices]
        )  # Coords, Gamma should be there
        sp_basis = np.array(
            [np.linalg.solve(self.basis.T, v) for v in sp_carts]
        )  # Kpoints

        order = np.linalg.norm(
            sp_carts - orderby, axis=1
        )  # order by cartesian distance, so it appears where it looks
        order = np.argsort(order)
        sp_carts = np.insert(
            sp_carts[order], 0, np.zeros(3), axis=0
        )  # Gamma should be first
        sp_basis = np.insert(
            sp_basis[order], 0, np.zeros(3), axis=0
        )  # Gamma should be first

        return SpecialPoints({"coords": sp_carts, "kpoints": sp_basis})

    @property
    def specials(self):
        "Returns the special points in the brillouin zone ordered by point (1,1,1) in cartesian coordinates. Gamma is always first."
        return self.get_special_points()

    @property
    def faces_coords(self):
        "Returns the coordinates of the faces of the brillouin zone in list of N faces of shape (M,3) where M is the number of vertices of the face."
        return tuple(
            self.vertices[(face,)] for face in self.faces
        )  # (face,) is to pick items from first dimension, face would try many dimensions

    @property
    def normals(self):
        "Get normal vectors to the faces of BZ. Returns a tuple of 6 arrays as (X,Y,Z,U,V,W) where (X,Y,Z) is the center of the faces and (U,V,W) is the normal direction."
        faces = self.faces_coords  # get once
        centers = np.array(
            [np.mean(np.unique(face, axis=0), axis=0) for face in faces]
        )  # unique is must to avoid a duplicate point at end
        other_points = []
        for center, face in zip(centers, faces):
            a = face[0] - center
            b = face[1] - center
            perp = np.cross(a, b)  # a and b are anti-clockwise
            perp = perp / np.linalg.norm(perp)  # Normalize
            other_points.append(perp)

        other_points = np.array(other_points)
        return namedtuple("Normals", ["X", "Y", "Z", "U", "V", "W"])(
            *centers.T, *other_points.T
        )  # Keep this as somewhere it will be used _asdict()

    def to_fractional(self, coords):
        "Converts cartesian coordinates to fractional coordinates in the basis of the brillouin zone."
        return to_basis(self.basis, coords)

    def to_cartesian(self, points):
        "Converts fractional coordinates in the basis of the brillouin zone to cartesian coordinates."
        return to_R3(self.basis, points)

    def map_kpoints(self, other, kpoints):
        """Map kpoints (fractional) from this to other Brillouin zone.
        This operation is useful when you do POSCAR.transform() and want to map kpoints between given and transformed BZ.
        """
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"other must be a {self.__class__} object, got {type(other)}"
            )
        return other.to_fractional(self.to_cartesian(kpoints))

    @_sub_doc(kpoints2bz, {"bz_data :.*kpoints :": "kpoints :"})
    def translate_inside(self, kpoints, shift=0, keep_geometry=False):
        return kpoints2bz(self, kpoints, shift=shift, keep_geomerty=keep_geometry)

    def subzone(self, func):
        """Returns a subzone of the brillouin zone by applying a function on the fractional special points of the zone.

        .. warning::
            We do not check if output is an irreducible zone or not. It is just a subzone based on the function.
        """
        if not callable(func):
            raise TypeError(f"func must be callable, got {type(func)}")

        if not isinstance(func(0, 0, 0), bool):
            raise TypeError(f"func must return bool, got {type(func(0,0,0))}")

        spoints = self.specials.kpoints
        fverts = []
        for vert in spoints:
            if func(*vert):
                fverts.append(vert)

        cverts = self.to_cartesian(fverts)

        chull = ConvexHull(cverts)
        vertices = cverts[chull.vertices]  # those are indices

        vertidx = chull.vertices.tolist()
        faces = []
        for sim in chull.simplices.tolist():
            face = [vertidx.index(s) for s in sim]
            faces.append(face)

        # TODO: Merge faces if they share an edge and are coplanar

        d = self.copy().to_dict()
        d.update({"faces": faces, "vertices": vertices})
        d["_specials"] = SpecialPoints({"kpoints": np.array(fverts), "coords": cverts})
        return self.__class__(d)


class CellData(Dict2Data):
    splot, iplot = _methods_imported()
    _req_keys = ("basis", "faces", "vertices")

    def __init__(self, d):
        super().__init__({k: v for k, v in d.items() if k != "primitive"})

    @property
    def faces_coords(self):
        "Returns the coordinates of the faces of the cell in list of N faces of shape (M,3) where M is the number of vertices of the face."
        return BrZoneData.faces_coords.fget(self)

    @property
    def normals(self):
        "Get normal vectors to the faces of Cell. Returns a tuple of 6 arrays as (X,Y,Z,U,V,W) where (X,Y,Z) is the center of the faces and (U,V,W) is the normal direction."
        return BrZoneData.normals.fget(self)

    def to_fractional(self, coords):
        "Converts cartesian coordinates to fractional coordinates in the basis of the cell."
        return BrZoneData.to_fractional(self, coords)

    def to_cartesian(self, points):
        "Converts fractional coordinates in the basis of the cell to cartesian coordinates."
        return BrZoneData.to_cartesian(self, points)


class GridData(Dict2Data):
    _req_keys = ("path", "poscar", "SYSTEM")

    def __init__(self, d):
        super().__init__(d)

    @property
    def coords(self):
        """
        Returns coordinates of the grid points in shape (3,Nx, Ny,Nz) given by equation

        .. math::
            (x,y,z) = \\frac{i}{N_x}a + \\frac{j}{N_y}b + \\frac{k}{N_z}c

        where (a,b,c) are lattice vectors. and i,j,k are the grid indices as in intervals [0, Nx-1], [0, Ny-1], [0, Nz-1].
        """
        shape = self.values.shape
        Nx, Ny, Nz = shape
        ix, iy, iz = np.indices(shape)
        a1, a2, a3 = self.poscar.basis
        return np.array(
            [
                ix * a1[0] / Nx + iy * a2[0] / Ny + iz * a3[0] / Nz,
                ix * a1[1] / Nx + iy * a2[1] / Ny + iz * a3[1] / Nz,
                ix * a1[2] / Nx + iy * a2[2] / Ny + iz * a3[2] / Nz,
            ]
        )


class OutcarData(Dict2Data):
    _req_keys = ("site_pot", "ion_pot", "basis")

    def __init__(self, d):
        super().__init__(d)

    def masked(self, func):
        "Returns a data with only the sites given by mask function over fractional coordinates, e.g. func = lambda x, y, z: x == 1"
        if not callable(func):
            raise TypeError("func must be callable like lambda x,y,z: x == 1")

        if len(inspect.signature(func).parameters) != 3:
            raise ValueError(
                "func takes exactly 3 arguments as x,y,z in fractional coordinates"
            )

        if not isinstance(func(0, 0, 0), bool):
            raise ValueError("func must return a boolean value")

        raise NotImplementedError("Not implemented yet")
        return  # ion_pot, site_pot, rename these with better names


class EncodeFromNumpy(json.JSONEncoder):
    """
    Serializes python/Numpy objects via customizing json encoder.


    >>> json.dumps(python_dict, cls=EncodeFromNumpy) # to get json string.
    >>> json.dump(*args, cls=EncodeFromNumpy) # to create a file.json.
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {"_kind_": "ndarray", "_value_": obj.tolist()}
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, range):
            value = list(obj)
            return {"_kind_": "range", "_value_": [value[0], value[-1] + 1]}
        return super(EncodeFromNumpy, self).default(obj)


class DecodeToNumpy(json.JSONDecoder):
    """
    Deserilizes JSON object to Python/Numpy's objects.


    >>> json.loads(json_string,cls=DecodeToNumpy) #  from string
    >>> json.load(path) # from file.
    """

    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if "_kind_" not in obj:
            return obj
        kind = obj["_kind_"]
        if kind == "ndarray":
            return np.array(obj["_value_"])
        elif kind == "range":
            value = obj["_value_"]
            return range(value[0], value[-1])
        return obj


def dump(data, format: str = "pickle", outfile: str = None, indent: int = 1) -> None:
    """Dump ``Dict2Data`` or subclass object or any dictionary to json or pickle string/file.

    Parameters
    ----------
    data : dict or instance of Dict2Data
        Any dictionary/Dict2Data(or subclass Data) object can be saved.
    format : str,
        Defualt is ``pickle``. Should be ``pickle`` or ``json``.
    outfile : str
        Defualt is None and return string. File name does not require extension as it is added from ``format``.
    indent : int
        Defualt is 1. Only works for json.
    """
    if format not in ["pickle", "json"]:
        raise ValueError("`format` expects 'pickle' or 'json', got '{}'".format(format))
    try:
        dict_obj = data.to_dict()  # Change Data object to dictionary
        dict_obj = {
            "_loader_": data.__class__.__name__,
            "_data_": dict_obj,
        }  # Add class name to dictionary for reconstruction
    except:
        dict_obj = data
    if format == "pickle":
        if outfile == None:
            return pickle.dumps(dict_obj)
        outfile = Path(outfile).stem + ".pickle"
        with open(outfile, "wb") as f:
            pickle.dump(dict_obj, f)
    if format == "json":
        if outfile == None:
            return json.dumps(dict_obj, cls=EncodeFromNumpy, indent=indent)
        outfile = Path(outfile).stem + ".json"
        with open(outfile, "w") as f:
            json.dump(dict_obj, f, cls=EncodeFromNumpy, indent=indent)
    return None


def load(file_or_str: str):
    """
    Loads a json/pickle dumped file or string by auto detecting it.

    Parameters
    ----------
    file_or_str : str, Filename of pickl/json or their string.
    """
    out = {}
    if not isinstance(file_or_str, bytes):
        try:  # must try, else fails due to path length issue
            if (p := Path(file_or_str)).is_file():
                if ".pickle" in p.suffix:
                    with p.open("rb") as f:
                        out = pickle.load(f)

                elif ".json" in p.suffix:
                    with p.open("r") as f:
                        out = json.load(f, cls=DecodeToNumpy)

            else:
                out = json.loads(file_or_str, cls=DecodeToNumpy)
            # json.loads required in else and except both as long str > 260 causes issue in start of try block
        except:
            out = json.loads(file_or_str, cls=DecodeToNumpy)
    elif isinstance(file_or_str, bytes):
        out = pickle.loads(file_or_str)

    if type(out) is dict:
        if "_loader_" in out:
            return globals()[out["_loader_"]](out["_data_"])
    else:
        if hasattr(out, "_loader_"):
            return globals()[out._loader_](out._data_)

    return out  # Retruns usual dictionaries
