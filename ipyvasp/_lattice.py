import re
import json
import numpy as np
from pathlib import Path
import requests as req
import inspect
from itertools import combinations, product
from functools import lru_cache
from typing import NamedTuple

from scipy.spatial import ConvexHull, KDTree
import plotly.graph_objects as go

import matplotlib.pyplot as plt  # For viewpoint
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.colors as mplc

from ipywidgets import interactive, IntSlider

# Inside packages import
from .core import plot_toolkit as ptk
from .core import parser as vp, serializer
from .core.spatial_toolkit import (
    to_plane,
    rotation,
    inside_convexhull, # be there for export
    to_basis,
    to_R3,
    get_TM,
    get_bz,
    coplanar,
)
from .core.plot_toolkit import quiver3d
from .utils import color as tcolor


# These colors are taken from Mathematica's ColorData["Atoms"]
_atom_colors = {
    "H": (0.7, 0.8, 0.7),
    "He": (0.8367, 1.0, 1.0),
    "Li": (0.7994, 0.9976, 0.5436),
    "Be": (0.7706, 0.0442, 0.9643),
    "B": (1.0, 0.5, 0),
    "C": (0.4, 0.4, 0.4),
    "N": (143 / 255, 143 / 255, 1),
    "O": (0.8005, 0.1921, 0.2015),
    "F": (128 / 255, 1, 0),
    "Ne": (0.6773, 0.9553, 0.9284),
    "Na": (0.6587, 0.8428, 0.4922),
    "Mg": (0.6283, 0.0783, 0.8506),
    "Al": (173 / 255, 178 / 255, 189 / 255),
    "Si": (248 / 255, 209 / 255, 152 / 255),
    "P": (1, 165 / 255, 0),
    "S": (1, 200 / 255, 50 / 255),
    "Cl": (0, 0.9, 0),
    "Ar": (0.5461, 0.8921, 0.8442),
    "K": (0.534, 0.7056, 0.4207),
    "Ca": (0.4801, 0.0955, 0.7446),
    "Sc": (0.902, 0.902, 0.902),
    "Ti": (0.749, 0.7804, 0.7608),
    "V": (0.651, 0.6706, 0.651),
    "Cr": (0.5412, 0.7804, 0.6),
    "Mn": (0.6118, 0.7804, 0.4784),
    "Fe": (0.32, 0.33, 0.35),
    "Co": (0.9412, 0.6275, 0.5647),
    "Ni": (141 / 255, 142 / 255, 140 / 255),
    "Cu": (184 / 255, 115 / 255, 51 / 255),
    "Zn": (186 / 255, 196 / 255, 200 / 255),
    "Ga": (90 / 255, 180 / 255, 189 / 255),
    "Ge": (0.6051, 0.5765, 0.6325),
    "As": (50 / 255, 71 / 255, 57 / 255),
    "Se": (0.9172, 0.0707, 0.6578),
    "Br": (161 / 255, 61 / 255, 45 / 255),
    "Kr": (0.426, 0.8104, 0.7475),
    "Rb": (0.4254, 0.5859, 0.3292),
    "Sr": (0.326, 0.096, 0.6464),
    "Y": (0.531, 1.0, 1.0),
    "Zr": (0.4586, 0.9186, 0.9175),
    "Nb": (0.385, 0.8417, 0.8349),
    "Mo": (0.3103, 0.7693, 0.7522),
    "Tc": (0.2345, 0.7015, 0.6694),
    "Ru": (0.1575, 0.6382, 0.5865),
    "Rh": (0.0793, 0.5795, 0.5036),
    "Pd": (0.0, 0.5252, 0.4206),
    "Ag": (0.7529, 0.7529, 0.7529),
    "Cd": (0.8, 0.67, 0.73),
    "In": (228 / 255, 228 / 255, 228 / 255),
    "Sn": (0.398, 0.4956, 0.4915),
    "Sb": (158 / 255, 99 / 255, 181 / 255),
    "Te": (0.8167, 0.0101, 0.4513),
    "I": (48 / 255, 25 / 255, 52 / 255),
    "Xe": (0.3169, 0.7103, 0.6381),
    "Cs": (0.3328, 0.4837, 0.2177),
    "Ba": (0.1659, 0.0797, 0.556),
    "La": (0.9281, 0.3294, 0.7161),
    "Ce": (0.8948, 0.3251, 0.7314),
    "Pr": (0.8652, 0.3153, 0.708),
    "Nd": (0.8378, 0.3016, 0.663),
    "Pm": (0.812, 0.2856, 0.6079),
    "Sm": (0.7876, 0.2683, 0.5499),
    "Eu": (0.7646, 0.2504, 0.4933),
    "Gd": (0.7432, 0.2327, 0.4401),
    "Tb": (0.7228, 0.2158, 0.3914),
    "Dy": (0.7024, 0.2004, 0.3477),
    "Ho": (0.68, 0.1874, 0.3092),
    "Er": (0.652, 0.1778, 0.2768),
    "Tm": (0.6136, 0.173, 0.2515),
    "Yb": (0.5579, 0.1749, 0.2346),
    "Lu": (0.4757, 0.1856, 0.2276),
    "Hf": (0.7815, 0.7166, 0.7174),
    "Ta": (0.7344, 0.6835, 0.5445),
    "W": (0.6812, 0.6368, 0.3604),
    "Re": (0.6052, 0.5563, 0.3676),
    "Os": (0.5218, 0.4692, 0.3821),
    "Ir": (0.4456, 0.3991, 0.3732),
    "Pt": (0.8157, 0.8784, 0.8157),
    "Au": (0.8, 0.7, 0.2),
    "Hg": (0.7216, 0.8157, 0.7216),
    "Tl": (0.651, 0.302, 0.3294),
    "Pb": (0.3412, 0.3804, 0.349),
    "Bi": (10 / 255, 49 / 255, 93 / 255),
    "Po": (0.6706, 0.0, 0.3608),
    "At": (0.4588, 0.2706, 0.3098),
    "Rn": (0.2188, 0.5916, 0.5161),
    "Fr": (0.2563, 0.3989, 0.0861),
    "Ra": (0.0, 0.0465, 0.4735),
    "Ac": (0.322, 0.9885, 0.7169),
    "Th": (0.3608, 0.943, 0.6717),
    "Pa": (0.3975, 0.8989, 0.628),
    "U": (0.432, 0.856, 0.586),
    "Np": (0.4645, 0.8145, 0.5455),
    "Pu": (0.4949, 0.7744, 0.5067),
    "Am": (0.5233, 0.7355, 0.4695),
    "Cm": (0.5495, 0.698, 0.4338),
    "Bk": (0.5736, 0.6618, 0.3998),
    "Cf": (0.5957, 0.6269, 0.3675),
    "Es": (0.6156, 0.5934, 0.3367),
    "Fm": (0.6335, 0.5612, 0.3075),
    "Md": (0.6493, 0.5303, 0.2799),
    "No": (0.663, 0.5007, 0.254),
    "Lr": (0.6746, 0.4725, 0.2296),
    "Rf": (0.6841, 0.4456, 0.2069),
    "Db": (0.6915, 0.42, 0.1858),
    "Sg": (0.6969, 0.3958, 0.1663),
    "Bh": (0.7001, 0.3728, 0.1484),
    "Hs": (0.7013, 0.3512, 0.1321),
    "Mt": (0.7004, 0.331, 0.1174),
    "Ds": (0.6973, 0.312, 0.1043),
    "Rg": (0.6922, 0.2944, 0.0928),
    "Cn": (0.6851, 0.2781, 0.083),
    "Nh": (0.6758, 0.2631, 0.0747),
    "Fl": (0.6644, 0.2495, 0.0681),
    "Mc": (0.6509, 0.2372, 0.0631),
    "Lv": (0.6354, 0.2262, 0.0597),
    "Ts": (0.6354, 0.2262, 0.0566),
    "Og": (0.6354, 0.2262, 0.0528),
}

_atom_numbers = {k: i for i, k in enumerate(_atom_colors.keys())}


def atomic_number(atom):
    "Return atomic number of atom"
    return _atom_numbers[atom]


def atoms_color():
    "Defualt color per atom used for plotting the crystal lattice"
    return serializer.Dict2Data(
        {k: [round(_v, 4) for _v in rgb] for k, rgb in _atom_colors.items()}
    )


def periodic_table(selection=None):
    "Display colorerd elements in periodic table. Use a list of atoms to only color a selection."
    _copy_names = np.array(
        [f"$^{{{str(i+1)}}}${k}" for i, k in enumerate(_atom_colors.keys())]
    )
    blank = []
    if isinstance(selection,(list, tuple, str)):
        if isinstance(selection, str):
            selection = selection.split()
        blank = [key for key in _atom_colors if not (key in selection)]   

    _copy_array = np.array([[1,1,1,0] if key in blank else [*value,1]  for key, value in _atom_colors.items()])

    names = ["" for i in range(180)]  # keep as list before modification
    fc = np.ones((180, 4))
    ec = np.zeros((180,3)) + (0.4 if blank else 0.9 )
    offsets = np.array([[(i,j) for i in range(18)] for j in range(10)]).reshape((-1,2)) - 0.5

    inds = np.array([
        (0, 0),
        (17, 1),
        (18, 2),
        (19, 3),
        *[(30 + i, 4 + i) for i in range(8)],
        *[(48 + i, 12 + i) for i in range(6)],
        *[(54 + i, 18 + i) for i in range(18)],
        *[(72 + i, 36 + i) for i in range(18)],
        *[(90 + i, 54 + i) for i in range(3)],
        *[(93 + i, 71 + i) for i in range(15)],
        *[(108 + i, 86 + i) for i in range(3)],
        *[(111 + i, 103 + i) for i in range(15)],
        *[(147 + i, 57 + i) for i in range(14)],
        *[(165 + i, 89 + i) for i in range(14)],
    ], dtype=int)

    for i, j in inds:
        fc[i,:] = _copy_array[j]
        names[i] = _copy_names[j]

    fidx = [i for i, _ in inds] # only plot at elements posistions,otherwise they overlap
    offsets = offsets[fidx]
    fc, ec = fc[fidx], ec[fidx]
    names = np.array(names)[fidx]
    
    # We are adding patches, because imshow does not properly appear in PDF of latex
    ax = ptk.get_axes(1, (7, 3.9),left=0.01,right=0.99,top=0.99,bottom=0.01)
    patches = np.array([Rectangle(offset,0.9 if i in [92,110] else 1,1) for i, offset in zip(fidx,offsets)])
    pc = PatchCollection(patches, facecolors=fc, edgecolors=ec,linewidths=(0.7,))
    ax.add_collection(pc)
    
    for (x,y), text, c in zip(offsets + 0.5, names, fc):
        c = "k" if np.linalg.norm(c[:3]) > 1 else "w"
        plt.text(x,y, text, color=c, ha="center", va="center")

    ax.set_axis_off()
    ax.set(xlim=[-0.6,17.6],ylim=[9.6,-0.6]) # to show borders correctly
    return ax

def _write_text(dest, text: str, *, encoding: str = "utf-8") -> None:
    "Write unicode text either to a path-like destination or to a writable text stream."
    # Treat file-like objects (streams) first (avoid Path("CON") / weird Windows devices, etc.)
    if hasattr(dest, "write") and callable(getattr(dest, "write")):
        dest.write(text)
        # Best-effort flush (sys.stdout has it, StringIO doesn't need it)
        flush = getattr(dest, "flush", None)
        if callable(flush):
            flush()
        return

    # Otherwise treat as a filesystem path
    path = Path(dest)
    with path.open("w", encoding=encoding) as f:
        f.write(text)

def write_poscar(poscar_data, outfile=None, selective_dynamics=None, overwrite=False, comment="", scale=None, system=None):
    """Writes POSCAR data to a file or returns string

    Parameters
    ----------
    outfile : PathLike
    selective_dynamics : callable
        If given, should be a function like `f(a) -> (a.p < 1/4)` or `f(a) -> (a.x < 1/4, a.y < 1/4, a.z < 1/4)` 
        which turns on/off selective dynamics for each atom based in each dimension.
        See `ipyvasp.POSCAR.data.get_selective_dynamics` for more info.
    overwrite: bool
        If file already exists, overwrite=True changes it.
    comment: str
        Add comment, previous comment will be there too.
    scale: float
        Scale factor for the basis vectors. Default is provided by loaded data.
    system: str
        System name to be used in POSCAR file instead of the one in `poscar_data.SYSTEM`.


    .. note::
        POSCAR is only written in direct format even if it was loaded from cartesian format.
    """
    _comment = poscar_data.metadata.comment + comment
    out_str = f"{system or poscar_data.SYSTEM}  # " + (_comment or "Created by ipyvasp")

    if scale is None:
        scale = poscar_data.metadata.scale
    elif not isinstance(scale, (int, float)):
        raise TypeError("scale must be a number or None.")
    elif scale == 0:
        raise ValueError("scale can not be zero.")
    
    out_str += "\n  {:<20.14f}\n".format(scale)
    out_str += "\n".join(
        ["{:>22.16f}{:>22.16f}{:>22.16f}".format(*a) for a in poscar_data.basis / scale]
    )
    uelems = poscar_data.types.to_dict()
    out_str += "\n  " + "    ".join(uelems.keys())
    out_str += "\n  " + "    ".join([str(len(v)) for v in uelems.values()])

    if selective_dynamics is not None:
        out_str += "\nSelective Dynamics"

    out_str += "\nDirect\n"
    positions = poscar_data.positions
    pos_list = ["{:>21.16f}{:>21.16f}{:>21.16f}".format(*a) for a in positions]

    if selective_dynamics is not None:
        sd = poscar_data.get_selective_dynamics(selective_dynamics).values()
        pos_list = [f"{p}   {s}" for p, s in zip(pos_list, sd)]

    out_str += "\n".join(pos_list)
    if outfile is not None:
        # If it's a writable stream (sys.stdout, StringIO, open file handle), write directly.
        if hasattr(outfile, "write") and callable(getattr(outfile, "write")):
            _write_text(outfile, out_str)
        else:
            # Otherwise treat as path-like with overwrite protection.
            path = Path(outfile)
            if path.exists() and not overwrite:
                raise FileExistsError(
                    f"{str(path)!r} exists, can not overwrite; use overwrite=True."
                )
            _write_text(path, out_str)
    else:
        print(out_str)


def export_poscar(path=None, content=None):
    """Export POSCAR file to python objects.

    Parameters
    ----------
    path : PathLike
        Path/to/POSCAR file. Auto picks in CWD.
    content : str
        POSCAR content as string, This takes precedence to path.
    """
    if content and isinstance(content, str):
        file_lines = [f"{line}\n" for line in content.splitlines()]
    else:
        P = Path(path or "./POSCAR")
        if not P.is_file():
            raise FileNotFoundError(f"{str(P)} not found.")

        with P.open("r", encoding="utf-8") as f:
            file_lines = f.readlines()

    header = file_lines[0].split("#", 1)
    SYSTEM = header[0].strip()
    comment = header[1].strip() if len(header) > 1 else "Exported by Pivopty"

    scale = float(file_lines[1].strip().split()[0]) # some people add comments here too
    if scale < 0:  # If that is for volume
        scale = 1

    basis = scale * vp.gen2numpy(file_lines[2:5], (3, 3), [-1, -1], exclude=None)
    # volume = np.linalg.det(basis)
    # rec_basis = np.linalg.inv(basis).T # general formula
    out_dict = {
        "SYSTEM": SYSTEM,  #'volume':volume,
        "basis": basis,  #'rec_basis':rec_basis,
        "metadata": {"comment": comment, "scale": scale},
    }

    elems = file_lines[5].split()
    ions = [int(i) for i in file_lines[6].split()]
    N = int(np.sum(ions))  # Must be py int, not numpy
    inds = np.cumsum([0, *ions]).astype(int)
    # Check Cartesian and Selective Dynamics
    lines = [l.strip() for l in file_lines[7:9]]  # remove whitespace or tabs
    out_dict["metadata"]["cartesian"] = (
        True if ((lines[0][0] in "cCkK") or (lines[1][0] in "cCkK")) else False
    )

    poslines = vp.gen2numpy(
        file_lines[7:],
        (N, 6),
        (-1, [0, 1, 2]),
        exclude="^\s+[a-zA-Z]|^[a-zA-Z]",
        raw=True,
    ).splitlines()  # handle selective dynamics word here
    positions = np.array(
        [line.split()[:3] for line in poslines], dtype=float
    )  # this makes sure only first 3 columns are taken

    if out_dict["metadata"]["cartesian"]:
        positions = scale * to_basis(basis, positions)
        print(("Cartesian format found in POSCAR file, converted to direct format."))

    unique_d = {}
    for i, e in enumerate(elems):
        unique_d.update({e: range(inds[i], inds[i + 1])})

    elem_labels = []
    for i, name in enumerate(elems):
        for ind in range(inds[i], inds[i + 1]):
            elem_labels.append(f"{name} {str(ind - inds[i] + 1)}")
    out_dict.update({"positions": positions, "types": unique_d})  #'labels':elem_labels,
    return serializer.PoscarData(out_dict)


# Cell
def _save_mp_API(api_key):
    "Save materials project api key for autoload in functions. This works only for legacy API."
    path = Path.home() / ".ipyvasprc"
    lines = []
    if path.is_file():
        with path.open("r") as fr:
            lines = fr.readlines()
            lines = [line for line in lines if "MP_API_KEY" not in line]

    with path.open("w") as fw:
        fw.write("MP_API_KEY = {}".format(api_key))
        for line in lines:
            fw.write(line)


# Cell
def _load_mp_data(formula, api_key=None, mp_id=None, max_sites=None, min_sites=None):
    if api_key is None:
        try:
            path = Path.home() / ".ipyvasprc"
            with path.open("r") as f:
                lines = f.readlines()
                for line in lines:
                    if "MP_API_KEY" in line:
                        api_key = line.split("=")[1].strip()
        except:
            raise ValueError(
                "api_key not given. provide in argument or generate in file using `_save_mp_API(your_mp_api_key)"
            )

    # url must be a raw string
    url = r"https://legacy.materialsproject.org/rest/v2/materials/{}/vasp?API_KEY={}".format(
        formula, api_key
    )
    resp = req.request(method="GET", url=url)
    if resp.status_code != 200:
        raise ValueError("Error in fetching data from materials project. Try again!")

    jl = json.loads(resp.text)
    if not "response" in jl:  # check if response
        raise ValueError("Either formula {!r} or API_KEY is incorrect.".format(formula))

    all_res = jl["response"]

    if max_sites != None and min_sites != None:
        lower, upper = min_sites, max_sites
    elif max_sites == None and min_sites != None:
        lower, upper = min_sites, min_sites + 1
    elif max_sites != None and min_sites == None:
        lower, upper = max_sites - 1, max_sites
    else:
        lower, upper = "-1", "-1"  # Unknown

    if lower != "-1" and upper != "-1":
        sel_res = []
        for res in all_res:
            if res["nsites"] <= upper and res["nsites"] >= lower:
                sel_res.append(res)
        return sel_res
    # Filter to mp_id at last. more preferred
    if mp_id != None:
        for res in all_res:
            if mp_id == res["material_id"]:
                return [res]
    return all_res


def _cif_str_to_poscar_str(cif_str, comment=None):
    # Using it in other places too
    lines = [
        line for line in cif_str.splitlines() if line.strip()
    ]  # remove empty lines

    abc = []
    abc_ang = []
    index = 0
    for ys in lines:
        if "_cell" in ys:
            if "_length" in ys:
                abc.append(ys.split()[1])
            if "_angle" in ys:
                abc_ang.append(ys.split()[1])
            if "_volume" in ys:
                volume = float(ys.split()[1])
        if "_structural" in ys:
            top = ys.split()[1] + f" # {comment}" if comment else ys.split()[1]
    for i, ys in enumerate(lines):
        if "_atom_site_occupancy" in ys:
            index = i + 1  # start collecting pos.
    poses = lines[index:]
    pos_str = ""
    for pos in poses:
        s_p = pos.split()
        pos_str += "{0:>12}  {1:>12}  {2:>12}  {3}\n".format(*s_p[3:6], s_p[0])

    names = [re.sub("\d+", "", pos.split()[1]).strip() for pos in poses]
    types = []
    for name in names:
        if name not in types:
            types.append(name)  # unique types, don't use numpy here.

    # ======== Cleaning ===========
    abc_ang = [float(ang) for ang in abc_ang]
    abc = [float(a) for a in abc]
    a = "{0:>22.16f} {1:>22.16f} {2:>22.16f}".format(1.0, 0.0, 0.0)  # lattic vector a.
    to_rad = 0.017453292519
    gamma = abc_ang[2] * to_rad
    bx, by = abc[1] * np.cos(gamma), abc[1] * np.sin(gamma)
    b = "{0:>22.16f} {1:>22.16f} {2:>22.16f}".format(
        bx / abc[0], by / abc[0], 0.0
    )  # lattic vector b.
    cz = volume / (abc[0] * by)
    cx = abc[2] * np.cos(abc_ang[1] * to_rad)
    cy = (abc[1] * abc[2] * np.cos(abc_ang[0] * to_rad) - bx * cx) / by
    c = "{0:>22.16f} {1:>22.16f} {2:>22.16f}".format(
        cx / abc[0], cy / abc[0], cz / abc[0]
    )  # lattic vector b.

    elems = "\t".join(types)
    nums = [str(len([n for n in names if n == t])) for t in types]
    nums = "\t".join(nums)
    content = (
        f"{top}\n  {abc[0]}\n {a}\n {b}\n {c}\n  {elems}\n  {nums}\nDirect\n{pos_str}"
    )
    return content


class InvokeMaterialsProject:
    """Connect to materials project and get data using `api_key` from their site.

    Usage
    -----
    >>> from ipyvaspr.sio import InvokeMaterialsProject # or import ipyvasp.InvokeMaterialsProject as InvokeMaterialsProject
    >>> mp = InvokeMaterialsProject(api_key='your_api_key')
    >>> outputs = mp.request(formula='NaCl') #returns list of structures from response
    >>> outupts[0].export_poscar() #returns poscar data
    >>> outputs[0].cif #returns cif data
    """

    def __init__(self, api_key=None):
        "Request Materials Project acess. api_key is on their site. Your only need once and it is saved for later."
        self.api_key = api_key
        self.__response = None
        self.success = False

    def save_api_key(self, api_key):
        "Save api_key for auto reloading later."
        _save_mp_API(api_key)

    @lru_cache(maxsize=2)  # cache for 2 calls
    def request(self, formula, mp_id=None, max_sites=None, min_sites=None):
        "Fetch data using request api of python form materials project website. After request, you can access `cifs` and `poscars`."
        self.__response = _load_mp_data(
            formula=formula,
            api_key=self.api_key,
            mp_id=mp_id,
            max_sites=max_sites,
            min_sites=min_sites,
        )
        if self.__response == []:
            raise req.HTTPError("Error in request. Check your api_key or formula.")

        class Structure:
            def __init__(self, response):
                self._cif = response["cif"]
                self.symbol = response["spacegroup"]["symbol"]
                self.crystal = response["spacegroup"]["crystal_system"]
                self.unit = response["unit_cell_formula"]
                self.mp_id = response["material_id"]

            @property
            def cif(self):
                return self._cif

            def __repr__(self):
                return f"Structure(unit={self.unit},mp_id={self.mp_id!r},symbol={self.symbol!r},crystal={self.crystal!r},cif='{self._cif[:10]}...')"

            def write_cif(self, outfile=None):
                if outfile is not None:
                    _write_text(outfile, self._cif)
                else:
                    print(self._cif)

            def write_poscar(self, outfile=None, overwrite=False, comment="",scale=None):
                "Use `ipyvasp.lattice.POSCAR.write` if you need extra options."
                write_poscar(self.export_poscar(), outfile=outfile, overwrite=overwrite, comment=comment, scale=scale)

            def export_poscar(self):
                "Export poscar data form cif content."
                content = _cif_str_to_poscar_str(
                    self._cif,
                    comment=f"[{self.mp_id!r}][{self.symbol!r}][{self.crystal!r}] Created by ipyvasp using Materials Project Database",
                )
                return export_poscar(content=content)

        # get cifs
        structures = []
        for res in self.__response:
            structures.append(Structure(res))

        self.success = True  # set success flag
        return structures


def _str2kpoints(kpts_str):
    try:
        with open(kpts_str, "r", encoding="utf-8") as f:
            kpts_str = f.read()
    except:
        pass

    hsk_list = []
    for j, line in enumerate(kpts_str.splitlines()):
        if line.strip():  # Make sure line is not empty
            data = line.split()
            if len(data) < 3:
                raise ValueError(f"Line {j + 1} has less than 3 values.")

            point = [float(i) for i in data[:3]]

            if len(data) == 4:
                _4th = (
                    data[3]
                    if re.search("\$\\\\[a-zA-Z]+\$|[a-zA-Z]+|[α-ωΑ-Ω]+|\|_", line)
                    else int(data[3])
                )
                point.append(_4th)
            elif len(data) == 5:
                _5th = int(data[4])
                point = point + [data[3], _5th]

            hsk_list.append(point)
    return hsk_list


def get_kpath(
    kpoints,
    n: int = 10,
    weight: float = None,
    ibzkpt: str = None,
    outfile: str = None,
    rec_basis=None,
):
    """Generate list of kpoints along high symmetry path. Options are write to file or return KPOINTS list.
    It generates uniformly spaced point with input `n` as just a scale factor of number of points per average length of `rec_basis`.

    Parameters
    ----------
    kpoints : list, tuple, np.ndarray, iterable, or str
        Any number of points as ``[(x,y,z,[label],[N]), ...]``, or any iterable
        (including ``zip`` objects or generators) that yields such tuples. Each
        point may also use a nested-array form ``(array_xyz, [label], [N])`` where
        ``array_xyz`` is any array-like of length ≥ 3 — it will be flattened
        automatically to ``(x, y, z)``. ``N`` is a numeric (int or float) density
        interpreted as **points per Angstrom** for the interval starting at that
        point whenever ``rec_basis`` is supplied (except when ``N=0`` to break
        the path). If `kpoints` is a multiline string, it is converted to a list
        of points; each line should be in format ``"x y z [label] [N]"``.
    n : int
        Number of points **per Angstrom** along each interval when ``rec_basis`` is
        provided. If (x,y,z,[label], N) is provided, this is ignored for that specific
        interval. If ``rec_basis`` is not provided, each interval has exactly ``n``
        points. Number of points in each interval is at least 2 even if ``n`` is less
        than 2 to keep end points anyway. You can use ``n = int(1/distance)`` based on known distance resolution.
        Both ``n`` and ``N`` do not guarantee exact distance between points beacause an interval can only be divided 
        into integer number of points and almost no interval length is an exact multiple of the distance between points.
    weight : float
        None by default to auto generates weights.
    ibzkpt : PathLike
        Path to ibzkpt file, required for HSE calculations.
    outfile : PathLike
        Path/to/file to write kpoints. Use sys.stdout to print to console.
    rec_basis : array_like
        Reciprocal basis 3x3 array to use for calculating uniform points.


    If `outfile = None`, kpoints array (Nx3) is returned.
    """
    if isinstance(kpoints, str):
        kpoints = _str2kpoints(kpoints)
    elif not isinstance(kpoints, (list, tuple, np.ndarray)):
        kpoints = list(kpoints)  # consume zip/generator

    if len(kpoints) < 2:
        raise ValueError("At least two points are required.")

    fixed_patches = []
    where_zero = []
    for idx, point in enumerate(kpoints):
        # flatten nested coords: (array, label, n) → [x, y, z, label, n]
        if isinstance(point, (list, tuple, np.ndarray)) and len(point) > 0:
            if isinstance(point[0], (list, tuple, np.ndarray)):
                point = [*np.asarray(point[0]).ravel()[:3], *point[1:]]

        if not isinstance(point, (list, tuple)):
            raise TypeError(
                f"kpoint must be a list or tuple as (x,y,z,[label],[N]),  got {point}"
            )

        cpt = point  # same for length 5, 4 with last entry as string

        if len(point) == 3:
            cpt = [*point, ""]  # make (x,y,z,label)
        elif len(point) == 4:
            if isinstance(point[3], (int, float, np.integer, np.floating)):
                cpt = [*point[:3], "", point[3]]
            elif not isinstance(point[3], str):
                raise TypeError(
                    f"4th entry in kpoint should be string label or numeric density, got {point}"
                )
        elif len(point) == 5:
            if not isinstance(point[3], str):
                raise TypeError(
                    f"4th entry in kpoint should be string label when 5 entries are given, got {point}"
                )
            if not isinstance(point[4], (int, float, np.integer, np.floating)):
                raise TypeError(
                    f"5th entry in kpoint should be numeric density, got {point}"
                )
        else:
            raise ValueError(f"Expects kpoint as (x,y,z,[label],[N]), got {point}")

        if isinstance(cpt[-1], (int, np.integer)) and cpt[-1] == 0:
            if idx - 1 in where_zero:
                raise ValueError(
                    f"Break at adjacent kpoints {idx}, {idx+1} is not allowed!"
                )
            if any([idx < 1, idx > (len(kpoints) - 3)]):
                raise ValueError("Bad break at edges!")
            where_zero.append(idx)

        fixed_patches.append(cpt)

    def add_points(p1, p2, npts, rec_basis):
        lab = p2[3]  # end point label
        segment_len = None
        if rec_basis is not None and np.size(rec_basis) == 9:
            basis = np.array(rec_basis)*2*np.pi # convert to 2pi factor for physical distance
            coords = to_R3(basis, [p1[:3], p2[:3]])
            segment_len = np.linalg.norm(coords[0] - coords[1])
        
        if len(p1) == 5:
            density = p1[4]
            if density > 0 and segment_len is not None:
                m = int(np.rint(density * segment_len))
            else:
                m = density
            lab = f"<={p1[3]}|{lab.replace('<=','')}" if density == 0 else lab  # merge labels in case user wants to break path
        elif segment_len is not None:
            m = int(np.rint(npts * segment_len))  # points per Angstrom based on rec_basis
        else:
            m = npts  # equal number of points in each interval, given by n.

        Np = max(m, 1)  # At least 2 points. one is given by end point of interval.
        X = np.linspace(p1[0], p2[0], Np, endpoint=False)
        Y = np.linspace(p1[1], p2[1], Np, endpoint=False)
        Z = np.linspace(p1[2], p2[2], Np, endpoint=False)

        kpts = [(x, y, z) for x, y, z in zip(X, Y, Z)]
        return (kpts, Np, lab)

    points, numbers, labels = [], [0], [fixed_patches[0][3]]
    for p1, p2 in zip(fixed_patches[:-1], fixed_patches[1:]):
        kp, m, lab = add_points(p1, p2, n, rec_basis)
        points.extend(kp)
        numbers.append(numbers[-1] + m)
        labels.append(lab)
        if lab.startswith("<="):
            labels[-2] = ""  # remove label for end of interval if broken, added here
    else:  # Add last point at end of for loop
        points.append(p2[:3])

    if weight is None and points:
        weight = 0 if ibzkpt else 1 / len(points)  # With IBZKPT, we need zero weight

    out_str = [
        "{0:>16.10f}{1:>16.10f}{2:>16.10f}{3:>12.6f}".format(x, y, z, weight)
        for x, y, z in points
    ]
    out_str = "\n".join(out_str)

    N = len(points)
    if (PI := Path(ibzkpt or "")).is_file():  # handles None automatically
        with PI.open("r") as f:
            lines = f.readlines()

        N = int(lines[1].strip()) + N  # Update N.
        slines = lines[3 : N + 4]
        ibz_str = "".join(slines)
        out_str = "{}\n{}".format(ibz_str.strip("\n"), out_str)

    path_info = ", ".join(
        f"{idx}:{lab}" for idx, lab in zip(numbers, labels) if lab != ""
    )

    top_str = "Automatically generated using ipyvasp for HSK-PATH {}\n\t{}\nReciprocal Lattice".format(
        path_info, N
    )
    out_str = "{}\n{}".format(top_str, out_str)
    if outfile != None:
        _write_text(outfile, out_str)
    else:
        return np.array(points, dtype=float)  # return points for any processing by user.


# Cell
def get_kmesh(
    poscar_data,
    *args,
    shift=0,
    weight=None,
    cartesian=False,
    ibzkpt=None,
    outfile=None,
    endpoint=True,
):
    """Generates uniform mesh of kpoints. Options are write to file, or return KPOINTS list.

    Parameters
    ----------
    poscar_data : ipyvasp.POSCAR.data
    *args : tuple
        1 or 3 integers which decide shape of mesh. If 1, mesh points equally spaced based on data from POSCAR.
    shift : float
        Only works if cartesian = False. Defualt is 0. Could be a number or list of three numbers to add to interval [0,1].
    weight : float
        If None, auto generates weights.
    cartesian : bool
        If True, generates cartesian mesh.
    ibzkpt : PathLike
        Path to ibzkpt file, required for HSE calculations.
    outfile : PathLike
        Path/to/file to write kpoints. Use sys.stdout to print to console.
    endpoint : bool
        Default True, include endpoints in mesh at edges away from origin.


    If `outfile = None`, kpoints array (Nx3) is returned.

    """
    if len(args) not in [1, 3]:
        raise ValueError("get_kmesh() takes 1 or 3 args!")

    if cartesian:
        norms = np.ptp(poscar_data.rec_basis, axis=0)
    else:
        norms = np.linalg.norm(poscar_data.rec_basis, axis=1)

    if len(args) == 1:
        if not isinstance(args[0], (int, np.integer)):
            raise ValueError("get_kmesh expects integer for first positional argument!")
        nx, ny, nz = [args[0] for _ in range(3)]

        weights = norms / np.max(norms)  # For making largest side at given n
        nx, ny, nz = np.rint(weights * args[0]).astype(int)

    elif len(args) == 3:
        for i, a in enumerate(args):
            if not isinstance(a, (int, np.integer)):
                raise ValueError("get_kmesh expects integer at position {}!".format(i))
        nx, ny, nz = list(args)

    low, high = np.array([[0, 0, 0], [1, 1, 1]]) + shift
    if cartesian:
        verts = get_bz(poscar_data.rec_basis, primitive=False).vertices
        low, high = np.min(verts, axis=0), np.max(verts, axis=0)
        low = (low * 2 * np.pi / poscar_data.metadata.scale).round(
            12
        )  # Cartesian KPOINTS are in unit of 2pi/SCALE
        high = (high * 2 * np.pi / poscar_data.metadata.scale).round(12)

    (lx, ly, lz), (hx, hy, hz) = low, high
    points = []
    for k in np.linspace(lz, hz, nz, endpoint=endpoint):
        for j in np.linspace(ly, hy, ny, endpoint=endpoint):
            for i in np.linspace(lx, hx, nx, endpoint=endpoint):
                points.append([i, j, k])

    points = np.array(points)
    points[np.abs(points) < 1e-10] = 0

    if len(points) == 0:
        raise ValueError("No KPOINTS in BZ from given input. Try larger input!")

    if weight == None and len(points) != 0:
        weight = float(1 / len(points))

    out_str = [
        "{0:>16.10f}{1:>16.10f}{2:>16.10f}{3:>12.6f}".format(x, y, z, weight)
        for x, y, z in points
    ]
    out_str = "\n".join(out_str)
    N = len(points)
    if ibzkpt and (PI := Path(ibzkpt)):
        with PI.open("r", encoding="utf-8") as f:
            lines = f.readlines()

        if (cartesian == False) and (lines[2].strip()[0] in "cCkK"):
            raise ValueError(
                "ibzkpt file is in cartesian coordinates, use get_kmesh(...,cartesian = True)!"
            )

        N = int(lines[1].strip()) + N  # Update N.
        slines = lines[3 : N + 4]
        ibz_str = "".join(slines)
        out_str = "{}\n{}".format(ibz_str, out_str)  # Update out_str
    mode = "Reciprocal" if cartesian == False else "Cartesian"
    top_str = "Generated uniform mesh using ipyvasp, GRID-SHAPE = [{},{},{}]\n\t{}\n{}".format(
        nx, ny, nz, N, mode
    )
    out_str = "{}\n{}".format(top_str, out_str)
    if outfile != None:
        _write_text(outfile, out_str)
    else:
        return np.array(points, dtype=float) # return points for any processing by user.


# Cell
def splot_bz(
    bz_data,
    plane=None,
    ax=None,
    color="blue",
    fill=False,
    fill_zorder=0,
    vectors=(0, 1, 2),
    colormap=None,
    shade=True,
    alpha=0.4,
    zoffset=0,
    center=(0,0,0),
    **kwargs,
):
    """Plots matplotlib's static figure of BZ/Cell. You can also plot in 2D on a 3D axes.

    Parameters
    ----------
    bz_data : Output of `get_bz`.
    plane : str
        Default is None and plots 3D surface. Can take 'xy','yz','zx' to plot in 2D.
    fill : bool
        True by defult, determines whether to fill surface of BZ or not.
    fill_zorder : int
        Default is 0, determines zorder of filled surface in 2D plots if `fill=True`.
    color : Any
        Color to fill surface and stroke color. Default is 'blue'. Can be any valid matplotlib color.
    vectors : tuple
        Tuple of indices of basis vectors to plot. Default is (0,1,2). All three are plotted in 3D
        (you can turn of by None or empty tuple), whhile you can specify any two/three in 2D.
        Vectors do not appear if given data is subzone data.
    ax : matplotlib.pyplot.Axes
        Auto generated by default, 2D/3D axes, auto converts in 3D on demand as well.
    colormap : str
        If None, single color is applied, only works in 3D and `fill=True`. Colormap is applied along z.
    shade : bool
        Shade polygons or not. Only works in 3D and `fill=True`.
    alpha : float
        Opacity of filling in range [0,1]. Increase for clear viewpoint.
    zoffset : float
        Only used if plotting in 2D over a 3D axis. Default is 0. Any plane 'xy','yz' etc can be offset to it's own normal.
    center : (3,) array_like
        Translation of origin in *basis coordinates* (fractional along the plotted basis). Use this to tile BZ with help of ``BrZoneData.tile`` fuction.

    kwargs are passed to `plt.plot` or `Poly3DCollection` if `fill=True`.

    Returns
    -------
    matplotlib.pyplot.Axes
        Matplotlib's 2D axes if `plane=None` otherswise 3D axes.
    """
    vname = "a" if bz_data.__class__.__name__ == "CellData" else "b"
    label = r"$k_{}/2π$" if vname == "b" else "{}"

    _label = r"\vec{" + vname + "}"  # For both

    if vectors and not isinstance(vectors, (tuple, list)):
        raise ValueError(f"`vectors` expects tuple or list, got {vectors!r}")

    if vectors is None:
        vectors = ()  # Empty tuple to make things work below

    for v in vectors:
        if v not in [0, 1, 2]:
            raise ValueError(f"`vectors` expects values in [0,1,2], got {vectors!r}")

    if not isinstance(center, (tuple, list, np.ndarray)) or len(center) != 3:
        raise ValueError("`center` must be a 3-sequence like (0,0,0) in basis coordinates.")
    try:
        center = np.array(center, dtype=float).reshape(3)
    except Exception as e:
        raise ValueError(f"`center` must be numeric, got {center!r}") from e
    
    origin = to_R3(bz_data.basis, [center])[0]  # (3,) cartesian shift
    bz_data = bz_data.copy()
    bz_data.vertices[:,:] += origin # apply on view, assignment is restricted
    
    name = kwargs.pop("label", None)  # will set only on single line
    kwargs.pop("zdir", None)  # remove , no need
    is_subzone = hasattr(bz_data, "_specials")  # For subzone

    if plane:  # Project 2D, works on 3D axes as well
        if not ax:  # Create 2D axes if not given
            ax = ptk.get_axes(figsize=(3.4, 3.4))  # For better display

        kwargs = {"solid_capstyle": "round", **kwargs}
        is3d = getattr(ax, "name", "") == "3d"
        normals = {
            "xy": (0, 0, 1),
            "yz": (1, 0, 0),
            "zx": (0, 1, 0),
            "yx": (0, 0, -1),
            "zy": (-1, 0, 0),
            "xz": (0, -1, 0),
        }
        if plane not in normals:
            raise ValueError(
                f"`plane` expects value in 'xyzxzyx' or None, got {plane!r}"
            )

        z0 = (
            [0, 0, zoffset]
            if plane in "xyx"
            else [0, zoffset, 0]
            if plane in "xzx"
            else [zoffset, 0, 0]
        )
        idxs = {
            "xy": [0, 1],
            "yz": [1, 2],
            "zx": [2, 0],
            "yx": [1, 0],
            "zy": [2, 1],
            "xz": [0, 2],
        }
        for idx, f in enumerate(bz_data.faces_coords):
            g = to_plane(normals[plane], f) + z0
            (line,) = ax.plot(
                *(g.T if is3d else g[:, idxs[plane]].T), color=color, **kwargs
            )
            if idx == 0:
                line.set_label(name)  # only one line

            if fill and not is3d:
                ax.fill(
                    *g[:, idxs[plane]].T,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0.0001,
                    alpha=alpha,
                    zorder=fill_zorder,
                )
            elif fill and is3d:
                poly = Poly3DCollection(
                    [g],  # 3D fill in plane
                    edgecolors=[color],
                    facecolors=[color],
                    alpha=alpha,
                    shade=shade,
                    zorder=fill_zorder,
                )
                ax.add_collection(poly)
                ax.autoscale_view()

        if vectors and not is_subzone:
            s_basis = to_plane(normals[plane], bz_data.basis[(vectors,)])
            s_origin = to_plane(normals[plane], [origin]*len(vectors))

            for k, b in zip(vectors, s_basis):
                x, y = b[idxs[plane]]
                l = r" ${}_{} $".format(_label, k + 1)
                l = l + "\n" if y < 0 else "\n" + l
                ha = "right" if x < 0 else "left"
                xyz = 0.8 * b + z0 + s_origin[0] if is3d else np.array([0.8 * x, 0.8 * y]) + s_origin[0, idxs[plane]]
                ax.text(
                    *xyz, l, va="center", ha=ha, clip_on=True
                )  # must clip to have limits of axes working.
                ax.scatter(
                    *(xyz / 0.8), color="w", s=0.0005
                )  # Must be to scale below arrow.
            if is3d:
                XYZ, UVW = (np.ones_like(s_basis) * z0 + s_origin).T, s_basis.T
                quiver3d(
                    *XYZ,
                    *UVW,
                    C=color,
                    L=0.7,
                    ax=ax,
                    arrowstyle="-|>",
                    mutation_scale=7,
                )
            else:
                ax.quiver(
                    *s_origin[:, idxs[plane]].T,
                    *s_basis[:, idxs[plane]].T,
                    lw=0.7,
                    color=color,
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                )

        ax.set_xlabel(label.format(plane[0]))
        ax.set_ylabel(label.format(plane[1]))
        if is3d:
            lab = [v for v in "xyz" if v not in plane][0]
            ax.set_zlabel(label.format(lab))
            ax.set_aspect("equal")
            zmin, zmax = ax.get_zlim()
            if zoffset > zmax:
                zmax = zoffset
            elif zoffset < zmin:
                zmin = zoffset
            ax.set_zlim([zmin, zmax])
        else:
            ax.set_aspect("equal")  # Must for 2D axes to show actual lengths of BZ

    else:  # Plot 3D
        if not ax:  # For 3D.
            ax = ptk.get_axes(figsize=(3.4, 3.4), axes_3d=True)

        if getattr(ax, "name", "") != "3d":
            raise ValueError("3D axes required for 3D plot.")

        if fill:
            if colormap:
                colormap = colormap if colormap in plt.colormaps() else "viridis"
                cz = [
                    np.mean(np.unique(f, axis=0), axis=0)[2]
                    for f in bz_data.faces_coords
                ]
                levels = (cz - np.min(cz)) / np.ptp(cz)  # along Z.
                colors = plt.cm.get_cmap(colormap)(levels)
            else:
                colors = np.array(
                    [[*mplc.to_rgb(color)] for f in bz_data.faces_coords]
                )  # Single color.

            poly = Poly3DCollection(
                bz_data.faces_coords,
                edgecolors=[color],
                facecolors=colors,
                alpha=alpha,
                shade=shade,
                label=name,
                **kwargs,
            )

            ax.add_collection(poly)
            ax.autoscale_view()
        else:
            kwargs = {"solid_capstyle": "round", **kwargs}
            (line,) = [
                ax.plot3D(f[:, 0], f[:, 1], f[:, 2], color=(color), **kwargs)
                for f in bz_data.faces_coords
            ][0]
            line.set_label(name)  # only one line

        if vectors and not is_subzone:
            for k, v in enumerate(0.35 * bz_data.basis):
                ax.text(*(v + origin), r"${}_{}$".format(_label, k + 1), va="center", ha="center")

            XYZ, UVW = np.array([origin] * 3).T, 0.3 * bz_data.basis.T
            quiver3d(
                *XYZ, *UVW, C="k", L=0.7, ax=ax, arrowstyle="-|>", mutation_scale=7
            )

        l_ = np.min(bz_data.vertices, axis=0)
        h_ = np.max(bz_data.vertices, axis=0)
        ax.set_xlim([l_[0], h_[0]])
        ax.set_ylim([l_[1], h_[1]])
        ax.set_zlim([l_[2], h_[2]])

        # Set aspect to same as data.
        ax.set_box_aspect(np.ptp(bz_data.vertices, axis=0))

        ax.set_xlabel(label.format("x"))
        ax.set_ylabel(label.format("y"))
        ax.set_zlabel(label.format("z"))

    if vname == "b":  # These needed for splot_kpath internally
        type(bz_data)._splot_kws = dict(plane=plane, zoffset=zoffset, ax=ax, shift=origin)

    return ax


def splot_kpath(
    bz_data, kpoints, labels=None, fmt_label=lambda x: (x, {"color": "blue"}), **kwargs
):
    """Plot k-path over last plotted BZ. It will take ``ax``, ``plane`` and ``zoffset`` internally from most recent call to ``splot_bz``/``bz.splot``.

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
    if not hasattr(bz_data, "_splot_kws"):
        raise ValueError("Plot BZ first to get ax, plane and zoffset.")

    if not np.ndim(kpoints) == 2 and np.shape(kpoints)[-1] == 3:
        raise ValueError("kpoints must be 2D array of shape (N,3)")

    plane, ax, zoffset, shift = [
        bz_data._splot_kws.get(attr, default)  # class level attributes
        for attr, default in zip(["plane", "ax", "zoffset", "shift"], [None, None, 0,np.array([0.0, 0.0, 0.0])])
    ]

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
    if plane:
        _zoffset = (
            [0, 0, zoffset]
            if plane in "xyx"
            else [0, zoffset, 0]
            if plane in "xzx"
            else [zoffset, 0, 0]
        )

    if isinstance(plane, str) and plane in _mapping:
        if getattr(ax, "name", None) != "3d":
            ijk = _mapping[
                plane
            ]  # only change indices if axes is not 3d, even if plane is given

    if not labels:
        labels = [
            "[{0:5.2f}, {1:5.2f}, {2:5.2f}]".format(x, y, z) for x, y, z in kpoints
        ]
    
    if fmt_label is None:
        fmt_label = lambda x: (x, {"color": "blue"})

    _validate_label_func(fmt_label,labels[0]) 

    coords = bz_data.to_cartesian(kpoints) + shift
    if _zoffset and plane:
        normal = (
            [0, 0, 1] if plane in "xyx" else [0, 1, 0] if plane in "xzx" else [1, 0, 0]
        )
        coords = to_plane(normal, coords) + _zoffset

    coords = coords[:, ijk]  # select only required indices
    kwargs = {
        **dict(color="blue", linewidth=0.8, marker=".", markersize=10),
        **kwargs,
    }  # need some defaults
    ax.plot(*coords.T, **kwargs)

    for c, text in zip(coords, labels):
        lab, textkws = fmt_label(text), {}
        if isinstance(lab, (list, tuple)):
            lab, textkws = lab
        ax.text(*c, lab, **textkws)

    return ax


# Cell
def iplot_bz(
    bz_data,
    fill=False,
    color="rgba(84,102,108,0.8)",
    special_kpoints=True,
    alpha=0.4,
    ortho3d=True,
    fig=None,
    **kwargs,
):
    """Plots interactive figure showing axes,BZ surface, special points and basis, each of which could be hidden or shown.

    Parameters
    ----------
    bz_data : Output of `get_bz`.
    fill : bool
        False by defult, determines whether to fill surface of BZ or not.
    color : str
        Color to fill surface 'rgba(84,102,108,0.8)` by default. This sholud be a valid Plotly color.
    special_kpoints : bool or callable
        True by default, determines whether to plot special points or not.
        You can also proivide a mask function f(x,y,z) -> bool which will be used to filter special points
        based on their fractional coordinates. This is ignored if BZ is primitive.
    alpha : float
        Opacity of BZ planes.
    ortho3d : bool
        Default is True, decides whether x,y,z are orthogonal or perspective.
    fig : plotly.graph_objects.Figure
        Plotly's `go.Figure`. If you want to plot on another plotly's figure, provide that.


    kwargs are passed to `plotly.graph_objects.Scatter3d` for BZ lines.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if not fig:
        fig = go.Figure()
    # Name fixing
    vname = "a" if bz_data.__class__.__name__ == "CellData" else "b"
    axes_text = [
        "<b>k</b><sub>x</sub>/2π",
        "",
        "<b>k</b><sub>y</sub>/2π",
        "",
        "<b>k</b><sub>z</sub>/2π",
    ]
    if vname == "a":
        axes_text = ["<b>x</b>", "", "<b>y</b>", "", "<b>z</b>"]  # Real space

    zone_name = kwargs.pop("name", "BZ" if vname == "b" else "Lattice")
    is_subzone = hasattr(bz_data, "_specials")  # For subzone

    if not is_subzone:  # No basis, axes for subzone
        # Axes
        _len = 0.5 * np.mean(bz_data.basis)
        fig.add_trace(
            go.Scatter3d(
                x=[_len, 0, 0, 0, 0],
                y=[0, 0, _len, 0, 0],
                z=[0, 0, 0, 0, _len],
                mode="lines+text",
                text=axes_text,
                line_color="skyblue",
                legendgroup="Axes",
                name="Axes",
            )
        )
        fig.add_trace(
            go.Cone(
                x=[_len, 0, 0],
                y=[0, _len, 0],
                z=[0, 0, _len],
                u=[1, 0, 0],
                v=[0, 1, 0],
                w=[0, 0, 1],
                showscale=False,
                sizemode="absolute",
                sizeref=0.5,
                anchor="tail",
                colorscale=["skyblue" for _ in range(3)],
                legendgroup="Axes",
                name="Axes",
            )
        )

        # Basis
        for i, b in enumerate(bz_data.basis):
            fig.add_trace(
                go.Scatter3d(
                    x=[0, b[0]],
                    y=[0, b[1]],
                    z=[0, b[2]],
                    mode="lines+text",
                    legendgroup="{}<sub>{}</sub>".format(vname, i + 1),
                    line_color="red",
                    name="<b>{}</b><sub>{}</sub>".format(vname, i + 1),
                    text=["", "<b>{}</b><sub>{}</sub>".format(vname, i + 1)],
                )
            )

            uvw = b / np.linalg.norm(b)  # Unit vector for cones
            fig.add_trace(
                go.Cone(
                    x=[b[0]],
                    y=[b[1]],
                    z=[b[2]],
                    u=uvw[0:1],
                    v=uvw[1:2],
                    w=uvw[2:],
                    showscale=False,
                    colorscale="Reds",
                    sizemode="absolute",
                    sizeref=0.02,
                    anchor="tail",
                    legendgroup="{}<sub>{}</sub>".format(vname, i + 1),
                    name="<b>{}</b><sub>{}</sub>".format(vname, i + 1),
                )
            )

    # Rest of the code is same for both subzone and BZ/Cell
    # Faces
    legend = True
    for pts in bz_data.faces_coords:
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="lines",
                line_color=color,
                legendgroup=zone_name,
                name=zone_name,
                showlegend=legend,
                surfaceaxis=2 if fill and coplanar(bz_data.vertices) else -1,
                surfacecolor=color,  # fills z-axis because its where 2D projection is
                opacity=alpha,
                **kwargs,
            )
        )

        legend = False  # Only first legend to show for all

    if fill and not coplanar(bz_data.vertices):
        # coplanar fill is not supported by Mesh3d
        xc = bz_data.vertices[ConvexHull(bz_data.vertices).vertices]
        fig.add_trace(
            go.Mesh3d(
                x=xc[:, 0],
                y=xc[:, 1],
                z=xc[:, 2],
                color=color,
                opacity=alpha,
                alphahull=0,# convex body
                lighting=dict(diffuse=0.5),
                legendgroup=zone_name,
                name=zone_name,
            )
        )

    # Special Points only if in reciprocal space and regular BZ
    if vname == "b" and (not getattr(bz_data, "primitive", False)) and special_kpoints:
        if callable(special_kpoints):
            skpts = bz_data.specials.masked(special_kpoints)
        else:
            skpts = bz_data.specials

        for tr in fig.data:  # hide all traces hover made before
            tr.hoverinfo = "none"  # avoid overlapping with special points

        texts, values = [], []
        norms = np.round(np.linalg.norm(skpts.coords, axis=1), 8)
        for key, value, norm in zip(skpts.kpoints.round(6), skpts.coords, norms):
            texts.append("K = {}</br>d = {}".format(key, norm))
            values.append([[*value, norm]])

        values = np.array(values).reshape((-1, 4))
        norm_max = np.max(values[:, 3])
        c_vals = np.array([int(v * 255 / norm_max) for v in values[:, 3]])
        colors = [0 for i in c_vals]
        _unique = np.unique(np.sort(c_vals))[::-1]
        _lnp = np.linspace(0, 255, len(_unique) - 1)
        _u_colors = ["rgb({},0,{})".format(r, b) for b, r in zip(_lnp, _lnp[::-1])]
        for _un, _uc in zip(_unique[:-1], _u_colors):
            _index = np.where(c_vals == _un)[0]
            for _ind in _index:
                colors[_ind] = _uc

        colors[0] = "rgb(255,215,0)"  # Gold color at Gamma!.
        fig.add_trace(
            go.Scatter3d(
                x=values[:, 0],
                y=values[:, 1],
                z=values[:, 2],
                hovertext=texts,
                name="HSK",
                marker=dict(color=colors, size=4),
                mode="markers",
            )
        )

    proj = dict(projection=dict(type="orthographic")) if ortho3d else {}
    camera = dict(center=dict(x=0.1, y=0.1, z=0.1), **proj)
    fig.update_layout(
        template="plotly_white",
        scene_camera=camera,
        font_family="Times New Roman",
        font_size=14,
        scene=dict(
            aspectmode="data",
            xaxis=dict(showbackground=False, visible=False),
            yaxis=dict(showbackground=False, visible=False),
            zaxis=dict(showbackground=False, visible=False),
        ),
        margin=dict(r=10, l=10, b=10, t=30),
    )
    return fig


# Cell
def _fix_sites(
    poscar_data, tol=1e-2, eqv_sites=False, translate=None, origin=(0, 0, 0)
):
    """Add equivalent sites to make a full data shape of lattice. Returns same data after fixing.
    It should not be exposed mostly be used in visualizations"""
    if not isinstance(origin, (tuple, list, np.ndarray)) or len(origin) != 3:
        raise ValueError("origin must be a list, tuple or numpy array of length 3.")

    pos = (
        poscar_data.positions.copy()
    )  # We can also do poscar_data.copy().positions that copies all contents.

    labels = np.array(poscar_data.labels)  # We need to store equivalent labels as well
    out_dict = poscar_data.to_dict()  # For output

    if isinstance(translate, (int, np.integer, float)):
        pos = pos + (translate - int(translate))  # Only translate in 0 - 1
    elif isinstance(translate,(tuple, list, np.ndarray)) and len(translate) == 3:
        txyz = np.array([translate])
        pos = pos + (txyz - txyz.astype(int))

    # Fix coordinates of sites distributed on edges and faces
    if getattr(poscar_data.metadata, 'eqv_fix', True): # no more fixing over there
        pos -= (pos > (1 - tol)).astype(int)  # Move towards orign for common fixing like in joining POSCARs
    
    out_dict["positions"] = pos
    out_dict["metadata"]["comment"] = "Modified by ipyvasp"

    # Add equivalent sites on edges and faces if given,handle each sepecies separately
    if eqv_sites and getattr(poscar_data.metadata, 'eqv_fix', True): 
        new_dict, start = {}, 0
        for k, v in out_dict["types"].items():
            vpos = pos[v]
            vlabs = labels[v]
            inds = np.array(v)

            ivpos = np.concatenate([np.indices((len(vpos),)).reshape((-1,1)),vpos],axis=1) # track of indices
            ivpos = np.array([ivpos + [0, *p] for p in product([-1,0,1],[-1,0,1],[-1,0,1])]).reshape((-1,4))
            ivpos = ivpos[(ivpos[:,1:] > -tol).all(axis=1) & (ivpos[:,1:] < 1 + tol).all(axis=1)]
            ivpos = ivpos[ivpos[:,0].argsort()]
            idxs = ivpos[:,0].ravel().astype(int).tolist()

            new_dict[k] = {"pos": ivpos[:,1:], "lab": vlabs[idxs], "inds": inds[idxs]}
            new_dict[k]["range"] = range(start, start + len(new_dict[k]["pos"]))
            start += len(new_dict[k]["pos"])

        out_dict["positions"] = np.vstack([new_dict[k]["pos"] for k in new_dict.keys()])
        out_dict["positions"] -= origin  # origin given by user to subtract

        out_dict["metadata"]["eqv_labels"] = np.hstack(
            [new_dict[k]["lab"] for k in new_dict.keys()]
        )

        out_dict["metadata"]["eqv_indices"] = np.hstack(
            [new_dict[k]["inds"] for k in new_dict.keys()]
        )
        out_dict["types"] = {k: new_dict[k]["range"] for k in new_dict.keys()}

    return serializer.PoscarData(out_dict)


def translate_poscar(poscar_data, offset):
    """Translate sites of a POSCAR with a given offset as a number or list of three number.
    Usully a farction of integarers like 1/2,1/4 etc."""
    return _fix_sites(poscar_data, translate=offset, eqv_sites=False)


def get_pairs(coords, r, tol=1e-3):
    """Returns a tuple of Points(coords,pairs, dist), so coords[pairs] given nearest site bonds.

    Parameters
    ----------
    coords : array_like
        Array(N,3) of cartesian positions of lattice sites.
    r : float
        Cartesian distance between the pairs in units of Angstrom e.g. 1.2 -> 1.2E-10.
    tol : float
        Tolerance value. Default is 10^-3.
    """
    if np.ndim(coords) != 2 and np.shape(coords)[1] != 3:
        raise ValueError("coords must be a 2D array of shape (N,3).")

    tree = KDTree(coords)
    inds = np.array([[*p] for p in tree.query_pairs(r, eps=tol)])
    if len(inds) > 0:
        dist = np.linalg.norm(coords[inds[:, 0],] - coords[inds[:, 1],], axis=1)
    else:
        dist = np.array([])
    return serializer.dict2tuple(
        "Points", {"coords": coords, "pairs": inds, "dist": dist}
    )


def _get_bond_length(poscar_data, bond_length=None):
    "Given `bond_length` should be in unit of Angstrom, and can be a number of dict like {'Fe-O':1.2,...}"
    if bond_length is not None:
        if isinstance(bond_length, (int, float, np.integer)):
            return bond_length
        elif isinstance(bond_length, dict):
            for k, v in bond_length.items():
                if not isinstance(v, (int, float, np.integer)):
                    raise TypeError(
                        f"Value to key `{k}` should be a number in unit of Angstrom."
                    )
                if not isinstance(k, str) or k.count("-") != 1:
                    raise TypeError(
                        f"key `{k}` should be a string connecting two elements like 'Fe-O'."
                    )

            return max(
                list(bond_length.values())
            )  # return the maximum distance, will filter later
        else:
            raise TypeError("`bon_length` should be a number or a dict.")
    else:
        keys = list(poscar_data.types.keys())
        if len(keys) == 1:
            keys = [*keys, *keys]  # still need it to be a list of two elements

        dists = [poscar_data.get_distance(k1, k2) for k1, k2 in combinations(keys, 2)]
        return (
            np.mean(dists) * 1.05
        )  # Add 5% margin over mean distance, this covers same species too, and in multiple species, this will stop bonding between same species.


class _Atom(NamedTuple):
    "Object passed to POSCAR operations `func` where atomic sites are modified. Additinal property p -> array([x,y,z])."
    symbol : str
    number : int
    index : int
    x : float
    y : float
    z : float

    @property
    def p(self): return np.array([self.x,self.y,self.z]) # for robust operations

class _AtomLabel(str):
    "Object passed to `fmt_label` in plotting. `number` and `symbol` are additional attributes and `to_latex` is a method."
    @property
    def number(self): return int(self.split()[1])
    @property
    def symbol(self): return self.split()[0]
    def to_latex(self): return "{}$_{{{}}}$".format(*self.split())

def _validate_func(func):
    if not callable(func):
        raise ValueError("`func` must be a callable function with single parameter `Atom(symbol,number, index,x,y,z)`.")
    
    if len(inspect.signature(func).parameters) != 1:
        raise ValueError(
            "`func` takes exactly 1 argument: `Atom(symbol, number, index,x,y,z)` in fractional coordinates"
        )
    
    ret = func(_Atom('',0,0,0,0,0))
    if not isinstance(ret, (bool, np.bool_)):
        raise ValueError(
            f"`func` must be a function that returns a bool, got {type(ret)}."
        )

def _masked_data(poscar_data, func):
    "Returns indices of sites which satisfy the func."
    _validate_func(func)
    eqv_inds  = tuple(getattr(poscar_data.metadata, "eqv_indices",[]))

    pick = []
    for i, pos in enumerate(poscar_data.positions):
        idx = eqv_inds[i] if eqv_inds else i  # map to original index
        if func(_Atom(*poscar_data._sn[i], idx, *pos)): # labels based on i, not eqv_idx
            pick.append(i)
    return pick  # could be duplicate indices


def _filter_pairs(labels, pairs, dist, bond_length):
    """Filter pairs based on bond_length dict like {1.2:['Fe','O'],...}. Returns same pairs otherwise."""
    if isinstance(bond_length, dict):
        new_pairs = []
        for pair, d in zip(pairs, dist):
            t1, t2 = [labels[idx].split()[0] for idx in pair]
            for k, v in bond_length.items():
                p = tuple(k.split("-"))
                if p in [(t1, t2), (t2, t1)] and d <= v:
                    new_pairs.append(pair)

        return np.unique(new_pairs, axis=0)  # remove duplicates

    # Return all pairs otherwise
    return pairs  # None -> auto calculate bond_length, number -> use that number

def filter_atoms(poscar_data, func, tol = 0.01):
    """Filter atomic sites based on a function that acts on an atom such as `lambda a: (a.p < 1/2).all()`. 
    `atom` passed to function is a namedtuple like `Atom(symbol,number,index,x,y,z)` which has extra attribute `p = array([x,y,z])`.
    This may include equivalent sites, so it should be used for plotting purpose only, e.g. showing atoms on a plane.
    An attribute `source_indices` is added to metadata which is useful to pick other things such as `OUTCAR.ion_pot[POSCAR.filter(...).data.metadata.source_indices]`. 

    >>> filter_atoms(..., lambda a: a.symbol=='Ga' or a.number in range(2)) # picks all Ga atoms and first two atoms of every other types.

    Note: If you are filtering a plane with more than one non-zero hkl like 110, you may first need to translate or set boundary on POSCAR to bring desired plane in full view to include all atoms.
    """
    if hasattr(poscar_data.metadata, 'source_indices'):
        raise ValueError("Cannot filter an already filtered POSCAR data.")
    
    poscar_data = _fix_sites(poscar_data, tol = tol, eqv_sites=True)
    idxs = _masked_data(poscar_data, func)
    data = poscar_data.to_dict() 
    eqvi = data['metadata'].pop('eqv_indices', []) # no need of this
    
    all_pos, npos, eqv_labs, finds = [], [0,],[],[]
    for value in poscar_data.types.values():
        indices = [i for i in value if i in idxs] # search from value make sure only non-equivalent sites added
        finds.extend(eqvi[indices] if len(eqvi) else indices)
        eqv_labs.extend(poscar_data.labels[indices])
        pos = data['positions'][indices]
        all_pos.append(pos)
        npos.append(len(pos))
    
    if not np.sum(npos):
        raise ValueError("No sites found with given filter func!")
    
    data['positions'] = np.concatenate(all_pos, axis = 0)
    data['metadata']['source_indices'] = np.array(finds)
    data['metadata']['eqv_fix'] = False
    data['metadata']['eqv_labels']  = np.array(eqv_labs) # need these for compare to previous

    ranges = np.cumsum(npos)
    data['types'] = {key: range(i,j) for key, i,j in zip(data['types'],ranges[:-1],ranges[1:]) if range(i,j)} # avoid empty
    return serializer.PoscarData(data)

# Cell
def iplot_lattice(
    poscar_data,
    sizes=10,
    colors=None,
    bond_length=None,
    tol=1e-2,
    eqv_sites=True,
    translate=None,
    origin=(0, 0, 0),
    fig=None,
    ortho3d=True,
    bond_kws=dict(line_width=4),
    site_kws=dict(line_color="rgba(1,1,1,0)", line_width=0.001, opacity=1),
    plot_cell=True,
    label_sites = False,
    **kwargs,
):
    """Plotly's interactive plot of lattice.

    Parameters
    ----------
    sizes : float or dict of type -> float
        Size of sites. Either one int/float or a mapping like {'Ga': 2, ...}.
    colors : color or dict of type -> color
        Mapping of colors like {'Ga': 'red, ...} or a single color. Automatically generated color for missing types.
    bond_length : float or dict
        Length of bond in Angstrom. Auto calculated if not provides. Can be a dict like {'Fe-O':3.2,...} to specify bond length between specific types.
    bond_kws : dict
        Keyword arguments passed to `plotly.graph_objects.Scatter3d` for bonds.
        Default is jus hint, you can use any keyword argument that is accepted by `plotly.graph_objects.Scatter3d`.
    site_kws : dict
        Keyword arguments passed to `plotly.graph_objects.Scatter3d` for sites.
        Default is jus hint, you can use any keyword argument that is accepted by `plotly.graph_objects.Scatter3d`.
    plot_cell : bool
        Defult is True. Plot unit cell with default settings.
        If you want to customize, use `POSCAR.iplot_cell(fig = <return of iplot_lattice>)` function.


    kwargs are passed to `iplot_bz`.
    """
    if len(poscar_data.positions) < 1:
        raise ValueError("Need at least 1 atom!")
    
    poscar_data = _fix_sites(
        poscar_data, tol=tol, eqv_sites=eqv_sites, translate=translate, origin=origin
    )
    
    blen = _get_bond_length(poscar_data, bond_length)

    coords, pairs, dist = get_pairs(poscar_data.coords, r=blen)
    _labels = poscar_data.labels
    pairs = _filter_pairs(_labels, pairs, dist, bond_length)

    if not fig:
        fig = go.Figure()

    uelems = poscar_data.types.to_dict()
    _fcs = _fix_color_size(uelems, colors, sizes, 10, backend = 'plotly')
    sizes  = [v['size'] for v in _fcs.values()]
    colors = [v['color'] for v in _fcs.values()]

    _colors = np.array([colors[i] for i, vs in enumerate(uelems.values()) for v in vs],dtype=object) # could be mixed color types

    if np.any(pairs):
        coords_p = coords[pairs]  # paired points
        _colors = _colors[pairs]  # Colors at pairs
        coords_n = []
        colors_n = []
        for c_p, _c in zip(coords_p, _colors):
            mid = np.mean(c_p, axis=0)
            arr = np.concatenate([c_p[0], mid, mid, c_p[1]]).reshape((-1, 2, 3))
            coords_n = [*coords_n, *arr]  # Same shape
            colors_n = [*colors_n, *_c]  # same shape.

        coords_n = np.array(coords_n)
        colors_n = np.array(colors_n, dtype=object)

        # Instead of plotting for each pair, we can make only as little lines as types of atoms to speec up
        unqc = [] # mixed colors type can't be sorted otherwise
        for c in colors_n:
            if c not in unqc:
                unqc.append(c)

        clabs = [unqc.index(c) for c in colors_n] # few colors categories
        corder = np.argsort(clabs) # coordinates order for those categories

        groups = dict([(i,[]) for i in range(len(unqc))])
        for co in corder:
            groups[clabs[co]].append(coords_n[co])
            groups[clabs[co]].append([[np.nan, np.nan, np.nan]]) # nan to break links outside bonds

        for i in range(len(unqc)):
            groups[i] = np.concatenate(groups[i], axis=0)

        bond_kws = {"line_width": 4, **bond_kws}

        for i, cp in groups.items():
            showlegend = True if i == 0 else False
            fig.add_trace(go.Scatter3d(x=cp[:, 0].T, y=cp[:, 1].T, z=cp[:, 2].T,
                mode="lines",
                line_color=unqc[i],
                legendgroup="Bonds",
                showlegend=showlegend,
                hoverinfo='skip',
                name="Bonds",
                **bond_kws,
            ))

    site_kws = {
        **dict(line_color="rgba(1,1,1,0)", line_width=0.001, opacity=1),
        **site_kws,
    }

    eqv_idxs = getattr(poscar_data.metadata, 'eqv_indices',np.array(range(poscar_data.positions.shape[0])))

    for (k, v), c, s in zip(uelems.items(), colors, sizes):
        coords = poscar_data.coords[v]
        labs = poscar_data.labels[v]
        idxs = eqv_idxs[v]
        hovertext = [f"<br>{x:7.3f} {y:7.3f} {z:7.3f}<br>Index: {idx}  Label: {lab}" for lab,idx, (x,y,z) in zip(labs,idxs, poscar_data.positions[idxs])]

        fig.add_trace(
            go.Scatter3d(
                x=coords[:, 0].T,
                y=coords[:, 1].T,
                z=coords[:, 2].T,
                mode="markers+text" if label_sites else "markers",
                marker_color=c,
                hovertext=hovertext,
                text=["{}<sub>{}</sub>".format(*l.split()) for l in labs] if label_sites else None,
                marker_size=s,
                name=k,
                **site_kws,
            )
        )

    if plot_cell:
        bz_data = serializer.CellData(
            get_bz(poscar_data.basis, primitive=True).to_dict()
        )  # Make cell for correct vector notations
        iplot_bz(bz_data, fig=fig, ortho3d=ortho3d, special_kpoints=False, **kwargs)
    else:
        if kwargs:
            print("Warning: kwargs are ignored as plot_cell is False.")
        # These thing are update in iplot_bz function, but if plot_cell is False, then we need to update them here.
        proj = dict(projection=dict(type="orthographic")) if ortho3d else {}
        camera = dict(center=dict(x=0.1, y=0.1, z=0.1), **proj)
        fig.update_layout(
            template="plotly_white",
            scene_camera=camera,
            font_family="Times New Roman",
            font_size=14,
            scene=dict(
                aspectmode="data",
                xaxis=dict(showbackground=False, visible=False),
                yaxis=dict(showbackground=False, visible=False),
                zaxis=dict(showbackground=False, visible=False),
            ),
            margin=dict(r=10, l=10, b=10, t=30),
        )
    return fig


def _validate_label_func(fmt_label,label):
    if not callable(fmt_label):
        raise ValueError("fmt_label must be a callable function.")
    if len(inspect.signature(fmt_label).parameters.values()) != 1:
        raise ValueError("fmt_label must have only one argument that accepts a str like 'Ga 1'.")

    test_out = fmt_label(_AtomLabel(label))
    if isinstance(test_out, (list, tuple)):
        if len(test_out) != 2:
            raise ValueError(
                "fmt_label must return string or a list/tuple of length 2."
            )

        if not isinstance(test_out[0], str):
            raise ValueError(
                "Fisrt item in return of `fmt_label` must return a string! got {}".format(
                    type(test_out[0])
                )
            )

        if not isinstance(test_out[1], dict):
            raise ValueError(
                "Second item in return of `fmt_label` must return a dictionary of keywords to pass to `plt.text`! got {}".format(
                    type(test_out[1])
                )
            )

    elif not isinstance(test_out, str):
        raise ValueError("fmt_label must return a string or a list/tuple of length 2.")

def _fix_color_size(types, colors, sizes, default_size, backend=None):
    cs = {key: {'color': _atom_colors.get(key, 'blue'), 'size': default_size} for key in types}
    for k in cs:
        if len(cs[k]['color']) == 3: # otherwise its blue
            if backend == 'plotly':
                cs[k]['color'] = "rgb({},{},{})".format(*[int(255*c) for c in cs[k]['color']])
            elif backend == 'ngl':
                cs[k]['color'] = mplc.to_hex(cs[k]['color'])
    
    if isinstance(sizes,(int,float,np.integer)):
        for k in cs:
            cs[k]['size'] = sizes 
    elif isinstance(sizes, dict):
        for k,v in sizes.items():
            cs[k]['size'] = v 
    else:
        raise TypeError("sizes should be a single int/float or dict as {'Ga':10,'As':15,...}")
    
    if isinstance(colors,dict):
        for k,v in colors.items():
            cs[k]['color'] = v 
    elif isinstance(colors,(str,list,tuple,np.ndarray)):
        for k in cs:
            cs[k]['color'] = colors
    elif colors is not None:
        raise TypeError("colors should be a single valid color or dict as {'Ga':'red','As':'blue',...}")
    return cs

# Cell
def splot_lattice(
    poscar_data,
    plane=None,
    sizes=50,
    colors=None,
    bond_length=None,
    tol=1e-2,
    eqv_sites=True,
    translate=None,
    origin=(0, 0, 0),
    ax=None,
    showlegend=True,
    fmt_label=None,
    site_kws=dict(alpha=0.7),
    bond_kws=dict(alpha=0.7, lw=1),
    plot_cell=True,
    **kwargs,
):
    """Matplotlib Static plot of lattice.

    Parameters
    ----------
    plane : str
        Plane to plot. Either 'xy','xz','yz' or None for 3D plot.
    sizes : float or dict of type -> float
        Size of sites. Either one int/float or a mapping like {'Ga': 2, ...}.
    colors : color or dict of type -> color
        Mapping of colors like {'Ga': 'red, ...} or a single color. Automatically generated color for missing types.
    bond_length : float or dict
        Length of bond in Angstrom. Auto calculated if not provides. Can be a dict like {'Fe-O':3.2,...} to specify bond length between specific types.
    alpha : float
        Opacity of points and bonds.
    showlegend : bool
        Default is True, show legend for each ion type.
    site_kws : dict
        Keyword arguments to pass to `plt.scatter` for plotting sites.
        Default is just hint, you can pass any keyword argument that `plt.scatter` accepts.
    bond_kws : dict
        Keyword arguments to pass to `LineCollection`/`Line3DCollection` for plotting bonds.
    fmt_label : callable
        If given, each site label is passed to it as a subclass of str 'Ga 1' with extra attributes `symbol` and `number` and a method `to_latex`.
        You can show specific labels based on condition, e.g. `lambda lab: lab.to_latex() if lab.number in [1,5] else ''` will show 1st and 5th atom of each types.
        It must return a string or a list/tuple of length 2 with first item as label and second item as dictionary of keywords to pass to `plt.text`.
    plot_cell : bool
        Default is True, plot unit cell with default settings.
        To customize options, use `plot_cell = False` and do `POSCAR.splot_cell(ax = <return of splot_lattice>)`.


    kwargs are passed to `splot_bz`.

    .. tip::
        Use `plt.style.use('ggplot')` for better 3D perception.
    """
    if len(poscar_data.positions) < 1:
        raise ValueError("Need at least 1 atom!")
    
    # Plane fix
    if plane and plane not in "xyzxzyx":
        raise ValueError("plane expects in 'xyzxzyx' or None.")
    if plane:
        ind = "xyzxzyx".index(plane)
        arr = [0, 1, 2, 0, 2, 1, 0]
        ix, iy = arr[ind], arr[ind + 1]

    poscar_data = _fix_sites(
        poscar_data, tol=tol, eqv_sites=eqv_sites, translate=translate, origin=origin
    )
    blen = _get_bond_length(poscar_data, bond_length)
    labels = poscar_data.labels
    coords, pairs, dist = get_pairs(poscar_data.coords, r=blen)
    pairs = _filter_pairs(labels, pairs, dist, bond_length)

    if fmt_label is not None:
        _validate_label_func(fmt_label,labels[0])

    if plot_cell:
        bz_data = serializer.CellData(
            get_bz(poscar_data.basis, primitive=True).to_dict()
        )  # For correct vectors
        ax = splot_bz(bz_data, plane=plane, ax=ax, **kwargs)
    else:
        ax = ax or ptk.get_axes(axes_3d=True if plane is None else False)
        if kwargs:
            print(f"Warning: Parameters {list(kwargs.keys())} are not used when `plot_cell = False`.")

    uelems = poscar_data.types.to_dict()

    _fcs = _fix_color_size(uelems, colors, sizes, 50)
    sizes  = [v['size'] for v in _fcs.values()]
    colors = [v['color'] for v in _fcs.values()]

    # Before doing other stuff, create something for legend.
    if showlegend:    
        for key, c, s in zip(uelems.keys(), colors, sizes):
            ax.scatter([], [], s=s, color=c, label=key, **site_kws)  # Works both for 3D and 2D.
        ptk.add_legend(ax)

    # Now change colors and sizes to whole array size
    colors = np.array(
        [mplc.to_rgb(colors[i]) for i, vs in enumerate(uelems.values()) for v in vs]
    )
    sizes = np.array([sizes[i] for i, vs in enumerate(uelems.values()) for v in vs])

    if np.any(pairs):
        coords_p = coords[pairs]  # paired points
        _colors = colors[pairs]  # Colors at pairs
        coords_n = []
        colors_n = []
        for c_p, _c in zip(coords_p, _colors):
            mid = np.mean(c_p, axis=0)
            arr = np.concatenate([c_p[0], mid, mid, c_p[1]]).reshape((-1, 2, 3))
            coords_n = [*coords_n, *arr]  # Same shape
            colors_n = [*colors_n, *_c]  # same shape.

        coords_n = np.array(coords_n)
        colors_n = np.array(colors_n)

        bond_kws = {
            "alpha": 0.7,
            "capstyle": "butt",
            **bond_kws,
        }  # bond_kws overrides alpha and capstyle only
        # 3D LineCollection by default, very fast as compared to plot one by one.
        lc = Line3DCollection(coords_n, colors=colors_n, **bond_kws)
        if plane and plane in "xyzxzyx":  # Avoid None
            lc = LineCollection(coords_n[:, :, [ix, iy]], colors=colors_n, **bond_kws)

        ax.add_collection(lc)
        ax.autoscale_view()

    if not plane:
        site_kws = {
            **dict(alpha=0.7, depthshade=False),
            **site_kws,
        }  # site_kws overrides alpha only
        ax.scatter(
            coords[:, 0], coords[:, 1], coords[:, 2], c=colors, s=sizes, **site_kws
        )
        if fmt_label:
            for i, coord in enumerate(coords):
                lab, textkws = fmt_label(_AtomLabel(labels[i])), {}
                if isinstance(lab, (list, tuple)):
                    lab, textkws = lab
                ax.text(*coord, lab, **textkws)
        # Set aspect to same as data if cell plotted
        if plot_cell:
            ax.set_box_aspect(np.ptp(bz_data.vertices, axis=0))

    elif plane in "xyzxzyx":
        site_kws = {**dict(alpha=0.7, zorder=3), **site_kws}
        (iz,) = [i for i in range(3) if i not in (ix, iy)]
        zorder = coords[:, iz].argsort()
        if plane in "yxzy":  # Left handed
            zorder = zorder[::-1]
        ax.scatter(
            coords[zorder][:, ix],
            coords[zorder][:, iy],
            c=colors[zorder],
            s=sizes[zorder],
            **site_kws,
        )

        if fmt_label:
            labels = [labels[i] for i in zorder]  # Reorder labels
            for i, coord in enumerate(coords[zorder]):
                lab, textkws = fmt_label(_AtomLabel(labels[i])), {}
                if isinstance(lab, (list, tuple)):
                    lab, textkws = lab
                ax.text(*coord[[ix, iy]], lab, **textkws)

        # Set aspect to display real shape.
        ax.set_aspect("equal")

    ax.set_axis_off()
    return ax


# Cell
def join_poscars(poscar_data, other, direction="c", tol=1e-2, system=None):
    """Joins two POSCARs in a given direction. In-plane lattice parameters are kept from
    first poscar and out of plane basis vector of other is modified while volume is kept same.

    Parameters
    ----------
    other : type(self)
        Other POSCAR to be joined with this POSCAR.
    direction : str
        The joining direction. It is general and can join in any direction along basis. Expect one of ['a','b','c'].
    tol : float
        Default is 0.01. It is used to bring sites near 1 to near zero in order to complete sites in plane.
        Vasp relaxation could move a point, say at 0.00100 to 0.99800 which is not useful while merging sites.
    system : str
        If system is given, it is written on top of file. Otherwise, it is infered from atomic species.
    """
    _poscar1 = _fix_sites(poscar_data, tol=tol, eqv_sites=False)
    _poscar2 = _fix_sites(other, tol=tol, eqv_sites=False)
    pos1 = _poscar1.positions.copy()
    pos2 = _poscar2.positions.copy()

    s1, s2 = 0.5, 0.5  # Half length for each.
    a1, b1, c1 = np.linalg.norm(_poscar1.basis, axis=1)
    a2, b2, c2 = np.linalg.norm(_poscar2.basis, axis=1)
    basis = _poscar1.basis.copy()  # Must be copied, otherwise change outside.

    # Processing in orthogonal space since a.(b x c) = abc sin(theta)cos(phi), and theta and phi are same for both.
    if direction in "cC":
        c2 = (
            (a2 * b2) / (a1 * b1) * c2
        )  # Conservation of volume for right side to stretch in c-direction.
        netc = c1 + c2
        s1, s2 = c1 / netc, c2 / netc
        pos1[:, 2] = s1 * pos1[:, 2]
        pos2[:, 2] = s2 * pos2[:, 2] + s1
        basis[2] = netc * basis[2] / np.linalg.norm(basis[2])  # Update 3rd vector

    elif direction in "bB":
        b2 = (
            (a2 * c2) / (a1 * c1) * b2
        )  # Conservation of volume for right side to stretch in b-direction.
        netb = b1 + b2
        s1, s2 = b1 / netb, b2 / netb
        pos1[:, 1] = s1 * pos1[:, 1]
        pos2[:, 1] = s2 * pos2[:, 1] + s1
        basis[1] = netb * basis[1] / np.linalg.norm(basis[1])  # Update 2nd vector

    elif direction in "aA":
        a2 = (
            (b2 * c2) / (b1 * c1) * a2
        )  # Conservation of volume for right side to stretch in a-direction.
        neta = a1 + a2
        s1, s2 = a1 / neta, a2 / neta
        pos1[:, 0] = s1 * pos1[:, 0]
        pos2[:, 0] = s2 * pos2[:, 0] + s1
        basis[0] = neta * basis[0] / np.linalg.norm(basis[0])  # Update 1st vector

    else:
        raise Exception("direction expects one of ['a','b','c']")

    scale = np.linalg.norm(basis[0])
    u1 = _poscar1.types.to_dict()
    u2 = _poscar2.types.to_dict()
    u_all = ({**u1, **u2}).keys()  # Union of unique atom types to keep track of order.

    pos_all = []
    i_all = []
    for u in u_all:
        _i_ = 0
        if u in u1.keys():
            _i_ = len(u1[u])
            pos_all = [*pos_all, *pos1[u1[u]]]
        if u in u2.keys():
            _i_ = _i_ + len(u2[u])
            pos_all = [*pos_all, *pos2[u2[u]]]
        i_all.append(_i_)

    i_all = np.cumsum([0, *i_all])  # Do it after labels
    uelems = {_u: range(i_all[i], i_all[i + 1]) for i, _u in enumerate(u_all)}
    sys = system or "".join(uelems.keys())
    iscartesian = poscar_data.metadata.cartesian or other.metadata.cartesian
    metadata = {
        "cartesian": iscartesian,
        "scale": scale,
        "comment": "Modified by ipyvasp",
    }
    out_dict = {
        "SYSTEM": sys,
        "basis": basis,
        "metadata": metadata,
        "positions": np.array(pos_all),
        "types": uelems,
    }
    return serializer.PoscarData(out_dict)


# Cell
def repeat_poscar(poscar_data, n, direction):
    """Repeat a given POSCAR.

    Parameters
    ----------
    n : int
        Number of repetitions.
    direction : str
        Direction of repetition. Can be 'a', 'b' or 'c'.
    """
    if not isinstance(n, (int, np.integer)) and n < 2:
        raise ValueError("n must be an integer greater than 1.")
    given_poscar = poscar_data
    for i in range(1, n):
        poscar_data = join_poscars(given_poscar, poscar_data, direction=direction)
    return poscar_data


def scale_poscar(poscar_data, scale=(1, 1, 1), tol=1e-2):
    """Create larger/smaller cell from a given POSCAR. Can be used to repeat a POSCAR with integer scale values.

    Parameters
    ----------
    scale : tuple
        Tuple of three values along (a,b,c) vectors. int or float values. If number of sites are not as expected in output,
        tweak `tol` instead of `scale`. You can put a minus sign with `tol` to get more sites and plus sign to reduce sites.
    tol : float
        It is used such that site positions are blow `1 - tol`, as 1 belongs to next cell, not previous one.

    .. note::
        ``scale = (2,2,2)`` enlarges a cell and next operation of ``(1/2,1/2,1/2)`` should bring original cell back.

    .. warning::
        A POSCAR scaled with Non-integer values should only be used for visualization purposes, Not for any other opration such as making supercells, joining POSCARs etc.
    """
    if not isinstance(scale, (tuple, list)) or len(scale) != 3:
        raise ValueError("scale must be a tuple of three values.")

    ii, jj, kk = np.ceil(scale).astype(int)  # Need int for joining.

    if tuple(scale) == (1, 1, 1):  # No need to scale.
        return poscar_data

    if ii >= 2:
        poscar_data = repeat_poscar(poscar_data, ii, direction="a")

    if jj >= 2:
        poscar_data = repeat_poscar(poscar_data, jj, direction="b")

    if kk >= 2:
        poscar_data = repeat_poscar(poscar_data, kk, direction="c")

    if np.all([s == int(s) for s in scale]):
        return poscar_data  # No need to prcess further in case of integer scaling.

    new_poscar = poscar_data.to_dict()  # Update in it

    # Get clip fraction
    fi, fj, fk = scale[0] / ii, scale[1] / jj, scale[2] / kk

    # Clip at end according to scale, change length of basis as fractions.
    pos = poscar_data.positions.copy() / np.array([fi, fj, fk])  # rescale for clip
    basis = poscar_data.basis.copy()

    for i, f in zip(range(3), [fi, fj, fk]):
        basis[i] = f * basis[i]  # Basis rescale for clip

    new_poscar["basis"] = basis
    new_poscar["metadata"]["scale"] = np.linalg.norm(basis[0])
    new_poscar["metadata"]["comment"] = f"Modified by ipyvasp"

    uelems = poscar_data.types.to_dict()
    # Minus in below for block is because if we have 0-2 then 1 belongs to next cell not original.
    positions, shift = [], 0
    for key, value in uelems.items():
        s_p = pos[value]  # Get positions of key
        s_p = s_p[(s_p < 1 - tol).all(axis=1)]  # Get sites within tolerance

        if len(s_p) == 0:
            raise Exception(
                f"No sites found for {key!r}, cannot scale down. Increase scale!"
            )

        uelems[key] = range(shift, shift + len(s_p))
        positions = [*positions, *s_p]  # Pick sites
        shift += len(s_p)  # Update for next element

    new_poscar["types"] = uelems
    new_poscar["positions"] = np.array(positions)
    return serializer.PoscarData(new_poscar)

def set_boundary(poscar_data, a = [0,1], b = [0,1], c = [0,1]):
    "View atoms in a given boundary along a,b,c directions."
    for d, name in zip([a,b,c],'abc'):
        if not isinstance(d,(list,tuple)) or len(d) != 2:
            raise ValueError(f"{name} should be a list/tuple of type [min, max]")
        if d[1] < d[0]:
            raise ValueError(f"{name} should be in increasing order as [min, max]")
        
    data = poscar_data.to_dict()
    upos = {}
    for key, value in poscar_data.types.items():
        pos = data['positions'][value]
        for i, (l,h), shift in zip(range(3), [a,b,c],np.eye(3)):
            pos = np.concatenate([pos + shift*k for k in np.arange(np.floor(l), np.ceil(h))],axis=0)
            pos = pos[(pos[:,i] >= l) & (pos[:,i] <= h)]
        
        upos[key] = pos
    
    data['positions'] = np.concatenate(list(upos.values()), axis = 0)
    data['metadata']['eqv_fix'] = False

    ranges = np.cumsum([0, *[len(v) for v in upos.values()]])
    data['types'] = {key: range(i,j) for key, i,j in zip(upos,ranges[:-1],ranges[1:])}
    del upos
    return serializer.PoscarData(data)


def rotate_poscar(poscar_data, angle_deg, axis_vec):
    """Rotate a given POSCAR.

    Parameters
    ----------
    angle_deg : float
        Rotation angle in degrees.
    axis_vec : array_like
        Vector (x,y,z) of axis about which rotation takes place. Axis passes through origin.
    """
    rot = rotation(angle_deg=angle_deg, axis_vec=axis_vec)
    p_dict = poscar_data.to_dict()
    p_dict["basis"] = rot.apply(
        p_dict["basis"]
    )  # Rotate basis so that they are transpose
    p_dict["metadata"]["comment"] = f"Modified by ipyvasp"
    return serializer.PoscarData(p_dict)

def set_zdir(poscar_data, hkl, phi=0):
    """Set z-direction of POSCAR along a given hkl direction and returns new data.

    Parameters
    ----------
    hkl : tuple
        (h,k,l) of the direction along which z-direction is to be set.
        Vector is constructed as h*a + k*b + l*c in cartesian coordinates.
    phi: float
        Rotation angle in degrees about z-axis to set a desired rotated view.

    Returns
    -------
    New instance of poscar with z-direction set along hkl.
    """
    if not isinstance(hkl, (list, tuple, np.ndarray)) and len(hkl) != 3:
        raise ValueError("hkl must be a list, tuple or numpy array of length 3.")

    p_dict = poscar_data.to_dict()
    basis = p_dict["basis"]
    zvec = to_R3(basis, [hkl])[0]  # in cartesian coordinates
    angle = np.arccos(
        zvec.dot([0, 0, 1]) / np.linalg.norm(zvec)
    )  # Angle between zvec and z-axis
    rot = rotation(
        angle_deg=np.rad2deg(angle), axis_vec=np.cross(zvec, [0, 0, 1])
    )  # Rotation matrix
    new_basis = rot.apply(basis)  # Rotate basis so that zvec is along z-axis
    p_dict["basis"] = new_basis
    p_dict["metadata"]["comment"] = f"Modified by ipyvasp"
    new_pos = serializer.PoscarData(p_dict)

    if phi:  # Rotate around z-axis
        return rotate_poscar(new_pos, angle_deg=phi, axis_vec=[0, 0, 1])

    return new_pos


def mirror_poscar(poscar_data, direction):
    "Mirror a POSCAR in a given direction. Sometime you need it before joining two POSCARs"
    poscar = poscar_data.to_dict()  # Avoid modifying original
    idx = "abc".index(direction)  # Check if direction is valid
    poscar["positions"][:, idx] = (
        1 - poscar["positions"][:, idx]
    )  # Trick: Mirror by subtracting from 1. not by multiplying with -1.
    return serializer.PoscarData(poscar)  # Return new POSCAR


def convert_poscar(poscar_data, atoms_mapping, basis_factor):
    """Convert a POSCAR to a similar structure of other atomic types or same type with strained basis.

    Parameters
    ----------
    atoms_mapping : dict
        A dictionary of {old_atom: new_atom} like {'Ga':'Al'} will convert GaAs to AlAs structure.
    basis_factor : float
        A scaling factor multiplied with basis vectors, single value (useful for conversion to another type)
        or list of three values to scale along (a,b,c) vectors (useful for strained structures).


    .. note::
        This can be used to strain basis vectors uniformly only. For non-uniform strain, use :func:`ipyvasp.POSCAR.deform`.
    """
    poscar_data = poscar_data.to_dict()  # Avoid modifying original
    poscar_data["types"] = {
        atoms_mapping.get(k, k): v for k, v in poscar_data["types"].items()
    }  # Update types
    basis = poscar_data["basis"].copy()  # Get basis to avoid modifying original

    if isinstance(basis_factor, (int, np.integer, float)):
        poscar_data["basis"] = basis_factor * basis  # Rescale basis
    elif isinstance(basis_factor, (list, tuple, np.ndarray)):
        if len(basis_factor) != 3:
            raise Exception("basis_factor should be a list/tuple/array of length 3")

        if np.ndim(basis_factor) != 1:
            raise Exception(
                "basis_factor should be a list/tuple/array of 3 int/float values"
            )

        poscar_data["basis"] = np.array(
            [
                basis_factor[0] * basis[0],
                basis_factor[1] * basis[1],
                basis_factor[2] * basis[2],
            ]
        )
    else:
        raise Exception(
            "basis_factor should be a list/tuple/array of 3 int/float values, got {}".format(
                type(basis_factor)
            )
        )
    poscar_data["SYSTEM"] = "".join(poscar_data["types"].keys())  # Update system name
    return serializer.PoscarData(poscar_data)  # Return new POSCAR


def transform_poscar(poscar_data, transformation, fill_factor=2, tol=1e-2):
    """Transform a POSCAR with a given transformation matrix or function that takes old basis and return target basis.
    Use `get_TM(basis1, basis2)` to get transformation matrix from one basis to another or function to return new basis of your choice.
    An example of transformation function is `lambda a,b,c: a + b, a-b, c` which will give a new basis with a+b, a-b, c as basis vectors.

    You may find errors due to missing atoms in the new basis, use `fill_factor` and `tol` to include any possible site in new cell.

    Examples
    --------
    - FCC primitive → 111 hexagonal cell: ``lambda a,b,c: (a-c,b-c,a+b+c) ~ [[1,0,-1],[0,1,-1],[1,1,1]]``
    - FCC primitive → FCC unit cell: ``lambda a,b,c: (b+c -a,a+c-b,a+b-c) ~ [[-1,1,1],[1,-1,1],[1,1,-1]]``
    - FCC unit cell → 110 tetragonal cell: ``lambda a,b,c: (a-b,a+b,c) ~ [[1,-1,0],[1,1,0],[0,0,1]]``


    .. note::
        This function keeps underlying lattice same. To apply strain, use `deform` function instead.
    """
    if callable(transformation):
        new_basis = np.array(transformation(*poscar_data.basis))  # mostly a tuple
        if new_basis.shape != (3, 3):
            raise Exception(
                "transformation function should return a tuple equivalent to 3x3 matrix"
            )
    elif np.ndim(transformation) == 2 and np.shape(transformation) == (3, 3):
        new_basis = np.matmul(transformation, poscar_data.basis)
    else:
        raise Exception(
            "transformation should be a function that accept 3 arguemnts or 3x3 matrix"
        )
    if not isinstance(fill_factor,int):
        raise TypeError(f'fill_factor should be int, got {type(fill_factor)}')

    _p = range(-fill_factor, fill_factor + 1)
    pos = np.concatenate([poscar_data.positions,[[i] for i,_ in enumerate(poscar_data.positions)]], axis=1) # keep track of index
    pos = np.concatenate([pos + [*p,0] for p in product(_p,_p,_p)],axis=0) # increaser by fill_factor^3
    pos[:,:3] = to_basis(new_basis, poscar_data.to_cartesian(pos[:,:3])) # convert to coords in this and to points in new
    pos = pos[(pos[:,:3] <= 1 - tol).all(axis=1) & (pos[:,:3] >= -tol).all(axis=1)]
    pos = pos[pos[:,-1].argsort()] # sort for species

    new_poscar = poscar_data.to_dict()  # Update in it
    new_poscar["basis"] = new_basis
    new_poscar["metadata"]["scale"] = np.linalg.norm(new_basis[0])
    new_poscar["metadata"]["comment"] = f"Transformed by ipyvasp"
    new_poscar["metadata"]["TM"] = get_TM(poscar_data.basis, new_basis)  # save Transformation matrix
    old_numbers = [len(v) for v in poscar_data.types.values()]


    uelems, start = {}, 0
    for k, v in poscar_data.types.items():
        uelems[k] = range(start, start + len(pos[(pos[:,-1] >= v.start) & (pos[:,-1] < v.stop)]))
        start = uelems[k].stop

    # warn if crystal formula changes
    new_numbers = [len(v) for v in uelems.values()]
    ratio = (np.array(new_numbers)/old_numbers).round(4) # Round to avoid floating point errors,can cover 1 to 10000 atoms transformation
    if len(np.unique(ratio)) != 1:
        print(tcolor.rb(f"WARNING: Transformation failed, atoms proportion changed: {old_numbers} -> {new_numbers}." 
              " If your transformation is an allowed one for this structure, increase `fill_factor` or `tol`."))

    new_poscar["types"] = uelems
    new_poscar["positions"] = np.array(pos[:,:3])
    return serializer.PoscarData(new_poscar)


def add_vaccum(poscar_data, thickness, direction, left=False):
    """Add vacuum to a POSCAR.

    Parameters
    ----------
    thickness : float
        Thickness of vacuum in Angstrom.
    direction : str
        Direction of vacuum. Can be 'a', 'b' or 'c'.
    left : bool
        If True, vacuum is added to left of sites. By default, vacuum is added to right of sites.
    """
    if direction not in "abc":
        raise Exception("Direction must be a, b or c.")

    poscar_dict = poscar_data.to_dict()  # Avoid modifying original
    basis = poscar_dict["basis"].copy()  # Copy basis to avoid modifying original
    pos = poscar_dict["positions"].copy()  # Copy positions to avoid modifying original
    idx = "abc".index(direction)
    norm = np.linalg.norm(basis[idx])  # Get length of basis vector
    s1, s2 = norm / (norm + thickness), thickness / (norm + thickness)  # Get scaling
    basis[idx, :] *= (thickness + norm) / norm  # Add thickness to basis
    poscar_dict["basis"] = basis
    if left:
        pos[:, idx] *= s2  # Scale down positions
        pos[:, idx] += s1  # Add vacuum to left of sites
        poscar_dict["positions"] = pos
    else:
        pos[:, idx] *= s1  # Scale down positions
        poscar_dict["positions"] = pos

    return serializer.PoscarData(poscar_dict)  # Return new POSCAR


def transpose_poscar(poscar_data, axes=[1, 0, 2]):
    "Transpose a POSCAR by switching basis from [0,1,2] -> `axes`. By Default, x and y are transposed."
    if isinstance(axes, (list, tuple, np.ndarray)) and len(axes) == 3:
        if not all(isinstance(i, (int, np.integer)) for i in axes):
            raise ValueError("`axes` must be a list of three integers.")

        poscar_data = poscar_data.to_dict()  #
        basis = poscar_data["basis"].copy()  # Copy basis to avoid modifying original
        positions = poscar_data[
            "positions"
        ].copy()  # Copy positions to avoid modifying original
        poscar_data["basis"] = basis[axes]  # Transpose basis
        poscar_data["positions"] = positions[:, axes]  # Transpose positions
        return serializer.PoscarData(poscar_data)  # Return new POSCAR
    else:
        raise Exception("`axes` must be a squence of length 3.")


def add_atoms(poscar_data, name, positions):
    "Add atoms with a `name` to a POSCAR at given `positions` in fractional coordinates."
    positions = np.array(positions)
    if (not np.ndim(positions) == 2) or (not positions.shape[1] == 3):
        raise ValueError("`positions` must be a 2D array of shape (n,3)")

    new_pos = np.vstack([poscar_data.positions, positions])  # Add new pos

    unique = poscar_data.types.to_dict()  # avoid modifying original
    unique[name] = range(len(poscar_data.positions), len(new_pos))

    data = poscar_data.to_dict()  # Copy data to avoid modifying original
    data["types"] = unique  # Update unique dictionary
    data["positions"] = new_pos  # Update positions
    data["metadata"][
        "comment"
    ] = f'{data["metadata"]["comment"]} + Added {name!r}'  # Update comment

    data["SYSTEM"] = "".join(data["types"].keys())  # Update system name
    return serializer.PoscarData(data)  # Return new POSCAR


def replace_atoms(poscar_data, func, name):
    """Replace atoms satisfying a `func(atom) -> bool` with a new `name`. Like `lambda a: a.symbol == 'Ga'`"""
    data = poscar_data.to_dict()  # Copy data to avoid modifying original
    mask = _masked_data(poscar_data, func)
    new_types = {**{k: [] for k in poscar_data.types.keys()}, name: []}

    for k, vs in data["types"].items():
        for idx in vs:
            if idx in mask:
                new_types[name].append(idx)
            else:
                new_types[k].append(idx)

    data["positions"] = np.vstack([data["positions"][t] for t in new_types.values()])
    idxs = np.cumsum([0, *map(len, new_types.values())])
    data["types"] = {
        k: range(idxs[i], idxs[i + 1])
        for i, k in enumerate(new_types.keys())
        if len(new_types[k]) != 0
    }
    data["SYSTEM"] = "".join(data["types"].keys())  # Update system name
    return serializer.PoscarData(data)  # Return new POSCAR

def sort_poscar(poscar_data, new_order):
    "sort poscar with new_order list/tuple of species."
    if not isinstance(new_order, (list, tuple)):
        raise TypeError(f"new_order should be a list/tuple of types, got {type(new_order)}")
    
    data = poscar_data.to_dict() 
    if not all([set(new_order).issubset(data["types"]), set(data["types"]).issubset(new_order)]):
        raise ValueError(f"new_order should contain all existings types {list(data['types'])}")
    
    data["types"] = {key:data["types"][key] for key in new_order}
    data["positions"] = data["positions"][[i for tp in data["types"].values() for i in tp]]
    idxs = np.cumsum([0, *map(len, data["types"].values())])
    data["types"] = {
        k: range(idxs[i], idxs[i + 1])
        for i, k in enumerate(data["types"].keys())
        if len(data["types"][k]) != 0
    }
    data["SYSTEM"] = "".join(data["types"].keys())  # Update system name
    return serializer.PoscarData(data)

def remove_atoms(poscar_data, func, fillby=None):
    """Remove atoms that satisfy `func(atom) -> bool` on their fractional coordinates like `lambda a: all(a.p < 1/2)`.
    `atom` passed to function is a namedtuple like `Atom(symbol,number,index,x,y,z)` which has extra attribute `p = array([x,y,z])`.
    If `fillby` is given, it will fill the removed atoms with atoms from fillby POSCAR.

    >>> remove_atoms(..., lambda a: sum((a.p - 0.5)**2) <= 0.25**2) # remove atoms in center of cell inside radius of 0.25

    .. note::
        The coordinates of fillby POSCAR are transformed to basis of given POSCAR, before filling.
        So a good filling is only guaranteed if both POSCARs have smaller lattice mismatch.
    """
    _validate_func(func) # need to validate for fillbay
    data = poscar_data.to_dict()  # Copy data to avoid modifying original
    positions = data["positions"]
    mask = _masked_data(poscar_data, lambda s: not func(s))

    new_types = {k: [] for k in poscar_data.types.keys()}
    for k, vs in data["types"].items():
        for idx in vs:
            if idx in mask:
                new_types[k].append(idx)

    if fillby:
        if not isinstance(fillby, serializer.PoscarData):
            raise ValueError("`fillby` must be instance of PoscarData class.")

        filldata = fillby.to_dict()
        positions = np.vstack(
            [data["positions"], to_basis(poscar_data.basis, fillby.coords)]
        )  # update positions of fillby in given data basis, not fillby basis

        def keep_pos(i, x, y, z):  # keep positions in basis of given data
            u, v, w = to_basis(poscar_data.basis, to_R3(fillby.basis, [[x, y, z]]))[0]
            return bool(func(_Atom('', 0, 0, u, v, w)))

        mask = _masked_data(fillby, keep_pos)
        N_prev = len(data["positions"])  # before filling
        new_types = {
            **{k: [] for k in filldata["types"]},
            **new_types,
        }  # Add new types from fillby but keep old types values

        for k, vs in filldata["types"].items():
            for idx in vs:
                if idx in mask:
                    new_types[k].append(N_prev + idx)

    data["positions"] = np.vstack([positions[t] for t in new_types.values()])
    idxs = np.cumsum([0, *map(len, new_types.values())])
    data["types"] = {
        k: range(idxs[i], idxs[i + 1])
        for i, k in enumerate(new_types.keys())
        if len(new_types[k]) != 0
    }
    data["SYSTEM"] = "".join(data["types"].keys())  # Update system name
    return serializer.PoscarData(data)  # Return new POSCAR


def deform_poscar(poscar_data, deformation):
    """Deform a POSCAR by a deformation as 3x3 ArrayLike, or a function that takee basis and returns a 3x3 ArrayLike,
    to be multiplied with basis (elementwise) and return a new POSCAR.

    .. note::
        This function can change underlying crystal structure if cell shape changes, to just change cell shape, use `transform` function instead.
    """
    poscar_dict = poscar_data.to_dict()  # make a copy

    if callable(deformation):
        try:
            poscar_dict["basis"] = np.array(
                deformation(*poscar_data.basis)
            )  # mostly tuple
        except:
            raise ValueError(
                "`deformation` function must be a function(a,b,c) -> 3x3 matrix to multiply with basis."
            )
    else:
        dmatrix = deformation

        if not isinstance(dmatrix, np.ndarray):
            dmatrix = np.array(dmatrix)

        if dmatrix.shape != (3, 3):
            raise ValueError(
                "`deformation` must be a 3x3 matrix or a function(a,b,c) -> 3x3 matrix to multiply with basis."
            )

        # Update basis by elemetwise multiplication
        poscar_dict["basis"] = poscar_data.basis * dmatrix

    poscar_dict["metadata"][
        "comment"
    ] = f'{poscar_data["metadata"]["comment"]} + Deformed POSCAR'
    return serializer.PoscarData(poscar_dict)  # Return new POSCAR


def view_poscar(poscar_data, **kwargs):
    "View a POSCAR in a jupyter notebook. kwargs are passed to splot_lattice. After setting a view, you can do view.f(**view.kwargs) to get same plot in a cell."

    def view(elev, azim, roll):
        ax = splot_lattice(poscar_data, **kwargs)
        ax.view_init(elev=elev, azim=azim, roll=roll)
    
    elev = IntSlider(description='elev', min=0,max=180,value=30, continuous_update=False)
    azim = IntSlider(description='azim', min=0,max=360,value=30, continuous_update=False)
    roll = IntSlider(description='roll', min=0,max=360,value=0, continuous_update=False)

    return interactive(view, elev=elev, azim=azim, roll=roll)
