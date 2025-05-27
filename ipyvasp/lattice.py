__all__ = [
    "POSCAR",
    "download_structure",
    "periodic_table",
    "get_kpath",
    "get_kmesh",
    "splot_bz",
    "iplot_bz",
    "ngl_viewer",
]

from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO
from itertools import permutations
from contextlib import suppress
from collections import namedtuple

import numpy as np
from pandas.io.clipboard import clipboard_get, clipboard_set
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go 


from .core import serializer
from .core import spatial_toolkit as stk
from .core.plot_toolkit import get_axes, iplot2widget
from .utils import _sig_kwargs, _sub_doc
from . import _lattice as plat
from ._lattice import (
    periodic_table,
    get_kpath,
    get_kmesh,
    splot_bz,
    iplot_bz,
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


def ngl_viewer(
    poscar,
    colors=None,
    sizes=0.5,
    plot_cell=True,
    linewidth=0.05,
    color=[0, 0, 0.2],
    bond_color="whitesmoke",
    width="400px",
    height="400px",
    plot_vectors=True,
    dashboard=False,
    origin=(0, 0, 0),
    eqv_sites = True,
):
    """Display structure in Jupyter notebook using nglview.

    Parameters
    ----------
    poscar : ipyvasp.POSCAR
    sizes : float or dict of type -> float
        Size of sites. Either one int/float or a mapping like {'Ga': 2, ...}.
    colors : color or sheme name or dict of type -> color
        Mapping of colors like {'Ga': 'red, ...} or a single color. Automatically generated color for missing types.
        If `colors = 'element'`, then element colors from `nglview` will be used.
        You can use `nglview.color.COLOR_SCHEME` to see available color schemes to use.
    plot_cell : bool
        Plot unit cell. Default True.
    linewidth : float
        Linewidth of cell edges.
    color : list or str
        Color of cell edges. Must be a valid color to support conversion to RGB via matplotlib.
    bond_color : str
        Color of bonds. Default "whitesmoke". Can be "element" or any other scheme from `nglview`.
    width : str
        Width of viewer. Default "400px".
    height : str
        Height of viewer. Default "400px".
    plot_vectors : bool
        Plot vectors. Default True. Only works if `plot_cell = True`.
    dashboard : bool
        Show dashboard. Default False. It just sets `view.gui_style = 'NGL'`.

    Returns
    -------
    NGLWidget
        An instance of nglview.NGLWidget object. You can use `.render_image()`, `.download_image()` etc. on it or can add more representations.

    .. note::
        `nglview` sometimes does not work in Jupyter lab. You can switch to classic notebook in that case.

    .. tip::
        See `Advanced NGLView usage <https://projects.volkamerlab.org/teachopencadd/talktorials/T017_advanced_nglview_usage.html>`_.
    """
    if not isinstance(poscar, POSCAR):
        raise TypeError("poscar must be an instance of POSCAR class.")

    try:
        import nglview as nv
    except ImportError:
        raise ImportError("Please install nglview to use this function.")

    # Only show equivalent sites if plotting cell, only shift origin otherwise
    poscar = POSCAR(  # don't change instance itself, make new one
        data=plat._fix_sites(poscar.data, eqv_sites=eqv_sites, origin=origin)
    )

    _types = poscar.data.types.to_dict()
    _fcs = plat._fix_color_size(_types,colors,sizes,0.5, backend='ngl')
    _sizes = [v['size'] for v in _fcs.values()]
    _colors = [v['color'] for v in _fcs.values()]

    view = nv.NGLWidget(
        nv.ASEStructure(poscar.to_ase()),
        width=width,
        height=height,
        default_representation=False,
    )
    view.clear()

    for e, r, c in zip(_types, _sizes, _colors):
        view.add_spacefill(radius=r, selection=f"#{e}", color=c)

    view.add_ball_and_stick(color=bond_color)
    view.camera = "orthographic"
    view.center()

    if plot_cell:
        # arrays.tolist() is important for nglview to write json
        shape = nv.shape.Shape(view=view)
        _color = mcolors.to_rgb(color)  # convert to rgb for nglview
        cell = poscar.get_cell()
        for face in cell.faces_coords:
            for p1, p2 in zip(face[1:], face[:-1]):
                shape.add_cylinder(p1.tolist(), p2.tolist(), _color, linewidth)

        for v in cell.vertices:
            shape.add_sphere(v.tolist(), _color, linewidth)

        if plot_vectors:
            for i, b in enumerate(cell.basis, start=1):
                tail = b - b / np.linalg.norm(b)
                shape.add_cone(
                    tail.tolist(), b.tolist(), _color, linewidth * 3, f"a{i}"
                )
    if dashboard:
        view.gui_style = "NGL"
    return view


def weas_viewer(poscar,
    sizes=1,
    colors=None,
    bond_length=None,
    model_style = 1,
    plot_cell=True,
    origin = (0,0,0),
    eqv_sites = True,
    ):
    """
    sizes : float or dict of type -> float
        Size of sites. Either one int/float or a mapping like {'Ga': 2, ...}.
    colors : color, color scheme or dict of type -> color
        Mapping of colors like {'Ga': 'red, ...} or a single color. Automatically generated color for missing types.
        You can use color schemes as 'VESTA','JMOL','CPK'.
    sizes : list
        List of sizes for each atom type.
    model_type: int
        whether to show Balls (0), Ball + Stick (1), Polyheda (2) or Sticks (3).
    plot_cell : bool
        Plot unit cell. Default True.
    bond_length : float or dict
        Length of bond in Angstrom. Auto calculated if not provides. Can be a dict like {'Fe-O':3.2,...} to specify bond length between specific types.
    
    Returns a WeasWidget instance. You can use `.export_image`, `save_image` and other operations on it.
    Read what you can do more with `WeasWidget` [here](https://weas-widget.readthedocs.io/en/latest/index.html).
    """
    
    from weas_widget import WeasWidget

    if len(poscar.data.positions) < 1:
        raise ValueError("Need at least 1 atom!")
    
    # Only show equivalent sites if plotting cell, only shift origin otherwise
    poscar = POSCAR(  # don't change instance itself, make new one
        data=plat._fix_sites(poscar.data, eqv_sites=eqv_sites, origin=origin)
    )

    w = WeasWidget(from_ase=poscar.to_ase())
    w.avr.show_bonded_atoms = False if plot_cell else True # plot_cell fix atoms itself
    w.avr.model_style = model_style
    w.avr.show_cell = plot_cell         

    if bond_length:
        if isinstance(bond_length,(int,float)):
            for a,b in permutations(list(poscar.data.types.keys()),2):
                with suppress(KeyError):
                    w.avr.bond.settings[f'{a}-{b}'].update({'max': bond_length})
        elif isinstance(bond_length, dict):
            for key, value in bond_length.items():
                w.avr.bond.settings[key].update({'max': value})
    
    _fcs = plat._fix_color_size(poscar.data.types.to_dict(), colors, sizes, 1)

    for key, value in _fcs.items():
        w.avr.species.settings[key].update({"radius": value["size"]})

    if isinstance(colors, str) and colors in ['VESTA','JMOL','CPK']:
        w.avr.color_type = colors
    else:
        colors = {key: value["color"] for key, value in _fcs.items()}
        for key,value in colors.items():
            w.avr.species.settings[key].update({"color": value})
        for (k1,c1), (k2,c2) in permutations(colors.items(),2):
            with suppress(KeyError):
                w.avr.bond.settings[f'{k1}-{k2}'].update({'color1':c1,'color2':c2})
    
    return w


class _AutoRenderer:
    _figw = None
    _kws = {} 

    def __init__(self, pc_cls):
        self._pc = pc_cls

    def on(self, template=None):
        "Enable auto rendering. In Jupyterlab, you can use `Create New View for Output` to drag a view on side."
        self.off()
        type(self)._figw = iplot2widget(self._pc._last.iplot_lattice(**self._kws), fig_widget=self._figw,template=template)
        
        def ip_display(that):
            iplot2widget(that.iplot_lattice(**self._kws), fig_widget=self._figw, template=template)
        
        self._pc._ipython_display_ = ip_display
        
        from ipywidgets import Button, VBox
        btn = Button(description='Disable Auto Rendering',icon='close',layout={'width': 'max-content'})
        btn.on_click(lambda btn: self.off())
        type(self)._box = VBox([btn, self._figw])
        return display(self._box)

    def off(self):
        "Disable auto rendering."
        if hasattr(self, '_box'):
            self._box.close()
            type(self)._figw = None # no need to close figw, it raise warning, but now garbage collected
        
        if hasattr(self._pc, '_ipython_display_'):
            del self._pc._ipython_display_

    @_sig_kwargs(plat.iplot_lattice,('poscar_data',))
    def update_params(self, **kwargs):
        type(self)._kws = kwargs
        if hasattr(self._pc, '_ipython_display_'):
            self._pc._last._ipython_display_()
    
    @property
    def params(self):
        return self._kws.copy() # avoid messing original
    
    def __repr__(self):
        return f"AutoRenderer(params = {self._kws})"

class POSCAR:
    _cb_instance = {}  # Loads last clipboard data if not changed
    _mp_instance = {}  # Loads last mp data if not changed

    def __init__(self, path=None, content=None, data=None):
        """
        POSCAR class to contain data and related methods, data is PoscarData, json/tuple file/string.

        Parameters
        ----------
        path : path to file
        content : string of POSCAR content
        data : PoscarData object. This assumes positions are in fractional coordinates.

        Prefrence order: data > content > path

        Note: POSCAR operations that need a `func` accept basis, atom tuple, label etc. Read their documentation.
        
        ```python
        pc = POSCAR()
        pc.filter_atoms(lambda a: a.symbol == 'Ga') # a is namedtuple `Atom(symbol,number,index,x,y,z)` which has extra attribute `p = array([x,y,z])`.
        pc.transform(lambda a,b,c: (a+b,a-b,c)) # basis or transform matrix
        pc.splot_lattice(lambda lab: lab.to_latex()) # lab is str subclass like `AtomLabel('Ga 1')` with extra attributes `symbol,number, to_latex()` that can be used to show specific sites labels only.
        ```

        Tip: You can use `self.auto_renderer.on()` to keep doing opertions and visualize while last line of any cell is a POSCAR object.
        """
        self._path = Path(path or "POSCAR")  # Path to file
        self._content = content
        self.__class__._last = self # need this to access in lambda in chain operations

        if data:
            self._data = serializer.PoscarData.validated(data)
        else:
            self._data = plat.export_poscar(path=str(self.path), content=content)

    def __repr__(self):
        atoms = ", ".join([f"{k}={len(v)}" for k, v in self.data.types.items()])
        lat = ", ".join(
            [
                f"{k}={v}"
                for k, v in zip(
                    "abcαβγ", (*self.data.norms.round(3), *self.data.angles.round(3))
                )
            ]
        )
        return f"{self.__class__.__name__}({atoms}, {lat})"

    def __str__(self):
        return self.content

    @property
    def path(self):
        return self._path
    
    @property
    def last(self):
        """Points to last created POSCAR instance during chained operations! You don't need to store results.
        
        ```python
        pc = POSCAR()
        pc.filter_atoms(lambda a: a.index in pc.data.types.Ga) # FINE, can use a.symbol == 'Ga' too, but we need to show a point below
        pc.set_boundary([-2,2]).filter_atoms(lambda a: a.index in pc.data.types.Ga) # INCORRECT sites picked
        pc.set_boundary([-2,2]).filter_atoms(lambda a: a.index in pc.last.data.types.Ga) # PERFECT, pc.last is output of set_boundary
        ```
        """
        return self._last
    
    @property
    def auto_renderer(self):
        """A renderer for auto viewing POSCAR when at last line of cell.

        Use `auto_renderer.on()` to enable it.
        Use `auto_renderer.off()` to disable it.
        Use `auto_renderer.[params, update_params()]` to view and update parameters.
        
        In Jupyterlab, you can use `Create New View for Output` to drag a view on side.
        In VS Code, you can open another view of Notebook to see it on side while doing operations.
        """
        if not hasattr(self, '_renderer'):
            self.__class__._renderer = _AutoRenderer(self.__class__) # assign to class
        return self._renderer

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

        return Atoms(  # ASE positions are cartesian, not fractional
            symbols=self.data.symbols, positions=self.data.coords, cell=self.data.basis
        )

    def view(self, viewer=None, **kwargs):
        """View POSCAR in notebook. If viewer is given it will be passed ase.visualize.view. You need to have ase installed.

        kwargs are passed to self.splot_lattice if viewer is None, otherwise a  single keyword argument `data` is passed to ase viewer.
        data should be volumetric data for ase.visualize.view, such as charge density, spin density, etc.

        .. tip::
            Use ``self.view_ngl()`` if you don't want to pass ``viewer = 'nglview'`` to ASE viewer or not showing volumetric data.
        """
        if viewer is None:
            return plat.view_poscar(self.data, **kwargs)
        elif viewer in "weas":
            return weas_viewer(self, **kwargs)
        elif viewer in "plotly":
            return self.view_plotly(**kwargs)
        elif viewer in "nglview":
            return print(
                f"Use `self.view_ngl()` for better customization in case of viewer={viewer!r}"
            )
        else:
            from ase.visualize import view

            return view(self.to_ase(), viewer=viewer, data=kwargs.get("data", None))

    @_sub_doc(ngl_viewer, {"poscar :.*colors :": "colors :"})
    @_sig_kwargs(ngl_viewer, ("poscar",))
    def view_ngl(self, **kwargs):
        return ngl_viewer(self, **kwargs)
    
    @_sub_doc(weas_viewer)
    @_sig_kwargs(weas_viewer, ("poscar",))
    def view_weas(self, **kwargs):
        return weas_viewer(self, **kwargs)

    def view_kpath(self, height='400px'):
        "Initialize a KPathWidget instance to view kpath for current POSCAR, and you can select others too."
        from .widgets import KPathWidget

        return KPathWidget([self.path,],height=height)

    @_sub_doc(plat.iplot_lattice)
    @_sig_kwargs(plat.iplot_lattice, ("poscar_data",))
    def view_plotly(self, **kwargs):
        return iplot2widget(self.iplot_lattice(**kwargs))

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
    def download(cls, formula, mp_id, api_key=None, save_key=False):
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

    def to_dict(self):
        "Returns a dictionary that can be modified generally and passed to `POSCAR.new` method."
        data = self.data.copy()  # avoid overwriting numpy arrays
        out = {"system": data.SYSTEM, "basis": data.basis}
        out["sites"] = {k: data.positions[v] for k, v in data.types.items()}
        return out

    @classmethod
    def new(cls, basis, sites, scale=1, system=None):
        """
        Crate a new POSCAR instance from basis and sites.

        Parameters
        ----------
        basis : array_like of shape (3, 3)
        sites : dict, key is element and value is array_like of shape (n, 3)
            For example, {'Mg': [[0, 0, 0]], 'Cl': [[1/3, 2/3,0.214],[2/3,1/3,0.786]]} for MgCl2.
        scale : int or float used to scale basis and kept in metadata to use when writing to file.
        system : str, name of the structure, if None, will be inferred from sites.
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
            "SYSTEM": system if system else "".join(type_dict.keys()),
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
    
    @property
    def metadata(self):
        "Metadata associated with this POSCAR."
        return self._data.metadata

    def copy(self):
        "Copy POSCAR object. It avoids accidental changes to numpy arrays in original object."
        return self.__class__(data=self.data.copy())
    
    @_sub_doc(plat.sort_poscar)
    def sort(self, new_order):
        return self.__class__(data=plat.sort_poscar(self.data, new_order))


    @property
    def content(self):
        "POSCAR content."
        with redirect_stdout(StringIO()) as f:
            self.write(outfile=None)  # print to stdout
            return f.getvalue()

    @property
    def bz(self):
        "Regular BZ data. Shortcut for `get_bz(primitive = False)`."
        if not hasattr(self, "_bz"):
            self._bz = self.get_bz(primitive=False)
        return self._bz

    @property
    def pbz(self):
        "Primitive BZ data. Shortcut for `get_bz(primitive = True)`."
        return self.get_bz(primitive=True)

    @property
    def cell(self):
        if not hasattr(self, "_cell"):
            self._cell = self.get_cell()
        return self._cell
    
    def get_plane(self, hkl, d=1/2,tol=1e-2):
        """Returns tuple `Plane(point, normal, vertices)` of a plane bound inside cell. .
        hkl should be list of three miller indices. d is fractional distance in range 0,1 in direction of hkl. 
        e.g. if there are 8 planes of atoms in a cubic cell, d = 0, 1/8,...7/8, 1 match position of those planes.
        """
        from sympy import Point3D, Line3D, Plane
        V = self.data.rec_basis.dot(hkl)
        normal = V/np.linalg.norm(V)
        point = d*normal*np.linalg.norm(self.data.basis.dot(hkl)) # to make d 0-1
        P = Plane(Point3D(*point),normal_vector=normal)

        pts = []
        for e in self.cell.vertices[self.cell.edges]:
            L = Line3D(Point3D(*e[0]),Point3D(*e[1]))
            if (isc := P.intersection(L)) and isinstance(isc[0],Point3D):
                pts.append(isc[0])

        pts = np.unique(np.array(pts,dtype=float),axis=0)
        pts = pts[stk.order(pts,)]
        qts = self.cell.to_fractional(pts)
        qts = qts[(qts >= -tol).all(axis=1) & (qts <= 1 + tol).all(axis=1)]
        pts = self.cell.to_cartesian(qts)
        return namedtuple("Plane","point normal vertices")(point, normal, pts)
    
    def splot_plane(self, hkl, d=1/2,tol=1e-2,ax=None, **kwargs):
        """Provide hkl and a 3D axes to plot plane. kwargs are passed to `mpl_toolkits.mplot3d.art3d.Poly3DCollection`
        Note: You may get wrong plane if your basis are not aligned to axes. So you can use `transpose` or `set_zdir` methods before plottling cell.
        """
        P = self.get_plane(hkl,d=d,tol=tol).vertices
        if ax is None:
            ax = get_axes(axes_3d=True)
            ax.set( # it does not show otherwise
                xlim=[P[:,0].min(),P[:,0].max()],
                ylim=[P[:,1].min(),P[:,1].max()],
                zlim=[P[:,2].min(),P[:,2].max()]
            )
        kwargs = {'alpha':0.5,'color':'#898','shade': False, 'label':str(hkl), **kwargs}
        ax.add_collection(Poly3DCollection([P],**kwargs))
        ax.autoscale_view()
        return ax

    def iplot_plane(self, hkl, d = 1/2, tol=1e-3, fig=None,**kwargs):
        "Plot plane on a plotly Figure. kwargs are passed to `plotly.graph_objects.Mesh3d`."
        if fig is None:
            fig = go.Figure()

        P = self.get_plane(hkl,d=d,tol=tol)
        kwargs['delaunayaxis'] = ('xyz')[np.abs(np.eye(3).dot(P.normal)).argmax()] # with alphahull=-1, delaunayaxis to be set properly
        kwargs = {**dict(color='#8a8',opacity=0.7,alphahull=-1, showlegend=True,name=str(hkl)),**kwargs}
        fig.add_trace(go.Mesh3d({k:v for v,k in zip(P.vertices.T, 'xyz')},**kwargs))
        return fig

    @_sub_doc(stk.get_bz, {"basis :.*loop :": "loop :"})
    @_sig_kwargs(stk.get_bz, ("basis",))
    def get_bz(self, **kwargs):
        return stk.get_bz(self.data.rec_basis, **kwargs)

    def get_cell(self, loop=True):
        "See docs of `get_bz`, same except space is inverted and no factor of 2pi."
        return serializer.CellData(
            stk.get_bz(  # data.basis makes prmitive cell in direct space
                basis=self.data.basis, loop=loop, primitive=True
            ).to_dict()
        )  # cell must be primitive

    @_sub_doc(plat.splot_bz, {"bz_data :.*plane :": "plane :"})
    @_sig_kwargs(plat.splot_bz, ("bz_data",))
    def splot_bz(self, plane=None, **kwargs):
        return plat.splot_bz(bz_data=self.bz, plane=plane, **kwargs)

    @_sub_doc(plat.splot_kpath)
    @_sig_kwargs(plat.splot_kpath, ("bz_data",))
    def splot_kpath(self, kpoints, **kwargs):
        return plat.splot_kpath(self.bz, kpoints=kpoints, **kwargs)

    @_sig_kwargs(plat.splot_bz, ("bz_data",))
    def splot_cell(self, plane=None, **kwargs):
        "See docs of `splot_bz`, everything is same except space is inverted."
        return plat.splot_bz(bz_data=self.cell, plane=plane, **kwargs)

    @_sub_doc(plat.iplot_bz, {"bz_data :.*fill :": "fill :"})
    @_sig_kwargs(plat.iplot_bz, ("bz_data",))
    def iplot_bz(self, **kwargs):
        return plat.iplot_bz(bz_data=self.bz, **kwargs)

    @_sig_kwargs(plat.iplot_bz, ("bz_data", "special_kpoints"))
    def iplot_cell(self, **kwargs):
        "See docs of `iplot_bz`, everything is same except space is iverted."
        return plat.iplot_bz(bz_data=self.cell, special_kpoints=False, **kwargs)

    @_sub_doc(plat.splot_lattice)
    @_sig_kwargs(plat.splot_lattice, ("poscar_data", "plane"))
    def splot_lattice(self, plane=None, **kwargs):
        return plat.splot_lattice(self.data, plane=plane, **kwargs)

    @_sub_doc(plat.iplot_lattice)
    @_sig_kwargs(plat.iplot_lattice, ("poscar_data",))
    def iplot_lattice(self, **kwargs):
        return plat.iplot_lattice(self.data, **kwargs)

    @_sub_doc(plat.write_poscar)
    @_sig_kwargs(plat.write_poscar, ("poscar_data",))
    def write(self, outfile=None, **kwargs):
        return plat.write_poscar(self.data, outfile=outfile, **kwargs)

    @_sub_doc(plat.join_poscars)
    @_sig_kwargs(plat.join_poscars, ("poscar_data", "other"))
    def join(self, other, direction="c", **kwargs):
        return self.__class__(
            data=plat.join_poscars(
                poscar_data=self.data, other=other.data, direction=direction, **kwargs
            )
        )

    @_sub_doc(plat.scale_poscar)
    @_sig_kwargs(plat.scale_poscar, ("poscar_data",))
    def scale(self, scale=(1, 1, 1), **kwargs):
        return self.__class__(data=plat.scale_poscar(self.data, scale, **kwargs))
    
    @_sub_doc(plat.set_boundary)
    @_sig_kwargs(plat.set_boundary,("poscar_data",))
    def set_boundary(self, a = [0,1], b=[0,1],c=[0,1]):
        return self.__class__(data = plat.set_boundary(self.data, a=a,b=b,c=c))
    
    @_sub_doc(plat.filter_atoms)
    @_sig_kwargs(plat.filter_atoms,("poscar_data",))
    def filter_atoms(self, func, tol=0.01):
        return self.__class__(data = plat.filter_atoms(self.data, func,tol=tol))

    @_sub_doc(plat.rotate_poscar)
    def rotate(self, angle_deg, axis_vec):
        return self.__class__(
            data=plat.rotate_poscar(self.data, angle_deg=angle_deg, axis_vec=axis_vec)
        )

    @_sub_doc(plat.set_zdir)
    def set_zdir(self, hkl, phi=0):
        return self.__class__(data=plat.set_zdir(self.data, hkl, phi=phi))

    @_sub_doc(plat.translate_poscar)
    def translate(self, offset):
        return self.__class__(data=plat.translate_poscar(self.data, offset=offset))

    @_sub_doc(plat.repeat_poscar)
    def repeat(self, n, direction):
        return self.__class__(
            data=plat.repeat_poscar(self.data, n=n, direction=direction)
        )

    @_sub_doc(plat.mirror_poscar)
    def mirror(self, direction):
        return self.__class__(data=plat.mirror_poscar(self.data, direction=direction))

    @_sub_doc(stk.get_TM, replace={"basis1": "self.basis"})
    def get_TM(self, target_basis):
        return stk.get_TM(self.data.basis, target_basis)

    @_sub_doc(plat.transform_poscar)
    def transform(self, transformation, fill_factor=2, tol=1e-2):
        return self.__class__(
            data=plat.transform_poscar(self.data, transformation, fill_factor=fill_factor, tol=tol)
        )

    @_sub_doc(plat.transpose_poscar)
    def transpose(self, axes=[1, 0, 2]):
        return self.__class__(data=plat.transpose_poscar(self.data, axes=axes))

    @_sub_doc(plat.add_vaccum)
    def add_vaccum(self, thickness, direction, left=False):
        return self.__class__(
            data=plat.add_vaccum(
                self.data, thickness=thickness, direction=direction, left=left
            )
        )

    @_sub_doc(plat.add_atoms)
    def add_atoms(self, name, positions):
        return self.__class__(
            data=plat.add_atoms(self.data, name=name, positions=positions)
        )

    @_sub_doc(plat.remove_atoms)
    def remove_atoms(self, func, fillby=None):
        if fillby and not isinstance(fillby, POSCAR):
            raise TypeError("fillby should be an instance of POSCAR class.")

        return self.__class__(
            data=plat.remove_atoms(
                self.data, func=func, fillby=fillby.data if fillby else None
            )
        )

    @_sub_doc(plat.replace_atoms)
    def replace_atoms(self, func, name):
        return self.__class__(data=plat.replace_atoms(self.data, func=func, name=name))

    @_sub_doc(plat.convert_poscar)
    def convert(self, atoms_mapping, basis_factor):
        return self.__class__(
            data=plat.convert_poscar(
                self.data, atoms_mapping=atoms_mapping, basis_factor=basis_factor
            )
        )

    @_sub_doc(plat.deform_poscar)
    def deform(self, deformation):
        return self.__class__(
            data=plat.deform_poscar(self.data, deformation=deformation)
        )

    @_sub_doc(get_kmesh, {"poscar_data :.*\*args :": "*args :"})
    @_sig_kwargs(get_kmesh, ("poscar_data",))
    def get_kmesh(self, *args, **kwargs):
        return get_kmesh(self.data, *args, **kwargs)

    @_sub_doc(get_kpath, {"rec_basis :.*\n\n": "\n\n"})
    @_sig_kwargs(get_kpath, ("rec_basis",))
    def get_kpath(self, kpoints, n: int = 5, **kwargs):
        return get_kpath(kpoints, n=n, **kwargs, rec_basis=self.data.rec_basis)
