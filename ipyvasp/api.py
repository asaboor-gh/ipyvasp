
__all__ = ['download_structure', '__all__', 'parse_text', 'POSCAR', 'LOCPOT', 'CHG', 'ELFCAR', 'PARCHG', 'OUTCAR',
           'get_axes', 'Bands', 'DOS']

from pathlib import Path
from itertools import islice, product
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
from IPython.display import display
from pandas.io.clipboard import clipboard_get, clipboard_set


from . import parser as vp
from . import splots as sp
from . import iplots as ip
from . import sio as sio
from . import widgets as wdg
from . import utils as gu
from . import serializer
from . import surfaces as srf
from .utils import _sig_kwargs, _sub_doc


def download_structure(formula, mp_id=None, max_sites=None,min_sites=None, api_key=None,save_key = False):
    """Download structure data from Materials project website.
    Args:
        - formula: chemical formula of the material.
        - mp_id: Materials project id of material.
        - max_sites: maximum number of sites in structure to download.
        - min_sites: minimum number of sites in structure to download.
    > max_sites and min_sites are used to filter the number of sites in structure, or use mp_id to download a specific structure.
    - **One Time API Key**
        - api_key: API key from Materials project websit, if you use save_key=True, never required again.
        - save_key: Save API key to file. You can save any time of key or device changed.
    - **Return**
        List of Structure data containing attribute/method `cif`/`export_poscar, write_cif` etc.
    """
    mp = sio.InvokeMaterialsProject(api_key= api_key)
    output = mp.request(formula=formula,mp_id=mp_id,max_sites=max_sites,min_sites=min_sites) # make a request
    if save_key and isinstance(api_key,str):
        mp.save_api_key(api_key)
    if mp.success:
        return output
    else:
        raise ConnectionError('Connection was not sccessful. Try again!')

# Direct function exports from modules
_memebers = (
    gu.set_dir,
    gu.list_files,
    gu.transform_color,
    gu.create_colormap,
    gu.interpolate_data,
    sio.get_kpath,
    sio.quiver3d,
    sio.rotation,
    sio.to_basis,
    sio.to_R3,
    sio.get_kpath,
    sio.periodic_table,
    wdg.summarize,
    wdg.load_results,
    vp.minify_vasprun,
    vp.xml2dict,
    ip.iplot2html,
    ip.iplot2widget,
    sp.plt2html,
    sp.plt2text,
    sp.show,
    sp.savefig,
    sp.append_axes,
    sp.join_axes,
    sp.add_colorbar,
    sp.color_cube,
    sp.color_wheel,
    sp.add_legend,
    sp.add_text,
    wdg.FilesWidget,
    wdg.BandsWidget,
    wdg.KpathWidget,
    srf.SpinDataFrame,
)

# Subset of functions from modules in __all__ to make exportable as *
__all__ = [*[_m.__name__ for _m in _memebers],*[a for a in __all__ if a != '__all__']]
for _m in _memebers:
    locals()[_m.__name__] = _m # Assign to local namespace that can be exported, classes only have __name__, not name

@_sig_kwargs(vp.gen2numpy,skip_params = ('gen',))
@_sub_doc(vp.gen2numpy,'gen :', replace= {'shape :':'path : Path to file containing data.\nshape :'})
def parse_text(path, shape, slice, **kwargs):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File {path!r} does not exists")

    with p.open('r', encoding='utf-8') as f:
        gen = islice(f, 0, None)
        data = vp.gen2numpy(gen, shape, slice, **kwargs) # should be under open file context
    return data


class POSCAR:
    _cb_instance = {} # Loads last clipboard data if not changed
    _mp_instance = {} # Loads last mp data if not changed
    
    def __init__(self,path = None,content = None,data = None):
        """
        POSACR class to contain data and related methods, data is PoscarData, json/tuple file/string.
        Do not use `data` yourself, it's for operations on poscar.
        
        Parameters
        ----------
        path : path to file
        content : string of POSCAR content
        data : PoscarData object. This assumes positions are in fractional coordinates.

        Prefrence order: data > content > path
        """
        self._path = Path(path or 'POSACR') # Path to file
        self._content = content
        self._sd = None # Selective dynamics Array will be stored here if applied.

        if data:
            self._data = serializer.PoscarData.validated(data)
        else:
            self._data = sio.export_poscar(path = str(self.path),content = content)
        # These after data to work with data
        self._primitive = False
        self._bz = self.get_bz(primitive = False) # Get defualt regular BZ from sio
        self._cell = self.get_cell() # Get defualt cell
        self._plane = None # Get defualt plane, changed with splot_bz
        self._ax = None # Get defualt axis, changed with splot_bz

    def __repr__(self):
        atoms = ', '.join([f'{k}={len(v)}' for k,v in self._data.types.items()])
        lat = ', '.join([f'{k}={v}' for k,v in zip('abcαβγ',(*self._data.norms.round(3), *self._data.angles.round(3)))])
        return f"{self.__class__.__name__}({atoms}, {lat})"

    def __str__(self):
        return self.content
    
    @property
    def path(self): return self._path
    
    def to_ase(self):
        """Convert to ase.Atoms format. You need to have ase installed.
        You can apply all ase methods on this object after conversion.
        
        Example
        -------
        ```python
        from ase.visualize import view
        structure = poscar.to_ase()
        view(structure) # POSCAR.view() also uses this method if viewer is given.
        reciprocal_lattice = structure.cell.get_bravais_lattice()
        reciprocal_lattice.plt_bz() # Plot BZ usinn ase, it also plots suggested band path.
        ```
        """
        from ase import Atoms
        symbols = [lab.split()[0] for lab in self.data.labels] # Remove numbers from labels
        return Atoms(symbols = symbols, positions = self.data.positions,cell = self.data.basis)
    
    def view(self, viewer = None, **kwargs):
        """View POSCAR in notebook. If viewer is given it will be passed ase.visualize.view. You need to have ase installed.
        
        kwargs are passed to self.splot_lattice if viewer is None, otherwise a  single keyword argument `data` is passed to ase viewer.
        data should be volumetric data for ase.visualize.view, such as charge density, spin density, etc.
        """
        if viewer is None:
            return sio.view_poscar(self.data, **kwargs)
        else:
            from ase.visualize import view
            return view(self.to_ase(), viewer = viewer, data = kwargs.get('data',None))
    
    def view_kpath(self):
        "Initialize a KpathWidget instance to view kpath for current POSCAR, and you can select others too."
        return wdg.KpathWidget(path = str(self.path.parent), glob = self.path.name)

    @classmethod
    def from_file(cls,path):
        "Load data from POSCAR file"
        return cls(path = path)
    

    @classmethod
    def from_string(cls,content):
        "content should be a valid POSCAR string"
        try:
            return cls(content = content)
        except:
            raise ValueError(f"Invalid POSCAR string!!!!!\n{content}")

    @classmethod
    def from_materials_project(cls,formula, mp_id, api_key = None, save_key = False):
        """Downloads POSCAR from materials project. `mp_id` should be string associated with a material on their website. `api_key` is optional if not saved.
        Get your API key from https://legacy.materialsproject.org/open and save it using `save_key` to avoid entering it everytime.
        """
        if cls._mp_instance and cls._mp_instance['kwargs'] == {'formula':formula,'mp_id':mp_id}:
            if api_key and save_key: # If user wants to save key even if data is loaded from cache
                sio._save_mp_API(api_key)
            
            return cls._mp_instance['instance']
        
        instance = cls(data = download_structure(formula=formula,mp_id=mp_id,api_key=api_key,save_key=save_key)[0].export_poscar())
        cls._mp_instance = {'instance':instance,'kwargs':{'formula':formula,'mp_id':mp_id}}
        return instance
    
    @classmethod
    def from_cif(cls, path):
        "Load data from cif file"
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"File {path!r} does not exists")
        
        with p.open('r', encoding='utf-8') as f:
            cif_str = f.read()
            poscar_content = sio._cif_str_to_poscar_str(cif_str, "Exported from cif file using ipyvasp")
        
        return cls.from_string(poscar_content)
        
        
    @classmethod
    def new(cls, scale, basis, sites, name = None):
        raise NotImplementedError("This method is not implemented yet.")

    @classmethod
    def from_clipborad(cls):
        "Read POSCAR from clipboard (based on clipboard reader impelemented by pandas library) It picks the latest from clipboard."
        try:
            instance = cls.from_string(content = clipboard_get()) # read from clipboard while allowing error for bad data
            if isinstance(instance,cls): # if valid POSCAR string
                cls._cb_instance = {'instance':instance} # cache instance
                return instance
        except:
            if cls._cb_instance:
                print("Loading from previously cached clipboard data, as current data is not valid POSCAR string.")
                return cls._cb_instance['instance']
            else:
                raise ValueError("Clipboard does not contain valid POSCAR string and no previous data is cached.")

    def to_clipboard(self):
        "Writes POSCAR to clipboard (as implemented by pandas library) for copy in other programs such as vim."
        clipboard_set(self.content) # write to clipboard

    @property
    def data(self):
        "Data object in POSCAR."
        return self._data

    def copy(self):
        "Copy POSCAR object. It avoids accidental changes to numpy arrays in original object."
        return self.__class__(data = self._data.copy())

    @property
    def content(self):
        "POSCAR content."
        with redirect_stdout(StringIO()) as f:
            self.write(outfile = None) # print to stdout
            return f.getvalue()

    @property
    def bz(self):
        return self._bz

    @property
    def sd(self):
        return self._sd

    @property
    def cell(self):
        return self._cell

    @_sub_doc(sio.get_bz,'path_pos :')
    @_sig_kwargs(sio.get_bz,('path_pos',))
    def get_bz(self, **kwargs):
        self._bz = sio.get_bz(path_pos = self._data.basis, **kwargs)
        self._primitive = kwargs.get('primitive',False)
        return self._bz

    @_sig_kwargs(sio.get_bz,('path_pos',))
    def set_bz(self,**kwargs):
        """Set BZ in primitive or regular shape. returns None, just set self.bz, Useful in switching between primitive and regular BZ during plotting."""
        self.get_bz(**kwargs)

    def get_cell(self, loop = True):
        "See docs of `get_bz`, same except space is inverted and no factor of 2pi."
        self._cell = serializer.CellData(sio.get_bz(path_pos = self._data.rec_basis,loop = loop, primitive = True).to_dict()) # cell must be primitive
        return self._cell

    @_sub_doc(sio.splot_bz,'bz_data :')
    @_sig_kwargs(sio.splot_bz,('bz_data',))
    def splot_bz(self, plane = None, **kwargs):
        self._plane = plane # Set plane for splot_kpath
        new_ax = sio.splot_bz(bz_data = self._bz, plane = plane, **kwargs)
        self._ax = new_ax # Set ax for splot_kpath
        return new_ax

    def splot_kpath(self, kpoints, labels = None, fmt_label = lambda x: (x, {'color': 'blue'}), **kwargs):
        """
        Plot k-path over existing BZ.
        
        Parameters
        ----------
        kpoints : list of k-points in fractional coordinates. e.g. [(0,0,0),(0.5,0.5,0.5),(1,1,1)] in order of path.
        labels : list of labels for each k-point in same order as `knn_inds`.
        fmt_label : function, takes a label from labels and should return a string or (str, dict) of which dict is passed to `plt.text`.
        
        kwargs are passed to `plt.plot` with some defaults.

        You can get `kpoints = POSCAR.get_bz().specials.masked(lambda x,y,z : (-0.1 < z 0.1) & (x >= 0) & (y >= 0))` to get k-points in positive xy plane.
        Then you can reorder them by an indexer like `kpoints = kpoints[[0,1,2,0,7,6]]`, note double brackets, and also that point at zero index is taken twice.

        > Tip: You can use this function multiple times to plot multiple/broken paths over same BZ.
        """
        if not self._bz or not self._ax:
            raise ValueError("BZ not found, use `splot_bz` first")
        
        if not np.ndim(kpoints) == 2 and np.shape(kpoints)[-1] == 3:
            raise ValueError("kpoints must be 2D array of shape (N,3)")

        ijk = [0,1,2]
        _mapping = {'xy':[0,1],'xz':[0,2],'yz':[1,2],'zx':[2,0],'zy':[2,1],'yx':[1,0]}
        if isinstance(self._plane, str) and self._plane in _mapping:
            ijk = _mapping[self._plane]

        if not labels:
            labels = ["[{0:5.2f}, {1:5.2f}, {2:5.2f}]".format(x,y,z) for x,y,z in kpoints]
            
        sio._validate_label_func(fmt_label, labels[0])
            
        coords = self.to_R3(kpoints, reciprocal = True)[:,ijk]
        kwargs = dict(color = 'blue',linewidth=0.8,marker='.',markersize=10) | kwargs # need some defaults
        self._ax.plot(*coords.T,**kwargs)
        
        for c,text in zip(coords, labels):
            lab, textkws = fmt_label(text), {}
            if isinstance(lab, (list,tuple)):
                lab, textkws = lab
            self._ax.text(*c,lab,**textkws)
            
        return self._ax

    @_sig_kwargs(sio.splot_bz,('bz_data',))
    def splot_cell(self, plane = None, **kwargs):
        "See docs of `splot_bz`, everything is same except space is inverted."
        return sio.splot_bz(bz_data = self._cell,plane = plane, **kwargs)

    @_sub_doc(sio.iplot_bz,'bz_data :')
    @_sig_kwargs(sio.iplot_bz,('bz_data',))
    def iplot_bz(self, **kwargs):
        return sio.iplot_bz(bz_data = self._bz, **kwargs)

    @_sig_kwargs(sio.iplot_bz,('bz_data','special_kpoints'))
    def iplot_cell(self, **kwargs):
        "See docs of `iplot_bz`, everything is same except space is iverted."
        return sio.iplot_bz(bz_data = self._cell, special_kpoints = False, **kwargs)

    @_sub_doc(sio.splot_lattice,'poscar_data :')
    @_sig_kwargs(sio.splot_lattice, ('poscar_data','plane'))
    def splot_lattice(self, plane = None, **kwargs):
        return sio.splot_lattice(self._data, plane = plane, **kwargs)
    
    @_sub_doc(sio.iplot_lattice,'poscar_data :')
    @_sig_kwargs(sio.iplot_lattice, ('poscar_data',))
    def iplot_lattice(self, **kwargs):
        return sio.iplot_lattice(self._data, **kwargs)

    @_sub_doc(sio.write_poscar,'poscar_data :')
    @_sig_kwargs(sio.write_poscar, ('poscar_data',))
    def write(self, outfile = None, **kwargs):
        return sio.write_poscar(self._data, outfile = outfile, **kwargs)

    @_sub_doc(sio.join_poscars,'poscar_data :')
    @_sig_kwargs(sio.join_poscars, ('poscar_data','other'))
    def join(self,other, direction = 'c', **kwargs):
        return self.__class__(data = sio.join_poscars(poscar_data = self._data, other = other.data, direction = direction,**kwargs))

    @_sub_doc(sio.scale_poscar,'- poscar_data')
    @_sig_kwargs(sio.scale_poscar, ('poscar_data',))
    def scale(self, scale = (1,1,1), **kwargs):
        return self.__class__(data = sio.scale_poscar(self._data, scale, **kwargs))

    @_sub_doc(sio.rotate_poscar,'- poscar_data')
    def rotate(self,angle_deg, axis_vec):
        return self.__class__(data = sio.rotate_poscar(self._data, angle_deg = angle_deg, axis_vec = axis_vec)) 

    @_sub_doc(sio.set_zdir,'- poscar_data')
    @_sig_kwargs(sio.set_zdir, ('poscar_data','hkl'))
    def set_zdir(self, hkl, **kwargs):
        return self.__class__(data = sio.set_zdir(self._data, hkl, **kwargs))
    
    @_sub_doc(sio.translate_poscar,'- poscar_data')
    def translate(self, offset):
        return self.__class__(data = sio.translate_poscar(self._data, offset=offset))

    @_sub_doc(sio.repeat_poscar,'- poscar_data')
    def repeat(self, n, direction):
        return self.__class__(data = sio.repeat_poscar(self._data, n=n, direction=direction))

    @_sub_doc(sio.mirror_poscar,'- poscar_data')
    def mirror(self, direction):
        return self.__class__(data = sio.mirror_poscar(self._data, direction=direction))

    @_sub_doc(sio.get_TM,'- poscar_data')
    def get_TM(self, target_basis):
        return sio.get_TM(self._data.basis, target_basis)
    @_sub_doc(sio.transform_poscar)
    def transform(self, transformation, zoom = 2, tol = 1e-2):
        return self.__class__(data = sio.transform_poscar(self._data, transformation, zoom = zoom, tol=tol))

    @_sub_doc(sio.transpose_poscar,'- poscar_data')
    def transpose(self, axes = [1,0,2]):
        return self.__class__(data = sio.transpose_poscar(self._data, axes=axes))

    @_sub_doc(sio.add_vaccum,'- poscar_data')
    def add_vaccum(self, thickness, direction, left = False):
        return self.__class__(data = sio.add_vaccum(self._data, thickness=thickness, direction=direction, left=left))

    @_sub_doc(sio.add_atoms,'- poscar_data')
    def add_atoms(self,name, positions):
        return self.__class__(data = sio.add_atoms(self._data, name=name, positions=positions))

    @_sub_doc(sio.convert_poscar,'- poscar_data')
    def convert(self, atoms_mapping, basis_factor):
        return self.__class__(data = sio.convert_poscar(self._data, atoms_mapping = atoms_mapping, basis_factor=basis_factor))

    @_sub_doc(sio.strain_poscar, '- poscar_data')
    def strain(self, strain_matrix):
        return self.__class__(data = sio.strain_poscar(self._data, strain_matrix = strain_matrix))
    
    @_sub_doc(sio.get_kmesh,'- poscar_data')
    @_sig_kwargs(sio.get_kmesh, ('poscar_data',))
    def get_kmesh(self, *args, **kwargs):
        return sio.get_kmesh(self.data, *args, **kwargs)

    @_sub_doc(sio.get_kpath,'- rec_basis')
    @_sig_kwargs(sio.get_kpath, ('rec_basis',))
    def get_kpath(self,kpoints, n = 5, **kwargs):
        return sio.get_kpath(kpoints, n = n, **kwargs, rec_basis = self.data.rec_basis)


    def bring_in_cell(self,points):
        """Brings atoms's positions inside Cell and returns their R3 coordinates.
        """
        # Cartesain POSCAR is also loaded as relative to basis in memeory, so both same
        return self.to_R3(points, reciprocal = False)

    def bring_in_bz(self,kpoints, shift = 0):
        """Brings kpoints inside already set BZ, (primitive or regular).
        If basis is not None, returns kpoints relative to those basis.
        `shift` is a number or a list of three numbers that will be added to kpoints before any other operation.
        """
        if not self._bz:
            raise RuntimeError('No BZ found. Please run `get_bz()` first.')
        return sio.kpoints2bz(self._bz, kpoints = kpoints,primitive = self._primitive, shift = shift)
    
    def to_R3(self, points, reciprocal = False):
        "Converts points relative to basis/rec_basis of POSCAR to R3 coordinates. If reciprocal is True, converts to R3 in reciprocal basis."
        points = np.array(points) # In case list of lists
        if reciprocal:
            return sio.to_R3(self.data.rec_basis, points)
        return sio.to_R3(self.data.basis, points)
    
    def to_basis(self, coords, reciprocal = False):
        "Converts coords to fractional points in basis/rec_basis of POSCAR. If reciprocal is True, converts to fractional in reciprocal basis."
        coords = np.array(coords) # In case list of lists
        if reciprocal:
            return sio.to_basis(self.data.rec_basis, coords)
        return sio.to_basis(self.data.basis, coords)
    
    def map_kpoints(self, other, kpoints):
        """Map kpoints (fractional) from this to other POSCAR's Brillouin zone. This is equivalent to:
        other.to_basis(self.to_R3(kpoints, reciprocal = True), reciprocal = True)
        
        This operation is useful when you do POSCAR.transform() and want to map kpoints from original POSCAR to transformed POSCAR.
        """
        if not isinstance(other, self.__class__):
            raise TypeError('other must be a POSCAR object.')
        return other.to_basis(self.to_R3(kpoints, reciprocal = True), reciprocal = True)
        


class LOCPOT:
    """
    - Returns Data from LOCPOT and similar structure files. Loads only single set out of 2/4 magnetization data to avoid performance/memory cost while can load electrostatic and one set of magnetization together.
    Args:
        - path: path/to/LOCPOT. LOCPOT is auto picked in CWD.
        - data_set: 0 for electrostatic data, 1 for magnetization data if ISPIN = 2. If non-colinear calculations, 1,2,3 will pick Mx,My,Mz data sets respectively. Only one data set is loaded, so you should know what you are loading.
    - **Exceptions**
        - Would raise index error if magnetization density set is not present in LOCPOT/CHG in case `m` is not False.

    **Note:** To avoid memory issues while loading multiple LOCPOT files, use this class as a context manager which cleans up the memory after use.
    ```python
    with LOCPOT('path/to/LOCPOT') as tmp:
        tmp.splot()
    # The object tmp is destroyed here and memory is freed.
    ```
    """
    def __init__(self,path = None,data_set = 0):
        self._path = path # Must be
        self._data = vp.export_locpot(path = path, data_set = data_set)

        self.rolling_mean = gu.rolling_mean # For quick access to rolling mean function.

    def __enter__(self):
        import weakref
        return weakref.proxy(self)

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    @property
    def poscar(self):
        "POSCAR class object"
        return POSCAR(data = self._data.poscar)

    @property
    def data(self):
        return self._data

    @_sub_doc(sp.plot_potential,'- values')
    @_sig_kwargs(sp.plot_potential,('values',))
    def splot(self, operation = 'mean_c', **kwargs):
        return sp.plot_potential(basis = self._data.poscar.basis,values = self._data.values, operation = operation, **kwargs)

    def view_period(self, operation: str = 'mean_c',interface = 0.5,lr_pos = (0.25,0.75), smoothness = 2, figsize = (5,3),**kwargs):
        """Check periodicity using ipywidgets interactive plot.
        - operation: What to do, such as 'mean_c' or 'mean_a' etc.
        - interface: Interface in range [0,1] to divide left and right halves.
        - lr_pos: Tuple of (left,right) positions in range [0,1] to get ΔV of right relative to left.
        - smoothness: int. Default is 2. Smoothing parameter for rolling mean. Larger is better.
        - figsize: Tuple of (width,height) of figure. Since each time a figure is created, we can't reuse it, so we need to specify the size.
        kwargs are passed to the plt.Axes.set(kwargs) method to handle the plot styling.
        You can use return value to retrieve information, like output.f(*output.args, **output.kwargs) in a cell to plot the current state and save it.
        """
        check = ['mean_a','min_a','max_a','mean_b','min_b','max_b','mean_c','min_c','max_c']
        if operation not in check:
            raise ValueError("operation expects any of {!r}, got {}".format(check,operation))

        opr, _dir = operation.split('_')
        x_ind = 'abc'.index(_dir)
        other_inds = tuple([i for i in [0,1,2] if i != x_ind])
        _func_ = getattr(np,opr)
        X_1 = _func_(self._data.values,axis = other_inds)

        _step = round(1/X_1.size,4)
        _min = round(4*_step,4) # At least 4 steps per period

        import ipywidgets as ipw
        import matplotlib.pyplot as plt

        def checker(period, period_right):
            fig, ax = plt.subplots(1,1,figsize=figsize)
            ax.plot(X_1,label=operation,lw=1)
            X_2 = self.rolling_mean(X_1,period,period_right=period_right,interface=interface, smoothness=smoothness)
            ax.plot(X_2,label='rolling_mean',ls='dashed',lw=1)

            x = [int(X_2.size*p) for p in lr_pos]
            y = X_2[x]
            ax.step(x,y,where = 'mid',marker='.',lw=0.7)
            ax.text(0,y.mean(),f'$V_{{R}} - V_{{L}}$ : {y[1]-y[0]:.6f}',backgroundcolor=[1,1,1,0.5])
            plt.legend(bbox_to_anchor=(0, 1),loc='lower left',ncol=2,frameon=False)
            ax.set(**kwargs)
            return ax

        return ipw.interactive(checker,
                period = ipw.FloatSlider(min=_min,max=0.5,value=0.125,step=_step,readout_format='.4f', continuous_update=False),
                period_right=ipw.FloatSlider(min=_min,max=0.5,value=0.125,step=_step,readout_format='.4f', continuous_update=False),
                )
        
    def view_slice(self,*argse, **kwargs):
        # Use interactive here to select the slice, digonal slices and so on..., tell user to get output results back
        raise NotImplementedError('Coming soon...')


class CHG(LOCPOT):
    __doc__ = LOCPOT.__doc__.replace('LOCPOT','CHG')
    def __init__(self, path = None,data_set = 0):
        super().__init__(path or 'CHG',data_set = data_set)

class ELFCAR(LOCPOT):
    __doc__ = LOCPOT.__doc__.replace('LOCPOT','ELFCAR')
    def __init__(self, path = None,data_set = 0):
        super().__init__(path or 'ELFCAR',data_set=data_set)


class PARCHG(LOCPOT):
    __doc__ = LOCPOT.__doc__.replace('LOCPOT','PARCHG')
    def __init__(self, path = None,data_set = 0):
        super().__init__(path or 'PARCHG',data_set = data_set)

class OUTCAR:
    "Parse some required data from OUTCAR file."
    def __init__(self,path = None):
        self._path = Path(path or 'OUTCAR')
        self._data = vp.export_outcar(self._path)

    @property
    def data(self):
        return self._data

    @property
    def path(self):
        return self._path
    
    @_sub_doc(vp.Vasprun.read)
    @_sig_kwargs(vp.Vasprun.read, ('self',))
    def read(self, start_match, stop_match, **kwargs):
        return vp.Vasprun.read(self, start_match, stop_match, **kwargs) # Pass all the arguments to the function

@_sub_doc(sp.get_axes)
@_sig_kwargs(sp.get_axes)
def get_axes(figsize = (3.4,2.6), **kwargs):
    axes = sp.get_axes(figsize, **kwargs)
    for ax in np.array([axes]).flatten():
        for f in [sp.add_text,sp.add_legend,sp.add_colorbar,sp.color_wheel,sp.break_spines,sp.modify_axes,sp.append_axes, sp.join_axes]:
            if ax.name != '3d':
                setattr(ax,f.__name__,f.__get__(ax,type(ax)))
    return axes
get_axes.__doc__ = get_axes.__doc__ + '''
**There are extra methods added to each axes (only 2D) object.**
- add_text
- add_legend
- add_colorbar
- color_wheel
- break_spines
- modify_axes
- append_axes
- join_axes
'''

def _format_input(projections, sys_info):
    """
    Format input spins, atoms, orbs and labels according to selected `projections`.
    For example: {'Ga-s':(0,[1]),'Ga-p':(0,[1,2,3]),'Ga-d':(0,[4,5,6,7,8])} #for Ga in GaAs, to pick Ga-1, use [0] instead of 0 at first place
    In case of 3 items in tuple, the first item is spin index, the second is atoms, the third is orbs.
    """
    if not isinstance(projections,dict):
        raise TypeError("`projections` must be a dictionary, with keys as labels and values from picked projection indices.")
    
    if not hasattr(sys_info,'orbs'): 
        raise ValueError("No orbitals found to pick from given data.")
    
    types = list(sys_info.types.values())
    names = list(sys_info.types.keys())
    max_ind = np.max([t for tt in types for t in tt]) # will be error if two ranges there to compare for max
    norbs = len(sys_info.orbs)
    
    # Set default values for different situations
    spins, atoms, orbs, labels = [], [], [], []

    for i, (k, v) in enumerate(projections.items()):
        if len(v) not in [2,3]:
            raise ValueError(f"{k!r}: {v} expects 2 items (atoms, orbs), or 3 items (spin, atoms, orbs), got {len(v)}.")
        
        if not isinstance(k,str):
            raise TypeError(f"{k!r} is not a string. Use string as key for labels.")
        
        labels.append(k)
        
        if len(v) == 2:
            A, B = v # A is atom, B is orbs only two cases: (1) int (2) list of int
        else:
            S, A, B = v # 
            
            if not isinstance(S,(int,np.integer)):
                raise TypeError(f"First itme in a squence of size 3 should be integer to pick spin.")
            
            spins.append(S) # Only add spins if given
        
        if not isinstance(A,(int,np.integer,list,tuple, range)):
            raise TypeError(f"{A!r} is not an integer or list/tuple/range of integers.")
        
        if not isinstance(B,(int,np.integer,list,tuple,range)):
            raise TypeError(f"{B!r} is not an integer or list/tuple/range of integers.")
        
        # Fix orbs
        B = [B] if isinstance(B,(int,np.integer)) else B
        
        if np.max(B) >= norbs:
            raise IndexError("index {} is out of bound for {} orbs".format(np.max(B),norbs))
        if np.min(B) < 0:
            raise IndexError("Only positive integers are allowed for selection of orbitals.")
        
        orbs.append(np.unique(B).tolist())
        
        # Fix atoms
        if isinstance(A, (int,np.integer)):
            if A < 0:
                raise IndexError("Only positive integers are allowed for selection of atoms.")
            
            if A < len(types):
                atoms.append(types[A])
                info = f"Given {A} at position {i+1} of sequence => {names[A]!r}: {atoms[i]}. "
                print(gu.color.g(info + f"To just pick one ion, write it as [{A}]."))
            else:
                raise IndexError(f"index {A}  at is out of bound for {len(types)} types of ions. Wrap {A} in [] to pick single ion if that was what you meant.")
        else:
            if np.max(A) > max_ind:
                raise IndexError(f"index {np.max(A)} is out of bound for {max_ind+1} ions")
            
            if np.min(A) < 0:
                raise IndexError("Only positive integers are allowed for selection of atoms.")
            
            atoms.append(np.unique(A).tolist())
    
    if spins and len(atoms) != len(spins):
        raise ValueError("You should provide spin for each projection or none at all. If not provided, spin is picked from corresponding eigenvalues (up/down) for all projections.")
    
    uatoms = np.unique([a for aa in atoms for a in aa]) # don't use set, need asceding order
    uorbs = np.unique([o for oo in orbs for o in oo])
    uorbs = tuple(uorbs) if len(uorbs) < norbs else -1 # -1 means all orbitals
    uatoms = tuple(uatoms) if len(uatoms) == (max_ind + 1) else -1 # -1 means all atoms
    uspins = tuple(spins)

    return (spins, uspins), (atoms,uatoms), (orbs,uorbs), labels

_spin_doc = 'spin : int, 0 by default. Use 0 for spin up and 1 for spin down for spin polarized calculations. Data for both channel is loaded by default, so when you plot one spin channel, plotting other with same parameters will use the same data.'
_kind_doc = 'kpairs : list/tuple of pair of indices to rearrange a computed path. For example, you computed 0:L, 15:G, 25:X, 34:M path and want to plot it as X-G|M-X, use [(25,15), (34,25)] as kpairs.'
_proj_doc = "projections : dict, str -> [atoms, orbs]. Use dict to select specific projections, e.g. {'Ga-s': (0,[0]), 'Ga1-p': ([0],[1,2,3])} in case of GaAs. If values of the dict are callable, they must accept two arguments evals/tdos, occs/idos of from data and should return array of shape[1:] (all but spin dimension)."

class _BandsDosBase:
    def __init__(self, source):
        if not isinstance(source, vp.DataSource):
            raise TypeError('`source` must be a subclass of `ipyvasp.parser.DataSource`.')
        self._source = source # source is instance of DataSource
        self._data = None # will be updated on demand
    
    @property
    def source(self):
        return self._source
    
    @property
    def data(self):
        "Returns a dictionary of information about the picked data after a plotting function called."
        return self._data
    
    def _fix_projections(self, projections):
        labels, spins, atoms, orbs, uspins, uatoms, uorbs =  [], None, None, None, None, None, None
        
        funcs = []
        if isinstance(projections, dict):
            if not projections:
                raise ValueError('`projections` dictionary should have at least one item.')
            
            _funcs = [callable(value) for _, value in projections.items()]
            if any(_funcs) and not all(_funcs): # Do not allow mixing of callable and non-callable values, as comparison will not make sense
                raise TypeError('Either all or none of the values of `projections` must be callable with two arguments evals, occs and return array of same shape as evals.')
            elif all(_funcs):
                funcs = [value for _, value in projections.items()]
                labels = list(projections.keys())
            else:
                (spins, uspins), (atoms,uatoms), (orbs,uorbs), labels = _format_input(projections, self.source.summary)           
        elif projections is not None:
            raise TypeError('`projections` must be a dictionary or None.')
        
        return (spins, uspins), (atoms,uatoms), (orbs,uorbs), (funcs, labels)


class Bands(_BandsDosBase):
    """
    Class to handle and plot bandstructure data.
    
    Parameters
    ----------
    source : instance of `ipyvasp.DataSource` such as `ipyvasp.Vasprun` or `ipyvasp.Vaspout`. You can define your own class to parse data with same attributes and methods by subclassing `ipyvasp.DataSource`.
    """
    def __init__(self, source):
        super().__init__(source)
        self._data_args = () # will be updated on demand
        
    def get_kticks(self, rel_path = 'KPOINTS'):
        """
        Reads associated KPOINTS file form a relative path of calculations and returns kticks. If KPOINTS file does not exist or was not created by this module, returns empty dict.
        
        .. note:: 
            kticks become useless when you interploate data in plotting, in that case write kticks manually.
        """
        path = Path(self.source.path).parent/rel_path
        if path.is_file():
            return sio.read_kticks(path)
        return []
    
    def get_plot_coords(self, kindices, eindices):
        """Returns coordinates of shape (len(zip(kindices, eindices)), 2) from most recent bandstructure plot. 
        Use in a plot command as `plt.plot(*get_plot_coords(kindices, eindices).T)`.
        Enegy values are shifted by `ezero` from a plot command or data. Use coords[:,1] + ezero to get data values.
        """
        for inds in (kindices, eindices):
            if not isinstance(inds, (list, tuple, range, np.ndarray)):
                raise TypeError('`kindices` and `eindices` must be list, tuple, range or numpy array.')
            
        if not hasattr(self,'_breaks'): # There will be data when a plotting function is called, and set _breaks attribute
            raise ValueError('You must call a plotting function first to get band gap coordinates from plot.')
        
        kpath = self.data.kpath
        if self._breaks:
            for i in self._breaks:
                kpath[i:] -= (kpath[i] - kpath[i-1]) # remove distance
        
            kpath = kpath/np.max(kpath) # normalize to in this case again
        
        kvs = [kpath[k] for k,e in zip(kindices, eindices)] # need same size as eindices
        evs = [self.data.evals[self._spin][k,e] - self.data.ezero for k,e in zip(kindices, eindices)]
        return np.array([kvs, evs]).T # shape (len(kindices), 2)
        
    @property
    def gap(self): # no need for get_here, useful as .gap.coords immediately after a plot
        """Retruns band gap data with following attributes:
        
        coords : array of shape (2, 2) -> (K,E) band gap in coordinates of most recent plot if exists, otherwise in data coordinates. `X,Y = coords.T` for plotting purpose.
        vbm, cbm : valence and conduction band energies in data coordinates. No shift is applied.
        kvbm, kcbm : kpoints of vbm and cbm in fractional coordinates. Useful to know which kpoint is at the band gap.
        
        These attributes will be None if band gap cannot be found. coords will be empty array of size (0,2) in that case.
        """
        if not self.data:
            self.get_data() # This assigns back to self._data
        
        if not hasattr(self,'_breaks'): # even if no plot
            vs = np.array([self.data.kpath[self.data.kvc,], self.data.evc]).T # shape (K,E) -> (2,2)
        else:
            vs = self.get_plot_coords(self.data.kvc, [0,0])
        
        out = {'coords': vs, 'vbm': None, 'cbm': None, 'gap': None, 'kvbm': None, 'kcbm': None} # will be updated below if vs.size > 0
        if vs.size: # May not exist, but still same shape in plotting with empty arrays
            if hasattr(self,'_breaks'): # only shift by ezero if plot exists
                vs[:,1] = [v - self.data.ezero for v in self.data.evc] 
            vbm, cbm = self.data.evc
            kvbm, kcbm = self.data.kpoints[self.data.kvc,]
            out.update({'coords': vs, 'vbm': vbm, 'cbm': cbm, 'gap': cbm - vbm, 'kvbm': tuple(kvbm), 'kcbm': tuple(kcbm)})
        
        return serializer.Dict2Data(out)
                
    def get_data(self, elim = None, ezero = None, projections: dict = None, kpairs = None):
        """
        Selects bands and projections to use in plotting functions. If input arguments are same as previous call, returns cached data.
        
        Parameters
        ----------
        elim : list, tuple of two floats to pick bands in this energy range. If None, picks all bands.
        ezero : float, None by default. If not None, elim is applied around this energy.
        projections : dict, str -> [atoms, orbs]. Use dict to select specific projections, e.g. {'Ga-s': (0,[0]), 'Ga1-p': ([0],[1,2,3])} in case of GaAs. If values of the dict are callable, they must accept two arguments evals, occs of shape (spin,kpoints, bands) and return array of shape (kpoints, bands).
        kpairs : list, tuple of integers, None by default to select all kpoints in given order. Use this to select specific kpoints intervals in specific order.
        
        Returns
        -------
        data : Selected bands and projections data to be used in bandstructure plotting functions under this class as `data` argument.
        """
        if self.data and self._data_args == (elim, ezero, projections, kpairs):
            return self.data
        
        if kpairs and not isinstance(kpairs, (list, tuple)):
            raise TypeError('`kpairs` must be a list/tuple of pair of indices of edge kpoints of intervals.')
        
        kinds = [] # plain indices of kpoints to select
        if kpairs:
            if np.ndim(kpairs) != 2:
                raise ValueError('`kpairs` must be a list/tuple of pairs indices to select intervals.')
            
            for inds in kpairs:
                if len(inds) != 2:
                    raise ValueError('`kpairs` must be a list/tuple of pairs indices to select intervals.')
            
            all_inds = [range(k1, k2, -1 if k2 < k1 else 1) for k1, k2 in kpairs]
            kinds = [*[ind for inds in all_inds for ind in inds], kpairs[-1][-1]] # flatten and add last index
        
        self._data_args = (elim, ezero, projections)

        (spins, uspins), (atoms,uatoms), (orbs,uorbs), (funcs, labels) = self._fix_projections(projections)
            
        kpts = self.source.get_kpoints()
        data = self.source.get_evals(elim = elim, ezero = ezero, atoms = uatoms, orbs = uorbs, spins = uspins or None, bands = None) # picks available spins if uspins is None
        
        if not spins:
            spins = data.spins # because they will be loaded anyway
            if len(spins) == 1 and labels: # in case projections not given, check label
                spins = [spins[0] for _ in labels] # only one spin channel is available, so use it for all projections
        
        output = {'kpath': kpts.kpath, 'kpoints': kpts.kpoints, 'coords': kpts.coords, **data.to_dict()}
        
        if kinds:
            coords = output['coords'][kinds]
            kpath = np.cumsum([0, *np.linalg.norm(coords[1:] - coords[:-1],axis = 1)])
            output['kpath'] = kpath/np.max(kpath) # normalize to 1
            output['kpoints'] = output['kpoints'][kinds]
            output['coords'] = coords
            output['evals'] = output['evals'][:,kinds,:]
            output['occs'] = output['occs'][:,kinds,:]
            if 'pros' in output:
                output['pros'] = output['pros'][...,kinds,:]
            
            # Have to see kvc, but it makes no sense when there is some break in kpath
            vbm, cbm = output['evc'] # evc is a tuple of two tuples
            kvbm = [k for k in np.where(output['evals'] == vbm)[1]] # keep as indices here, we don't know the cartesian coordinates of kpoints here
            kcbm = [k for k in np.where(output['evals'] == cbm)[1]]
            kvc = tuple(sorted(product(kvbm,kcbm),key = lambda K: np.ptp(K))) # bring closer points first by sorting
            output['kvc']  = kvc[0] if kvc else (0,0) # Only relative minimum indices matter, set to (0,0) if no kvc found
        
        output['labels'] = labels # works for both functions and picks
        
        data = serializer.Dict2Data(output) # replacing data with updated one
        if funcs:
            pros = []
            for func in funcs:
                out = func(data.evals, data.occs)
                if np.shape(out) != data.evals.shape[1:]: # evals shape is (spin, kpoints, bands), but we need single spin
                    raise ValueError(f'Projections returned by {func} must be of same shape as last two dimensions of input evals.')
                pros.append(out)
                
            output['pros'] = np.array(pros)
            output['info'] = '"Custom projections by user"'
        
        elif hasattr(data, 'pros'): # Data still could be there, but prefer if user provides projections as functions
            arrays = []
            for sp,atom,orb in zip(spins, atoms, orbs):
                if uatoms != -1:
                    atom = [i for i, a in enumerate(data.atoms) if a in atom] # indices for partial data loaded
                if uorbs != -1:
                    orb = [i for i, o in enumerate(data.orbs) if o in orb]
                sp = list(data.spins).index(sp) # index for spin is single
                _pros  = np.take(data.pros[sp],atom,axis = 0).sum(axis = 0) # take dimension of spin and then sum over atoms leaves 3D array
                _pros = np.take(_pros,orb,axis = 0).sum(axis = 0) # Sum over orbitals leaves 2D array
                arrays.append(_pros)

            output['pros'] = np.array(arrays)
            output.pop('atoms', None) # No more needed
            output.pop('orbs', None)
            output.pop('spins', None) # No more needed
        
        output['shape'] = '(spin[evals,occs]/selection[pros], kpoints, bands)'
        
        self._data = serializer.Dict2Data(output) # Assign for later use
        return self._data
    
    def _handle_kwargs(self, **kwargs):
        "Returns fixed kwargs and new elim relative to fermi energy for gettig data."
        if kwargs.get('spin',None) not in [0,1]:
            raise ValueError('spin must be 0 or 1')
        
        self._spin = kwargs.pop('spin',None) # remove from kwargs as plots don't need it, but get_plot_coords need it
        kpairs = kwargs.pop('kpairs',None) # not needed for plots but need here
        
        if kpairs is None:
            kwargs['kticks'] = kwargs.get('kticks', None) or self.get_kticks() # User can provide kticks, but if not, use default
        
        # Need to fetch data for gap and plot later
        self._breaks = [tick[0] for tick in (kwargs['kticks'] or []) if tick[1].lstrip().startswith('<=')]   
        return kwargs, kwargs.get('elim',None) # need in plots
    
    @_sub_doc(sp.splot_bands,['K :','E :'],replace = {'ax :': f"{_spin_doc}\n{_kind_doc}\nax :"})
    @_sig_kwargs(sp.splot_bands, ('K','E'))
    def splot_bands(self, spin = 0, kpairs = None, ezero = None, **kwargs):
        kwargs, elim = self._handle_kwargs(spin = spin, **kwargs)
        data = self.get_data(elim = elim, ezero = ezero, kpairs = kpairs)
        return sp.splot_bands(data.kpath, data.evals[spin] - data.ezero, **kwargs)
    
    @_sub_doc(sp.splot_rgb_lines,['K :','E :', 'pros :', 'labels :'], replace = {'ax :': f"{_proj_doc}\n{_spin_doc}\n{_kind_doc}\nax :"})
    @_sig_kwargs(sp.splot_rgb_lines, ('K','E','pros','labels'))
    def splot_rgb_lines(self, projections, spin = 0, kpairs = None,ezero = None,**kwargs):
        kwargs, elim = self._handle_kwargs(spin = spin, **kwargs)
        data = self.get_data(elim, ezero, projections, kpairs = kpairs) 
        return sp.splot_rgb_lines(data.kpath, data.evals[spin] - data.ezero, data.pros, data.labels, **kwargs)
    
    @_sub_doc(sp.splot_color_lines,['K :','E :', 'pros :', 'labels :'], replace = {'ax :': f"{_proj_doc}\n{_spin_doc}\n{_kind_doc}\nax :"})
    @_sig_kwargs(sp.splot_color_lines, ('K','E','pros','labels'))
    def splot_color_lines(self, projections, spin = 0,  kpairs = None, ezero = None, **kwargs ):
        kwargs, elim = self._handle_kwargs(spin = spin, **kwargs)
        data = self.get_data(elim, ezero, projections, kpairs = kpairs) # picked relative limit
        return sp.splot_color_lines(data.kpath, data.evals[spin] - data.ezero, data.pros, data.labels, **kwargs)
    
    @_sub_doc(ip.iplot_rgb_lines,['K :','E :', 'pros :', 'labels :','occs :','kpoints :'], replace = {'fig :': f"{_proj_doc}\n{_spin_doc}\n{_kind_doc}\nfig :"})
    @_sig_kwargs(ip.iplot_rgb_lines, ('K','E', 'pros', 'labels', 'occs', 'kpoints'))
    def iplot_rgb_lines(self, projections, spin = 0, kpairs = None, ezero = None, **kwargs):
        kwargs, elim = self._handle_kwargs(spin = spin, **kwargs)
        data = self.get_data(elim, ezero, projections, kpairs = kpairs)
        # Send K and bands in place of K for use in iplot_rgb_lines to depict correct band number 
        return ip.iplot_rgb_lines({"K": data.kpath, 'indices':data.bands}, data.evals[spin] - data.ezero, data.pros, data.labels, data.occs[spin], data.kpoints, **kwargs)
    
    @_sub_doc(ip.iplot_bands,['K :','E :'], replace = {'fig :': f"{_proj_doc}\n{_spin_doc}\n{_kind_doc}\nfig :"})
    @_sig_kwargs(ip.iplot_bands, ('K','E'))
    def iplot_bands(self,spin   = 0,kpairs = None,ezero  = None,**kwargs):
        kwargs, elim = self._handle_kwargs(spin = spin, **kwargs)
        data = self.get_data(elim, ezero, kpairs = kpairs)
        # Send K and bands in place of K for use in iplot_rgb_lines to depict correct band number
        return ip.iplot_bands({"K": data.kpath, 'indices':data.bands}, data.evals[spin] - data.ezero, **kwargs) 
    
    def view_bands(self):
        "Initialize and return `ipyvasp.widgets.BandsWidget` to view bandstructure interactively."
        use_vaspout = True if isinstance(self.source, vp.Vaspout) else False
        return wdg.BandsWidget(use_vaspout = use_vaspout, path = self.source.path.parent, glob = self.source.path.name)
    
_multiply_doc = "multiply : float, multiplied by total dos and sign is multiply by partial dos to flip plot in case of spin down."
_total_doc = "total : bool, True by default. If False, total dos is not plotted, sign of multiply parameter is still used for partial dos"
class DOS(_BandsDosBase):
    """
    Class to handle and plot density of states data.
    
    Parameters
    ----------
    source : instance of `ipyvasp.DataSource` such as `ipyvasp.Vasprun` or `ipyvasp.Vaspout`. You can define your own class to parse data with same attributes and methods by subclassing `ipyvasp.DataSource`.
    """
    def __init__(self, source):
        super().__init__(source)
        self._data_args =  () # updated on demand
    
    def get_data(self, elim = None, ezero = None, projections: dict = None):
        if self.data and self._data_args == (elim, ezero, projections):
            return self.data
        
        self._data_args = (elim, ezero, projections)
        
        (spins, uspins), (atoms,uatoms), (orbs,uorbs), (funcs, labels) = self._fix_projections(projections)
        dos = self.source.get_dos(elim = elim, ezero = ezero, atoms = uatoms, orbs = uorbs, spins = uspins or None)
        
        if not spins:
            spins = dos.spins # because they will be loaded anyway
            if len(spins) == 1 and labels: # in case projections not given, check label
                spins = [spins[0] for _ in labels] # only one spin channel is available, so use it for all projections
        
        out = dos.to_dict()
        out['labels'] = labels
        
        if funcs:
            pdos = []
            for func in funcs:
                p = func(dos.tdos, dos.idos)
                if np.shape(p) != dos.energy.shape[1:]: # energy shape is (spin, grid), but we need single spin
                    raise ValueError(f'Projections returned by {func} must be of same shape as last dimension of input energy.')
                pdos.append(p)
                
            out['pdos'] = np.array(pdos)
            out['info'] = '"Custom projections by user"'
        
        elif hasattr(dos, 'pdos'): # Data still could be there, but prefer if user provides projections as functions
            arrays = []
            for sp,atom,orb in zip(spins, atoms, orbs):
                if uatoms != -1:
                    atom = [i for i, a in enumerate(dos.atoms) if a in atom] # indices for partial data loaded
                if uorbs != -1:
                    orb = [i for i, o in enumerate(dos.orbs) if o in orb]
                sp = list(dos.spins).index(sp) # index for spin is single
                _pdos  = np.take(dos.pdos[sp],atom,axis = 0).sum(axis = 0) # take dimension of spin and then sum over atoms leaves 2D array
                _pdos = np.take(_pdos,orb,axis = 0).sum(axis = 0) # Sum over orbitals leaves 1D array
                arrays.append(_pdos)

            out['pdos'] = np.array(arrays)
            out.pop('atoms', None) # No more needed
            out.pop('orbs', None)
            out.pop('spins', None) # No more needed
        
        out['shape'] = '(spin[energy,tdos,idos]/selection[pdos], NEDOS)'
        
        self._data = serializer.Dict2Data(out) # Assign for later use
        return self._data
    
    def _handle_kwargs(self, **kwargs):
        "Returns fixed kwargs and new elim relative to fermi energy for gettig data."
        if kwargs.get('spin',None) not in [0,1,2,3]:
            raise ValueError('spin must be 0,1,2,3 for dos')
        
        kwargs.pop('spin',None) # remove from kwargs as plots don't need it
        return kwargs, kwargs.get('elim', None)
        
    @_sub_doc(sp.splot_dos_lines,['energy :', 'dos_arrays :', 'labels :'], replace = {'ax :': f"{_proj_doc}\n{_spin_doc}\n{_multiply_doc}\n{_total_doc}\nax :"})
    @_sig_kwargs(sp.splot_dos_lines, ('energy', 'dos_arrays', 'labels'))
    def splot_dos_lines(self, projections = None, # dos should allow only total dos as well
        spin = 0,multiply = 1,total = True,ezero = None, **kwargs
        ):
        kwargs, elim = self._handle_kwargs(spin = spin, **kwargs)
        data = self.get_data(elim, ezero, projections)
        energy, labels = data['energy'][spin], data['labels']
        
        dos_arrays = []
        if projections is not None:
            dos_arrays = np.sign(multiply)*data['pdos'] # filp if asked, 
        
        tlab = kwargs.pop('label', None) # pop in any case
        if total:
            dos_arrays = [data['tdos'][spin]*multiply, *dos_arrays]
            labels = [tlab or 'Total', *labels]
        elif len(dos_arrays) == 0:
            raise ValueError("Either total should be True or projections given!")
        
        return sp.splot_dos_lines(energy - data.ezero, dos_arrays, labels, **kwargs)
    
    @_sub_doc(ip.iplot_dos_lines,['energy :', 'dos_arrays :', 'labels :'], replace = {'fig :': f"{_proj_doc}\n{_spin_doc}\n{_multiply_doc}\n{_total_doc}\nfig :"})
    @_sig_kwargs(ip.iplot_dos_lines, ('energy', 'dos_arrays', 'labels'))
    def iplot_dos_lines(self, projections = None, # dos should allow only total dos as well
        spin = 0,multiply = 1,total = True, ezero = None, **kwargs
        ):
        kwargs, elim = self._handle_kwargs(spin = spin, **kwargs)
        data = self.get_data(elim, ezero, projections)
        energy, labels = data['energy'][spin], data['labels']
        
        dos_arrays = []
        if projections is not None:
            dos_arrays = np.sign(multiply)*data['pdos'] # filp if asked
            
        tname = kwargs.pop('name', None) # pop in any case
        if total:
            dos_arrays = [data['tdos'][spin]*multiply, *dos_arrays]
            labels = [tname or 'Total', *labels]
        elif len(dos_arrays) == 0:
            raise ValueError("Either total should be True or projections given!")
        
        return ip.iplot_dos_lines(energy - data.ezero, dos_arrays, labels, **plot_kws, **kwargs)
            
          