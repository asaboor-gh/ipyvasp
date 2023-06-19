
__all__ = ['atomic_number', 'atoms_color', 'periodic_table', 'Arrow3D', 'quiver3d',
           'write_poscar', 'export_poscar', 'InvokeMaterialsProject', 'get_kpath', 'read_kticks',
           'get_kmesh', 'order', 'rotation', 'get_bz', 'splot_bz', 'iplot_bz', 'to_R3', 'to_basis', 'kpoints2bz',
           'fix_sites', 'translate_poscar', 'get_pairs', 'iplot_lattice', 'splot_lattice', 'join_poscars', 'repeat_poscar',
           'scale_poscar', 'rotate_poscar', 'mirror_poscar', 'convert_poscar', 'get_TM',
           'transform_poscar', 'add_vaccum', 'transpose_poscar', 'add_atoms','strain_poscar', 'view_poscar']


import re
import json
import numpy as np
from pathlib import Path
import requests as req
import inspect
from collections import namedtuple
from itertools import product, combinations
from functools import lru_cache

from scipy.spatial import ConvexHull, Voronoi, KDTree
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import matplotlib.colors as mplc #For viewpoint
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from ipywidgets import interactive

# Inside packages import 
from . import parser as vp, serializer
from . import splots as sp


# These colors are taken from Mathematica's ColorData["Atoms"]
_atom_colors = {'H': (0.7, 0.8, 0.7), 'He': (0.8367, 1.0, 1.0),
    'Li': (0.7994, 0.9976, 0.5436), 'Be': (0.7706, 0.0442, 0.9643), 'B': (1.0, 0.5, 0), 'C': (0.4, 0.4, 0.4), 'N': (143/255,143/255,1), 'O': (0.8005, 0.1921, 0.2015), 'F': (128/255, 1, 0), 'Ne': (0.6773, 0.9553, 0.9284),
    'Na': (0.6587, 0.8428, 0.4922),'Mg': (0.6283, 0.0783, 0.8506),'Al': (173/255, 178/255, 189/255),'Si': (248/255, 209/255, 152/255),'P': (1,165/255,0),'S': (1,200/255,50/255),'Cl': (0,0.9,0),'Ar': (0.5461, 0.8921, 0.8442),
    'K':  (0.534, 0.7056, 0.4207), 'Ca': (0.4801, 0.0955, 0.7446), 'Sc': (0.902, 0.902, 0.902), 'Ti': (0.749, 0.7804, 0.7608), 'V': (0.651, 0.6706, 0.651), 'Cr': (0.5412, 0.7804, 0.6), 'Mn': (0.6118, 0.7804, 0.4784), 'Fe': (0.32,0.33,0.35),
    'Co': (0.9412, 0.6275, 0.5647), 'Ni': (141/255, 142/255, 140/255), 'Cu': (184/255, 115/255, 51/255), 'Zn': (186/255, 196/255, 200/255),'Ga': (90/255, 180/255, 189/255),'Ge': (0.6051, 0.5765, 0.6325),'As': (50/255,71/255,57/255),'Se': (0.9172, 0.0707, 0.6578),
    'Br': (161/255, 61/255, 45/255),'Kr': (0.426, 0.8104, 0.7475),'Rb': (0.4254, 0.5859, 0.3292),'Sr': (0.326, 0.096, 0.6464),'Y': (0.531, 1.0, 1.0),'Zr': (0.4586, 0.9186, 0.9175),'Nb': (0.385, 0.8417, 0.8349),'Mo': (0.3103, 0.7693, 0.7522),
    'Tc': (0.2345, 0.7015, 0.6694), 'Ru': (0.1575, 0.6382, 0.5865), 'Rh': (0.0793, 0.5795, 0.5036), 'Pd': (0.0, 0.5252, 0.4206), 'Ag': (0.7529, 0.7529, 0.7529), 'Cd': (0.8,0.67,0.73), 'In': (228/255, 228/255, 228/255), 'Sn': (0.398, 0.4956, 0.4915),
    'Sb': (158/255,99/255,181/255), 'Te': (0.8167, 0.0101, 0.4513), 'I': (48/255, 25/255, 52/255), 'Xe': (0.3169, 0.7103, 0.6381), 'Cs': (0.3328, 0.4837, 0.2177), 'Ba': (0.1659, 0.0797, 0.556), 'La': (0.9281, 0.3294, 0.7161), 'Ce': (0.8948, 0.3251, 0.7314),
    'Pr': (0.8652, 0.3153, 0.708), 'Nd': (0.8378, 0.3016, 0.663), 'Pm': (0.812, 0.2856, 0.6079), 'Sm': (0.7876, 0.2683, 0.5499), 'Eu': (0.7646, 0.2504, 0.4933), 'Gd': (0.7432, 0.2327, 0.4401), 'Tb': (0.7228, 0.2158, 0.3914), 'Dy': (0.7024, 0.2004, 0.3477),
    'Ho': (0.68, 0.1874, 0.3092), 'Er': (0.652, 0.1778, 0.2768), 'Tm': (0.6136, 0.173, 0.2515), 'Yb': (0.5579, 0.1749, 0.2346), 'Lu': (0.4757, 0.1856, 0.2276), 'Hf': (0.7815, 0.7166, 0.7174), 'Ta': (0.7344, 0.6835, 0.5445), 'W': (0.6812, 0.6368, 0.3604),
    'Re': (0.6052, 0.5563, 0.3676), 'Os': (0.5218, 0.4692, 0.3821), 'Ir': (0.4456, 0.3991, 0.3732), 'Pt': (0.8157, 0.8784, 0.8157), 'Au': (0.8, 0.7, 0.2), 'Hg': (0.7216, 0.8157, 0.7216), 'Tl': (0.651, 0.302, 0.3294), 'Pb': (0.3412, 0.3804, 0.349),
    'Bi': (10/255, 49/255, 93/255), 'Po': (0.6706, 0.0, 0.3608), 'At': (0.4588, 0.2706, 0.3098), 'Rn': (0.2188, 0.5916, 0.5161), 'Fr': (0.2563, 0.3989, 0.0861), 'Ra': (0.0, 0.0465, 0.4735), 'Ac': (0.322, 0.9885, 0.7169), 'Th': (0.3608, 0.943, 0.6717),
    'Pa': (0.3975, 0.8989, 0.628), 'U': (0.432, 0.856, 0.586), 'Np': (0.4645, 0.8145, 0.5455), 'Pu': (0.4949, 0.7744, 0.5067), 'Am': (0.5233, 0.7355, 0.4695), 'Cm': (0.5495, 0.698, 0.4338), 'Bk': (0.5736, 0.6618, 0.3998), 'Cf': (0.5957, 0.6269, 0.3675),
    'Es': (0.6156, 0.5934, 0.3367), 'Fm': (0.6335, 0.5612, 0.3075), 'Md': (0.6493, 0.5303, 0.2799), 'No': (0.663, 0.5007, 0.254), 'Lr': (0.6746, 0.4725, 0.2296), 'Rf': (0.6841, 0.4456, 0.2069), 'Db': (0.6915, 0.42, 0.1858), 'Sg': (0.6969, 0.3958, 0.1663),
    'Bh': (0.7001, 0.3728, 0.1484), 'Hs': (0.7013, 0.3512, 0.1321), 'Mt': (0.7004, 0.331, 0.1174), 'Ds': (0.6973, 0.312, 0.1043), 'Rg': (0.6922, 0.2944, 0.0928), 'Cn': (0.6851, 0.2781, 0.083), 'Nh': (0.6758, 0.2631, 0.0747), 'Fl': (0.6644, 0.2495, 0.0681),
    'Mc': (0.6509, 0.2372, 0.0631), 'Lv': (0.6354, 0.2262, 0.0597), 'Ts': (0.6354, 0.2262, 0.0566), 'Og': (0.6354, 0.2262, 0.0528)}

_atom_numbers = {k:i for i,k in enumerate(_atom_colors.keys())}

def atomic_number(atom):
    "Return atomic number of atom"
    return _atom_numbers[atom]

def atoms_color():
    "Defualt color per atom used for plotting the crystal lattice"
    return serializer.Dict2Data({k:[round(_v,4) for _v in rgb] for k,rgb in _atom_colors.items()})

def periodic_table():
    "Display colorerd elements in periodic table."
    _copy_names = np.array([f'$^{{{str(i+1)}}}${k}' for i,k in enumerate(_atom_colors.keys())])
    _copy_array = np.array(list(_atom_colors.values()))

    array = np.ones((180,3))
    names = ['' for i in range(180)] # keep as list before modification

    inds = [(0,0),(17,1),
            (18,2),(19,3),*[(30+i,4+i) for i in range(8)],
            *[(48+i,12+i) for i in range(6)],
            *[(54+i,18+i) for i in range(18)],
            *[(72+i,36+i) for i in range(18)],
            *[(90+i,54+i) for i in range(3)],*[(93+i,71+i) for i in range(15)],
            *[(108+i,86+i) for i in range(3)],*[(111+i,103+i) for i in range(15)],
            *[(147+i,57+i) for i in range(14)],
            *[(165+i,89+i) for i in range(14)]
            ]

    for i,j in inds:
        array[i] = _copy_array[j]
        names[i] = _copy_names[j]

    array = np.reshape(array,(10,18,3))
    names = np.reshape(names,(10,18))
    ax = sp.get_axes((9,4.5))
    ax.imshow(array)

    for i in range(18):
        for j in range(10):
            c = 'k' if np.linalg.norm(array[j,i]) > 1 else 'w'
            plt.text(i,j,names[j,i],color = c,ha='center',va='center')
    ax.set_axis_off()
    plt.show()


class Arrow3D(FancyArrowPatch):
    """Draw 3D fancy arrow."""
    def __init__(self, x, y, z, u, v, w, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = [x,x+u], [y,y+v], [z,z+w]

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M) #renderer>M for < 3.4 but we don't need it
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer = None): # For matplotlib >= 3.5
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return min(zs)

    def on(self,ax):
        ax.add_artist(self)

def quiver3d(X,Y,Z,U,V,W,ax=None,C = 'r',L = 0.7,mutation_scale=10,**kwargs):
    """Plots 3D arrows on a given ax. See [FancyArrowPatch](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.patches.FancyArrowPatch.html).
    Args:
        - X, Y, Z : 1D arrays of coordinates of arrows' tail point.
        - U, V, W : 1D arrays of dx,dy,dz of arrows.
        - ax: 3D axes, if not given, auto created.
        - C : 1D colors array mapping for arrows. Could be one color.
        - L : 1D linwidths array mapping for arrows. Could be one linewidth.
        - mutation_scale: Arrow head width/size scale. Default is 10.
        - kwargs: FancyArrowPatch's keyword arguments excluding positions,color, lw and mutation_scale, shrinkA, shrinkB which are already used. An important keyword argument is `arrowstyle` which could be '->','-|>', their inverted forms and many more. See on matplotlib.
    """
    if not ax:
        ax = sp.get_axes(figsize=(3.4,3.4),axes_3d=True) # Same aspect ratio.
    if not isinstance(C,(list,tuple,np.ndarray)):
        C = [[*mplc.to_rgb(C)] for x in X]
    C = np.array(C) # Safe for list

    if not isinstance(L,(list,tuple,np.ndarray)):
        L = [L for x in X]
    args_dict = dict(mutation_scale=mutation_scale,shrinkA=0, shrinkB=0)
    for x,y,z,u,v,w,c,l in zip(X,Y,Z,U,V,W,C,L):
        Arrow3D(x, y, z, u, v, w, color=c,lw=l,**args_dict,**kwargs).on(ax)

    return ax


def write_poscar(poscar_data, outfile = None, selective_dynamics = None, overwrite = False):
    """Writes poscar data object to a file or returns string
    Parameters
    ----------
    poscar_data : Output of `export_poscar`,`join_poscars` etc.
    outfile : str,file path to write on.
    selective_dynamics : callable, if given, should be a function like `f(index,x,y,z) -> (bool, bool, bool)` 
        which turns on/off selective dynamics for each atom based in each dimension. See `ipyvasp.POSCAR.data.get_selective_dynamics` for more info.
    overwrite: bool, if file already exists, overwrite=True changes it.

    **Note**: POSCAR is only written in direct format even if it was loaded from cartesian format.
    """
    _comment = poscar_data.metadata.comment
    out_str = f'{poscar_data.SYSTEM}  # ' + (_comment or 'Created by Pivopty')
    scale = poscar_data.metadata.scale
    out_str += "\n  {:<20.14f}\n".format(scale)
    out_str += '\n'.join(["{:>22.16f}{:>22.16f}{:>22.16f}".format(*a) for a in poscar_data.basis/scale])
    uelems = poscar_data.types.to_dict()
    out_str += "\n  " + '    '.join(uelems.keys())
    out_str += "\n  " + '    '.join([str(len(v)) for v in uelems.values()])
    
    if selective_dynamics is not None:
        out_str += "\nSelective Dynamics"

    out_str += "\nDirect\n"
    positions = poscar_data.positions
    pos_list = ["{:>21.16f}{:>21.16f}{:>21.16f}".format(*a) for a in positions]
    
    if selective_dynamics is not None:
        sd = poscar_data.get_selective_dynamics(selective_dynamics).values()
        pos_list = [f"{p}   {s}" for p,s in zip(pos_list,sd)]
    
    out_str += '\n'.join(pos_list)
    if outfile:
        path = Path(outfile)
        if not path.is_file():
            with path.open('w', encoding='utf-8') as f:
                f.write(out_str)

        elif overwrite and path.is_file():
            with path.open('w', encoding='utf-8') as f:
                f.write(out_str)
        else:
            raise FileExistsError(f"{outfile!r} exists, can not overwrite, \nuse overwrite=True if you want to chnage.")
    else:
        print(out_str)

def export_poscar(path = None,content = None):
    """Export POSCAR file to python objects. Only Direct POSCAR supported.
    Args:
        - path: Path/to/POSCAR file. Auto picks in CWD.
        - content: POSCAR content as string, This takes precedence to path.
    """
    if content and isinstance(content,str):
        file_lines = [f'{line}\n' for line in content.splitlines()] # Split by lines strips \n which should be there
    else:
        P = Path(path or './POSCAR')
        if not P.is_file():
            raise FileNotFoundError(f"{str(P)} not found.")
        
        with P.open('r', encoding='utf-8') as f:
            file_lines = f.readlines()
            
    header = file_lines[0].split('#',1)
    SYSTEM = header[0].strip()
    comment = header[1].strip() if len(header) > 1 else 'Exported by Pivopty'

    scale = float(file_lines[1].strip())
    if scale < 0: # If that is for volume
        scale = 1
        
    basis = scale*vp.gen2numpy(file_lines[2:5],(3,3),[-1,-1],exclude = None)
    #volume = np.linalg.det(basis)
    #rec_basis = np.linalg.inv(basis).T # general formula
    out_dict = {'SYSTEM':SYSTEM,#'volume':volume,
                'basis':basis,#'rec_basis':rec_basis,
                'metadata':{'comment':comment,'scale':scale}}

    elems = file_lines[5].split()
    ions = [int(i) for i in file_lines[6].split()]
    N = int(np.sum(ions)) # Must be py int, not numpy
    inds = np.cumsum([0,*ions]).astype(int)
    # Check Cartesian and Selective Dynamics
    lines = [l.strip() for l in file_lines[7:9]] # remove whitespace or tabs
    out_dict['metadata']['cartesian'] = True if ((lines[0][0] in 'cCkK') or (lines[1][0] in 'cCkK')) else False

    poslines = vp.gen2numpy(file_lines[7:],(N,6),(-1,[0,1,2]),exclude="^\s+[a-zA-Z]|^[a-zA-Z]", raw = True).splitlines() # handle selective dynamics word here
    positions = np.array([line.split()[:3] for line in poslines],dtype=float) # this makes sure only first 3 columns are taken
    
    if out_dict['metadata']['cartesian']:
        positions = scale*to_basis(basis, positions)
        print(("Cartesian format found in POSCAR file, converted to direct format."))

    unique_d = {}
    for i,e in enumerate(elems):
        unique_d.update({e:range(inds[i],inds[i+1])})

    elem_labels = []
    for i, name in enumerate(elems):
        for ind in range(inds[i],inds[i+1]):
            elem_labels.append(f"{name} {str(ind - inds[i] + 1)}")
    out_dict.update({'positions':positions,#'labels':elem_labels,
                     'types':unique_d})
    return serializer.PoscarData(out_dict)

# Cell
def _save_mp_API(api_key):
    """
    - Save materials project api key for autoload in functions. This works only for legacy API.
    """
    path = Path.home()/'.ipyvasprc'
    lines = []
    if path.is_file():
        with path.open('r') as fr:
            lines = fr.readlines()
            lines = [line for line in lines if 'MP_API_KEY' not in line]

    with path.open('w') as fw:
        fw.write("MP_API_KEY = {}".format(api_key))
        for line in lines:
            fw.write(line)

# Cell
def _load_mp_data(formula,api_key=None,mp_id=None,max_sites = None, min_sites = None):
    """
    - Returns fetched data using request api of python form materials project website.
    Args:
        - formula  : Material formula such as 'NaCl'.
        - api_key  : API key for your account from material project site. Auto picks if you already used `_save_mp_API` function.
        - mp_id     : Optional, you can specify material ID to filter results.
        - max_sites : Maximum number of sites. If None, sets `min_sites + 1`, if `min_sites = None`, gets all data.
        - min_sites : Minimum number of sites. If None, sets `max_sites + 1`, if `max_sites = None`, gets all data.
    """
    if api_key is None:
        try:
            path = Path.home()/'.ipyvasprc'
            with path.open('r') as f:
                lines=f.readlines()
                for line in lines:
                    if 'MP_API_KEY' in line:
                        api_key = line.split('=')[1].strip()
        except:
            raise ValueError("api_key not given. provide in argument or generate in file using `_save_mp_API(your_mp_api_key)")

    #url must be a raw string
    url = r"https://legacy.materialsproject.org/rest/v2/materials/{}/vasp?API_KEY={}".format(formula,api_key)
    resp = req.request(method='GET',url=url)
    if resp.status_code != 200:
        raise ValueError("Error in fetching data from materials project. Try again!")

    jl = json.loads(resp.text)
    if not 'response' in jl: #check if response
        raise ValueError("Either formula {!r} or API_KEY is incorrect.".format(formula))

    all_res = jl['response']

    if max_sites != None and min_sites != None:
        lower, upper = min_sites, max_sites
    elif max_sites == None and min_sites != None:
        lower, upper = min_sites, min_sites + 1
    elif max_sites != None and min_sites == None:
        lower, upper = max_sites - 1, max_sites
    else:
        lower, upper = '-1', '-1' # Unknown

    if lower != '-1' and upper != '-1':
        sel_res=[]
        for res in all_res:
            if res['nsites'] <= upper and res['nsites'] >= lower:
                sel_res.append(res)
        return sel_res
    # Filter to mp_id at last. more preferred
    if mp_id !=None:
        for res in all_res:
            if mp_id == res['material_id']:
                return [res]
    return all_res


def _cif_str_to_poscar_str(cif_str, comment = None):
    # Using it in other places too
    lines = [line for line in cif_str.splitlines() if line.strip()] # remove empty lines
    
    abc = []
    abc_ang = []
    index = 0
    for ys in lines:
        if '_cell' in ys:
            if '_length' in ys:
                abc.append(ys.split()[1])
            if '_angle' in ys:
                abc_ang.append(ys.split()[1])
            if '_volume' in ys:
                volume = float(ys.split()[1])
        if '_structural' in ys:
            top = ys.split()[1] + f" # {comment}" if comment else ys.split()[1]
    for i,ys in enumerate(lines):
        if '_atom_site_occupancy' in ys:
            index = i +1 # start collecting pos.
    poses = lines[index:]
    pos_str = ""
    for pos in poses:
        s_p = pos.split()
        pos_str += "{0:>12}  {1:>12}  {2:>12}  {3}\n".format(*s_p[3:6],s_p[0])
    
    names = [re.sub('\d+', '', pos.split()[1]).strip() for pos in poses]
    types = []
    for name in names:
        if name not in types:
            types.append(name) # unique types, don't use numpy here.

    # ======== Cleaning ===========
    abc_ang = [float(ang) for ang in abc_ang]
    abc     = [float(a) for a in abc]
    a = "{0:>22.16f} {1:>22.16f} {2:>22.16f}".format(1.0,0.0,0.0) # lattic vector a.
    to_rad = 0.017453292519
    gamma = abc_ang[2]*to_rad
    bx,by = abc[1]*np.cos(gamma),abc[1]*np.sin(gamma)
    b = "{0:>22.16f} {1:>22.16f} {2:>22.16f}".format(bx/abc[0],by/abc[0],0.0) # lattic vector b.
    cz = volume/(abc[0]*by)
    cx = abc[2]*np.cos(abc_ang[1]*to_rad)
    cy = (abc[1]*abc[2]*np.cos(abc_ang[0]*to_rad)-bx*cx)/by
    c = "{0:>22.16f} {1:>22.16f} {2:>22.16f}".format(cx/abc[0],cy/abc[0],cz/abc[0]) # lattic vector b.

    elems = '\t'.join(types)
    nums  = [str(len([n for n in names if n == t])) for t in types]
    nums  = '\t'.join(nums)
    content = f"{top}\n  {abc[0]}\n {a}\n {b}\n {c}\n  {elems}\n  {nums}\nDirect\n{pos_str}"
    return content
                

class InvokeMaterialsProject:
    """Connect to materials project and get data using `api_key` from their site.
    Usage:
    ```python
    from ipyvaspr.sio import InvokeMaterialsProject # or import ipyvasp.InvokeMaterialsProject as InvokeMaterialsProject
    mp = InvokeMaterialsProject(api_key='your_api_key')
    outputs = mp.request(formula='NaCl') #returns list of structures from response
    outupts[0].export_poscar() #returns poscar data
    outputs[0].cif #returns cif data
    ```"""
    def __init__(self,api_key=None):
        "Request Materials Project acess. api_key is on their site. Your only need once and it is saved for later."
        self.api_key = api_key
        self.__response = None
        self.success = False

    def save_api_key(self,api_key):
        "Save api_key for auto reloading later."
        _save_mp_API(api_key)

    @lru_cache(maxsize=2) #cache for 2 calls
    def request(self,formula,mp_id=None,max_sites = None,min_sites=None):
        """Fetch data using request api of python form materials project website. After request, you can access `cifs` and `poscars`.
        Args:
            - formula  : Material formula such as 'NaCl'.
            - mp_id     : Optional, you can specify material ID to filter results.
            - max_sites : Maximum number of sites. If None, sets `min_sites + 1`, if `min_sites = None`, gets all data.
            - min_sites : Minimum number of sites. If None, sets `max_sites + 1`, if `max_sites = None`, gets all data.
        """
        self.__response = _load_mp_data(formula = formula,api_key = self.api_key, mp_id = mp_id, max_sites = max_sites,min_sites=min_sites)
        if self.__response == []:
            raise req.HTTPError("Error in request. Check your api_key or formula.")

        class Structure:
            def __init__(self,response):
                self._cif    = response['cif']
                self.symbol  = response['spacegroup']['symbol']
                self.crystal = response['spacegroup']['crystal_system']
                self.unit    = response['unit_cell_formula']
                self.mp_id   = response['material_id']

            @property
            def cif(self):
                return self._cif

            def __repr__(self):
                return f"Structure(unit={self.unit},mp_id={self.mp_id!r},symbol={self.symbol!r},crystal={self.crystal!r},cif='{self._cif[:10]}...')"

            def write_cif(self,outfile = None):
                if isinstance(outfile,str):
                    with open(outfile,'w') as f:
                        f.write(self._cif)
                else:
                    print(self._cif)

            def write_poscar(self,outfile = None, overwrite = False):
                "Use `ipyvasp.api.POSCAR.write/ipyvasp.sio.write_poscar` if you need extra options."
                write_poscar(self.export_poscar(),outfile = outfile, overwrite = overwrite)

            def export_poscar(self):
                "Export poscar data form cif content."
                content = _cif_str_to_poscar_str(self._cif, comment = f"[{self.mp_id!r}][{self.symbol!r}][{self.crystal!r}] Created by ipyvasp using Materials Project Database")
                return export_poscar(content = content)


        # get cifs
        structures = []
        for res in self.__response:
            structures.append(Structure(res))

        self.success = True # set success flag
        return structures
    
def _str2kpoints(kpts_str):
    hsk_list = []
    for j,line in enumerate(kpts_str.splitlines()):
        if line.strip(): # Make sure line is not empty
            data = line.split()
            if len(data) < 3:
                raise ValueError(f"Line {j + 1} has less than 3 values.")

            point = [float(i) for i in data[:3]]

            if len(data) == 4:
                _4th = data[3] if re.search('\$\\\\[a-zA-Z]+\$|[a-zA-Z]+|[α-ωΑ-Ω]+|\|_', line) else int(data[3])
                point.append(_4th)
            elif len(data) == 5:
                _5th = int(data[4])
                point = point + [data[3],_5th]

            hsk_list.append(point)
    return hsk_list

def get_kpath(kpoints, n = 5, weight= None ,ibzkpt = None,outfile=None, rec_basis = None):
    """
    Generate list of kpoints along high symmetry path. Options are write to file or return KPOINTS list.
    It generates uniformly spaced point with input `n` as just a scale factor of number of points per average length of `rec_basis`.
    
    Parameters   
    ----------
    kpoints : list, str
        Any number points as [(x,y,z,[label],[N]), ...]. N adds as many points in current interval. 
        To disconnect path at a point, provide it as (x,y,z,[label], 0), next point will be start of other patch.
        If `kpoints` is a multiline string, it is converted to list of points. Each line should be in format "x y z [label] [N]". 
    n : int
        Number of point per averge length of `rec_basis`, this makes uniform steps based on distance between points. 
        If (x,y,z,[label], N) is provided, this is ignored for that specific interval. If `rec_basis` is not provided, each interval has exactly `n` points.
        Number of points in each interval is at least 2 even if `n` is less than 2 to keep end points anyway.
    weight : float, None by default to auto generates weights.
    ibzkpt : PathLike, Path to ibzkpt file, required for HSE calculations.
    outfile : PathLike, Path/to/file to write kpoints.
    rec_basis : Reciprocal basis 3x3 array to use for calculating uniform points.

    If `outfile = None`, KPONITS file content is printed.
    """
    if isinstance(kpoints, str):
        kpoints = _str2kpoints(kpoints)
    elif not isinstance(kpoints, (list, tuple)):
        raise TypeError(f"kpoints must be a list or tuple as [(x,y,z,[label],[N]), ...], or multiline string got {kpoints}")
    
    if len(kpoints) < 2:
        raise ValueError("At least two points are required.")

    fixed_patches = []
    where_zero = []
    for idx, point in enumerate(kpoints):
        if not isinstance(point, (list, tuple)):
            raise TypeError(f"kpoint must be a list or tuple as (x,y,z,[label],[N]),  got {point}")
        
        cpt = point # same for length 5, 4 with last entry as string
        
        if len(point) == 3:
             cpt = [*point,''] # make (x,y,z,label)
        elif len(point) == 4:
            if isinstance(point[3],(int,np.integer)):
                 cpt = [*point[:3], '', point[-1]] # add full point as (x,y,z,label, N)
            elif not isinstance(point[3],str):
                raise TypeError(f"4th entry in kpoint should be string label or int number of points for next interval if label is skipped, got {point}")
        elif len(point) == 5:
            if not isinstance(point[3],str):
                raise TypeError(f"4th entry in kpoint should be string label when 5 entries are given, got {point}")
            if not isinstance(point[4],(int, np.integer)):
                raise TypeError(f"5th entry in kpoint should be an integer to add that many points in interval, got {point}")
        else:
            raise ValueError(f"Expects kpoint as (x,y,z,[label],[N]), got {point}")
        
        if isinstance(cpt[-1], (int, np.integer)) and cpt[-1] == 0:
            if idx - 1 in where_zero:
                raise ValueError(f'Break at adjacent kpoints {idx}, {idx+1} is not allowed!')
            if any([idx < 1,  idx > (len(kpoints) - 3)]):
                raise ValueError('Bad break at edges!')
            where_zero.append(idx)
            
        fixed_patches.append(cpt)
    
    def add_points(p1, p2, npts, rec_basis):
        lab = p2[3] # end point label
        if len(p1) == 5:
            m = p1[4] # number of points given explicitly. 
            lab = f'<={p1[3]}|{lab}' # merge labels in case user wants to break path
        elif rec_basis is not None and np.size(rec_basis) == 9:
            basis = np.array(rec_basis)
            coords = to_R3(basis,[p1[:3],p2[:3]])
            _mean = np.mean(np.linalg.norm(basis,axis = 1)) # average length of basis vectors
            m = np.rint(npts*np.linalg.norm(coords[0] - coords[1])/_mean).astype(int) # number of points in interval
        else:
            m = npts # equal number of points in each interval, given by n.
        
        # Doing m - 1 in an interval, so along with last point, total n points are generated per interval.
        Np = max(m - 1, 1) # At least 2 points. one is given by end point of interval.
        X = np.linspace(p1[0],p2[0],Np,endpoint = False)
        Y = np.linspace(p1[1],p2[1],Np,endpoint = False)
        Z = np.linspace(p1[2],p2[2],Np,endpoint = False)
        
        kpts = [(x,y,z) for x,y,z in zip(X,Y,Z)]
        return kpts, Np, lab # return kpoints, number of points, label of end of interval

    points, numbers, labels = [], [0], [fixed_patches[0][3]]
    for p1,p2 in zip(fixed_patches[:-1],fixed_patches[1:]):
        kp, m, lab = add_points(p1,p2, n, rec_basis)
        points.extend(kp) 
        numbers.append(numbers[-1] + m)
        labels.append(lab)
        if lab.startswith('<='):
            labels[-2] = '' # remove label for end of interval if broken, added here
    else: # Add last point at end of for loop
        points.append(p2[:3])

    if weight is None and points:
        weight = 0 if ibzkpt else 1/len(points) # With IBZKPT, we need zero weight, still allow user to override.

    out_str = ["{0:>16.10f}{1:>16.10f}{2:>16.10f}{3:>12.6f}".format(x,y,z,weight) for x,y,z in points]
    out_str = '\n'.join(out_str)
    
    N = len(points)
    if (PI := Path(ibzkpt or '')).is_file(): # handles None automatically
        with PI.open('r') as f:
            lines = f.readlines()

        N = int(lines[1].strip())+N # Update N.
        slines = lines[3:N+4]
        ibz_str = ''.join(slines)
        out_str = "{}\n{}".format(ibz_str.strip('\n'),out_str) # Update out_str, ibz_str is stripped of trailing newline.
    
    path_info = ', '.join(f'{idx}:{lab}' for idx, lab in zip(numbers,labels) if lab != '')
    
    top_str = "Automatically generated using ipyvasp for HSK-PATH {}\n\t{}\nReciprocal Lattice".format(path_info,N)
    out_str = "{}\n{}".format(top_str,out_str)
    if outfile != None:
        with open(outfile,'w', encoding='utf-8') as f: # write to allow unicode characters in path
            f.write(out_str)
    else:
        print(out_str)


def read_kticks(kpoints_path):
    "Reads ticks values and labels in header of kpoint file. Returns dictionary of `kticks` that can be used in plotting functions. If not exist in header, returns empty values(still valid)."
    kticks = []
    if (path := Path(kpoints_path)).is_file():
        with path.open('r', encoding='utf-8') as f: # Read header, important to use utf-8 to include greek letters.
            top_line = f.readline()
        if 'HSK-PATH' in top_line:
            head = top_line.split('HSK-PATH')[1].strip() # Only update head if HSK-PATH is found.
    
            hsk = [[v.strip() for v in vs.split(':')] for vs in head.split(',')]
            for k,v in hsk:
                kticks.append((int(k),v))
                
    return kticks

# Cell
def _get_basis(path_pos):
    """Returns given(computed) and inverted basis as tuple(given,inverted).
    Args:
        - path_pos: path/to/POSCAR or 3 given vectors as rows of a matrix."""
    if isinstance(path_pos,(list,tuple,np.ndarray)) and np.ndim(path_pos) ==2:
        basis = np.array(path_pos)
    elif isinstance(path_pos,str) or isinstance(path_pos,type(None)):
        basis = export_poscar(path_pos).basis
    else:
        raise FileNotFoundError("{!r} does not exist or not 3 by 3 list.".format(path_pos))
    # Process. 2π is not included in vasp output
    rec_basis = np.linalg.inv(basis).T # Compact Formula
    Basis = namedtuple('Basis', ['given', 'inverted'])
    return Basis(basis,rec_basis)

# Cell
def get_kmesh(poscar_data, *args, shift = 0, weight = None, cartesian = False, ibzkpt= None, outfile=None, endpoint = True):
    """**Note**: Use `ipyvasp.POSCAR.get_kmesh` to get k-mesh based on current POSCAR.
    - Generates uniform mesh of kpoints. Options are write to file, or return KPOINTS list.
    Args:
        - poscar_data: export_poscar() or export_vasprun().poscar().
        - *args: 1 or 3 integers which decide shape of mesh. If 1, mesh points equally spaced based on data from POSCAR.
        - shift  : Only works if cartesian = False. Defualt is 0. Could be a number or list of three numbers to add to interval [0,1].
        - weight : Float, if None, auto generates weights.
        - cartesian: If True, generates cartesian mesh.
        - ibzkpt : Path to ibzkpt file, required for HSE calculations.
        - outfile: Path/to/file to write kpoints.
        - endpoint: Default True, include endpoints in mesh at edges away from origin.

    If `outfile = None`, KPOINTS file content is printed."""
    if len(args) not in [1,3]:
        raise ValueError("get_kmesh() takes 1 or 3 args!")

    if cartesian:
        norms = np.ptp(poscar_data.rec_basis,axis=0)
    else:
        norms = np.linalg.norm(poscar_data.rec_basis, axis = 1)

    if len(args) == 1:
        if not isinstance(args[0],(int, np.integer)):
            raise ValueError("get_kmesh expects integer for first positional argument!")
        nx,ny,nz = [args[0] for _ in range(3)]

        weights = norms/np.max(norms) # For making largest side at given n
        nx, ny, nz = np.rint(weights*args[0]).astype(int)

    elif len(args) == 3:
        for i,a in enumerate(args):
            if not isinstance(a,(int, np.integer)):
                raise ValueError("get_kmesh expects integer at position {}!".format(i))
        nx,ny,nz = list(args)

    low,high = np.array([[0,0,0],[1,1,1]]) + shift
    if cartesian:
        verts = get_bz(poscar_data.basis, primitive=False).vertices
        low, high = np.min(verts,axis=0), np.max(verts,axis=0)
        low = (low * 2 * np.pi / poscar_data.metadata.scale).round(12) # Cartesian KPOINTS are in unit of 2pi/SCALE
        high = (high * 2 * np.pi / poscar_data.metadata.scale).round(12)

    (lx,ly,lz),(hx,hy,hz) = low,high
    points = []
    for k in np.linspace(lz,hz,nz, endpoint = endpoint):
        for j in np.linspace(ly,hy,ny, endpoint = endpoint):
            for i in np.linspace(lx,hx,nx, endpoint = endpoint):
                points.append([i,j,k])

    points = np.array(points)
    points[np.abs(points) < 1e-10] = 0

    if len(points) == 0:
        raise ValueError('No KPOINTS in BZ from given input. Try larger input!')

    if weight == None and len(points) != 0:
        weight = float(1/len(points))

    out_str = ["{0:>16.10f}{1:>16.10f}{2:>16.10f}{3:>12.6f}".format(x,y,z,weight) for x,y,z in points]
    out_str = '\n'.join(out_str)
    N = len(points)
    if ibzkpt and (PI := Path(ibzkpt)):
        with PI.open('r', encoding='utf-8') as f:
            lines = f.readlines()

        if (cartesian == False) and (lines[2].strip()[0] in 'cCkK'):
            raise ValueError("ibzkpt file is in cartesian coordinates, use get_kmesh(...,cartesian = True)!")

        N = int(lines[1].strip())+N # Update N.
        slines = lines[3:N+4]
        ibz_str = ''.join(slines)
        out_str = "{}\n{}".format(ibz_str,out_str) # Update out_str
    mode = 'Reciprocal' if cartesian == False else 'Cartesian'
    top_str = "Generated uniform mesh using ipyvasp, GRID-SHAPE = [{},{},{}]\n\t{}\n{}".format(nx,ny,nz,N,mode)
    out_str = "{}\n{}".format(top_str,out_str)
    if outfile != None:
        with open(outfile,'w', encoding='utf-8') as f:
            f.write(out_str)
    else:
        print(out_str)

# Cell
def _tan_inv(vy,vx):
    """
    - Returns full angle from x-axis counter clockwise.
    Args:
        - vy : Perpendicular componet of vector including sign.
        - vx : Base compoent of vector including sign.
    """
    angle = 0  # Place hodler to handle exceptions
    if vx == 0 and vy == 0:
        angle = 0
    elif vx == 0 and np.sign(vy) == -1:
        angle = 3*np.pi/2
    elif vx == 0 and np.sign(vy) == 1:
        angle = np.pi/2
    else:
        theta = abs(np.arctan(vy/vx))
        if np.sign(vx) == 1 and np.sign(vy) == 1:
            angle = theta
        if np.sign(vx) == -1 and np.sign(vy) == 1:
            angle = np.pi - theta
        if np.sign(vx) == -1 and np.sign(vy) == -1:
            angle = np.pi + theta
        if np.sign(vx) == 1 and np.sign(vy) == -1:
            angle = 2*np.pi - theta
        if np.sign(vx) == -1 and vy == 0:
            angle = np.pi
        if np.sign(vx) == 1 and vy == 0:
            angle = 2*np.pi
    return angle

def order(points,loop=True):
    """
    - Returns indices of counterclockwise ordered vertices of a plane in 3D.
    Args:
        - points: numpy array of shape (N,3) or List[List(len=3)].
        - loop  : Default is True and appends start point at end to make a loop.
    - **Example**
        > pts = np.array([[1,0,3],[0,0,0],[0,1,2]])
        > inds = order(pts)
        > pts[inds]
        ```
        array([[1, 2, 3],
               [0, 0, 0],
               [1, 0, 3]
               [0, 1, 2]])
        ```
    """
    points = np.array(points) # Make array.
    # Fix points if start point is zero.
    if np.sum(points[0]) == 0:
        points = points + 0.5

    center = np.mean(points,axis=0) # 3D cent point.
    vectors = points - center # Relative to center

    ex = vectors[0]/np.linalg.norm(vectors[0])  # i
    ey = np.cross(center,ex)
    ey = ey/np.linalg.norm(ey)  # j

    angles= []
    for i, v in enumerate(vectors):
        vx = np.dot(v,ex)
        vy = np.dot(v,ey)
        angle = _tan_inv(vy,vx)
        angles.append([i,angle])

    s_angs = np.array(angles)
    ss = s_angs[s_angs[:,1].argsort()] #Sort it.

    if loop: # Add first at end for completing loop.
        ss = np.concatenate((ss,[ss[0]]))

    return ss[:,0].astype(int) # Order indices.


def _out_bz_plane(test_point,plane):
    """
    - Returns True if test_point is between plane and origin. Could be used to sample BZ mesh in place of ConvexHull.
    Args:
        - test_points: 3D point.
        - plane      : List of at least three coplanar 3D points.
    """
    outside = True
    p_test = np.array(test_point)
    plane = np.unique(plane,axis=0) #Avoid looped shape.
    c = np.mean(plane,axis=0) #center
    _dot_ = np.dot(p_test-c,c)
    if _dot_ < -1e-5:
        outside = False
    return outside


def _rad_angle(v1,v2):
    """
    - Returns interier angle between two vectors.
    Args:
        - v1,v2 : Two vectors/points in 3D.
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    norm  = np.linalg.norm(v1)*np.linalg.norm(v2)
    dot_p = np.round(np.dot(v1,v2)/norm,12)
    angle = np.arccos(dot_p)
    return angle

def to_plane(normal, points):
    "Project points to a plane defined by `normal`. shape of normal should be (3,) and of points (N,3)."
    if np.ndim(normal) + 1 != np.ndim(points):
        raise ValueError("Shape of points should be (N,3) and of normal (3,).")
    points = np.array(points)
    nu = normal/np.linalg.norm(normal) # Normal unit vector
    along_normal = points.dot(nu) 
    points = points - along_normal[:,None]*nu # v - (v.n)n
    return points

from scipy.spatial.transform import Rotation
def rotation(angle_deg,axis_vec):
    """Get a scipy Rotation object at given `angle_deg` around `axis_vec`.
    Usage:
        rot = rotation(60,[0,0,1])
        rot.apply([1,1,1])
        [-0.3660254  1.3660254  1.] #give this
    """
    axis_vec = np.array(axis_vec)/np.linalg.norm(axis_vec) # Normalization
    angle_rad = np.deg2rad(angle_deg)
    return Rotation.from_rotvec(angle_rad * axis_vec)

# Cell
def get_bz(path_pos = None,loop = True,primitive=False):
    """
    Return required data to construct first Brillouin zone.
    
    Parameters
    ----------
    path_pos : POSCAR file path or list of 3 Real space vectors in 3D as list[list,list,list].
    loop : If True, joins the last vertex of a BZ plane to starting vertex in order to complete loop.
    primitive : Defualt is False and returns Wigner-Seitz cell, If True returns parallelipiped in rec_basis.

    Returns
    -------
    BrZoneData(basis, vertices, faces). 
    
    - You can access special points with `.get_special_points` method or by default from `.specials` property.
    - You can access coordinates of faces with `.faces_coords` property.
    - You can access normals vectors of faces with `.normals` property.
    """
    basis = _get_basis(path_pos).inverted # Reads
    b1, b2, b3 = basis # basis are reciprocal basis
    # Get all vectors for BZ
    if primitive:
        # verts, faces, below are in order, if you cange 1, change all
        verts = np.array([
            [0,0,0],
            b1,
            b2,
            b3,
            b1+b2,
            b1+b3,
            b2+b3,
            b1+b2+b3
        ])
        idx_faces = ( # Face are kept anti-clockwise sorted.
            (0,1,5,3,0),
            (0,2,4,1,0),
            (0,3,6,2,0),
            (2,6,7,4,2),
            (1,4,7,5,1),
            (3,5,7,6,3)
        )
        if loop is False:
            idx_faces = tuple(face[:-1] for face in idx_faces)
        
    else:
        vectors = []
        for i,j,k in product([0,1,-1],[0,1,-1],[0,1,-1]):
            vectors.append(i*b1+j*b2+k*b3)
            
        vectors = np.array(vectors)
        # Generate voronoi diagram
        vor = Voronoi(vectors)
        faces = []
        vrd = vor.ridge_dict
        for r in vrd:
            if r[0] == 0 or r[1] == 0:
                verts_in_face = np.array([vor.vertices[i] for i in vrd[r]])
                faces.append(verts_in_face)
        
        verts = [v for vs in faces for v in vs]
        verts = np.unique(verts,axis=0)
        
        # make faces as indices over vertices because that what most programs accept
        idx_faces = []
        for face in faces:
            vert_inds = [i for i,v in enumerate(verts) if tuple(v) in [tuple(f) for f in face]] # having tuple comparsion is important here.
            idx_faces.append(vert_inds) # other zero is to pick single index out of same three

        # order faces
        idx_faces = [tuple(face[i] for i in order(verts[face], loop = loop)) for face in idx_faces]
    
    out_dict = {'basis' : basis, 'vertices' : verts,'faces' : idx_faces}
    return serializer.BrZoneData(out_dict)


# Cell
def splot_bz(bz_data, plane = None, ax = None, color='blue',fill=True,vectors = (0,1,2),colormap=None,shade = True, alpha = 0.4, zoffset = 0, **kwargs):
    """
    Plots matplotlib's static figure of Brillouin zone. You can also plot in 2D on a 3D axes.
    
    Parameters
    ----------
    bz_data : Output of `get_bz`.
    plane : Default is None and plots 3D surface. Can take 'xy','yz','zx' to plot in 2D.
    fill : True by defult, determines whether to fill surface of BZ or not.
    color : color to fill surface and stroke color.
    vectors : Tuple of indices of basis vectors to plot. Default is (0,1,2). All three are plotted in 3D (you can turn of by None or empty tuple), whhile you can specify any two/three in 2D.
    ax : Auto generated by default, 2D/3D axes, auto converts in 3D on demand as well.
    colormap : If None, single color is applied, only works in 3D and `fill=True`. Colormap is applied along z.
    shade : Shade polygons or not. Only works in 3D and `fill=True`.
    alpha : Opacity of filling in range [0,1]. Increase for clear viewpoint.
    zoffset : Only used if plotting in 2D over a 3D axis. Default is 0. Any plane 'xy','yz' etc. can be plotted but it will be in xy plane of 3D axes.
    
    kwargs are passed to `plt.plot` or `Poly3DCollection` if `fill=True`.
    
    Returns
    -------
    ax : Matplotlib's 2D axes if `plane=None` otherswise 3D axes.
    """
    vname = 'a' if bz_data.__class__.__name__ == 'CellData' else 'b'
    label = r"$k_{}/2π$" if vname == 'b' else "{}"
    if not ax: #For both 3D and 2D, initialize 2D axis.
        ax = sp.get_axes(figsize=(3.4,3.4)) #For better display

    _label = r'\vec{' + vname + '}' # For both
    valid_planes = 'xyzxzyx' # cylic
    
    if vectors and not isinstance(vectors,(tuple,list)):
        raise ValueError(f"`vectors` expects tuple or list, got {vectors!r}")
    
    if vectors is None:
        vectors = () # Empty tuple to make things work below
    
    for v in vectors:
        if v not in [0,1,2]:
            raise ValueError(f"`vectors` expects values in [0,1,2], got {vectors!r}")
    
    name = kwargs.pop('label',None) # will set only on single line
    kwargs.pop('zdir',None) # 2D plot on 3D axes is only supported in xy plane.
    
    if plane: #Project 2D, works on 3D axes as well
        ind = valid_planes.index(plane)
        
        arr = [0,1,2,0,2,1,0]
        i, j = arr[ind], arr[ind+1]
        is3d = getattr(ax,'name','') == '3d'
        normals = {'xy' : (0,0,1), 'yz' : (1,0,0), 'zx' : (0,1,0), 'yx' : (0,0,-1), 'zy' : (-1,0,0), 'xz' : (0,-1,0)}
        if plane not in normals:
            raise ValueError(f"`plane` expects value in 'xyzxzyx' or None, got {plane!r}")
    
        z0 = [0,0,zoffset] if plane in 'xyx' else [0,zoffset,0] if plane in 'xzx' else [zoffset,0,0]
        idxs = {'xy' : [0,1], 'yz' : [1,2], 'zx' : [2,0], 'yx' : [1,0], 'zy' : [2,1], 'xz' : [0,2]}
        for idx, f in enumerate(bz_data.faces_coords):
            g = to_plane(normals[plane],f) + z0
            line, = ax.plot(*(g.T if is3d else g[:,idxs[plane]].T),color = color,**kwargs)
            if idx == 0:
                line.set_label(name) # only one line
        
        if vectors:
            s_basis = to_plane(normals[plane],bz_data.basis[(vectors,)])

            for k,b in zip(vectors,s_basis):
                x,y = b[idxs[plane]]
                l = r" ${}_{} $".format(_label,k+1)
                l = l + '\n' if y < 0 else '\n' + l
                ha = 'right' if x < 0 else 'left'
                xyz = 0.8*b + z0 if is3d else (0.8*x,0.8*y)
                ax.text(*xyz, l, va = 'center',ha=ha,clip_on = True) # must clip to have limits of axes working.
                ax.scatter(*(xyz if is3d else 0.8*xyz),color='w',s=0.0005) # Must be to scale below arrow.
            if is3d:
                XYZ,UVW = (np.ones_like(s_basis)*z0).T, s_basis.T
                quiver3d(*XYZ,*UVW,C='k',L=0.7,ax = ax,arrowstyle="-|>",mutation_scale=7)
            else:
                s_zero = [0 for _ in s_basis] # either 3 or 2.
                ax.quiver(s_zero,s_zero,*s_basis[:,idxs[plane]].T,lw=0.9,color='navy',angles='xy', scale_units='xy', scale=1)

        ax.set_xlabel(label.format(idxs[plane][0]))
        ax.set_ylabel(label.format(idxs[plane][1]))
        if is3d:
            ind = [i for i in range(3) if i not in idxs[plane]][0]
            ax.set_zlabel(label.format(ind))
            ax.set_aspect('equal')
            zmin, zmax = ax.get_zlim()
            if zoffset > zmax:
                zmax = zoffset
            elif zoffset < zmin:
                zmin = zoffset
            ax.set_zlim([zmin,zmax])
        else:
            ax.set_aspect(1) # Must for 2D axes to show actual lengths of BZ
        return ax
    else: # Plot 3D
        if getattr(ax, 'name', '') == "3d": # handle None or 2D axes passed.
            ax3d = ax
        else:
            pos = ax.get_position()
            fig = ax.get_figure()
            ax.remove()
            ax3d = fig.add_axes(pos,projection='3d',azim=45,elev=30,proj_type='ortho')

        if fill:
            if colormap:
                colormap = colormap if colormap in plt.colormaps() else 'viridis'
                cz = [np.mean(np.unique(f,axis=0),axis=0)[2] for f in bz_data.faces_coords]
                levels = (cz - np.min(cz))/np.ptp(cz) # along Z.
                colors = plt.cm.get_cmap(colormap)(levels)
            else:
                colors = np.array([[*mplc.to_rgb(color)] for f in bz_data.faces_coords]) # Single color.
            
            poly = Poly3DCollection(bz_data.faces_coords,edgecolors = [color,],facecolors = colors, alpha=alpha, shade = shade, label = name, **kwargs)
            
            ax3d.add_collection(poly)
            ax3d.autoscale_view()
        else:
            line, = [ax3d.plot3D(f[:,0],f[:,1],f[:,2],color=(color),**kwargs) for f in bz_data.faces_coords][0]
            line.set_label(name) # only one line

        if vectors:
            for k,v in enumerate(0.35*bz_data.basis):
                ax3d.text(*v,r"${}_{}$".format(_label,k+1),va='center',ha='center')

            XYZ,UVW = [[0,0,0],[0,0,0],[0,0,0]], 0.3*bz_data.basis.T
            quiver3d(*XYZ,*UVW,C='k',L=0.7,ax=ax3d,arrowstyle="-|>",mutation_scale=7)

        l_ = np.min(bz_data.vertices,axis=0)
        h_ = np.max(bz_data.vertices,axis=0)
        ax3d.set_xlim([l_[0],h_[0]])
        ax3d.set_ylim([l_[1],h_[1]])
        ax3d.set_zlim([l_[2],h_[2]])

        # Set aspect to same as data.
        ax3d.set_box_aspect(np.ptp(bz_data.vertices,axis=0))

        ax3d.set_xlabel(label.format('x'))
        ax3d.set_ylabel(label.format('y'))
        ax3d.set_zlabel(label.format('z'))
        return ax3d

# Cell
def iplot_bz(bz_data, fill = False,color = 'rgba(168,204,216,0.4)', special_kpoints = True, alpha = 0.4,ortho3d=True,fig = None, **kwargs):
    """
    Plots interactive figure showing axes,BZ surface, special points and basis, each of which could be hidden or shown.
    
    Parameters
    ----------
    bz_data : Output of `get_bz`.
    fill : False by defult, determines whether to fill surface of BZ or not.
    color : color to fill surface 'rgba(168,204,216,0.4)` by default.
    background : Plot background color, default is 'rgb(255,255,255)'.
    special_kpoints : True by default, determines whether to plot special points or not.
    alpha : Opacity of BZ planes.
    ortho3d : Default is True, decides whether x,y,z are orthogonal or perspective.
    fig : (Optional) Plotly's `go.Figure`. If you want to plot on another plotly's figure, provide that.
    
    kwargs are passed to `plotly.graph_objects.Scatter3d` for BZ lines.
    Returns
    -------
    fig : plotly.graph_object's Figure instance.
    """
    if not fig:
        fig = go.Figure()
    # Name fixing
    vname = 'a' if bz_data.__class__.__name__ == 'CellData' else 'b'
    axes_text = ["<b>k</b><sub>x</sub>/2π","","<b>k</b><sub>y</sub>/2π","","<b>k</b><sub>z</sub>/2π"]
    if vname == 'a':
        axes_text = ["<b>x</b>","","<b>y</b>","","<b>z</b>"] # Real space
       
    zone_name = kwargs.pop('name','BZ' if vname == 'b' else 'Lattice')
    # Axes
    _len = 0.5*np.mean(bz_data.basis)
    fig.add_trace(go.Scatter3d(x=[_len,0,0,0,0],y=[0,0,_len,0,0],z=[0,0,0,0,_len],
        mode = 'lines+text', text = axes_text,
        line_color='skyblue', legendgroup = 'Axes',name = 'Axes'))
    fig.add_trace(go.Cone(x=[_len,0,0],y=[0,_len,0],z=[0,0,_len],
        u = [1,0,0],v = [0,1,0],w = [0,0,1], showscale = False,
        sizemode='absolute',sizeref = 0.5, anchor = 'tail',
        colorscale=['skyblue' for _ in range(3)],legendgroup = 'Axes',name = 'Axes'))
    
    # Basis
    for i,b in enumerate(bz_data.basis):
        fig.add_trace(go.Scatter3d(x=[0,b[0]], y=[0,b[1]],z=[0,b[2]],
            mode='lines+text',legendgroup="{}<sub>{}</sub>".format(vname,i+1), line_color='red',
            name="<b>{}</b><sub>{}</sub>".format(vname,i+1),text=["","<b>{}</b><sub>{}</sub>".format(vname,i+1)]))
        
        uvw = b/np.linalg.norm(b) # Unit vector for cones
        fig.add_trace(go.Cone(x=[b[0]],y=[b[1]],z=[b[2]],
            u = uvw[0:1],v = uvw[1:2],w = uvw[2:],showscale = False,colorscale = 'Reds',
            sizemode = 'absolute', sizeref = 0.02, anchor = 'tail',
            legendgroup="{}<sub>{}</sub>".format(vname,i+1),name="<b>{}</b><sub>{}</sub>".format(vname,i+1)))
    
    # Faces
    legend = True
    for pts in bz_data.faces_coords:
        fig.add_trace(go.Scatter3d(x=pts[:,0], y=pts[:,1],z=pts[:,2],
            mode='lines',line_color=color, legendgroup = zone_name,name = zone_name,
            showlegend = legend, **kwargs)) 
        
        legend = False # Only first legend to show for all

    if fill:
        xc = bz_data.vertices[ConvexHull(bz_data.vertices).vertices]
        fig.add_trace(go.Mesh3d(x=xc[:, 0], y=xc[:, 1], z=xc[:, 2],
                        color = color,
                        opacity = alpha,
                        alphahull=0,
                        lighting = dict(diffuse=0.5),
                        legendgroup = zone_name,name=zone_name))
        

    # Special Points only if in reciprocal space.
    if vname == 'b' and special_kpoints:
        for tr in fig.data: # hide all traces hover made before
            tr.hoverinfo = 'none' # avoid overlapping with special points
        
        texts,values =[],[]
        norms = np.round(np.linalg.norm(bz_data.specials.coords,axis=1),8)
        sps = bz_data.specials
        for key,value, norm in zip(sps.kpoints.round(6), sps.coords, norms):
            texts.append("K = {}</br>d = {}".format(key,norm))
            values.append([[*value,norm]])

        values = np.array(values).reshape((-1,4))
        norm_max = np.max(values[:,3])
        c_vals = np.array([int(v*255/norm_max) for v in values[:,3]])
        colors = [0 for i in c_vals]
        _unique = np.unique(np.sort(c_vals))[::-1]
        _lnp = np.linspace(0,255,len(_unique)-1)
        _u_colors = ["rgb({},0,{})".format(r,b) for b,r in zip(_lnp,_lnp[::-1])]
        for _un,_uc in zip(_unique[:-1],_u_colors):
            _index = np.where(c_vals == _un)[0]
            for _ind in _index:
                colors[_ind]=_uc

        colors[0]= "rgb(255,215,0)" # Gold color at Gamma!.
        fig.add_trace(go.Scatter3d(x=values[:,0], y=values[:,1],z=values[:,2],
                hovertext=texts,name="HSK",marker=dict(color=colors,size=4),mode='markers'))
    
    proj = dict(projection=dict(type = "orthographic")) if ortho3d else {}
    camera = dict(center=dict(x=0.1, y=0.1, z=0.1),**proj)
    fig.update_layout(template = 'plotly_white', scene_camera=camera,
        font_family="Times New Roman",font_size= 14,
        scene = dict(aspectmode='data',xaxis = dict(showbackground=False,visible=False),
                        yaxis = dict(showbackground=False,visible=False),
                        zaxis = dict(showbackground=False,visible=False)),
                        margin=dict(r=10, l=10,b=10, t=30))
    return fig

# Cell
def to_R3(basis,points):
    """Transforms coordinates of points (relative to non-othogonal basis) into orthogonal space.
    Parameters
    ----------
    basis : 3x3 matrix with basis vectors as rows like [[b1x, b1y, b1z],[b2x, b2y, b2z],[b3x, b3y, b3z]].
    points : Nx3 points relative to basis, such as KPOINTS and Lattice Points.
        
    Conversion formula:   
    [x,y,z] = n1*b1 + n2*b2 +n3*b3 = [n1, n2, n3] @ [[b1x, b1y, b1z],[b2x, b2y, b2z],[b3x, b3y, b3z]]

    **Note**: Do not use this function if points are Cartesian or provide identity basis.
    """
    return np.array(points) @ basis

def to_basis(basis,coords):
    """Transforms coordinates of points (relative to othogonal basis) into basis space.
    Parameters
    ---------
    basis : 3x3 matrix with basis vectors as rows like [[b1x, b1y, b1z],[b2x, b2y, b2z],[b3x, b3y, b3z]].
    coords : Nx3 points relative to cartesian axes.
    
    Conversion formula:     
    [n1, n2, n3] = [x,y,z] @ inv([[b1x, b1y, b1z],[b2x, b2y, b2z],[b3x, b3y, b3z]])
    """
    return np.array(coords) @ np.linalg.inv(basis)

# Cell
def kpoints2bz(bz_data,kpoints, primitive = False, shift = 0):
    """Brings KPOINTS inside BZ. Applies `to_R3` only if `primitive=True`.
    Args:
        - bz_data  : Output of get_bz(), make sure use same value of `primitive` there and here.
        - kpoints  : List or array of KPOINTS to transorm into BZ or R3.
        - primitive: Default is False and brings kpoints into regular BZ. If True, returns `to_R3()`.
        - shift    : This value is added to kpoints before any other operation, single number of list of 3 numbers for each direction.

    """
    kpoints = np.array(kpoints) + shift
    if primitive:
        return to_R3(bz_data.basis,kpoints)

    cent_planes = [np.mean(np.unique(face,axis=0),axis=0) for face in bz_data.faces_coords]

    out_coords = np.empty(np.shape(kpoints)) # To store back

    def inside(coord,cent_planes):
        _dots_ = np.max([np.dot(coord-c, c) for c in cent_planes]) #max in all planes
        #print(_dots_)
        if np.max(_dots_) > 1e-8: # Outside
            return [] # empty for comparison
        else: # Inside
            return list(coord) # Must be in list form
    
    for i,p in enumerate(kpoints):
        for q in product([0,1,-1],[0,1,-1],[0,1,-1]):
            # First translate, then make coords, then feed it back
            #print(q)
            pos = to_R3(bz_data.basis, p + np.array(q))
            r = inside(pos,cent_planes)
            if r:
                #print(p,'-->',r)
                out_coords[i] = r
                StopIteration

    return out_coords # These may have duplicates, apply np.unique(out_coords,axis=0). do this in surface plots

def inside_convexhull(hull, points):
    if not isinstance(hull, ConvexHull):
        raise TypeError("hull must be a scipy.spatial.ConvexHull object")
    
    if np.shape(points)[-1] != hull.points.shape[-1]:
        raise ValueError("points must have same physical dimension as hull.points")
    
    # A.shape = (facets, d) and b.shape = (facets, 1)
    A, b = hull.equations[:, :-1], hull.equations[:, -1:]
    eps = np.finfo(np.float32).eps # do not use custom tolerance here
    
    # Points inside CovexHull satifies equation Ax + b <= 0.
    return np.all(A @ points.T + b < eps, axis = 0)

# Cell
def _fix_sites(poscar_data,tol=1e-2,eqv_sites = False,translate = None): # should not be exposed mostly be used in visualizations
    """Add equivalent sites to make a full data shape of lattice. Returns same data after fixing.
    
    Parameters
    ----------
    eqv_sites : If True, add sites on edges and faces. If False, just fix coordinates, i.e. `pos > 1 - tol -> pos - 1`, useful for merging poscars to make slabs.
    translate : A number(+/-) or list of three numbers to translate in a,b,c directions.
    """
    pos = poscar_data.positions.copy() # We can also do poscar_data.copy().positions that copies all contents.
    labels = np.array(poscar_data.labels) # We need to store equivalent labels as well
    out_dict = poscar_data.to_dict() # For output

    if translate and isinstance(translate,(int,np.integer,float)):
        pos = pos + (translate - int(translate)) # Only translate in 0 - 1
    elif translate and len(translate) == 3:
        txyz = np.array([translate])
        pos = pos + (txyz - txyz.astype(int))

    # Fix coordinates of sites distributed on edges and faces
    pos -= (pos > (1 - tol)).astype(int) # Move towards orign for common fixing like in joining POSCARs
    out_dict['positions'] = pos
    out_dict['metadata']['comment'] = 'Modified by ipyvasp'


    # Add equivalent sites on edges and faces if given,handle each sepecies separately
    if eqv_sites:
        new_dict, start = {}, 0
        for k,v in out_dict['types'].items():
            vpos = pos[v]
            vlabs = labels[v]
            cond_ops = [
                (((vpos[:,0] + 1) < (tol +1)), [[1,0,0]]), # Add 1 to x if within tol
                (((vpos[:,1] + 1) < (tol +1)), [[0,1,0]]), # Add 1 to y on modified and if within tol
                (((vpos[:,2] + 1) < (tol +1)), [[0,0,1]]), # Add 1 to z and if within tol
                (((vpos[:,0:2] + 1) < (tol +1)).all(axis = 1), [[1,1,0]]), # Add 1 to x and y if within tol
                (((vpos[:,1:3] + 1) < (tol +1)).all(axis = 1), [[0,1,1]]), # Add 1 to y and z if within tol
                (((vpos[:,[0,2]] + 1) < (tol +1)).all(axis = 1), [[1,0,1]]), # Add 1 to x and z if within tol
                (((vpos + 1) < (tol +1)).all(axis=1), [[1,1,1]]), # Add 1 to all if within tol  
            ]
            spos = [vpos[c] + op for c, op in cond_ops]
            slab = [vlabs[c] for c, op in cond_ops]
            
            new_dict[k] = {'pos':np.vstack([vpos,*spos]), 'lab':np.hstack([vlabs,*slab])}
            new_dict[k]['range'] = range(start,start+len(new_dict[k]['pos']))
            start += len(new_dict[k]['pos'])

        out_dict['positions'] = np.vstack([new_dict[k]['pos'] for k in new_dict.keys()])
        out_dict['metadata']['eqv_labels'] = np.hstack([new_dict[k]['lab'] for k in new_dict.keys()])
        out_dict['types'] = {k:new_dict[k]['range'] for k in new_dict.keys()}

    return serializer.PoscarData(out_dict)

def translate_poscar(poscar_data, offset):
    """ Translate sites of a PPSCAR. Usully a farction of integarers like 1/2,1/4 etc.
    Args:
        - poscar_data: Output of `export_poscar` or `export_vasprun().poscar`.
        - offset: A number(+/-) or list of three numbers to translate in a,b,c directions.
    """
    return _fix_sites(poscar_data, translate = offset, eqv_sites = False)

def get_pairs(poscar_data, positions, r, tol=1e-3):
    """Returns a tuple of Lattice (coords,pairs), so coords[pairs] given nearest site bonds.
    Args:
        - poscar_data: Output of `export_poscar` or `export_vasprun().poscar`.
        - positions: Array(N,3) of fractional positions of lattice sites. If coordinates positions, provide unity basis.
        - r        : Cartesian distance between the pairs in units of Angstrom e.g. 1.2 -> 1.2E-10.
        - tol      : Tolerance value. Default is 10^-3.
    """
    coords = to_R3(poscar_data.basis,positions)
    tree = KDTree(coords)
    inds = np.array([[*p] for p in tree.query_pairs(r,eps=tol)])
    return serializer.dict2tuple('Lattice',{'coords':coords,'pairs':inds})

def _get_bond_length(poscar_data, given = None):
    "`given` bond length should be in range [0,1] which is scaled to V^(1/3)."
    if given is not None:
        return given*poscar_data.volume**(1/3)
    else:
        keys = list(poscar_data.types.keys())
        if len(keys) == 1:
            keys = [*keys,*keys] # strill need it to be a list of two elements
            
        dists = [poscar_data.get_distance(k1,k2) for k1, k2 in combinations(keys,2)]
        return np.mean(dists)*1.05 # Add 5% margin over mean distance, this covers same species too, and in multiple species, this will stop bonding between same species.
        
def _masked_data(poscar_data, mask_sites):
    "Returns indices of sites which satisfy the mask_sites function."
    if not callable(mask_sites):
        raise TypeError('`mask_sites` should be a callable function.')
    
    if len(inspect.signature(mask_sites).parameters) != 4:
        raise ValueError("`mask_sites` takes exactly 4 arguments: (index,x,y,z) in fractional coordinates")
    
    if not isinstance(mask_sites(0,0,0,0),bool):
        raise TypeError('`mask_sites` should return a boolean value.')
    
    
    pick = []
    for i, pos in enumerate(poscar_data.positions):
        if mask_sites(i,*pos):
            pick.append(i)
    return pick
        
# Cell
def iplot_lattice(poscar_data, sizes = 10, colors = None, bond_length = None,tol = 1e-2,bond_tol = 1e-3,eqv_sites = True,
    translate = None, fig = None, ortho3d = True, mask_sites = None, bond_kws = dict(line_width = 4),site_kws = dict(line_color='rgba(1,1,1,0)',line_width=0.001, opacity = 1), plot_cell = True, **kwargs):
    """
    Plotly's interactive plot of lattice.
    Parameters
    ----------
    poscar_data : Output of export_poscar or export_vasprun().poscar.
    sizes : Size of sites. Either one int/float or list equal to type of ions.
    colors : Sequence of colors for each type. Automatically generated if not provided.
    bond_length : Length of bond in fractional unit [0,1]. It is scaled to V^1/3 and auto calculated if not provides.
    mask_sites : Provide a mask function `f(index, x,y,z) -> bool` to show only selected sites. For example, to show only sites with z > 0.5, use `mask_sites = lambda i, x,y,z: x > 0.5`.
    bond_kws : Keyword arguments passed to `plotly.graph_objects.Scatter3d` for bonds. Default is jus hint, you can use any keyword argument that is accepted by `plotly.graph_objects.Scatter3d`.
    site_kws : Keyword arguments passed to `plotly.graph_objects.Scatter3d` for sites. Default is jus hint, you can use any keyword argument that is accepted by `plotly.graph_objects.Scatter3d`.
    plot_cell : bool, defult is True. Plot unit cell with default settings. If you want to customize, use `POSCAR.iplot_cell(fig = <return of iplot_lattice>)` function.
    
    kwargs are passed to `iplot_bz`.
    """
    poscar_data = _fix_sites(poscar_data,tol=tol,eqv_sites=eqv_sites,translate=translate)
    bond_length = _get_bond_length(poscar_data,given = bond_length)
    
    sites = None
    pos = poscar_data.positions
    if mask_sites is not None: # not None is important, as it can be False given by user
        sites = _masked_data(poscar_data,mask_sites)
        pos = poscar_data.positions[sites]
        if not sites:
            raise ValueError('No sites found with given mask_sites function.')
    
    coords, pairs = get_pairs(poscar_data, pos, r = bond_length,tol = bond_tol) # bond tolernce should be smaller than cell tolernce.
    
    if not fig:
        fig = go.Figure()

    uelems = poscar_data.types.to_dict()
    if not isinstance(sizes,(list,tuple,np.ndarray)):
        sizes = [sizes for elem in uelems.keys()]

    if colors and len(colors) != len(uelems.keys()):
        print('Warning: Number of colors does not match number of atom types. Using default colors.')

    if (colors is None) or len(colors) != len(uelems.keys()):
        colors = [_atom_colors[elem] for elem in uelems.keys()]
        colors = ['rgb({},{},{})'.format(*[int(_c*255) for _c in c]) for c in colors]

    _colors = np.array([colors[i] for i,vs in enumerate(uelems.values()) for v in vs])

    if np.any(pairs):
        coords_p = coords[pairs] #paired points
        _colors = _colors[pairs] # Colors at pairs
        coords_n = []
        colors_n = []
        for c_p, _c in zip(coords_p,_colors):
            mid = np.mean(c_p,axis=0)
            arr = np.concatenate([c_p[0],mid,mid,c_p[1]]).reshape((-1,2,3))
            coords_n = [*coords_n,*arr] # Same shape
            colors_n = [*colors_n,*_c] # same shape.

        coords_n = np.array(coords_n)
        colors_n = np.array(colors_n)
        
        bond_kws = {'line_width': 4, **bond_kws}

        for (i, cp),c in zip(enumerate(coords_n),colors_n):
            showlegend = True if i == 0 else False
            fig.add_trace(go.Scatter3d(
                x = cp[:,0].T,
                y = cp[:,1].T,
                z = cp[:,2].T,
                mode='lines',line_color = c, legendgroup='Bonds',showlegend = showlegend,
                name='Bonds',**bond_kws))
    
    site_kws = {**dict(line_color='rgba(1,1,1,0)',line_width=0.001, opacity = 1), **site_kws}
    for (k,v),c,s in zip(uelems.items(),colors,sizes):
        if sites:
            v = [i for i in v if i in sites] # Only show selected sites.
            coords = poscar_data.coords[v] 
            labs = poscar_data.labels[v]
        else:
            coords = poscar_data.coords[v]
            labs = poscar_data.labels[v]

        fig.add_trace(go.Scatter3d(
            x = coords[:,0].T,
            y = coords[:,1].T,
            z = coords[:,2].T,
            mode='markers',marker_color = c, hovertext = labs,
            marker_size = s,name = k, **site_kws))
    
    if plot_cell:
        bz_data = serializer.CellData(get_bz(path_pos = poscar_data.rec_basis, primitive=True).to_dict()) # Make cell for correct vector notations
        iplot_bz(bz_data,fig = fig, ortho3d = ortho3d, special_kpoints = False, **kwargs)
    else:
        if kwargs:
            print('Warning: kwargs are ignored as plot_cell is False.')
        # These thing are update in iplot_bz function, but if plot_cell is False, then we need to update them here.
        proj = dict(projection = dict(type = "orthographic")) if ortho3d else {}
        camera = dict(center = dict(x = 0.1, y = 0.1, z = 0.1),**proj)
        fig.update_layout(template = 'plotly_white', scene_camera = camera,
            font_family="Times New Roman",font_size= 14,
            scene = dict(aspectmode='data',
                        xaxis = dict(showbackground = False, visible = False),
                        yaxis = dict(showbackground = False, visible = False),
                        zaxis = dict(showbackground = False, visible = False)),
                        margin=dict(r=10, l=10,b=10, t=30))
    return fig

def _validate_label_func(fmt_label, parameter):
    if not callable(fmt_label):
        raise ValueError('fmt_label must be a callable function.')
    if len(inspect.signature(fmt_label).parameters.values()) != 1:
        raise ValueError('fmt_label must have only one argument.')
    
    test_out = fmt_label(parameter)
    if isinstance(test_out,(list,tuple)):
        if len(test_out) != 2:
            raise ValueError('fmt_label must return string or a list/tuple of length 2.')
        
        if not isinstance(test_out[0],str):
            raise ValueError('Fisrt item in return of `fmt_label` must return a string! got {}'.format(type(test_out[0])))
        
        if not isinstance(test_out[1],dict):
            raise ValueError('Second item in return of `fmt_label` must return a dictionary of keywords to pass to `plt.text`! got {}'.format(type(test_out[1])))
    
    elif not isinstance(test_out, str):
        raise ValueError('fmt_label must return a string or a list/tuple of length 2.')

# Cell
def splot_lattice(poscar_data, plane = None, sizes = 50, colors = None, bond_length = None,tol = 1e-2,bond_tol = 1e-3,eqv_sites = True,
    translate = None, ax = None, mask_sites = None, showlegend = True, fmt_label = None,
    site_kws = dict(alpha = 0.7), bond_kws = dict(alpha = 0.7, lw = 1),
    plot_cell = True, **kwargs):
    """
    Matplotlib Static plot of lattice.
    
    Parameters
    ----------
    poscar_data : Output of export_poscar or export_vasprun().poscar.
    plane : Plane to plot. Either 'xy','xz','yz' or None for 3D plot.
    sizes : Size of sites. Either one int/float or list equal to type of ions.
    colors : Sequence of colors for each ion type. If None, automatically generated.
    bond_length : Length of bond in fractional unit [0,1]. It is scaled to V^1/3 and auto calculated if not provides.
    alpha : Opacity of points and bonds.
    mask_sites : Provide a mask function `f(index, x,y,z) -> bool` to show only selected sites. For example, to show only sites with z > 0.5, use `mask_sites = lambda i,x,y,z: x > 0.5`.
    showlegend : bool, default is True, show legend for each ion type.
    site_kws : Keyword arguments to pass to `plt.scatter` for plotting sites. Default is just hint, you can pass any keyword argument that `plt.scatter` accepts.
    bond_kws : Keyword arguments to pass to `plt.plot` for plotting bonds. Default is just hint, you can pass any keyword argument that `plt.plot` accepts.
    fmt_label : If given, each site label is passed to it like fmt_label('Ga 1'). It must return a string or a list/tuple of length 2. First item is the label and second item is a dictionary of keywords to pass to `plt.text`.
    plot_cell : bool, default is True, plot unit cell with default settings. To customize options, use `plot_cell = False` and do `POSCAR.splot_cell(ax = <return of splot_lattice>)`.
    
    kwargs are passed to `splot_bz`.
    
    > Tip: Use `plt.style.use('ggplot')` for better 3D perception.
    """
    #Plane fix
    if plane and plane not in 'xyzxzyx':
        raise ValueError("plane expects in 'xyzxzyx' or None.")
    if plane:
        ind = 'xyzxzyx'.index(plane)
        arr = [0,1,2,0,2,1,0]
        ix,iy = arr[ind], arr[ind+1]
        
    poscar_data = _fix_sites(poscar_data,tol=tol,eqv_sites=eqv_sites,translate=translate)
    bond_length = _get_bond_length(poscar_data,given = bond_length)
    
    sites = None
    pos = poscar_data.positions # take all sites
    if mask_sites is not None: # not None is important, user can give anything
        sites = _masked_data(poscar_data,mask_sites)
        pos = poscar_data.positions[sites]
        if not sites:
            raise ValueError('No sites found with given mask_sites function.')
    
    coords, pairs = get_pairs(poscar_data,positions = pos,r = bond_length,tol = bond_tol) # bond tolernce should be smaller than cell tolernce.
    
    labels = [poscar_data.labels[i] for i in sites] if sites else poscar_data.labels
    if fmt_label is not None:
        _validate_label_func(fmt_label, labels[0])
    
    if plot_cell:
        bz_data = serializer.CellData(get_bz(poscar_data.rec_basis, primitive=True).to_dict()) # For correct vectors
        ax = splot_bz(bz_data,plane = plane, ax = ax, **kwargs)
    else:
        ax = ax or sp.get_axes(axes_3d = True if plane is None else False)
        if kwargs:
            print('Warning: kwargs are not used when plot_cell = False.')

    uelems = poscar_data.types.to_dict()
    if not isinstance(sizes,(list,tuple, np.ndarray)):
        sizes = [sizes for elem in uelems.keys()]

    if colors and len(colors) != len(uelems.keys()):
        print('Warning: Number of colors does not match number of atom types. Using default colors.')

    if (colors is None) or len(colors) != len(uelems.keys()):
        colors = [_atom_colors[elem] for elem in uelems.keys()]

    # Before doing other stuff, create something for legend.
    for (k,v),c,s in zip(uelems.items(),colors,sizes):
        ax.scatter([],[],s=s,color=c,label=k) # Works both for 3D and 2D.

    # Now change colors and sizes to whole array size
    colors = np.array([colors[i] for i,vs in enumerate(uelems.values()) for v in vs])
    sizes = np.array([sizes[i] for i,vs in enumerate(uelems.values()) for v in vs])
    
    if sites:
        colors = colors[sites]
        sizes = sizes[sites]

    if np.any(pairs):
        coords_p = coords[pairs] #paired points
        _colors = colors[pairs] # Colors at pairs
        coords_n = []
        colors_n = []
        for c_p, _c in zip(coords_p,_colors):
            mid = np.mean(c_p,axis=0)
            arr = np.concatenate([c_p[0],mid,mid,c_p[1]]).reshape((-1,2,3))
            coords_n = [*coords_n,*arr] # Same shape
            colors_n = [*colors_n,*_c] # same shape.

        coords_n = np.array(coords_n)
        colors_n = np.array(colors_n)

        bond_kws = {'alpha':0.7, **bond_kws} # bond_kws overrides alpha only
        if not plane:
            _ = [ax.plot(*c.T,c=_c,**bond_kws) for c,_c in zip(coords_n,colors_n)]
        elif plane in 'xyzxzyx':
            _ = [ax.plot(c[:,ix],c[:,iy],c=_c,**bond_kws) for c,_c in zip(coords_n,colors_n)]
    
    if not plane:
        site_kws = {**dict(alpha = 0.7, depthshade = False), **site_kws} # site_kws overrides alpha only
        ax.scatter(coords[:,0],coords[:,1],coords[:,2],c = colors ,s = sizes,**site_kws)
        if fmt_label:
            for i,coord in enumerate(coords):
                lab, textkws = fmt_label(labels[i]), {}
                if isinstance(lab, (list,tuple)):
                    lab, textkws = lab
                ax.text(*coord,lab,**textkws)
        # Set aspect to same as data.
        ax.set_box_aspect(np.ptp(bz_data.vertices,axis=0))
                
    elif plane in 'xyzxzyx':
        site_kws = {**dict(alpha = 0.7, zorder = 3), **site_kws} 
        iz, = [i for i in range(3) if i not in (ix,iy)]
        zorder = coords[:,iz].argsort()
        if plane in 'yxzy': # Left handed
            zorder = zorder[::-1]
        ax.scatter(coords[zorder][:,ix],coords[zorder][:,iy],c = colors[zorder] ,s =sizes[zorder],**site_kws)
        
        if fmt_label:
            labels = [labels[i] for i in zorder] # Reorder labels
            for i,coord in enumerate(coords[zorder]):
                lab, textkws = fmt_label(labels[i]), {}
                if isinstance(lab, (list,tuple)):
                    lab, textkws = lab
                ax.text(*coord[[ix,iy]],lab,**textkws)
        
        # Set aspect to display real shape.
        ax.set_aspect(1)

    ax.set_axis_off()
    if showlegend:
        sp.add_legend(ax)
    return ax

# Cell
def join_poscars(poscar_data,other,direction='c',tol=1e-2, system = None):
    """Joins two POSCARs in a given direction. In-plane lattice parameters are kept from first poscar and out of plane basis vector of other is modified while volume is kept same.
    
    Parameters
    ----------
    poscar_data :  Base POSCAR. Output of `export_poscar` or similar object from other functions.
    other : Other POSCAR to be joined with this POSCAR.
    direction : The joining direction. It is general and can join in any direction along basis. Expect one of ['a','b','c'].
    tol : Default is 0.01. It is used to bring sites near 1 to near zero in order to complete sites in plane. Vasp relaxation could move a point, say at 0.00100 to 0.99800 which is not useful while merging sites.
    system : If system is given, it is written on top of file. Otherwise, it is infered from atomic species.
    """
    _poscar1 = _fix_sites(poscar_data,tol = tol,eqv_sites = False)
    _poscar2 = _fix_sites(other,tol = tol,eqv_sites = False)
    pos1 = _poscar1.positions.copy()
    pos2 = _poscar2.positions.copy()

    s1,s2 = 0.5, 0.5 # Half length for each.
    a1,b1,c1 = np.linalg.norm(_poscar1.basis,axis=1)
    a2,b2,c2 = np.linalg.norm(_poscar2.basis,axis=1)
    basis = _poscar1.basis.copy() # Must be copied, otherwise change outside.

    # Processing in orthogonal space since a.(b x c) = abc sin(theta)cos(phi), and theta and phi are same for both.
    if direction in 'cC':
        c2 = (a2*b2)/(a1*b1)*c2 # Conservation of volume for right side to stretch in c-direction.
        netc = c1+c2
        s1, s2 = c1/netc, c2/netc
        pos1[:,2] = s1*pos1[:,2]
        pos2[:,2] = s2*pos2[:,2] + s1
        basis[2] = netc*basis[2]/np.linalg.norm(basis[2]) #Update 3rd vector

    elif direction in 'bB':
        b2 = (a2*c2)/(a1*c1)*b2 # Conservation of volume for right side to stretch in b-direction.
        netb = b1+b2
        s1, s2 = b1/netb, b2/netb
        pos1[:,1] = s1*pos1[:,1]
        pos2[:,1] = s2*pos2[:,1] + s1
        basis[1] = netb*basis[1]/np.linalg.norm(basis[1]) #Update 2nd vector

    elif direction in 'aA':
        a2 = (b2*c2)/(b1*c1)*a2 # Conservation of volume for right side to stretch in a-direction.
        neta = a1+a2
        s1, s2 = a1/neta, a2/neta
        pos1[:,0] = s1*pos1[:,0]
        pos2[:,0] = s2*pos2[:,0] + s1
        basis[0] = neta*basis[0]/np.linalg.norm(basis[0]) #Update 1st vector

    else:
        raise Exception("direction expects one of ['a','b','c']")

    scale = np.linalg.norm(basis[0])
    u1 = _poscar1.types.to_dict()
    u2 = _poscar2.types.to_dict()
    u_all = ({**u1,**u2}).keys() # Union of unique atom types to keep track of order.


    pos_all = []
    i_all = []
    for u in u_all:
        _i_ = 0
        if u in u1.keys():
            _i_ = len(u1[u])
            pos_all = [*pos_all,*pos1[u1[u]]]
        if u in u2.keys():
            _i_ = _i_ + len(u2[u])
            pos_all = [*pos_all,*pos2[u2[u]]]
        i_all.append(_i_)

    i_all = np.cumsum([0,*i_all]) # Do it after labels
    uelems = {_u:range(i_all[i],i_all[i+1]) for i,_u in enumerate(u_all)}
    sys = system or ''.join(uelems.keys())
    iscartesian = poscar_data.metadata.cartesian or other.metadata.cartesian
    metadata = {'cartesian':iscartesian, 'scale': scale, 'comment': 'Modified by ipyvasp'}
    out_dict = {'SYSTEM':sys,'basis':basis,'metadata':metadata,'positions':np.array(pos_all),'types':uelems}
    return serializer.PoscarData(out_dict)


# Cell
def repeat_poscar(poscar_data, n, direction):
    """Repeat a given POSCAR.
    Args:
        - path_poscar: Path/to/POSCAR or `poscar` data object.
        - n: Number of repetitions.
        - direction: Direction of repetition. Can be 'a', 'b' or 'c'.
    """
    if not isinstance(n, (int, np.integer)) and n < 2:
        raise ValueError("n must be an integer greater than 1.")
    given_poscar = poscar_data
    for i in range(1,n):
        poscar_data = join_poscars(given_poscar, poscar_data,direction = direction)
    return poscar_data

def scale_poscar(poscar_data,scale = (1,1,1),tol=1e-2):
    """Create larger/smaller cell from a given POSCAR. Can be used to repeat a POSCAR with integer scale values.
    Args:
        - poscar_data: `poscar` data object.
        - scale: Tuple of three values along (a,b,c) vectors. int or float values. If number of sites are not as expected in output, tweak `tol` instead of `scale`. You can put a minus sign with `tol` to get more sites and plus sign to reduce sites.
        - tol: It is used such that site positions are blow `1 - tol`, as 1 belongs to next cell, not previous one.
    **Tip:** scale = (2,2,2) enlarges a cell and next operation of (1/2,1/2,1/2) should bring original cell back.
    **Caveat:** A POSACR scaled with Non-integer values should only be used for visualization purposes, Not for any other opration such as making supercells, joining POSCARs.
    """
    ii, jj, kk = np.ceil(scale).astype(int) # Need int for joining.

    if tuple(scale) == (1,1,1): # No need to scale.
        return poscar_data

    if ii >= 2:
        poscar_data = repeat_poscar(poscar_data,ii,direction='a')

    if jj >= 2:
        poscar_data = repeat_poscar(poscar_data,jj,direction='b')

    if kk >= 2:
        poscar_data = repeat_poscar(poscar_data,kk,direction='c')

    if np.all([s == int(s) for s in scale]):
        return poscar_data # No need to prcess further in case of integer scaling.

    new_poscar = poscar_data.to_dict() # Update in it

    # Get clip fraction
    fi, fj, fk = scale[0]/ii, scale[1]/jj, scale[2]/kk

    # Clip at end according to scale, change length of basis as fractions.
    pos   = poscar_data.positions.copy()/np.array([fi,fj,fk]) # rescale for clip
    basis = poscar_data.basis.copy()
    for i,f in zip(range(3),[fi,fj,fk]):
        basis[i] = f*basis[i] # Basis rescale for clip

    new_poscar['basis'] = basis
    new_poscar['metadata']['scale'] = np.linalg.norm(basis[0])
    new_poscar['metadata']['comment'] = f'Modified by ipyvasp'

    uelems = poscar_data.types.to_dict()
    # Minus in below for block is because if we have 0-2 then 1 belongs to next cell not original.
    positions,shift = [],0
    for key,value in uelems.items():
        s_p = pos[value] # Get positions of key
        s_p = s_p[(s_p < 1 - tol).all(axis=1)] # Get sites within tolerance

        if len(s_p) == 0:
            raise Exception(f'No sites found for {key!r}, cannot scale down. Increase scale!')

        uelems[key] = range(shift,shift + len(s_p))
        positions = [*positions,*s_p] # Pick sites
        shift += len(s_p) #Update for next element

    new_poscar['types']    = uelems
    new_poscar['positions'] = np.array(positions)
    return serializer.PoscarData(new_poscar)

def rotate_poscar(poscar_data,angle_deg,axis_vec):
    """Rotate a given POSCAR.
    Args:
        - path_poscar: Path/to/POSCAR or `poscar` data object.
        - angle_deg: Rotation angle in degrees.
        - axis_vec : (x,y,z) of axis about which rotation takes place. Axis passes through origin.
    """
    rot = rotation(angle_deg = angle_deg,axis_vec = axis_vec)
    p_dict = poscar_data.to_dict()
    p_dict['basis'] = rot.apply(p_dict['basis']) # Rotate basis so that they are transpose
    p_dict['metadata']['comment'] = f'Modified by ipyvasp'
    return serializer.PoscarData(p_dict)

def set_zdir(poscar_data, hkl, phi = 0):
    """
    Set z-direction of POSCAR along a given hkl direction and returns new data.
    
    Parameters
    ----------
    path_poscar : Path/to/POSCAR or `poscar` data object.
    hkl : (h,k,l) of the direction along which z-direction is to be set. Vector is constructed as h*a + k*b + l*c in cartesian coordinates.
    phi: Rotation angle in degrees about z-axis to set a desired rotated view.
    
    Returns
    -------
    New instance of poscar with z-direction set along hkl.
    """
    if not isinstance(hkl, (list, tuple, np.ndarray)) and len(hkl) != 3:
        raise ValueError("hkl must be a list, tuple or numpy array of length 3.")
    
    p_dict = poscar_data.to_dict()
    basis = p_dict['basis']
    zvec = to_R3(basis, [hkl])[0] # in cartesian coordinates
    angle = np.arccos(zvec.dot([0,0,1])/np.linalg.norm(zvec)) # Angle between zvec and z-axis
    rot = rotation(angle_deg = np.rad2deg(angle), axis_vec = np.cross(zvec,[0,0,1])) # Rotation matrix
    new_basis = rot.apply(basis) # Rotate basis so that zvec is along z-axis
    p_dict['basis'] = new_basis
    p_dict['metadata']['comment'] = f'Modified by ipyvasp'
    new_pos = serializer.PoscarData(p_dict)
    
    if phi:
        return rotate_poscar(new_pos,angle_deg = phi,axis_vec = [0,0,1]) # Rotate around z-axis 
        
    return new_pos

def mirror_poscar(poscar_data, direction):
    "Mirror a POSCAR in a given direction. Sometime you need it before joining two POSCARs"
    poscar = poscar_data.to_dict() # Avoid modifying original
    idx = 'abc'.index(direction) # Check if direction is valid
    poscar['positions'][:,idx] = 1 - poscar['positions'][:,idx] # Trick: Mirror by subtracting from 1. not by multiplying with -1.
    return serializer.PoscarData(poscar) # Return new POSCAR

def convert_poscar(poscar_data, atoms_mapping, basis_factor):
    """Convert a POSCAR to a similar structure of other atomic types or same type with strained basis.
    `atoms_mapping` is a dictionary of {old_atom: new_atom} like {'Ga':'Al'} will convert GaAs to AlAs structure.
    `basis_factor` is a scaling factor multiplied with basis vectors, single value (useful for conversion to another type)
    or list of three values to scale along (a,b,c) vectors (useful for strained structures).
    """
    poscar_data = poscar_data.to_dict() # Avoid modifying original
    poscar_data['types'] = {atoms_mapping.get(k,k):v for k,v in poscar_data['types'].items()} # Update types
    basis = poscar_data['basis'].copy() # Get basis to avoid modifying original

    if isinstance(basis_factor,(int, np.integer, float)):
        poscar_data['basis'] = basis_factor*basis # Rescale basis
    elif isinstance(basis_factor,(list,tuple,np.ndarray)):
        if len(basis_factor) != 3:
            raise Exception('basis_factor should be a list/tuple/array of length 3')

        if np.ndim(basis_factor) != 1:
            raise Exception('basis_factor should be a list/tuple/array of 3 int/float values')

        poscar_data['basis'] = np.array([
            basis_factor[0]*basis[0],
            basis_factor[1]*basis[1],
            basis_factor[2]*basis[2]
        ])
    else:
        raise Exception('basis_factor should be a list/tuple/array of 3 int/float values, got {}'.format(type(basis_factor)))

    return serializer.PoscarData(poscar_data) # Return new POSCAR

def get_TM(basis1, basis2):
    """Returns a transformation matrix that gives `basis2` when applied on `basis1`.
    basis are 3x3 matrices with basis vectors as rows like [[b1x, b1y, b1z],[b2x, b2y, b2z],[b3x, b3y, b3z]].
    
    ```python
    TM = get_TM(basis1, basis2)
    assert np.allclose(basis2, TM @ basis1)
    Q = P @ TM.T # Transform points from P in basis1 to Q in basis2
    # Both P and Q are N x D matrices where N is number of points and D is dimension of space
    ```
    """
    return to_basis(basis2, basis1) # basis1 in basis2 is simply the transformation matrix

def transform_poscar(poscar_data, transformation, zoom = 2, tol = 1e-2):
    """Transform a POSCAR with a given transformation matrix or function that takes old basis and return target basis.
    Use `get_TM(basis1, basis2)` to get transformation matrix from one basis to another or function to return new basis of your choice.
    An example of transformation function is `lambda a,b,c: a + b, a-b, c` which will give a new basis with a+b, a-b, c as basis vectors.
    
    You may find errors due to missing atoms in the new basis, use `zoom` to increase the size of given cell to include any possible site in new cell.
    
    Examples:
    - FCC primitive -> 111 hexagonal cell  
        lambda a,b,c: (a-c,b-c,a+b+c) ~ [[1,0,-1],[0,1,-1],[1,1,1]]
    - FCC primitive --> FCC unit cell 
        lambda a,b,c: (b+c -a,a+c-b,a+b-c) ~ [[-1,1,1],[1,-1,1],[1,1,-1]]
    - FCC unit cell --> 110 tetragonal cell
        lambda a,b,c: (a-b,a+b,c) ~ [[1,-1,0],[1,1,0],[0,0,1]]
    """
    if callable(transformation):
        new_basis = np.array(transformation(*poscar_data.basis)) # mostly a tuple
        if new_basis.shape != (3,3):
            raise Exception('transformation function should return a tuple equivalent to 3x3 matrix')
    elif np.ndim(transformation) == 2 and np.shape(transformation) == (3,3):
        new_basis = np.matmul(transformation,poscar_data.basis)
    else:
        raise Exception('transformation should be a function that accept 3 arguemnts or 3x3 matrix')
    
    def get_cell_vertices(basis):
        verts = get_bz(np.linalg.inv(basis).T,primitive = True).vertices
        return verts - np.mean(verts, axis = 0) # Center around origin, otherwise it never going to be inside convex hull
    
    TM = get_TM(poscar_data.basis, new_basis) # Transformation matrix to save before scaling
    old_numbers = [len(v) for v in poscar_data.types.to_dict().values()]
    chull = ConvexHull(np.max([zoom,2])*get_cell_vertices(new_basis)) # Convex hull of new cell, make it at least two times to avoid losing sites because of translation
    
    while any(inside_convexhull(chull, get_cell_vertices(poscar_data.basis))): # Check if all vertices of old cell are inside new cell
        poscar_data = scale_poscar(poscar_data, [2,2,2],tol = tol) # Repeat in all directions
            
    points = to_basis(new_basis,poscar_data.coords) # Transform coordinates to new basis around origin
    points = points - np.mean(points, axis = 0) # Center around origin, to include all
    
    new_poscar = poscar_data.to_dict() # Update in it
    new_poscar['basis'] = new_basis
    new_poscar['metadata']['scale'] = np.linalg.norm(new_basis[0])
    new_poscar['metadata']['comment'] = f'Transformed by ipyvasp'
    new_poscar['metadata']['TM'] = TM # Save transformation matrix in both function and matrix given

    uelems = poscar_data.types.to_dict()
    positions,shift, unique_dict = [],0, {}
    for key,value in uelems.items():
        s_p = points[value]
        s_p = s_p[((s_p > -tol) & (s_p < 1 - tol)).all(axis = 1)] # Get sites within tolerance, for very far sites

        if s_p.size == 0:
            raise Exception(f'No sites found for {key!r}, transformation stopped! You may need to modify `transformation` or increase `zoom` value.')

        unique_dict[key] = range(shift,shift + len(s_p))
        positions = [*positions,*s_p] # Pick sites
        shift += len(s_p) #Update for next element
        
    # Final check if crystal is still same
    new_numbers = [len(v) for v in unique_dict.values()]
    ratio = [round(new/old,4) for new,old in zip(new_numbers, old_numbers)] # Round to avoid floating point errors,can cover 1 to 10000 atoms transformation
    if len(np.unique(ratio)) != 1:
        raise Exception(f'Transformation failed, atoms proportion changed: {old_numbers} -> {new_numbers}, if your transformation is an allowed one for this structure, increase `zoom` value.')
    
    new_poscar['types']  = unique_dict
    new_poscar['positions'] = np.array(positions)
    return serializer.PoscarData(new_poscar)

def add_vaccum(poscar_data, thickness, direction, left = False):
    """Add vacuum to a POSCAR.
    Args:
        - poscar_data: `poscar` data object.
        - thickness: Thickness of vacuum in Angstrom.
        - direction: Direction of vacuum. Can be 'a', 'b' or 'c'.
        - left: If True, vacuum is added to left of sites. By default, vacuum is added to right of sites.
    """
    if direction not in 'abc':
        raise Exception('Direction must be a, b or c.')

    poscar_dict = poscar_data.to_dict() # Avoid modifying original
    basis = poscar_dict['basis'].copy() # Copy basis to avoid modifying original
    pos = poscar_dict['positions'].copy() # Copy positions to avoid modifying original
    idx = 'abc'.index(direction)
    norm = np.linalg.norm(basis[idx]) # Get length of basis vector
    s1, s2 = norm/(norm + thickness), thickness/(norm + thickness) # Get scaling factors
    basis[idx,:] *= (thickness + norm)/norm # Add thickness to basis
    poscar_dict['basis'] = basis
    if left:
        pos[:,idx] *= s2 # Scale down positions
        pos[:,idx] += s1 # Add vacuum to left of sites
        poscar_dict['positions'] = pos
    else:
        pos[:,idx] *= s1 # Scale down positions
        poscar_dict['positions'] = pos

    return serializer.PoscarData(poscar_dict) # Return new POSCAR


# Cell
def transpose_poscar(poscar_data, axes = [1,0,2]):
    "Transpose a POSCAR by switching basis from [0,1,2] -> `axes`. By Default, x and y are transposed."
    if isinstance(axes,(list,tuple, np.ndarray)) and len(axes) == 3:
        if not all(isinstance(i,(int,np.integer)) for i in axes):
            raise ValueError('`axes` must be a list of three integers.')

        poscar_data = poscar_data.to_dict() #
        basis = poscar_data['basis'].copy() # Copy basis to avoid modifying original
        positions = poscar_data['positions'].copy() # Copy positions to avoid modifying original
        poscar_data['basis'] = basis[axes] # Transpose basis
        poscar_data['positions'] = positions[:,axes] # Transpose positions
        return serializer.PoscarData(poscar_data) # Return new POSCAR
    else:
        raise Exception('`axes` must be a squence of length 3.')

def add_atoms(poscar_data, name, positions):
    "Add atoms with a `name` to a POSCAR at given `positions` in fractional coordinates."
    if name in poscar_data.types.keys():
        raise Exception(f'{name!r} already exists in POSCAR. Cannot add duplicate atoms.')

    positions = np.array(positions)
    if (not np.ndim(positions) == 2) or (not positions.shape[1] == 3):
        raise ValueError('`positions` must be a 2D array of shape (n,3)')

    new_pos = np.vstack([poscar_data.positions,positions]) # Add new positions to existing ones

    unique = poscar_data.types.to_dict() # Copy unique dictionary to avoid modifying original
    unique[name] = range(len(poscar_data.positions),len(new_pos)) # Add new unique element

    data = poscar_data.to_dict() # Copy data to avoid modifying original
    data['types'] = unique # Update unique dictionary
    data['positions'] = new_pos # Update positions
    data['SYSTEM'] = f'{data["SYSTEM"]}+{name}' # Update SYSTEM
    data['metadata']['comment'] = f'{data["metadata"]["comment"]} + Added {name!r}' # Update comment

    return serializer.PoscarData(data) # Return new POSCAR

def strain_poscar(poscar_data, strain_matrix):
    "Strain a POSCAR by a given 3x3 `strain_matrix` to be multiplied with basis (elementwise) and return a new POSCAR."
    if not isinstance(strain_matrix,np.ndarray):
        strain_matrix = np.array(strain_matrix)
        
    if strain_matrix.shape != (3,3):
        raise ValueError('`strain_matrix` must be a 3x3 matrix to multiply with basis.')
    
    poscar_data = poscar_data.to_dict() #
    poscar_data['basis'] = poscar_data['basis'] * strain_matrix# Update basis by elemetwise multiplication
    poscar_data['metadata']['comment'] = f'{poscar_data["metadata"]["comment"]} + Strained POSCAR' # Update comment
    return serializer.PoscarData(poscar_data) # Return new POSCAR

def view_poscar(poscar_data, **kwargs):
    "View a POSCAR in a jupyter notebook. kwargs are passed to splot_lattice. After setting a view, you can do view.f(**view.kwargs) to get same plot in a cell."
    def view(elev = 30, azim = 30, roll = 0):
        ax = splot_lattice(poscar_data, **kwargs)
        ax.view_init(elev = elev, azim = azim, roll = roll)
        
    return interactive(view, elev = (0,180), azim = (0,360), roll=(0,360))
    
        
    
    
    