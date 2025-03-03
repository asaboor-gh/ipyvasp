__all__ = ["parse_text", "get_bib", "get_E0", "OUTCAR"]

import requests
from pathlib import Path
from itertools import islice
import numpy as np

from .core import parser as vp
from .utils import _sig_kwargs, _sub_doc


@_sig_kwargs(vp.gen2numpy, skip_params=("gen",))
@_sub_doc(
    vp.gen2numpy,
    {"gen :.*shape :": "path : Path to file containing data.\nshape :"},
)
def parse_text(path, shape, slice, **kwargs):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File {path!r} does not exists")

    with p.open("r", encoding="utf-8") as f:
        gen = islice(f, 0, None)
        data = vp.gen2numpy(
            gen, shape, slice, **kwargs
        )  # should be under open file context
    return data

def get_bib(DOI):
    "Get bibligraphy entry from DOI"
    response = requests.get(
        f'http://dx.doi.org/{DOI}', 
        headers={'Accept':'text/bibliography;style=bibtex'})
    if response.status_code == 200:
        return response.content.decode('utf-8')
    
    return response.status_code

def get_E0(path: Path = 'OSZICAR',nsteps=0):
    "Get the E0 from the last line of OSZICAR. If `nsteps > 0`, returns as many available last steps ."
    fh = Path(path)
    if fh.is_file():
        lines = fh.read_text().splitlines()
    else:
        raise FileNotFoundError(f"File: {path!r} does not exists!")
    if not lines:
        raise ValueError(f"File: {path!r} is empty!")
    
    line = lines[-1]
    if 'F=' not in line:
        raise ValueError(f"File: {path!r} is not a valid OSZICAR file or calculation is not finished!")
    
    if not line.lstrip().startswith('1'):
        print(f"Calculation may not be converged for {path!r}\n{line}")
    
    if nsteps > 0: # not allow negative ones
        lines = [float(line.split()[2]) for line in lines if (not 'dE' in line) and (not 'F=' in line)]
        return np.array(lines[-nsteps:]) # as many last steps
        
    return float(line.split('=')[1].split()[0])


class OUTCAR:
    "Parse some required data from OUTCAR file."

    def __init__(self, path=None):
        self._path = Path(path or "OUTCAR")
        self._data = vp.export_outcar(self._path)

    @property
    def data(self):
        return self._data

    @property
    def path(self):
        return self._path

    @_sub_doc(vp.read)
    def read(self, start_match, stop_match=r'\n', nth_match=1, skip_last=False,apply=None):
        kws = {k:v for k,v in locals().items() if k !='self'}
        return vp.read(self.path, **kws)  # Pass all the arguments to the function
        
