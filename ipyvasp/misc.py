__all__ = ["parse_text", "OUTCAR"]

from pathlib import Path
from itertools import islice

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

    @_sub_doc(vp.Vasprun.read)
    @_sig_kwargs(vp.Vasprun.read, ("self",))
    def read(self, start_match, stop_match, **kwargs):
        return vp.Vasprun.read(
            self, start_match, stop_match, **kwargs
        )  # Pass all the arguments to the function
