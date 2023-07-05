# Testing a flexible command line interface for ipyvasp

import os
import typer
from . import minify_vasprun
from . import POSCAR, plt2text, get_kpath
from .utils import _sig_kwargs

app = typer.Typer(no_args_is_help=True)
poscar_app = typer.Typer(no_args_is_help=True)
app.add_typer(poscar_app, name="poscar", help="POSCAR related operations")


@app.callback()
def callback():
    """
    A flexible command line interface for ipyvasp.
    """


@poscar_app.command("view")
def poscar_view(
    path: str = "POSCAR", width: int = 144, nocolor: bool = False, invert: bool = False
):
    poscar = POSCAR(path)  # it prints to stdout
    poscar.splot_lattice()
    plt2text(width=width, crop=True, colorful=not nocolor, invert=invert)


@poscar_app.command("download")
@_sig_kwargs(POSCAR.download, ("self",))
def download_poscar(
    formula: str, mp_id: str, api_key: str = None, save_key: bool = False
):
    return POSCAR.download(formula, mp_id, api_key=api_key, save_key=save_key).write()


@app.command("get-kpath")
@_sig_kwargs(POSCAR.get_kpath, ("self",))
def _get_kpath(kpoints, n: int = 5, poscar: str = "POSCAR", **kwargs):
    """kpoints should be a multiline string. See POSCAR.get_kpath for more details.
    If poscar file is not found, number of kpoints in each seqgment will be same unless specifies per segment.
    """
    if os.path.isfile(poscar):
        return POSCAR(poscar).get_kpath(kpoints, n=n, **kwargs)
    else:
        return get_kpath(kpoints, n=n, **kwargs)


app.command("minify-vasprun")(minify_vasprun)
