# Testing a flexible command line interface for ipyvasp
from typing import List
from pathlib import Path

import typer
from typing_extensions import Annotated

from .core.parser import minify_vasprun
from .lattice import POSCAR, get_kpath
from .utils import _sig_kwargs

app = typer.Typer(no_args_is_help=True)
poscar_app = typer.Typer(no_args_is_help=True)
app.add_typer(poscar_app, name="poscar", help="POSCAR related operations")
vasprun_app = typer.Typer(no_args_is_help=True)
app.add_typer(vasprun_app, name="vasprun", help="vasprun.xml related operations")


@app.callback()
def callback():
    """
    A flexible command line interface for ipyvasp.
    """


@poscar_app.command("view")
def poscar_view(
    path: str = "POSCAR", width: int = 144, nocolor: bool = False, invert: bool = False
):
    from .core.plot_toolkit import plt2text

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
def _get_kpath(kpoints: str, n: int = 5, poscar: str = "POSCAR", **kwargs):
    """kpoints should be a multiline string or a file consist of kpoints. See POSCAR.get_kpath for more details.
    If poscar file is not found, number of kpoints in each seqgment will be same unless specifies per segment.
    """
    from pathlib import Path

    if Path(poscar).is_file():
        return POSCAR(poscar).get_kpath(kpoints, n=n, **kwargs)
    else:
        return get_kpath(kpoints, n=n, **kwargs)


vasprun_app.command("minify")(minify_vasprun)


@vasprun_app.command("get-gap")
def get_gap(glob: List[Path]):
    from .core.parser import Vasprun
    from .widgets import summarize

    def gap_summary(path):
        gap = Vasprun(path).bands.gap
        delattr(gap, "coords")  # remove coords info
        d = gap.to_dict()
        d["kvbm"] = "(" + ",".join(str(round(k, 4)) for k in d["kvbm"]) + ")"
        d["kcbm"] = "(" + ",".join(str(round(k, 4)) for k in d["kcbm"]) + ")"
        return d

    name = str(Path(".").absolute())
    print("\n", name, "\n", "=" * len(name), "\n")
    print(summarize(glob, gap_summary).to_string())  # make all data visible


@vasprun_app.command("get-summary")
def get_summary(glob: List[Path]):
    from .core.parser import Vasprun

    for path in glob:
        name = str(path.absolute())
        print("\n", name, "\n", "=" * len(name))
        print(Vasprun(path).summary)


@app.command("set-dir")
def _set_dir(glob: List[Path], command: str = ""):
    from platform import system
    from subprocess import Popen
    from .utils import set_dir

    os = system()  # operating system

    dirs = [f.absolute() for f in glob if f.is_dir()]  # only dirs
    if not dirs:
        raise RuntimeError(
            "Provided paths do not exist or are not directories. Exiting..."
        )

    for path in dirs:
        with set_dir(path) as p:
            print("Working in -> ", p)  # to give info about the cwd
            if os == "Windows":
                try:
                    p = Popen("pwsh.exe -NoProfile -c " + command, shell=False)
                except:
                    p = Popen("powershell.exe -NoProfile -c " + command, shell=False)

            else:
                p = Popen(command, shell=False)  # Linux, MacOS

            p.wait()
            if p.returncode != 0:
                raise RuntimeError(f"Command {command} failed in {path}. Exiting...")
