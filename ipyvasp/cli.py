# Testing a flexible command line interface for ipyvasp
from typing import List
from pathlib import Path

import typer
from typing_extensions import Annotated

from .core.parser import minify_vasprun
from .lattice import POSCAR, get_kpath
from .utils import _sig_kwargs

app_kwargs = dict(no_args_is_help=True, pretty_exceptions_show_locals=False)

app = typer.Typer(**app_kwargs)
poscar_app = typer.Typer(**app_kwargs)
app.add_typer(poscar_app, name="poscar", help="POSCAR related operations")
vasprun_app = typer.Typer(**app_kwargs)
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


@vasprun_app.command("minify")
def minify(files: List[Path]):
    "Remove projected data from vasprun.xml file to reduce file size."
    for file in files:
        minify_vasprun(file)


@vasprun_app.command("get-gap")
def get_gap(files: List[Path]):
    "Get band gap information as table."
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
    print(summarize(files, gap_summary).to_string())  # make all data visible


@vasprun_app.command("get-summary")
def get_summary(files: List[Path]):
    "Get summary of calculation output."
    from .core.parser import Vasprun

    for path in files:
        name = str(path.absolute())
        print("\n", name, "\n", "=" * len(name))
        print(Vasprun(path).summary)

@app.command("get-E0")
def _get_E0(files: List[Path]):
    "Get the E0 from the last line of OSZICAR file(s)."
    from .misc import get_E0
    from .widgets import summarize
    
    print(summarize(files, lambda path: {'E0': get_E0(path)}).to_string())


@app.command("set-dir")
def _set_dir(
    paths: List[Path], command: Annotated[str, typer.Option("-c", "--command")] = "",
    ignore_error: Annotated[bool, typer.Option("-i", "--ignore-error")] = False,
    exclude: Annotated[str, typer.Option("-e", "--exclude", help="Regex to exclude directories")] = "",
):
    """Set multiple directories like a for loop to execute a shell command within each of them.
    It will raise an error if the command fails in any of the directories. 
    To ignore the error and keep running in other directiories in sequence, use -i/--ignore-error. 
    It will raise the shell errors but python will go through all the directories.
    To exclude certain directories, use -e/--exclude. It will exclude the directories that match the regex pattern.
    Regex pattern is applied to relative path of the directory only.
    """
    import re
    from platform import system
    from subprocess import Popen
    from .utils import set_dir, color

    os = system()  # operating system

    dirs = [f for f in paths if f.is_dir()]  # only dirs
    if exclude:
        dirs = [f for f in dirs if not re.search(exclude, str(f))]  # exclude
    
    if not dirs:
        raise RuntimeError(
            "Provided paths do not exist or are not directories. Exiting..."
        )
    
    abs_paths = [f.absolute() for f in dirs] # absolute path is must but before filtering
    
    for path, d in zip(abs_paths,dirs):
        with set_dir(path):
            print(color.gb(f"📁 {str(d)!r}"))
            
            if command:
                if os == "Windows":
                    try:
                        p = Popen("pwsh.exe -NoProfile -c " + command, shell=False)
                    except:
                        p = Popen("powershell.exe -NoProfile -c " + command, shell=False)

                else:
                    p = Popen(command, shell=True)  # Linux, MacOS, shell to get args

                p.wait()
                if not ignore_error and p.returncode != 0:
                    raise RuntimeError(f"Command {command} failed in {path}. Exiting...")
