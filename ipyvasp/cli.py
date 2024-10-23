# Testing a flexible command line interface for ipyvasp
from typing import List
from pathlib import Path

import typer
from typing_extensions import Annotated


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
    from .core.parser import minify_vasprun

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
    from .core.serializer import dump

    output = []
    for path in files:
        summary = {'file': str(path.absolute()), **Vasprun(path).summary.to_dict()}
        output.append(dump(summary, format="json", indent=2))
        
    print("[" + ',\n'.join(output) + "]")  # make all data json compatible

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
    time_interval: Annotated[int, typer.Option("-t",'--time-interval')] = 0,
    extra_command: Annotated[str, typer.Option("-e", "--extra-command")] = "",
):
    """Set multiple directories like a for loop to execute a shell command within each of them.
    It will raise an error if the command fails in any of the directories. 
    To ignore the error and keep running in other directiories in sequence, use -i/--ignore-error. 
    It will raise the shell errors but python will go through all the directories.

    To keep repeating a command after some time interval, use -t/--time-interval (seconds). Only works for a command.
    Use -e/--extra-command to run (in pwd) before looping over directories. This can be used to cleanup (previous results etc.) with each time recursion.
    
    Examples:
    
    ipyvasp set-dir /path/to/dir1 /path/to/dir2 -c "echo $PWD" # print dir1 and dir2 paths
    
    ipyvasp set-dir /path/to/dir*[1-2] -c "echo $PWD" # same as above but using glob
    """
    from platform import system
    from subprocess import Popen
    from time import sleep
    from .utils import set_dir, color

    os = system()  # operating system

    dirs = [f for f in paths if f.is_dir()]  # only dirs
    
    if not dirs:
        raise RuntimeError(
            "Provided paths do not exist or are not directories. Exiting..."
        )
    
    abs_paths = [f.absolute() for f in dirs] # absolute path is must but after filtering
    
    def exec_cmd(command, path, ignore_error=ignore_error):
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
                raise RuntimeError(f"Command {command} failed in {path}. Exiting...\n" +
                "Use -i or --ignore-error switch to suppress error and continue in other directories silently."
                )

    def run(abs_paths, dirs, command, ec=extra_command):
        if ec:
            exec_cmd(ec, Path('.').absolute())
            print(color.bb("\n"+f" {command} ".center(80,"-")))

        for path, d in zip(abs_paths,dirs):
            with set_dir(path):
                print(color.gb(f"📁  {str(d)!r}"))
                exec_cmd(command, path)
                
    if time_interval > 0 and command: # repeat only if command given
        while True: # keep going until interrupted
            run(abs_paths, dirs, command)
            sleep(time_interval)
    else:
        run(abs_paths, dirs, command)

@app.command('get-bib')
def _get_bib(doi: List[str]):
    "Get bibliography entries from doi. Can provide many DOIs."
    from .misc import get_bib
    for d in doi: # all of them
        print(get_bib(d))
