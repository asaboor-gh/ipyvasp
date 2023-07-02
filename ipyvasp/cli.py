# Testing a flexible command line interface for ipyvasp
# Find a better software for this purpose.

import sys
from . import minify_vasprun as minify_vasprun


def main():
    print("Welcome to ipyvasp command line interface!\n")
    args = sys.argv[1:]
    print(sys.argv[0] + " " + " ".join(args))
    if args and args[0].startswith("poscar"):
        raise NotImplementedError("poscar is not implemented yet.")
    elif args and args[0] == "minify_vasprun":
        path = args[1] if len(args) > 1 else "vasprun.xml"
        minify_vasprun(path)
