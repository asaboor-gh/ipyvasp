import os, sys
from shutil import rmtree

from setuptools import setup, Command, find_packages

if sys.version_info[:2] < (3, 9):
    sys.exit("Sorry, Python < 3.9 is not supported for ipyvasp")

# Package meta-data.
NAME = "ipyvasp"
DESCRIPTION = (
    "A processing tool for VASP DFT input/output processing in Jupyter Notebook."
)
URL = "https://github.com/massgh/ipyvasp"
EMAIL = "mass_qau@outlook.com"
AUTHOR = "Abdul Saboor"
REQUIRES_PYTHON = ">=3.9"


# What packages are required for this module to be executed?
REQUIRED = [
    "matplotlib>=3.7.5",
    "numpy>=1.23.2",
    "scipy>=1.9.1",
    "ipywidgets>=8.0.4",
    "pillow>=9.3.0",
    "pandas>=1.4.4",
    "plotly>=6.2.0",
    "requests>=2.28.1",
    "typer>=0.9.0",
    "einteract", # any latest version of einteract 
    "sympy",
]

# What packages are optional?
EXTRAS = {
    "extra": ["jupyterlab>=3.5.2", "ipython>=9.0", "ase>=3.22.1", "nglview>=3.0.4"],
}

KEYWORDS = ["Jupyter", "Widgets", "IPython", "VASP", "DFT"]
PROJECT_URLS = {
    "Bug Tracker": "https://github.com/massgh/ipyvasp/issues",
}

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
with open(os.path.join(here, NAME, "_version.py")) as f:
    exec(f.read(), about)


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        print(f'\n\033[1;92m{s}\033[0m\n{"-" * 50}')

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous dist …")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution …")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine …")
        os.system("twine upload dist/*")

        yes_no = input("Upload this version to GitHub? [y/n]: ")
        if yes_no.lower() == "y":
            self.status("Pushing git tags…")
            os.system("git tag v{0}".format(about["__version__"]))
            os.system("git push --tags")

        sys.exit()

class BuildDocsCommand(UploadCommand):
    description = "Build docs using sphinx-build."
    user_options = []

    def run(self):
        source = os.path.join(here, "docs","source")
        build = os.path.join(here, "docs","build", "html")
        os.makedirs(build, exist_ok=True)
        ret = os.system(f"sphinx-build -b html {source} {build}")
        if ret != 0:
            raise RuntimeError("Sphinx docs build failed!")
        print(f"✅ Documentation successfully built as {build}")

# Where the magic happens:
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="MIT",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    keywords=KEYWORDS,
    project_urls=PROJECT_URLS,
    # $ setup.py publish support.
    cmdclass={
        "upload": UploadCommand,
        "build_docs": BuildDocsCommand,
    },
    # for command line interface
    entry_points={
        "console_scripts": [
            "ipyvasp=ipyvasp.cli:app",
        ]
    },
)
