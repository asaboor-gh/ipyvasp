# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from ipyvasp import __version__

project = "ipyvasp"
copyright = "2022, Abdul Saboor"
author = "Abdul Saboor"
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.autosectionlabel",  # Add this for better cross-references
]

templates_path = ["_templates"]
exclude_patterns = []

# Add intersphinx mapping to other projects
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'numpy': ('https://numpy.org/doc/stable', None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

import furo
from sphinx_gallery.sorting import FileNameSortKey 

html_theme = "furo"
# html_theme_path = furo.__path__  # Unfortunately this is a list itself


html_static_path = ['_static']

autodoc_member_order = "bysource"
nbsphinx_allow_errors = True # on github, packages get confused
autosectionlabel_prefix_document = True

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # IMPORTANT: avoid nbsphinx picking gallery notebooks
    "auto_examples/*.ipynb",
    "auto_examples/**/*.ipynb",
]

# Sphinx Gallery configuration
sphinx_gallery_conf = {
    'examples_dirs': ['examples'],  # Directory containing example Python scripts (relative to docs/)
    'gallery_dirs': ['auto_examples'],  # Directory where the gallery will be generated (relative to docs/)
    'first_notebook_cell': '%matplotlib inline',  # Ensures plots display inline in the gallery
    'filename_pattern': '.*',  # Match all files (or customize, e.g., 'plot.*' or '.*\.py')
    'doc_module': ('ipyvasp',),  # Module to import for examples (helps with autodoc integration)
    'image_scrapers': ('matplotlib',),  # Scraper for matplotlib plots
    'ignore_pattern': r'__init__\.py',  # Ignore __init__.py files in examples
    'show_memory': False,  # Optional: Show memory usage in examples
    'reference_url': {
        'ipyvasp': None,  # Optional: Base URL for linking to source code
    },
    'backreferences_dir': None,  # Disable if not needed
    'expected_failing_examples': [],  # Helps reduce warnings
    'plot_gallery': True,  # Ensure plots are generated
    'download_all_examples': False,  # Disable if causing issues
    'show_signature': False,
    'capture_repr': ('_repr_html_',),  # Capture HTML representations for richer output
    'within_subsection_order': FileNameSortKey,  # Sort examples by filename within subsections
}
