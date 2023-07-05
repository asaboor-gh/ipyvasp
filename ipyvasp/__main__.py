"""
You can use ipyvasp as cli tool for subset of its features as follows:

ipyvasp [OPTIONS] COMMAND [ARGS]...
python -m ipyvasp [OPTIONS] COMMAND [ARGS]...
"""

from .cli import app
app(prog_name="ipyvasp")