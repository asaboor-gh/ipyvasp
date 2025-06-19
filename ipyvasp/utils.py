__all__ = [
    "get_file_size",
    "take",
    "set_dir",
    "interpolate_data",
    "rolling_mean",
    "color",
    "transform_color",
    "create_colormap",
]

import re
import os
import io
from contextlib import contextmanager
from pathlib import Path
from inspect import signature, getdoc
from itertools import islice


import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.ndimage.filters import convolve1d

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt


def get_file_size(path: str):
    """Return file size"""
    if (p := Path(path)).is_file():
        size = p.stat().st_size
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024.0:
                return "%3.2f %s" % (size, unit)
            size /= 1024.0
    else:
        return ""
    
def take(f, rows, cols=None, dtype=None, exclude=None,sep=None):
    """Read data from an opened file pointer `f` by indexing. `rows=None` picks all lines. Negative indexing is supported to read lines from end.
    Negative indexing is not supported in cols because of variable length of each line.
    If `cols=None`, returns a single str of line if one integer given, otherwise a list of lines.
    If `cols` is int ot sequence of int, each line is splitted by `sep` (default all whitespaces) and `dtype` is applied over resulting fields.
    `exclude` should be regex. It removes matching lines after selection by `rows`. Empty lines are also discarded if `cols` is given.

    Returns list (nested or plain) or single value or None based on `rows` and `cols` selection.

    `take(f, -1, 1, float) == float(f.readlines()[-1].split()[1])` with advantage for consuming almost no memory as compared to `f.readlines()` on a huge file.

    .. note::
        For more robust reading of structured files like `PROCAR` use `ipyvasp.parse_text` function.
    
    .. tip::
        If your output is matrix-like, you can cast it to numpy array like `take(...)*np.array(1)`.
   
    >>> with open('some_file','r') as f:
    >>>     take(f, -1, 1, float) # read last line, second column as float
    >>>     take(f, range(5)) # first 5 lines
    >>>     take(f, range(-5,0)) # last 5 lines
    """
    if not isinstance(f, io.TextIOWrapper):
        raise TypeError(f"f should be file-like object. got {type(f)}")
    
    return_line = False
    if isinstance(rows, int):
        rows = [rows]
        return_line = True

    if rows and not isinstance(rows, (tuple,list, range)):
        raise TypeError(f"rows should int/list/tuple/range, got {type(rows)}")
    
    f.seek(0)
    if rows and min(rows) < 0:
        if not hasattr(f, '_nlines'): # do this once, assuming file is not changed while reading
            f._nlines = sum(1 for _ in enumerate(f))
            f.seek(0)

        rows = [i + (f._nlines if i < 0 else 0) for i in rows] # make all positive
    
    if rows is None:
        lines = islice(f, None)
    else:
        lines = (l for i, l in enumerate(f) if i in rows)
    
    if exclude:
        lines = (l for l in lines if not re.search(exclude,l))
    
    if cols is not None:
        conv = dtype if callable(dtype) else (lambda v: v)
        return_col = False
        if isinstance(cols, int):
            cols = [cols]
            return_col = True 

        if not isinstance(cols, (list,tuple, range)):
            raise TypeError(f"cols should be a sequce of integers or single int, got {type(cols)}")
        
        lines = (l for l in lines if l.strip()) # remove empty lines after indexing and only if cols are given
        lines = ([conv(v) for i, v in enumerate(l.split(sep)) if i in cols] for l in lines)
        
        if return_col:
            lines = (line[0] if line else None for line in lines)
    else:
        return ''.join(lines) or None # just raw format as it is

    # Try to return None where there is nothing
    return next(lines,None) if return_line else (list(lines) or None)


def _sig_kwargs(from_func, skip_params=()):
    "Add signature to decorated function form other function"

    def wrapper(func, skip_params=skip_params):
        sig = signature(from_func)
        if not isinstance(skip_params, (list, tuple)):
            raise TypeError("skip_params must be list or tuple")

        this_sig = signature(func)
        all_params = list(this_sig.parameters.values())
        other_params = [
            value for value in all_params if value.kind.name != "VAR_KEYWORD"
        ]

        if other_params == all_params:  # no **kwargs
            return func

        skip_params = list(skip_params) + [
            value.name for value in other_params
        ]  # skip params already in func as positional or keyword
        target_params = [
            value for value in sig.parameters.values() if value.name not in skip_params
        ]
        target_sig = sig.replace(parameters=other_params + target_params)
        func.__signature__ = target_sig
        return func

    return wrapper

def _md_code_blocks_to_rst(text):
    def repl(match):
        language = match.group(1) or 'python'  # default to 'python' if no language
        code = match.group(2)
        return f'.. code-block:: {language}\n\n   ' + '\n   '.join(code.strip().splitlines())

    return re.sub(
        r'```([a-zA-Z0-9]*)\s*\n(.*?)```',
        repl,
        text,
        flags=re.DOTALL
    )


def _sub_doc(from_func, replace={}):
    """Assing __doc__ from other function. Replace words in docs where need."""

    def wrapper(func):
        docs = getdoc(from_func)
        if not isinstance(replace, dict):
            raise TypeError("replace must be dict of 'match':'replace'")

        for k, v in replace.items():
            docs = re.sub(k, v, docs, count=1, flags=re.DOTALL)
        func.__doc__ = _md_code_blocks_to_rst(docs)
        return func

    return wrapper


def _fmt_doc(fmt_dict):
    "Format docstring with keys from given dict"

    def wrapper(func):
        docs = func.__doc__  # Not by getdoc here, needs proper formatting
        if not isinstance(fmt_dict, dict):
            raise TypeError("fmt_dict must be dict of 'match':'replace'")
        func.__doc__ = docs.format(**fmt_dict)
        return func

    return wrapper


@contextmanager
def set_dir(path: str):
    """Context manager to work in some directory and come back.

    >>> with set_dir('some_folder'):
    >>>    do_something()
    >>> # Now you are back in starting directory
    """
    current = os.getcwd()  # not available in pathlib yet
    abspath = Path(path).resolve(strict=True).absolute()
    try:
        os.chdir(abspath)
        yield abspath
    finally:
        os.chdir(current)


def interpolate_data(x: np.ndarray, y: np.ndarray, n: int = 10, k: int = 3) -> tuple:
    """
    Returns interpolated xnew,ynew. If two points are same, it will add 0.1*min(dx>0) to compensate it.

    Parameters
    ----------
    x : ndarry, 1D array of size p,
    y : ndarray, ndarray of size p*q*r,....
    n : int, Number of points to add between two given points.
    k : int, Polynomial order to interpolate.


    Example
    -------
    For ``K(p),E(p,q)`` input from bandstructure, do ``Knew, Enew = interpolate_data(K,E,n=10,k=3)`` for cubic interploation.

    Returns
    -------
    tuple: (xnew, ynew) after interpolation.


    .. note::
        Only axis 0 will be interpolated. If you want general interploation, use ``from scipy.interpolate import make_interp_spline, BSpline``.
    """
    # Avoid adding points between same points, like in kpath patches
    inds = [i for i in range(0, len(x)) if x[i - 1] == x[i]]  # Duplicate indices
    if inds:
        inds = [0, *inds, len(x)]  # Indices to split x
        ranges = list(zip(inds[:-1], inds[1:]))  # we are using this twice,so make list
        for p, q in ranges:
            if q - p == 1:  # means three consecutive points have same value
                raise ValueError(
                    f"Three or more duplicate values found at index {p} in array `x`, at most two allowed for broken kpath like scenario."
                )
        arrays = [[x[i:j], y[i:j]] for i, j in ranges]  # Split x,y into arrays
    else:
        arrays = [(x, y)]

    new_as, new_bs = [], []
    for a, b in arrays:
        anew = [np.linspace(a[i], a[i + 1], n) for i in range(len(a) - 1)]
        anew = np.reshape(anew, (-1))
        spl = make_interp_spline(a, b, k=k)  # BSpline object
        bnew = spl(anew)
        new_as.append(anew)
        new_bs.append(bnew)

    if len(new_as) == 1:
        return new_as[0], new_bs[0]

    return np.concatenate(new_as, axis=0), np.concatenate(new_bs, axis=0)


def rolling_mean(
    X: np.ndarray,
    period: float,
    period_right: float = None,
    interface: float = None,
    mode: str = "wrap",
    smoothness: int = 2,
) -> np.ndarray:
    """
    Caluate rolling mean of array X using scipy.ndimage.filters.convolve1d.

    Parameters
    ----------
    X : ndarray, 1D numpy array.
    period : float, In range [0,1]. Period of rolling mean. Applies left side of X from center if period_right is given.
    period_right : float, In range [0,1]. Period of rolling mean on right side of X from center.
    interface : float, In range [0,1]. The point that divides X into left and right, like in a slab calculation.
    mode : string, Mode of convolution. Default is wrap to keep periodcity of crystal in slab caluation. Read scipy.ndimage.filters.convolve1d for more info.
    smoothness : int, Default is 2. Order of making the output smoother. Larger is better but can't be too large as there will be points lost at each convolution.

    Returns
    -------
    ndarray: convolved array of same size as X if mode is 'wrap'. May be smaller if mode is something else.
    """
    if period_right is None:
        period_right = period

    if interface is None:
        interface = 0.5

    if smoothness < 1:
        raise ValueError("smoothness must be >= 1")

    wx = np.linspace(
        0, 1, X.size, endpoint=False
    )  # x-axis for making weights for convolution, 0 to 1 - (last point is not included in VASP grid).
    x1, x2, x3, x4 = (
        period_right,
        interface - period,
        interface + period_right,
        1 - period,
    )
    m1, m2, m3 = 0.5 / x1, 1 / (x2 - x3), 0.5 / (1 - x4)  # Slopes
    weights_L = np.piecewise(
        wx,  # .----.____. Looks like this
        [
            wx < x1,  # left side reflected by right side
            (wx >= x1) & (wx <= x2),  # left side
            (wx > x2) & (wx < x3),  # middle contribution from left and right
            (wx >= x3) & (wx <= x4),  # right side
            wx > x4,  # right side reflected by left side
        ],
        [
            lambda z: m1 * (z - x1) + 1,
            1,
            lambda z: m2 * (z - x2) + 1,
            0,
            lambda z: m3 * (z - x4),
        ],
    )

    weights_R = 1 - weights_L  # .____.----. Looks like this

    L = int(period * X.size)  # Left periodictity
    R = int(period_right * X.size)  # Right Periodicity

    kernel_L = np.ones((L,)) / L
    kernel_R = np.ones((R,)) / R

    mean_L = convolve1d(X, kernel_L, mode=mode)
    mean_R = convolve1d(X, kernel_R, mode=mode)

    mean_all = weights_L * mean_L + weights_R * mean_R

    if smoothness > 1:
        p_l, p_r = (
            period / 2,
            period_right / 2,
        )  # Should be half of period for smoothing each time
        for _ in range(smoothness - 1):
            mean_all = rolling_mean(
                mean_all,
                period=p_l,
                period_right=p_r,
                interface=interface,
                mode=mode,
                smoothness=1,
            )
            p_l, p_r = p_l / 2, p_r / 2

    return mean_all

@contextmanager
def prevent_overwrite(path) -> Path:
    """Contextmanager to prevents overwiting as file by adding numbers in given path.

    >>> with prevent_overwrite("file.txt") as path:
    >>>     print(path) # file.txt if it doesn't exist, file-1.txt if it exists, file-2.txt if file-1.txt exists and so on.
    """
    out_path = Path(path)
    if out_path.exists():
        # Check existing files
        i = 0
        name = (out_path.parent / out_path.stem) + "-{}" + out_path.suffix
        while Path(name.format(i)).is_file():
            i += 1
        out_path = Path(name.format(i))
        print(f"Found existing path: {path!r}\nConverting to: {out_path!r}")

    yield out_path


class color:
    def bg(text, r, g, b):
        """Provide r,g,b component in range 0-255"""
        return f"\033[48;2;{r};{g};{b}m{text}\033[00m"

    def fg(text, r, g, b):
        """Provide r,g,b component in range 0-255"""
        return f"\033[38;2;{r};{g};{b}m{text}\033[00m"

    # Usual Colos
    r = lambda text: f"\033[0;91m {text}\033[00m"
    rb = lambda text: f"\033[1;91m {text}\033[00m"
    g = lambda text: f"\033[0;92m {text}\033[00m"
    gb = lambda text: f"\033[1;92m {text}\033[00m"
    b = lambda text: f"\033[0;34m {text}\033[00m"
    bb = lambda text: f"\033[1;34m {text}\033[00m"
    y = lambda text: f"\033[0;93m {text}\033[00m"
    yb = lambda text: f"\033[1;93m {text}\033[00m"
    m = lambda text: f"\033[0;95m {text}\033[00m"
    mb = lambda text: f"\033[1;95m {text}\033[00m"
    c = lambda text: f"\033[0;96m {text}\033[00m"
    cb = lambda text: f"\033[1;96m {text}\033[00m"


def transform_color(
    arr: np.ndarray,
    s: float = 1,
    c: float = 1,
    b: float = 0,
    mixing_matrix: np.ndarray = None,
) -> np.ndarray:
    """
    Color transformation such as brightness, contrast, saturation and mixing of an input color array. ``c = -1`` would invert color,keeping everything else same.

    Parameters
    ----------
    arr : ndarray, input array, a single RGB/RGBA color or an array with inner most dimension equal to 3 or 4. e.g. [[[0,1,0,1],[0,0,1,1]]].
    c : float, contrast, default is 1. Can be a float in [-1,1].
    s : float, saturation, default is 1. Can be a float in [-1,1]. If s = 0, you get a gray scale image.
    b : float, brightness, default is 0. Can be a float in [-1,1] or list of three brightnesses for RGB components.
    mixing_matrix : ndarray, A 3x3 matrix to mix RGB values, such as `ipyvas.utils.color_matrix`.

    Returns
    -------
    ndarray: Transformed color array of same shape as input array.

    See `Recoloring <https://docs.microsoft.com/en-us/windows/win32/gdiplus/-gdiplus-recoloring-use?redirectedfrom=MSDN>`_
    and `Rainmeter <https://docs.rainmeter.net/tips/colormatrix-guide/>`_ for useful information on color transformation.
    """
    arr = np.array(arr)  # Must
    t = (1 - c) / 2  # For fixing gray scale when contrast is 0.
    whiteness = np.array(b) + t  # need to clip to 1 and 0 after adding to color.
    sr = (1 - s) * 0.2125  # red saturation from red luminosity
    sg = (1 - s) * 0.7154  # green saturation from green luminosity
    sb = (1 - s) * 0.0721  # blue saturation from blue luminosity
    # trans_matrix is multiplied from left, or multiply its transpose from right.
    # trans_matrix*color is not normalized but value --> value - int(value) to keep in [0,1].
    trans_matrix = np.array(
        [
            [c * (sr + s), c * sg, c * sb],
            [c * sr, c * (sg + s), c * sb],
            [c * sr, c * sg, c * (sb + s)],
        ]
    )
    if np.ndim(arr) == 1:
        new_color = np.dot(trans_matrix, arr)
    else:
        new_color = np.dot(arr[..., :3], trans_matrix.T)
    if mixing_matrix is not None and np.size(mixing_matrix) == 9:
        new_color = np.dot(new_color, np.transpose(mixing_matrix))
    new_color[new_color > 1] = new_color[new_color > 1] - new_color[
        new_color > 1
    ].astype(int)
    new_color = np.clip(new_color + whiteness, a_max=1, a_min=0)
    if np.shape(arr)[-1] == 4:
        axis = len(np.shape(arr)) - 1  # Add back Alpha value if present
        new_color = np.concatenate([new_color, arr[..., 3:]], axis=axis)
    return new_color


# color_marices for quick use
color_matrix = np.array(
    [[0.5, 0, 0.5, 1], [0.5, 0.5, 0, 1], [0, 0.5, 0.5, 0.2], [1, 1, 0.2, 0]]
)  # lights up to see colors a little bit
rbg_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])  # Red, Blue, Green
cmy_matrix = np.array(
    [[0, 0.5, 0.5, 1], [0.5, 0, 0.5, 1], [0.5, 0.5, 0, 0.2], [1, 1, 0.2, 0]]
)  # Generates CMYK color palette


# Register 'RGB' colormap in current session
RGB = LinearSegmentedColormap.from_list(
    "RGB", [(0.9, 0, 0), (0.9, 0.9, 0), (0, 0.9, 0), (0, 0.9, 0.9), (0, 0, 0.9)]
)
CMY = LinearSegmentedColormap.from_list(
    "CMY", [(0, 0.9, 0.9), (0, 0, 0.9), (0.9, 0, 0.9), (0.9, 0, 0), (0.9, 0.9, 0)]
)
plt.register_cmap("RGB", RGB)
plt.register_cmap("CMY", CMY)


def create_colormap(name="RB", colors=[(0.9, 0, 0), (0, 0, 0.9)]):
    """
    Create and register a custom colormap from a list of RGB colors. and then use it's name in plottoing functions to get required colors.

    Parameters
    ----------
    name: str, name of the colormap
    colors: list of RGB colors, e.g. [(0.9,0,0),(0,0,0.9)] or named colors, e.g. ['red','blue'], add as many colors as you want.

    Returns
    -------
    Colormap object which you can use to get colors from. like cm = create_colormap(); cm(0.5) which will return a color at center of map
    """
    RB = LinearSegmentedColormap.from_list(name, colors)
    plt.register_cmap(name, RB)
    return RB
