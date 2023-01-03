import re
import os
import glob
from collections import namedtuple
from subprocess import Popen, PIPE
from contextlib import contextmanager


import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.ndimage.filters import convolve1d


def get_file_size(path:str):
    """Return file size"""
    if os.path.isfile(path):
        size = os.stat(path).st_size
        for unit in ['Bytes','KB','MB','GB','TB']:
            if size < 1024.0:
                return "%3.2f %s" % (size,unit)
            size /= 1024.0
    else:
        return ''

@contextmanager
def set_dir(path:str):
    "Context manager to work in some directory and come back"
    current = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(current)

def interpolate_data(x:np.ndarray,y:np.ndarray,n:int=10,k:int=3) -> tuple:
    """Returns interpolated xnew,ynew. If two points are same, it will add 0.1*min(dx>0) to compensate it.
    
    Args:
        x (ndarry): 1D array of size p,
        y (ndarray): ndarray of size p*q*r,....
        n (int): Number of points to add between two given points.
        k (int): Polynomial order to interpolate.


    Example: 
        For ``K(p),E(p,q)`` input from bandstructure, do ``Knew, Enew = interpolate_data(K,E,n=10,k=3)`` for cubic interploation.
    
    Returns:
        tuple: (xnew, ynew) after interpolation.
        
    .. note::
        Only axis 0 will be interpolated. If you want general interploation, use ``from scipy.interpolate import make_interp_spline, BSpline``.
    """
    #Add very small values at simliar points to make interpolation work.
    ind=[i for i in range(0,len(x)) if x[i-1]==x[i]] #Duplicate indices
    xa=np.unique(x)
    dx=0.001*np.min(xa[1:]-xa[:-1]) #Very small value to add to duplicate points.
    if(ind):
        for pt in ind:
            x[pt:]=x[pt:]-x[pt]+x[pt-1]+dx
    # Now Apply interpolation
    xnew=[np.linspace(x[i],x[i+1],n) for i in range(len(x)-1)]
    xnew=np.reshape(xnew,(-1))
    spl = make_interp_spline(x, y, k=k) #BSpline object
    ynew = spl(xnew)
    return xnew,ynew

def rolling_mean(X:np.ndarray, period:float, period_right:float = None, interface:float = None, mode:str = 'wrap', smoothness:int = 2) -> np.ndarray:
    """Caluate rolling mean of array X using scipy.ndimage.filters.convolve1d.
    
    Args:
        X (ndarray): 1D numpy array.
        period (float): In range [0,1]. Period of rolling mean. Applies left side of X from center if period_right is given.
        period_right (float): In range [0,1]. Period of rolling mean on right side of X from center.
        interface (float): In range [0,1]. The point that divides X into left and right, like in a slab calculation.
        mode (string): Mode of convolution. Default is wrap to keep periodcity of crystal in slab caluation. Read scipy.ndimage.filters.convolve1d for more info.
        smoothness (int): Default is 2. Order of making the output smoother. Larger is better but can't be too large as there will be points lost at each convolution.
    
    Returns:
        ndarray: convolved array of same size as X if mode is 'wrap'. May be smaller if mode is something else.
    """
    if period_right is None:
        period_right = period
    
    if interface is None:
        interface = 0.5
        
    if smoothness < 1:
        raise ValueError('smoothness must be >= 1')
    
    wx = np.linspace(0, 1, X.size, endpoint = False) # x-axis for making weights for convolution, 0 to 1 - (last point is not included in VASP grid).
    x1, x2, x3, x4 = period_right, interface - period, interface + period_right, 1 - period
    m1, m2, m3 = 0.5/x1, 1/(x2-x3), 0.5/(1-x4) # Slopes
    weights_L = np.piecewise(wx,  # .----.____. Looks like this
                [
                    wx < x1, # left side reflected by right side
                    (wx >= x1) & (wx <= x2), # left side
                    (wx > x2) & (wx<x3), # middle contribution from left and right
                    (wx>=x3) & (wx<=x4), # right side
                    wx > x4 # right side reflected by left side
                ], 
                [
                    lambda z:  m1*(z-x1)+1,
                    1, 
                    lambda z: m2*(z-x2) + 1,
                    0, 
                    lambda z: m3*(z-x4)
                ])
    
    weights_R =  1 - weights_L # .____.----. Looks like this
    
    L = int(period*X.size) # Left periodictity
    R = int(period_right*X.size) # Right Periodicity
    
    kernel_L = np.ones((L,))/L
    kernel_R = np.ones((R,))/R
    
    mean_L = convolve1d(X,kernel_L,mode = mode)
    mean_R = convolve1d(X,kernel_R,mode = mode)
    
    mean_all = weights_L*mean_L + weights_R*mean_R
    
    if smoothness > 1:
        p_l, p_r = period/2, period_right/2 # Should be half of period for smoothing each time
        for _ in range(smoothness - 1):
            mean_all = rolling_mean(mean_all, period = p_l, period_right = p_r, interface = interface, mode = mode, smoothness = 1)
            p_l, p_r = p_l/2, p_r/2
    
    return mean_all

def ps2py(ps_command :str ='Get-ChildItem', exec_type :str='-Command', path_to_ps:str='powershell.exe') -> list:
    """Captures powershell output in python.
    
    Args:
        ps_command (str): enclose ps_command in ' ' or " ".
        exec_type  (str): type of execution, default '-Command', could be '-File'.
        path_to_ps (str): path to powerhell.exe if not added to PATH variables.
    
    Returns:
        list: list of lines of output.
    """
    try: # Works on Linux and Windows if PS version > 5.
        cmd = ['pwsh', '-ExecutionPolicy', 'Bypass', exec_type, ps_command]
        proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
    except FileNotFoundError:
        try: # Works only on Windows.
            cmd = ['powershell', '-ExecutionPolicy', 'Bypass', exec_type, ps_command]
            proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
        except FileNotFoundError:
            # Works in case nothing above works and you know where is executable.
            cmd = [path_to_ps, '-ExecutionPolicy', 'Bypass', exec_type, ps_command]
            proc = Popen(cmd, stdout=PIPE, stderr=PIPE)

    out=[]; #save to out.
    while True:
        line = proc.stdout.readline()
        if line!=b'':
            line=line.strip()
            u_line=line.decode("utf-8")
            out.append(u_line)
        else:
            break
    out=[item for item in out if item!=''] #filter out empty lines
    return out

def ps2std(ps_command :str='Get-ChildItem', exec_type:str='-Command', path_to_ps:str='powershell.exe') -> None:
    """Prints powershell output in python std.
    Args:
        ps_command (str): enclose ps_command in ' ' or " ".
        exec_type  (str): type of execution, default '-Command', could be '-File'.
        path_to_ps (str): path to powerhell.exe if not added to PATH variables.
    """
    out = ps2py(path_to_ps=path_to_ps,exec_type=exec_type,ps_command=ps_command)
    for item in out:
        print(item)
    return None


def get_child_items(path:str = os.getcwd(),depth:int = None,recursive:bool=True,include:str=None,exclude:str=None,files_only:bool=False,dirs_only:bool= False) -> tuple:
    """Returns selected directories/files recursively from a parent directory.
    
    Args:
        path (str): path to a parent directory, default is `"."`
        depth (int): subdirectories depth to get recursively, default is None to list all down.
        recursive (bool): If False, only list current directory items, if True,list all items recursively down the file system.
        include (str): Default is None and includes everything. String of patterns separated by | to keep, could be a regular expression.
        exclude (str): Default is None and removes nothing. String of patterns separated by | to drop,could be a regular expression.
        files_only (bool): Boolean, if True, returns only files.
        dirs_only  (bool): Boolean, if True, returns only directories.
    
    Returns:
        tuple: (children,parent), children is list of selected directories/files and parent is given path. Access by index of by `get_child_items().{children,path}`.
    
    """
    path = os.path.abspath(path) # important
    pattern = path + '**/**' # Default pattern
    if depth != None and type(depth) == int:
        pattern = path + '/'.join(['*' for i in range(depth+1)])
        if glob.glob(pattern) == []: #If given depth is more, fall back.
            pattern = path + '**/**' # Fallback to default pattern if more depth to cover all.
    glob_files = glob.iglob(pattern, recursive=recursive)
    if dirs_only == True:
        glob_files = filter(lambda f: os.path.isdir(f),glob_files)
    if files_only == True:
        glob_files = filter(lambda f: os.path.isfile(f),glob_files)
    list_dirs=[]
    for g_f in glob_files:
        list_dirs.append(os.path.relpath(g_f,path))
    # Include check
    if include:
        list_dirs = [l for l in list_dirs if re.search(include,l)]
    # Exclude check
    if exclude:
        list_dirs = [l for l in list_dirs if not re.search(exclude,l)]
    # Keep only unique
    req_dirs = list(np.unique(list_dirs))
    out_files = namedtuple('GLOB',['children','parent'])
    return out_files(req_dirs,os.path.abspath(path))

def prevent_overwrite(path: str) -> str:
    """Prevents overwiting as file/directory by adding numbers in given file/directory path."""
    if os.path.exists(path):
        name, ext = os.path.splitext(path)
        # Check existing files
        i = 0
        _path = name + '-{}' + ext
        while os.path.isfile(_path.format(i)):
            i +=1
        out_path = _path.format(i)
        print(f"Found existing path: {path!r}\nConverting to: {out_path!r}")
        return out_path
    return path

class color:
     def bg(text,r,g,b):
          """Provide r,g,b component in range 0-255"""
          return f"\033[48;2;{r};{g};{b}m{text}\033[00m"
     def fg(text,r,g,b):
          """Provide r,g,b component in range 0-255"""
          return f"\033[38;2;{r};{g};{b}m{text}\033[00m"
     # Usual Colos
     r  = lambda text: f"\033[0;91m {text}\033[00m"
     rb = lambda text: f"\033[1;91m {text}\033[00m"
     g  = lambda text: f"\033[0;92m {text}\033[00m"
     gb = lambda text: f"\033[1;92m {text}\033[00m"
     b  = lambda text: f"\033[0;34m {text}\033[00m"
     bb = lambda text: f"\033[1;34m {text}\033[00m"
     y  = lambda text: f"\033[0;93m {text}\033[00m"
     yb = lambda text: f"\033[1;93m {text}\033[00m"
     m  = lambda text: f"\033[0;95m {text}\033[00m"
     mb = lambda text: f"\033[1;95m {text}\033[00m"
     c  = lambda text: f"\033[0;96m {text}\033[00m"
     cb = lambda text: f"\033[1;96m {text}\033[00m"


def transform_color(arr: np.ndarray,s:float=1,c:float=1,b:float=0,mixing_matrix:np.ndarray=None) -> np.ndarray:
    """Color transformation such as brightness, contrast, saturation and mixing of an input color array. ``c = -1`` would invert color,keeping everything else same.
    
    Args:
        arr (ndarray): input array, a single RGB/RGBA color or an array with inner most dimension equal to 3 or 4. e.g. [[[0,1,0,1],[0,0,1,1]]].
        c (float): contrast, default is 1. Can be a float in [-1,1].
        s (float): saturation, default is 1. Can be a float in [-1,1]. If s = 0, you get a gray scale image.
        b (float): brightness, default is 0. Can be a float in [-1,1] or list of three brightnesses for RGB components.
        mixing_matrix (ndarray): A 3x3 matrix to mix RGB values, such as `pp.color_matrix`.

    Returns:
        ndarray: Transformed color array of same shape as input array.
    `Recoloring <https://docs.microsoft.com/en-us/windows/win32/gdiplus/-gdiplus-recoloring-use?redirectedfrom=MSDN>`_
    
    `Rainmeter <https://docs.rainmeter.net/tips/colormatrix-guide/>`_
    """
    arr = np.array(arr) # Must
    t = (1-c)/2 # For fixing gray scale when contrast is 0.
    whiteness = np.array(b)+t # need to clip to 1 and 0 after adding to color.
    sr = (1-s)*0.2125 #red saturation from red luminosity
    sg = (1-s)*0.7154 #green saturation from green luminosity
    sb = (1-s)*0.0721 #blue saturation from blue luminosity
    # trans_matrix is multiplied from left, or multiply its transpose from right.
    # trans_matrix*color is not normalized but value --> value - int(value) to keep in [0,1].
    trans_matrix = np.array([
        [c*(sr+s), c*sg,      c*sb],
        [c*sr,   c*(sg+s),    c*sb],
        [c*sr,     c*sg,  c*(sb+s)]])
    if np.ndim(arr) == 1:
        new_color = np.dot(trans_matrix,arr)
    else:
        new_color = np.dot(arr[...,:3],trans_matrix.T)
    if mixing_matrix is not None and np.size(mixing_matrix)==9:
        new_color = np.dot(new_color,np.transpose(mixing_matrix))
    new_color[new_color > 1] = new_color[new_color > 1] - new_color[new_color > 1].astype(int)
    new_color = np.clip(new_color + whiteness,a_max=1,a_min=0)
    if np.shape(arr)[-1]==4:
        axis = len(np.shape(arr))-1 #Add back Alpha value if present
        new_color = np.concatenate([new_color,arr[...,3:]],axis=axis)
    return new_color