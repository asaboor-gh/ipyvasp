
__all__ = ['iplot2html', 'iplot_rgb_lines', 'iplot_dos_lines']


import re
from collections import Iterable
import numpy as np

import plotly.graph_objects as go

# Inside packages import 
from . import parser as vp
from . import splots as sp
from . import utils as gu
    

def _format_rgb_data(K, E, pros, labels, interp, occs, kpoints, maxwidth = 10, indices = None):
    "Transform data to 1D for rgb lines to plot effectently. Output is a dictionary."
    data = sp._fix_data(K, E, pros, labels, interp, rgb = True, occs = occs, kpoints = kpoints)
    # Note that data['pros'] is normalized to 0-1
    rgb = np.zeros((*np.shape(data['evals']),3)) # Initialize rgb array, because there could be less than three channels
    if data['pros'].shape[2] == 3:
        rgb = data['pros']
    elif data['pros'].shape[2] == 2:
        rgb[:,:,:2] = data['pros'] # Normalized overall color data
        labels = [*labels, '']
    elif data['pros'].shape[2] == 1:
        rgb[:,:,:1] = data['pros'] # Normalized overall color data
        labels = [*labels, '','']
    
    # Since normalized data is Y = (X - X_min)/(X_max - X_min), so X = Y*(X_max - X_min) + X_min is the actual data.
    low, high = data['ptp']
    data['norms'] = np.round(rgb*(high - low) + low, 3) # Read actual data back from normalized data.
    if data['pros'].shape[2] == 2:
        data['norms'][:,:,2] = np.nan # Avoid wrong info here
    elif data['pros'].shape[2] == 1:
        data['pros'][:,:,1:] = np.nan
    
    lws = np.sum(rgb,axis = 2) # Sum of all colors
    lws = maxwidth*lws/(float(np.max(lws)) or 1) # Normalize to maxwidth
    data['widths'] = 0.0001 + lws #should be before scale colors, almost zero size of a data point with no contribution.
    

    # Now scale colors to 1 at each point.
    cl_max = np.max(data['pros'],axis=2)
    cl_max[cl_max==0.0] = 1 # avoid divide by zero. Contributions are 4 digits only.
    data['pros'] = (rgb/cl_max[:,:,np.newaxis]*255).astype(int) # Normalized per point and set rgb data back to data.
    
    if indices is None: # make sure indices are in range
        indices = range(np.shape(data['evals'])[1])
        
    # Now process data to make single data for faster plotting.
    txt = 'Projection: [{}]</br>Value:'.format(', '.join(labels))
    K, E, C, S, PT, OT, KT, ET = [], [], [], [], [], [], [], []
    for i, b in enumerate(indices):
        K  = [*K, *data['kpath'], np.nan]
        E  = [*E, *data['evals'][:,i], np.nan]
        C  = [*C, *[f'rgb({r},{g},{b})' for (r,g,b) in data['pros'][:,i,:]], 'rgb(0,0,0)']
        S  = [*S, *data['widths'][:,i], data['widths'][-1,i]]
        PT = [*PT, *[f'{txt} [{s}, {p}, {d}]' for (s,p,d) in data['norms'][:,i]], ""]
        OT = [*OT, *[f'Occ: {t:>7.4f}' for t in data['occs'][:,i]], ""]
        KT = [*KT, *[f'K<sub>{j+1}</sub>: {x:>7.3f}{y:>7.3f}{z:>7.3f}' for j, (x,y,z) in enumerate(data['kpoints'])], ""]
        ET = [*ET, *["{}".format(b + 1) for _ in data['kpath']],""] # Add bands subscripts to labels.
    
    T = [f"</br>{p} </br></br>Band: {e}  {o}</br>{k}" for (p,e,o,k) in zip(PT,ET, OT,KT)]
    return {'K':K, 'E':E, 'C':C, 'S':S, 'T':T, 'labels': labels} # K, energy, marker color, marker size, text, labels that get changed

def _fmt_labels(ticklabels):
    if isinstance(ticklabels, Iterable):
        labels = [re.sub(r'\$\_\{(.*)\}\$|\$\_(.*)\$', r'<sub>\1\2</sub>', lab, flags = re.DOTALL) for lab in ticklabels] # will match _{x} or _x not both at the same time.
        return [re.sub(r'\$\^\{(.*)\}\$|\$\^(.*)\$', r'<sup>\1\2</sup>', lab, flags = re.DOTALL) for lab in labels]
    return ticklabels

def iplot2html(fig,filename = None, out_string = False, modebar = True):
    """
    - Writes plotly's figure as HTML file or display in IPython which is accessible when online. It is different than plotly's `fig.to_html` as it is minimal in memory. If you need to have offline working file, just use `fig.write_html('file.html')` which will be larger in size.
    Args:
        - fig      : A plotly's figure object.
        - filename : Name of file to save fig. Defualt is None and show plot in Colab/Online or return hrml string.
        - out_string: If True, returns HTML string, if False displays graph if possible.
    """
    import uuid # Unique div-id required,otherwise jupyterlab renders at one place only and overwite it.
    div_id = "graph-{}".format(uuid.uuid1())
    fig_json = fig.to_json()
    # a simple HTML template
    if filename:
        _filename = gu.prevent_overwrite(filename)
        template = """<html>
        <head>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <div id='{}'></div>
            <script>
                var fig_data = {}
                Plotly.react('{}', fig_data.data, fig_data.layout);
            </script>
        </body>
        </html>"""

        # write the JSON to the HTML template
        with open(_filename, 'w') as f:
            f.write(template.format(div_id,fig_json,div_id))

    else:
        if modebar==True: #Only for docs issue
            config = "{displayModeBar: true,scrollZoom: true}"
        else:
            config = "{displayModeBar: false,scrollZoom: true}"
        template = """<div>
        <script src='https://cdn.plot.ly/plotly-latest.min.js'></script>
            <div id='{}'><!-- Plotly chart DIV --></div>
            <script>
                var data = {};
                var config = {};
                Plotly.newPlot('{}', data.data,data.layout,config);
            </script>
        </div>""".format(div_id,fig_json,config,div_id)
        if out_string is True:
            return template
        else:
            from IPython.display import HTML
            return HTML(template)
        
def iplot2widget(fig, fig_widget = None, template = None): 
    "Converts plotly's figure to FigureWidget by copying attributes and data. If fig_widget is provided, it will update it. Adds template if provided."
    if not isinstance(fig,go.Figure):
        raise ValueError("fig must be instance of plotly.graph_objects.Figure")
    
    if fig_widget is None:
        fig_widget = go.FigureWidget()
    elif not isinstance(fig_widget,go.FigureWidget):
        raise ValueError("fig_widget must be FigureWidget")
    
    fig_widget.data = [] # Clear previous data
    if template is not None:
        fig.layout.template = template # will make white flash if not done before
    
    fig_widget.layout = fig.layout
        
    with fig_widget.batch_animate():
        for data in fig.data:
            fig_widget.add_trace(data)
        
    return fig_widget


def iplot_bands(K, E,
    fig    = None,
    elim   = None,
    kticks = None, 
    interp = None,   
    title  = None,
    **kwargs
    ):
    """
    Plot band structure using plotly.
    
    Parameters
    ----------
    K : array_like of shape (Nk,)
    E : array_like of shape (Nk, Nb)
    fig : plotly.graph_objects.Figure, created if None
    elim : tuple, energy limits for plot
    kticks : [(int, str),...] for indices of high symmetry k-points. To join a broken path, use '<=' before symbol, e.g.  [(0, 'G'),(40, '<=K|M'), ...] will join 40 back to 39. You can also use shortcut like zip([0,10,20],'GMK').
    interp : int or list/tuple of (n,k) for interpolation. If int, n is number of points to interpolate. If list/tuple, n is number of points and k is the order of spline.
    title : str, title of plot
    
    kwargs are passed to plotly.graph_objects.Scatter
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    if isinstance(K, dict): # Provided by Bands class, don't do is yourself
        K, indices = K['K'], K['indices']
    else:
        K, indices = K, range(np.shape(E)[1]) # Assume K is provided by user
        
    K, E, xticks, xticklabels = sp._validate_data(K,E,elim,kticks,interp)
    data = _format_rgb_data(K, E, [E], ['X'], interp, E, np.array([K,K,K]).reshape((-1,3)),maxwidth = 1, indices = indices) # moking other arrays, we need only 
    K, E, T = data['K'], data['E'] , data['T'] # Fixed K and E as single line data
    T = ['Band' + t.split('Band')[1].split('Occ')[0] for t in T] # Just Band number here
    
    if fig is None:
        fig = go.Figure()
        
    kwargs = {'mode': 'markers + lines', 'marker': dict(size = 0.1), **kwargs} # marker so that it is selectable by box, otherwise it does not
    fig.add_trace(go.Scatter(x = K, y = E, hovertext = T, **kwargs))
    
    fig.update_layout(template = 'plotly_white', title = (title or ''), # Do not set autosize = False, need to be responsive in widgets boxes
            margin = go.layout.Margin(l=60,r=50,b=40,t=75,pad=0),
            yaxis = go.layout.YAxis(title_text = 'Energy (eV)',range = elim or [min(E), max(E)]),
            xaxis = go.layout.XAxis(ticktext = _fmt_labels(xticklabels), tickvals = xticks,tickmode = "array",range = [min(K), max(K)]),
            font = dict(family="stix, serif",size=14)
    )
    return fig
        
def iplot_rgb_lines(K, E, pros, labels, occs, kpoints,
    fig        = None,
    elim       = None,
    kticks     = None, 
    interp     = None, 
    maxwidth   = 10,   
    mode       = 'markers + lines',
    title      = None,
    **kwargs        
    ):
    """
    Interactive plot of band structure with rgb data points using plotly.
    
    Parameters
    ----------
    K : array-like, shape (nk,)
    E : array-like, shape (nk,nb)
    pros : array-like, shape (m,nk,nb), m is the number of projections
    labels : list of str, length m
    occs : array-like, shape (nk,nb)
    kpoints : array-like, shape (nk,3)
    fig : plotly.graph_objects.Figure, if not provided, a new figure will be created
    elim : tuple, (emin,emax), energy range to plot
    kticks : [(int, str),...] for indices of high symmetry k-points. To join a broken path, use '<=' before symbol, e.g.  [(0, 'G'),(40, '<=K|M'), ...] will join 40 back to 39. You can also use shortcut like zip([0,10,20],'GMK').
    interp : int or list/tuple of (n,k) for interpolation. If int, n is number of points to interpolate. If list/tuple, n is number of points and k is the order of spline.
    maxwidth : float, maximum linewidth, 10 by default
    mode : str, plotly mode, 'markers + lines' by default, see modes in `plotly.graph_objects.Scatter`.
    title : str, title of the figure, labels are added to the end of the title.
    
    kwargs are passed to `plotly.graph_objects.Scatter`.
    
    Returns
    -------
    fig : plotly.graph_objects.Figure that can be displayed in Jupyter notebook or saved as html using `iplot2html`.
    """
    if isinstance(K, dict): # Provided by Bands class, don't do is yourself
        K, indices = K['K'], K['indices']
    else:
        K, indices = K, range(np.shape(E)[1]) # Assume K is provided by user
    
    K, E, xticks, xticklabels = sp._validate_data(K,E,elim,kticks,interp)
    data = _format_rgb_data(K, E, pros, labels, interp, occs, kpoints,maxwidth = maxwidth, indices = indices)
    K, E, C, S, T, labels = data['K'], data['E'], data['C'], data['S'], data['T'], data['labels']
    
    if fig is None:
        fig = go.Figure()
        
    kwargs.pop('marker_color',None) # Provided by C
    kwargs.pop('marker_size',None) # Provided by S
    kwargs.update({'hovertext': T, 'marker': {'line_color': 'rgba(0,0,0,0)', **kwargs.get('marker',{}), 'color': C, 'size': S}}) # marker edge should be free
    
    fig.add_trace(go.Scatter(x = K, y = E, mode = mode, **kwargs))
    
    fig.update_layout(template = 'plotly_white', title = (title or '') + '[' + ', '.join(labels) + ']', # Do not set autosize = False, need to be responsive in widgets boxes
            margin = go.layout.Margin(l=60,r=50,b=40,t=75,pad=0),
            yaxis = go.layout.YAxis(title_text = 'Energy (eV)',range = elim or [min(E), max(E)]),
            xaxis = go.layout.XAxis(ticktext = _fmt_labels(xticklabels), tickvals = xticks,tickmode = "array",range = [min(K), max(K)]),
            font = dict(family="stix, serif",size=14)
    )
    return fig

def iplot_dos_lines(energy, dos_arrays, labels,
    fig = None,
    elim = None,
    colormap = 'tab10',
    colors = None,
    fill = True,
    vertical = False,
    stack = False, 
    mode = 'lines',
    interp = None,
    **kwargs):
    """
    Plot density of states (DOS) lines.
    
    Parameters
    ----------
    energy : array-like, shape (n,)
    dos_arrays : list of array_like, each of shape (n,) or array-like (m,n)
    labels : list of str, length = len(dos_arrays) should hold.
    fig : plotly.graph_objects.Figure, if not provided, a new figure will be created
    elim : list of length 2, (emin, emax), if None, (min(energy), max(energy)) is used.
    colormap : str, default 'tab10', any valid matplotlib colormap name. Note that colormap is take from matplotlib, not plotly.
    colors : list of str, length = len(dos_arrays) should hold if given, and will override colormap. Should be valid CSS colors. 
    fill : bool, default True, if True, fill the area under the DOS lines.
    vertical : bool, default False, if True, plot DOS lines vertically.
    mode : str, default 'lines', plotly mode, see modes in `plotly.graph_objects.Scatter`.
    stack : bool, default False, if True, stack the DOS lines. Only works for horizontal plots.
    interp : int or list/tuple of (n,k), default None, if given, interpolate the DOS lines using spline.
    
    keyword arguments are passed to `plotly.graph_objects.Scatter`.
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    energy, dos_arrays, labels, colors = sp._fix_dos_data(energy, dos_arrays, labels, colors, interp)
    if fig is None:
        fig = go.Figure()
        fig.update_layout(margin = go.layout.Margin(l=60,r=50,b=40,t=75,pad=0),
                          font = dict(family="stix, serif",size=14)) # Do not set autosize = False, need to be responsive in widgets boxes
    if elim:
        ylim = [min(elim),max(elim)]
    else:
        ylim = [min(energy),max(energy)]
        
    if colors is None:
        from matplotlib.pyplot import cm
        _colors = cm.get_cmap(colormap)(np.linspace(0,1,2*len(labels)))
        colors  = ['rgb({},{},{})'.format(*[int(255*x) for x in c[:3]]) for c in _colors]
    if vertical:
        if stack:
            raise NotImplementedError('stack is not supported for vertical plots')
        
        _fill = 'tozerox' if fill else None
        fig.update_yaxes(range = ylim,title='Energy (eV)')
        fig.update_xaxes(title='DOS')
        for arr,label,color in zip(dos_arrays,labels,colors):
                fig.add_trace(go.Scatter(y = energy,x = arr,line_color=color,fill=_fill,mode=mode,name=label,**kwargs))
    else:
        extra_args = {'stackgroup':'one'} if stack else {}
        _fill = 'tozeroy' if fill else None
        fig.update_xaxes(range = ylim,title='Energy (eV)')
        fig.update_yaxes(title='DOS')
        for arr,label,color in zip(dos_arrays,labels,colors):
            fig.add_trace(go.Scatter(x = energy,y = arr,line_color=color,fill=_fill,mode=mode,name=label,**kwargs,**extra_args))
    
    return fig

