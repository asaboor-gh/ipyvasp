# AUTOGENERATED! DO NOT EDIT! File to edit: InteractivePlots.ipynb (unless otherwise specified).

__all__ = ['iplot2html', 'iplot_rgb_lines', 'iplot_dos_lines']

# Cell
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import plotly.graph_objects as go

# Inside packages import to work both with package and jupyter notebook.
try:
    from ipyvasp import parser as vp
    from ipyvasp import splots as sp
    from ipyvasp import utils as gu
except:
    import ipyvasp.parser as vp
    import ipyvasp.splots as sp
    import ipyvasp.utils as gu
    

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
    
    if np.shape(data['evals'])[1] == 1: # If only one band, then remove last nan.
        K, E, C, S, PT, OT, KT, ET = K[:-1], E[:-1], C[:-1], S[:-1], PT[:-1], OT[:-1], KT[:-1], ET[:-1]
    
    T = [f"</br>{p} </br></br>Band: {e}  {o}</br>{k}" for (p,e,o,k) in zip(PT,ET, OT,KT)]
    return {'K':K, 'E':E, 'C':C, 'S':S, 'T':T, 'labels': labels} # K, energy, marker color, marker size, text, labels that get changed


# Cell
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
    
    fig.update_layout(title = (title or '') + '[' + ', '.join(labels) + ']',
            margin = go.layout.Margin(l=60,r=50,b=40,t=75,pad=0),
            yaxis = go.layout.YAxis(title_text = 'Energy (eV)',range = elim or [min(E), max(E)]),
            xaxis = go.layout.XAxis(ticktext = xticklabels, tickvals = xticks,tickmode = "array",range = [min(K), max(K)]),
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
                          font = dict(family="stix, serif",size=14))
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
            raise NotImplementedError('stack is not implemented for vertical plots')
        
        _fill = 'tozerox' if fill else None
        fig.update_yaxes(range = ylim,title='Energy (eV)')
        fig.update_xaxes(title='DOS')
        for arr,label,color in zip(dos_arrays,labels,colors):
                fig.add_trace(go.Scatter(y = energy,x = arr,line_color=color,fill=_fill,mode='lines',name=label,**kwargs))
    else:
        extra_args = {'stackgroup':'one'} if stack else {}
        _fill = 'tozeroy' if fill else None
        fig.update_xaxes(range = ylim,title='Energy (eV)')
        fig.update_yaxes(title='DOS')
        for arr,label,color in zip(dos_arrays,labels,colors):
            fig.add_trace(go.Scatter(x = energy,y = arr,line_color=color,fill=_fill,mode='lines',name=label,**kwargs,**extra_args))
    
    return fig

