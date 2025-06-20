# IPyVASP

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15482350.svg)](https://doi.org/10.5281/zenodo.15482350)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/asaboor-gh/ipyvasp/HEAD?urlpath=%2Fdoc%2Ftree%2Fdocs%2Fsource%2Fnotebooks%2Fquickstart.ipynb)
[![PyPI Downloads](https://static.pepy.tech/badge/ipyvasp)](https://pepy.tech/projects/ipyvasp)

An VASP-based DFT pre and post processing tool.

## Install
Currently the package is being built and not stable. If you want to use development version, install this way:(recommended to install in a virtual environment)
```
git clone https://github.com/massgh/ipyvasp.git
cd ipyvasp
pip install -e .
```

## Documentataion
- [Latest at Github Pages](https://asaboor-gh.github.io/ipyvasp/) 
- [Read the Docs](https://ipyvasp.readthedocs.io/)

## Showcase Examples
Plot 2D BZ layer on top of 3D!

```python
import ipyvasp as ipv
pos =  ipv.POSCAR('FCC POSACR FILE').set_zdir([1,1,1])
ax = pos.splot_bz(vectors = None,color='skyblue',lw=0.2,alpha=0.2,fill=True)

kpts = [[0,-1/2,0],[0,0,0]]
pos.splot_kpath(kpts,labels=[str(k) for k in kpts],zorder=-1) # At 3D BZ

pos2 = pos.transform(lambda a,b,c: (a-c, b-c, a+b+c)) # 111 plane
pos2.splot_bz('xy',ax=ax,zoffset=0.15,vectors=None,color='navy')

kp2 = pos.bz.map_kpoints(pos2.bz, kpts)
pos2.splot_kpath(kp2,labels=[str(k) for k in kp2.round(1).tolist()],color='navy',fmt_label=lambda lab: (lab+'\n', dict(va='center',color='navy')),zorder=3) 

ax.set_axis_off()
```

![BZ](BZ.png)

Interactively select bandstructure path by clicking on high symmetry points on plot!

![KP](KP.png)

Apply operations on POSCAR and simultaneously view using plotly's `FigureWidget` in Jupyterlab side by side.

![snip](op.png)

![BandsWidget](Bands.jpg)
More coming soon! You can test these examples on [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/asaboor-gh/ipyvasp/HEAD?urlpath=%2Fdoc%2Ftree%2Fdocs%2Fsource%2Fnotebooks%2Fquickstart.ipynb).
