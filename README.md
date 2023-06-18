# ipyvasp

An successor of [pivotpy](https://github.com/massgh/pivotpy) for VASP-based DFT pre and post processing tool.

## Install
Currently the package is being built and not stable. If you want to use development version, install this way:(recommended to install in a virtual environment)
```
git clone https://github.com/massgh/ipyvasp.git
cd ipyvasp
pip install -e .
```

## Showcase Example
Plot 2D BZ layer on top of 3D!

```python
import ipyvasp as ipv
pos =  ipv.POSCAR('FCC POSACR FILE').rotate(35, [0,0,1])
ax = pos.splot_bz(vectors = None,color='skyblue',lw=0.2,alpha=0.5)
pos.splot_bz('xy', ax=ax, zoffset=0, vectors=None, lw=0.7,color='black')
```

![BZ](BZ.png)


