"""
POSCAR Example
==============

This example shows how to create and plot crystal structures."""

#%%
import numpy as np
import matplotlib.pyplot as plt

import ipyvasp as ipv

#%%
poscar = ipv.POSCAR.new(
    [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]],
    {"Ga": [[0.0, 0.0, 0.0]],"As": [[0.25, 0.25, 0.25]]},

    scale = 5
)

poscar.write("POSCAR",overwrite=True) # need it later

site_kws = dict(alpha=1)
ax1, ax2, ax3 = ipv.get_axes(3, (8,3), axes_3d=[0,1,2])

poscar.splot_lattice( ax = ax1, fill = False,label='original',
    site_kws=site_kws)

_ = poscar.transform(lambda a,b,c: (b+c-a, a+c-b, a+b-c)).transpose([1,2,0]) # bring a1 to x
print(poscar.last.data.metadata.TM)
poscar.last.splot_lattice( # .last points to last created POSCAR in transform
    ax=ax2, fill = False, color='red', label='transformed',
    site_kws=site_kws
)
poscar.last.splot_plane([1,1,0],1/2,ax=ax2)

poscar.transform(lambda a,b,c: (a-c,b-c,a+b+c)
).set_zdir([0,0,1]).splot_lattice(ax=ax3,color='red')

print(poscar.last.data.metadata.TM)
ax1.view_init(azim=-25, elev=15)
ax2.view_init(azim=-35, elev=15)
ipv.plt2html() 

#%%
view = poscar.view_weas(colors={'Ga':'red','As':'blue'})
view

#%%
view.download_image('weas.png')