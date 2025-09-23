"""
KPath Example
=============
"""
#%%
import numpy as np
import matplotlib.pyplot as plt

import ipyvasp as ipv

#%%
poscar = ipv.POSCAR.from_file("POSCAR")
poscar.get_kpath([
    (0,0,0,'Γ'), 
    (1/4,1/2,-1/4,'K'), 
    (0.5,0.5,0,'X'), 
    (0,0,0,'Γ'), 
    (0,0.5,0,'L')
    ], n=8)

#%%
poscar.get_kmesh(2,2,2, weight=0)

#%%
pos =  poscar.set_zdir([1,1,1])
ax = pos.splot_bz(vectors = None,color='skyblue',lw=0.2,alpha=0.2,fill=True)

kpts = [[0,-1/2,0],[0,0,0]]
pos.splot_kpath(kpts,labels=[str(k) for k in kpts],zorder=-1) # At 3D BZ

pos2 = pos.transform(lambda a,b,c: (a-c, b-c, a+b+c)) # 111 plane
pos2.splot_bz('xy',ax=ax,zoffset=0.15,vectors=None,color='navy')

kp2 = pos.bz.map_kpoints(pos2.bz, kpts)
pos2.splot_kpath(kp2,labels=[str(k) for k in kp2.round(1).tolist()],color='navy',fmt_label=lambda lab: (lab+'\n', dict(va='center',color='navy')),zorder=3) 

ax.set_axis_off()
ipv.plt2html()