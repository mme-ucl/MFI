#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 12:00:49 2019

@author: veselina
"""
import numpy as np
from numpy import seterr,isneginf,array
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import scipy.io as spio
from matplotlib import cm


# close previous figures
plt.close('all')
# END

FES = spio.loadmat('BIG_FES.mat', squeeze_me=True) 
FES=np.array(FES['BIG_FES'])

pi=3.1415
min_grid=[2.7,-0.5,-pi]
max_grid=[5.3,5.6,pi]
nbins=50
kT=2.49
b=1/kT

gridx=np.linspace(min_grid[0],max_grid[0],nbins)
gridy=np.linspace(min_grid[1],max_grid[1],nbins)
gridz=np.linspace(min_grid[2],max_grid[2],nbins)
X, Y, Z = np.meshgrid(gridx, gridy, gridz)


#remove very high FES values
#default_value = np.NaN
#FES[(FES >100)] = default_value

FES_new=np.reshape(FES,125000)
P=np.exp(-FES*b)
P=P/np.sum(P)

P_new=np.exp(-FES_new*b)
P_new=P_new/sum(P_new)

P_cryst=0
P_flipped=0

for i in range(nbins):
    for j in range(nbins):
        for k in range(nbins):
            if (X[i,j,k]<3.3 and Z[i,j,k]>2.8) or (X[i,j,k]<3.3 and Z[i,j,k]<-2.8):
                P_cryst=P_cryst+P[i,j,k]
            #    print(gridx[i],gridz[k],P[i,j,k])
            elif (X[i,j,k]<3.4 and Z[i,j,k]>-0.5) or (X[i,j,k]<3.4 and Z[i,j,k]<0.5):
                P_flipped=P_flipped+P[i,j,k]
    

P_cryst=P_cryst*100
P_flipped=P_flipped*100
fig = plt.figure()
ax = fig.gca(projection='3d')
p = ax.scatter3D(X, Y, Z, c=P_new, s=2, cmap=plt.viridis()) 
fig.colorbar(p)

ax.set_facecolor('white')                
#ax.set_xlabel(r' distance [nm]', fontsize=18)
#ax.set_ylabel(r' coordination number', fontsize=18)
#ax.set_zlabel(r' alignment angle [rad]', fontsize=18)

plt.tight_layout()
#z_ticks=[-3.1415,-1.5708,0,1.5708,3.1415]
#plt.zticks(z_ticks,('-$\pi$','-$\pi$/2','0','$\pi$/2','$\pi$'))
#plt.savefig('FES_water_example.png',dpi=500, transparent=True, edgecolor='w',orientation='portrait')

