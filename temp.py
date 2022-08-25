import os
os.chdir("/home/antoniu/Desktop/MFI_git/MFI_master/MFI")
from pyMFI import MFI
import time
import numpy as np
import matplotlib.pyplot as plt


gridx = np.linspace(-2, 2, 200)
gridy = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(gridx, gridy)

a = 0
b = 0
bw = 0.1
bw2 = bw**2
stride = 10
const = (1 / (bw * np.sqrt(2 * np.pi) * stride))

Z = const * np.exp( - (1 / (2 * bw2)) * ((X - a) ** 2 + (Y - b) ** 2))

z = const * np.exp( - (1 / (2 * bw2)) * ((gridx - a) ** 2)) * 1



os.chdir("/home/antoniu/Desktop/MFI_development/compare_bw")
os.chdir("/home/antoniu/Desktop/MFI_git/MFI_master/MFI/Antoniu_2D_potential")

lim  = -1

HILLS=MFI.load_HILLS_2D(hills_name="HILLS_1M")
[position_x, position_y] = MFI.load_position_2D(position_name="position_1M")

Ftot_den = np.zeros_like(X)
for i in range(len(position_x[:lim])):
    Ftot_den = Ftot_den +  const * np.exp( - (1 / (2 * bw2)) * ((X - position_x[i]) ** 2 + (Y - position_y[i]) ** 2))
    if i % 50 == 0: print(i)
    
Ftot_den_plt = np.array(Ftot_den)
        
Ftot_den_plt[Ftot_den_plt < 0.4 * (len(position_x) / 10000)] = np.nan        

plt.figure(figsize=(10,10))
plt.contourf(X,Y, Ftot_den_plt, cmap="Greys")
plt.scatter(position_x[:lim], position_y[:lim], s=1)
plt.scatter(HILLS[:int(lim/10),1], HILLS[:int(lim/10),2], s=1)
plt.colorbar()
plt.show()

