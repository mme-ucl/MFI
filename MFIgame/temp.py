import MFI1Dgame as MFI
import run_plumedgame as plumed
import numpy as np
import os
import matplotlib.pyplot as plt


#Run simulation
plumed.run_langevin1D(50000, gaus_width=0.1, gaus_height=1, biasfactor=5)


os.system("rm bck.*")

#Read the Colvar File
position = MFI.load_position(position_name="position")

#Read the HILLS file
HILLS=MFI.load_HILLS(hills_name="HILLS")

# #Compute the time-independent mean force
# [grid, Ftot_den, Ftot, FES, AD, AAD, AD_f, AAD_f] = MFI.MFI_1D(HILLS = HILLS, position = position, bw = 0.05)

# print("AAD: ", AAD)

# MFI.plot_FES(grid, FES)

# MFI.plot_FES(grid, AD)


plt.plot(range(len(position)), position)
plt.show()

print(np.shape(position))
