
import numpy as np
import matplotlib.pyplot as plt
from pyMFI import MFI1D

#Read the HILLS file
HILLS=MFI1D.load_HILLS(hills_name="HILLS")

#Read the Colvar File
position = MFI1D.load_position(position_name="COLVAR")

#Compute the time-independent mean force
[X, Ftot_den, Ftot, ofe_map, ofe_history] = MFI1D.MFI_1D(HILLS = HILLS, position = position, bw = 0.02, kT = 1, log_pace = 50, error_pace = 500, min_grid=-10, max_grid=10)

# Integrate Ftot, obtain FES 
FES = MFI1D.intg_1D(X,Ftot)

# Plot Recap
MFI1D.plot_recap(X, FES, Ftot_den, ofe_map, ofe_history)