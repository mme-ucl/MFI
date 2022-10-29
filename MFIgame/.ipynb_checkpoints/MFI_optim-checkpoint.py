import MFI1Dgame as MFI
import run_plumedgame as plumed
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
from scipy.optimize import minimize

from noisyopt import minimizeCompass

# torch.set_grad_enabled(True) 


x = np.linspace(-2,2,201)
y = 23*x**4 - 7*x**2
y = y - min(y)
z = np.zeros(201)


WIDTH = 0.1
HEIGHT = 0.1
BIASFACTOR = 10 
BW = 0.1

aad_history = []


def MFI_optim(input):
    #Run simulation
    plumed.run_langevin1D(50000, gaus_width=input[2], gaus_height=input[0], biasfactor=input[1])
    
    os.system("rm bck.*")

    #Read the Colvar File
    position = MFI.load_position(position_name="position")

    #Read the HILLS file
    HILLS=MFI.load_HILLS(hills_name="HILLS")

    #Compute the time-independent mean force
    [grid, Ftot_den, Ftot, FES, AD, AAD, AD_f, AAD_f] = MFI.MFI_1D(HILLS = HILLS, position = position, bw = 0.1)
    
    aad_history.append(AAD)
    
    print("width:", input[2], "   |||  height: ", input[0], "   |||  bf: ", input[1], "   |||  AAD: ", AAD)
    
    return AAD


res = minimize(MFI_optim, [1, 5, 0.1],  options={"disp":True})

print(res.x)

print("\n\n")

print(res)

plt.plot(range(len(aad_history)), aad_history)
plt.xlabel("Itterations")
plt.ylabel("AAD of FES to theoratical y")
plt.title("Parameter optimisation\nusing scipy")
plt.show()

