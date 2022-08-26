import MFI1Dgame as MFI
import run_plumedgame as plumed
import numpy as np
import os
import matplotlib.pyplot as plt
# from enum import Enum
import time


# class parameters(Enum):
#    WIDTH = 0.1
#    HEIGHT = 1
#    BIASFACTOR = 10 
#    BW = 0.1

class MFIgame:
    
    def __init__(self):
        self.time = time.time()
        self.reset()

    
    def MFI_action(self, action):
        #Run simulation
        plumed.run_langevin1D(action[3], gaus_width=action[0], gaus_height=action[1], biasfactor=action[2])
        os.system("rm bck.*")

        #Read the Colvar File
        position = MFI.load_position(position_name="position")

        #Read the HILLS file
        HILLS=MFI.load_HILLS(hills_name="HILLS")

        #Compute the time-independent mean force
        [X, Ftot_den, Ftot, FES, AD, AAD, AD_f, AAD_f] = MFI.MFI_1D(HILLS = HILLS, position = position, bw = 0.1)

        #Append force to master
        self.master.append([Ftot_den, Ftot])
        
        #Patch master
        [X, Ftot_den, Ftot, FES, AD, AAD, AD_f, AAD_f] = MFI.patch_FES_error(self.master, X, 201)
        self.master_patch.append([Ftot_den, Ftot])
        self.aad_history.append(AAD)
        self.aad_f_history.append(AAD_f)
        
        
        # MFI.plot_FES(X, FES)

        print("AAD FES =", round(AAD,5), "  ||  AAD Force ", round(AAD_f,5))
        
        return AAD, AAD_f
    
    
    def MFI_new(self):
        #Run simulation
        plumed.run_langevin1D(100000, gaus_width=0.1, gaus_height=1, biasfactor=10)
        os.system("rm bck.*")

        #Read the Colvar File
        position = MFI.load_position(position_name="position")

        #Read the HILLS file
        HILLS=MFI.load_HILLS(hills_name="HILLS")

        #Compute the time-independent mean force
        [X, Ftot_den, Ftot, FES, AD, AAD, AD_f, AAD_f] = MFI.MFI_1D(HILLS = HILLS, position = position, bw = 0.1)

        #Append force to master
        self.master.append([Ftot_den, Ftot])
        
        #Patch master
        [X, Ftot_den, Ftot, FES, AD, AAD, AD_f, AAD_f] = MFI.patch_FES_error(self.master, X, 201)
        self.master_patch.append([Ftot_den, Ftot])
        self.aad_history.append(AAD)
        self.aad_f_history.append(AAD_f)
        
        
        # MFI.plot_FES(X, FES)

        print("AAD FES =", round(AAD,5), "  ||  AAD Force ", round(AAD_f,5))
        
        return AAD, AAD_f

    def reset(self):
        self.master = []
        self.master_patch = []
        self.aad_history = []
        self.aad_f_history = []
        
        # self.WIDTH = 0.1
        # self.HEIGHT = 1
        # self.BIASFACTOR = 5
        # self.BW = 0.1
        
        
        
