import MFI1Dgame as MFI
import run_plumedgame as plumed
import numpy as np
import os
import matplotlib.pyplot as plt
# from enum import Enum
import time

#analytical function
min_grid=-2
max_grid=2
nbins=201
grid = np.linspace(min_grid, max_grid, nbins)
y = 7*grid**4 - 23*grid**2
y = y - min(y)


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
        plumed.run_langevin1D(action[0], initial_position=action[1], gaus_width=round(action[2],5), gaus_height=round(action[3],4), biasfactor=round(action[4],2))
        # os.system("rm bck.*")

        #Read the Colvar File
        position = MFI.load_position(position_name="position")

        #Read the HILLS file
        HILLS=MFI.load_HILLS(hills_name="HILLS")

        #Compute the time-independent mean force
        [X, Ftot_den, Ftot_den2, Ftot, ofv_num] = MFI.MFI_1D(HILLS = HILLS, position = position, bw = 0.05)

        #Append force to master
        self.master.append([Ftot_den, Ftot_den2, Ftot, ofv_num])
        
        #Patch master
        [X, PD_patch, F_patch, FES, AD, AAD, OFE, AOFE] = MFI.patch_FES_AD_ofe(self.master, X, y, nbins)
        self.master_patch.append([PD_patch, F_patch, OFE])
        
        return [X, y, Ftot_den, FES, AD, AAD, OFE, AOFE]
    
    
    def MFI_new(self):
        
        #Run simulation
        plumed.run_langevin1D(25000, gaus_width=0.1, gaus_height=1, biasfactor=10)
        os.system("rm bck.*")

        #Read the Colvar File
        position = MFI.load_position(position_name="position")

        #Read the HILLS file
        HILLS=MFI.load_HILLS(hills_name="HILLS")

        #Compute the time-independent mean force
        [X, Ftot_den, Ftot_den2, Ftot, ofv_num] = MFI.MFI_1D(HILLS = HILLS, position = position, bw = 0.05)

        #Append force to master
        self.master.append([Ftot_den, Ftot_den2, Ftot, ofv_num])
        
        #Patch master
        [X, PD_patch, F_patch, FES, AD, AAD, OFE, AOFE] = MFI.patch_FES_AD_ofe(self.master, X, y, nbins)
        self.master_patch.append([PD_patch, F_patch, OFE])
        
        return [X, y, Ftot_den, FES, AD, AAD, OFE, AOFE]

    def reset(self):
        self.master = []
        self.master_patch = []

        
        # self.WIDTH = 0.1
        # self.HEIGHT = 1
        # self.BIASFACTOR = 5
        # self.BW = 0.1
        
        
        
