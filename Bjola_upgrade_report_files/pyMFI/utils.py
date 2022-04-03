
import numpy as np
import os
import statistics
import scipy.integrate as integrate
import plumed
import pandas as pd
from labellines import labelLines
from labellines import labelLine
import scipy.io
import matplotlib as mpl

#Load the HILLS data
def load_HILLS_1D(hills_name = "HILLS"):
    for file in glob.glob(hills_name):
        hills = np.loadtxt(file)
        hills = hills[:-1]
        hills0 = hills[0]
        hills0[3] = 0
        hills = np.concatenate(([hills0],hills))
    return hills

#Load the trajectory (position) data
def load_position_1D(position_name = "position"):
    for file1 in glob.glob(position_name):
        colvar = np.loadtxt(file1)
    return colvar[:-1, 1]

### Algorithm to run 1D MFI
def MFI_1D_simple():
    #initialise force terms
    Fbias = np.zeros(len(x));
    Ftot_num = np.zeros(len(x));
    Ftot_den = np.zeros(len(x));

    for i in range(total_number_of_hills):
        # Build metadynamics potential
        s = HILLS[i, 1]  # center position of gausian
        sigma_meta2 = HILLS[i, 2] ** 2  # width of gausian
        gamma = HILLS[i, 4]  #scaling factor of gausian
        height_meta = HILLS[i, 3] * ((gamma - 1) / (gamma))  # Height of Gausian
        kernelmeta = np.exp(-0.5 * (((x - s) ** 2) / (sigma_meta2)))
        Fbias = Fbias + height_meta * kernelmeta * ((x - s) / (sigma_meta2)) #Bias force due to Metadynamics potentials

        # Estimate the biased proabability density
        pb_t = np.zeros(len(x))
        Fpbt = np.zeros(len(x))
        data = position[i * stride: (i + 1) * stride] #positons of window of constant bias force.
        for j in range(stride):
            kernel = const * np.exp(- (x - data[j])**2 / (2*bw2) ) #probability density of 1 datapoint
            pb_t = pb_t + kernel #probability density of window 
            Fpbt = Fpbt + kT * kernel * (x - data[j]) / bw2

        # Estimate of the Mean Force
        Ftot_den = Ftot_den + pb_t   #total probability density
        dfds = np.divide(Fpbt, pb_t, out=np.zeros_like(Fpbt), where=pb_t != 0) + Fbias
        Ftot_num = Ftot_num + pb_t * dfds
        Ftot = np.divide(Ftot_num, Ftot_den, out=np.zeros_like(Ftot_num), where=Ftot_den != 0) #total force

        if (i+1) % (total_number_of_hills/10) == 0:
            print(str(i+1) + " / " + str(total_number_of_hills))
            
    return [Ftot_den, Ftot]

        
# Integrate Ftot, obtain FES 
def intg_1D(F):
    fes = []
    for j in range(len(x)): fes.append(integrate.simps(F[:j + 1], x[:j + 1]))
    fes = fes - min(fes)
    return fes

# Calculate average deviation of 1D FES
def error_1D(FES):
    AD = abs(FES - y)
    AAD = sum(AD) / len(AD)
    print("The AAD of the FES is: " + str(AAD))
    return (AD, AAD)

#define indexing
def index(position, min_grid=min_x, _grid_space=grid_space):
    return int((position-min_grid)//_grid_space) + 1

# Calculate average deviation of 1D FES in central region [range_min, range_max]. e.g. [-1.75, 1.75]
def error_1D_centre(FES, range_min, range_max):
    AD = abs(FES[index(-1.75):index(1.75)+1] - y[index(-1.75):index(1.75)+1])
    AAD = sum(AD) / len(AD)
    print("The AAD of the FES from x=" + str(range_min) +" to x=" + str(range_max) + " is: " + str(AAD))
    return (AD, AAD)

#Load the HILLS data in 1D
def load_HILLS_1D(hills_name = "HILLS"):
    """This function loads a HILLS

    Args:
        hills_name (str, optional): HILLS file name. Defaults to "HILLS".

    Returns:
        _type_: _description_
    """
    for file in glob.glob(hills_name):
        hills = np.loadtxt(file)
        hills = np.concatenate(([hills[0]], hills[:-1]))
        hills[0][3] = 0
    return hills

#Load the trajectory (position) data
def load_position_1D(position_name = "position"):
    """Loads the position in CV space, typically from a COLVAR file, printed at a higher frequency than the corresponding HILLS file. 

    Args:
        position_name (str, optional): Name of the COLVAR files. Defaults to "position".

    Returns:
        _type_: _description_
    """
    for file1 in glob.glob(position_name):
        colvar = np.loadtxt(file1)
    return colvar[:-1, 1]

def load_positiontime_1D(position_name = "position"):
    """Loads the timestep associated with the position in CV space, in 1D
    MS comment: this be incorporated with load_position (?)

    Args:
        position_name (str, optional): Name of the COLVAR . Defaults to "position".

    Returns:
        _type_: _description_
    """
    for file1 in glob.glob(position_name):
        colvar = np.loadtxt(file1)
        time = colvar[:-1, 0]
    return time

def load_HILLS_2D(hills_name = "HILLS"):
    """Loads a 2D HILLS file

    Args:
        hills_name (str, optional): Filename. Defaults to "HILLS".

    Returns:
        _type_: _description_
    """
    for file in glob.glob(hills_name):
        hills = np.loadtxt(file)
        hills = np.concatenate(([hills[0]], hills[:-1]))
        hills[0][5] = 0
    return hills

def laod_position_2D(position_name = "position"):
    for file1 in glob.glob(position_name):
        colvar = np.loadtxt(file1)
        position_x = colvar[:-1, 1]
        position_y = colvar[:-1, 2]
    return [position_x, position_y]

##################################################################

def save_pkl(object, file_name):
    with open(file_name, "wb") as fw:
        pickle.dump(object, fw, pickle.HIGHEST_PROTOCOL)

def load_pkl(name):
    with open(name, "rb") as fr:
        return pickle.load(fr)

def save_npy(object, file_name):
    with open(file_name, "wb") as fw:
        np.save(fw, object)

def load_npy(name):
    with open(name, "rb") as fr:
        return np.load(fr)