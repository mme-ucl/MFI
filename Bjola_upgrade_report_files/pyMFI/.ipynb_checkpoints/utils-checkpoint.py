
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



#Load the HILLS data in 1D
def load_HILLS_1D(hills_name = "HILLS"):

    for file in glob.glob(hills_name):
        hills = np.loadtxt(file)
        hills = np.concatenate(([hills[0]], hills[:-1]))
        hills[0][3] = 0
    return hills

#Load the trajectory (position) data
def laod_position_1D(position_name = "position"):
    for file1 in glob.glob(position_name):
        colvar = np.loadtxt(file1)
    return colvar[:-1, 1]

def laod_positiontime_1D(position_name = "position"):
    for file1 in glob.glob(position_name):
        colvar = np.loadtxt(file1)
        time = colvar[:-1, 0]
    return time

def load_HILLS_2D(hills_name = "HILLS"):
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