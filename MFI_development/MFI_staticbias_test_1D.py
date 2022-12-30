import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
from scipy import interpolate
from scipy.interpolate import griddata
# import scipy.integrate as integrate
# import matplotlib.colors as colors
# import time
# from matplotlib import ticker, cm
# import subprocess
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib.colors import LogNorm
# import random

data_path = "/home/ucecabj/Desktop/pyMFI_git/MFI_development/data/1D_wall_pot"
MFI_path = "/home/ucecabj/Desktop/pyMFI_git"
sys.path.insert(0, MFI_path)
from pyMFI import MFI1D
from pyMFI import run_plumed

os.chdir(data_path)
# sys.path.insert(0, data_path)
print(os.getcwd())


#Define constants 
bw = 0.05; bw2 = bw**2     
kT = 2.49         
# const = (1 / (bw*(2*np.pi)*stride))

# define grid and analytical FES  
nbins = 301
min_grid = -np.pi
max_grid = np.pi
grid_space = (max_grid - min_grid) / (nbins - 1)
grid = np.linspace(min_grid,max_grid,nbins)
X, Y = np.meshgrid(grid, grid)
#Use periodic extension for defining PBC
periodic = 1
periodic_extension = 1 / 2
grid_ext = (1/2) * periodic_extension * (max_grid-min_grid)
Flim=50

# def gaus2(amp,mue_x,sigma,grid):
#     return (amp * np.exp(-(grid - mue_x)**2 / sigma) )

# def make_rand_fes():
    

#     function  = "sin(x)" #"0.8*x^4" #+50*exp(-(x-1.4)^2)+65*exp(-(x+0.4)^2)+40*exp(-2.5*(x+2.0)^2)"
#     grid = np.linspace(min_grid,max_grid,nbins)
#     Z_plot = np.sin(grid)


#     for _ in range(1):
#         sigma = round(random.uniform(0.1, 0.9), 2)
#         mue_x = round(random.uniform(-2, 2), 2)
#         amp = round(random.uniform(-1, 1), 2)
#         function = function + "+(" + str(amp) + "*exp(-(x-(" + str(mue_x) + "))^2/" + str(sigma) + "))"
#         Z_plot += gaus2(amp,mue_x,sigma,grid)

#     Z_plot = Z_plot - Z_plot.min()
    
#     return grid, function, Z_plot


# grid, y_string, y = make_rand_fes()

# print(y_string)

# plt.plot(grid, y)
# plt.plot([i + 2*np.pi for i in grid], y)
# plt.show()
# exit()

master = []
master_patch = []


os.chdir(data_path)

# run_plumed.find_alanine_dipeptide_input()
# run_plumed.run_alanine_dipeptide_1D(simulation_steps = 500000, tors_angle="phi", gaus_width=0.1, grid_min="-pi", grid_max="pi", gaus_height=3, biasfactor=10)
HILLS=MFI1D.load_HILLS(hills_name="HILLS")
position = MFI1D.load_position(position_name="position")

[X, Ftot_den, Ftot_den2, Ftot, ofv_num, FES, ofv, ofe, cutoff, ofv_history, ofe_history, time_history] = MFI1D.MFI_1D(
                                HILLS = HILLS, position = position, bw = 0.05, kT = 1, periodic=1,
                                log_pace = 1, error_pace = 5, min_grid=min_grid, max_grid=max_grid, nbins=nbins,
                                WellTempered=1, truncation_limit=10**-10, FES_cutoff=50)



#Append force to master
master.append([Ftot_den, Ftot_den2, Ftot, ofv_num])

#Patch master
# [X, PD_patch, F_patch, FES, AD, AAD, OFE, AOFE] = MFI1D.patch_FES_AD_ofe(master, X, y, nbins)
[X, PD_patch, F_patch, FES, OFE, AOFE] = MFI1D.patch_FES_ofe(master, X, nbins)
master_patch.append([PD_patch, F_patch, OFE])

cutoff = np.ones_like(FES)

# plt.figure(1)
# MFI1D.plot_recap(X, MFI1D.zero_to_nan(FES*cutoff), MFI1D.zero_to_nan(Ftot_den*cutoff), MFI1D.zero_to_nan(ofe*cutoff), ofe_history, time_history, FES_lim=50, ofe_lim = 15, error_log_scale=0)
# plt.show()


###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# # run_plumed.find_alanine_dipeptide_input()
# run_plumed.run_alanine_dipeptide_1D(simulation_steps = 200000, tors_angle="phi", gaus_width=0.05, grid_min="-pi", grid_max="pi", gaus_height=0.1, biasfactor=10, hp_centre=0.0, hp_kappa=100, print_bias=1, file_extension="_HP")
HILLS=MFI1D.load_HILLS(hills_name="HILLS_HP")
position = MFI1D.load_position(position_name="position_HP")

[X, Ftot_den, Ftot_den2, Ftot, ofv_num, FES, ofv, ofe, cutoff, ofv_history, ofe_history, time_history] = MFI1D.MFI_1D(
                                HILLS = HILLS, position = position, bw = 0.05, kT = 1, periodic=1, hp_centre=0.0, hp_kappa=100,
                                log_pace = 1, error_pace = 5, min_grid=min_grid, max_grid=max_grid, nbins=nbins, 
                                WellTempered=1, truncation_limit=10**-10, FES_cutoff=50)

cutoff = np.ones_like(FES)
# plt.figure(2)
# MFI1D.plot_recap(X, MFI1D.zero_to_nan(FES*cutoff), MFI1D.zero_to_nan(Ftot_den*cutoff), MFI1D.zero_to_nan(OFE*cutoff), ofe_history, time_history, FES_lim=50, ofe_lim = 15, error_log_scale=0)
# plt.show()
# exit()

#Append force to master
master.append([Ftot_den, Ftot_den2, Ftot, ofv_num])

#Patch master
# [X, PD_patch, F_patch, FES, AD, AAD, OFE, AOFE] = MFI1D.patch_FES_AD_ofe(master, X, y, nbins)
[X, PD_patch, F_patch, FES, OFE, AOFE] = MFI1D.patch_FES_ofe(master, X, nbins)
master_patch.append([PD_patch, F_patch, OFE])

# plt.figure(3)
# MFI1D.plot_recap(X, MFI1D.zero_to_nan(FES*cutoff), MFI1D.zero_to_nan(PD_patch*cutoff), MFI1D.zero_to_nan(OFE*cutoff), ofe_history, time_history, FES_lim=50, ofe_lim = 15, error_log_scale=0)
# plt.show()

def index(position, min_grid, grid_space):
    """Finds (approximate) index of a position in a grid. Independent of CV-type.

    Args:
        position (float): position of interest
        min_grid (float): minimum value of grid
        grid_space (float): grid spacing

    Returns:
        int: index of position
    """
    return int((position-min_grid)//grid_space) + 1


def find_hp_force_and_energy(hp_centre, hp_kappa, grid, min_grid, max_grid, grid_space, periodic):
    """Find 1D harmonic potential force. 

    Args:
        hp_centre (float): position of harmonic potential
        hp_kappa (float): force_constant of harmonic potential
        grid (array): CV grid positions
        min_grid (float): minimum value of grid
        max_grid (float): maximum value of grid
        grid_space (float): space between two consecutive grid values
        periodic (binary): information if system is periodic. value of 0 corresponds to non-periodic system. Value of 1 corresponds to periodic system.

    Returns:
        array: harmonic force array
    """
    F_harmonic = hp_kappa * (grid - hp_centre)
    P_harmonic = 0.5 * hp_kappa * (grid - hp_centre)**2
    if periodic == 1:
        grid_length = max_grid - min_grid
        grid_centre = min_grid + grid_length/2
        if hp_centre < grid_centre:
            index_period = index(hp_centre + grid_length / 2, min_grid, grid_space)
            F_harmonic[index_period:] = hp_kappa * (grid[index_period:] - hp_centre - grid_length)
            P_harmonic[index_period:] = 0.5 * hp_kappa * (grid[index_period:] - hp_centre - grid_length)**2
        elif hp_centre > grid_centre:
            index_period = index(hp_centre - grid_length / 2, min_grid, grid_space)
            F_harmonic[:index_period] = hp_kappa * (grid[:index_period] - hp_centre + grid_length)
            P_harmonic[:index_period] = 0.5 * hp_kappa * (grid[:index_period] - hp_centre + grid_length)**2

    return F_harmonic, P_harmonic



F_harmonic, P_harmonic = find_hp_force_and_energy(0.0, 100, grid, min_grid, max_grid, grid_space, periodic=1)


for file1 in glob.glob("restraint_HP"):
    data = np.loadtxt(file1)
    restr_pos = data[:, 1]
    plumed_bias = data[:, 2]
    plumed_force2 = data[:, 3]


bias_error = []
bias_interpolate = interpolate.interp1d(grid, P_harmonic, kind="quadratic")
P_interpolate = [ bias_interpolate(restr_pos[j]) for j in range(len(restr_pos))]
for ii in range(len(plumed_bias)):
    bias_error.append(abs(P_interpolate[ii] - plumed_bias[ii])/plumed_bias[ii]*100)
# bias_error = np.array(bias_error)#.reshape((np.shape(bias_error)[0],))


force_error = []
force_interpolate = interpolate.interp1d(grid, F_harmonic, kind="quadratic")
F_interpolate = [force_interpolate(restr_pos[j]) for j in range(len(restr_pos))]
for ii in range(len(plumed_force2)):
    force_error.append(np.sqrt(abs(F_interpolate[ii]**2 - plumed_force2[ii])/plumed_force2[ii])*100)
# bias_error = np.array(force_error)#.reshape((np.shape(force_error)[0],))


if 1 == 1: #sum(bias_error) / (len(bias_error)) > 1:
    # print("\n\n******** ATTENTION, BIAS DIFFERENCE IS HIGH **********")
    print("\nPlumed Bias vs theo. Bias [%%-diff]: " , round(sum(bias_error)/(len(bias_error)),5), "%")
    print("SQRT( Plumed Force vs theo. Force) [%%-diff]: " , round(sum(force_error)/(len(force_error)),5), "%\n")

    bias_error_interploation = griddata(restr_pos, np.array(bias_error), grid, method="linear")
    force_error_interploation = griddata(restr_pos, np.array(force_error), grid, method="linear")

    plumed_bias_interpolate = griddata(restr_pos, np.array(plumed_bias), grid, method="linear")
    plumed_force_interpolate = griddata(restr_pos, np.array(plumed_force2), grid, method="linear")

    fig_restr = plt.figure(22, figsize=plt.figaspect(0.3))

    plt.subplot(1,3,1)
    plt.plot(grid, bias_error_interploation, label="bias_error_interploation")
    plt.plot(grid, force_error_interploation, label="force_error_interploation")
    plt.xlim(-np.pi, np.pi)
    plt.ylabel("percentage differnce")
    plt.xlabel("CV")
    plt.legend()

    plt.subplot(1,3,2)
    plt.plot(grid, plumed_bias_interpolate, label="plumed_bias_interpolate")
    plt.scatter(grid, P_harmonic, label="P_harmonic", c="r")
    # plt.plot(grid, plumed_force_interpolate, label="plumed_force_interpolate")
    # plt.plot(grid, F_harmonic**2, label="F_harmonic**2")
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-1, 70)
    plt.ylabel("hamrmonic potential")
    plt.xlabel("CV")
    plt.legend()
    
    plt.subplot(1,3,3)
    # plt.plot(grid, P_harmonic)
    plt.plot(grid, plumed_force_interpolate, label="plumed_force_interpolate")
    plt.scatter(grid, F_harmonic**2, label="F_harmonic**2", c="r", s=20)
    plt.scatter(restr_pos, [F_interpolate[i]**2 for i in range(len(F_interpolate))], c="g", label="F_interpolate**2", s=15)
    plt.scatter(restr_pos, plumed_force2, color="cyan", label="plumed_force2", s=10)
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-1, 10000)
    plt.ylabel("force**2")
    plt.xlabel("CV")
    plt.legend()
    plt.tight_layout()


###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# run_plumed.find_alanine_dipeptide_input()
# run_plumed.run_alanine_dipeptide_1D(simulation_steps = 200000, tors_angle="phi", gaus_width=0.1, grid_min="-pi", grid_max="pi", gaus_height=3, biasfactor=10, hp_centre=2.5, hp_kappa=100, print_bias=1, file_extension="_HP2")
HILLS=MFI1D.load_HILLS(hills_name="HILLS_HP2")
position = MFI1D.load_position(position_name="position_HP2")

[X, Ftot_den, Ftot_den2, Ftot, ofv_num, FES, ofv, ofe, cutoff, ofv_history, ofe_history, time_history] = MFI1D.MFI_1D(
                                HILLS = HILLS, position = position, bw = 0.05, kT = 1, periodic=1, hp_centre=2.5, hp_kappa=100,
                                log_pace = 1, error_pace = 5, min_grid=min_grid, max_grid=max_grid, nbins=nbins, 
                                WellTempered=1, truncation_limit=10**-10, FES_cutoff=50)

cutoff = np.ones_like(FES)
# plt.figure(2)
# MFI1D.plot_recap(X, MFI1D.zero_to_nan(FES*cutoff), MFI1D.zero_to_nan(Ftot_den*cutoff), MFI1D.zero_to_nan(OFE*cutoff), ofe_history, time_history, FES_lim=70, ofe_lim = 15, error_log_scale=0)
# plt.show()
# exit()

#Append force to master
master.append([Ftot_den, Ftot_den2, Ftot, ofv_num])

#Patch master
# [X, PD_patch, F_patch, FES, AD, AAD, OFE, AOFE] = MFI1D.patch_FES_AD_ofe(master, X, y, nbins)
[X, PD_patch, F_patch, FES, OFE, AOFE] = MFI1D.patch_FES_ofe(master, X, nbins)
master_patch.append([PD_patch, F_patch, OFE])


F_harmonic, P_harmonic = find_hp_force_and_energy(2.5, 100, grid, min_grid, max_grid, grid_space, periodic=1)


for file1 in glob.glob("restraint_HP2"):
    data = np.loadtxt(file1)
    restr_pos = data[:, 1]
    plumed_bias = data[:, 2]
    plumed_force2 = data[:, 3]


bias_error = []
bias_interpolate = interpolate.interp1d(grid, P_harmonic, kind="quadratic")
P_interpolate = [ bias_interpolate(restr_pos[j]) for j in range(len(restr_pos))]
for ii in range(len(plumed_bias)):
    bias_error.append(abs(P_interpolate[ii] - plumed_bias[ii])/plumed_bias[ii]*100)
# bias_error = np.array(bias_error)#.reshape((np.shape(bias_error)[0],))


force_error = []
force_interpolate = interpolate.interp1d(grid, F_harmonic, kind="quadratic")
F_interpolate = [force_interpolate(restr_pos[j]) for j in range(len(restr_pos))]
for ii in range(len(plumed_force2)):
    force_error.append(np.sqrt(abs(F_interpolate[ii]**2 - plumed_force2[ii])/plumed_force2[ii])*100)
# bias_error = np.array(force_error)#.reshape((np.shape(force_error)[0],))


if 1 == 1: #sum(bias_error) / (len(bias_error)) > 1:
    # print("\n\n******** ATTENTION, BIAS DIFFERENCE IS HIGH **********")
    print("\nPlumed Bias vs theo. Bias [%%-diff]: " , round(sum(bias_error)/(len(bias_error)),5), "%")
    print("SQRT( Plumed Force vs theo. Force) [%%-diff]: " , round(sum(force_error)/(len(force_error)),5), "%\n")

    bias_error_interploation = griddata(restr_pos, np.array(bias_error), grid, method="linear")
    force_error_interploation = griddata(restr_pos, np.array(force_error), grid, method="linear")

    plumed_bias_interpolate = griddata(restr_pos, np.array(plumed_bias), grid, method="linear")
    plumed_force_interpolate = griddata(restr_pos, np.array(plumed_force2), grid, method="linear")

    fig_restr = plt.figure(23, figsize=plt.figaspect(0.3))

    plt.subplot(1,3,1)
    plt.plot(grid, bias_error_interploation, label="bias_error_interploation")
    plt.plot(grid, force_error_interploation, label="force_error_interploation")
    plt.xlim(-np.pi, np.pi)
    plt.ylabel("percentage differnce")
    plt.xlabel("CV")
    plt.legend()

    plt.subplot(1,3,2)
    plt.plot(grid, plumed_bias_interpolate, label="plumed_bias_interpolate")
    plt.scatter(grid, P_harmonic, label="P_harmonic", c="r")
    # plt.plot(grid, plumed_force_interpolate, label="plumed_force_interpolate")
    # plt.plot(grid, F_harmonic**2, label="F_harmonic**2")
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-1, 70)
    plt.ylabel("hamrmonic potential")
    plt.xlabel("CV")
    plt.legend()
    
    plt.subplot(1,3,3)
    # plt.plot(grid, P_harmonic)
    plt.plot(grid, plumed_force_interpolate, label="plumed_force_interpolate")
    plt.scatter(grid, F_harmonic**2, label="F_harmonic**2", c="r", s=20)
    plt.scatter(restr_pos, [F_interpolate[i]**2 for i in range(len(F_interpolate))], c="g", label="F_interpolate**2", s=15)
    plt.scatter(restr_pos, plumed_force2, color="cyan", label="plumed_force2", s=10)
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-1, 10000)
    plt.ylabel("force**2")
    plt.xlabel("CV")
    plt.legend()
    plt.tight_layout()



# plt.figure(4)
MFI1D.plot_recap(X, MFI1D.zero_to_nan(FES*cutoff), MFI1D.zero_to_nan(PD_patch*cutoff), MFI1D.zero_to_nan(OFE*cutoff), ofe_history, time_history, FES_lim=70, ofe_lim = 15, error_log_scale=0)
plt.show()