import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
import scipy.integrate as integrate
import matplotlib.colors as colors
import time
import plumed
from scipy import interpolate
from matplotlib import ticker, cm
import subprocess
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from matplotlib.colors import LogNorm



# import sklearn
# from MFI_functions import *


# #Take reference pdb file |->  create structure.gro and topology.top
# os.system("gmx pdb2gmx -f reference.pdb -o structure.gro -p topology.top")
# #Prepare input for simulation (input0.tpr)
# os.system("gmx grompp -f gromppvac.mdp -c structure.gro -p topology.top -o input0.tpr")
# #Run simulation with 10^6 steps
# os.system("gmx mdrun -s input0.tpr -nsteps 1000000 -plumed plumed_first.dat")

# with open("AP_10E9_fes.npy","rb") as fr:
#     FES10E9 = np.load(fr)

#get reference fes
data = np.genfromtxt('fes_ap_10E9.dat') #np.load('AP_10E9_fes.npy')

FREF=np.reshape(data[:,2],(60,60));
XREF=np.reshape(data[:,0],(60,60));
YREF=np.reshape(data[:,1],(60,60));

FREF = FREF - np.min(FREF)

#Change value of pi. min(XREF) is too small
XREF[XREF == -3.141593] = -np.pi
YREF[YREF == -3.141593] = -np.pi


#Define constants ####################################################################################################################################################################
bw = 0.05; bw2 = bw**2        # bw: bandwidth for the KDE of the biased probability density
kT = 2.49          # kT:       value of kT
# const = (1 / (bw*(2*np.pi)*stride))

# define grid and analytical FES  ####################################################################################################################################################################
nbins = 201 # number of bin in grid  #---------------------------------------
min_grid = -np.pi
max_grid = np.pi
grid_space = (max_grid - min_grid) / (nbins - 1)
grid = np.linspace(min_grid,max_grid,nbins)
X, Y = np.meshgrid(grid, grid)
#Use periodic extension for defining PBC
periodic_extension = 1 / 2
grid_ext = (1/2) * periodic_extension * (max_grid-min_grid)
Flim=50


def find_periodic_point(x_coord,y_coord):
    coord_list = []
    #There are potentially 4 points, 1 original and 3 periodic copies
    coord_list.append([x_coord,y_coord])
    copy_record = [0,0,0,0]
    #check for x-copy
    if x_coord < min_grid+grid_ext:
        coord_list.append([x_coord + 2*np.pi,y_coord])
        copy_record[0] = 1
    elif x_coord > max_grid-grid_ext:
        coord_list.append([x_coord - 2*np.pi,y_coord])
        copy_record[1] = 1
    #check for y-copy
    if y_coord < min_grid+grid_ext:
        coord_list.append([x_coord, y_coord + 2 * np.pi])
        copy_record[2] = 1
    elif y_coord > max_grid-grid_ext:
        coord_list.append([x_coord, y_coord - 2 * np.pi])
        copy_record[3] = 1
    #check for xy-copy
    if sum(copy_record) == 2:
        if copy_record[0] == 1 and copy_record[2] == 1: coord_list.append([x_coord + 2 * np.pi, y_coord + 2 * np.pi])
        elif copy_record[1] == 1 and copy_record[2] == 1: coord_list.append([x_coord - 2 * np.pi, y_coord + 2 * np.pi])
        elif copy_record[0] == 1 and copy_record[3] == 1: coord_list.append([x_coord + 2 * np.pi, y_coord - 2 * np.pi])
        elif copy_record[1] == 1 and copy_record[3] == 1: coord_list.append([x_coord - 2 * np.pi, y_coord - 2 * np.pi])

    return coord_list


def find_cutoff_matrix(input_FES):
    len_x, len_y = np.shape(input_FES)
    cutoff_matrix = np.ones((len_x, len_y))
    for ii in range(len_x):
        for jj in range(len_y):
            if input_FES[ii][jj] >= Flim: cutoff_matrix[ii][jj] = 0
    return cutoff_matrix


def zero_to_nan(input_array):
    len_x, len_y = np.shape(input_array)
    for ii in range(len_x):
        for jj in range(len_y):
            if input_array[ii][jj] == 0: input_array[ii][jj] = float("Nan")
    return input_array


def find_FES_adj(X_old, Y_old, FES_old):
    # r = np.stack(["x_old_grid_mesh".ravel(), "y_old_grid_mesh".ravel()]).T
    r = np.stack([X_old.ravel(), Y_old.ravel()]).T
    # Sx = interpolate.CloughTocher2DInterpolator(r, "Z_values".ravel())
    Sx = interpolate.CloughTocher2DInterpolator(r, FES_old.ravel())
    # ri = np.stack(["x_new_grid_mesh".ravel(), "y_new_grid_mesh".ravel()]).T
    ri = np.stack([XREF.ravel(), YREF.ravel()]).T
    FES_new = Sx(ri).reshape(XREF.shape)

    return FES_new

# def find_FES_adj_plus(FES_old):
#     # r = np.stack(["x_old_grid_mesh".ravel(), "y_old_grid_mesh".ravel()]).T
#     r = np.stack([PHIi.ravel(), PSIi.ravel()]).T
#     # Sx = interpolate.CloughTocher2DInterpolator(r, "Z_values".ravel())
#     Sx = interpolate.CloughTocher2DInterpolator(r, FES_old.ravel())
#     # ri = np.stack(["x_new_grid_mesh".ravel(), "y_new_grid_mesh".ravel()]).T
#     ri = np.stack([XREF.ravel(), YREF.ravel()]).T
#     FES_new = Sx(ri).reshape(XREF.shape)
#
#     return FES_new


def MSintegral(FX,FY,X_old=X, Y_old=Y, i_bins=(nbins,nbins)):

    if i_bins != (nbins,nbins):

        r = np.stack([X_old.ravel(), Y_old.ravel()]).T
        Sx = interpolate.CloughTocher2DInterpolator(r, FX.ravel())
        Sy = interpolate.CloughTocher2DInterpolator(r, FY.ravel())
        Nx, Ny = i_bins

        x_new = np.linspace(grid.min(), grid.max(), Nx)
        y_new = np.linspace(grid.min(), grid.max(), Ny)
        X_new, Y_new = np.meshgrid(x_new, y_new)

        ri = np.stack([X_new.ravel(), Y_new.ravel()]).T
        FX = Sx(ri).reshape(X_new.shape)
        FY = Sy(ri).reshape(Y_new.shape)

        grid_diff = np.diff(x_new)[0]

    else:
        grid_diff = grid_space
        X_new = X_old
        Y_new = Y_old


    SdZx = np.cumsum(FX, axis=1) * grid_diff  # cumulative sum along x-axis
    SdZy = np.cumsum(FY, axis=0) * grid_diff  # cumulative sum along y-axis
    SdZx3 = np.cumsum(FX[::-1], axis=1) * grid_diff  # cumulative sum along x-axis
    SdZy3 = np.cumsum(FY[::-1], axis=0) * grid_diff  # cumulative sum along y-axis
    SdZx5 = np.cumsum(FX[:, ::-1], axis=1) * grid_diff  # cumulative sum along x-axis
    SdZy5 = np.cumsum(FY[:, ::-1], axis=0) * grid_diff  # cumulative sum along y-axis
    SdZx7 = np.cumsum(FX[::-1, ::-1], axis=1) * grid_diff  # cumulative sum along x-axis
    SdZy7 = np.cumsum(FY[::-1, ::-1], axis=0) * grid_diff  # cumulative sum along y-axis


    FES = np.zeros(i_bins)
    FES2 = np.zeros(i_bins)
    FES3 = np.zeros(i_bins)
    FES4 = np.zeros(i_bins)
    FES5 = np.zeros(i_bins)
    FES6 = np.zeros(i_bins)
    FES7 = np.zeros(i_bins)
    FES8 = np.zeros(i_bins)

    for i in range(FES.shape[0]):
        for j in range(FES.shape[1]):
            FES[i, j]  += np.sum([SdZy[i, 0], -SdZy[0, 0], SdZx[i, j], -SdZx[i, 0]])
            FES2[i, j] += np.sum([SdZx[0, j], -SdZx[0, 0], SdZy[i, j], -SdZy[0, j]])
            FES3[i, j] += np.sum([-SdZy3[i, 0], SdZy3[0, 0], SdZx3[i, j], -SdZx3[i, 0]])
            FES4[i, j] += np.sum([SdZx3[0, j], -SdZx3[0, 0], -SdZy3[i, j], SdZy3[0, j]])
            FES5[i, j] += np.sum([SdZy5[i, 0], -SdZy5[0, 0], -SdZx5[i, j], SdZx5[i, 0]])
            FES6[i, j] += np.sum([-SdZx5[0, j], SdZx5[0, 0], SdZy5[i, j], -SdZy5[0, j]])
            FES7[i, j] += np.sum([-SdZy7[i, 0], SdZy7[0, 0], -SdZx7[i, j], SdZx7[i, 0]])
            FES8[i, j] += np.sum([-SdZx7[0, j], SdZx7[0, 0], -SdZy7[i, j], SdZy7[0, j]])

    FES = FES - np.min(FES)
    FES2 = FES2 - np.min(FES2)
    FES3 = FES3[::-1] - np.min(FES3)
    FES4 = FES4[::-1] - np.min(FES4)
    FES5 = FES5[:,::-1] - np.min(FES5)
    FES6 = FES6[:,::-1] - np.min(FES6)
    FES7 = FES7[::-1,::-1] - np.min(FES7)
    FES8 = FES8[::-1,::-1] - np.min(FES8)
    FES_a = (FES + FES2 + FES3 + FES4 + FES5 + FES6 + FES7 + FES8) / 8
    FES_a = FES_a - np.min(FES_a)

    if i_bins != (nbins,nbins):
        return (FES_a, X_new, Y_new)
    else:
        return FES_a


def FWIntegration(FX, FY, i_bins=(nbins,nbins)):

    if i_bins == (nbins,nbins):
        freq_1d = np.fft.fftfreq(nbins, grid_space)
        freq_x, freq_y = np.meshgrid(freq_1d, freq_1d)

    else: #interpolate
        grid_new_x = np.linspace(grid.min(), grid.max(), i_bins[0])
        grid_new_y = np.linspace(grid.min(), grid.max(), i_bins[1])
        X_new, Y_new = np.meshgrid(grid_new_x, grid_new_y)
        grid_space_x = (grid.max() - grid.min()) / (i_bins[0] - 1)
        grid_space_y = (grid.max() - grid.min()) / (i_bins[1] - 1)

        r = np.stack([X.ravel(), Y.ravel()]).T
        Sx = interpolate.CloughTocher2DInterpolator(r, FX.ravel())
        Sy = interpolate.CloughTocher2DInterpolator(r, FY.ravel())
        ri = np.stack([X_new.ravel(), Y_new.ravel()]).T

        FX = Sx(ri).reshape(X_new.shape)
        FY = Sy(ri).reshape(Y_new.shape)

        freq_1d_x = np.fft.fftfreq(i_bins[0], grid_space_x)
        freq_1d_y = np.fft.fftfreq(i_bins[1], grid_space_y)
        freq_x, freq_y = np.meshgrid(freq_1d_x, freq_1d_y)


    freq_hypot = np.hypot(freq_x, freq_y)
    freq_sq = np.where(freq_hypot != 0, freq_hypot ** 2, 1E-10)

    fourier_x = (np.fft.fft2(FX) * freq_x) / (2 * np.pi * 1j * freq_sq)
    fes_x = np.real(np.fft.ifft2(fourier_x))

    fourier_y = (np.fft.fft2(FY) * freq_y) / (2 * np.pi * 1j * freq_sq)
    fes_y = np.real(np.fft.ifft2(fourier_y))

    fes = fes_x + fes_y
    fes = fes - np.min(fes)

    if i_bins == (nbins, nbins): return fes
    else: return (X_new, Y_new, fes)


def find_hp_force(hp_centre_x, hp_centre_y, hp_kappa_x, hp_kappa_y, X , Y, min_grid, max_grid, grid_space, periodic):
    #Calculate x-force
    F_harmonic_x = hp_kappa_x * (X - hp_centre_x)
    if periodic == 1:
        grid_length = max_grid[0] - min_grid[0]
        grid_centre = min_grid[0] + grid_length/2
        if hp_centre_x < grid_centre:
            index_period = index(hp_centre_x + grid_length/2, min_grid[0], grid_space[0])
            F_harmonic_x[:, index_period:] = hp_kappa_x * (X[:, index_period:] - hp_centre_x - grid_length)
        elif hp_centre_x > grid_centre:
            index_period = index(hp_centre_x - grid_length/2, min_grid[0], grid_space[0])
            F_harmonic_x[:, :index_period] = hp_kappa_x * (X[:, :index_period] - hp_centre_x + grid_length)
    #Calculate y-force
    F_harmonic_y = hp_kappa_y * (Y - hp_centre_y)
    if periodic == 1:
        grid_length = max_grid[0] - min_grid[0]
        grid_centre = min_grid[0] + grid_length / 2
        if hp_centre_y < grid_centre:
            index_period = index(hp_centre_y + grid_length/2, min_grid[1], grid_space[1])
            F_harmonic_y[index_period:, :] = hp_kappa_y * (Y[index_period:, :] - hp_centre_y - grid_length)
        elif hp_centre_y > grid_centre:
            index_period = index(hp_centre_y - grid_length/2, min_grid[1], grid_space[1])
            F_harmonic_y[:index_period, :] = hp_kappa_y * (Y[:index_period, :] - hp_centre_y + grid_length)


    #Calculate x-force
    P_harmonic_x = hp_kappa_x/2 * (X - hp_centre_x)**2
    if periodic == 1:
        grid_length = max_grid[0] - min_grid[0]
        grid_centre = min_grid[0] + grid_length/2
        if hp_centre_x < grid_centre:
            index_period = index(hp_centre_x + grid_length/2, min_grid[0], grid_space[0])
            P_harmonic_x[:, index_period:] = hp_kappa_x/2 * (X[:, index_period:] - hp_centre_x - grid_length)**2
        elif hp_centre_x > grid_centre:
            index_period = index(hp_centre_x - grid_length/2, min_grid[0], grid_space[0])
            P_harmonic_x[:, :index_period] = hp_kappa_x/2 * (X[:, :index_period] - hp_centre_x + grid_length)**2
    #Calculate y-force
    P_harmonic_y = hp_kappa_y/2 * (Y - hp_centre_y)**2
    if periodic == 1:
        grid_length = max_grid[0] - min_grid[0]
        grid_centre = min_grid[0] + grid_length / 2
        if hp_centre_y < grid_centre:
            index_period = index(hp_centre_y + grid_length/2, min_grid[1], grid_space[1])
            P_harmonic_y[index_period:, :] = hp_kappa_y/2 * (Y[index_period:, :] - hp_centre_y - grid_length)**2
        elif hp_centre_y > grid_centre:
            index_period = index(hp_centre_y - grid_length/2, min_grid[1], grid_space[1])
            P_harmonic_y[:index_period, :] = hp_kappa_y/2 * (Y[:index_period, :] - hp_centre_y + grid_length)**2

    return [F_harmonic_x, F_harmonic_y, P_harmonic_x, P_harmonic_y]

Ftot_master = []; Ftot_patch = [] ; sd_patch = []; error_patch = [];
count = 0; simulation_count = 1;

'''
### import HILLS and position file ####################################################################################################################################################################
for file in glob.glob("HILLS40"):
    hills = np.loadtxt(file)
    hills = np.concatenate(([hills[0]], hills[:-1]))
    hills[0][5] = 0
    HILLS = hills

for file1 in glob.glob("position40"):
    colvar = np.loadtxt(file1)
    position_x = colvar[:-1, 1]
    position_y = colvar[:-1, 2]

stride = int(len(position_x)/len(HILLS))     # stride:   number of points in the COLVAR file per point in the HILLS file
const = (1 / (bw*np.sqrt(2*np.pi)*stride))
#Shorten HILLS
HILLS = HILLS[:10000]

#initialize terms
Fbias_x = np.zeros((nbins, nbins)); Fbias_y = np.zeros((nbins, nbins)); Ftot_num_x = np.zeros((nbins, nbins)); Ftot_num_y = np.zeros((nbins, nbins)); Ftot_den = np.zeros((nbins, nbins));
ofe_x = np.zeros((nbins,nbins)); ofe_y = np.zeros((nbins,nbins)); ofe = np.zeros((nbins,nbins));
total_number_of_hills=len(HILLS[:,1]); print("Total no. of HILLS for following simulation: " + str(total_number_of_hills))

Fbias_x_test = np.zeros((nbins,nbins))
Fpbt_x_test = np.zeros((nbins,nbins))

#Cycle over the updates of the bias potential V_t(s) ####################################################################################################################################################################
for i in range(total_number_of_hills):
    count += 1
    # Build metadynamics potential
    s_x = HILLS[i, 1]  # center x-position of gausian
    s_y = HILLS[i, 2]  # center y-position of gausian
    sigma_meta2_x = HILLS[i, 3] ** 2  # width of gausian
    sigma_meta2_y = HILLS[i, 4] ** 2  # width of gausian
    gamma = HILLS[i, 6]
    height_meta = HILLS[i, 5] * ((gamma - 1) / (gamma))  # Height of Gausian
    periodic_points = find_periodic_point(s_x,s_y)
    for j in range(len(periodic_points)):
        kernelmeta = np.exp(-0.5 * (((X - periodic_points[j][0]) ** 2) / sigma_meta2_x + ((Y - periodic_points[j][1]) ** 2) / sigma_meta2_y))  # potential erorr in calc. of s-s_t
        # kernelmeta = np.where(((X-periodic_points[j][0]) ** 2 + (Y - periodic_points[j][1]) ** 2) ** (1 / 2) >= grid_ext, 0, kernelmeta)
        Fbias_x = Fbias_x + height_meta * kernelmeta * ((X - periodic_points[j][0]) / sigma_meta2_x);  ##potential erorr in calc. of s-s_t
        Fbias_y = Fbias_y + height_meta * kernelmeta * ((Y - periodic_points[j][1]) / sigma_meta2_y);  ##potential erorr in calc. of s-s_t


    # Initialise probability density
    pb_t = np.zeros((nbins, nbins))
    Fpbt_x = np.zeros((nbins, nbins))
    Fpbt_y = np.zeros((nbins, nbins))
    # Estimate the biased proabability density p_t ^ b(s)
    data_x = position_x[i * stride: (i + 1) * stride]
    data_y = position_y[i * stride: (i + 1) * stride]


    # Biased probability density component of the force
    for j in range(stride):
        periodic_points = find_periodic_point(data_x[j], data_y[j])
        for k in range(len(periodic_points)):
            kernel = const * np.exp(- (1 / (2 * bw2)) * ((X - periodic_points[k][0]) ** 2 + (Y - periodic_points[k][1]) ** 2));  # check index of j
            # kernel = np.where(((X - periodic_points[k][0]) ** 2 + (Y - periodic_points[k][1]) ** 2) ** (1 / 2) >= grid_ext, 0,kernel)
            pb_t = pb_t + kernel;
            Fpbt_x = Fpbt_x + kernel * kT * (X - periodic_points[k][0]) / bw2
            Fpbt_y = Fpbt_y + kernel * kT * (Y - periodic_points[k][1]) / bw2

    Fpbt_x_test += Fpbt_x

    # Calculate Mean Force
    Ftot_den = Ftot_den + pb_t;
    # Calculate x-component of Force
    Fpbt_x = np.divide((Fpbt_x), pb_t, out=np.zeros_like(Fpbt_x), where=pb_t != 0)
    dfds_x = Fpbt_x + Fbias_x
    Ftot_num_x = Ftot_num_x + pb_t * dfds_x
    Ftot_x = np.divide(Ftot_num_x, Ftot_den, out=np.zeros_like(Fpbt_x), where=Ftot_den != 0)
    # Calculate y-component of Force
    Fpbt_y = np.divide((Fpbt_y), pb_t, out=np.zeros_like(Fpbt_y), where=pb_t != 0)
    dfds_y = Fpbt_y + Fbias_y
    Ftot_num_y = Ftot_num_y + pb_t * dfds_y
    Ftot_y = np.divide(Ftot_num_y, Ftot_den, out=np.zeros_like(Fpbt_y), where=Ftot_den != 0)
    #modulus of force
    Ftot_m = np.sqrt(Ftot_x ** 2 + Ftot_y ** 2)
    #calculate on the fly error (ofe)
    ofe_x += pb_t * (dfds_x - Ftot_x)**2
    ofe_y += pb_t * (dfds_y - Ftot_y)**2
    ofe = np.sqrt(np.sqrt(ofe_x**2 + ofe_y**2))
    ofe = np.divide(ofe, Ftot_den, out=np.zeros_like(ofe), where=Ftot_den > 0.001)

    Fbias_x_test += Fbias_x * pb_t

    if count % 500 == 0 or count == total_number_of_hills:
        # print(str(count) + " / " + str(total_number_of_hills))

        # Ftot_num_x_test = Fbias_x_test + Fpbt_x_test
        # print("\nTest:\nDifference: ", sum(sum(abs(Ftot_num_x - Ftot_num_x_test))) / (nbins ** 2), "\n")
        #
        #
        # Fbias_x_test2 = np.divide(Fbias_x_test, Ftot_den, out=np.zeros_like(Fbias_x_test), where=Ftot_den != 0)
        # Fpbt_x_test2 = np.divide(Fpbt_x_test, Ftot_den, out=np.zeros_like(Fpbt_x_test), where=Ftot_den != 0)
        #
        # Ftot_x_test = Fbias_x_test2 + Fpbt_x_test2
        # print("\nTest:\nDifference: " , sum(sum( abs(Ftot_x  -  Ftot_x_test) ))/(nbins**2), "\n")

        # fig_fes = plt.figure(99)
        #
        # ax1 = fig_fes.add_subplot(141)
        # limit = np.max([abs(np.min(Fbias_x_test2)), np.max(Fbias_x_test2)])
        # im1 = ax1.contourf(X, Y, (Fbias_x_test2), levels=np.linspace(-limit, limit, 50), cmap='coolwarm')
        # ax1.set_title("F_bias X")
        # ax1.set_xlabel("phi")
        # ax1.set_ylabel("psi")
        # divider = make_axes_locatable(ax1)
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        # fig_fes.colorbar(im1, cax=cax, orientation='vertical');
        #
        # ax2 = fig_fes.add_subplot(142)
        # limit = np.max([abs(np.min(Fpbt_x_test2)), np.max(Fpbt_x_test2)])
        # im2 = ax2.contourf(X, Y, (Fpbt_x_test2), levels=np.linspace(-limit, limit, 50), cmap='coolwarm')
        # ax2.set_title("Fpbt")
        # ax2.set_xlabel("phi")
        # ax2.set_ylabel("psi")
        # divider = make_axes_locatable(ax2)
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        # fig_fes.colorbar(im2, cax=cax, orientation='vertical')
        #
        # ax3 = fig_fes.add_subplot(143)
        # limit = np.max([abs(np.min(Ftot_x)), np.max(Ftot_x)])
        # im3 = ax3.contourf(X, Y, (Ftot_x), levels=np.linspace(-limit, limit, 50), cmap='coolwarm')
        # ax3.set_title("Ftot X")
        # ax3.set_xlabel("phi")
        # ax3.set_ylabel("psi")
        # divider = make_axes_locatable(ax3)
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        # fig_fes.colorbar(im3, cax=cax, orientation='vertical')
        #
        # ax4 = fig_fes.add_subplot(144)
        # limit = np.max([abs(np.min(abs(Ftot_x - Ftot_x_test))), np.max(abs(Ftot_x - Ftot_x_test))])
        # im4 = ax4.contourf(X, Y, abs(Ftot_x - Ftot_x_test), levels=np.linspace(-limit, limit, 50), cmap='coolwarm')
        # ax4.set_title("Diff")
        # ax4.set_xlabel("phi")
        # ax4.set_ylabel("psi")
        # divider = make_axes_locatable(ax4)
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        # fig_fes.colorbar(im4, cax=cax, orientation='vertical')
        #
        # plt.show()



        #Integration M.S.
        SdZx = np.cumsum(Ftot_x, axis=1) * grid_space  # cumulative sum along x-axis
        SdZy = np.cumsum(Ftot_y, axis=0) * grid_space  # cumulative sum along y-axis
        FES = np.zeros(SdZx.shape)
        for i in range(FES.shape[0]):
            for j in range(FES.shape[1]):
                FES[i, j] += np.sum([SdZy[i, 0], -SdZy[0, 0], SdZx[i, j], -SdZx[i, 0]])
        FES = FES - np.min(FES)

        #Find adjusted FES (60x60) to compare to reference FES (60x60)
        FES_adj = find_FES_adj(X, Y, FES)
        FES_adj = FES_adj - np.min(FES_adj)

        #Find cutoff matrix for FES_adj > Flim
        cutoff = find_cutoff_matrix(FES_adj)

        # Calculate FES, only take every 5th (factor) value to comopare to fes[60x60]
        FES_error = abs(FES_adj - FREF) * cutoff
        error_patch.append(sum(sum(FES_error)) / (np.count_nonzero(FES_error)))

        # Find cutoff matrix for FES > Flim
        cutoff = find_cutoff_matrix(FES)
        ofe = np.array(ofe*cutoff)

        print(str(count) + " / " + str(total_number_of_hills) + "\nofe = "  + str(sum(sum(ofe))/(np.count_nonzero(ofe))) )#+ "\n")
        # print("ofe2222 = "  + str(sum(sum(ofe*cutoff2))/(np.count_nonzero(ofe*cutoff2))))
        print("AAD (to FREF) is: " + str(error_patch[-1]))
        print("With non-zero ratio: " , np.count_nonzero(cutoff)/(np.shape(FES)[0]*np.shape(FES)[1]))
        print("\n")

        ###########plot fes
        # plt.figure(22)
        # plt.contourf(XREF, YREF, FES_adj, levels=np.linspace(0,50,11), cmap='coolwarm')
        # plt.colorbar();
        # plt.contour(XREF, YREF, FREF, levels=np.linspace(0,50,11), alpha=0.2, colors="grey")
        # plt.show()



# Activate this to save Force terms
file_name = "Ftot_master_10M005.npy"
Ftot_master.append([Ftot_den, Ftot_x,Ftot_y, Ftot_m])
os.system("rm " + file_name)
with open(file_name,"wb") as fw:
    np.save(fw, Ftot_master)
# sys.exit()
'''

# Activete this to load and overwirte force terms
with open("Ftot_master_2.npy","rb") as fr:
    Ftot_master = np.load(fr)
# [Ftot_den, Ftot_x, Ftot_y, Ftot_m, ofe] = Ftot_master[0]
[Ftot_den, Ftot_x, Ftot_y, Ftot_m] = Ftot_master[0]
Ftot_master = []
Ftot_master.append([Ftot_den, Ftot_x, Ftot_y, Ftot_m])

try:
    sum(ofe)
except:
    ofe = np.zeros((nbins,nbins))

#'''

#Save Force terms to Patched Force Terms
Ftot_patch.append([Ftot_den, Ftot_x,Ftot_y, Ftot_m])





# FES, X_new, Y_new = MSintegral(Ftot_x,Ftot_y,i_bins=(800,800))

# FES = MSintegral(Ftot_x,Ftot_y)
FES = FWIntegration(Ftot_x,Ftot_y)


if len(FES) == len(X): FES_adj = find_FES_adj(X, Y, FES)
else: FES_adj = find_FES_adj(X_new, Y_new, FES)
FES_adj = FES_adj - np.min(FES_adj)
cutoff_adj = find_cutoff_matrix(FES_adj)
FES_error = abs(FES_adj - FREF) * cutoff_adj
FES_error_uncut = abs(FES_adj - FREF)
FES_error_first = np.array(FES_error)
cutoff = find_cutoff_matrix(FES)
error_patch.append(sum(sum(FES_error)) / (np.count_nonzero(FES_error)))

print("AAD (to FREF) is: " + str(error_patch[-1]))
print("With non-zero ratio: " , np.count_nonzero(cutoff)/(np.shape(FES)[0]*np.shape(FES)[1]), "\n")






sd_patch.append(0)

#'''

# #Plot probability density
# plt.figure(199)
# plt.contourf(X, Y, (Ftot_den), levels=np.logspace(-13,3,17),cmap='coolwarm', norm = LogNorm())#,cmap='coolwarm', locator=ticker.LogLocator()
# plt.colorbar()
# plt.title("Probability density (after 1st sim)")
# plt.xlabel("phi")
# plt.ylabel("psi")



fig_fes = plt.figure(3, figsize=plt.figaspect(0.4))
plt.suptitle("FES 0th simulation")
#########################
ax1 = fig_fes.add_subplot(121)
if len(FES) == len(X):im1 = ax1.contourf(X, Y, (FES), levels=np.linspace(0, 50, 21), cmap='coolwarm')
else:im1 = ax1.contourf(X_new, Y_new, (FES), levels=np.linspace(0, 50, 21), cmap='coolwarm')
ax1.set_title("Simulated FES")
ax1.set_xlabel("phi")
ax1.set_ylabel("psi")
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig_fes.colorbar(im1, cax=cax, orientation='vertical');
im1 = ax1.contour(XREF, YREF, FREF, levels=np.linspace(0, 50, 21), alpha=0.4, colors="grey")
#########################
ax2 = fig_fes.add_subplot(122)
if len(FES) == len(X):im2 = ax2.contourf(X, Y, (FES), levels=np.linspace(0, 150, 21), cmap='coolwarm')
else:im2 = ax2.contourf(X_new, Y_new, (FES), levels=np.linspace(0, 150, 21), cmap='coolwarm')
ax2.set_title("Simulated FES (0 - 150)")
ax2.set_xlabel("phi")
ax2.set_ylabel("psi")
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig_fes.colorbar(im2, cax=cax, orientation='vertical')
im2 = ax2.contour(XREF, YREF, FREF, levels=np.linspace(0, 150, 21), alpha=0.4, colors="grey")
#########################
# ax3 = fig_fes.add_subplot(223)
# im3 = ax3.contourf(XREF, YREF, FREF, levels=np.linspace(0, 50, 21), cmap='coolwarm')
# ax3.set_title("Reference FES")
# ax3.set_xlabel("phi")
# ax3.set_ylabel("psi")
# divider = make_axes_locatable(ax3)
# cax = divider.append_axes('right', size='5%', pad=0.05)
# fig_fes.colorbar(im3, cax=cax, orientation='vertical')
# #########################
# ax4 = fig_fes.add_subplot(224)
# im4 = ax4.contourf(XREF, YREF, FREF, levels=np.linspace(0, 150, 21), cmap='coolwarm')
# ax4.set_title("Reference FES (0 - 150)")
# ax4.set_xlabel("phi")
# ax4.set_ylabel("psi")
# divider = make_axes_locatable(ax4)
# cax = divider.append_axes('right', size='5%', pad=0.05)
# fig_fes.colorbar(im4, cax=cax, orientation='vertical')


fig_fes = plt.figure(5, figsize=plt.figaspect(0.4))
plt.suptitle("FES error 0th simulation")

#########################
ax1 = fig_fes.add_subplot(121)
im1 = ax1.contourf(XREF, YREF, zero_to_nan(FES_error), levels=np.linspace(0, 10, 21), cmap='coolwarm')
ax1.set_title("AD to FES10E9")
ax1.set_xlabel("phi")
ax1.set_ylabel("psi")
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig_fes.colorbar(im1, cax=cax, orientation='vertical');
#########################
ax2 = fig_fes.add_subplot(122)
im2 = ax2.contourf(XREF, YREF, zero_to_nan(FES_error_uncut), levels=np.linspace(0, 80, 21), cmap='coolwarm')
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig_fes.colorbar(im2, cax=cax, orientation='vertical')
im2 = ax2.contour(XREF, YREF, zero_to_nan(FES_error_uncut), levels=np.linspace(50, 150, 5), alpha=0.5, colors="grey")
ax2.set_title("AD to FES10E9 UNCUT")
ax2.set_xlabel("phi")
ax2.set_ylabel("psi")



# fig_fes = plt.figure(6)#, figsize=plt.figaspect(0.4))
#
# ax1 = fig_fes.add_subplot(221)
# limit = np.max([abs(np.min(Ftot_x)), np.max(Ftot_x)])
# im1 = ax1.contourf(X, Y, (Ftot_x), levels=np.linspace(-300, 300, 50), cmap='coolwarm')
# ax1.set_title("Ftot X (Simulation 1)")
# ax1.set_xlabel("phi")
# ax1.set_ylabel("psi")
# divider = make_axes_locatable(ax1)
# cax = divider.append_axes('right', size='5%', pad=0.05)
# fig_fes.colorbar(im1, cax=cax, orientation='vertical');
#
# ax2 = fig_fes.add_subplot(222)
# limit = np.max([abs(np.min(Ftot_y)), np.max(Ftot_y)])
# im2 = ax2.contourf(X, Y, (Ftot_y), levels=np.linspace(-300, 300, 50), cmap='coolwarm')
# ax2.set_title("Ftot Y (Simulation 1)")
# ax2.set_xlabel("phi")
# ax2.set_ylabel("psi")
# divider = make_axes_locatable(ax2)
# cax = divider.append_axes('right', size='5%', pad=0.05)
# fig_fes.colorbar(im2, cax=cax, orientation='vertical')


# plt.show()
# sys.exit()
#'''



####REINITIALISE WITH WALL POTWNTIAL
ipos_x = ["0.0"]
ipos_y = ["0.0"]


# ipos_x = ["-2.0"]
# ipos_y = ["-2.0"]

# ipos_x = ["-1.0","-0.5","0.0","0.0","0.0","1.5","2.0"]#,"0.5","1.0","1.0","1.0"]#,"-2.5"]
# ipos_y = ["1.0","0.75","0.5","1.0","0.0","0.0","-0.5"]#,"0.25","0.0","-1.0","-2.0"]
#
#
# ipos_x = ["-1.0","-2.0","-2.0","-2.0","-1.5","-1.5"]#,"0.0","1.5","2.0"]#,"0.5","1.0","1.0","1.0"]#,"-2.5"]
# ipos_y = ["1.0","0.0","-1.0","-1.5","-1.5","-1.0","-0.5"]#,"0.25","0.0","-1.0","-2.0"]
#
# # kappa = [50,50,50,50,50,50,50,50,50]
# kappa = [150,150,150,150,150,150,150,150,150]
kappa = [100,100,100,100,100,100,100,100,100]

# kappa = [0,0]#[150,150]





nsteps = 100000
os.system("rm bck.*")
os.system("rm structure*")
os.system("rm input*")
os.system("rm *.*#")


for simulation in range(1):
    simulation_count += 1

    #Find index of x restraint
    IPOSX = int((float(ipos_x[simulation]) - np.min(grid)) // grid_space)
    ipos_x[simulation] = str(grid[IPOSX])
    # Find index of y restraint
    IPOSY = int((float(ipos_y[simulation]) - np.min(grid)) // grid_space)
    ipos_y[simulation] = str(grid[IPOSY])
    # print("New x-wall (lower/upper) adjusted to grid: ", (grid[LX_wall], grid[UX_wall]), (diff1x, diff2x))


    #'''
    #####>>>Analyse trajectorry <- input traj_comp2.xtc>>>##############################
    #PBC havent been used for start_region (below)
    start_region = [str( float(ipos_x[simulation]) - 0.5), str( float(ipos_x[simulation]) + 0.5), str( float(ipos_y[simulation]) - 0.5), str( float(ipos_y[simulation]) + 0.5)]
    print("Preparing new inpit files ...")
    with open("plumed_traj.dat") as f:
        lines = f.readlines()
    lines[3] = "UPDATE_IF ARG=phi,psi MORE_THAN=" + start_region[0] + "," + start_region[2] + " LESS_THAN=" + start_region[1] + "," + start_region[3] + "\n"
    lines[4] = "DUMPATOMS FILE=structure" + str(simulation_count) + ".gro ATOMS=1-22\n"
    with open("plumed_traj.dat", "w") as f:
        f.writelines(lines)

    os.system("plumed driver --plumed plumed_traj.dat --mf_xtc 0traj_comp.xtc > /dev/null") #
    #####<<<Analyse trajectorry -> output new structure(n).gro<<<##############################


    #####>>>Prepare new input file<- input structure(n).gro, topolvac.top, gromppvac.mdp >>>##############################
    # print("Writing new .trp file ...")
    # os.system("gmx grompp -f gromppvac.mdp -c structure" + str(simulation_count) + ".gro -p topology.top -o input" + str(simulation_count) + ".tpr > /dev/null") #
    find_input_structure = subprocess.Popen(["gmx", "grompp", "-f", "gromppvac.mdp", "-c", "structure" + str(simulation_count)+".gro", "-p", "topology.top", "-o", "input" + str(simulation_count)+ ".tpr"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    find_input_structure.wait()
    output_find_input_structure, errors_find_input_structure = find_input_structure.communicate()
    if "Error" in errors_find_input_structure:
        print("*****There is an error message:*****\n\n")
        print(errors_find_input_structure)
    #####<<<Prepare new input file input(n).tpr<<<##############################


    #####>>>Change simulation parameters in plumed.dat>>>##############################
    with open("plumed_restr.dat", "w") as f:
        print("""MOLINFO STRUCTURE=reference.pdb
phi: TORSION ATOMS=@phi-2
psi: TORSION ATOMS=@psi-2
RESTRAINT ARG=phi,psi AT=0.0,0.0 KAPPA=125.0,125.0 LABEL=restraint
METAD ARG=phi,psi SIGMA=0.2,0.2 HEIGHT=1.2 PACE=500 TEMP=300 BIASFACTOR=8 GRID_MIN=-pi,-pi GRID_MAX=pi,pi GRID_BIN=500,500
PRINT FILE=position ARG=phi,psi STRIDE=10
PRINT FILE=restraint ARG=phi,psi,restraint.bias,restraint.force2 STRIDE=100

#gmx mdrun -s topolA.tpr -nsteps 1000000 -plumed plumed_first.dat -v""", file=f)

    with open("plumed_restr.dat") as f:
        lines = f.readlines()

    lines[3] = "RESTRAINT ARG=phi,psi AT=" + ipos_x[simulation] + "," + ipos_y[simulation] + " KAPPA=" + str(kappa[simulation]) + "," + str(kappa[simulation]) + " LABEL=restraint\n"
    lines[4] = "METAD ARG=phi,psi PACE=100 SIGMA=0.1,0.1 HEIGHT=0.5 TEMP=300 BIASFACTOR=3 GRID_MIN=-pi,-pi GRID_MAX=pi,pi GRID_BIN=1000,1000\n"

    with open("plumed_restr.dat", "w") as f:
        f.writelines(lines)
    #####<<<Change simulation parameters in plumed.dat<<<##############################


    #####>>>run new simulations>>>##############################
    print("Running simulation with harmonic potential at: (" , round(float(ipos_x[simulation]),2) , "," , round(float(ipos_y[simulation]),2) , ")  with kappa =" , str(kappa[simulation]))
    # os.system("gmx mdrun -s input" + str(simulation_count) + ".tpr -nsteps " + str(int(nsteps)) + " -plumed plumed_restr.dat -v") # > /dev/null
    find_input_file = subprocess.Popen(["gmx", "mdrun", "-s", "input" + str(simulation_count) + ".tpr", "-nsteps" , str(int(nsteps)), "-plumed", "plumed_restr.dat"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    find_input_file.wait()
    output_find_input_file, errors_find_input_file = find_input_file.communicate()

    if "Error" in errors_find_input_file:
        print("There is an error message")
        print(errors_find_input_file)

    print("... Simulation finished. Analysing data ...\n")

    #####<<<run new simulations<<<##############################
    #'''

    ### import HILLS and position file ####################################################################################################################################################################
    for file in glob.glob("HILLS"):
        hills = np.loadtxt(file)
        hills = np.concatenate(([hills[0]], hills[:-1]))
        hills[0][5] = 0
        HILLS = hills

    for file1 in glob.glob("position"):
        colvar = np.loadtxt(file1)
        position_x = colvar[:-1, 1]
        position_y = colvar[:-1, 2]


    stride = int(len(position_x)/len(HILLS))
    const = (1 / (bw * np.sqrt(2 * np.pi) * stride))

    plt.figure(0)
    plt.scatter(position_x, position_y,s=1, label="Sim. " + str(simulation))
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi, np.pi)
    plt.xlabel("phi")
    plt.ylabel("psi")
    plt.title("COLVAR coordinates")
    plt.legend()
    # plt.show()


    ######### FIND HARMONIC FORCE (taking into account PBC)##################
    F_harmonic_x = np.zeros((len(grid), len(grid)))
    F_harmonic_y = np.zeros((len(grid), len(grid)))
    #Calculate x-force
    if abs(grid[IPOSX]) < grid_space: F_harmonic_x = kappa[simulation] * (X - grid[IPOSX])
    elif grid[IPOSX] < 0:
        SEPR = round((grid[IPOSX] + 2*np.pi) / grid_space)
        F_harmonic_x[:, :SEPR] += kappa[simulation] * (X[:, :SEPR] - grid[IPOSX])
        F_harmonic_x[:, SEPR:] += kappa[simulation] * (X[:, SEPR:] - grid[IPOSX] - (np.max(grid) - np.min(grid)))
    elif grid[IPOSX] > 0:
        SEPR = round((grid[IPOSX]) / grid_space)
        F_harmonic_x[:, :SEPR] += kappa[simulation] * (X[:, :SEPR] - grid[IPOSX] + (np.max(grid) - np.min(grid)))
        F_harmonic_x[:, SEPR:] += kappa[simulation] * (X[:, SEPR:] - grid[IPOSX])
    #Calculate y-force
    if abs(grid[IPOSY]) < grid_space: F_harmonic_y = kappa[simulation] * (Y - grid[IPOSY])
    elif grid[IPOSY] < 0:
        SEPR = round((grid[IPOSY] + 2*np.pi) / grid_space)
        F_harmonic_y[:SEPR, :] += kappa[simulation] * (Y[:SEPR, :] - grid[IPOSY])
        F_harmonic_y[SEPR:, :] += kappa[simulation] * (Y[SEPR:, :] - grid[IPOSY] - (np.max(grid) - np.min(grid)))
    elif grid[IPOSY] > 0:
        SEPR = round((grid[IPOSY]) / grid_space)
        F_harmonic_y[:SEPR, :] += kappa[simulation] * (Y[:SEPR, :] - grid[IPOSY] + (np.max(grid) - np.min(grid)))
        F_harmonic_y[SEPR:, :] += kappa[simulation] * (Y[SEPR:, :] - grid[IPOSY])

    F_harmonic = F_harmonic_x + F_harmonic_y




    P_harmonic_x = np.zeros((len(grid), len(grid)))
    P_harmonic_y = np.zeros((len(grid), len(grid)))
    #Calculate x-force
    if abs(grid[IPOSX]) < grid_space: P_harmonic_x = kappa[simulation]/2 * (X - grid[IPOSX])**2
    elif grid[IPOSX] < 0:
        SEPR = round((grid[IPOSX] + 2*np.pi) / grid_space)
        P_harmonic_x[:, :SEPR] += kappa[simulation]/2 * (X[:, :SEPR] - grid[IPOSX])**2
        P_harmonic_x[:, SEPR:] += kappa[simulation]/2 * (X[:, SEPR:] - grid[IPOSX] - (np.max(grid) - np.min(grid)))**2
    elif grid[IPOSX] > 0:
        SEPR = round((grid[IPOSX]) / grid_space)
        P_harmonic_x[:, :SEPR] += kappa[simulation]/2 * (X[:, :SEPR] - grid[IPOSX] + (np.max(grid) - np.min(grid)))**2
        P_harmonic_x[:, SEPR:] += kappa[simulation]/2 * (X[:, SEPR:] - grid[IPOSX])**2
    #Calculate y-force
    if abs(grid[IPOSY]) < grid_space: P_harmonic_y = kappa[simulation]/2 * (Y - grid[IPOSY])**2
    elif grid[IPOSY] < 0:
        SEPR = round((grid[IPOSY] + 2*np.pi) / grid_space)
        P_harmonic_y[:SEPR, :] += kappa[simulation]/2 * (Y[:SEPR, :] - grid[IPOSY])**2
        P_harmonic_y[SEPR:, :] += kappa[simulation]/2 * (Y[SEPR:, :] - grid[IPOSY] - (np.max(grid) - np.min(grid)))**2
    elif grid[IPOSY] > 0:
        SEPR = round((grid[IPOSY]) / grid_space)
        P_harmonic_y[:SEPR, :] += kappa[simulation]/2 * (Y[:SEPR, :] - grid[IPOSY] + (np.max(grid) - np.min(grid)))**2
        P_harmonic_y[SEPR:, :] += kappa[simulation]/2 * (Y[SEPR:, :] - grid[IPOSY])**2

    P_harmonic = P_harmonic_x + P_harmonic_y



    for file1 in glob.glob("restraint"):
        data = np.loadtxt(file1)
        restr_pos = data[:, 1:3]
        # restr_psi = data[:, 2]
        plumed_bias = data[:, 3]
        plumed_force2 = data[:, 4]


    bias_error = []
    bias_interpolate = interpolate.interp2d(grid, grid, P_harmonic, kind="quintic")
    # force_interpolate = interpolate.interp2d(grid, grid, F_harmonic, kind="quintic")

    for ii in range(len(plumed_bias)):
        bias_error.append(bias_interpolate(restr_pos[ii][0], restr_pos[ii][1]) - plumed_bias[ii])
    bias_error = np.array(bias_error).reshape((np.shape(bias_error)[0],))

    print("\n\n******** ATTENTION, BIAS DIFFERENCE IS HIGH **********")
    print("Plumed Bias vs theoretical Bias difference: " , sum(bias_error)/(len(bias_error)))

    bias_error_interploation = griddata(restr_pos, np.array(bias_error), (X, Y), method="linear")

    plumed_bias_interpolate = griddata(restr_pos, np.array(plumed_bias), (X, Y), method="linear")

    fig_restr = plt.figure(2)

    ax1 = fig_restr.add_subplot(131)
    im1 = ax1.contourf(X, Y, bias_error_interploation, cmap='coolwarm')
    ax1.set_title("bias_error_interploation")
    ax1.set_xlabel("phi")
    ax1.set_ylabel("psi")
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig_restr.colorbar(im1, cax=cax, orientation='vertical');

    ax3 = fig_restr.add_subplot(132)
    im3 = ax3.contourf(X, Y, plumed_bias_interpolate, cmap='coolwarm')
    ax3.set_xlabel("phi")
    ax3.set_ylabel("psi")
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig_restr.colorbar(im3, cax=cax, orientation='vertical');

    ax4 = fig_restr.add_subplot(133)
    im4 = ax4.contourf(X, Y, P_harmonic, cmap='coolwarm')
    ax4.set_title("P_harmonic")
    ax4.set_xlabel("phi")
    ax4.set_ylabel("psi")
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig_restr.colorbar(im4, cax=cax, orientation='vertical');

    plt.show()


    #"""

    # initialize Force terms
    Fbias_x = np.zeros((nbins, nbins)) ; Fbias_y = np.zeros((nbins, nbins)) ; Ftot_num_x = np.zeros((nbins, nbins)) ; Ftot_num_y = np.zeros((nbins, nbins))
    Ftot_den = np.zeros((nbins, nbins)) ; Ftot_x = np.zeros((nbins, nbins)) ; Ftot_y = np.zeros((nbins, nbins))
    count = 0 ; total_number_of_hills = len(HILLS)
    print("Calculate Force Terms (" + str(total_number_of_hills) + " HILLS) ...")

    Fbias_x_test = np.zeros((nbins, nbins))
    Fpbt_x_test = np.zeros((nbins, nbins))
    F_harmonic_x_test = np.zeros((nbins, nbins))

    # Cycle over the updates of the bias potential V_t(s) ####################################################################################################################################################################
    for i in range(total_number_of_hills):
        count += 1

        # Build metadynamics potential
        s_x = HILLS[i, 1] ;s_y = HILLS[i, 2]
        sigma_meta2_x = HILLS[i, 3] ** 2 ; sigma_meta2_y = HILLS[i, 4] ** 2  # width of gausian
        gamma = HILLS[i, 6]; height_meta = HILLS[i, 5] * ((gamma - 1) / (gamma))  # Height of Gausian

        periodic_points = find_periodic_point(s_x, s_y)
        for j in range(len(periodic_points)):
            kernelmeta = np.exp(-0.5 * (((X - periodic_points[j][0]) ** 2) / sigma_meta2_x + ((Y - periodic_points[j][1]) ** 2) / sigma_meta2_y))  # potential erorr in calc. of s-s_t
            # kernelmeta = np.where(((X-periodic_points[j][0]) ** 2 + (Y - periodic_points[j][1]) ** 2) ** (1 / 2) >= grid_ext, 0, kernelmeta)
            Fbias_x = Fbias_x + height_meta * kernelmeta * ((X - periodic_points[j][0]) / sigma_meta2_x);  ##potential erorr in calc. of s-s_t
            Fbias_y = Fbias_y + height_meta * kernelmeta * ((Y - periodic_points[j][1]) / sigma_meta2_y);  ##potential erorr in calc. of s-s_t


        # Initialise probability density
        pb_t = np.zeros((nbins, nbins))
        Fpbt_x = np.zeros((nbins, nbins))
        Fpbt_y = np.zeros((nbins, nbins))
        # Estimate the biased proabability density p_t ^ b(s)
        data_x = position_x[i * stride: (i + 1) * stride]
        data_y = position_y[i * stride: (i + 1) * stride]

        # Biased probability density component of the force
        for j in range(stride):
            periodic_points = find_periodic_point(data_x[j], data_y[j])
            for k in range(len(periodic_points)):
                # kernel = const * np.exp( - (1 / (2 * bw2)) * ((X - data_x[j]) ** 2 + (Y - data_y[j]) ** 2));  # check index of j
                kernel = const * np.exp(- (1 / (2 * bw2)) * ((X - periodic_points[k][0]) ** 2 + (Y - periodic_points[k][1]) ** 2));  # check index of j
                # kernel = np.where(((X - periodic_points[k][0]) ** 2 + (Y - periodic_points[k][1]) ** 2) ** (1 / 2) >= grid_ext, 0,kernel)
                pb_t = pb_t + kernel;
                Fpbt_x = Fpbt_x + kernel * kT * (X - periodic_points[k][0]) / bw2
                Fpbt_y = Fpbt_y + kernel * kT * (Y - periodic_points[k][1]) / bw2

        Fpbt_x_test += Fpbt_x

        # Calculate Mean Force
        Ftot_den = Ftot_den + pb_t;
        # Calculate x-component of Force
        Fpbt_x = np.divide((Fpbt_x), pb_t, where=pb_t != 0)
        dfds_x = Fpbt_x + Fbias_x - F_harmonic_x
        Ftot_num_x = Ftot_num_x + pb_t * dfds_x
        Ftot_x = np.divide(Ftot_num_x, Ftot_den, where=Ftot_den != 0)
        # Calculate y-component of Force
        Fpbt_y = np.divide((Fpbt_y), pb_t, where=pb_t != 0)
        dfds_y = Fpbt_y + Fbias_y - F_harmonic_y
        Ftot_num_y = Ftot_num_y + pb_t * dfds_y
        Ftot_y = np.divide(Ftot_num_y, Ftot_den, where=Ftot_den != 0)
        # modulus
        Ftot_m = np.sqrt(Ftot_x ** 2 + Ftot_y ** 2)

        Fbias_x_test += Fbias_x * pb_t

        F_harmonic_x_test += - F_harmonic_x * pb_t


    Ftot_master.append([Ftot_den,Ftot_x, Ftot_y,Ftot_m])

    # Ftot_num_x_test = Fbias_x_test + Fpbt_x_test + F_harmonic_x_test
    # print("\nTest Ftot_num_x difference: ", sum(sum(abs(Ftot_num_x - Ftot_num_x_test))) / (nbins ** 2))
    #
    # Fbias_x_test2 = np.divide(Fbias_x_test, Ftot_den, out=np.zeros_like(Fbias_x_test), where=Ftot_den != 0)
    # Fpbt_x_test2 = np.divide(Fpbt_x_test, Ftot_den, out=np.zeros_like(Fpbt_x_test), where=Ftot_den != 0)
    # F_harmonic_x_test2 = np.divide(F_harmonic_x_test, Ftot_den, out=np.zeros_like(F_harmonic_x_test), where=Ftot_den != 0)
    # Ftot_x_test = Fbias_x_test2 + Fpbt_x_test2 + F_harmonic_x_test2
    # print("Test Ftot_x difference: ", sum(sum(abs(Ftot_x - Ftot_x_test))) / (nbins ** 2), "\n")

    # fig_fes = plt.figure(49, figsize = plt.figaspect(0.22))
    # ax1 = fig_fes.add_subplot(141)
    # limit = np.max([abs(np.min(Fbias_x_test2)), np.max(Fbias_x_test2)])
    # im1 = ax1.contourf(X, Y, (Fbias_x_test2), levels=np.linspace(-limit, limit, 50), cmap='coolwarm')
    # ax1.set_title("F_bias X")
    # ax1.set_xlabel("phi")
    # ax1.set_ylabel("psi")
    # divider = make_axes_locatable(ax1)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig_fes.colorbar(im1, cax=cax, orientation='vertical');
    #
    # ax2 = fig_fes.add_subplot(142)
    # limit = np.max([abs(np.min(Fpbt_x_test2)), np.max(Fpbt_x_test2)])
    # im2 = ax2.contourf(X, Y, (Fpbt_x_test2), levels=np.linspace(-limit, limit, 50), cmap='coolwarm')
    # ax2.set_title("Fpbt")
    # ax2.set_xlabel("phi")
    # ax2.set_ylabel("psi")
    # divider = make_axes_locatable(ax2)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig_fes.colorbar(im2, cax=cax, orientation='vertical')
    #
    # try:
    #     ax4 = fig_fes.add_subplot(143)
    #     limit = np.max([abs(np.min(F_harmonic_x_test2)), np.max(F_harmonic_x_test2)])
    #     im4 = ax4.contourf(X, Y, F_harmonic_x_test2, levels=np.linspace(-limit, limit, 50), cmap='coolwarm')
    #     ax4.set_title("F harmonic")
    #     ax4.set_xlabel("phi")
    #     ax4.set_ylabel("psi")
    #     divider = make_axes_locatable(ax4)
    #     cax = divider.append_axes('right', size='5%', pad=0.05)
    #     fig_fes.colorbar(im4, cax=cax, orientation='vertical')
    # except: pass
    #
    # ax3 = fig_fes.add_subplot(144)
    # limit = np.max([abs(np.min(Ftot_x_test)), np.max(Ftot_x_test)])
    # im3 = ax3.contourf(X, Y, (Ftot_x_test), levels=np.linspace(-limit, limit, 50), cmap='coolwarm')
    # ax3.set_title("Ftot X")
    # ax3.set_xlabel("phi")
    # ax3.set_ylabel("psi")
    # divider = make_axes_locatable(ax3)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig_fes.colorbar(im3, cax=cax, orientation='vertical')



    print("Patching Force Terms ( Shape of master: ",np.shape(Ftot_master))
    Ftot_x = np.zeros((nbins, nbins))
    Ftot_y = np.zeros((nbins, nbins))
    Ftot_den = np.zeros((nbins, nbins))

    for i in range(len(Ftot_master)):
        Ftot_x += Ftot_master[i][0] * Ftot_master[i][1]
        Ftot_y += Ftot_master[i][0] * Ftot_master[i][2]
        Ftot_m += Ftot_master[i][0] * Ftot_master[i][3]
        Ftot_den += Ftot_master[i][0]

    Ftot_x = np.divide(Ftot_x, Ftot_den, out=np.zeros_like(Ftot_x), where=Ftot_den != 0)
    Ftot_y = np.divide(Ftot_y, Ftot_den, out=np.zeros_like(Ftot_y), where=Ftot_den != 0)
    Ftot_m = np.divide(Ftot_m, Ftot_den, out=np.zeros_like(Ftot_m), where=Ftot_den != 0)

    Ftot_patch.append([Ftot_den, Ftot_x, Ftot_y, Ftot_m])



    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FW Integration~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    start = time.time()
    FES = FWIntegration(Ftot_x, Ftot_y)
    time_Four = time.time() - start

    # print("shape FES: " , np.shape(FES_a))
    if len(FES) == len(X):
        FES_adj = find_FES_adj(X, Y, FES)
    else:
        FES_adj = find_FES_adj(X_new, Y_new, FES_a)
    FES_adj = FES_adj - np.min(FES_adj)

    # Find cutoff matrix for FES_adj > Flim
    cutoff_adj = find_cutoff_matrix(FES_adj)

    # Calculate FES error to FREF[60x60]
    FES_error = abs(FES_adj - FREF) * cutoff_adj
    FES_error_uncut = abs(FES_adj - FREF)
    error_patch.append(sum(sum(FES_error)) / (np.count_nonzero(FES_error)))

    # Find cutoff matrix for FES > Flim
    cutoff = find_cutoff_matrix(FES)

    print("AAD (to FREF) is: " + str(error_patch[-1]) + str("     <-------------"))
    print("With non-zero ratio: ", np.count_nonzero(cutoff) / (np.shape(cutoff)[0] * np.shape(cutoff)[1]))
    print("with duration ", round(time_Four, 5), "\n")
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FW Integration~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")





# file_name = "Ftot_master_corner3.npy"
# Ftot_save = [X, Y, Ftot_den, Ftot_x,Ftot_y, FES]
# os.system("rm " + file_name)
# with open(file_name,"wb") as fw:
#     np.save(fw, Ftot_master)
#
# print("file saved\n~~~~~~~~~~~~~~~~~~~")







fig_fes = plt.figure(203, figsize=plt.figaspect(0.4))
plt.suptitle("Fourier FES")
#########################
ax1 = fig_fes.add_subplot(121)
im1 = ax1.contourf(X, Y, (FES), levels=np.linspace(0, 50, 21), cmap='coolwarm')
ax1.set_title("Fourier FES")
ax1.set_xlabel("phi")
ax1.set_ylabel("psi")
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig_fes.colorbar(im1, cax=cax, orientation='vertical');
im1 = ax1.contour(XREF, YREF, FREF, levels=np.linspace(0, 50, 21), alpha=0.4, colors="grey")
#########################
ax2 = fig_fes.add_subplot(122)
im2 = ax2.contourf(X, Y, (FES), levels=np.linspace(0, 150, 21), cmap='coolwarm')
ax2.set_title("Fourier FES (0 - 150)")
ax2.set_xlabel("phi")
ax2.set_ylabel("psi")
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig_fes.colorbar(im2, cax=cax, orientation='vertical')
im2 = ax2.contour(XREF, YREF, FREF, levels=np.linspace(0, 150, 21), alpha=0.4, colors="grey")



fig_fes = plt.figure(205, figsize=plt.figaspect(0.4))
plt.suptitle("Error Fourier")

ax1 = fig_fes.add_subplot(121)
im1 = ax1.contourf(XREF, YREF, zero_to_nan(FES_error), levels=np.linspace(0, 10, 21), cmap='coolwarm')
ax1.set_title("AD to FES10E9")
ax1.set_xlabel("phi")
ax1.set_ylabel("psi")
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig_fes.colorbar(im1, cax=cax, orientation='vertical');

ax2 = fig_fes.add_subplot(122)
im2 = ax2.contourf(XREF, YREF, zero_to_nan(FES_error_uncut), levels=np.linspace(0, 80, 21), cmap='coolwarm')
ax2.set_title("AD to FES10E9 UNCUT")
ax2.set_xlabel("phi")
ax2.set_ylabel("psi")
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig_fes.colorbar(im2, cax=cax, orientation='vertical')
im2 = ax2.contour(XREF, YREF, zero_to_nan(FES_error_uncut), levels=np.linspace(50, 150, 5), alpha=0.5, colors="grey")



plt.figure(23)
plt.plot(range(len(error_patch)),error_patch)
plt.title("AAD progession")
plt.xlabel("Number of simulations")
plt.ylabel("AAD in kJ/mol")


plt.show()


#Reuse old plots
#
# fes3d_woco = plt.figure(81)
# ax = fes3d_woco.gca(projection='3d')
# surf = ax.plot_surface(X, Y, FES, cmap="coolwarm", linewidth=0, antialiased=False)
# fes3d_woco.colorbar(surf, shrink=0.5, aspect=5)
# plt.title("Theoretical FES")
# ax.set_xlabel('phi in rad')
# ax.set_ylabel('psi in rad')
# ax.set_zlabel('Free energy in kJ');

# plt.figure(100)
# plt.contourf(X + 2 * np.pi, Y, (Ftot_x), cmap='coolwarm')  # ;
# # plt.contourf(X, Y + 2 * np.pi, (Ftot_x), cmap='coolwarm')  # ;
# plt.contourf(X - 2 * np.pi, Y, (Ftot_x), cmap='coolwarm')  # ;
# plt.contourf(X, Y - 2 * np.pi, (Ftot_x), cmap='coolwarm')  # ;
# plt.contourf(X + 2 * np.pi, Y + 2 * np.pi, (Ftot_x), cmap='coolwarm')  # ;
# plt.contourf(X + 2 * np.pi, Y - 2 * np.pi, (Ftot_x), cmap='coolwarm')  # ;
# plt.contourf(X - 2 * np.pi, Y + 2 * np.pi, (Ftot_x), cmap='coolwarm')  # ;
# plt.contourf(X - 2 * np.pi, Y - 2 * np.pi, (Ftot_x), cmap='coolwarm')  # ;
# plt.contourf(X, Y, (Ftot_x), cmap='coolwarm')  # ;
# plt.plot([-3 * np.pi, 3 * np.pi], [-np.pi, -np.pi], '--', color="green")
# plt.plot([-3 * np.pi, 3 * np.pi], [np.pi, np.pi], '--', color="green")
# plt.plot([np.pi, np.pi], [-3 * np.pi, 3 * np.pi], '--', color="green")
# plt.plot([-np.pi, -np.pi], [-3 * np.pi, 3 * np.pi], '--', color="green")
# plt.colorbar();
# plt.title("Ftot_x")
# plt.xlabel("phi")
# plt.ylabel("psi")
#
# plt.figure(101)
# plt.contourf(X + 2 * np.pi, Y, (Ftot_y), cmap='coolwarm')  # ;
# plt.contourf(X, Y + 2 * np.pi, (Ftot_y), cmap='coolwarm')  # ;
# plt.contourf(X - 2 * np.pi, Y, (Ftot_y), cmap='coolwarm')  # ;
# # plt.contourf(X, Y - 2 * np.pi, (Ftot_y), cmap='coolwarm')  # ;
# plt.contourf(X + 2 * np.pi, Y + 2 * np.pi, (Ftot_y), cmap='coolwarm')  # ;
# plt.contourf(X + 2 * np.pi, Y - 2 * np.pi, (Ftot_y), cmap='coolwarm')  # ;
# plt.contourf(X - 2 * np.pi, Y + 2 * np.pi, (Ftot_y), cmap='coolwarm')  # ;
# plt.contourf(X - 2 * np.pi, Y - 2 * np.pi, (Ftot_y), cmap='coolwarm')  # ;
# plt.contourf(X, Y, (Ftot_y), cmap='coolwarm')  # ;
# plt.plot([-3 * np.pi, 3 * np.pi], [-np.pi, -np.pi], '--', color="green")
# plt.plot([-3 * np.pi, 3 * np.pi], [np.pi, np.pi], '--', color="green")
# plt.plot([np.pi, np.pi], [-3 * np.pi, 3 * np.pi], '--', color="green")
# plt.plot([-np.pi, -np.pi], [-3 * np.pi, 3 * np.pi], '--', color="green")
# plt.colorbar();
# plt.title("Ftot_y")
# plt.xlabel("phi")
# plt.ylabel("psi")