from __future__ import print_function
import glob
import matplotlib.pyplot as plt
#from numba import jit 
#from numba import njit
import numpy as np
import pickle
import random
from matplotlib import ticker
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import os



### Load files ####
def load_HILLS_2D(hills_name="HILLS"):
	"""Load 2-dimensional hills data (includes time, position_x, position_y, hills_parameters ).
	
	Args:
		hills_name (str, optional): Name of hills file. Defaults to "HILLS".
	Returns:
		np.array: hills data with length equal to the total number of hills. Information: [time [ps], position_x [nm], position_y [nm], MetaD_sigma_x [nm], MetaD_sigma_y [nm], MetaD_height [nm], MetaD_biasfactor]
	"""
	for file in glob.glob(hills_name):
		hills = np.loadtxt(file)
		hills = np.concatenate(([hills[0]], hills[:-1]))
		hills[0][5] = 0
	return hills

def load_position_2D(position_name="position"):
	"""Load 2-dimensional position/trajectory data.

	Args:
		position_name (str, optional): Name of position file. Defaults to "position".

	Returns:
		list: [position_x, position_y]
		position_x (np.array of shape (number_of_positions,)): position (or COLVAR) data of x-dimension (or CV1)
		position_y (np.array of shape (number_of_positions,)): position (or COLVAR) data of y-dimension (or CV2)
	"""
	for file1 in glob.glob(position_name):
		colvar = np.loadtxt(file1)
		position_x = colvar[:-1, 1]
		position_y = colvar[:-1, 2]
	return [position_x, position_y]

def find_periodic_point(x_coord, y_coord, min_grid, max_grid, periodic):
	"""Finds periodic copies of input coordinates. First checks if systems is periodic. If not, returns input coordinate array. Next, it checks if each coordinate is within the boundary range (grid min/max +/- grid_ext). If it is, periodic copies will be made on the other side of the CV-domain. 
	
	Args:
		x_coord (float): CV1-coordinate
		y_coord (float): CV2-coordinate
		min_grid (list): list of CV1-minimum value of grid and CV2-minimum value of grid
		max_grid (list): list of CV1-maximum value of grid and CV2-maximum value of grid
		periodic (list or array of shape (2,)): Specifies if system is periodic. First entry sepcifies if CV1 is periodic. Second entry sepcifies if CV2 is periodic. value of 0 corresponds to non-periodic CV1. Value of 1 corresponds to periodic CV1.
	Returns:
		list: list of [x-coord, y-coord] pairs (i.e. [[x1,y1], [x2,y2], ..., [xn,yn]])
	"""

	coord_list = []
	coord_list.append([x_coord, y_coord])
	
	if periodic[0] == 1 or periodic[1] == 1:
		# Use periodic extension for defining PBC
		grid_length = max_grid - min_grid
		grid_ext = (1 / 4) * grid_length

		# There are potentially 4 points, 1 original and 3 periodic copies, or less.
		copy_record = [0, 0, 0, 0]
		# check for x-copy
		if x_coord < min_grid[0] + grid_ext[0] and periodic[0] == 1:
			coord_list.append([x_coord + grid_length[0], y_coord])
			copy_record[0] = 1
		elif x_coord > max_grid[0] - grid_ext[0] and periodic[0] == 1:
			coord_list.append([x_coord - grid_length[0], y_coord])
			copy_record[1] = 1
		# check for y-copy
		if y_coord < min_grid[1] + grid_ext[1] and periodic[1] == 1:
			coord_list.append([x_coord, y_coord + grid_length[1]])
			copy_record[2] = 1
		elif y_coord > max_grid[1] - grid_ext[1] and periodic[1] == 1:
			coord_list.append([x_coord, y_coord - grid_length[1]])
			copy_record[3] = 1
		# check for xy-copy
		if sum(copy_record) == 2:
			if copy_record[0] == 1 and copy_record[2] == 1:
				coord_list.append([x_coord + grid_length[0], y_coord + grid_length[1]])
			elif copy_record[1] == 1 and copy_record[2] == 1:
				coord_list.append([x_coord - grid_length[0], y_coord + grid_length[1]])
			elif copy_record[0] == 1 and copy_record[3] == 1:
				coord_list.append([x_coord + grid_length[0], y_coord - grid_length[1]])
			elif copy_record[1] == 1 and copy_record[3] == 1:
				coord_list.append([x_coord - grid_length[0], y_coord - grid_length[1]])       
	
	return coord_list


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

def reduce_to_window(input_array, min_grid, grid_space, x_min=-0.5, x_max=0.5, y_min=-1.5, y_max=1.5):
	"""Reduces an 2D input array to a specified range.

	Args:
		input_array (array): 2D array to be reduced
		min_grid (list): list of CV1-minimum value of grid and CV2-minimum value of grid
		grid_space (list): list of CV1-grid spacing and CV2-grid spacing
		x_min (float, optional): lower CV1-value of output array. Defaults to -0.5.
		x_max (float, optional): upper CV1-value of output array. Defaults to 0.5.
		y_min (float, optional): lower CV2-value of output array. Defaults to -1.5.
		y_max (float, optional): upper CV2-value of output array. Defaults to 1.5.

	Returns:
		array: reduced array
	"""
	return input_array[index(y_min, min_grid[1], grid_space[1]): index(y_max, min_grid[1], grid_space[1]), index(x_min, min_grid[0], grid_space[0]): index(x_max, min_grid[0], grid_space[0])]

def find_hp_force(hp_centre_x, hp_centre_y, hp_kappa_x, hp_kappa_y, X , Y, min_grid, max_grid, grid_space, periodic):
	"""Find 2D harmonic potential force. 

	Args:
		hp_centre_x (float): CV1-position of harmonic potential
		hp_centre_y (float): CV2-position of harmonic potential
		hp_kappa_x (float): CV1-force_constant of harmonic potential
		hp_kappa_y (float): CV2-force_constant of harmonic potential
		X (array): 2D array of CV1 grid positions
		Y (array): 2D array of CV2 grid positions
		min_grid (list): list of CV1-minimum value of grid and CV2-minimum value of grid
		max_grid (list): list of CV1-maximum value of grid and CV2-maximum value of grid
		grid_space (list): list of CV1-grid spacing and CV2-grid spacing
		periodic (list or array of shape (2,))): Specifies if system is periodic. First entry sepcifies if CV1 is periodic. Second entry sepcifies if CV2 is periodic. value of 0 corresponds to non-periodic CV1. Value of 1 corresponds to periodic CV1.

	Returns:
		list: [F_harmonic_x, F_harmonic_y]
		F_harmonic_x
		F_harmonic_y
	"""
	#Calculate x-force
	F_harmonic_x = hp_kappa_x * (X - hp_centre_x)
	if periodic[0] == 1:
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
	if periodic[1] == 1:
		grid_length = max_grid[0] - min_grid[0]
		grid_centre = min_grid[0] + grid_length / 2
		if hp_centre_y < grid_centre:
			index_period = index(hp_centre_y + grid_length/2, min_grid[1], grid_space[1])
			F_harmonic_y[index_period:, :] = hp_kappa_y * (Y[index_period:, :] - hp_centre_y - grid_length)
		elif hp_centre_y > grid_centre:
			index_period = index(hp_centre_y - grid_length/2, min_grid[1], grid_space[1])
			F_harmonic_y[:index_period, :] = hp_kappa_y * (Y[:index_period, :] - hp_centre_y + grid_length)

	return [F_harmonic_x, F_harmonic_y]


def find_lw_force(lw_centre_x, lw_centre_y, lw_kappa_x, lw_kappa_y, X , Y, min_grid, max_grid, grid_space, periodic):
	"""Find lower half of 2D harmonic potential force equivalent to f = 2 * lw_kappa * (grid - lw_centre) for grid < lw_centre and f = 0 otherwise. This can change for periodic cases.

	Args:
		lw_centre_x (float): CV1-position of lower wall potential
		lw_centre_y (float): CV2-position of lower wall potential
		lw_kappa_x (float): CV1-force_constant of lower wall potential
		lw_kappa_y (float): CV2-force_constant of lower wall potential
		X (array): 2D array of CV1 grid positions
		Y (array): 2D array of CV2 grid positions
		min_grid (list): list of CV1-minimum value of grid and CV2-minimum value of grid
		max_grid (list): list of CV1-maximum value of grid and CV2-maximum value of grid
		grid_space (list): list of CV1-grid spacing and CV2-grid spacing
		periodic (list or array of shape (2,))): Specifies if system is periodic. First entry sepcifies if CV1 is periodic. Second entry sepcifies if CV2 is periodic. value of 0 corresponds to non-periodic CV1. Value of 1 corresponds to periodic CV1.

	Returns:
		list: [F_wall_x, F_wall_y]
		F_wall_x
		F_wall_y
	"""
	#Calculate x-force
	F_wall_x = np.where(X < lw_centre_x, 2 * lw_kappa_x * (X - lw_centre_x), 0)
	if periodic[0] == 1:
		grid_length = max_grid[0] - min_grid[0]
		grid_centre = min_grid[0] + grid_length/2
		if lw_centre_x < grid_centre:
			index_period = index(lw_centre_x + grid_length/2, min_grid[0], grid_space)
			F_wall_x[:, index_period:] =  2 * lw_kappa_x * (X[:, index_period:] - lw_centre_x - grid_length)        
		elif lw_centre_x > grid_centre:
			index_period = index(lw_centre_x - grid_length/2, min_grid[0], grid_space)
			F_wall_x[:, :index_period] = 0

	#Calculate y-force
	F_wall_y = np.where(Y < lw_centre_y, 2 * lw_kappa_y * (Y - lw_centre_y), 0)
	if periodic[1] == 1:
		grid_length = max_grid[1] - min_grid[1]
		grid_centre = min_grid[1] + grid_length/2
		if lw_centre_y < grid_centre:
			index_period = index(lw_centre_y + grid_length/2, min_grid[1], grid_space)
			F_wall_y[index_period:, :] = 2 * lw_kappa_y * (Y[index_period:, :] - lw_centre_y - grid_length)
		elif lw_centre_y > grid_centre:
			index_period = index(lw_centre_y - grid_length/2, min_grid[1], grid_space)
			F_wall_y[:index_period, :] = 0
	return [F_wall_x, F_wall_y]


def find_uw_force(uw_centre_x, uw_centre_y, uw_kappa_x, uw_kappa_y, X , Y, min_grid, max_grid, grid_space, periodic):
	"""Find upper half of 2D harmonic potential force equivalent to f = 2 * uw_kappa * (grid - uw_centre) for grid > uw_centre and f = 0 otherwise. This can change for periodic cases.

	Args:
		lw_centre_x (float): CV1-position of upper wall potential
		lw_centre_y (float): CV2-position of upper wall potential
		lw_kappa_x (float): CV1-force_constant of upper wall potential
		lw_kappa_y (float): CV2-force_constant of upper wall potential
		X (array): 2D array of CV1 grid positions
		Y (array): 2D array of CV2 grid positions
		min_grid (list): list of CV1-minimum value of grid and CV2-minimum value of grid
		max_grid (list): list of CV1-maximum value of grid and CV2-maximum value of grid
		grid_space (list): list of CV1-grid spacing and CV2-grid spacing
		periodic (list or array of shape (2,))): Specifies if system is periodic. First entry sepcifies if CV1 is periodic. Second entry sepcifies if CV2 is periodic. value of 0 corresponds to non-periodic CV1. Value of 1 corresponds to periodic CV1.

	Returns:
		list: [F_wall_x, F_wall_y]
		F_wall_x
		F_wall_y
	"""

	#Calculate x-force
	F_wall_x = np.where(X > uw_centre_x, 2 * uw_kappa_x * (X - uw_centre_x), 0)
	if periodic[0] == 1:
		grid_length = max_grid[0] - min_grid[0]
		grid_centre = min_grid[0] + grid_length/2
		if uw_centre_x < grid_centre:
			index_period = index(uw_centre_x + grid_length/2, min_grid[0], grid_space)
			F_wall_x[:, index_period:] = 0       
		elif uw_centre_x > grid_centre:
			index_period = index(uw_centre_x - grid_length/2, min_grid[0], grid_space)
			F_wall_x[:, :index_period] = 2 * uw_kappa_x * (X[:, :index_period] - uw_centre_x + grid_length)   
	#Calculate y-force
	F_wall_y = np.where(Y > uw_centre_y, 2 * uw_kappa_y * (Y - uw_centre_y), 0)
	if periodic[1] == 1:
		if uw_centre_y < grid_centre:
			index_period = index(uw_centre_y + grid_length/2, min_grid[1], grid_space)
			F_wall_y[index_period:, :] = 0
		elif uw_centre_y > grid_centre:
			index_period = index(uw_centre_y - grid_length/2, min_grid[1], grid_space)
			F_wall_y[:index_period, :] = 2 * uw_kappa_y * (Y[:index_period, :] - uw_centre_y + grid_length)
	return [F_wall_x, F_wall_y]


# @jit
def mean_force_variance(Ftot_den, Ftot_den2, Ftot_x, Ftot_y, ofv_num_x, ofv_num_y, Ftot_den_limit=1E-10, use_weighted_st_dev=True):
	"""Calculates the variance of the mean force

	Args:
		Ftot_den (array of size (nbins[1], nbins[0])): Cumulative biased probability density
		25Ftot_den2 (array of size (nbins[1], nbins[0])):  Cumulative squared biased probability density
		Ftot_x (array of size (nbins[1], nbins[0])): CV1 component of the Mean Force.
		Ftot_y (array of size (nbins[1], nbins[0])): CV2 component of the Mean Force.
		ofv_num_x (array of size (nbins[1], nbins[0])): intermediate component in the calculation of the CV1 "on the fly variance" ( sum of: pb_t * dfds_x ** 2)
		ofv_num_y (array of size (nbins[1], nbins[0])): intermediate component in the calculation of the CV2 "on the fly variance" ( sum of: pb_t * dfds_y ** 2)
		Ftot_den_limit (scalar): threshold in the cumulative biased probability density where data is discarded. Defaults to 0.
	Returns:
		list : [ofv, ofe]\n
		ofv (array of size (nbins[1], nbins[0])): "on the fly variance"\n
		ofe (array of size (nbins[1], nbins[0])): "on the fly error"
  
	"""    
	# calculate ofe (standard error)
	Ftot_den_sq = np.square(Ftot_den)
	Ftot_den_diff = Ftot_den_sq-Ftot_den2
	if use_weighted_st_dev == True: bessel_corr = np.divide(Ftot_den_sq , Ftot_den_diff, out=np.zeros_like(Ftot_den), where=Ftot_den_diff > 0)
	else: bessel_corr = np.divide(Ftot_den2 , Ftot_den_diff, out=np.zeros_like(Ftot_den), where=Ftot_den_diff > 0)

	ofv_x = np.multiply(np.divide(ofv_num_x , Ftot_den, out=np.zeros_like(Ftot_den), where=Ftot_den > Ftot_den_limit) - np.square(Ftot_x) , bessel_corr )
	ofv_y = np.multiply(np.divide(ofv_num_y , Ftot_den, out=np.zeros_like(Ftot_den), where=Ftot_den > Ftot_den_limit) - np.square(Ftot_y) , bessel_corr )
	
	ofv = np.sqrt(np.square(ofv_x) + np.square(ofv_y))	
	ofe = np.sqrt(abs(ofv_x) + abs(ofv_y))

	return [ofv, ofe]


def patch_to_base_variance(master0, master, Ftot_den_limit=1E-10, use_weighted_st_dev=True):
	"""Patches force terms of a base simulation (alaysed prior to current simulation) with current simulation to return patched "on the fly variance".

	Args:
		master0 (list): Force terms of base simulation (alaysed prior to current simulation) [Ftot_den, Ftot_den2, Ftot_x, Ftot_y, ofv_num_x, ofv_num_y]
		master (list): Force terms of current simulation [Ftot_den, Ftot_den2, Ftot_x, Ftot_y, ofv_num_x, ofv_num_y]
		Ftot_den_limit (int): Truncates the probability density below Ftot_den_limit. Default set to 10**-10.
		use_weighted_st_dev (bool)

	Returns:
		list : [PD_patch, FX_patch, FY_patch, OFV, OFE]\n
		PD_patch (array of size (nbins[1], nbins[0])): Patched probability density\n
		FX_patch (array of size (nbins[1], nbins[0])): Patched Froce gradient in x-direction (CV1 direction)\n
		FY_patch (array of size (nbins[1], nbins[0])): Patched Froce gradient in y-direction (CV2 direction)\n
  		OFV (array of size (nbins[1], nbins[0])): modulus of patched "on the fly variance" \n
		OFE(array of size (nbins[1], nbins[0])): modulus of patched "on the fly error" 
	"""
	#Define names
	[PD0, PD20, FX0, FY0, OFV_num_X0, OFV_num_Y0] = master0
	[PD, PD2, FX, FY, OFV_num_X, OFV_num_Y] = master

	#Patch base_terms with current_terms
	PD_patch = PD0 + PD
	PD2_patch = PD20 + PD2
	FX_patch = np.multiply(PD0, FX0) + np.multiply(PD, FX)
	FY_patch = np.multiply(PD0, FY0) + np.multiply(PD, FY)
	OFV_num_X_patch = OFV_num_X0 + OFV_num_X
	OFV_num_Y_patch = OFV_num_Y0 + OFV_num_Y

	FX_patch = np.divide(FX_patch, PD_patch, out=np.zeros_like(FX_patch), where=PD_patch > Ftot_den_limit)
	FY_patch = np.divide(FY_patch, PD_patch, out=np.zeros_like(FY_patch), where=PD_patch > Ftot_den_limit)
	#Ftot_patch.append([PD_patch, PD2_patch, FX_patch, FY_patch, OFV_num_X_patch, OFV_num_Y_patch])

	#Calculate variance of mean force
	PD_sq = np.square(PD_patch)
	PD_diff = PD_sq - PD2_patch
	if use_weighted_st_dev == True: bessel_corr = np.divide(PD_sq , PD_diff, out=np.zeros_like(PD_patch), where=PD_diff > 0)
	else: bessel_corr = np.divide(PD2_patch , PD_diff, out=np.zeros_like(PD_patch), where=PD_diff > 0)

	OFV_X = np.multiply(np.divide(OFV_num_X_patch, PD_patch, out=np.zeros_like(PD_patch), where=PD_patch > 0) - np.square(FX_patch) , bessel_corr )
	OFV_Y = np.multiply(np.divide(OFV_num_Y_patch, PD_patch, out=np.zeros_like(PD_patch), where=PD_patch > 0) - np.square(FY_patch) , bessel_corr )
	
	OFV = np.sqrt(np.square(OFV_X) + np.square(OFV_Y))
	OFE = np.sqrt(abs(OFV_X) + abs(OFV_Y))
	
	return [PD_patch, FX_patch, FY_patch, OFV, OFE]


# Get the cutoff array
def get_cutoff(Ftot_den=None, Ftot_den_cutoff=0.1, FX=None, FY=None, FES_cutoff=-1):
	"""Finds the cutoff array according to the specifications.

	Args:
		Ftot_den (np.array, optional): If a probability density (Ftot_den) cutoff should be applied, this argument in necessary. Defaults to None.
		Ftot_den_cutoff (float, optional): Specifies the cutoff limit of the probability density. When negative or zero, no probability density cutoff is applied. Defaults to -1.
		FX (np.array, optional): Force gradient of X or CV1. If a free energy surface (FES) cutoff should be applied, this argument in necessary. Defaults to None.
		FY (np.array, optional): Force gradient of Y or CV2. If a free energy surface (FES) cutoff should be applied, this argument in necessary. Defaults to None.
		FES_cutoff (list or float, optional): Required list: [FES_cutoff_limit, min_grid, max_grid, periodic]. If list is not provided, FES_cutoff will not be applied. Defaults to -1.

	Returns:
		np.array: cutoff array with the shape of FY. Elements that correspond to the probability density above the Ftot_den_cutoff or the FES below the FES_cutoff will be 1. Elements outside the cutoff will be 0.
	"""
	
	cutoff = 0
	if hasattr(Ftot_den, "__len__") == True: cutoff = np.where(Ftot_den > Ftot_den_cutoff, 1, 0)
	elif hasattr(FX, "__len__") == True: cutoff = np.ones_like(FX)
	else: print("\n\n*** ERROR***\nPlease either provide a probabilit density (PD or Ftot_den), or a positive FES_cutoff value and provide Ftot_x, Ftot_y")
	
	if hasattr(FES_cutoff, "__len__") == True:
		[X, Y, FES] = FFT_intg_2D(FX, FY, min_grid=FES_cutoff[1], max_grid=FES_cutoff[2], periodic=FES_cutoff[3])
		cutoff = np.where(FES <= FES_cutoff[0], cutoff, 0)
	
	return cutoff


### Integration using Fast Fourier Transform (FFT integration) in 2D
def FFT_intg_2D(FX, FY, min_grid=np.array((-np.pi, -np.pi)), max_grid=np.array((np.pi, np.pi)), periodic=np.array((0,0))):
	"""2D integration of force gradient (FX, FY) to find FES using Fast Fourier Transform.

	Args:
		FX (array of size (nbins[1], nbins[0])): CV1 component of the Mean Force.
		FY (array of size (nbins[1], nbins[0])): CV1 component of the Mean Force.
		min_grid (array, optional): Lower bound of the simulation domain. Defaults to np.array((-np.pi, -np.pi)).
		min_grid (array, optional): Upper bound of the simulation domain. Defaults to np.array((np.pi, np.pi)).
		periodic (list or array of shape (2,))): Specifies if system is periodic. First entry sepcifies if CV1 is periodic. Second entry sepcifies if CV2 is periodic. value of 0 corresponds to non-periodic CV1. Value of 1 corresponds to periodic CV1.

	Returns:
		list : [X, Y, fes]\n
		X (array of size (nbins[1], nbins[0])): CV1 grid positions\n
		Y (array of size (nbins[1], nbins[0])): CV2 grid positions\n
		fes (array of size (nbins[1], nbins[0])): Free Energy Surface
	"""

	#create grid
	nbins_yx = np.shape(FX)
	gridx = np.linspace(min_grid[0], max_grid[0], nbins_yx[1])
	gridy = np.linspace(min_grid[1], max_grid[1], nbins_yx[0])
	grid_spacex = (max_grid[0] - min_grid[0]) / (nbins_yx[0] - 1)
	grid_spacey = (max_grid[1] - min_grid[1]) / (nbins_yx[1] - 1)	
	X, Y = np.meshgrid(gridx, gridy)

	#If system is non-periodic, make (anti-)symmetic copies so that the system appears symmetric.
	if periodic[0] == 0:
		nbins_yx = np.array((nbins_yx[0], nbins_yx[1]*2))
		FX = np.block([[-FX[:,::-1],FX]])
		FY = np.block([[FY[:,::-1],FY]])		
	if periodic[1] == 0:
		nbins_yx = np.array((nbins_yx[0]*2, nbins_yx[1]))
		FX = np.block([[FX],[FX[::-1,:]]])
		FY = np.block([[FY],[-FY[::-1,:]]])

	# Calculate frequency
	freq_1dx = np.fft.fftfreq(nbins_yx[1], grid_spacey)
	freq_1dy = np.fft.fftfreq(nbins_yx[0], grid_spacex)
	freq_x, freq_y = np.meshgrid(freq_1dx, freq_1dy)
	freq_hypot = np.hypot(freq_x, freq_y)
	freq_sq = np.where(freq_hypot != 0, freq_hypot ** 2, 1E-10)
	
 	# FFTransform and integration
	fourier_x = (np.fft.fft2(FX) * freq_x) / (2 * np.pi * 1j * freq_sq)
	fourier_y = (np.fft.fft2(FY) * freq_y) / (2 * np.pi * 1j * freq_sq)
	# Reverse FFT
	fes_x = np.real(np.fft.ifft2(fourier_x))
	fes_y = np.real(np.fft.ifft2(fourier_y))
	# Construct whole FES
	fes = fes_x + fes_y
	
	#if non-periodic, cut FES back to original domain.
	if periodic[0] == 0: fes = fes[:,int(nbins_yx[1]/2):]
	if periodic[1] == 0: fes = fes[:int(nbins_yx[0]/2),:]

	fes = fes - np.min(fes)
	return [X, Y, fes]


# Equivalent to integration MS in Alanine dipeptide notebook.
def intg_2D(FX, FY, min_grid=np.array((-np.pi, -np.pi)), max_grid=np.array((np.pi, np.pi)), nbins=np.array((200, 200))):
	"""2D integration of force gradient (FX, FY) to find FES using finite difference method.
	
	Args:
		FX (array of size (nbins[1], nbins[0])): CV1 component of the Mean Force.
		FY (array of size (nbins[1], nbins[0])): CV2 component of the Mean Force.
		min_grid (array, optional): Lower bound of the simulation domain. Defaults to np.array((-np.pi, -np.pi)).
		min_grid (array, optional): Upper bound of the simulation domain. Defaults to np.array((np.pi, np.pi)).
		nbins (int, optional): number of bins in CV1,CV2. Defaults to np.array((200,200)).
	Returns:
		list : [X, Y, fes]\n
		X (array of size (nbins[1], nbins[0])): CV1 grid positions\n
		Y (array of size (nbins[1], nbins[0])): CV2 grid positions\n
		fes (array of size (nbins[1], nbins[0])): Free Energy Surface
	"""

	gridx = np.linspace(min_grid[0], max_grid[0], nbins[0])
	gridy = np.linspace(min_grid[1], max_grid[1], nbins[1])
	X, Y = np.meshgrid(gridx, gridy)

	FdSx = np.cumsum(FX, axis=1) * np.diff(gridx)[0]
	FdSy = np.cumsum(FY, axis=0) * np.diff(gridy)[0]

	fes = np.zeros(FdSx.shape)
	for i in range(fes.shape[0]):
		for j in range(fes.shape[1]):
			fes[i, j] += np.sum([FdSy[i, 0], -FdSy[0, 0], FdSx[i, j], -FdSx[i, 0]])

	fes = fes - np.min(fes)

	return [X, Y, fes]


#@jit(nopython=True)
def intgrad2(fx, fy, min_grid=np.array((-2, -2)), max_grid=np.array((2, 2)), periodic=np.array((0,0)), intconst=0):
	
	"""This function uses the inverse of the gradient to reconstruct the free energy surface from the mean force components.
	[John D'Errico (2022). Inverse (integrated) gradient (https://www.mathworks.com/matlabcentral/fileexchange/9734-inverse-integrated-gradient), MATLAB Central File Exchange. Retrieved May 17, 2022.]
	[Translated from MatLab to Python by Francesco Serse (https://github.com/Fserse)]

	Args:
		fx (array): (ny by nx) array. X-gradient to be integrated.
		fy (array): (ny by nx) array. X-gradient to be integrated.
		nx (integer): nuber of datapoints in x-direction. Default to 0: will copy the shape of the input gradient.
		ny (integer): nuber of datapoints in y-direction. Default to 0: will copy the shape of the input gradient.
		intconst (float): Minimum value of output FES
		periodic (list or array of shape (2,))): Specifies if system is periodic. First entry sepcifies if CV1 is periodic. Second entry sepcifies if CV2 is periodic. value of 0 corresponds to non-periodic CV1. Value of 1 corresponds to periodic CV1.
		min_grid (list/array of length=2): list/array of minimum value of [x-grid, y-grid]
		max_grid (list/array of length=2):  list/array of maximum value of [x-grid, y-grid]
		nbins (list/array of length=2): list/array of number of data pointis of [x-grid, y-grid]. Default to 0: will copy the shape of the input gradient.

	Returns:
		list : [X, Y, fhat]\n
		X (ny by nx array): X-component of meshgrid\n
		Y (ny by nx array): Y-component of meshgrid\n
		fhat (ny by nx array): integrated free energy surface
	"""

	nx = np.shape(fx)[1]    
	ny = np.shape(fx)[0]
	
	gridx = np.linspace(min_grid[0], max_grid[0], nx)
	gridy = np.linspace(min_grid[1], max_grid[1], ny)
	dy = abs(gridx[1] - gridx[0])
	dx = abs(gridy[1] - gridy[0])
	X, Y = np.meshgrid(gridx, gridy)

	rhs = np.ravel((fx,fy))
	
	Af=np.zeros((4*nx*ny,3))
	
	n=0
	#Equations in x
	for i in range(0,ny):
		#Leading edge
		Af[2*nx*i][0] = 2*nx*i/2
		
		if(periodic[1]): 
			Af[2*nx*i][1] = nx*i+(nx-1)
		else: 
			Af[2*nx*i][1] = nx*i

		Af[2*nx*i][2] = -0.5/dx

		Af[2*nx*i+1][0] = 2*nx*i/2
		Af[2*nx*i+1][1] = nx*i+1
		Af[2*nx*i+1][2] = 0.5/dx
	
		#Loop over inner space
		for j in range(1,nx-1):
			Af[2*nx*i+2*j][0] = int((2*nx*i+2*j)/2)
			Af[2*nx*i+2*j][1] = nx*i+j
			Af[2*nx*i+2*j][2] = -1/dx
	
			Af[2*nx*i+2*j+1][0] = int((2*nx*i+2*j)/2)
			Af[2*nx*i+2*j+1][1] = nx*i+j+1
			Af[2*nx*i+2*j+1][2] = 1/dx
	
		#Trailing edge
		Af[2*nx*(i+1)-2][0] = int((2*nx*(i+1)-2)/2)
		Af[2*nx*(i+1)-2][1] = nx*i+(nx-2)
		Af[2*nx*(i+1)-2][2] = -0.5/dx
	
		Af[2*nx*(i+1)-1][0] = int((2*nx*(i+1)-2)/2)
		if(periodic[1]):
			Af[2*nx*(i+1)-1][1] = nx*i
		else:
			Af[2*nx*(i+1)-1][1] = nx*i+(nx-1)
		Af[2*nx*(i+1)-1][2] = 0.5/dx
	
	
	n=2*nx*ny
	
	#Equations in y
	#Leading edge
	for j in range(0,nx):
	
		Af[2*j+n][0] = 2*j/2 + n/2
		
		if(periodic[0]):
			Af[2*j+n][1] = (ny-1)*nx+j
		else:
			Af[2*j+n][1] = j
		Af[2*j+n][2] = -0.5/dy
	
		Af[2*j+n+1][0] = 2*j/2 + n/2
		Af[2*j+n+1][1] = j+nx
		Af[2*j+n+1][2] = 0.5/dy
	
	#Loop over inner space
	for i in range(1,ny-1):
		for j in range(0,nx):
			
			Af[2*nx*i+2*j+n][0] = int((2*nx*i+2*j+n)/2)
			Af[2*nx*i+2*j+n][1] = j+(i)*nx
			Af[2*nx*i+2*j+n][2] = -1/dy
	
			Af[2*nx*i+2*j+n+1][0] = int((2*nx*i+2*j+n)/2)
			Af[2*nx*i+2*j+n+1][1] = j+(i+1)*nx
			Af[2*nx*i+2*j+n+1][2] = 1/dy
			a=2*nx*i+2*j+n+1
	n=n+2*(ny-1)*nx
	
	#Trailing edge
	for j in range(0,nx):
		Af[2*j+n][0] = int((2*j+n)/2)
		Af[2*j+n][1] = (ny-2)*nx+j
		Af[2*j+n][2] = -0.5/dy
	
		Af[2*j+n+1][0] = int((2*j+n)/2)
		if(periodic[0]):
			Af[2*j+n+1][1] = j
		else:
			Af[2*j+n+1][1] = (ny-1)*nx+j
		Af[2*j+n+1][2] = 0.5/dy
	
	
	#Boundary conditions
	Af[0][2]=1
	Af[1][:]=0
	rhs[0] = intconst
	
	#Solve
	A=sps.csc_matrix((Af[:,2],(Af[:,0],Af[:,1])),shape=(2*ny*nx,ny*nx))
	fhat=spsl.lsmr(A,rhs)
	fhat=fhat[0]
	fhat = np.reshape(fhat,(ny,nx)) 
	#print(fhat.shape)   
	fhat = fhat - np.min(fhat)    

	return [X, Y, fhat]

def plot_recap_2D(X, Y, FES, TOTAL_DENSITY, CONVMAP, CONV_history, CONV_history_time, FES_lim=50, ofe_map_lim=50, FES_step=1, ofe_step=1):
	"""Plots 1. FES, 2. varinace_map, 3. Cumulative biased probability density, 4. Convergece of variance.
	
	Args:
		X (array of size (nbins[1], nbins[0])): CV1 grid positions
		Y (array of size (nbins[1], nbins[0])): CV2 grid positions
		FES (array of size (nbins[1], nbins[0])): Free Energy Surface
		TOTAL_DENSITY (array of size (nbins[1], nbins[0])): Cumulative biased probability density
		CONVMAP (array of size (nbins[1], nbins[0])): varinace_map
		CONV_history (list): Convergece of variance
		CONV_history_time (list): Simulation time corresponding to CONV_history

	"""
	fig, axs = plt.subplots(1, 4, figsize=(16, 3))
	cp = axs[0].contourf(X, Y, FES, levels=range(0, FES_lim, FES_step), cmap='coolwarm', antialiased=False, alpha=0.8);
	cbar = plt.colorbar(cp, ax=axs[0])
	cbar.set_label("Free Energy [kJ/mol]", fontsize=11)
	axs[0].set_ylabel('CV2', fontsize=11)
	axs[0].set_xlabel('CV1', fontsize=11)
	axs[0].set_xlim(np.min(X),np.max(X))
	axs[0].set_ylim(np.min(Y),np.max(Y))
	axs[0].set_title('Free Energy Surface', fontsize=11)

	cp = axs[1].contourf(X, Y, zero_to_nan(CONVMAP), levels=range(0, ofe_map_lim, ofe_step), cmap='coolwarm', antialiased=False, alpha=0.8);
	cbar = plt.colorbar(cp, ax=axs[1])
	cbar.set_label("Standard Deviation [kJ/mol]", fontsize=11)
	axs[1].set_ylabel('CV2', fontsize=11)
	axs[1].set_xlabel('CV1', fontsize=11)
	axs[1].set_xlim(np.min(X),np.max(X))
	axs[1].set_ylim(np.min(Y),np.max(Y))
	axs[1].set_title('Standard Deviation of the Mean Force', fontsize=11)

	cp = axs[2].contourf(X, Y, (TOTAL_DENSITY), cmap='gray_r', antialiased=False, alpha=0.8);  #, locator=ticker.LogLocator()
	cbar = plt.colorbar(cp, ax=axs[2])
	cbar.set_label("Relative count [-]", fontsize=11)
	axs[2].set_ylabel('CV2', fontsize=11)
	axs[2].set_xlabel('CV1', fontsize=11)
	axs[2].set_xlim(np.min(X),np.max(X))
	axs[2].set_ylim(np.min(Y),np.max(Y))
	axs[2].set_title('Total Biased Probability Density', fontsize=11)

	axs[3].plot( CONV_history_time, CONV_history, label="global ofe");
	axs[3].set_ylabel('Standard Deviation [kJ/mol]', fontsize=11)
	axs[3].set_xlabel('Simulation time', fontsize=11)
	axs[3].set_title('Global Convergence of Standard Deviation', fontsize=11)
 
	plt.tight_layout()


# Patch independent simulations
def patch_2D(master_array):
	"""Takes in a collection of force terms and patches them togehter to return the patched force terms

	Args:
		master_array (list): collection of force terms (n * [Ftot_den, Ftot_den2, Ftot_x, Ftot_y, ofv_x, ofv_y])

	Returns:
		list : [FP, FP2, FX, FY, OFV_X, OFV_Y]\n
		FP(array of size (nbins[1], nbins[0])): Patched probability density\n
		FP2(array of size (nbins[1], nbins[0])): Patched (probability density squared)\n
		FX(array of size (nbins[1], nbins[0])): Patched Froce gradient in x-direction (CV1 direction)\n
		FY(array of size (nbins[1], nbins[0])): Patched Froce gradient in y-direction (CV2 direction)\n
  		OFV_X (array of size (nbins[1], nbins[0])): "on the fly variance"-term for the calculation of the variance of the froce gradient in x-direction (CV1 direction)\n
  		OFV_Y (array of size (nbins[1], nbins[0])): "on the fly variance"-term for the calculation of the variance of the froce gradient in y-direction (CV2 direction)
	"""
	nbins_yx = np.shape(master_array[0][0])
	FP = np.zeros(nbins_yx)
	FP2 = np.zeros(nbins_yx)
	FX = np.zeros(nbins_yx)
	FY = np.zeros(nbins_yx)
	OFV_X = np.zeros(nbins_yx)
	OFV_Y = np.zeros(nbins_yx)

	for i in range(len(master_array)):
		FP += master_array[i][0]
		FP2 += master_array[i][1]
		FX += master_array[i][0] * master_array[i][2]
		FY += master_array[i][0] * master_array[i][3]
		OFV_X += master_array[i][4]
		OFV_Y += master_array[i][5]

	FX = np.divide(FX, FP, out=np.zeros_like(FX), where=FP != 0)
	FY = np.divide(FY, FP, out=np.zeros_like(FY), where=FP != 0)

	return [FP, FP2, FX, FY, OFV_X, OFV_Y]

# Patch independent simulations
def patch_2D_simple(master_array):
	"""Takes in a collection of force and patches only the probability density and mean forces

	Args:
		master_array (list): collection of force terms (n * [Ftot_den, Ftot_x, Ftot_y])

	Returns:
		list : [FP, FP2, FX, FY, OFV_X, OFV_Y]\n
		FP (array of size (nbins[1], nbins[0])): Patched probability density\n
		FX (array of size (nbins[1], nbins[0])): Patched Froce gradient in x-direction (CV1 direction)\n
		FY (array of size (nbins[1], nbins[0])): Patched Froce gradient in y-direction (CV2 direction)
	"""
	nbins_yx = np.shape(master_array[0][0])
	FP = np.zeros(nbins_yx)
	FX = np.zeros(nbins_yx)
	FY = np.zeros(nbins_yx)


	for i in range(len(master_array)):
		FP += master_array[i][0]
		FX += master_array[i][0] * master_array[i][1]
		FY += master_array[i][0] * master_array[i][2]

	FX = np.divide(FX, FP, out=np.zeros_like(FX), where=FP != 0)
	FY = np.divide(FY, FP, out=np.zeros_like(FY), where=FP != 0)

	return [FP, FX, FY]


def plot_patch_2D(X, Y, FES, TOTAL_DENSITY, lim=50):
	"""Plots 1. FES, 2. Cumulative biased probability density
	
	Args:
		X (array of size (nbins[1], nbins[0])): CV1 grid positions
		Y (array of size (nbins[1], nbins[0])): CV2 grid positions
		FES (array of size (nbins[1], nbins[0])): Free Energy Surface
		TOTAL_DENSITY (array of size (nbins[1], nbins[0])): Cumulative biased probability density
	"""
	fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
	cp = axs[0].contourf(X, Y, FES, levels=range(0, lim, 1), cmap='coolwarm', antialiased=False, alpha=0.8);
	cbar = plt.colorbar(cp, ax=axs[0])
	axs[0].set_ylabel('CV2', fontsize=11)
	axs[0].set_xlabel('CV1', fontsize=11)
	axs[0].set_title('Free Energy Surface', fontsize=11)

	cp = axs[1].contourf(X, Y, TOTAL_DENSITY, cmap='gray_r', antialiased=False, alpha=0.8);
	cbar = plt.colorbar(cp, ax=axs[1])
	axs[1].set_ylabel('CV2', fontsize=11)
	axs[1].set_xlabel('CV1', fontsize=11)
	axs[1].set_title('Total Biased Probability Density', fontsize=11)

def bootstrap_2D_new(X, Y, force_array, n_bootstrap, min_grid=np.array((-3, -3)), max_grid=np.array((3, 3)), periodic=np.array((0,0)), FES_cutoff=0, Ftot_den_cutoff=0, non_exploration_penalty=0):
	"""Algorithm to determine bootstrap error. Takes in a collection of force-terms and with each itteration, a random selection of force-terms will be used to calculate a FES. The average and st.dev of all FESes will be calculated.

	Args:
		X (array of size (nbins[1], nbins[0])): CV1 grid positions
		Y (array of size (nbins[1], nbins[0])): CV2 grid positions
		force_array (list): collection of force terms (n * [Ftot_den, Ftot_x, Ftot_y])
		n_bootstrap (int): bootstrap iterations
		min_grid (array, optional): Lower bound of the force domain. Defaults to np.array((-np.pi, -np.pi)).
		max_grid (array, optional): Upper bound of the force domain. Defaults to np.array((np.pi, np.pi)).
		periodic (list or array of shape (2,))): Specifies if system is periodic. First entry sepcifies if CV1 is periodic. Second entry sepcifies if CV2 is periodic. value of 0 corresponds to non-periodic CV1. Value of 1 corresponds to periodic CV1.
		FES_cutoff (float, optional): Cutoff applied to error calculation for FES values over the FES_cutoff. If the cutoff applies, the error will be set to zero, otherwise the error will stay the same. Defaults to 0. When FES_cutoff <= 0, no cufoff is applied. Use with care, computing the fes in the loop renders the calculation slow.
		Ftot_den_cutoff (float, optional): Cutoff applied to error calculation for probability density (Ftot_den) values below the Ftot_den_cutoff. If the cutoff applies, the error will be set to zero, otherwise the error will stay the same. Defaults to 0.1. When Ftot_den_cutoff <= 0, no cufoff is applied. 
		non_exploration_penalty (float, optional): Turns zero-value error to the non_exploration_penalty value. This should be used in combination with the cutoff. If some part of CV space hasn't been explored, or has a FES value that is irrelevanlty high, the cutoff will set the error of that region to zero. If the non_exploration_penalty is larger than zero, the error of that region will take the value of the non_exploration_penalty instead of zero. Default is set to 0.

	Returns:
		list: [FES_avr, sd_fes, sd_fes_prog ]\n
		FES_avr (array of size (nbins[1], nbins[0])): Average of all FESes generated.\n
		sd_fes (array of size (nbins[1], nbins[0])): Map of standard deviation of all FESes generated.\n
		sd_fes_prog (array of size (n_bootstrap,)): The standard deviation of all FESes generated after each itteration.
	"""
   
	#Define constants and lists
	nbins_yx = np.shape(X)
	n_forces = int(len(force_array))
	sd_fes_prog = np.zeros(n_bootstrap)    
	FES_avr= np.zeros_like(X)
	M2 = np.zeros_like(X)

	#Patch force array without randdom choice. If applicable, find Ftot_den and FES cutoff
	[Ftot_den, Ftot_x, Ftot_y] = patch_2D_simple(force_array)
	cutoff = np.ones(nbins_yx)
	if Ftot_den_cutoff > 0: cutoff = np.where(Ftot_den > Ftot_den_cutoff, cutoff, 0) 
	if FES_cutoff > 0: 
		[X, Y, FES] = FFT_intg_2D(Ftot_x, Ftot_y, min_grid=min_grid, max_grid=max_grid, periodic=periodic)
		cutoff = np.where(FES < FES_cutoff, cutoff, 0)


	for iteration in range(n_bootstrap):
		
		#Randomly choose forward forces and backward forces and save to force array
		random_sample_index =  np.random.choice(n_forces-1, size=n_forces)      
		force = force_array[random_sample_index]
  
		#Patch forces
		[Ftot_den, Ftot_x, Ftot_y] = patch_2D_simple(force)
  
		#Calculate FES. if there is a FES_cutoff, find cutoff. 
		[X, Y, FES] = FFT_intg_2D(Ftot_x, Ftot_y, min_grid=min_grid, max_grid=max_grid, periodic=periodic)
  
		# calculate standard devaition using Welford’s method
		delta = FES - FES_avr
		FES_avr += delta/(iteration+1)
		delta2 = FES - FES_avr
		M2 += delta*delta2
		if iteration > 0:
			sd_fes = np.sqrt(M2 / (iteration))
			if Ftot_den_cutoff > 0 or FES_cutoff > 0: sd_fes *= cutoff
			if non_exploration_penalty > 0: sd_fes = np.where(cutoff > 0.5, sd_fes, non_exploration_penalty)
			sd_fes_prog[iteration] = np.sum(sd_fes)/(nbins_yx[0]*nbins_yx[1])
		
			#print progress
			print_progress(iteration+1,n_bootstrap,variable_name='Bootstrap Average Standard Deviation',variable=round(sd_fes_prog[iteration],3))        
			
	# return [FES_avr, cutoff, var_fes, sd_fes, variance_prog, stdev_prog, var_fes_prog, sd_fes_prog ]
	return [FES_avr, sd_fes, sd_fes_prog]


def plot_bootstrap(X, Y, FES, sd_fes, sd_fes_prog, FES_lim=11, sd_lim=11, FES_levels=None, sd_levels=None):
	"""Plots result of bootstrap analysis. 1. Average FES, 2. average varinace, 3. variance progression

	Args:
		X (array of size (nbins[1], nbins[0])): CV1 grid positions
		Y (array of size (nbins[1], nbins[0])): CV2 grid positions
		FES (array of size (nbins[1], nbins[0])): Free Energy Surface
		sd_fes (array of size (nbins[1], nbins[0])): Map of standard deviation of all FESes generated.
		sd_fes_prog (list / np.array of size (bootstrap_iterations,)): Progression of the standard deviation (of all FESes generated after each bootstrap itteration).
		FES_lim (int, optional): Upper energy limit of FES plot. Defaults to 11.
		sd_lim (int, optional): Upper variance limit of variance plot. Defaults to 11.
		FES_levels (int, optional): Amout of contour levels shown in FES plot. Default is set to None, in which case FES_levels = int(FES_lim + 1).
		ofe_levels (int, optional): Amout of contour levels shown in standard deviation plot. Default is set to None, in which case FES_levels = int(FES_lim + 1).

	"""
 
	if FES_levels == None: FES_levels = int(FES_lim + 1)
	if sd_levels == None: sd_levels = int(sd_lim + 1)
	
	fig, axs = plt.subplots(1, 3, figsize=(15, 4))
	cp = axs[0].contourf(X, Y, FES, levels=np.linspace(0, FES_lim, FES_levels), cmap='coolwarm', antialiased=False, alpha=0.8);
	cbar = plt.colorbar(cp, ax=axs[0])
	axs[0].set_ylabel('CV2', fontsize=11)
	axs[0].set_xlabel('CV1', fontsize=11)
	axs[0].set_title('Average FES', fontsize=11)

	cp = axs[1].contourf(X, Y, sd_fes, levels=np.linspace(0, sd_lim, sd_levels), cmap='coolwarm', antialiased=False, alpha=0.8);
	cbar = plt.colorbar(cp, ax=axs[1])
	cbar.set_label("Variance of Average FES [kJ/mol]$^2$", rotation=270)
	axs[1].set_ylabel('CV2', fontsize=11)
	axs[1].set_xlabel('CV1', fontsize=11)
	axs[1].set_title('Bootstrap Variance of FES', fontsize=11)


	axs[2].plot( range(len(sd_fes_prog)), sd_fes_prog);
	axs[2].set_ylabel('Average Variance of Average FES [kJ/mol]$^2$', fontsize=11)
	axs[2].set_xlabel('Bootstrap iterations', fontsize=11)
	axs[2].set_title('Global Convergence of Bootstrap Variance', fontsize=11)

	plt.rcParams["figure.figsize"] = (5,4)



def save_npy(object, file_name):
	"""Saves np.array in a file with .npy format

	Args:
		object (np.array): object to be saved. Must be a numpy array.
		file_name (string): Name of file
	"""
	with open(file_name, "wb") as fw:
		np.save(fw, object)


def load_npy(name):
	"""Loads np.array of a file with .npy format

	Args:
		name (string): Name of file

	Returns:
		np.array: object to be loaded. Must be a numpy array.
	"""
	with open(name, "rb") as fr:
		return np.load(fr)

def save_pkl(object, file_name):
	"""Saves a list/array in a file with .pkl format

	Args:
		object (any): object to be saved
		file_name (string): Name of file
	"""
	with open(file_name, "wb") as fw:
		pickle.dump(object, fw)


def load_pkl(name):
	"""Loads list/array of a file with .pkl format

	Args:
		name (string): Name of file

	Returns:
		any: object to be loaded
	"""
	with open(name, "rb") as fr:
		return pickle.load(fr)
	
		
def zero_to_nan(input_array):
	"""Function to turn all zero-elements to np.nan. Works for any shapes.

	Args:
		input_array (array of arbitrary shape): non specific array

	Returns:
		array: input array with zero-elements turned to np.nan
	"""
	output_array = np.zeros_like(input_array)
	for ii in range(len(input_array)):
		for jj in range(len(input_array[ii])):
			if input_array[ii][jj] <= 0: output_array[ii][jj] = np.nan
			else: output_array[ii][jj] = input_array[ii][jj]
	return output_array



def coft(HILLS="HILLS",FES="FES",kT=1,WellTempered=-1,total_number_of_hills=100,stride=10,
			min_grid=np.array((-np.pi, -np.pi)), max_grid=np.array((np.pi, np.pi)), nbins=np.array((200, 200))):
	"""Compute a time-independent estimate of the Mean Thermodynamic Force, i.e. the free energy gradient in 2D CV spaces.

	Args:
		HILLS (array): HILLS array. Defaults to "HILLS".
		FES (array of size (nbins[0], nbins[1]))
		kT (float, optional): Boltzmann constant multiplied with temperature (reduced format, 120K -> 1).
		WellTempered (binary, optional): Is the simulation well tempered? 1 or yes and 0 for no. Defaults to 1.
		total_number_of_hills (int, optional): Number of HILLS to analyse. Defaults to 100.
		min_grid (array, optional): Lower bound of the force domain. Defaults to np.array((-np.pi, -np.pi)).
		max_grid (array, optional): Upper bound of the force domain. Defaults to np.array((np.pi, np.pi)).
		nbins (array, optional): number of bins in CV1,CV2. Defaults to np.array((200,200)).
  
	Returns:
		list : [c,Bias]\n
		c (array of size (total_numer_of_hills,)): Ensemble average of the bias in the unperturbed ensemble, calculated after the deposition of each metadynamics hill.\n
		Bias (array of size (nbins[0], nbins[1])): Metadynamics Bias reconstructed from HILLS data.
	"""

	gridx = np.linspace(min_grid[0], max_grid[0], nbins[0])
	gridy = np.linspace(min_grid[1], max_grid[1], nbins[1])
	grid_space = np.array(((max_grid[0] - min_grid[0]) / (nbins[0]-1), (max_grid[1] - min_grid[1]) / (nbins[1]-1)))
	X, Y = np.meshgrid(gridx, gridy)

	# Initialize force terms
	Bias = np.zeros(nbins[::-1])

	# Definition Gamma Factor, allows to switch between WT and regular MetaD
	if WellTempered < 1:
		Gamma_Factor = 1
	else:
		gamma = HILLS[0, 6]
		Gamma_Factor = (gamma - 1) / (gamma)
		
	for i in range(total_number_of_hills):
		
		# Build metadynamics potential
		s_x = HILLS[i, 1]  # centre x-position of Gaussian
		s_y = HILLS[i, 2]  # centre y-position of Gaussian
		sigma_meta2_x = HILLS[i, 3] ** 2  # width of Gaussian
		sigma_meta2_y = HILLS[i, 4] ** 2  # width of Gaussian
		height_meta = HILLS[i, 5] * Gamma_Factor  # Height of Gaussian
		
		kernelmeta_x = np.exp( - np.square(gridx - s_x) / (2 * sigma_meta2_x)) * height_meta
		kernelmeta_y = np.exp( - np.square(gridy - s_y) / (2 * sigma_meta2_y))
		Bias += np.outer(kernelmeta_y, kernelmeta_x)
		
		if i==0: 
			for k in range(stride):
				c=np.sum(np.exp(-FES/kT))/np.sum(np.exp(-Bias/kT)*np.exp(-FES/kT))
			print_progress(i, total_number_of_hills,variable_name='exp(c(t)/kT)',variable=c)
		else: 
			for k in range(stride):
				c=np.append(c,np.sum(np.exp(-FES/kT))/np.sum(np.exp(-Bias/kT)*np.exp(-FES/kT)))
			print_progress(i, total_number_of_hills,variable_name='exp(c(t)/kT)',variable=c[-1])
	return [c,Bias]
  
  
def print_progress(iteration, total, bar_length=50, variable_name='progress variable' , variable=0):
	"""Function to show a progress bar, that fills up as the iteration number reaches the total. Prints a variable at the end of the progress bar, that can be continiously updated.

	Args:
		iteration (int): Currrent iteration
		total (int): Total iterations
		bar_length (int, optional): Length of the progress bar. Defaults to 50.
		variable_name (str, optional): Name of variable that is being shown at the end of the progress bar. Defaults to 'progress variable'.
		variable (float, optional): Variable that is being shown at the end of the progress bar. Defaults to 0.
	"""
	progress = (iteration / total)
	arrow = '*' * int(round(bar_length * progress))
	spaces = ' ' * (bar_length - len(arrow))
	print(f'\r|{arrow}{spaces}| {int(progress * 100)}% | {variable_name}: {variable}', end='', flush=True)


## 
def MFI_2D(HILLS="HILLS", position_x="position_x", position_y="position_y", bw=np.array((0.1,0.1)), kT=1,
			min_grid=np.array((-np.pi, -np.pi)), max_grid=np.array((np.pi, np.pi)), nbins=np.array((200, 200)),
			error_pace=-1, base_terms = 0, window_corners=[], WellTempered=1, nhills=-1, periodic = np.array((0,0)), 
			Ftot_den_limit = 1E-10, FES_cutoff = -1, Ftot_den_cutoff = 0.1, non_exploration_penalty = 0, use_weighted_st_dev = True,
			hp_centre_x=0.0, hp_centre_y=0.0, hp_kappa_x=0, hp_kappa_y=0,
			lw_centre_x=0.0, lw_centre_y=0.0, lw_kappa_x=0, lw_kappa_y=0,
			uw_centre_x=0.0, uw_centre_y=0.0, uw_kappa_x=0, uw_kappa_y=0, 
			F_static_x = np.zeros((1,1)), F_static_y = np.zeros((1,1)), ref_fes = np.zeros((1,1))):
	"""Compute a time-independent estimate of the Mean Thermodynamic Force, i.e. the free energy gradient in 2D CV spaces.

	Args:
		HILLS (array): HILLS array. The colums have format: [time [ps], position_x [nm], position_y [nm], MetaD_sigma_x [nm], MetaD_sigma_y [nm], MetaD_height [nm], MetaD_biasfactor]
		position_x (array): CV1 array. Defaults to "position_x".
		position_y (array): CV2 array. Defaults to "position_y".
		bw (list or array of shape (2,), optional): Scalar, bandwidth for the construction of the KDE estimate of the biased probability density. First entry is the bandwidth for CV1 and second entry is the bandwidth for CV2. Defaults to np.array((0.1,0.1)).
		kT (float, optional): Boltzmann constant multiplied with temperature (reduced format, 120K -> 1).
		min_grid (array, optional): Lower bound of the force domain. Defaults to np.array((-np.pi, -np.pi)).
		max_grid (array, optional): Upper bound of the force domain. Defaults to np.array((np.pi, np.pi)).
		nbins (array, optional): number of bins in CV1,CV2. First enrty is the number of bins in CV1 and the second entry is the number of bins in CV2! Defaults to np.array((200,200)). 
		error_pace (int, optional): Pace for the calculation of the on-the-fly measure of global convergence. Defaults to 1, change it to a higher value if FES_cutoff>0 is used. 
		base_terms (int or list, optional): When set to 0, inactive. When activated, "on the fly" variance is calculated as a patch to base (previous) simulation. To activate, put force terms of base simulation ([Ftot_den, Ftot_den2, Ftot_x, Ftot_y, ofv_x, ofv_y]). Defaults to 0.
		window_corners (list, optional): When set to [], inactive. When activated, error is ALSO calculated for mean force in the window. To activate, put the min and max values of the window ([min_x, max_x, min_y, max_y]). Defaults to [].
		WellTempered (binary, optional): Is the simulation well tempered? 1 for yes and 0 for no. Defaults to 1.
		nhills (int, optional): Number of HILLS to analyse, -1 for the entire HILLS array. Defaults to -1, i.e. the entire dataset.
		periodic (list or array of shape (2,), optional): Specifies if system is periodic. First entry sepcifies if CV1 is periodic. Second entry sepcifies if CV2 is periodic. value of 0 corresponds to non-periodic CV1. Value of 1 corresponds to periodic CV1. Defaults to np.array((0,0)).
		Ftot_den_limit (float, optional): Truncation of the probability density for numerical reasons, to avaiod devisions by zero (or suare root of negative numbers). If the probability density (Ftot_den) of some CV region is lover than the Ftot_den_limit, it will be set to zero. Default is set to 1E-10.
		FES_cutoff (float, optional): Cutoff applied to error calculation for FES values over the FES_cutoff. If the cutoff applies, the error will be set to zero, otherwise the error will stay the same. Defaults to 0. When FES_cutoff <= 0, no cufoff is applied. Use with care, computing the fes in the loop renders the calculation slow.
		Ftot_den_cutoff (float, optional): Cutoff applied to error calculation for probability density (Ftot_den) values below the Ftot_den_cutoff. If the cutoff applies, the error will be set to zero, otherwise the error will stay the same. Defaults to 0.1. When Ftot_den_cutoff <= 0, no cufoff is applied. 
		non_exploration_penalty (float, optional): Turns zero-value error to the non_exploration_penalty value. This should be used in combination with the cutoff. If some part of CV space hasn't been explored, or has a FES value that is irrelevanlty high, the cutoff will set the error of that region to zero. If the non_exploration_penalty is larger than zero, the error of that region will take the value of the non_exploration_penalty instead of zero. Default is set to 0.
		use_weighted_st_dev (bool, optional): When set to True, the calculated error will be the weighted standard deviation ( var^0.5 ). When set to False, the calculated error will be the standard error ( (var/n_sample)^0.5 ). Defaults to True. (The standard devaition is expected to converge after enough time, while the standard error is expected to decrease as more datapoints are added.)
		hp_centre_x (float, optional): CV1-position of harmonic potential. Defaults to 0.0.
		hp_centre_y (float, optional): CV2-position of harmonic potential. Defaults to 0.0.
		hp_kappa_x (int, optional): CV1-force_constant of harmonic potential. Defaults to 0.
		hp_kappa_y (int, optional): CV2-force_constant of harmonic potential. Defaults to 0.
		lw_centre_x (float, optional): CV1-position of lower wall potential. Defaults to 0.0.
		lw_centre_y (float, optional): CV2-position of lower wall potential. Defaults to 0.0.
		lw_kappa_x (int, optional): CV1-force_constant of lower wall potential. Defaults to 0.
		lw_kappa_y (int, optional): CV2-force_constant of lower wall potential. Defaults to 0.
		uw_centre_x (float, optional): CV1-position of upper wall potential. Defaults to 0.0.
		uw_centre_y (float, optional): CV2-position of upper wall potential. Defaults to 0.0.
		uw_kappa_x (int, optional): CV1-force_constant of upper wall potential. Defaults to 0.
		uw_kappa_y (int, optional): CV2-force_constant of upper wall potential. Defaults to 0.
        F_static (array, optional): Option to provide a starting bias potential that remains constant through the algorithm. This could be a harmonic potential, an previously used MetaD potential or any other bias potential defined on the grid. Defaults to np.zeros((1,1)), which will automatically set F_static to a zero-array with shape=nbins.
        compare_to_reference_FES (int, optional): Do you wanto to compare with a reference FES (dev option). Defaults to 0. 
		ref_fes (array, optional): Reference FES (dev options). Defaults to an array of zeros.

	Returns:
 
		list: [X, Y, Ftot_den, Ftot_x, Ftot_y, ofv, ofe, cutoff, volume_history, ofe_history, time_history, Ftot_den2, ofv_num_x, ofv_num_y]
		
		X (array of size (nbins[1], nbins[0])): CV1 grid positions\n
		Y (array of size (nbins[1], nbins[0])): CV2 grid positions\n
		Ftot_den (array of size (nbins[1], nbins[0])): Cumulative biased probability density, equivalent to an unbiased histogram of samples in CV space.\n
		Ftot_x (array of size (nbins[1], nbins[0])): CV1 component of the Mean Force.\n
		Ftot_y (array of size (nbins[1], nbins[0])): CV2 component of the Mean Force.\n
		ofv (array of size (nbins[1], nbins[0])): on the fly variance estimate of the local convergence\n
		ofe (array of size (nbins[1], nbins[0])): on the fly error estimate of the local convergence\n
		cutoff (array of size (nbins[1], nbins[0])): Array of ones and zeros. By default array of only ones. When the FES or probablity density (Ftot_den) values are outside their respective cutoff, the corresponding cutoff value will be a zero.\n
		volume_history (list of size (nbins[1], nbins[0])): List of scalars indicating the explored volume, as a percentage of the ones in the cutoff array.\n
		ofe_history (list of size (total_number_of_hills/error_pace,)): Running estimate of the global convergence of the mean force by calculating the statistical error of the mean force. Error calculated and added to list with error_pace.\n
		ofe_history_window (optional)(list of size (total_number_of_hills/error_pace)): running estimate of the ofe within the "window" (specified and activated by using the input window_corners).\n
		time_history (list of size (total_number_of_hills/error_pace,)): time array of volume_history, ofe_history and ofe_history_window if applicable.\n
		Ftot_den2 (array of size (nbins[1], nbins[0])): Cumulative squared biased probability density\n
		ofv_x (array of size (nbins[1], nbins[0])): intermediate component in the calculation of the CV1 "on the fly variance" ( sum of: pb_t * dfds_x ** 2)\n
		ofv_y (array of size (nbins[1], nbins[0])): intermediate component in the calculation of the CV2 "on the fly variance" ( sum of: pb_t * dfds_y ** 2)
	"""

	gridx = np.linspace(min_grid[0], max_grid[0], nbins[0])
	gridy = np.linspace(min_grid[1], max_grid[1], nbins[1])
	grid_space = np.array(((max_grid[0] - min_grid[0]) / (nbins[0]-1), (max_grid[1] - min_grid[1]) / (nbins[1]-1)))
	X, Y = np.meshgrid(gridx, gridy)
	stride = int(len(position_x) / len(HILLS))
	bw_xy2  = bw[0] * bw[1]
	bw_x2 = bw[1] ** 2
	bw_y2 = bw[1] ** 2
	const = (1 / (np.sqrt(bw[0] * bw[1]) * np.sqrt(2 * np.pi) * stride))

	# Optional - analyse only nhills, if nhills is set
	if nhills > 0: total_number_of_hills = nhills
	else: total_number_of_hills = len(HILLS)
	if error_pace == -1: error_pace = int(total_number_of_hills / 100)
	elif error_pace == -2: error_pace = int(total_number_of_hills / 10)
	elif error_pace < 0: error_pace = total_number_of_hills
	if FES_cutoff > 0: FES_cutoff = [FES_cutoff, min_grid, max_grid, periodic]
 
	# Initialize force terms
	Fbias_x = np.zeros(nbins[::-1])
	Fbias_y = np.zeros(nbins[::-1])
	Ftot_num_x = np.zeros(nbins[::-1])
	Ftot_num_y = np.zeros(nbins[::-1])
	Ftot_den = np.zeros(nbins[::-1])
	Ftot_den2 = np.zeros(nbins[::-1])
	ofv_num_x = np.zeros(nbins[::-1])
	ofv_num_y = np.zeros(nbins[::-1])
	cutoff = np.ones(nbins[::-1])
	volume_history = []
	ofe_history = []
	aad_history = []
	time_history = []
	ofeMAP = []
	aadMAP = []
	if len(window_corners) == 4: ofe_history_window = []

	#Calculate static force
	if np.shape(F_static_x) != (nbins[1], nbins[0]): F_static_x = np.zeros(nbins[::-1])
	if np.shape(F_static_y) != (nbins[1], nbins[0]): F_static_y = np.zeros(nbins[::-1])
	if hp_kappa_x > 0 or hp_kappa_y > 0:
		[Force_x, Force_y] = find_hp_force(hp_centre_x, hp_centre_y, hp_kappa_x, hp_kappa_y, X , Y, min_grid, max_grid, grid_space, periodic)
		F_static_x += Force_x
		F_static_y += Force_y
	if lw_kappa_x > 0 or lw_kappa_y > 0:
		[Force_x, Force_y] = find_lw_force(lw_centre_x, lw_centre_y, lw_kappa_x, lw_kappa_y, X , Y, periodic)
		F_static_x += Force_x
		F_static_y += Force_y
	if uw_kappa_x > 0 or uw_kappa_y > 0:
		[Force_x, Force_y] = find_uw_force(uw_centre_x, uw_centre_y, uw_kappa_x, uw_kappa_y, X , Y, periodic)
		F_static_x += Force_x
		F_static_y += Force_y

	# Definition Gamma Factor, allows to switch between WT and regular MetaD
	if WellTempered < 1: Gamma_Factor = 1
	else: Gamma_Factor = (HILLS[0, 6] - 1) / (HILLS[0, 6])
		
	for i in range(total_number_of_hills):
		
		# Build metadynamics potential
		s_x = HILLS[i, 1]  # centre x-position of Gaussian
		s_y = HILLS[i, 2]  # centre y-position of Gaussian
		sigma_meta2_x = HILLS[i, 3] ** 2  # width of Gaussian
		sigma_meta2_y = HILLS[i, 4] ** 2  # width of Gaussian
		height_meta = HILLS[i, 5] * Gamma_Factor  # Height of Gaussian

		periodic_images = find_periodic_point(s_x, s_y, min_grid, max_grid, periodic)
		for j in range(len(periodic_images)):
			kernelmeta_x = np.exp( - np.square(gridx - periodic_images[j][0]) / (2 * sigma_meta2_x)) * height_meta
			kernelmeta_y = np.exp( - np.square(gridy - periodic_images[j][1]) / (2 * sigma_meta2_y))
			Fbias_x += np.outer(kernelmeta_y, np.multiply(kernelmeta_x, (gridx - periodic_images[j][0])) / sigma_meta2_x )
			Fbias_y += np.outer(np.multiply(kernelmeta_y, (gridy - periodic_images[j][1])) / sigma_meta2_y, kernelmeta_x )

		# Estimate the biased proabability density p_t ^ b(s)
		pb_t = np.zeros(nbins[::-1])
		Fpbt_x = np.zeros(nbins[::-1])
		Fpbt_y = np.zeros(nbins[::-1])

		data_x = position_x[i * stride: (i + 1) * stride]
		data_y = position_y[i * stride: (i + 1) * stride]

		for j in range(stride):
			periodic_images = find_periodic_point(data_x[j], data_y[j], min_grid, max_grid, periodic)
			for k in range(len(periodic_images)):
				kernel_x = np.exp( - np.square(gridx - periodic_images[k][0]) / (2 * bw_x2)) * const #add constant here for less computations
				kernel_y = np.exp( - np.square(gridy - periodic_images[k][1]) / (2 * bw_y2))
				kernel = np.outer(kernel_y, kernel_x)
				kernel_x *= kT / bw_xy2 #add constant here for less computations

				pb_t += kernel
				Fpbt_x += np.outer(kernel_y, np.multiply(kernel_x, (gridx - periodic_images[k][0])) )
				Fpbt_y += np.outer(np.multiply(kernel_y, (gridy - periodic_images[k][1])) , kernel_x )

		# Calculate total probability density
		pb_t = np.where(pb_t > Ftot_den_limit, pb_t, 0)  # truncated probability density of window
		Ftot_den += pb_t
		
		# Calculate x-component of Force
		dfds_x = np.divide(Fpbt_x, pb_t, out=np.zeros_like(Fpbt_x), where=pb_t > 0) + Fbias_x - F_static_x
		Ftot_num_x += np.multiply(pb_t, dfds_x)
		
		# Calculate y-component of Force
		dfds_y = np.divide(Fpbt_y, pb_t, out=np.zeros_like(Fpbt_y), where=pb_t > 0) + Fbias_y - F_static_y
		Ftot_num_y += np.multiply(pb_t, dfds_y)

		# Calculate on the fly error components
		Ftot_den2 += np.square(pb_t)
		ofv_num_x += np.multiply(pb_t, np.square(dfds_x))
		ofv_num_y += np.multiply(pb_t, np.square(dfds_y))

		if (i + 1) % int(error_pace) == 0 or (i+1) == total_number_of_hills:
			#calculate forces
			Ftot_x = np.divide(Ftot_num_x, Ftot_den, out=np.zeros_like(Fpbt_x), where=Ftot_den > 0)
			Ftot_y = np.divide(Ftot_num_y, Ftot_den, out=np.zeros_like(Fpbt_y), where=Ftot_den > 0)

			# calculate ofe (standard error)
			if base_terms == 0:
				[ofv, ofe] = mean_force_variance(Ftot_den, Ftot_den2, Ftot_x, Ftot_y, ofv_num_x, ofv_num_y, use_weighted_st_dev=use_weighted_st_dev)
				[Ftot_den_tot, Ftot_x_tot, Ftot_y_tot] = [np.array(Ftot_den), np.array(Ftot_x), np.array(Ftot_y)]
			elif len(base_terms) == 6:
				[Ftot_den_tot, Ftot_x_tot, Ftot_y_tot, ofv, ofe] = patch_to_base_variance(base_terms, [Ftot_den, Ftot_den2, Ftot_x, Ftot_y, ofv_num_x, ofv_num_y], use_weighted_st_dev=use_weighted_st_dev)

			# If specified, apply cutoff to error and/or non_exploration_pentaly
			if Ftot_den_cutoff > 0 or hasattr(FES_cutoff, "__len__"): cutoff = get_cutoff(Ftot_den_tot, Ftot_den_cutoff=Ftot_den_cutoff, FX=Ftot_x_tot, FY=Ftot_y_tot, FES_cutoff=FES_cutoff)			
			if non_exploration_penalty > 0: ofe = np.where(cutoff > 0.5, ofe, non_exploration_penalty)
			else: ofe = np.where(cutoff > 0.5, ofe, 0) 
			#ofeMAP.append(ofe)

			#Calculate averaged global error
			absolute_explored_volume = np.count_nonzero(cutoff)
			volume_history.append(absolute_explored_volume/cutoff.size)
			if non_exploration_penalty <= 0: ofe_history.append( np.sum(ofe) / absolute_explored_volume)
			else: ofe_history.append( np.sum(ofe) / (nbins[0]*nbins[1]))
			time_history.append(HILLS[i,0] + HILLS[2,0] - HILLS[1,0])
			if len(window_corners) == 4:
				ofe_cut_window = reduce_to_window(ofe, min_grid, grid_space, x_min=window_corners[0], x_max=window_corners[1], y_min=window_corners[2], y_max=window_corners[3]) 
				ofe_history_window.append(np.sum(ofe_cut_window) / (np.count_nonzero(ofe_cut_window)))

			# Check if aadMAP really needed here
         	#Find Absolute deviation
			if np.shape(ref_fes) == (nbins[1], nbins[0]): 
				[X, Y, FES] = FFT_intg_2D(Ftot_x_tot, Ftot_y_tot, min_grid=min_grid, max_grid=max_grid)
				AD=abs(ref_fes - FES) * cutoff
				AAD = np.sum(AD)/(np.count_nonzero(cutoff))		
				aad_history.append(AAD)
				#aadMAP.append(AD)
				
			#print progress
			print_progress(i+1,total_number_of_hills,variable_name='Average Mean Force Error',variable=round(ofe_history[-1],3))        
			# if len(window_corners) == 4: print("    ||    Error in window", ofe_history_window[-1])		

			
	if len(window_corners) == 4: return [X, Y, Ftot_den, Ftot_x, Ftot_y, ofv, ofe, cutoff, volume_history, ofe_history, ofe_history_window, time_history, Ftot_den2, ofv_num_x, ofv_num_y]
  
	else: return [X, Y, Ftot_den, Ftot_x, Ftot_y, ofv, ofe, cutoff, volume_history, ofe_history, aad_history, time_history, Ftot_den2, ofv_num_x, ofv_num_y] #aadMAP, ofeMAP]
    