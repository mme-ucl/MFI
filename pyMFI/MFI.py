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
		np.array: Array with hills data
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
		list: 2 * np.array with position data of each dimension ([position_x, position_y])
	"""
	for file1 in glob.glob(position_name):
		colvar = np.loadtxt(file1)
		position_x = colvar[:-1, 1]
		position_y = colvar[:-1, 2]
	return [position_x, position_y]

def find_periodic_point(x_coord, y_coord, min_grid, max_grid, periodic):
	"""Finds periodic copies of input coordinates. 
	
	Args:
		x_coord (float): CV1-coordinate
		y_coord (float): CV2-coordinate
		min_grid (list): list of CV1-minimum value of grid and CV2-minimum value of grid
		max_grid (list): list of CV1-maximum value of grid and CV2-maximum value of grid
		periodic (binary): information if system is periodic. value of 0 corresponds to non-periodic system; function will only return input coordinates. Value of 1 corresponds to periodic system; function will return input coordinates with periodic copies.
	Returns:
		list: list of [x-coord, y-coord] pairs
	"""

	coord_list = []
	coord_list.append([x_coord, y_coord])
	
	if periodic == 1:
		# Use periodic extension for defining PBC
		periodic_extension = periodic * 1 / 2
		grid_ext = (1 / 2) * periodic_extension * (max_grid - min_grid)

		# There are potentially 4 points, 1 original and 3 periodic copies, or less.

		copy_record = [0, 0, 0, 0]
		# check for x-copy
		if x_coord < min_grid[0] + grid_ext[0]:
			coord_list.append([x_coord + 2 * np.pi, y_coord])
			copy_record[0] = 1
		elif x_coord > max_grid[0] - grid_ext[0]:
			coord_list.append([x_coord - 2 * np.pi, y_coord])
			copy_record[1] = 1
		# check for y-copy
		if y_coord < min_grid[1] + grid_ext[1]:
			coord_list.append([x_coord, y_coord + 2 * np.pi])
			copy_record[2] = 1
		elif y_coord > max_grid[1] - grid_ext[1]:
			coord_list.append([x_coord, y_coord - 2 * np.pi])
			copy_record[3] = 1
		# check for xy-copy
		if sum(copy_record) == 2:
			if copy_record[0] == 1 and copy_record[2] == 1:
				coord_list.append([x_coord + 2 * np.pi, y_coord + 2 * np.pi])
			elif copy_record[1] == 1 and copy_record[2] == 1:
				coord_list.append([x_coord - 2 * np.pi, y_coord + 2 * np.pi])
			elif copy_record[0] == 1 and copy_record[3] == 1:
				coord_list.append([x_coord + 2 * np.pi, y_coord - 2 * np.pi])
			elif copy_record[1] == 1 and copy_record[3] == 1:
				coord_list.append([x_coord - 2 * np.pi, y_coord - 2 * np.pi])        

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
		periodic (binary): information if system is periodic. value of 0 corresponds to non-periodic system. Value of 1 corresponds to periodic system.

	Returns:
		list: CV1 harmonic force array and CV2 harmonic force array.
	"""
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

	return [F_harmonic_x, F_harmonic_y]


def find_lw_force(lw_centre_x, lw_centre_y, lw_kappa_x, lw_kappa_y, X , Y, min_grid, max_grid, grid_space, periodic):
	"""Find 2D lower wall force.

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
		periodic (binary): information if system is periodic. value of 0 corresponds to non-periodic system. Value of 1 corresponds to periodic system.

	Returns:
		list: CV1 lower wall force array and CV2 lower wall force array
	"""
	#Calculate x-force
	F_wall_x = np.where(X < lw_centre_x, 2 * lw_kappa_x * (X - lw_centre_x), 0)
	if periodic == 1:
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
	if periodic == 1:
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
	"""Find 2D upper wall force.

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
		periodic (binary): information if system is periodic. value of 0 corresponds to non-periodic system. Value of 1 corresponds to periodic system.

	Returns:
		[F_wall_x, F_wall_y] - list: CV1 upper wall force array and CV2 upper wall force array
	"""

	#Calculate x-force
	F_wall_x = np.where(X > uw_centre_x, 2 * uw_kappa_x * (X - uw_centre_x), 0)
	if periodic == 1:
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
	if periodic == 1:
		if uw_centre_y < grid_centre:
			index_period = index(uw_centre_y + grid_length/2, min_grid[1], grid_space)
			F_wall_y[index_period:, :] = 0
		elif uw_centre_y > grid_centre:
			index_period = index(uw_centre_y - grid_length/2, min_grid[1], grid_space)
			F_wall_y[:index_period, :] = 2 * uw_kappa_y * (Y[:index_period, :] - uw_centre_y + grid_length)
	return [F_wall_x, F_wall_y]



### Main Mean Force Integration

def MFI_2D(HILLS="HILLS", position_x="position_x", position_y="position_y", bw=1, kT=1,
			min_grid=np.array((-np.pi, -np.pi)), max_grid=np.array((np.pi, np.pi)), nbins=np.array((200, 200)),
			log_pace=10, error_pace=1, base_terms = 0, window_corners=[], WellTempered=1, nhills=-1, periodic=0, 
			FES_cutoff = -1, FFT_integration = 0, Ftot_den_limit = 1E-10, Ftot_den_cutoff = 0.1,
			hp_centre_x=0.0, hp_centre_y=0.0, hp_kappa_x=0, hp_kappa_y=0,
			lw_centre_x=0.0, lw_centre_y=0.0, lw_kappa_x=0, lw_kappa_y=0,
			uw_centre_x=0.0, uw_centre_y=0.0, uw_kappa_x=0, uw_kappa_y=0):
	"""Compute a time-independent estimate of the Mean Thermodynamic Force, i.e. the free energy gradient in 2D CV spaces.

	Args:
		HILLS (str): HILLS array. Defaults to "HILLS".
		position_x (str): CV1 array. Defaults to "position_x".
		position_y (str): CV2 array. Defaults to "position_y".
		bw (int, optional): Scalar, bandwidth for the construction of the KDE estimate of the biased probability density. Defaults to 1.
		kT (int, optional): Scalar, kT. Defaults to 1.
		min_grid (array, optional): Lower bound of the force domain. Defaults to np.array((-np.pi, -np.pi)).
		max_grid (array, optional): Upper bound of the force domain. Defaults to np.array((np.pi, np.pi)).
		nbins (array, optional): number of bins in CV1,CV2. Defaults to np.array((200,200)).
		log_pace (int, optional): Progress and convergence are outputted every log_pace steps. Defaults to 10.
		error_pace (int, optional): Pace for the calculation of the on-the-fly measure of global convergence. Defaults to 1, change it to a higher value if FES_cutoff>0 is used. 
		base_terms (int or list, optional): When set to 0, inactive. When activated, "on the fly" variance is calculated as a patch to base (previous) simulation. To activate, put force terms of base simulation ([Ftot_den, Ftot_den2, Ftot_x, Ftot_y, ofv_x, ofv_y]). Defaults to 0.
		window_corners (list, optional): When set to [], inactive. When activated, error is ALSO calculated for mean force in the window. To activate, put the min and max values of the window ([min_x, max_x, min_y, max_y]). Defaults to [].
		WellTempered (binary, optional): Is the simulation well tempered? . Defaults to 1.
		nhills (int, optional): Number of HILLS to analyse, -1 for the entire HILLS array. Defaults to -1, i.e. the entire dataset.
		periodic (int, optional): Is the CV space periodic? 1 for yes. Defaults to 0.
		FES_cutoff (float, optional): Cutoff applied to FES and error calculation for FES values over the FES_cutoff. Defaults to -1. When FES_cutoff = 0, no cufoff is applied. Use with care, computing the fes in the loop renders the calculation extremely slow.
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

	Returns:
		X: array of size (nbins[0], nbins[1]) - CV1 grid positions
		Y: array of size (nbins[0], nbins[1]) - CV2 grid positions
		Ftot_den: array of size (nbins[0], nbins[1]) - Cumulative biased probability density, equivalent to an unbiased histogram of samples in CV space.
		Ftot_x:  array of size (nbins[0], nbins[1]) - CV1 component of the Mean Force.
		Ftot_y:  array of size (nbins[0], nbins[1]) - CV2 component of the Mean Force.
		ofe:  array of size (nbins[0], nbins[1]) - on the fly estimate of the local convergence
		ofe_history: array of size (1, total_number_of_hills) - running estimate of the global convergence of the mean force.
		(option with window corner activated: ofe_history_window: array of size (1, total_number_of_hills) - running estimate of the "window" convergence of the mean force.)
		ofe_history_time: array of size (1, total_number_of_hills) - time array of ofe_history
		Ftot_den2: array of size (nbins[0], nbins[1]) - Cumulative squared biased probability density
		ofv_x: array of size (nbins[0], nbins[1]) - intermediate component in the calculation of the CV1 "on the fly variance" ( sum of: pb_t * dfds_x ** 2)
		ofv_y: array of size (nbins[0], nbins[1]) - intermediate component in the calculation of the CV2 "on the fly variance" ( sum of: pb_t * dfds_y ** 2)
	"""

	if FES_cutoff > 0 and FFT_integration == 0: print("I will integrate the FES every ",str(error_pace)," steps. This may take a while." )

	gridx = np.linspace(min_grid[0], max_grid[0], nbins[0])
	gridy = np.linspace(min_grid[1], max_grid[1], nbins[1])
	grid_space = np.array(((max_grid[0] - min_grid[0]) / (nbins[0]-1), (max_grid[1] - min_grid[1]) / (nbins[1]-1)))
	X, Y = np.meshgrid(gridx, gridy)
	stride = int(len(position_x) / len(HILLS))
	const = (1 / (bw * np.sqrt(2 * np.pi) * stride))

	# Optional - analyse only nhills, if nhills is set
	if nhills > 0:
		total_number_of_hills = nhills
	else:
		total_number_of_hills = len(HILLS)
	bw2 = bw ** 2

	# Initialize force terms
	Fbias_x = np.zeros(nbins)
	Fbias_y = np.zeros(nbins)
	Ftot_num_x = np.zeros(nbins)
	Ftot_num_y = np.zeros(nbins)
	Ftot_den = np.zeros(nbins)
	Ftot_den2 = np.zeros(nbins)
	cutoff=np.zeros(nbins)
	ofv_num_x = np.zeros(nbins)
	ofv_num_y = np.zeros(nbins)
	volume_history = []
	ofe_history = []
	time_history = []
	if len(window_corners) == 4: ofe_history_window = []

	#Calculate static force
	F_static_x = np.zeros(nbins)
	F_static_y = np.zeros(nbins)
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

	# print("Total no. of Gaussians analysed: " + str(total_number_of_hills))

	# Definition Gamma Factor, allows to switch between WT and regular MetaD
	if WellTempered < 1:
		Gamma_Factor = 1
	else:
		gamma = HILLS[0, 6]
		Gamma_Factor = (gamma - 1) / (gamma)
		
	for i in range(total_number_of_hills):
		
		#Probability density limit, below which (fes or error) values aren't considered.
		# Ftot_den_limit = (i+1)*stride * 10**-5
		
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
			# kernelmeta = np.outer(kernelmeta_y, kernelmeta_x)
			Fbias_x += np.outer(kernelmeta_y, np.multiply(kernelmeta_x, (gridx - periodic_images[j][0])) / sigma_meta2_x )
			Fbias_y += np.outer(np.multiply(kernelmeta_y, (gridy - periodic_images[j][1])) / sigma_meta2_y, kernelmeta_x )

		# Estimate the biased proabability density p_t ^ b(s)
		pb_t = np.zeros(nbins)
		Fpbt_x = np.zeros(nbins)
		Fpbt_y = np.zeros(nbins)

		data_x = position_x[i * stride: (i + 1) * stride]
		data_y = position_y[i * stride: (i + 1) * stride]

		for j in range(stride):
			periodic_images = find_periodic_point(data_x[j], data_y[j], min_grid, max_grid, periodic)
			for k in range(len(periodic_images)):
				kernel_x = np.exp( - np.square(gridx - periodic_images[k][0]) / (2 * bw2)) * const #add constant here for less computations
				kernel_y = np.exp( - np.square(gridy - periodic_images[k][1]) / (2 * bw2))
				kernel = np.outer(kernel_y, kernel_x)
				kernel_x *= kT / bw2 #add constant here for less computations
    
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

		# calculate on the fly error components
		Ftot_den2 += np.square(pb_t)
		ofv_num_x += np.multiply(pb_t, np.square(dfds_x))
		ofv_num_y += np.multiply(pb_t, np.square(dfds_y))

		if (i + 1) % error_pace == 0 or (i+1) == total_number_of_hills:
			#calculate forces
			Ftot_x = np.divide(Ftot_num_x, Ftot_den, out=np.zeros_like(Fpbt_x), where=Ftot_den > 0)
			Ftot_y = np.divide(Ftot_num_y, Ftot_den, out=np.zeros_like(Fpbt_y), where=Ftot_den > 0)

			# calculate ofe (standard error)
			if base_terms == 0:
				[ofv, ofe] = mean_force_variance(Ftot_den, Ftot_den2, Ftot_x, Ftot_y, ofv_num_x, ofv_num_y)
				[Ftot_den_temp, Ftot_x_temp, Ftot_y_temp] = [np.array(Ftot_den), np.array(Ftot_x), np.array(Ftot_y)]
			elif len(base_terms) == 6:
				[Ftot_den_temp, Ftot_x_temp, Ftot_y_temp, ofv, ofe] = patch_to_base_variance(base_terms, [Ftot_den, Ftot_den2, Ftot_x, Ftot_y, ofv_num_x, ofv_num_y])

			# Define cutoff based on the biased probability density, unless FES_cut>0. In that case the cutoff is updated every error_pace steps. 
			#if there is a FES_cutoff, calculate fes ## Use with care, it costs a lot. 
			if FES_cutoff > 0: 
				if (i + 1) % int(error_pace) == 0 or (i+1) == total_number_of_hills:
					if periodic == 1 or FFT_integration == 1: [X, Y, FES] = FFT_intg_2D(Ftot_x_temp, Ftot_y_temp, min_grid=min_grid, max_grid=max_grid)
					else: [X, Y, FES] = intgrad2(Ftot_x_temp, Ftot_y_temp, min_grid=min_grid, max_grid=max_grid)
					cutoff = np.where(FES <= np.ones_like(FES) * FES_cutoff, 1, 0)
			else: cutoff = np.where(Ftot_den_temp >= np.ones_like(Ftot_den_temp) * Ftot_den_cutoff, 1, 0)
			
			ofe = np.multiply(ofe, cutoff) 

			#Calculate averaged global error
			absolute_explored_volume = np.count_nonzero(cutoff)
			volume_history.append( nbins[0]*nbins[1]/absolute_explored_volume)
			ofe_history.append( np.sum(ofe) / absolute_explored_volume)
			time_history.append(HILLS[i,0] + HILLS[2,0] - HILLS[1,0])
			if len(window_corners) == 4:
				ofe_cut_window = reduce_to_window(ofe, min_grid, grid_space, x_min=window_corners[0], x_max=window_corners[1], y_min=window_corners[2], y_max=window_corners[3]) 
				ofe_history_window.append(np.sum(ofe_cut_window) / (np.count_nonzero(ofe_cut_window)))

		print_progress(i,total_number_of_hills,variable_name='Average Mean Force Error',variable=ofe_history[-1])        
		#print progress
#		if (i + 1) % log_pace == 0:
#			print("|" + str(i + 1) + "/" + str(total_number_of_hills) + "|==> Average Mean Force Error: " + str(ofe_history[-1]), end='\x1b[1K\r')
#			if len(window_corners) == 4: print("    ||    Error in window", ofe_history_window[-1])
#			else: print("")
			
	if len(window_corners) == 4: return [X, Y, Ftot_den, Ftot_x, Ftot_y, ofv, ofe, cutoff, volume_history, ofe_history, ofe_history_window, time_history, Ftot_den2, ofv_num_x, ofv_num_y]
  
	else: return [X, Y, Ftot_den, Ftot_x, Ftot_y, ofv, ofe, cutoff, volume_history, ofe_history, time_history, Ftot_den2, ofv_num_x, ofv_num_y]


# @jit
def mean_force_variance(Ftot_den, Ftot_den2, Ftot_x, Ftot_y, ofv_num_x, ofv_num_y):
	"""Calculates the variance of the mean force

	Args:
		Ftot_den (array of size (nbins[0], nbins[1])): Cumulative biased probability density
		25Ftot_den2 (array of size (nbins[0], nbins[1])):  Cumulative squared biased probability density
		Ftot_x (array of size (nbins[0], nbins[1])): CV1 component of the Mean Force.
		Ftot_y (array of size (nbins[0], nbins[1])): CV2 component of the Mean Force.
		ofv_num_x (array of size (nbins[0], nbins[1])): intermediate component in the calculation of the CV1 "on the fly variance" ( sum of: pb_t * dfds_x ** 2)
		ofv_num_y (array of size (nbins[0], nbins[1])): intermediate component in the calculation of the CV2 "on the fly variance" ( sum of: pb_t * dfds_y ** 2)
		Ftot_den_limit (scalar): threshold in the cumulative biased probability density where data is discarded. Defaults to 0.
	Returns:
		var (array of size (nbins[0], nbins[1])): modulus of "on the fly variance" 
	"""    
	# calculate ofe (standard error)
	Ftot_den_sq = np.square(Ftot_den)
	Ftot_den_diff = Ftot_den_sq-Ftot_den2
	bessel_corr = np.divide(Ftot_den_sq , Ftot_den_diff, out=np.zeros_like(Ftot_den), where=Ftot_den_diff > 0)

	ofv_x = np.multiply(np.divide(ofv_num_x , Ftot_den, out=np.zeros_like(Ftot_den), where=Ftot_den > 0) - np.square(Ftot_x) , bessel_corr )
	ofv_y = np.multiply(np.divide(ofv_num_y , Ftot_den, out=np.zeros_like(Ftot_den), where=Ftot_den > 0) - np.square(Ftot_y) , bessel_corr )
	
	ofv = np.sqrt(np.square(ofv_x) + np.square(ofv_y))	
 
	ofe_x = np.sqrt(ofv_x)
	ofe_y = np.sqrt(ofv_y)
	ofe = np.sqrt(np.square(ofe_x) + np.square(ofe_y))

	return [ofv, ofe]


def patch_to_base_variance(master0, master):
	"""Patches force terms of a base simulation (alaysed prior to current simulation) with current simulation to return patched "on the fly variance".

	Args:
		master0 (list): Force terms of base simulation (alaysed prior to current simulation) [Ftot_den, Ftot_den2, Ftot_x, Ftot_y, ofv_num_x, ofv_num_y]
		master (list): Force terms of current simulation [Ftot_den, Ftot_den2, Ftot_x, Ftot_y, ofv_num_x, ofv_num_y]
		Ftot_den_limit (int): Truncates the probability density below Ftot_den_limit. Default set to 10**-10.

	Returns:
		OFV (array of size (nbins[0], nbins[1])): modulus of patched "on the fly variance" 
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

	FX_patch = np.divide(FX_patch, PD_patch, out=np.zeros_like(FX_patch), where=PD_patch > 0)
	FY_patch = np.divide(FY_patch, PD_patch, out=np.zeros_like(FY_patch), where=PD_patch > 0)
	#Ftot_patch.append([PD_patch, PD2_patch, FX_patch, FY_patch, OFV_num_X_patch, OFV_num_Y_patch])

	#Calculate variance of mean force
	PD_sq = np.square(PD_patch)
	PD_diff = PD_sq - PD2_patch
	bessel_corr = np.divide(PD_sq , PD_diff, out=np.zeros_like(PD_patch), where=PD_diff > 0)

	OFV_X = np.multiply(np.divide(OFV_num_X_patch, PD_patch, out=np.zeros_like(PD_patch), where=PD_patch > 0) - np.square(FX_patch) , bessel_corr )
	OFE_X = np.sqrt(OFV_X)
	
	OFV_Y = np.multiply(np.divide(OFV_num_Y_patch, PD_patch, out=np.zeros_like(PD_patch), where=PD_patch > 0) - np.square(FY_patch) , bessel_corr )
	OFE_Y = np.sqrt(OFV_Y)
	
	OFV = np.sqrt(np.square(OFV_X) + np.square(OFV_Y))
	OFE = np.sqrt(np.square(OFE_X) + np.square(OFE_Y))
	
	return [PD_patch, FX_patch, FY_patch, OFV, OFE]


### Integration using Fast Fourier Transform (FFT integration) in 2D
def FFT_intg_2D(FX, FY, min_grid=np.array((-np.pi, -np.pi)), max_grid=np.array((np.pi, np.pi)), nbins=0):
	"""2D integration of force gradient (FX, FY) to find FES using Fast Fourier Transform.

	Args:
		FX (array of size (nbins[0], nbins[1])): CV1 component of the Mean Force.
		FY (array of size (nbins[0], nbins[1])): CV1 component of the Mean Force.
		min_grid (array, optional): Lower bound of the simulation domain. Defaults to np.array((-np.pi, -np.pi)).
		min_grid (array, optional): Upper bound of the simulation domain. Defaults to np.array((np.pi, np.pi)).
		nbins (int, optional): number of bins in CV1,CV2. Defaults to 0. When nbins=0, nbins will take the shape of FX.

	Returns:
		X: array of size (nbins[0], nbins[1]) - CV1 grid positions
		Y: array of size (nbins[0], nbins[1]) - CV2 grid positions
		fes: array of size (nbins[0], nbins[1]) - Free Energy Surface
	"""
	if hasattr(nbins, "__len__") == False: nbins = np.shape(FX)        
	
	gridx = np.linspace(min_grid[0], max_grid[0], nbins[0])
	gridy = np.linspace(min_grid[1], max_grid[1], nbins[1])
	grid_spacex = (max_grid[0] - min_grid[0]) / (nbins[0] - 1)
	grid_spacey = (max_grid[1] - min_grid[1]) / (nbins[1] - 1)
	X, Y = np.meshgrid(gridx, gridy)

	# Calculate frequency
	freq_1dx = np.fft.fftfreq(nbins[0], grid_spacex)
	freq_1dy = np.fft.fftfreq(nbins[1], grid_spacey)
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
	fes = fes - np.min(fes)
	return [X, Y, fes]


# Equivalent to integration MS in Alanine dipeptide notebook.
def intg_2D(FX, FY, min_grid=np.array((-np.pi, -np.pi)), max_grid=np.array((np.pi, np.pi)), nbins=np.array((200, 200))):
	"""2D integration of force gradient (FX, FY) to find FES using finite difference method.
	
	Args:
		FX (array of size (nbins[0], nbins[1])): CV1 component of the Mean Force.
		FY (array of size (nbins[0], nbins[1])): CV2 component of the Mean Force.
		min_grid (array, optional): Lower bound of the simulation domain. Defaults to np.array((-np.pi, -np.pi)).
		min_grid (array, optional): Upper bound of the simulation domain. Defaults to np.array((np.pi, np.pi)).
		nbins (int, optional): number of bins in CV1,CV2. Defaults to np.array((200,200)).
	Returns:
		X: array of size (nbins[0], nbins[1]) - CV1 grid positions
		Y: array of size (nbins[0], nbins[1]) - CV2 grid positions
		fes: array of size (nbins[0], nbins[1]) - Free Energy Surface
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
def intgrad2(fx, fy, nx=0, ny=0, intconst=0, per1 = False, per2 = False, min_grid=np.array((-2, -2)), max_grid=np.array((2, 2)), nbins=0):
	
	"""This function uses the inverse of the gradient to reconstruct the free energy surface from the mean force components.
	[John D'Errico (2022). Inverse (integrated) gradient (https://www.mathworks.com/matlabcentral/fileexchange/9734-inverse-integrated-gradient), MATLAB Central File Exchange. Retrieved May 17, 2022.]
	[Translated from MatLab to Python by Francesco Serse (https://github.com/Fserse)]

	Args:
		fx (array): (ny by nx) array. X-gradient to be integrated.
		fy (array): (ny by nx) array. X-gradient to be integrated.
		nx (integer): nuber of datapoints in x-direction. Default to 0: will copy the shape of the input gradient.
		ny (integer): nuber of datapoints in y-direction. Default to 0: will copy the shape of the input gradient.
		intconst (float): Minimum value of output FES
		per1 (boolean): True if x-variable is periodic. False if non-periodic.
		per2 (boolean): True if y-variable is periodic. False if non-periodic
		min_grid (list/array of length=2): list/array of minimum value of [x-grid, y-grid]
		max_grid (list/array of length=2):  list/array of maximum value of [x-grid, y-grid]
		nbins (list/array of length=2): list/array of number of data pointis of [x-grid, y-grid]. Default to 0: will copy the shape of the input gradient.

	Returns:
		X (ny by nx array): X-component of meshgrid
		Y (ny by nx array): Y-component of meshgrid
		fhat (ny by nx array): integrated free energy surface
	"""

	if nx == 0: nx = np.shape(fx)[1]    
	if ny == 0: ny = np.shape(fx)[0]
	if hasattr(nbins, "__len__") == False: nbins = np.shape(fx) 
	
	gridx = np.linspace(min_grid[0], max_grid[0], nbins[0])
	gridy = np.linspace(min_grid[1], max_grid[1], nbins[1])
	dx = abs(gridx[1] - gridx[0])
	dy = abs(gridy[1] - gridy[0])
	X, Y = np.meshgrid(gridx, gridy)

	rhs = np.ravel((fx,fy))
	
	Af=np.zeros((4*nx*ny,3))
	
	n=0
	#Equations in x
	for i in range(0,ny):
		#Leading edge
		Af[2*nx*i][0] = 2*nx*i/2
		
		if(per2): 
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
		if(per2):
			Af[2*nx*(i+1)-1][1] = nx*i
		else:
			Af[2*nx*(i+1)-1][1] = nx*i+(nx-1)
		Af[2*nx*(i+1)-1][2] = 0.5/dx
	
	
	n=2*nx*ny
	
	#Equations in y
	#Leading edge
	for j in range(0,nx):
	
		Af[2*j+n][0] = 2*j/2 + n/2
		
		if(per1):
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
		if(per1):
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
	fhat = np.reshape(fhat,nbins) 
	#print(fhat.shape)   
	fhat = fhat - np.min(fhat)    

	return [X, Y, fhat]



def plot_recap_2D(X, Y, FES, TOTAL_DENSITY, CONVMAP, CONV_history, CONV_history_time, FES_lim=50, ofe_map_lim=50, FES_step=1, ofe_step=1):
	"""Plots 1. FES, 2. varinace_map, 3. Cumulative biased probability density, 4. Convergece of variance.
	
	Args:
		X: array of size (nbins[0], nbins[1]) - CV1 grid positions
		Y: array of size (nbins[0], nbins[1]) - CV2 grid positions
		FES: array of size (nbins[0], nbins[1]) - Free Energy Surface
		TOTAL_DENSITY: array of size (nbins[0], nbins[1]) - Cumulative biased probability density
		CONVMAP (array of size (nbins[0], nbins[1])): varinace_map
		CONV_history (list): Convergece of variance
	Returns: 
		(Plot)
	"""
	fig, axs = plt.subplots(1, 4, figsize=(16, 3))
	cp = axs[0].contourf(X, Y, FES, levels=range(0, FES_lim, FES_step), cmap='coolwarm', antialiased=False, alpha=0.8);
	cbar = plt.colorbar(cp, ax=axs[0])
	cbar.set_label("Free Energy [kJ/mol]")
	axs[0].set_ylabel('CV2 [nm]', fontsize=11)
	axs[0].set_xlabel('CV1 [nm]', fontsize=11)
	axs[0].set_xlim(np.min(X),np.max(X))
	axs[0].set_ylim(np.min(Y),np.max(Y))
	axs[0].set_title('Free Energy Surface', fontsize=11)

	cp = axs[1].contourf(X, Y, zero_to_nan(CONVMAP), levels=range(0, ofe_map_lim, ofe_step), cmap='coolwarm', antialiased=False, alpha=0.8);
	cbar = plt.colorbar(cp, ax=axs[1])
	cbar.set_label("Standard Deviation [kJ/(mol*nm)]")
	axs[1].set_ylabel('CV2 [nm]', fontsize=11)
	axs[1].set_xlabel('CV1 [nm]', fontsize=11)
	axs[1].set_xlim(np.min(X),np.max(X))
	axs[1].set_ylim(np.min(Y),np.max(Y))
	axs[1].set_title('Standard Deviation of the Mean Force', fontsize=11)

	cp = axs[2].contourf(X, Y, (TOTAL_DENSITY), cmap='gray_r', antialiased=False, alpha=0.8);  #, locator=ticker.LogLocator()
	cbar = plt.colorbar(cp, ax=axs[2])
	cbar.set_label("Relative count [-]")
	axs[2].set_ylabel('CV2 [nm]', fontsize=11)
	axs[2].set_xlabel('CV1 [nm]', fontsize=11)
	axs[2].set_xlim(np.min(X),np.max(X))
	axs[2].set_ylim(np.min(Y),np.max(Y))
	axs[2].set_title('Total Biased Probability Density', fontsize=11)

	axs[3].plot( [time/1000 for time in CONV_history_time], CONV_history, label="global ofe");
	axs[3].set_ylabel('Standard Deviation [kJ/(mol*nm)]', fontsize=11)
	axs[3].set_xlabel('Simulation time [ns]', fontsize=11)
	axs[3].set_title('Global Convergence of Standard Deviation', fontsize=11)
 
	plt.tight_layout()


# Patch independent simulations
def patch_2D(master_array, nbins=np.array((200, 200))):
	"""Takes in a collection of force terms and patches them togehter to return the patched force terms

	Args:
		master_array (list): collection of force terms (n * [Ftot_den, Ftot_den2, Ftot_x, Ftot_y, ofv_x, ofv_y])
		nbins (array, optional): number of bins in CV1,CV2. Defaults to np.array((200,200)).

	Returns:
		Patched force terms (list) -> ([Ftot_den, Ftot_den2, Ftot_x, Ftot_y, ofv_x, ofv_y])
	"""
	FP = np.zeros(nbins)
	FP2 = np.zeros(nbins)
	FX = np.zeros(nbins)
	FY = np.zeros(nbins)
	OFV_X = np.zeros(nbins)
	OFV_Y = np.zeros(nbins)

	for i in range(len(master_array)):
		FP += master_array[i][0]
		FP2 += master_array[i][1]
		FX += master_array[i][0] * master_array[i][2]
		FY += master_array[i][0] * master_array[i][3]
		OFV_X += master_array[i][4]
		OFV_Y += master_array[i][5]

	FX = np.divide(FX, FP, out=np.zeros_like(FX), where=FP != 0)
	FY = np.divide(FY, FP, out=np.zeros_like(FY), where=FP != 0)

	# #Calculate variance of mean force
	# PD_ratio = np.divide(PD2, (PD ** 2 - PD2), out=np.zeros_like(PD), where=(PD ** 2 - PD2) != 0)
	# OFE_X = np.divide(OFV_X, PD, out=np.zeros_like(OFV_X), where=PD > 1E-100) - FX ** 2
	# OFE_Y = np.divide(OFV_Y, PD, out=np.zeros_like(OFV_Y), where=PD > 1E-100) - FY ** 2
	# OFE_X = OFE_X * PD_ratio
	# OFE_Y = OFE_Y * PD_ratio
	# OFE = np.sqrt( abs(OFE_X) + abs(OFE_Y))

	return [FP, FP2, FX, FY, OFV_X, OFV_Y]

# Patch independent simulations
def patch_2D_simple(master_array, nbins=np.array((200, 200))):
	"""Takes in a collection of force and patches only the probability density and mean forces

	Args:
		master_array (list): collection of force terms (n * [X, Y, Ftot_den, Ftot_x, Ftot_y])
		nbins (array, optional): number of bins in CV1,CV2. Defaults to np.array((200,200)).

	Returns:
		Patched probability density and mean forces (list) -> ([Ftot_den, Ftot_x, Ftot_y])
	"""
	FP = np.zeros(nbins)
	FX = np.zeros(nbins)
	FY = np.zeros(nbins)


	for i in range(len(master_array)):
		FP += master_array[i][2]
		FX += master_array[i][2] * master_array[i][3]
		FY += master_array[i][2] * master_array[i][4]

	FX = np.divide(FX, FP, out=np.zeros_like(FX), where=FP != 0)
	FY = np.divide(FY, FP, out=np.zeros_like(FY), where=FP != 0)

	return [FP, FX, FY]


def plot_patch_2D(X, Y, FES, TOTAL_DENSITY, lim=50):
	"""Plots 1. FES, 2. Cumulative biased probability density
	
	Args:
		X: array of size (nbins[0], nbins[1]) - CV1 grid positions
		Y: array of size (nbins[0], nbins[1]) - CV2 grid positions
		FES: array of size (nbins[0], nbins[1]) - Free Energy Surface
		TOTAL_DENSITY: array of size (nbins[0], nbins[1]) - Cumulative biased probability density
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


def bootstrap_2D(X, Y, forces_all, n_bootstrap, FES_cutoff = 0, FFT_integration=0, min_grid=np.array((-3, -3)), max_grid=np.array((3, 3))):
	"""Algorithm to determine bootstrap error

	Args:
		X: array of size (nbins[0], nbins[1]) - CV1 grid positions
		Y: array of size (nbins[0], nbins[1]) - CV2 grid positions
		forces_all (list): collection of force terms (n * [Ftot_den, Ftot_x, Ftot_y])
		n_bootstrap (int): bootstrap iterations

	Returns:
		[FES_avr, var_fes, sd_fes, variance_prog, stdev_prog, var_fes_prog, sd_fes_prog ]
	"""
   
	# #Define terms that will be updated iteratively
	# Ftot_x_inter = np.zeros(np.shape(X))
	# Ftot_y_inter = np.zeros(np.shape(X))
	# Ftot_x_sum = np.zeros(np.shape(X))
	# Ftot_y_sum = np.zeros(np.shape(X))
	# Ftot_den_sum = np.zeros(np.shape(X))
	# Ftot_den2_sum = np.zeros(np.shape(X))
	FES_sum = np.zeros(np.shape(X))
	# FES2_sum = np.zeros(np.shape(X))

	# #store var and sd progression here
	# variance_prog = []
	# stdev_prog = []
	var_fes_prog  = []
	sd_fes_prog = []

	#Save patch force terms and FES
	FES_collection = []

	#Patch forces
	[Ftot_den, Ftot_x, Ftot_y] = patch_2D_simple(forces_all)

	#save non-random probability density
	Ftot_den_base = np.array(Ftot_den)


	for iteration in range(n_bootstrap):

		#Randomly choose forces
		force_rand_select = []    
		for i in range(len(forces_all)):
			force_rand_select.append(forces_all[random.randint(0,len(forces_all)-1)])  
				
		#patch forces to find average Ftot_den, Ftot and FES
		[Ftot_den, Ftot_x, Ftot_y] = patch_2D_simple(force_rand_select)
		
		#Calculate FES. if there is a FES_cutoff, find cutoff. 
		if FFT_integration == 1: [X, Y, FES] = FFT_intg_2D(Ftot_x, Ftot_y, min_grid=min_grid, max_grid=max_grid)
		else: [X, Y, FES] = intgrad2(Ftot_x, Ftot_y, min_grid=min_grid, max_grid=max_grid)



		#Save terms
		FES_collection.append(FES)

		# #calculate sums for variance
		# Ftot_x_inter += Ftot_den * Ftot_x**2
		# Ftot_y_inter += Ftot_den * Ftot_y**2
		# Ftot_x_sum += Ftot_x
		# Ftot_y_sum += Ftot_y
		# Ftot_den_sum += Ftot_den
		# Ftot_den2_sum += Ftot_den**2
		
		FES_sum += FES
		# FES2_sum += FES**2

		if iteration > 0:
			
			# #calculate force variance
			# Ftot_x_avr = Ftot_x_sum / (iteration+1)
			# Ftot_y_avr = Ftot_y_sum / (iteration+1)
			# Ftot2_x_weighted = np.divide(Ftot_x_inter, Ftot_den_sum, out=np.zeros_like(Ftot_x_inter), where=Ftot_den_base>10)
			# Ftot2_y_weighted = np.divide(Ftot_y_inter, Ftot_den_sum, out=np.zeros_like(Ftot_y_inter), where=Ftot_den_base>10)
			
			# Ftot_den_ratio = np.divide(Ftot_den_sum ** 2, (Ftot_den_sum ** 2 - Ftot_den2_sum), out=np.zeros_like(Ftot_den_sum), where=Ftot_den_base > 10)
			
			# variance_x = (Ftot2_x_weighted - Ftot_x_avr**2) * Ftot_den_ratio
			# variance_y = (Ftot2_y_weighted - Ftot_y_avr**2) * Ftot_den_ratio
						
			# stdev_x = np.sqrt(variance_x)
			# stdev_y = np.sqrt(variance_y)
			
			# stdev = np.sqrt(stdev_x**2 + stdev_y**2)
		
			#calculate FES variance
			FES_avr = FES_sum/ (iteration+1)
			
			# if FES_cutoff > 0: cutoff = np.where(FES <= np.ones_like(FES) * FES_cutoff, 1, 0)
			# else: cutoff = np.ones_like(Ftot_den)
			
			cutoff = np.where(Ftot_den >= np.ones_like(FES) * 0.1, 1, 0)
			
			var_fes = np.zeros(np.shape(X))
			for i in range(len(FES_collection)): 
				var_fes += (FES_collection[i] - FES_avr)**2
			var_fes = (1/(len(FES_collection)-1)) * var_fes
			sd_fes = np.sqrt(var_fes)
						
			#save variance
			# stdev_prog.append(sum(sum(stdev*cutoff))/(np.count_nonzero(stdev*cutoff)))
			var_fes_prog.append(sum(sum(var_fes*cutoff))/(np.count_nonzero(var_fes*cutoff))) 
			sd_fes_prog.append(sum(sum(sd_fes*cutoff))/(np.count_nonzero(sd_fes*cutoff)))
		
		
		#print progress
		if (iteration+1) % (n_bootstrap/5) == 0:
			# print(iteration+1, "Ftot: sd=", round(stdev_prog[-1],5), "      FES: var=", round(var_fes_prog[-1],3), "     sd=", round(sd_fes_prog[-1],3) )
			print(iteration+1, "FES: var=", round(var_fes_prog[-1],3), "     sd=", round(sd_fes_prog[-1],3) )
			
	# return [FES_avr, cutoff, var_fes, sd_fes, variance_prog, stdev_prog, var_fes_prog, sd_fes_prog ]
	return [FES_avr, cutoff, var_fes, sd_fes, var_fes_prog, sd_fes_prog ]


def bootstrap_2D_fes(X, Y, forces_all, n_bootstrap, FES_cutoff = 0, FFT_integration=0, min_grid=np.array((-3, -3)), max_grid=np.array((3, 3))):
	"""Algorithm to determine bootstrap error

	Args:
		X: array of size (nbins[0], nbins[1]) - CV1 grid positions
		Y: array of size (nbins[0], nbins[1]) - CV2 grid positions
		forces_all (list): collection of force terms (n * [Ftot_den, Ftot_x, Ftot_y])
		n_bootstrap (int): bootstrap iterations

	Returns:
		[FES_avr, var_fes, sd_fes, variance_prog, stdev_prog, var_fes_prog, sd_fes_prog ]
	"""
   
	# #Define terms that will be updated iteratively
	FES_sum = np.zeros(np.shape(X))
	Ftot_den_sum = np.zeros(np.shape(X))
	Ftot_den2_sum = np.zeros(np.shape(X))
	var_num = np.zeros(np.shape(X))

	#store var and sd progression here
	sd_fes_prog = []

	#Save patch force terms and FES
	FES_collection = []
	Ftot_den_collection = []

	#Patch forces
	[Ftot_den, Ftot_x, Ftot_y] = patch_2D_simple(forces_all)

	#cutoff all points that havent been visited
	cutoff = np.where(Ftot_den >= np.ones_like(Ftot_den) * 0.1, 1, 0)

	for iteration in range(n_bootstrap):

		#Randomly choose forces
		force_rand_select = []    
		for i in range(len(forces_all)):
			force_rand_select.append(forces_all[random.randint(0,len(forces_all)-1)])  
				
		#patch forces to find average Ftot_den, Ftot and FES
		[Ftot_den, Ftot_x, Ftot_y] = patch_2D_simple(force_rand_select)
		
		#Calculate FES. if there is a FES_cutoff, find cutoff. 
		if FFT_integration == 1: [X, Y, FES] = FFT_intg_2D(Ftot_x, Ftot_y, min_grid=min_grid, max_grid=max_grid)
		else: [X, Y, FES] = intgrad2(Ftot_x, Ftot_y, min_grid=min_grid, max_grid=max_grid)

		#Save FES
		FES_collection.append(FES) 
		Ftot_den_collection.append(Ftot_den)       
		FES_sum += FES
		# var_num += Ftot_den * FES**2
		Ftot_den_sum += Ftot_den
		Ftot_den2_sum += Ftot_den**2

		if iteration > 0:
					
			#calculate FES variance
			FES_avr = FES_sum/ (iteration+1)
									
			var_fes = np.zeros(np.shape(X))
			for i in range(len(FES_collection)): 
				var_fes += Ftot_den_collection[i] * (FES_collection[i] - FES_avr)**2
			var_fes = np.divide( var_fes , Ftot_den_sum, out=np.zeros_like(Ftot_den_sum), where=Ftot_den_sum > 10**-10)
			
			# var_fes = np.divide( var_num , Ftot_den_sum, out=np.zeros_like(Ftot_den_sum), where=Ftot_den_sum > 10**-10) - FES_avr**2
			
			bessel_corr = np.divide(Ftot_den_sum**2 , (Ftot_den_sum**2-Ftot_den2_sum), out=np.zeros_like(Ftot_den), where=(Ftot_den_sum**2-Ftot_den2_sum) > 0)
			
			var_fes = var_fes * bessel_corr
			sd_fes = np.sqrt(var_fes)
						
			#save variance
			sd_fes_prog.append(sum(sum(sd_fes*cutoff))/(np.count_nonzero(sd_fes*cutoff)))
		
		
		#print progress
		if (iteration+1) % (n_bootstrap/5) == 0:
			# print(iteration+1, "Ftot: sd=", round(stdev_prog[-1],5), "      FES: var=", round(var_fes_prog[-1],3), "     sd=", round(sd_fes_prog[-1],3) )
			print(iteration+1, "FES st. dev. =", round(sd_fes_prog[-1],3) )
			
	return [FES_avr, cutoff, sd_fes, sd_fes_prog ]

def plot_bootstrap(X, Y, FES, sd_fes, sd_fes_prog, FES_lim=11, ofe_lim=11, FES_step=1, ofe_step=1):
	"""Plots result of bootstrap analysis. 1. Average FES, 2. average varinace, 3. variance progression

	Args:
		X: array of size (nbins[0], nbins[1]) - CV1 grid positions
		Y: array of size (nbins[0], nbins[1]) - CV2 grid positions
		FES: array of size (nbins[0], nbins[1]) - Free Energy Surface
		var_fes (array of size (nbins[0], nbins[1])): _description_
		var_fes_prog (list): _description_
		FES_lim (int, optional): Upper energy limit of FES plot. Defaults to 11.
		ofe_map_lim (int, optional): Upper variance limit of variance plot. Defaults to 11.
		
	Returns:
		(Plot)
	"""
	
	fig, axs = plt.subplots(1, 3, figsize=(15, 4))
	cp = axs[0].contourf(X, Y, FES, levels=range(0, FES_lim, 1), cmap='coolwarm', antialiased=False, alpha=0.8);
	cbar = plt.colorbar(cp, ax=axs[0])
	axs[0].set_ylabel('CV2', fontsize=11)
	axs[0].set_xlabel('CV1', fontsize=11)
	axs[0].set_title('Average FES', fontsize=11)

	cp = axs[1].contourf(X, Y, sd_fes, levels=np.linspace(0, ofe_lim, 10), cmap='coolwarm', antialiased=False, alpha=0.8);
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
	output_array = np.zeros_like(input_array)
	for ii in range(len(input_array)):
		for jj in range(len(input_array[ii])):
			if input_array[ii][jj] <= 0: output_array[ii][jj] = np.nan
			else: output_array[ii][jj] = input_array[ii][jj]
	return output_array



def intg_FD8(FX,FY, min_grid=np.array((0, 0)), max_grid=np.array((3, 3)), nbins=np.array((201, 201))):

	if nbins[0] != np.shape(FX)[0] or nbins[1] != np.shape(FX)[1]:

		print("this part of the code not ready yet")
		pass

		# r = np.stack([X_old.ravel(), Y_old.ravel()]).T
		# Sx = interpolate.CloughTocher2DInterpolator(r, FX.ravel())
		# Sy = interpolate.CloughTocher2DInterpolator(r, FY.ravel())
		# Nx, Ny = i_bins

		# x_new = np.linspace(grid.min(), grid.max(), Nx)
		# y_new = np.linspace(grid.min(), grid.max(), Ny)
		# X_new, Y_new = np.meshgrid(x_new, y_new)

		# ri = np.stack([X_new.ravel(), Y_new.ravel()]).T
		# FX = Sx(ri).reshape(X_new.shape)
		# FY = Sy(ri).reshape(Y_new.shape)

		# grid_diff = np.diff(x_new)[0]

	else:
		gridx = np.linspace(min_grid[0], max_grid[0], nbins[0])
		gridy = np.linspace(min_grid[1], max_grid[1], nbins[1])
		X, Y = np.meshgrid(gridx, gridy) 


	SdZx = np.cumsum(FX, axis=1) * np.diff(gridx)[0]  # cumulative sum along x-axis
	SdZy = np.cumsum(FY, axis=0) * np.diff(gridy)[0]  # cumulative sum along y-axis
	SdZx3 = np.cumsum(FX[::-1], axis=1) * np.diff(gridx)[0]  # cumulative sum along x-axis
	SdZy3 = np.cumsum(FY[::-1], axis=0) * np.diff(gridy)[0]  # cumulative sum along y-axis
	SdZx5 = np.cumsum(FX[:, ::-1], axis=1) * np.diff(gridx)[0]  # cumulative sum along x-axis
	SdZy5 = np.cumsum(FY[:, ::-1], axis=0) * np.diff(gridy)[0]  # cumulative sum along y-axis
	SdZx7 = np.cumsum(FX[::-1, ::-1], axis=1) * np.diff(gridx)[0]  # cumulative sum along x-axis
	SdZy7 = np.cumsum(FY[::-1, ::-1], axis=0) * np.diff(gridy)[0]  # cumulative sum along y-axis


	FES = np.zeros(nbins)
	FES2 = np.zeros(nbins)
	FES3 = np.zeros(nbins)
	FES4 = np.zeros(nbins)
	FES5 = np.zeros(nbins)
	FES6 = np.zeros(nbins)
	FES7 = np.zeros(nbins)
	FES8 = np.zeros(nbins)

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

	return [X, Y, FES_a]
	
	


    
def coft(HILLS="HILLS",FES="FES",kT=1,WellTempered=-1,total_number_of_hills=100,stride=10,
			min_grid=np.array((-np.pi, -np.pi)), max_grid=np.array((np.pi, np.pi)), nbins=np.array((200, 200)),
			):
	"""Compute a time-independent estimate of the Mean Thermodynamic Force, i.e. the free energy gradient in 2D CV spaces.

	Args:
		HILLS (str): HILLS array. Defaults to "HILLS".
		position_x (str): CV1 array. Defaults to "position_x".
		position_y (str): CV2 array. Defaults to "position_y".
		bw (int, optional): Scalar, bandwidth for the construction of the KDE estimate of the biased probability density. Defaults to 1.
		min_grid (array, optional): Lower bound of the force domain. Defaults to np.array((-np.pi, -np.pi)).
		max_grid (array, optional): Upper bound of the force domain. Defaults to np.array((np.pi, np.pi)).
		nbins (array, optional): number of bins in CV1,CV2. Defaults to np.array((200,200)).
	"""

	gridx = np.linspace(min_grid[0], max_grid[0], nbins[0])
	gridy = np.linspace(min_grid[1], max_grid[1], nbins[1])
	grid_space = np.array(((max_grid[0] - min_grid[0]) / (nbins[0]-1), (max_grid[1] - min_grid[1]) / (nbins[1]-1)))
	X, Y = np.meshgrid(gridx, gridy)

	# Initialize force terms
	Bias = np.zeros(nbins)

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
    progress = (iteration / total)
    arrow = '*' * int(round(bar_length * progress))
    spaces = ' ' * (bar_length - len(arrow))
    print(f'\r|{arrow}{spaces}| {int(progress * 100)}% | {variable_name}: {variable}', end='', flush=True)