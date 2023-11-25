from __future__ import print_function
#from numba import jit 
#from numba import njit
import numpy as np
from MFI import * 

def MFI_2D(HILLS="HILLS", position_x="position_x", position_y="position_y", bw=np.array((0.1,0.1)), kT=1,
			min_grid=np.array((-np.pi, -np.pi)), max_grid=np.array((np.pi, np.pi)), nbins=np.array((200, 200)),
			error_pace=-1, base_terms = 0, window_corners=[], WellTempered=1, nhills=-1, periodic = np.array((0,0)), 
			Ftot_den_limit = 1E-10, FES_cutoff = -1, Ftot_den_cutoff = 0.1, non_exploration_penalty = 0, use_weighted_st_dev = True,
			hp_centre_x=0.0, hp_centre_y=0.0, hp_kappa_x=0, hp_kappa_y=0,
			lw_centre_x=0.0, lw_centre_y=0.0, lw_kappa_x=0, lw_kappa_y=0,
			uw_centre_x=0.0, uw_centre_y=0.0, uw_kappa_x=0, uw_kappa_y=0, 
			F_static_x = np.zeros((1,1)), F_static_y = np.zeros((1,1)),
	        compare_to_reference_FES = 0, ref_fes = np.zeros((200,200))):
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

	print(np.shape(X))

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
			ofeMAP.append(ofe)

			#Calculate averaged global error
			absolute_explored_volume = np.count_nonzero(cutoff)
			volume_history.append(absolute_explored_volume/absolute_explored_volume.size)
			if non_exploration_penalty <= 0: ofe_history.append( np.sum(ofe) / absolute_explored_volume)
			else: ofe_history.append( np.sum(ofe) / (nbins[0]*nbins[1]))
			time_history.append(HILLS[i,0] + HILLS[2,0] - HILLS[1,0])
			if len(window_corners) == 4:
				ofe_cut_window = reduce_to_window(ofe, min_grid, grid_space, x_min=window_corners[0], x_max=window_corners[1], y_min=window_corners[2], y_max=window_corners[3]) 
				ofe_history_window.append(np.sum(ofe_cut_window) / (np.count_nonzero(ofe_cut_window)))

         	#Find Absolute devaition
			if compare_to_reference_FES == 1:
				[X, Y, FES] = FFT_intg_2D(Ftot_x_tot, Ftot_y_tot, min_grid=min_grid, max_grid=max_grid)
				AD=abs(ref_fes - FES) * cutoff
				AAD = np.sum(AD)/(np.count_nonzero(cutoff))		
				aad_history.append([AAD])
				aadMAP.append(AD)
				
			#print progress
			print_progress(i+1,total_number_of_hills,variable_name='Average Mean Force Error',variable=round(ofe_history[-1],3))        
			# if len(window_corners) == 4: print("    ||    Error in window", ofe_history_window[-1])		

			
	if len(window_corners) == 4: return [X, Y, Ftot_den, Ftot_x, Ftot_y, ofv, ofe, cutoff, volume_history, ofe_history, ofe_history_window, time_history, Ftot_den2, ofv_num_x, ofv_num_y]
  
	else: return [X, Y, Ftot_den, Ftot_x, Ftot_y, ofv, ofe, cutoff, volume_history, ofe_history, aad_history, time_history, Ftot_den2, ofv_num_x, ofv_num_y, aadMAP, ofeMAP]
    