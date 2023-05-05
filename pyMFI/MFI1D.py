import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
from numba import jit, njit


def load_HILLS(hills_name="HILLS"):
    """Load 1-dimensional hills data (includes time, position_x, position_y, hills_parameters 

    Args:
        hills_name (str, optional): Name of hills file. Defaults to "HILLS".

    Returns:
        np.array: Array with hills data
    """
    for file in glob.glob(hills_name):
        hills = np.loadtxt(file)
        hills = hills[:-1]
        hills0 = np.array(hills[0])
        hills0[3] = 0
        hills = np.concatenate(([hills0], hills))
    return hills


# Load the trajectory (position) data
def load_position(position_name="position"):
    """Load 1-dimensional position/trajectory data.

    Args:
        position_name (str, optional): Name of position file. Defaults to "position".

    Returns:
        position (list):  np.array with position data
    """
    for file1 in glob.glob(position_name):
        colvar = np.loadtxt(file1)
    return colvar[:-1, 1]

#define indexing
@njit
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


def find_periodic_point(x_coord, min_grid, max_grid, periodic, grid_ext, grid_length):
    """Finds periodic copies of input coordinate. 

    Args:
        x_coord (float): CV-coordinate
        min_grid (float): minimum value of grid
        max_grid (float): maximum value of grid
        periodic (binary): information if system is periodic. value of 0 corresponds to non-periodic system; function will only return input coordinates. Value of 1 corresponds to periodic system; function will return input coordinates with periodic copies.


    Returns:
        list: list of input coord and possibly periodic copies
    """
    if periodic == 1:
        coord_list = []
        #There are potentially 2 points, 1 original and 1 periodic copy.
        coord_list.append(x_coord)
        #check for copy
        if x_coord < min_grid+grid_ext: coord_list.append(x_coord + grid_length)
        elif x_coord > max_grid-grid_ext: coord_list.append(x_coord - grid_length)

        return coord_list
    else:
        return [x_coord]
    
@jit
def find_periodic_point_numpy(coord_array, min_grid, max_grid, periodic, grid_ext, grid_length):

    len_coord_array = len(coord_array)

    if periodic == 0:
        return coord_array
    elif periodic == 1:
        for i in range(len_coord_array):
            if coord_array[i] < min_grid+grid_ext:
                coord_array = np.append(coord_array, coord_array[i] + grid_length)
            elif coord_array[i] > max_grid-grid_ext:
                coord_array = np.append(coord_array, coord_array[i] - grid_length)
        return coord_array  
    
@jit
def window_forces(periodic_positions, periodic_hills, grid, sigma_meta2, height_meta, kT, const, bw2, Ftot_den_limit):
    
    pb_t = np.zeros(len(grid))
    Fpbt = np.zeros(len(grid))
    Fbias_window = np.zeros(len(grid))
    
    for j in range(len(periodic_hills)):
        kernelmeta = np.exp( - np.square(grid - periodic_hills[j]) / (2*sigma_meta2) )
        Fbias_window += height_meta / sigma_meta2 * np.multiply(kernelmeta, (grid - periodic_hills[j]))
    
    # Estimate the biased proabability density
    for j in range(len(periodic_positions)):
        kernel = const * np.exp(- np.square(grid - periodic_positions[j]) / (2 * bw2))  
        pb_t += kernel
        Fpbt += kT / bw2 * np.multiply(kernel, (grid - periodic_positions[j]))

    pb_t = np.where(pb_t > Ftot_den_limit, pb_t, 0)  # truncated probability density of window
   
    return [pb_t, Fpbt, Fbias_window]  

@njit
def find_hp_force(hp_centre, hp_kappa, grid, min_grid, max_grid, grid_space, periodic):
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
    if periodic == 1:
        grid_length = max_grid - min_grid
        grid_centre = min_grid + grid_length/2
        if hp_centre < grid_centre:
            index_period = index(hp_centre + grid_length / 2, min_grid, grid_space)
            F_harmonic[index_period:] = hp_kappa * (grid[index_period:] - hp_centre - grid_length)
        elif hp_centre > grid_centre:
            index_period = index(hp_centre - grid_length / 2, min_grid, grid_space)
            F_harmonic[:index_period] = hp_kappa * (grid[:index_period] - hp_centre + grid_length)

    return F_harmonic

@njit
def find_lw_force(lw_centre, lw_kappa, grid, min_grid, max_grid, grid_space, periodic):
    """_summary_

    Args:
        lw_centre (float): position of lower wall potential
        lw_kappa (float): force_constant of lower wall potential
        grid (array): CV grid positions
        min_grid (float): minimum value of grid
        max_grid (float): maximum value of grid
        grid_space (float): space between two consecutive grid values
        periodic (binary): information if system is periodic. value of 0 corresponds to non-periodic system. Value of 1 corresponds to periodic system.

    Returns:
       array: lower wall force array
    """
    F_harmonic = np.where(grid < lw_centre, 2 * lw_kappa * (grid - lw_centre), 0)
    if periodic == 1:
        grid_length = max_grid - min_grid
        grid_centre = min_grid + grid_length/2
        if lw_centre < grid_centre:
            index_period = index(lw_centre + grid_length / 2, min_grid, grid_space)
            F_harmonic[index_period:] = 2 * lw_kappa * (grid[index_period:] - lw_centre - grid_length)
        elif lw_centre > grid_centre:
            index_period = index(lw_centre - grid_length / 2, min_grid, grid_space)
            F_harmonic[:index_period] = 0

    return F_harmonic

@njit
def find_uw_force(uw_centre, uw_kappa, grid, min_grid, max_grid, grid_space, periodic):
    """_summary_

    Args:
        uw_centre (float): position of upper wall potential
        uw_kappa (float): force_constant of upper wall potential
        grid (_type_): CV grid positions
        min_grid (float): minimum value of grid
        max_grid (float): maximum value of grid
        grid_space (float): space between two consecutive grid values
        periodic (binary): information if system is periodic. value of 0 corresponds to non-periodic system. Value of 1 corresponds to periodic system.

    Returns:
        array: upper wall force array
    """
    F_harmonic = np.where(grid > uw_centre, uw_kappa * (grid - uw_centre), 0)
    if periodic == 1:
        grid_length = max_grid - min_grid
        grid_centre = min_grid + grid_length/2
        if uw_centre < grid_centre:
            index_period = index(uw_centre + grid_length / 2, min_grid, grid_space)
            F_harmonic[index_period:] = 0
        elif uw_centre > grid_centre:
            index_period = index(uw_centre - grid_length / 2, min_grid, grid_space)
            F_harmonic[:index_period] = 2 * uw_kappa * (grid[:index_period] - uw_centre + grid_length)
    
    return F_harmonic

@njit
def intg_1D(Force, dx):
    """Integration of 1D gradient using finite difference method (simpson's method).

    Args:
        Force (array): Mean force
        dx (array): grid spacing

    Returns:
        FES (array): Free energy surface
    """
    fes = np.zeros_like(Force)
    
    for j in range(len(Force)): 
        y = Force[:j + 1]
        N = len(y)
        if N % 2 == 0: fes[j] = dx/6 * (np.sum(y[: N-3: 2] + 4*y[1: N-3+1: 2] + y[2: N-3+2: 2]) + np.sum(y[1: N-2: 2] + 4*y[1+1: N-1: 2] + y[1+2: N: 2])) + dx/4 * ( y[1] + y[0] + y[-1] + y[-2])
        else: fes[j] = dx / 3.0 * np.sum(y[: N-2: 2] + 4*y[1: N-1: 2] + y[2: N: 2])
            
    fes = fes - min(fes)
    return fes
   
### Algorithm to run 1D MFI
# Run MFI algorithm with on the fly error calculation
@njit
def MFI_1D(HILLS="HILLS", position="position", bw=0.1, kT=1, min_grid=-2, max_grid=2, nbins=201, log_pace=10,
           error_pace=200, WellTempered=1, nhills=-1, periodic=0, hp_centre=0.0, hp_kappa=0, lw_centre=0.0, lw_kappa=0,
           uw_centre=0.0, uw_kappa=0, intermediate_fes_number = 0, Ftot_den_limit = 1E-10, FES_cutoff = 0, Ftot_den_cutoff = 0, F_static = np.zeros(123)):
    """Compute a time-independent estimate of the Mean Thermodynamic Force, i.e. the free energy gradient in 1D CV spaces.

    Args:
        HILLS (str): HILLS array. Defaults to "HILLS".
        position (str): CV/position array. Defaults to "position".
        bw (float, optional): bandwidth for the construction of the KDE estimate of the biased probability density. Defaults to 1.
        kT (float, optional): kT. Defaults to 1.
        min_grid (int, optional): Lower bound of the force domain. Defaults to -2.
        max_grid (int, optional): Upper bound of the force domain. Defaults to 2.
        nbins (int, optional): number of bins in grid. Defaults to 101.
        log_pace (int, optional): Pace for outputting progress and convergence. Defaults to 10.
        error_pace (int, optional): Pace for the cutoffcalculation of the on-the-fly measure of global convergence. Defaults to 200.
        WellTempered (binary, optional): Is the simulation well tempered?. Defaults to 1.
        periodic (int, optional): Is the CV space periodic? 1 for yes. Defaults to 0.
        hp_centre (float, optional): position of harmonic potential. Defaults to 0.0.
        hp_kappa (int, optional): force_constant of harmonic potential. Defaults to 0.
        lw_centre (float, optional): position of lower wall potential. Defaults to 0.0.
        lw_kappa (int, optional): force_constant of lower wall potential. Defaults to 0.
        uw_centre (float, optional): position of upper wall potential. Defaults to 0.0.
        uw_kappa (int, optional): force_constant of upper wall potential. Defaults to 0.
        FES_cutoff (float, optional): Cutoff applied to FES and error calculation for FES values over the FES_cutoff. Defaults to 0. When FES_cutoff = 0, no cufoff is applied.

    Returns:
        grid (array of size (1, nbins)): CV-array
        Ftot_den (array of size (1, nbins)): Cumulative biased probability density
        Ftot (array of size (1, nbins)): Mean Force
        ofe (array of size (1, nbins)): on the fly estimate of the variance of the mean force
        ofe_history (list of size (1, error_pace)): running estimate of the global on the fly variance of the mean force
    """
    
    grid = np.linspace(min_grid, max_grid, nbins)
    grid_space = (max_grid - min_grid) / (nbins-1)
    grid_ext = 0.25 * (max_grid-min_grid)
    grid_length = max_grid - min_grid
    stride = int(len(position) / len(HILLS[:, 1]))
    const = (1 / (bw * np.sqrt(2 * np.pi) * stride))
    bw2 = bw ** 2
    if nhills > 0: total_number_of_hills = nhills
    else: total_number_of_hills = len(HILLS)
    
    # initialise force terms
    Fbias = np.zeros(len(grid))
    Ftot_num = np.zeros(len(grid))
    Ftot_den = np.zeros(len(grid))
    Ftot_den2 = np.zeros(len(grid))
    ofv_num = np.zeros(len(grid))
    
    #initialise error terms
    error_history_collection = np.zeros((int(error_pace), 3))
    error_count = 0
    volume_history = np.zeros(int(error_pace))
    
    intermediate_fes_collection = np.zeros((int(intermediate_fes_number), nbins))
    intermediate_cutoff_collection = np.zeros((int(intermediate_fes_number), nbins))
    intermediate_time_collection = np.zeros(int(intermediate_fes_number))
    intermediate_fes_count = 0
        
    #Calculate static force (form harmonic or wall potential)
    if len(F_static) != nbins: F_static = np.zeros(nbins)
    if hp_kappa > 0: F_static += find_hp_force(hp_centre, hp_kappa, grid, min_grid, max_grid, grid_space, periodic)
    if lw_kappa > 0: F_static += find_lw_force(lw_centre, lw_kappa, grid, min_grid, max_grid, grid_space, periodic)
    if uw_kappa > 0: F_static += find_uw_force(uw_centre, uw_kappa, grid, min_grid, max_grid, grid_space, periodic)


    # Definition Gamma Factor, allows to switch between WT and regular MetaD
    if WellTempered < 1: Gamma_Factor = 1
    else: Gamma_Factor = (HILLS[0, 4] - 1) / (HILLS[0, 4])
        
    #Cycle over windows of constant bias (for each deposition of a gaussian bias)
    for i in range(total_number_of_hills):
        
        #Get position data of window        
        s = HILLS[i, 1]  # centre position of Gaussian
        sigma_meta2 = HILLS[i, 2] ** 2  # width of Gaussian
        height_meta = HILLS[i, 3] * Gamma_Factor  # Height of Gaussian
        data = position[i * stride: (i + 1) * stride]  # positons of window of constant bias force.
        periodic_hills = find_periodic_point_numpy(np.array([s]), min_grid, max_grid, periodic, grid_ext, grid_length)
        periodic_positions = find_periodic_point_numpy(data, min_grid, max_grid, periodic, grid_ext, grid_length)
                
        # Find forces of window
        [pb_t, Fpbt, Fbias_window] = window_forces(periodic_positions, periodic_hills, grid, sigma_meta2, height_meta, kT, const, bw2, Ftot_den_limit)

        # update force terms
        Ftot_den += pb_t  # total probability density             
        Fbias += Fbias_window
        dfds = np.where(pb_t > Ftot_den_limit, Fpbt / pb_t, 0) + Fbias - F_static
        Ftot_num += np.multiply(pb_t, dfds)
        Ftot = np.where(Ftot_den > Ftot_den_limit, Ftot_num / Ftot_den, 0)

        # terms for error calculation
        Ftot_den2 += np.square(pb_t) 
        ofv_num += np.multiply(pb_t, np.square(dfds))  
  

        # Calculate error
        if (i + 1) % int(total_number_of_hills / error_pace) == 0 or (i+1) == total_number_of_hills:
            
    
            cutoff = np.ones(nbins, dtype=np.float64)
            if FES_cutoff > 0:
                FES = intg_1D(Ftot, grid_space)
                cutoff = np.where(FES < FES_cutoff, 1.0, 0)
            if Ftot_den_cutoff > 0:
                cutoff = np.where(Ftot_den > Ftot_den_cutoff, 1.0, 0)

            # print(np.shape(cutoff))
            ofv = np.where(Ftot_den > Ftot_den_limit, ofv_num / Ftot_den, 0) - np.square(Ftot)
            temp_diff = np.square(Ftot_den)-Ftot_den2
            ofv *= np.where(temp_diff > 0, np.square(Ftot_den) / temp_diff, 0)
            ofe = np.where(ofv != 0, np.sqrt(ofv), 0)  
        
            error_history_collection[error_count,0] = sum(ofv) / np.count_nonzero(ofv)
            error_history_collection[error_count,1] = sum(ofe) / np.count_nonzero(ofe)
            error_history_collection[error_count,2] = HILLS[i,0]
            
            absolute_explored_volume = np.count_nonzero(cutoff)
            # volume_history[error_count] = nbins / absolute_explored_volume

            error_count += 1
            
            #window error            
               
        # Print progress               
        if (i + 1) % int(total_number_of_hills / log_pace) == 0 or (i+1) == total_number_of_hills:
            print((round((i + 1) / total_number_of_hills * 100, 0)) , "%   OFE =", round(error_history_collection[error_count-1,1], 4))
                
        # #Calculate intermediate fes
        if intermediate_fes_number > 1:
            if (i+1) % (total_number_of_hills/intermediate_fes_number) == 0 or (i+1) == total_number_of_hills:
                
                FES = intg_1D(Ftot, grid_space)
                
                cutoff = np.ones(nbins, dtype=np.float64)
                if FES_cutoff > 0:
                    cutoff = np.where(FES < FES_cutoff, 1.0, 0)
                if Ftot_den_cutoff > 0:
                    cutoff = np.where(Ftot_den > Ftot_den_cutoff, 1.0, 0)
            
                intermediate_fes_collection[intermediate_fes_count] = FES
                intermediate_cutoff_collection[intermediate_fes_count] = cutoff
                intermediate_time_collection[intermediate_fes_count] = HILLS[i,0]
                intermediate_fes_count += 1

    FES = intg_1D(Ftot, grid_space)
    
    if intermediate_fes_number > 1: return grid, Ftot_den, Ftot_den2, Ftot, ofv_num, FES, ofv, ofe, cutoff, error_history_collection, volume_history, intermediate_fes_collection, intermediate_cutoff_collection, intermediate_time_collection
    else: return grid, Ftot_den, Ftot_den2, Ftot, ofv_num, FES, ofv, ofe, cutoff, error_history_collection, volume_history, intermediate_fes_collection, intermediate_cutoff_collection, intermediate_time_collection


### FAST Algorithm to run 1D MFI
@njit
def MFI_1D_fast(HILLS="HILLS", position="position", bw=1, kT=1, min_grid=-2, max_grid=2, nbins=201,
           WellTempered=1, nhills=-1, periodic=0, hp_centre=0.0, hp_kappa=0, lw_centre=0.0, lw_kappa=0,
           uw_centre=0.0, uw_kappa=0, Ftot_den_limit = 1E-10, F_static = np.zeros(123)):
    """Compute a time-independent estimate of the Mean Thermodynamic Force, i.e. the free energy gradient in 1D CV spaces.

    Args:
        HILLS (str): HILLS array. Defaults to "HILLS".
        position (str): CV/position array. Defaults to "position".
        bw (float, optional): bandwidth for the construction of the KDE estimate of the biased probability density. Defaults to 1.
        kT (float, optional): kT. Defaults to 1.
        min_grid (int, optional): Lower bound of the force domain. Defaults to -2.
        max_grid (int, optional): Upper bound of the force domain. Defaults to 2.
        nbins (int, optional): number of bins in grid. Defaults to 101.
        log_pace (int, optional): Pace for outputting progress and convergence. Defaults to 10.
        error_pace (int, optional): Pace for the cutoffcalculation of the on-the-fly measure of global convergence. Defaults to 200.
        WellTempered (binary, optional): Is the simulation well tempered?. Defaults to 1.
        periodic (int, optional): Is the CV space periodic? 1 for yes. Defaults to 0.
        hp_centre (float, optional): position of harmonic potential. Defaults to 0.0.
        hp_kappa (int, optional): force_constant of harmonic potential. Defaults to 0.
        lw_centre (float, optional): position of lower wall potential. Defaults to 0.0.
        lw_kappa (int, optional): force_constant of lower wall potential. Defaults to 0.
        uw_centre (float, optional): position of upper wall potential. Defaults to 0.0.
        uw_kappa (int, optional): force_constant of upper wall potential. Defaults to 0.

    Returns:
        grid (array of size (1, nbins)): CV-array
        Ftot_den (array of size (1, nbins)): Cumulative biased probability density
        Ftot (array of size (1, nbins)): Mean Force
        ofe (array of size (1, nbins)): on the fly estimate of the variance of the mean force
        ofe_history (list of size (1, error_pace)): running estimate of the global on the fly variance of the mean force
    """
    
    for initialise in [1]:
        grid = np.linspace(min_grid, max_grid, nbins)
        grid_space = (max_grid - min_grid) / (nbins-1)   
        grid_ext = 0.25 * (max_grid-min_grid)
        grid_length = max_grid - min_grid
        stride = int(len(position) / len(HILLS[:, 1]))
        const = (1 / (bw * np.sqrt(2 * np.pi) * stride))
        bw2 = bw ** 2
        if nhills > 0: total_number_of_hills = nhills
        else: total_number_of_hills = len(HILLS)
        
        # initialise force terms
        Fbias = np.zeros(len(grid))
        Ftot_num = np.zeros(len(grid))
        Ftot_den = np.zeros(len(grid))
        Ftot_den2 = np.zeros(len(grid))
        ofv_num = np.zeros(len(grid))

        #Calculate static force (form harmonic or wall potential)
        if len(F_static) != nbins: F_static = np.zeros(nbins)
        if hp_kappa > 0: F_static += find_hp_force(hp_centre, hp_kappa, grid, min_grid, max_grid, grid_space, periodic)
        if lw_kappa > 0: F_static += find_lw_force(lw_centre, lw_kappa, grid, min_grid, max_grid, grid_space, periodic)
        if uw_kappa > 0: F_static += find_uw_force(uw_centre, uw_kappa, grid, min_grid, max_grid, grid_space, periodic)
        

        # Definition Gamma Factor, allows to switch between WT and regular MetaD
        if WellTempered < 1: Gamma_Factor = 1
        else: Gamma_Factor = (HILLS[0, 4] - 1) / (HILLS[0, 4])
            
    for i in range(total_number_of_hills):
        
        #Get position data of window        
        s = HILLS[i, 1]  # centre position of Gaussian
        sigma_meta2 = HILLS[i, 2] ** 2  # width of Gaussian
        height_meta = HILLS[i, 3] * Gamma_Factor  # Height of Gaussian
        data = position[i * stride: (i + 1) * stride]  # positons of window of constant bias force.
        periodic_hills = find_periodic_point_numpy(np.array([s]), min_grid, max_grid, periodic, grid_ext, grid_length)
        periodic_positions = find_periodic_point_numpy(data, min_grid, max_grid, periodic, grid_ext, grid_length)

        #Calculate forces of window
        [pb_t, Fpbt, Fbias_window] = window_forces(periodic_positions, periodic_hills, grid, sigma_meta2, height_meta, kT, const, bw2, Ftot_den_limit)
        Fbias += Fbias_window                   
        dfds = np.where(pb_t > Ftot_den_limit, Fpbt / pb_t , 0) + Fbias - F_static
        Ftot_num += np.multiply(pb_t, dfds)
        
        # Calculate total force
        Ftot_den = Ftot_den + pb_t  # total probability density  
        Ftot = np.where(Ftot_den > Ftot_den_limit, Ftot_num / Ftot_den, 0)

        # terms for error calculation
        Ftot_den2 += np.square(pb_t)  # sum of (probability densities)^2
        ofv_num += np.multiply(pb_t, np.square(dfds))   # sum of (weighted mean force of window)^2
        
    return grid, Ftot_den, Ftot_den2, Ftot, ofv_num

### FAST Algorithm to run 1D MFI
@njit
def MFI_1D_long(y, HILLS="HILLS", position="position", bw=1, kT=1, min_grid=-2, max_grid=2, nbins=201, error_pace=10,
           WellTempered=1, nhills=-1, periodic=0, hp_centre=0.0, hp_kappa=0, lw_centre=0.0, lw_kappa=0,
           uw_centre=0.0, uw_kappa=0, Ftot_den_limit = 1E-10, F_static = np.zeros(123), FES_cutoff=0, Ftot_den_cutoff=0.1, ofe_non_exploration_penalty=30, use_weighted_st_dev=True):

    for initialise in [1]:
        grid = np.linspace(min_grid, max_grid, nbins)
        grid_space = (max_grid - min_grid) / (nbins-1)   
        grid_ext = 0.25 * (max_grid-min_grid)
        grid_length = max_grid - min_grid
        stride = int(len(position) / len(HILLS[:, 1]))
        const = (1 / (bw * np.sqrt(2 * np.pi) * stride))
        bw2 = bw ** 2
        if nhills > 0: total_number_of_hills = nhills
        else: total_number_of_hills = len(HILLS)
        
        # initialise force terms
        Fbias = np.zeros(len(grid))
        Ftot_num = np.zeros(len(grid))
        Ftot_den = np.zeros(len(grid))
        Ftot_den2 = np.zeros(len(grid))
        ofv_num = np.zeros(len(grid))
        
        error_history = []


        #Calculate static force (form harmonic or wall potential)
        if len(F_static) != nbins: F_static = np.zeros(nbins)
        if hp_kappa > 0: F_static += find_hp_force(hp_centre, hp_kappa, grid, min_grid, max_grid, grid_space, periodic)
        if lw_kappa > 0: F_static += find_lw_force(lw_centre, lw_kappa, grid, min_grid, max_grid, grid_space, periodic)
        if uw_kappa > 0: F_static += find_uw_force(uw_centre, uw_kappa, grid, min_grid, max_grid, grid_space, periodic)
    

        # Definition Gamma Factor, allows to switch between WT and regular MetaD
        if WellTempered < 1: Gamma_Factor = 1
        else: Gamma_Factor = (HILLS[0, 4] - 1) / (HILLS[0, 4])
            
    for i in range(total_number_of_hills):
        
        #Get position data of window        
        s = HILLS[i, 1]  # centre position of Gaussian
        sigma_meta2 = HILLS[i, 2] ** 2  # width of Gaussian
        height_meta = HILLS[i, 3] * Gamma_Factor  # Height of Gaussian
        data = position[i * stride: (i + 1) * stride]  # positons of window of constant bias force.
        periodic_hills = find_periodic_point_numpy(np.array([s]), min_grid, max_grid, periodic, grid_ext, grid_length)
        periodic_positions = find_periodic_point_numpy(data, min_grid, max_grid, periodic, grid_ext, grid_length)

        #Calculate forces of window
        [pb_t, Fpbt, Fbias_window] = window_forces(periodic_positions, periodic_hills, grid, sigma_meta2, height_meta, kT, const, bw2, Ftot_den_limit)
        Fbias += Fbias_window                   
        dfds = np.where(pb_t > Ftot_den_limit, Fpbt / pb_t , 0) + Fbias - F_static
        Ftot_num += np.multiply(pb_t, dfds)
        
        # Calculate total force
        Ftot_den = Ftot_den + pb_t  # total probability density  
        Ftot = np.where(Ftot_den > Ftot_den_limit, Ftot_num / Ftot_den, 0)

        # terms for error calculation
        Ftot_den2 += np.square(pb_t)  # sum of (probability densities)^2
        ofv_num += np.multiply(pb_t, np.square(dfds))   # sum of (weighted mean force of window)^2
        
        
        # Calculate error
        if (i + 1) % int(total_number_of_hills / error_pace) == 0 or (i+1) == total_number_of_hills:
            
            Ftot_den_sq = np.square(Ftot_den)
            Ftot_den_diff = (Ftot_den_sq-Ftot_den2)
            ofv = np.where(Ftot_den > Ftot_den_limit, ofv_num / Ftot_den, 0) - np.square(Ftot)
            if use_weighted_st_dev == True: ofv = np.multiply(ofv , np.where(Ftot_den_diff > 0, Ftot_den_sq / Ftot_den_diff, 0) )
            else: ofv = np.multiply(ofv , np.where(Ftot_den_diff > 0, Ftot_den2 / Ftot_den_diff, 0) )
            ofe = np.where(ofv > 10E-5 , np.sqrt(ofv), 0)
            if Ftot_den_cutoff != 0: ofe = np.where(Ftot_den > Ftot_den_cutoff, ofe, ofe_non_exploration_penalty)
            Aofe = np.sum(ofe) / np.count_nonzero(ofe)
                                    
            FES = intg_1D(Ftot, grid_space)
            AD = np.absolute(FES-y)
            if FES_cutoff != 0: AD = np.where(FES < FES_cutoff, AD, 0)
            AAD = np.sum(AD) / np.count_nonzero(AD)            
            error_history.append([Aofe, AAD])
            

    return grid, Ftot_den, Ftot, FES, ofe, AD, Aofe, AAD, error_history


@njit
def patch_FES_AD_ofe(force_vector, grid, y, nbins, PD_limit=1E-10):
    #initialisa terms
    PD_patch = np.zeros(nbins)
    PD2_patch = np.zeros(nbins)
    F_patch = np.zeros(nbins)
    OFV_patch = np.zeros(nbins)   
    
    #Patch force terms    
    for i in range(len(force_vector)):
        PD_patch += force_vector[i][0]
        PD2_patch += force_vector[i][1]
        F_patch += np.multiply(force_vector[i][0] ,force_vector[i][2])
        OFV_patch += force_vector[i][3]
        
    F_patch = np.where(PD_patch > PD_limit, F_patch / PD_patch, 0)

    #Calculate error
    PD_patch2 = np.square(PD_patch)
    PD_diff = PD_patch2 - PD2_patch
    PD_ratio = np.where( PD_diff > 0, PD2_patch / PD_diff, 0)    ### for stdev --->>> np.where( PD_diff > 0, PD2_patch / PD_diff)
    # PD_ratio = np.where( PD_diff > 0, PD_patch2 / PD_diff, 0)    ### for stdev --->>> np.where( PD_diff > 0, PD2_patch / PD_diff)
    OFV = np.multiply( np.where(PD_patch > 0, OFV_patch / PD_patch, 0) - np.square(F_patch) , PD_ratio)
    OFE = np.where(OFV > 1E-5, np.sqrt(OFV), 0)
    AOFE = np.sum(OFE) / nbins

    #Find FES and AAD
    FES = intg_1D(F_patch, grid[1] - grid[0])        
    AD = np.absolute(FES - y)
    AAD = np.sum(AD) / nbins
    
    if AOFE != AOFE:
        print("\n\n\n*************ATTENTION*****************\n THERE IS A --  NaN  --- SOMEWHERE IN THE OFE\n AOFE = " , AOFE,  "\n\n")
    
    return grid, PD_patch, F_patch, FES, AD, AAD, OFE, AOFE


@njit
def patch_FES_AD_ofe_cutoff(force_vector, grid, y, PD_limit=1E-10, FES_cutoff=0, PD_cutoff=0.1, ofe_non_exploration_penalty=30, use_weighted_st_dev=True):
    #initialise terms
    for _init_terms_ in range(1):
        PD_patch = np.zeros_like(grid)
        PD2_patch = np.zeros_like(grid)
        F_patch = np.zeros_like(grid)
        OFV_patch = np.zeros_like(grid)   
    
    #Patch force terms    
    for i in range(len(force_vector)):
        PD_patch += force_vector[i][0]
        PD2_patch += force_vector[i][1]
        F_patch += np.multiply(force_vector[i][0] ,force_vector[i][2])
        OFV_patch += force_vector[i][3]
        
    F_patch = np.where(PD_patch > PD_limit, F_patch / PD_patch, 0)

    #Calculate error
    PD_patch2 = np.square(PD_patch)
    PD_diff = PD_patch2 - PD2_patch
    if use_weighted_st_dev == True: PD_ratio = np.where( PD_diff > 0, PD_patch2 / PD_diff, 0)    ## standard error of the weighted mean(keeps decreasing as more datapoints are added)
    else: PD_ratio = np.where( PD_diff > 0, PD2_patch / PD_diff, 0)    ## weighted standard deviation (keeps converging as more datapoints are added)
    OFV = np.multiply( np.where(PD_patch > PD_limit, OFV_patch / PD_patch, 0) - np.square(F_patch) , PD_ratio)
    OFE = np.where(OFV > 1E-5, np.sqrt(OFV), 0)
    if PD_cutoff != 0: OFE = np.where(PD_patch > PD_cutoff, OFE, ofe_non_exploration_penalty)
    AOFE = np.sum(OFE) / np.count_nonzero(OFE)
    
    #Find FES and AAD
    FES = intg_1D(F_patch, grid[1] - grid[0])        
    AD = np.absolute(FES - y)
    if FES_cutoff != 0: AD = np.where(FES < FES_cutoff, AD, 0)
    AAD = np.sum(AD) / np.count_nonzero(AD)
    
    if AOFE != AOFE:
        print("\n\n\n*************ATTENTION*****************\n THERE IS A --  NaN  --- SOMEWHERE IN THE OFE\n AOFE = " , AOFE,  "\n\n")
    
    return grid, PD_patch, F_patch, FES, AD, AAD, OFE, AOFE


def patch_forces(force_vector):
    """Takes in a collection of force and probability density and patches them.
    Args:
        force_vector (list): collection of force terms (n * [Ftot_den, Ftot])

    Returns:
        Patched probability density and mean forces (list) -> ([Ftot_den, Ftot])
    """
    PD_patch = np.zeros(np.shape(force_vector[0][0]))
    F_patch = np.zeros(np.shape(force_vector[0][0]))
    for i in range(len(force_vector)):
        PD_patch += force_vector[i][0]
        F_patch += force_vector[i][0] * force_vector[i][1]
    F_patch = np.divide(F_patch, PD_patch, out=np.zeros_like(F_patch), where=PD_patch > 0.000001)

    return [PD_patch, F_patch]


def patch_FES_ofe(force_vector, grid, nbins):
    #initialisa terms
    PD_patch = np.zeros(nbins)
    PD2_patch = np.zeros(nbins)
    F_patch = np.zeros(nbins)
    OFV_patch = np.zeros(nbins)   
    
    #Patch force terms    
    for i in range(len(force_vector)):
        PD_patch += force_vector[i][0]
        PD2_patch += force_vector[i][1]
        F_patch += force_vector[i][0] * force_vector[i][2]
        OFV_patch += force_vector[i][3]
    F_patch = np.divide(F_patch, PD_patch, out=np.zeros_like(F_patch), where=PD_patch > 0)

    #Calculate error
    PD_ratio = np.divide(PD2_patch, (PD_patch ** 2 - PD2_patch), out=np.zeros_like(PD_patch), where=(PD_patch ** 2 - PD2_patch) > 0)
    OFV = (np.divide(OFV_patch, PD_patch, out=np.zeros_like(OFV_patch), where=PD_patch > 0) - F_patch ** 2) * PD_ratio
    OFE = np.where(OFV > 10E-10, np.sqrt(OFV), 0)
    AOFE = sum(OFE) / nbins

    #Find FES and AAD
    FES = intg_1D(F_patch, grid[1]-grid[0])        
    
    if AOFE != AOFE:
        print("\n\n\n*************ATTENTION*****************")
        print("AOFE = ", AOFE)
        
        print("\n\n")
        print("OFE = \n", OFE)
        
        print("\n\n")
        print("OFV = \n", OFV)
        
        print("\n\n")
        print("PD_ratio = \n", PD_ratio)
        
        print("\n\n")
        print("PD2_patch = \n", PD2_patch)
        
        print("\n\n")
        print("PD2_patch = \n", PD2_patch)
        
        print("\n\n")
        print("OFV_patch = \n", OFV_patch)
        
        exit()
    
    return [grid, PD_patch, F_patch, FES, OFE, AOFE]


def bootstrap_forw_back(grid, forward_force, backward_force, n_bootstrap):
    """Algorithm to determine bootstrap error

    Args:
        grid (array): CV grid positions
        forward_force (list): collection of force terms (n * [Ftot_den, Ftot]) from forward transition
        backward_force (list): collection of force terms (n * [Ftot_den, Ftot]) from backward transition
        n_bootstrap (int): bootstrap itterations

    Returns:
        [FES_avr, sd_fes, variance_prog, stdev_prog, var_fes_prog, sd_fes_prog ]
    """
   
    #Define terms that will be updated itteratively
    Ftot_inter = np.zeros(len(grid))
    Ftot_sum = np.zeros(len(grid))
    Ftot_den_sum = np.zeros(len(grid))
    Ftot_den2_sum = np.zeros(len(grid))
    FES_sum = np.zeros(len(grid))
    FES2_sum = np.zeros(len(grid))

    #store var and sd progression here
    variance_prog = []
    stdev_prog = []
    var_fes_prog  = []
    sd_fes_prog = []

    #Save patch force terms and FES
    force_patch_collection = []
    FES_collection = []

    #Patch forces
    [Ftot_den, Ftot] = patch_forces(np.concatenate((forward_force, backward_force)))

    #save non-random probability density
    Ftot_den_base = np.array(Ftot_den)


    for itteration in range(n_bootstrap):

        #Randomly choose 49 forward forces and backward forces
        force = []    
        for i in range(len(forward_force)):
            force.append(forward_force[random.randint(0,len(forward_force)-1)])
            force.append(backward_force[random.randint(0,len(forward_force)-1)])

                
                
        #patch forces to find average Ftot_den, Ftot and FES
        [Ftot_den, Ftot] = patch_forces(force)
        FES = intg_1D(F_patch, grid[1]-grid[0])
        FES = FES - FES[0]

        #Save terms
        force_patch_collection.append([Ftot_den, Ftot])
        FES_collection.append(FES)

        #calculate sums for variance
        Ftot_inter += Ftot_den * Ftot**2
        Ftot_sum += Ftot
        Ftot_den_sum += Ftot_den
        Ftot_den2_sum += Ftot_den**2
        
        FES_sum += FES
        FES2_sum += FES**2

        if itteration > 0:
            
            #calculate force variance
            Ftot_avr = Ftot_sum / (itteration+1)
            Ftot2_weighted = np.divide(Ftot_inter, Ftot_den_sum, out=np.zeros_like(Ftot_inter), where=Ftot_den_base>10)
            Ftot_den_ratio = np.divide(Ftot_den_sum ** 2, (Ftot_den_sum ** 2 - Ftot_den2_sum), out=np.zeros_like(Ftot_den_sum), where=Ftot_den_base > 10)
            variance = (Ftot2_weighted - Ftot_avr**2) * Ftot_den_ratio
            n_eff = np.divide(Ftot_den_sum ** 2, Ftot_den2_sum, out=np.zeros_like(Ftot_den), where=Ftot_den_base>10)
            stdev = np.where(Ftot_den_base > 10,  np.sqrt(variance / n_eff ), 0)
        
            #calculate FES variance
            FES_avr = FES_sum/ (itteration+1)
            var_fes = np.zeros(len(grid))
            for i in range(len(FES_collection)): 
                var_fes += (FES_collection[i] - FES_avr)**2
            var_fes = 1/(len(FES_collection)-1) * var_fes
            sd_fes = np.sqrt(var_fes)
            
            #save variance
            variance_prog.append(sum(variance)/len(grid))
            stdev_prog.append(sum(stdev)/len(grid))
            var_fes_prog.append(sum(var_fes)/len(grid))
            sd_fes_prog.append(sum(sd_fes)/len(grid))
        
        
        #print progress
        if (itteration+1) % 50 == 0:
            print(itteration+1, ": var:", round(variance_prog[-1],5), "     sd:", round(stdev_prog[-1],5), "      FES: var:", round(var_fes_prog[-1],3), "     sd:", round(sd_fes_prog[-1],3) )
            
    return [FES_avr, sd_fes, variance_prog, stdev_prog, var_fes_prog, sd_fes_prog ]



# Integrate Ftot, obtain FES
# import scipy.integrate as integrate

# def intg_1D_old(x, F):
#     """Integration of 1D gradient using finite difference method (simpson's method).

#     Args:
#         x (array): grid
#         F (array): Mean force

#     Returns:
#         FES (array): Free energy surface
#     """
#     fes = []
#     for j in range(len(x)): fes.append(integrate.simps(F[:j + 1], x[:j + 1]))
#     fes = fes - min(fes)
#     return fes


def plot_recap(X, FES, Ftot_den, ofe, ofe_history, time_history, FES_lim=40, ofe_lim = 10, error_log_scale = 1):
    """Plot result of 1D MFI algorithm. 1. FES, 2. varinace_map, 3. Cumulative biased probability density, 4. Convergece of variance.

    Args:
        X (array): gird
        FES (array): Free energy surface
        Ftot_den (array): _description_
        ofe (array): Cumulative biased probability density
        ofe_history (list): on the fly estimate of the local convergence
        time_history (_type_): _description_
        FES_lim (int, optional): Upper energy value in FES plot. Defaults to 40.
        ofe_lim (int, optional): Upper error value in FES plot. Defaults to 10.
        error_log_scale (boolean, optional): Option to make error_conversion plot with a log scale. 1 for log scale. Defaults to 1.
    """
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    #plot ref f
    y = 7*X**4 - 23*X**2
    y = y - min(y)  
    axs[0, 0].plot(X, y, color="red", alpha=0.3);
    
    axs[0, 0].plot(X, FES);
    axs[0, 0].set_ylim([0, FES_lim])
    axs[0, 0].set_ylabel('F(CV1) [kJ/mol]')
    axs[0, 0].set_xlabel('CV1')
    axs[0, 0].set_title('Free Energy Surface')
    axs[0, 0].set_xlim(np.min(X),np.max(X))

    axs[0, 1].plot(X, ofe);
    axs[0, 1].plot(X, np.zeros(len(X)), color="grey", alpha=0.3);
    axs[0, 1].set_ylabel('Mean Force Error [kJ/(mol*nm)]')
    axs[0, 1].set_xlabel('CV1')
    axs[0, 1].set_title('Local Error Map')
    # axs[0, 1].set_ylim(-0.1, ofe_lim)
    axs[0, 0].set_xlim(np.min(X),np.max(X))

    

    axs[1, 0].plot(X, Ftot_den);
    axs[1, 0].set_ylabel('Count [relative probability]')
    axs[1, 0].set_xlabel('CV1')
    axs[1, 0].set_title('Total Probability density')
    axs[0, 0].set_xlim(np.min(X),np.max(X))
    # axs[1, 0].set_yscale('log')


    axs[1, 1].plot([time/1000 for time in time_history], ofe_history);
    # axs[1, 1].plot([time/1000 for time in time_history], np.zeros(len(ofe_history)), color="grey", alpha=0.3);
    axs[1, 1].set_ylabel('Average Mean Force Error [kJ/(mol*nm)]')
    axs[1, 1].set_xlabel('Simulation time')
    axs[1, 1].set_title('Progression of Average Mean Force Error')
    if error_log_scale == 1: axs[1, 1].set_yscale('log')
    # axs[1, 1].set_ylim([-0.5, max(ofe_history)])

    fig.tight_layout()    





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
        if input_array[ii] == 0: output_array[ii] = np.nan
        else: output_array[ii] = input_array[ii]
    return output_array