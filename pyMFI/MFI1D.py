import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle
from numba import jit, njit


def load_HILLS(hills_name="HILLS"):
    """Load 1-dimensional hills data (includes time, position_x, position_y, hills_parameters)
                Adjustment 2 - Last hill is removed from output array, as no complete set of COLVAR data is available of that hill. 

    Args:
        hills_name (str, optional): Name of hills file. Defaults to "HILLS".

    Returns:
        np.array: Hills data with length equal to the total number of hills. Information: [time [ps], position [nm], MetaD_sigma [nm], MetaD_height [nm], MetaD_biasfactor]
    """
    for file in glob.glob(hills_name):
        hills = np.loadtxt(file)
        hills = hills[:-1] # Last hill is removed from output array, as no complete set of COLVAR data is available of that hill. 
        hills0 = np.array(hills[0])
        hills0[3] = 0
        hills = np.concatenate(([hills0], hills)) # One dummy hills with zero height is introduced at the beginning of the output array, so that the forces of the first few positions are calculated without the effect of any bias hill.
    return hills


# Load the trajectory (position) data
def load_position(position_name="position"):
    """Load 1-dimensional position (CV) data.

    Args:
        position_name (str, optional): Name of position file. Defaults to "position".

    Returns:
        np.array: position data
    """
    for file1 in glob.glob(position_name):
        colvar = np.loadtxt(file1)
    return colvar[:-1, 1]  #last position can be removed, since it is not part of a complete set of positions in a window of constant bias.

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


def find_periodic_point(x_coord, min_grid, max_grid, periodic, grid_ext):
    """Finds periodic copies of input coordinates. First checks if systems is periodic. If not, returns input coordinate array. Next, it checks if each coordinate is within the boundary range (grid min/max +/- grid_ext). If it is, periodic copies will be made on the other side of the CV-domain. 

Args:
    x_coord (float): CV-coordinate
    min_grid (float): minimum value of grid
    max_grid (float): maximum value of grid
    periodic (binary): information if system is periodic. value of 0 corresponds to non-periodic system; function will only return input coordinates. Value of 1 corresponds to periodic system; function will return input coordinates with periodic copies.
    grid_ext (float): how far outside the domain periodic copies are searched for. E.g. if the domain is from -2 to +2 and the grid ext(end) is set to 1, then periodic copies are only found for input_coordiante < -2 + 1 or input_coordiante > +2 - 1.


Returns:
    list: list of input coord and if applicable periodic copies

    """
    if periodic == 1:
        grid_length = max_grid - min_grid
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
def find_periodic_point_numpy(coord_array, min_grid, max_grid, periodic, grid_ext):
    """Finds periodic copies of input coordinates. First checks if systems is periodic. If not, returns input coordinate array. Next, it checks if each coordinate is within the boundary range (grid min/max +/- grid_ext). If it is, periodic copies will be made on the other side of the CV-domain. 

    Args:
        coord_array (array): array of CV-coordinate s
        min_grid (float): minimum value of grid
        max_grid (float): maximum value of grid
        periodic (binary): information if system is periodic. value of 0 corresponds to non-periodic system; function will only return input coordinates. Value of 1 corresponds to periodic system; function will return input coordinates with periodic copies.
        grid_ext (float): how far outside the domain periodic copies are searched for. E.g. if the domain is from -2 to +2 and the grid ext(end) is set to 1, then periodic copies are only found for input_coordiante < -2 + 1 or input_coordiante > +2 - 1.


    Returns:
        np.array: list of input coord and if applicable periodic copies
    """

    if periodic == 0:
        return coord_array
    elif periodic == 1:
        grid_length = max_grid - min_grid
        len_coord_array = len(coord_array)
        for i in range(len_coord_array):
            if coord_array[i] < min_grid+grid_ext:
                coord_array = np.append(coord_array, coord_array[i] + grid_length)
            elif coord_array[i] > max_grid-grid_ext:
                coord_array = np.append(coord_array, coord_array[i] - grid_length)
        return coord_array  
    
@jit
def window_forces(periodic_positions, periodic_hills, grid, sigma_meta2, height_meta, kT, const, bw2, Ftot_den_limit=1E-10):
    """Takes in two arrays of positions. The periodic_positions are collected from the COLVAR file during a period of constant bias and calculates the force component associated with the probability density. periodic_hills are also positions collected from the HILLS file and calculates the force component resulting from the metadynamics bias. In a periodic system, positions have periodic copies if applicable. 

    Args:
        periodic_positions (array of shape (n,)): This array contains the positions from the COLVAR file, that were collected during a period of constant bias. n is the number of positions including their periodic copies if applicable. 
        periodic_hills (array of shape (n,)): This array contains the position of the metadynamics hill (from the HILLS file), that was deposited at the beginning of the period of constant bias. If applicable the array also contains periodic copies. 
        grid (array of shape (nbins,)): CV-array.
        sigma_meta2 (float): width of metadynamics hill squared.
        height_meta (float): height of metadynamics hill
        kT (float): Temperature of the system.
        const (float): constant factor used for the calculation of the kernel. This constant enables the patching of data with different values of bw and stride . const = (1 / (bw * np.sqrt(2 * np.pi) * stride))
        bw2 (float): width of kernel density estimation (for the probability density) squared.
        Ftot_den_limit (float, optional): Probability density limit below which data will be set to zero (this is done for numerical stability reasons. For default Ftot_den_limit, numerical difference is negligable). Defaults to 1E-10.

    Returns:
        list: [pb_t, Fpbt, Fbias_window]\n
        pb_t (array of shape (nbins,)): Probability density of window\n
        Fpbt (array of shape (nbins,)): Force component associated with the probability density of the time-window. \n
        Fbias_window (array of shape (nbins,)): Force component associated with the metadynamics hill deposited at the beginning of the time-window. 
    """
    
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
    """Find 1D harmonic potential force equivalent to f = hp_kappa * (grid - hp_centre). 

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
    #First, find harmonic potential for non-periodic case
    F_harmonic = hp_kappa * (grid - hp_centre)
    #Second, if periodic, make harmonic potential periodic
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
    """Find lower half of 1D harmonic potential force equivalent to f = 2 * lw_kappa * (grid - lw_centre) for grid < lw_centre and f = 0 otherwise. This can change for periodic cases.

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
    #First, find harmonic potential for non-periodic case
    F_harmonic = np.where(grid < lw_centre, 2 * lw_kappa * (grid - lw_centre), 0)
    #Second, if periodic, make harmonic potential periodic
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
    """Find upper half of 1D harmonic potential force equivalent to f = 2 * uw_kappa * (grid - uw_centre) for grid > uw_centre and f = 0 otherwise. This can change for periodic cases.

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
    #First, find harmonic potential for non-periodic case
    F_harmonic = np.where(grid > uw_centre, uw_kappa * (grid - uw_centre), 0)
    #Second, if periodic, make harmonic potential periodic
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
        dx (float): grid spacing (i.e. space between consecutive grid entries)

    Returns:
        array: Free energy surface
    """
    fes = np.zeros_like(Force)
    
    for j in range(len(Force)): 
        y = Force[:j + 1]
        N = len(y)
        if N % 2 == 0: fes[j] = dx/6 * (np.sum(y[: N-3: 2] + 4*y[1: N-3+1: 2] + y[2: N-3+2: 2]) + np.sum(y[1: N-2: 2] + 4*y[1+1: N-1: 2] + y[1+2: N: 2])) + dx/4 * ( y[1] + y[0] + y[-1] + y[-2])
        else: fes[j] = dx / 3.0 * np.sum(y[: N-2: 2] + 4*y[1: N-1: 2] + y[2: N: 2])
            3
    fes = fes - min(fes)
    return fes
   
### Algorithm to run 1D MFI
# Run MFI algorithm with on the fly error calculation
@njit
def MFI_1D(HILLS="HILLS", position="position", bw=0.1, kT=1, min_grid=-2, max_grid=2, nbins=201, 
           log_pace=-1, error_pace=-1, WellTempered=1, nhills=-1, periodic=0, 
           hp_centre=0.0, hp_kappa=0, lw_centre=0.0, lw_kappa=0, uw_centre=0.0, uw_kappa=0, F_static = np.zeros(1), 
           Ftot_den_limit = 1E-10, FES_cutoff = 0, Ftot_den_cutoff = 0, non_exploration_penalty = 0, save_intermediate_fes_error_cutoff = False, use_weighted_st_dev = True):
    
    """Compute a time-independent estimate of the Mean Thermodynamic Force, i.e. the free energy gradient in 1D CV spaces.

    Args:
        HILLS (str): HILLS array. Defaults to "HILLS".
        position (str): CV/position array. Defaults to "position".
        bw (float, optional): bandwidth for the construction of the KDE estimate of the biased probability density. Defaults to 1.
        kT (float, optional): Boltzmann constant multiplied with temperature (reduced format, 120K -> 1).
        min_grid (float, optional): Lower bound of the force domain. Defaults to -2.
        max_grid (float, optional): Upper bound of the force domain. Defaults to 2.
        nbins (int, optional): number of bins in grid. Defaults to 101.
        log_pace (int, optional): Pace for outputting progress and convergence. Defaults to -1. When set to -1, progress will be outputted 5 times in total.
        error_pace (int, optional): Pace for calculating the on-the-fly error for  estimating global convergence. Defaults to -1. When set to -1, on-the-fly error will be calculated 50 times.
        WellTempered (binary, optional): Is the simulation well tempered?. Defaults to 1.
        nhills (binary, optional): Number of HILLS to be analysed. When set to -1, all HILLS will be analysed. Defaults to -1.
        periodic (binary, optional): Is the CV space periodic? 1 for yes. Defaults to 0.
        hp_centre (float, optional): position of harmonic potential. Defaults to 0.0.
        hp_kappa (flaot, optional): force_constant of harmonic potential. Defaults to 0.
        lw_centre (float, optional): position of lower wall potential. Defaults to 0.0.
        lw_kappa (flaot, optional): force_constant of lower wall potential. Defaults to 0.
        uw_centre (float, optional): position of upper wall potential. Defaults to 0.0.
        uw_kappa (flaot, optional): force_constant of upper wall potential. Defaults to 0.
        F_static (array, optional): Option to provide a starting bias potential that remains constant through the algorithm. This could be a harmonic potential, an previously used MetaD potential or any other bias potential defined on the grid. Defaults to np.zeros(1), which will automatically set F_static to a zero-array with length=nbins.
        Ftot_den_limit (float, optional): Probability density limit below which data will be set to zero (this is done for numerical stability reasons. For default Ftot_den_limit, numerical difference is negligable). Defaults to 1E-10.
        FES_cutoff (float, optional): Cutoff applied to FES and error calculation for FES values over the FES_cutoff. All FES values above the FES_cutoff won't contribuite towards the error. Useful when high FES values have no physical meaning. When FES_cutoff = 0, no cufoff is applied.Defaults to 0. 
        Ftot_den_cutoff (flaot, optional): Cutoff applied to FES and error calculation for Ftot_den (Probability density) values below the Ftot_den_cutoff. All FES values that are excluded by the cutoff won't contribuite towards the error. Useful when low Probability density values have little statistical significance or no physical meaning. When Ftot_den_cutoff = 0, no cufoff is applied. Defaults to 0.
		non_exploration_penalty (float, optional): Turns zero-value error to the non_exploration_penalty value. This should be used in combination with the cutoff. If some part of CV space hasn't been explored, or has a FES value that is irrelevanlty high, the cutoff will set the error of that region to zero. If the non_exploration_penalty is larger than zero, the error of that region will take the value of the non_exploration_penalty instead of zero. Default is set to 0.
        save_intermediate_fes_error_cutoff (bool, optional): If Ture, every time the error is calculated, the FES, variance, standard deviation and cutoff will be saved. Defaults to False.
        use_weighted_st_dev (bool, optional): When set to True, the calculated error will be the weighted standard deviation ( var^0.5 ). When set to False, the calculated error will be the standard error ( (var/n_sample)^0.5 ). Defaults to True. (The standard devaition is expected to converge after enough time, while the standard error is expected to decrease as more datapoints are added.)

    Returns:
        tuple: grid, Ftot_den, Ftot_den2, Ftot, ofv_num, FES, ofv, ofe, cutoff, error_evol, fes_error_cutoff_evol\n
        grid (array of size (nbins,)): CV-array.\n
        Ftot_den (array of size (nbins,)): Cumulative biased probability density.\n
        Ftot_den2 (array of size (nbins,): Cumulative (biased probability density squared). Used for error calculation.\n
        Ftot (array of size (nbins,)): Local Mean Force. When integrated will be the FES.\n
        ofv_num (array of size (nbins,)): Numerator of "on-the-fly" variance (ofv). Used for error calculation.\n
        FES (array of size (nbins,)): Free Energy Surface\n
        ofv (array of size (nbins,)): "on-the-fly" variance\n
        ofe (array of size (nbins,)): "on-the-fly" estimate of the standard deviation of the mean force\n
        cutoff (binary array of size (nbins,)): If FES_cutoff and/or Ftot_den_cutoff are specified, grid values outside the cufoff will be zero and grid values inside the cutoff will be one. If FES_cutoff and Ftot_den_cutoff not active, cutoff will be an array of ones.\n
        error_evol (array of size (4, total_number_of_hills//error_pace)): Evolution of error. First array is the collection of the global (average) "on-the-fly" variance, second array is the collection of the global (average) "on-the-fly" standard deviation, third array is the percentage of values within the specified cutoff, and the fourth array are the simulation times of the latter former arrays.\n
        fes_error_cutoff_evol (array of size (4, total_number_of_hills//error_pace, nbins)): First array is the collection of FES, second array is the collection of local "on-the-fly" variance, third array is the collection of local "on-the-fly" standard deviation, and the fourth is the collection of the cutoffs. The elements of the collections were calculated at the simulation times of the fourth array in error_evol.
        
        

    """
    
    #Specify grid
    grid = np.linspace(min_grid, max_grid, nbins)
    grid_space = (max_grid - min_grid) / (nbins-1)
    grid_ext = 0.25 * (max_grid-min_grid)
    
    #Specift constants
    stride = int(len(position) / len(HILLS[:, 1]))
    const = (1 / (bw * np.sqrt(2 * np.pi) * stride))
    bw2 = bw ** 2
    if nhills > 0: total_number_of_hills = nhills
    else: total_number_of_hills = len(HILLS)
    if log_pace < 0: log_pace = total_number_of_hills // 5
    if error_pace < 0: error_pace = total_number_of_hills // 50
    
    
    # initialise force terms
    Fbias = np.zeros(nbins)
    Ftot_num = np.zeros(nbins)
    Ftot_den = np.zeros(nbins)
    Ftot_den2 = np.zeros(nbins)
    ofv_num = np.zeros(nbins)
    
    #initialise other arrays and floats for intermediate results
    if total_number_of_hills % error_pace == 0: error_evol = np.zeros((4, total_number_of_hills // error_pace))
    else: error_evol = np.zeros((4, total_number_of_hills // error_pace +1))
    if save_intermediate_fes_error_cutoff == True: fes_error_cutoff_evol = np.zeros((4, len(error_evol[0]), nbins))
    else: fes_error_cutoff_evol = np.zeros((4, len(error_evol[0]), nbins))#np.zeros(1)
    error_count = 0
        
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
        periodic_hills = find_periodic_point_numpy(np.array([s]), min_grid, max_grid, periodic, grid_ext)
        periodic_positions = find_periodic_point_numpy(data, min_grid, max_grid, periodic, grid_ext)
                
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
        if (i + 1) % error_pace == 0 or (i+1) == total_number_of_hills:
            
            #If applicable, find FES and cutoff
            cutoff = np.ones(nbins, dtype=np.float64)
            if FES_cutoff > 0 or save_intermediate_fes_error_cutoff == True: FES = intg_1D(Ftot, grid_space)
            if FES_cutoff > 0: cutoff = np.where(FES < FES_cutoff, 1.0, 0)
            if Ftot_den_cutoff > 0: cutoff = np.where(Ftot_den > Ftot_den_cutoff, 1.0, 0)
            
                
            # calculate error
            ofv = np.where(Ftot_den > Ftot_den_limit, ofv_num / Ftot_den, 0) - np.square(Ftot)
            Ftot_den_sq = np.square(Ftot_den)
            Ftot_den_diff = Ftot_den_sq - Ftot_den2
            if use_weighted_st_dev == True: ofv *= np.where(Ftot_den_diff > 0, Ftot_den_sq / Ftot_den_diff, 0)
            else: ofv *= np.where(Ftot_den_diff > 0, Ftot_den2 / Ftot_den_diff, 0)
            ofv = np.multiply(ofv, cutoff)
            if non_exploration_penalty > 0: ofv = np.where(cutoff > 0.5, ofv, non_exploration_penalty**2) 
            ofe = np.where(ofv > 0, np.sqrt(ofv), 0)  

            #save global error evolution
            error_evol[0,error_count] = sum(ofv) / np.count_nonzero(ofv)
            error_evol[1,error_count] = sum(ofe) / np.count_nonzero(ofe)
            error_evol[2,error_count] = np.count_nonzero(cutoff) / nbins
            error_evol[3,error_count] = HILLS[i,0]
 
            # print(np.shape(FES), np.shape())
            
            # save local fes, error and cutoff
            if save_intermediate_fes_error_cutoff == True:
                fes_error_cutoff_evol[0,error_count] = FES
                fes_error_cutoff_evol[1,error_count] = ofv
                fes_error_cutoff_evol[2,error_count] = np.where(ofv != 0, np.sqrt(ofv), 0) 
                fes_error_cutoff_evol[3,error_count] = cutoff
            
            error_count += 1
            
            #window error             
                  
             
        if (i + 1) % log_pace == 0 or (i+1) == total_number_of_hills:
            print((round((i + 1) / total_number_of_hills * 100, 0)) , "%   OFE =", round(error_evol[1,error_count-1], 4))
                

    ofe = np.where(ofv != 0, np.sqrt(ofv), 0)
    FES = intg_1D(Ftot, grid_space)
    
    return grid, Ftot_den, Ftot_den2, Ftot, ofv_num, FES, ofv, ofe, cutoff, error_evol, fes_error_cutoff_evol


@njit
def patch_forces(force_vector, PD_limit=1E-10):
    """Takes in a collection of force and probability density and patches them.
    Args:
        force_vector (list): collection of force terms (n * [Ftot_den, Ftot])
        PD_limit (float, optional): Probability density limit below which data will be set to zero (this is done for numerical stability reasons. For default Ftot_den_limit, numerical difference is negligable). Defaults to 1E-10.

    Returns:
        list: [PD_patch, F_patch]\n
        PD_patch (array): Patched probability density \n
        F_patch (array): Patched mean force 
    """
    PD_patch = np.zeros(np.shape(force_vector[0][0]))
    F_patch = np.zeros(np.shape(force_vector[0][0]))
    for i in range(len(force_vector)):
        PD_patch += force_vector[i][0]
        F_patch += np.multiply(force_vector[i][0] , force_vector[i][1])
    F_patch = np.where(PD_patch > PD_limit, F_patch / PD_patch, 0)

    return [PD_patch, F_patch]


@njit
def get_cutoff_1D(Ftot_den=None, Ftot_den_cutoff=0.1, FES=None, FES_cutoff=-1):
    """Finds the cutoff array according to the specifications. If the "@njit" is deactivated, use get_cutoff_1D_nonjit() instead.

	Args:
		Ftot_den (np.array, optional): If a probability density (Ftot_den) cutoff should be applied, this argument in necessary. Defaults to None.
		Ftot_den_cutoff (float, optional): Specifies the cutoff limit of the probability density. Values below the limit will be cut-off. Will only be applied if Ftot_den is provided. Defaults to 0.1.
		FES (np.array, optional): free energy surfacee. If a free energy surface (FES) cutoff should be applied, this argument in necessary. Defaults to None.
		FES_cutoff (float, optional): Specifies the cutoff limit of the FES. Values above the limit will be cut-off. Will only be applied if FES is provided. Defaults to 23.

	Returns:
		array :cutoff array with the shape of Ftot_den or FES. Elements that correspond to the probability density above the Ftot_den_cutoff or the FES below the FES_cutoff will be 1. Elements outside the cutoff will be 0.
    """

    if Ftot_den != None: nbins = len(Ftot_den)
    elif FES != None: nbins = len(FES)
    else: print("\n**ERROR** Please specify either Ftot_den or FES or both!! **ERROR**\n\n")
    
    cutoff = np.ones(nbins)

    if Ftot_den != None: cutoff = np.where(Ftot_den > Ftot_den_cutoff, 1.0, 0.0)
    if FES != None: cutoff = np.where(FES <= FES_cutoff, cutoff, 0.0)

    return cutoff


def get_cutoff_1D_nonjit(Ftot_den=None, Ftot_den_cutoff=0.1, FES=None, FES_cutoff=23):
    """Finds the cutoff array according to the specifications. 

	Args:
		Ftot_den (np.array, optional): If a probability density (Ftot_den) cutoff should be applied, this argument in necessary. Defaults to None.
		Ftot_den_cutoff (float, optional): Specifies the cutoff limit of the probability density. Values below the limit will be cut-off. Will only be applied if Ftot_den is provided. Defaults to 0.1.
		FES (np.array, optional): free energy surfacee. If a free energy surface (FES) cutoff should be applied, this argument in necessary. Defaults to None.
		FES_cutoff (float, optional): Specifies the cutoff limit of the FES. Values above the limit will be cut-off. Will only be applied if FES is provided. Defaults to 23.

	Returns:
		array : cutoff array with the shape of Ftot_den or FES. Elements that correspond to the probability density above the Ftot_den_cutoff or the FES below the FES_cutoff will be 1. Elements outside the cutoff will be 0.
    """

    if hasattr(Ftot_den, "__len__") == True: cutoff = np.where(Ftot_den > Ftot_den_cutoff, 1, 0)
    elif hasattr(FES, "__len__") == True: cutoff = np.ones_like(FES)
    else: print("\n**ERROR** Please specify either Ftot_den or FES or both!! **ERROR**\n\n")
    
    if hasattr(FES, "__len__") == True: cutoff = np.where(FES <= FES_cutoff, cutoff, 0.0)

    return cutoff




@njit
def patch_forces_ofe(force_vector, PD_limit=1E-10, use_weighted_st_dev=True, ofe_progression=True, Ftot_den_cutoff=-1, non_exploration_penalty=0):
    
    """Takes in an array of force terms, patches them together and calculates the error. The force vector has to be in the format [Ftot_den, Ftot_den2, Ftot, ofv_num]. 

    Args:
        force_vector (array of shape (n,4, nbins)) where n is the number of independent force terms: The force terms are tipically outputted from the MFI_1D functions. They should be in the format: [Ftot_den (Probability density), Ftot_den2, Ftot (Mean Force), ofv_num (numerator of on the fly variance)]. 
        PD_limit (flaot, optional): Probability density values below the PD_limit will be approximated to be 0, to ensure the numerical stability of the algorithm. Defaults to 1E-10.
        ofe_non_exploration_penalty (flaot, optional): On-the-fly error values that that are ouside of the PD_cutoff will have a error equal to the ofe_non_exploration_penalty. Absolute deviation values that that are ouside of the PD_cutoff will have a error equal to the ofe_non_exploration_penalty/10. Defaults to 100.
        use_weighted_st_dev (bool, optional): When set to True, the calculated error will be the weighted standard deviation ( var^0.5 ). When set to False, the calculated error will be the standard error ( (var/n_sample)^0.5 ). Defaults to True. (The standard devaition is expected to converge after enough time, while the standard error is expected to decrease as more datapoints are added.).
        ofe_progression (bool, optional): If True, the error progression will be returned. If false, error progression will be an array will all elements being the final error. Default set to False.
		Ftot_den_cutoff (float, optional): Specifies the cutoff limit of the probability density. Values below the limit will be cut-off. Will only be applied if Ftot_den_cutoff is larger than 0. Defaults to -1.
		non_exploration_penalty (float, optional): Turns zero-value error to the non_exploration_penalty value. This should be used in combination with the cutoff. If some part of CV space hasn't been explored, or has a FES value that is irrelevanlty high, the cutoff will set the error of that region to zero. If the non_exploration_penalty is larger than zero, the error of that region will take the value of the non_exploration_penalty instead of zero. Default is set to 0.

    Returns:
        tuple : PD_patch, PD2_patch, F_patch, OFV_num_patch, OFE, Aofe_progression\n
        PD_patch (array of size (nbins,)): Patched Probability density\n
        PD2_patch (array of size (nbins,)): Patched (Probability density squared)\n
        F_patch (array of size (nbins,)): Patched Mean Force\n
        OFV_num_patch (array of size (nbins,)): Patched numerator of "on the fly" variance \n
        OFE (array of size (nbins,)): On-the-fly error map. Statistical error of the mean force\n
        Aofe_progression (float): List of average OFE. If ofe_progression=False, all values will be the last calcuated average OFE. If ofe_progression=True, the values correspond to the average OFE after each patch.
    """

    #initialisa terms
    nbins = len(force_vector[0][0])
    PD_patch = np.zeros(nbins)
    PD2_patch = np.zeros(nbins)
    F_patch = np.zeros(nbins)
    OFV_num_patch = np.zeros(nbins)   
    
    Aofe_progression = np.zeros(len(force_vector))
    
    if ofe_progression == False:
    
        #Patch force terms    
        for i in range(len(force_vector)):
            PD_patch += force_vector[i][0]
            PD2_patch += force_vector[i][1]
            F_patch += np.multiply(force_vector[i][0] ,force_vector[i][2])
            OFV_num_patch += force_vector[i][3]
            
        F_patch = np.where(PD_patch > PD_limit, F_patch / PD_patch, 0)
        F = F_patch

        #Calculate error
        PD_patch2 = np.square(PD_patch)
        PD_diff = PD_patch2 - PD2_patch
        if use_weighted_st_dev == True: PD_ratio = np.where( PD_diff > 0, PD_patch2 / PD_diff, 0)   
        else: PD_ratio = np.where( PD_diff > 0, PD2_patch / PD_diff, 0)  
        OFV = np.multiply( np.where(PD_patch > 0, OFV_num_patch / PD_patch, 0) - np.square(F_patch) , PD_ratio)
        if Ftot_den_cutoff > 0: OFV *= get_cutoff_1D(Ftot_den=PD_patch, Ftot_den_cutoff=Ftot_den_cutoff)
        if non_exploration_penalty > 0: OFV = np.where(PD_patch > Ftot_den_cutoff, OFV, non_exploration_penalty**2) 
        OFE = np.where(OFV > 1E-5, np.sqrt(OFV), 0)
        AOFE = np.sum(OFE) / np.count_nonzero(OFE)
        Aofe_progression = np.ones_like(Aofe_progression) * AOFE
        
    if ofe_progression == True:

        #Patch force terms    
        for i in range(len(force_vector)):
            PD_patch += force_vector[i][0]
            PD2_patch += force_vector[i][1]
            F_patch += np.multiply(force_vector[i][0] ,force_vector[i][2])
            OFV_num_patch += force_vector[i][3]
            
            F = np.where(PD_patch > PD_limit, F_patch / PD_patch, 0)

            #Calculate error
            PD_patch2 = np.square(PD_patch)
            PD_diff = PD_patch2 - PD2_patch
            if use_weighted_st_dev == True: PD_ratio = np.where( PD_diff > 0, PD_patch2 / PD_diff, 0)   
            else: PD_ratio = np.where( PD_diff > 0, PD2_patch / PD_diff, 0)  
            OFV = np.multiply( np.where(PD_patch > 0, OFV_num_patch / PD_patch, 0) - np.square(F) , PD_ratio)
            if Ftot_den_cutoff > 0: OFV *= get_cutoff_1D(Ftot_den=PD_patch, Ftot_den_cutoff=Ftot_den_cutoff)
            if non_exploration_penalty > 0: OFV = np.where(PD_patch > Ftot_den_cutoff, OFV, non_exploration_penalty**2) 
            OFE = np.where(OFV > 1E-5, np.sqrt(OFV), 0)
            AOFE = np.sum(OFE) / np.count_nonzero(OFE)
            Aofe_progression[i] = AOFE
    
    F_patch = F    
        
    
    return PD_patch, PD2_patch, F_patch, OFV_num_patch, OFE, Aofe_progression

    

@njit
def patch_forces_ofe_AD(force_vector, grid, y, PD_limit=1E-10, use_weighted_st_dev=True, error_progression=True, Ftot_den_cutoff=-1, FES_cutoff=-1, non_exploration_penalty=0, set_fes_minima=None):
    """Takes in an array of force terms, patches them together and calculates the error, FES and average deviation. The force vector has to be in the format [Ftot_den, Ftot_den2, Ftot, ofv_num]. 

    Args:
        force_vector (array of shape (n,4, nbins)) where n is the number of independent force terms: The force terms are tipically outputted from the MFI_1D functions. They should be in the format: [Ftot_den (Probability density), Ftot_den2, Ftot (Mean Force), ofv_num (numerator of on the fly variance)]. 
        grid (array of shape (nbins, )): Grid that CV-space is defined on.
        y (array of shape (nbins, )): Reference surface that is used to calculate the absolute devaition.
        PD_limit (flaot, optional): Probability density values below the PD_limit will be approximated to be 0, to ensure the numerical stability of the algorithm. Defaults to 1E-10.
        use_weighted_st_dev (bool, optional): When set to True, the calculated error will be the weighted standard deviation ( var^0.5 ). When set to False, the calculated error will be the standard error ( (var/n_sample)^0.5 ). Defaults to True. (The standard devaition is expected to converge after enough time, while the standard error is expected to decrease as more datapoints are added.).
        error_progression (bool, optional): When set to True, the error (AOFE, AAD, volume ratio) will be calculated after every patch. When set to False, all forces are patched and the error is calculated only at the end. Default set to True.
		Ftot_den_cutoff (float, optional): Specifies the cutoff limit of the probability density. Values below the limit will be cut-off. Will only be applied if Ftot_den_cutoff is larger than 0. Defaults to -1.
		FES_cutoff (float, optional): Specifies the cutoff limit of the FES. Values above the limit will be cut-off. Will only be applied if FES is provided. Defaults to 23.
		non_exploration_penalty (float, optional): Turns zero-value error to the non_exploration_penalty value. This should be used in combination with the cutoff. If some part of CV space hasn't been explored, or has a FES value that is irrelevanlty high, the cutoff will set the error of that region to zero. If the non_exploration_penalty is larger than zero, the error of that region will take the value of the non_exploration_penalty instead of zero. Default is set to 0.
        set_fes_minima (str, optional): USed to specify how to set the minima of the FES. When set to "first", the first element of the fes will be set to 0, and the rest of the FES array will be shifted by the same amount. When set to None, the smalles element of the FES will be set to 0, and the rest of the FES array will be shifted by the same amount. Defauts is None.
        
    Returns:
        tuple: PD_patch, PD2_patch, F_patch, OFV_num_patch, FES, AD, aad_progression, OFE, Aofe_progression, volume_progression\n
        PD_patch (array of size (nbins,)): Patched Probability density\n
        PD2_patch (array of size (nbins,)): Patched (Probability density squared)\n
        F_patch (array of size (nbins,)): Patched Mean Force\n
        OFV_num_patch (array of size (nbins,)): Patched numerator of "on the fly" variance \n
        FES (array of size (nbins,)): Free Energy Surface from patched Mean Force\n
        AD (array of size (nbins,)): Absulute Deviation from FES to reference surface\n
        aad_progression (list): List of average AAD. If error_progression=False, all values will be the last calcuated average AAD. If error_progression=True, the values correspond to the average AAD after each patch.\n
        OFE (array of size (nbins,)): On-the-fly error. Statistical error of the mean force\n
        Aofe_progression (list): List of average OFE. If error_progression=False, all values will be the last calcuated average OFE. If error_progression=True, the values correspond to the average OFE after each patch.\n
        volume_progression (list): List of average volume ratio (Percentage of the grid that is not cutoff, e.g. with PD>0.1). If error_progression=False, all values will be the last calcuated average volume ratio. If error_progression=True, the values correspond to the average volume ratio after each patch.
    """
    
    #initialisa terms
    nbins = len(grid)
    cutoff = np.ones(nbins)
    PD_patch = np.zeros(nbins)
    PD2_patch = np.zeros(nbins)
    F_patch = np.zeros(nbins)
    OFV_num_patch = np.zeros(nbins)   
    
    Aofe_progression = np.zeros(len(force_vector))
    aad_progression = np.zeros(len(force_vector))
    volume_progression = np.zeros(len(force_vector))
    
    if error_progression == False:
    
        #Patch force terms    
        for i in range(len(force_vector)):
            PD_patch += force_vector[i][0]
            PD2_patch += force_vector[i][1]
            F_patch += np.multiply(force_vector[i][0] ,force_vector[i][2])
            OFV_num_patch += force_vector[i][3]
            
        F_patch = np.where(PD_patch > PD_limit, F_patch / PD_patch, 0)
        F = F_patch
        
        #Find FES get cutoff if applicable
        FES = intg_1D(F_patch, grid[1] - grid[0])        
        if set_fes_minima == "first": FES = FES - FES[0]        
        if Ftot_den_cutoff > 0 or FES_cutoff > 0: cutoff = get_cutoff_1D(Ftot_den=PD_patch, Ftot_den_cutoff=Ftot_den_cutoff, FES=FES, FES_cutoff=FES_cutoff)

        #Calculate error
        PD_patch2 = np.square(PD_patch)
        PD_diff = PD_patch2 - PD2_patch
        if use_weighted_st_dev == True: PD_ratio = np.where( PD_diff > 0, PD_patch2 / PD_diff, 0)   
        else: PD_ratio = np.where( PD_diff > 0, PD2_patch / PD_diff, 0)  
        OFV = np.multiply( np.where(PD_patch > 0, OFV_num_patch / PD_patch, 0) - np.square(F_patch) , PD_ratio)
        OFV *= cutoff
        if non_exploration_penalty > 0: OFV = np.where(cutoff > 0.5, OFV, non_exploration_penalty**2) 
        OFE = np.where(OFV > 1E-5, np.sqrt(OFV), 0)
        AOFE = np.sum(OFE) / np.count_nonzero(OFE)

        AD = np.absolute(FES - y)
        AD *= cutoff
        AAD = np.sum(AD) / np.count_nonzero(AD)

        Aofe_progression = np.ones_like(Aofe_progression) * AOFE
        aad_progression = np.ones_like(Aofe_progression) * AAD
        volume_progression = np.ones_like(Aofe_progression) * (np.count_nonzero(cutoff) / nbins)
        
    if error_progression == True:

        #Patch force terms    
        for i in range(len(force_vector)):
            PD_patch += force_vector[i][0]
            PD2_patch += force_vector[i][1]
            F_patch += np.multiply(force_vector[i][0] ,force_vector[i][2])
            OFV_num_patch += force_vector[i][3]
            
            F = np.where(PD_patch > PD_limit, F_patch / PD_patch, 0)
            
            #Find FES get cutoff if applicable
            FES = intg_1D(F, grid[1] - grid[0])
            if set_fes_minima == "first": FES = FES - FES[0]        
            if Ftot_den_cutoff > 0 or FES_cutoff > 0: cutoff = get_cutoff_1D(Ftot_den=PD_patch, Ftot_den_cutoff=Ftot_den_cutoff, FES=FES, FES_cutoff=FES_cutoff)

            #Calculate error
            PD_patch2 = np.square(PD_patch)
            PD_diff = PD_patch2 - PD2_patch
            if use_weighted_st_dev == True: PD_ratio = np.where( PD_diff > 0, PD_patch2 / PD_diff, 0)   
            else: PD_ratio = np.where( PD_diff > 0, PD2_patch / PD_diff, 0)  
            OFV = np.multiply( np.where(PD_patch > 0, OFV_num_patch / PD_patch, 0) - np.square(F) , PD_ratio)
            OFV *= cutoff
            if non_exploration_penalty > 0: OFV = np.where(cutoff > 0.5, OFV, non_exploration_penalty**2) 
            OFE = np.where(OFV > 1E-5, np.sqrt(OFV), 0)
            AOFE = np.sum(OFE) / np.count_nonzero(OFE)

            AD = np.absolute(FES - y)
            AD *= cutoff
            AAD = np.sum(AD) / np.count_nonzero(AD)            
            
            Aofe_progression[i] = AOFE
            aad_progression[i] = AAD
            volume_progression[i] = np.count_nonzero(cutoff) / nbins
    
    F_patch = F    
    
    if AOFE != AOFE:
        print("\n\n\n*************ATTENTION*****************\n THERE IS A --  NaN  --- SOMEWHERE IN THE OFE\n AOFE = " , AOFE,  "\n\n")
    
    return PD_patch, PD2_patch, F_patch, OFV_num_patch, FES, AD, aad_progression, OFE, Aofe_progression, volume_progression



@njit
def bootstrap_forw_back(grid, forward_force, backward_force, n_bootstrap, set_fes_minima=None):
    """Algorithm to determine bootstrap error

    Args:
        grid (array of shape (nbins,)): CV grid positions
        forward_force (list): collection of force terms (n * [Ftot_den, Ftot]) from forward transition
        backward_force (list): collection of force terms (n * [Ftot_den, Ftot]) from backward transition
        n_bootstrap (int): bootstrap iterations
        set_fes_minima (str, optional): USed to specify how to set the minima of the FES. When set to "first", the first element of the fes will be set to 0, and the rest of the FES array will be shifted by the same amount. When set to None, the smalles element of the FES will be set to 0, and the rest of the FES array will be shifted by the same amount. Defauts is None.

    Returns:
        list : [FES_avr, sd_fes, sd_fes_prog]\n
        FES_avr (array of shape (nbins,)): Average of all FES generated during the bootstrap algorithm.\n
        sd_fes (array of shape (nbins,)): Standard deviation of all FES generated during the bootstrap algorithm.\n
        sd_fes_prog (array of shape (n_bootstrap,)): Global average of the standard deviation after each bootstrap iteration. When this array converges, enough iterations have been performed. If it does not converge, move iterations are necessary.
    """

    #Define constants and lists
    nbins = len(grid)
    n_forces = len(forward_force)
    sd_fes_prog = np.zeros(n_bootstrap)    
    FES_avr = np.zeros(nbins)
    M2 = np.zeros(nbins)

    #Patch forces
    [Ftot_den, Ftot] = patch_forces(np.concatenate((forward_force, backward_force)))        

    for iteration in range(n_bootstrap):

        #Randomly choose forward forces and backward forces and save to force array
        force = np.zeros((int(n_forces * 2), 2, nbins)) 
        random_sample_index =  np.random.choice(n_forces-1, size=(2, n_forces))
        force[:n_forces] = forward_force[random_sample_index[0]]
        force[n_forces:] = backward_force[random_sample_index[1]]
                
        #patch forces to find average Ftot_den, Ftot and FES
        [Ftot_den, Ftot] = patch_forces(force)
        FES = intg_1D(Ftot, grid[1]-grid[0])
        if set_fes_minima == "first_value":  FES = FES - FES[0]
        else: FES = FES - min(FES)

        # calculate standard devaition using Welford’s method
        delta = FES - FES_avr
        FES_avr += delta/(iteration+1)
        delta2 = FES - FES_avr
        M2 += delta*delta2
        sd_fes = np.sqrt(M2 / (iteration))
        sd_fes_prog[iteration] = sum(sd_fes)/nbins
        

        # print progress
        if (iteration+1) % 50 == 0: print("Iteration:", iteration+1, "- sd:", round(sd_fes_prog[iteration],5) )
       
    return [FES_avr, sd_fes, sd_fes_prog]


@njit
def bootstrap_1D(grid, force_array, n_bootstrap, set_fes_minima=None):
    """Algorithm to determine bootstrap error

    Args:
        grid (array of shape (nbins,)): CV grid positions
        force_array (list): collection of force terms (n * [Ftot_den, Ftot]).
        n_bootstrap (int): bootstrap iterations

    Returns:
        FES_avr (array of shape (nbins,)): Average of all FES generated during the bootstrap algorithm.
        sd_fes (array of shape (nbins,)): Standard deviation of all FES generated during the bootstrap algorithm.
        sd_fes_prog (array of shape (n_bootstrap,)): Global average of the standard deviation after each bootstrap iteration. When this array converges, enough iterations have been performed. If it does not converge, move iterations are necessary.
    """
    
    #Define constants and lists
    nbins = len(grid)
    n_forces = len(force_array)
    sd_fes_prog = np.zeros(n_bootstrap)    
    FES_avr = np.zeros(nbins)
    M2 = np.zeros(nbins)

    for iteration in range(n_bootstrap):
        
        #Randomly choose forward forces and backward forces and save to force array
        force = np.zeros((int(n_forces ), 2, nbins)) 
        random_sample_index =  np.random.choice(n_forces-1, size=n_forces)      
        force = force_array[random_sample_index]
     
        #patch forces to find average Ftot_den, Ftot and FES
        [Ftot_den, Ftot] = patch_forces(force)
        FES = intg_1D(Ftot, grid[1]-grid[0])
        if set_fes_minima == "first_value":  FES = FES - FES[0]
        else: FES = FES - min(FES)
        
        # calculate standard devaition using Welford’s method
        delta = FES - FES_avr
        FES_avr += delta/(iteration+1)
        delta2 = FES - FES_avr
        M2 += delta*delta2
        sd_fes = np.sqrt(M2 / (iteration))
        sd_fes_prog[iteration] = sum(sd_fes)/nbins
        
        # print progress
        if (iteration+1) % 50 == 0: print("Iteration:", iteration+1, "- sd:", round(sd_fes_prog[iteration],5) )
       
    return [FES_avr, sd_fes, sd_fes_prog]


def plot_recap(X, FES, Ftot_den, ofe, ofe_history, time_history, y_ref=None, FES_lim=40, ofe_lim = 10, error_log_scale = 1,
               cv = "CV", time_axis_name="simulation time"):
    """Plot result of 1D MFI algorithm. 1. FES, 2. varinace_map, 3. Cumulative biased probability density, 4. Convergece of variance.

    Args:
        X (array): gird
        FES (array): Free energy surface
        Ftot_den (array): _description_
        ofe (array): Cumulative biased probability density
        ofe_history (list): on the fly estimate of the local convergence
        time_history (list): List of time or simulation number corresponding to error history.
        FES_lim (int, optional): Upper energy value in FES plot. Defaults to 40.
        ofe_lim (int, optional): Upper error value in FES plot. Defaults to 10.
        error_log_scale (boolean, optional): Option to make error_conversion plot with a log scale. 1 for log scale. Defaults to 1.
    """
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    #plot ref f
    if hasattr(y_ref, "__len__") != True: y = 7*X**4 - 23*X**2
    else: y = y_ref
    y = y - min(y)  
    axs[0, 0].plot(X, y, label="Reference", color="red", alpha=0.3);
    
    axs[0, 0].plot(X, FES, label="FES");
    axs[0, 0].set_ylim([0, FES_lim])
    axs[0, 0].set_ylabel('F(' + cv + ') [kJ/mol]')
    axs[0, 0].set_xlabel(cv)
    axs[0, 0].set_title('Free Energy Surface')
    axs[0, 0].set_xlim(np.min(X),np.max(X))
    axs[0, 0].legend(fontsize=10)

    axs[0, 1].plot(X, ofe);
    axs[0, 1].plot(X, np.zeros(len(X)), color="grey", alpha=0.3);
    axs[0, 1].set_ylabel('Mean Force Error [kJ/(mol)]')
    axs[0, 1].set_xlabel(cv)
    axs[0, 1].set_title('Local Error Map')
    # axs[0, 1].set_ylim(-0.1, ofe_lim)
    axs[0, 0].set_xlim(np.min(X),np.max(X))

    

    axs[1, 0].plot(X, Ftot_den)
    axs[1, 0].set_ylabel('Count [relative probability]')
    axs[1, 0].set_xlabel(cv)
    axs[1, 0].set_title('Total Probability density')
    axs[0, 0].set_xlim(np.min(X),np.max(X))
    # axs[1, 0].set_yscale('log')


    axs[1, 1].plot(time_history, ofe_history);
    # axs[1, 1].plot([time/1000 for time in time_history], np.zeros(len(ofe_history)), color="grey", alpha=0.3);
    axs[1, 1].set_ylabel('Average Mean Force Error [kJ/(mol)]')
    axs[1, 1].set_xlabel(time_axis_name)
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
    """Turns all zero elements of an array into numpy NaN (Not a Number)

    Args:
        input_array (array of size (x,)): Input array.

    Returns:
        array: Input array with zero elements turned to Nan.
    """    
    output_array = np.zeros_like(input_array)
    for ii in range(len(input_array)):
        if input_array[ii] == 0: output_array[ii] = np.nan
        else: output_array[ii] = input_array[ii]
    return output_array


def print_progress(iteration, total, bar_length=50, variable_name='progress variable' , variable=0):
    progress = (iteration / total)
    arrow = '*' * int(round(bar_length * progress))
    spaces = ' ' * (bar_length - len(arrow))
    print(f'\r|{arrow}{spaces}| {int(progress * 100)}% | {variable_name}: {variable}', end='', flush=True)