import glob
import os
import subprocess
import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle
from numba import jit, njit

sys.path.append("/home/ucecabj/Desktop/pyMFI_git/pyMFI")
import MFI1D as MFI1D


def load_HILLS_1D_RT(hills_name="HILLS", hills_analysed=0):
    """Load 1-dimensional hills data for analysing simulations in real time (includes time, position_x, position_y, hills_parameters)
			Takes into account the hills aready analysed. Will only return hills data  after the already analysed hills. When only passing partial hills data to MFI, the previous hills data has to be passed as a starting bias with F_static.
    Args:
        hills_name (str, optional): Name of hills file. Defaults to "HILLS".
        hills_analysed (int, optional): Number of hills already analysed. Will only return the hills data after the analysed hills. Defaults to 0.

    Returns:
        np.array: Hills data with length equal to the total number of hills. Information: [time [ps], position [nm], MetaD_sigma [nm], MetaD_height [nm], MetaD_biasfactor]
    """

    #create copy of HILLS file. Same name with _cp ending.
    hills_name_cp = hills_name + "_cp"
    os.system("cp " + hills_name + " " + hills_name_cp)
    
    #Find the number of lines starting with #!
    count_, count_done = 0, 0
    cmd = ['head', '-n', "10" , hills_name_cp]
    output = subprocess.check_output(cmd).decode()
    output  = [line.split() for line in output.strip().split('\n')]
    while count_done == 0:
        if output[count_][0] == "#!": count_ += 1
        else: count_done = 1
        
        #Safety net incase there are more than 10 lines starting with "#!"
        if count_ > len(output)-1: 
            new_length_tobechecked = str(int(len(output) * 3))
            cmd = ['head', '-n', new_length_tobechecked , hills_name_cp]
            output = subprocess.check_output(cmd).decode()
            output  = [line.split() for line in output.strip().split('\n')]
    
    # Read Hills file                
    if hills_analysed == 0:
              
        cmd = ['cat', hills_name_cp]
        output = subprocess.check_output(cmd).decode()
        output  = [line.split() for line in output.strip().split('\n')]
        output = output[count_:-1] #remove the first few lines starting with "#!". remove the last line incase position file is not complete or last line is not complete.
        # if len(output[-1]) != len(output[-2]): output = output[:-1]   # if last line is not complete, remove it. THIS LINE WAS COMMENTED OUT; line above should fix it.
        
        hills = np.array(output, dtype=float)   
        hills0 = np.array(hills[0])
        hills0[3] = 0
        hills = np.concatenate(([hills0], hills))   # Add a hill at time 0 with height 0.   
               
    else:
        cmd = ['tail', '-n', "+" + str(hills_analysed+1+count_), hills_name_cp]  # loads file from line = hills_analysed [ignore previous hills] +1 [jump to new line to get new hill] + count_ [ignore the first few lines starting with "#!"]
        output = subprocess.check_output(cmd).decode()
        output  = [line.split() for line in output.strip().split('\n')]
        output = output[:-1]  #remove the last line incase position file is not complete
        
        hills = np.array(output, dtype=float)   
        
    os.system("rm " + hills_name_cp)
    return hills

def load_position_1D_RT(position_name="position", hills_analysed=0, stride=10):  #assumes a stride of 10
    """Load 1-dimensional position data for analysing simulations in real time.
			Takes into account the hills aready analysed. Will only return position data after the already analysed positions (found by multiplying hills with stride). When only passing partial hills data to MFI, the previous hills data has to be passed as a starting bias with F_static.
    Args:
        hills_name (str, optional): Name of hills file. Defaults to "HILLS".
        hills_analysed (int, optional): Number of hills already analysed. Will only return the hills data after the analysed hills. Defaults to 0.

    Returns:
        np.array: Hills data with length equal to the total number of hills. Information: [time [ps], position [nm], MetaD_sigma [nm], MetaD_height [nm], MetaD_biasfactor]
    """

    #create copy of position file. Same name with _cp ending.
    position_name_cp = position_name + "_cp"
    os.system("cp " + position_name + " " + position_name_cp)
    
    #Find the number of lines starting with #!
    count_, count_done = 0, 0
    cmd = ['head', '-n', "10" , position_name_cp]
    position = subprocess.check_output(cmd).decode()
    position  = [line.split() for line in position.strip().split('\n')]
    while count_done == 0:
        if position[count_][0] == "#!": count_ += 1
        else: count_done = 1
        
        #Safety net incase there are more than 10 lines starting with "#!"
        if count_ > len(position)-1: 
            new_length_tobechecked = str(int(len(position) * 3))
            cmd = ['head', '-n', new_length_tobechecked , position_name_cp]
            position = subprocess.check_output(cmd).decode()
            position  = [line.split() for line in position.strip().split('\n')]
            

    # Read position file                
    if hills_analysed == 0:
        cmd = ['cat', position_name_cp]
        position = subprocess.check_output(cmd).decode()
        position  = [line.split() for line in position.strip().split('\n')]
        position = position[count_:-1] #remove the first few lines starting with "#!". remove the last line incase last line is not complete.
        # if len(position[-1]) != len(position[-2]): position = position[:-1]  # if last line is not complete, remove it. THIS LINE WAS COMMENTED OUT; line above should fix it.
        
        position = np.array(position, dtype=float)   
        position = position[:, 1]    
     
               
    else:
        cmd = ['tail', '-n', "+" + str(int((hills_analysed+1)*int(stride)+1+count_)), position_name_cp]  # loads file from line = (hills_analysed+1)*int(stride) [ignore previous position for previous hills] +1 [jump to new line to get new hill] + count_ [ignore the first few lines starting with "#!"]
        position = subprocess.check_output(cmd).decode()
        position  = [line.split() for line in position.strip().split('\n')]
        position = position[:-1] #remove the last line incase last line is not complete.
        # if len(position[-1]) != len(position[-2]): position = position[:-1] # if last line is not complete, remove it. THIS LINE WAS COMMENTED OUT; line above should fix it.
        position = np.array(position, dtype=float)   
        position = position[:, 1]   
        
    os.system("rm position_cp")
    return position

   
@njit
def MFI_1D_fast_RT(HILLS="HILLS", position="position", stride=10, bw=1, kT=1, min_grid=-2, max_grid=2, nbins=201,
           WellTempered=1, nhills=-1, periodic=0, Ftot_den_limit = 1E-10, F_static = np.zeros(123)):
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
    
    
    for _initialise_ in [1]:
        grid = np.linspace(min_grid, max_grid, nbins)
        grid_ext = 0.25 * (max_grid-min_grid)        
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

        # Definition Gamma Factor, allows to switch between WT and regular MetaD
        if WellTempered < 1: Gamma_Factor = 1
        else: Gamma_Factor = (HILLS[1, 4] - 1) / (HILLS[1, 4])
    
            
    for i in range(total_number_of_hills):
                
        #Get position data of window        
        s = HILLS[i, 1]  # centre position of Gaussian
        sigma_meta2 = HILLS[i, 2] ** 2  # width of Gaussian
        height_meta = HILLS[i, 3] * Gamma_Factor  # Height of Gaussian
        data = position[i * stride: (i + 1) * stride]  # positons of window of constant bias force.
        periodic_hills = MFI1D.find_periodic_point_numpy(np.array([s]), min_grid, max_grid, periodic, grid_ext)
        periodic_positions = MFI1D.find_periodic_point_numpy(data, min_grid, max_grid, periodic, grid_ext)

        #Calculate forces of window
        [pb_t, Fpbt, Fbias_window] = MFI1D.window_forces(periodic_positions, periodic_hills, grid, sigma_meta2, height_meta, kT, const, bw2, Ftot_den_limit)
        Fbias += Fbias_window                   
        dfds = np.where(pb_t > Ftot_den_limit, Fpbt / pb_t , 0) + Fbias - F_static
        Ftot_num += np.multiply(pb_t, dfds)
        
        # Calculate total force
        Ftot_den = Ftot_den + pb_t  # total probability density  
        Ftot = np.where(Ftot_den > Ftot_den_limit, Ftot_num / Ftot_den, 0)

        # terms for error calculation
        Ftot_den2 += np.square(pb_t)  # sum of (probability densities)^2
        ofv_num += np.multiply(pb_t, np.square(dfds))   # sum of (weighted mean force of window)^2
        
    return grid, Ftot_den, Ftot_den2, Ftot, ofv_num, Fbias

