import os
import sys
import subprocess
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pickle
from numba import jit, njit

import random
import time
from datetime import datetime, timedelta 
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from dataclasses import dataclass, field

####  ---- Run Langevin Simulation with 1 CV (1D)  ----  ####

@dataclass
class RunLangevin1D:
    """Class for running langevin dynamics in 1D with some bias potential."""
    
    analytical_function: str = "7*x^4-23*x^2"
    n_steps: int = 1_000_000
    initial_position: float = None
    pl_grid: np.ndarray = field(default = None)
    pl_min: float = -3
    pl_max: float = 3
    pl_n: int = 201 #what grid does plumed create? should this be 200 to be compatible with np.linspace(-3,3,201)?
    periodic: bool = False

    temperature: float = 1
    time_step: float = 0.005
    friction: float = 1
    
    metad_width: float = 0.1
    metad_height: float = None
    biasfactor: float = 10.0
    metad_pace: int = 200
    position_pace: int = None
    n_pos_per_window: int = 10
    
    hp_centre: float = None; hp_kappa: float = None
    lw_centre: float = None; lw_kappa: float = None
    uw_centre: float = None; uw_kappa: float = None
    
    external_bias_file: str = None
    file_extension: str = ""
    
    terminal_input: str = "plumed pesmd < input >/dev/null 2>&1" 
    start_sim: bool = True
    print_info: bool = True
    info: str = None
    save_simulation_data_file: str = None
    
    def __post_init__(self):
        # Get the grid for plumed
        if self.pl_grid is None and (self.pl_min is None or self.pl_max is None or self.pl_n is None): print("\n ***** Please either plumed grid or plumed min, max, and nbins. ***** \n")
        elif self.pl_grid is None: self.pl_grid = np.linspace(self.pl_min, self.pl_max, self.pl_n)
        else: self.pl_min, self.pl_max, self.pl_n = self.pl_grid[0], self.pl_grid[-1], len(self.pl_grid)
        self.pl_l = self.pl_max - self.pl_min
            
        # Set the metadynamics parameters and initial position
        if self.metad_height is not None and self.metad_height > 0: 
            if self.position_pace is None: self.position_pace = int(round(self.metad_pace / self.n_pos_per_window))
            assert round(self.metad_pace / self.position_pace,4) == round(self.n_pos_per_window,4), "The metad_pace, position_pace, and n_pos_per_window do not match."
        elif self.position_pace is None: self.position_pace = 10
        self.periodic_boundaries = str(self.pl_min) + "," + str(self.pl_max) if self.periodic else "NO"
        if self.initial_position is None: self.initial_position = get_random_move(self.pl_min + self.pl_l/10, self.pl_max - self.pl_l/10) if self.periodic is False else get_random_move(self.pl_min, self.pl_max)

        # Write the input file or command
        input_file_text = f"nstep {self.n_steps}\n"  \
                            f"ipos {self.initial_position}\n" \
                            f"temperature {self.temperature}\n"  \
                            f"tstep {self.time_step}\n"  \
                            f"friction {self.friction}\n" \
                            f"dimension 1"
        input_file_text += f"\nperiodic on min {self.pl_min} max {self.pl_max}" if self.periodic else "\nperiodic false"
        with open("input" ,"w") as f: print( input_file_text,file=f)

        # Write the plumed file
        plumed_file_text = f"p: DISTANCE ATOMS=1,2 COMPONENTS\n"  \
                            f"ff: MATHEVAL ARG=p.x FUNC=({self.analytical_function}) PERIODIC={self.periodic_boundaries}\n"  \
                            f"bb: BIASVALUE ARG=ff\n"

        # Metadynamics bias. To activate, the height needs to be a positive number
        if self.metad_height is not None and self.metad_height > 0: plumed_file_text += f"METAD ARG=p.x SIGMA={self.metad_width} HEIGHT={self.metad_height} BIASFACTOR={self.biasfactor} GRID_MIN={self.pl_min} GRID_MAX={self.pl_max} GRID_BIN={self.pl_n-1} PACE={self.metad_pace} TEMP={self.temperature * 120} FILE=HILLS{self.file_extension}\n"
        # Harmonic potential bias. To activate, the force constant (kappa) needs to be a positive number
        if self.hp_kappa is not None: plumed_file_text += f"RESTRAINT ARG=p.x AT={self.hp_centre} KAPPA={self.hp_kappa} LABEL=restraint\n"
        # Lower wall bias. To activate, the force constant (kappa) needs to be a positive number
        if self.lw_kappa is not None: plumed_file_text += f"LOWER_WALLS ARG=p.x AT={self.lw_centre} KAPPA={self.lw_kappa} LABEL=lowerwall\n"
        # Upper wall bias. To activate, the force constant (kappa) needs to be a positive number
        if self.uw_kappa is not None: plumed_file_text += f"UPPER_WALLS ARG=p.x AT={self.uw_centre} KAPPA={self.uw_kappa} LABEL=upperwall\n"
        # External bias. To activate, the file name needs to be given
        if (self.external_bias_file is not None) and (self.external_bias_file != ""): plumed_file_text += f"EXTERNAL ARG=p.x FILE={self.external_bias_file} LABEL=external \n"
        # Print position of system. 
        plumed_file_text += f"PRINT FILE=position{self.file_extension} ARG=p.x STRIDE={self.position_pace}"
        # Write the plumed file
        with open("plumed.dat" ,"w") as f: print(plumed_file_text ,file=f)

        # Start the simulation
        if self.start_sim == True: self.start_simulation()
                
    def start_simulation(self):
        
        # Start the simulation
        process = subprocess.Popen(self.terminal_input, shell=True, preexec_fn=os.setsid)
        
        # Write simulation info text. If self.print_info, print the information about the simulation
        if self.print_info: start = time.time()
        if self.print_info or self.save_simulation_data_file is not None:
            info = f"\nRunning Langevin dynamics: n_steps={self.n_steps:,}, ipos={self.initial_position}, Pos_t={self.position_pace}, T={self.temperature}, t_Tot={self.n_steps*self.time_step/1000:,.2f}ns"
            if self.metad_height is not None: info += f"\nsigma={self.metad_width}, h={self.metad_height}, bf={self.biasfactor}, Gaus_t={self.metad_pace}"
            if self.hp_kappa is not None: info += f"\nHarmonic potential: centre={self.hp_centre}, kappa={self.hp_kappa}"
            if self.lw_kappa is not None: info += f"\nLower wall: centre={self.lw_centre}, kappa={self.lw_kappa}"
            if self.uw_kappa is not None: info += f"\nUpper wall: centre={self.uw_centre}, kappa={self.uw_kappa}"
            if self.external_bias_file != "": info += f"\nStatic bias used: {self.external_bias_file}"
                        
            if self.print_info: print(info, "\n") 
        
        # If self.print_info, print a progress bar, else wait for the simulation to finish
        if self.print_info:
            tot_pos = self.n_steps / self.position_pace               
            while process.poll() is None:
                time.sleep(1)
                n_lines = count_lines("position"+self.file_extension)
                live_print_progress(start, n_lines, tot_pos, bar_length=50, variable_name='Simulated time', variable=round(n_lines*self.time_step*self.position_pace/1_000,4))
        else: process.wait()
        if self.print_info: print(f"\nLangevin dynamics finished in {format_seconds(time.time()-start)}.")
        
        # if self.save_simulation_data_file is given save the data
        if self.save_simulation_data_file is not None:
            [HILLS, pos] = read_data(n_pos_per_window=self.n_pos_per_window, extension=self.file_extension)
            external_bias = read_plumed_grid_file(self.external_bias_file) if self.external_bias_file != "" else [None]
            save_pkl([HILLS, pos, external_bias, info], self.save_simulation_data_file)        
            
    def start_sim_return_process(self, print_info=False):
        # Start the simulation and return the process so that it can be interupted later.
        process = subprocess.Popen(self.terminal_input, shell=True, preexec_fn=os.setsid)
        if print_info: print("Simulation started with pid:", process.pid, os.getpgid(process.pid))
        return process
            
def wait_for_HILLS(new_hills, hills_analysed=0, hills_path="HILLS", return_nhills=False, lines_with_coments="default", periodic=False, sleep_between_checks=0.1):
    wait = True
    counter = 1
    if lines_with_coments == "default": lines_with_coments = 3 if periodic == False else 5
    while wait == True:
        
        result = subprocess.run(['wc', '-l', hills_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # The output will be in the format 'number filename', so split on whitespace and take the first component
        if result.returncode == 0: 
            line_count = int(result.stdout.split()[0])   # Check if the command was successful
            if (line_count - hills_analysed - lines_with_coments) > new_hills: wait = False 
            else: time.sleep(sleep_between_checks)        
        else: 
            counter += 1
            if counter > 3: raise Exception(f"Error counting lines in {hills_path}: {result.stderr}")
            else: time.sleep(sleep_between_checks*5)

    if return_nhills: return line_count - hills_analysed - lines_with_coments
    else: return

def wait_for_positions(new_positions, n_pos_analysed=0, position_path="position", return_n_pos=False, lines_with_coments="default", periodic=False, sleep_between_checks=0.1):
    wait = True
    counter = 1
    if lines_with_coments == "default": lines_with_coments = 1 if periodic == False else 3
    while wait == True:
        
        result = subprocess.run(['wc', '-l', position_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # The output will be in the format 'number filename', so split on whitespace and take the first component
        if result.returncode == 0: 
            line_count = int(result.stdout.split()[0])   # Check if the command was successful
            if (line_count - n_pos_analysed - lines_with_coments) > new_positions: wait = False 
            else: time.sleep(sleep_between_checks)        
        else: 
            counter += 1
            if counter > 10: raise Exception(f"Error counting lines in {position_path}: {result.stderr}")
            else: time.sleep(sleep_between_checks*5)

    if return_n_pos: return line_count - return_n_pos - lines_with_coments
    else: return

def get_plumed_grid_1D(X, pl_min=None, pl_max=None, print_info=False, periodic=False):

    x_min, x_max, nx = np.min(X), np.max(X), len(X)
    dx = X[1] - X[0]
    
    if pl_min is None and periodic is False: pl_min = x_min - 1
    if pl_max is None and periodic is False: pl_max = x_max + 1
    if pl_min is None and pl_max is None and periodic is True: 
        
        pl_x = X
        pl_x_min_new, pl_x_max_new, pl_nx = x_min, x_max, nx
        nx_low_extra, nx_up_extra = 0, 0
    
    else: 
        diff_x_low = x_min - pl_min
        diff_x_up = pl_max - x_max
        
        nx_low_extra = int(np.ceil(diff_x_low / dx))
        nx_up_extra = int(np.ceil(diff_x_up / dx))
        
        pl_x_min_new = x_min - nx_low_extra * dx
        pl_x_max_new = x_max + nx_up_extra * dx
        
        pl_nx = nx + nx_low_extra + nx_up_extra
        
        pl_x = np.linspace(pl_x_min_new, pl_x_max_new, pl_nx)
        
        if print_info: print("The MFI grid was: \nxmin=", round(x_min,2), " xmax=", round(x_max,2), " nx=", nx)
        if print_info: print("\nThe new PLUMED grid is: \npl_xmin=", round(pl_x_min_new,2), " pl_xmax=", round(pl_x_max_new,2), " pl_nx=", pl_nx) 
    
    return [pl_x, pl_x_min_new, pl_x_max_new, pl_nx, [nx_low_extra, nx_up_extra]]

def make_external_bias_1D(grid_mfi, FES=None, Bias=None, Bias_sf=1, gaus_filter_sigma=None, FES_cutoff=None, pl_min=None, pl_max=None, periodic=False, file_name_extension="", return_array=None, cv_name="p.x"):
    
    #Get plumed grid
    pl_x, pl_min, pl_max, pl_nx, pl_extra = get_plumed_grid_1D(grid_mfi, pl_min=pl_min, pl_max=pl_max, print_info=False, periodic=periodic)
    assert np.sum(pl_x[pl_extra[0]:-pl_extra[1]] - grid_mfi) < 1E-3, "The plumed grid does not match the MFI grid. Please check the grid creation."
    
    #Find plumed Bias
    pl_Bias = np.zeros(pl_nx)
    if FES is not None: 
        if FES_cutoff is not None: FES = np.where(FES < FES_cutoff, FES, FES_cutoff)
        pl_Bias[pl_extra[0]:-pl_extra[1]] = -FES - np.min(-FES)
    elif Bias is not None: 
        if FES_cutoff is not None:
            max_Bias = np.max(Bias)
            if max_Bias > FES_cutoff: Bias = np.where(Bias < (max_Bias - FES_cutoff), 0, Bias - (max_Bias - FES_cutoff))
        pl_Bias[pl_extra[0]:-pl_extra[1]] = Bias
    
    # Modify Bias by scaling factor and/or gaussian filter
    if gaus_filter_sigma is not None: 
        if periodic: pl_Bias = gaussian_filter(pl_Bias, sigma=gaus_filter_sigma, mode='wrap')
        else: pl_Bias = gaussian_filter(pl_Bias, sigma=gaus_filter_sigma)
    if Bias_sf > 0 and Bias_sf != 1: pl_Bias *= Bias_sf
    
    #Find gradient of Bias
    pl_F_bias = np.gradient(pl_Bias, pl_x[1]-pl_x[0])
    
    # get correct format to save in file "external_bias.dat" file
    if periodic: 
        pl_x = pl_x[:-1]
        pl_Bias = pl_Bias[:-1]
        pl_F_bias = pl_F_bias[:-1]
        if round(pl_min,1) + np.pi == 0.0: pl_min, pl_max = "-pi", "pi"
        periodic = "true"
    else: periodic = "false"

    # change the plumed nbins to the correct format (irrespective of periodicity)
    pl_nx = pl_nx - 1
    
    #Save to external_bias.dat file
    external_bias_vector = np.array([pl_x, pl_Bias, pl_F_bias]).T       
    head_text = f"#! FIELDS {cv_name} external.bias der_{cv_name}\n#! SET min_{cv_name} {pl_min}\n#! SET max_{cv_name} {pl_max}\n#! SET nbins_{cv_name} {pl_nx}\n#! SET periodic_{cv_name} {periodic}"
    np.savetxt(f"external_bias{file_name_extension}.dat", external_bias_vector, fmt="%.8f", delimiter="   ", header=head_text, comments="")

    if return_array != None: return [pl_Bias[pl_extra[0]:-pl_extra[1]], pl_F_bias[pl_extra[0]:-pl_extra[1]], f"external_bias{file_name_extension}.dat"]

def find_total_bias_from_hills(grid, HILLS, nhills=-1, periodic=False):
    #Specify grid variables
    nbins = len(grid)
    grid_min, grid_max = np.min(grid), np.max(grid)
    grid_length = grid_max - grid_min
    periodic_range = 0.25*grid_length

    # initialise Bias 
    Bias = np.zeros(nbins)

    # Specify Gaussian parameters
    sigma_meta2 = HILLS[1, 2]**2
    Gamma_Factor = (HILLS[1, 4] - 1) / (HILLS[1, 4])
    assert HILLS[2, 2]**2 == sigma_meta2, "The sigma_meta2 is not the same for all hills. Please check the code (account for non_WT MetaD)."
    assert HILLS[2, 4] == HILLS[1, 4], "The Gamma_Factor is not the same for all hills. Please check the code (account for non_WT MetaD)."
    if nhills == -1: nhills = len(HILLS)
    
    #Cycle over hills
    for i in range(nhills):
        # get position and height of Gaussian. Add the Gaussian to the Bias
        pos_meta, height_meta = HILLS[i, 1], HILLS[i, 3] * Gamma_Factor # centre position of Gaussian, and height of Gaussian
        pos_meta = find_periodic_point(pos_meta, grid_min, grid_max, periodic, periodic_range) if periodic else [pos_meta]
        for pos_m in pos_meta: Bias += height_meta * np.exp( - np.square(grid - pos_m) / (2*sigma_meta2) )
   
    return Bias
    
def get_random_move(grid_min, grid_max):
    random.seed()
    index = random.randint(0, 500)
    coordinate = grid_min + ( (grid_max - grid_min) / 500) * index
    return round(coordinate,2)

def set_up_folder(folder_path, remove_folder=False, print_info=False, copy_files_path=[]):
    
    if os.path.isdir(folder_path): 
        if remove_folder is False: 
            if print_info:print(f"*** Folder already exists: {folder_path} *** Moving into folder.") 
            os.chdir(folder_path)
            return os.getcwd()
        else: 
            if print_info: print(f"*** Removing folder: {folder_path} *** Creating a new folder and moving into it.")
            shutil.rmtree(folder_path)
    os.mkdir(folder_path)
    os.chdir(folder_path)
    
    if len(copy_files_path) > 0:
        for file in copy_files_path:
            if os.path.exists(file): shutil.copy(file, folder_path)
            else: print(f"*** File {file} does not exist. ***")
            
    return os.getcwd()

####  ---- Read simulation Data  ----  ####

for _read_simulation_data_ in [0]:

    def count_lines(filename):
        with open(filename, 'r') as file:
            return sum(1 for line in file if not line.startswith('#'))

    def read_data(hills_path = "HILLS", pos_path = "position", n_pos_analysed = 0, n_pos_per_window = 10, metad_h=1, extension=""):
        
        # add extension to the file names
        if extension != "": hills_path, pos_path = hills_path + extension, pos_path + extension
        
        #### ~~~~~ Copy hills and position Data ~~~~~ ####
        
        if n_pos_per_window is None: n_pos_per_window = 10
        n_pos_per_window = int(n_pos_per_window)
        n_pos_analysed = int(n_pos_analysed)
        
        # make a copy of the position and hills file, and read the data from the copy, ignoring lines starting with #. If copy already exists, remove it first.
        pos_path_cp = pos_path + "_cp"
        if os.path.exists(pos_path_cp): os.system("rm " + pos_path_cp)
                
        if metad_h is not None and metad_h > 0: 
            hills_path_cp = hills_path + "_cp"
            if os.path.exists(hills_path_cp): os.system("rm " + hills_path_cp)
        
        os.system("cp " + pos_path + " " + pos_path_cp)        
        if metad_h is not None and metad_h > 0: os.system("cp " + hills_path + " " + hills_path_cp)        
        
        #### ~~~~~ Load position Data ~~~~~ ####

        with open(pos_path_cp, 'r') as file: pos = [line.split() for line in file if not line.startswith("#")]
        
        # check if the position data is empty
        assert len(pos) > n_pos_per_window, f"The position data: {pos = } is empty ({len(pos) = }) (ec1)"
        
        # the first position is at time 0 is removed. This is done because the position is preset and to have the number of positions being a const multiple (=n_pos_per_window) of the number of hills.
        # n_pos_per_window is the same for every hill, but at the start there are usually n_pos_per_window+1 positions. 
        pos_time_00, pos_00 = pos[0][0], pos[0][1]
        pos = pos[1:]

        # remove the last line if it is not complete
        if len(pos[-1]) < len(pos[-2]): pos = pos[:-1]  
        last_index_pos_line = len(pos[-2])-1
        if len(pos[-1][last_index_pos_line]) < len(pos[-2][last_index_pos_line]) / 2: pos = pos[:-1]
        assert len(pos) > n_pos_per_window, f"The position data: {pos = } is empty ({len(pos) = }) (ec2) "

        # turn position data into a numpy array
        pos = np.array(pos, dtype=float)

        # check if there are more positions than the number of positions already analysed
        if len(pos) <= n_pos_analysed: 
            print(f"The position data is smaller than the number of positions already analysed {len(pos) = } <= {n_pos_analysed = }")
            return None, None
            
        # cut position data, removing the number of positions already analysed
        if n_pos_analysed > 0: pos = pos[n_pos_analysed:]
        assert len(pos) > n_pos_per_window, f"The position data: {pos = } is empty ({len(pos) = }) (ec3)"
            
        #### ~~~~~ Load HILLS Data ~~~~~ ####
        
        if metad_h is not None and metad_h > 0: 
            
            with open(hills_path_cp, 'r') as file: hills = [line.split() for line in file if not line.startswith("#")]

            # check if the hills data is empty
            assert len(hills) > 1, f"The hills data: {hills = } is empty ({len(hills) = }) (ec4)"
            
            # remove the last line if it is not complete
            if len(hills[-1]) < len(hills[-2]): hills = hills[:-1]  
            last_index_hills_line = len(hills[-2])-1
            if len(hills[-1][last_index_hills_line]) < len(hills[-2][last_index_hills_line]) / 2: hills = hills[:-1]
            
            # if the number of positions analysed is 0 (sim time=0 at beginning), add dummy hills at time 0: [time, CV, sigma_MetaD, height_MetaD, biasfactor_MetaD] = [pos_time_00, 0.0, sigma from first line, 0, biasfactor from first line]
            if n_pos_analysed == 0: hills = [[pos_time_00, pos_00, hills[0][2], 0, hills[0][4]]] + hills
            
            # turn hills data into a numpy array
            hills = np.array(hills, dtype=float)
            
            #### ~~~~~ Data Loadin Finished  --> Now cut data where necessary ~~~~~ ####
            
            # find the time between positions and hills
            dt_pos, dt_hills = float(pos[1][0]) - float(pos[0][0]), float(hills[2][0]) - float(hills[1][0])
            
            # make sure that dt_hills == dt_pos * n_pos_per_window
            assert round(dt_hills, 4) == round(dt_pos * n_pos_per_window, 4), f"The pace of the position (*n_pos_per_window) and the hills pace don't match: dt_pos * n_pos_per_window = {dt_pos} * {n_pos_per_window} = {dt_pos * n_pos_per_window} != {dt_hills = }"
        
            #### ~~~~~ If n_pos_analysed > 0 find the data after n_pos_analysed ~~~~~ ####        
            if n_pos_analysed > 0:
                
                # find poitions sim start time and expected hills start time
                t_pos_0 = pos[0][0]
                t_hills_0 = round(t_pos_0 - dt_pos, 4)  # the hill corresponding to the first position is deposited at t_pos_0 - dt_pos
            
                # find the index of the first hill. 
                t_hills_slided = np.array([round(hills[i,0] - t_hills_0, 4) for i in range(len(hills))])
                index = np.where(t_hills_slided == 0.0)
                assert len(index[0]) == 1, f"Could not find the (first) hill corresponding to the first position: t_pos_0 - dt_pos = {t_pos_0} - {dt_pos} = {t_hills_0 = } in the hills file: {hills_path} \n{hills[:5] = } \n{hills[-5:] = }\n{t_hills_slided = }"
            
                # cut hills data, removing the hills before the first hill corresponding to the first position
                hills = hills[index[0][0]:]
            
            # check if t_pos_0 - dt_pos == t_hills_0
            assert round(pos[0][0] - dt_pos, 4) == round(hills[0][0], 4), f"The first time in the position data and the first time in the hills data don't match: pos[0][0] - dt_pos = {pos[0][0]} - {dt_pos} = {pos[0][0] - dt_pos} != {hills[0][0] = }"

            #### ~~~~~ Cut from the end to make len(hills) * n_pos_per_window = len(pos) ~~~~~ ####        

            len_hills, len_pos = len(hills), len(pos)
            if len_hills * n_pos_per_window != len_pos:    
                
                #  if there are too many hills lines, remove extra hills lines
                if len_hills * n_pos_per_window > len_pos: 
                    extra_hills = int(np.ceil((len_hills * n_pos_per_window - len_pos) / n_pos_per_window))
                    hills = hills[:-extra_hills] # this will require removing some positions with the next if statement
                    len_hills = len(hills)
                
                # if there are too many position lines, remove extra position lines    
                if len_hills * n_pos_per_window < len_pos: 
                    extra_positions = int(len_pos - len_hills * n_pos_per_window)
                    pos = pos[:-extra_positions]
                    len_pos = len(pos)
                
                # if the lengths still don't match, print a warning message and return None        
                if len_hills * n_pos_per_window != len_pos: 
                    extra_hills = int(np.ceil((len_hills * n_pos_per_window - len_pos) / n_pos_per_window))
                    extra_positions = int(len_pos - len_hills * n_pos_per_window)
                    print(f"Attention: Problem with reading new data and cutting hills/position: {len(hills) = },  {len(pos) = }, {n_pos_per_window = }, {extra_hills = }, {extra_positions = }")
                    return None, None           

                #### ~~~~~ Make sure the end time of the position and hills data match (t_pos_end - dt_hills = t_hills_end) ~~~~~ ####        
                if round((pos[-1][0] - dt_hills) - hills[-1][0], 4) != 0.0:
                    print(f"Error in cutting the hills data: The (last time in the hills data) + (time between hills) = ({hills[-1][0] +  dt_hills}) does not match the last time in the position data ({pos[-1][0]})")
                    return None, None            
                        
            # consistency checks                                       
            assert round((pos[-1][0] - dt_hills) - hills[-1][0], 4) == 0.0, f"The last time in the position data and the last time in the hills data don't match: t_pos_end - dt_hills = {pos[-1][0] - dt_hills} != {hills[-1][0] = }"
            assert len(hills) > 1, f"The hills data: {hills = } is empty (after data has been cut)"
            assert len(pos) > 1, f"The position data: {pos = } is empty (after data has been cut)"
            assert int(round(len(pos) / len(hills))) == n_pos_per_window, f"The number of positions: {len(pos) = } divided by the number of hills: {len(hills) = } is not equal to the number of positions per hill: {n_pos_per_window = }"

            # remove the copies of the hills and position files
            os.system("rm " + hills_path_cp)
        
        # if metad_h is None or meta_h <= 0, hills is returned as None    
        else: hills = None
        os.system("rm " + pos_path_cp)

        return hills, pos

    def read_position(pos_path = "position", n_pos_analysed = 0, extension=""):
        
        # add extension to the file names
        if extension != "": pos_path = pos_path + extension
        
        n_pos_analysed = int(n_pos_analysed)
        
        # copy the position file and read the data from the copy, ignoring lines starting with #. If copy already exists, remove it first.
        pos_path_cp = pos_path + "_cp"
        if os.path.exists(pos_path_cp): os.system("rm " + pos_path_cp)
        os.system("cp " + pos_path + " " + pos_path_cp)
        with open(pos_path_cp, 'r') as file: pos = [line.split() for line in file if not line.startswith("#")]
        
        # the first position is at time 0 is removed. This is done because the position is preset.
        pos = pos[1:]
        
        # check if the position data is long enough and if the last lines are complete. If not, remove the last line.
        assert len(pos) > 2, f"The position data: {pos = } is empty ({len(pos) = }) (ec1)"
        if len(pos[-1]) < len(pos[-2]): pos = pos[:-1]  
        last_index_pos_line = len(pos[-2])-1
        if len(pos[-1][last_index_pos_line]) < len(pos[-2][last_index_pos_line]) / 2: pos = pos[:-1]

        # turn position data into a numpy array
        pos = np.array(pos, dtype=float)
        
        # check if there are more positions than the number of positions already analysed
        if len(pos) <= n_pos_analysed:
            print(f"The position data is smaller than the number of positions already analysed {len(pos) = } <= {n_pos_analysed = }")
            return None
        
        # cut position data, removing the number of positions already analysed
        if n_pos_analysed > 0: pos = pos[n_pos_analysed:]  

        # remove the copy of the position file and return the position data
        os.system("rm " + pos_path_cp)
        return pos

    def read_plumed_grid_file(filename):
        
        with open(filename, "r") as f: external_bias = f.read()    
        
        # read comment lines
        comment_lines =  [line for line in external_bias.strip().split('\n') if line.startswith("#")]
        
        # read data
        filtered_lines = [line for line in external_bias.strip().split('\n') if not line.startswith("#")]
        data_array = np.array([list(map(float, line.split())) for line in filtered_lines if line.strip() != ""])
    
        return [data_array[:, i] for i in range(len(data_array[0]))]

    def get_file_path(file_name, file_type):
        if file_name != "": print(f"\n\n *** Can not find {file_type} file (path): \"{file_name}\" ***\nPlease enter the {file_type} name (path) or \'exit\' to sys.exit(): ")
        else: print(f"\n\n*** No {file_type} data or file name (path) provided ***\nPlease enter the {file_type} name (path) or \'exit\' to sys.exit(): ")
        counter = 0 
        while True:
            file_name = input("File name (path): ")
            if file_name == "exit" or counter > 100: sys.exit()
            if os.path.exists(file_name): return file_name
            else: print(f"\n *** Can not find {file_type} file (path): \"{file_name}\". Please try again or \'exit\'")
            counter += 1

####  ---- MFI functions  ----  ####

def find_periodic_point(x_coord, min_grid, max_grid, periodic, periodic_range):
    """Finds periodic copies of input coordinates. First checks if systems is periodic. If not, returns input coordinate array. Next, it checks if each coordinate is within the boundary range (grid min/max +/- periodic_range). If it is, periodic copies will be made on the other side of the CV-domain. 

    Args:
    x_coord (float): CV-coordinate
    min_grid (float): minimum value of grid
    max_grid (float): maximum value of grid
    periodic (binary): information if system is periodic. value of 0 corresponds to non-periodic system; function will only return input coordinates. Value of 1 corresponds to periodic system; function will return input coordinates with periodic copies.
    periodic_range (float): how far outside the domain periodic copies are searched for. E.g. if the domain is from -2 to +2 and the grid ext(end) is set to 1, then periodic copies are only found for input_coordiante < -2 + 1 or input_coordiante > +2 - 1.


    Returns:
    list: list of input coord and if applicable periodic copies

    """
    if isinstance(x_coord, (float, int)): coord_list = [x_coord]
    elif isinstance(x_coord, (list, np.ndarray)): coord_list = list(x_coord)
    else: raise ValueError("Input coordinate has to be a float, int, list or numpy array.")
    
    
    if periodic == 1:
        grid_length = max_grid - min_grid
        #There are potentially 2 points, 1 original and 1 periodic copy.
        if x_coord < min_grid+periodic_range: coord_list.append(x_coord + grid_length)
        elif x_coord > max_grid-periodic_range: coord_list.append(x_coord - grid_length)
        return coord_list
    else:
        return x_coord

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
    # Harmonic = hp_kappa / 2 * (grid - hp_centre)**2
    #Second, if periodic, make harmonic potential periodic
    if periodic == 1:
        grid_length = max_grid - min_grid
        grid_centre = min_grid + grid_length/2
        if hp_centre < grid_centre:
            index_period = index(hp_centre + grid_length / 2, min_grid, grid_space)
            F_harmonic[index_period:] = hp_kappa * (grid[index_period:] - hp_centre - grid_length)
            # Harmonic[index_period:] = hp_kappa / 2 * (grid[index_period:] - hp_centre - grid_length)**2
        elif hp_centre > grid_centre:
            index_period = index(hp_centre - grid_length / 2, min_grid, grid_space)
            F_harmonic[:index_period] = hp_kappa * (grid[:index_period] - hp_centre + grid_length)
            # Harmonic[:index_period] = hp_kappa / 2 * (grid[:index_period] - hp_centre + grid_length)**2

    return F_harmonic

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

def intg_1D(Force, dx, remove_zeros=True):
    """Integration of 1D gradient using finite difference method (simpson's method).

    Args:
        Force (array): Mean force
        dx (float): grid spacing (i.e. space between consecutive grid entries)

    Returns:
        array: Free energy surface
    """
    
    if remove_zeros:
        first_non_zero = np.argmax(Force!=0) if Force[0] == 0 else 0
        last_non_zero = len(Force) - np.argmax(Force[::-1]!=0) - 1 if Force[-1] == 0 else len(Force) - 1
        non_zero_Force = Force[first_non_zero:last_non_zero+1]
        non_zero_fes = np.zeros_like(non_zero_Force)
    else: non_zero_Force, non_zero_fes = Force, np.zeros_like(Force)
    
    for j in range(len(non_zero_Force)): 
        y = non_zero_Force[:j + 1]
        N = len(y)
        if N % 2 == 0: non_zero_fes[j] = dx/6 * (np.sum(y[: N-3: 2] + 4*y[1: N-3+1: 2] + y[2: N-3+2: 2]) + np.sum(y[1: N-2: 2] + 4*y[1+1: N-1: 2] + y[1+2: N: 2])) + dx/4 * ( y[1] + y[0] + y[-1] + y[-2])
        else: non_zero_fes[j] = dx / 3.0 * np.sum(y[: N-2: 2] + 4*y[1: N-1: 2] + y[2: N: 2])
    
    if remove_zeros:
        fes = np.zeros_like(Force)    
        fes[first_non_zero:last_non_zero+1] = non_zero_fes
        if first_non_zero != 0: fes[:first_non_zero] = non_zero_fes[0]
        if last_non_zero != len(Force) - 1: fes[last_non_zero+1:] = non_zero_fes[-1]
    else: fes = non_zero_fes
    
    return fes - min(fes)

def window_forces(grid, pos, pos_meta, sigma_meta2, height_meta, kT, const, bw2, PD_limit=1E-10):
    """Takes in two arrays of positions. The periodic_positions are collected from the COLVAR file during a period of constant bias and calculates the force component associated with the probability density. periodic_hills are also positions collected from the HILLS file and calculates the force component resulting from the metadynamics bias. In a periodic system, positions have periodic copies if applicable. 

    Args:
        periodic_positions (array of shape (n,)): This array contains the positions from the COLVAR file, that were collected during a period of constant bias. n is the number of positions including their periodic copies if applicable. 
        periodic_hills (array of shape (n,)): This array contains the position of the metadynamics hill (from the HILLS file), that was deposited at the beginning of the period of constant bias. If applicable the array also contains periodic copies. 
        grid (array of shape (nbins,)): CV-array.
        sigma_meta2 (float): width of metadynamics hill squared.
        height_meta (float): height of metadynamics hill
        kT (float): Temperature of the system.
        const (float): constant factor used for the calculation of the kernel. This constant enables the patching of data with different values of bw and position_pace (cv recorded at every position_pace time_step) . const = position_pace/ bw
        bw2 (float): width of kernel density estimation (for the probability density) squared.
        PD_limit (float, optional): Probability density limit below which data will be set to zero (this is done for numerical stability reasons. For default PD_limit, numerical difference is negligable). Defaults to 1E-10.

    Returns:
        list: [pb_t, Fpbt, Fbias_window]\n
        pb_t (array of shape (nbins,)): Probability density of window\n
        Fpbt (array of shape (nbins,)): Force component associated with the probability density of the time-window. \n
        Fbias_window (array of shape (nbins,)): Force component associated with the metadynamics hill deposited at the beginning of the time-window. 
    """
    
    PD_i = np.zeros(len(grid))
    F_PD_i = np.zeros(len(grid))
    
    if height_meta is not None and height_meta >= 0: 
        F_bias_i = np.zeros(len(grid))
        Bias_i = np.zeros(len(grid))
        
        for j in range(len(pos_meta)):
            Bias_i += height_meta * np.exp( - np.square(grid - pos_meta[j]) / (2*sigma_meta2) )
            F_bias_i += 1 / sigma_meta2 * np.multiply(Bias_i, (grid - pos_meta[j]))
    else: F_bias_i, Bias_i = None, None
    
    # Estimate the biased proabability density
    for j in range(len(pos)):
        kernel = const * np.exp(- np.square(grid - pos[j]) / (2 * bw2))  
        PD_i += kernel
        F_PD_i += kT / bw2 * np.multiply(kernel, (grid - pos[j]))

    PD_i = np.where(PD_i > PD_limit, PD_i, 0)  # truncated probability density of window
    F_PD_i = np.divide(F_PD_i, PD_i, out=np.zeros_like(F_PD_i), where=PD_i > PD_limit)
   
    return [PD_i, F_PD_i, Bias_i, F_bias_i]  
   
def MFI_forces(HILLS, position, n_pos_per_window, const, bw2, kT, grid, Gamma_Factor, F_static, n_pos=-1, periodic=False, PD_limit = 1E-10, return_FES=True):

    #Specify grid variables
    nbins = len(grid)
    grid_min = grid[0]
    grid_max = grid[-1]
    periodic_range = 0.25 * (grid_max-grid_min)
    dx = grid[1] - grid[0]
    
    # initialise force terms
    PD = np.zeros(nbins)
    PD2 = np.zeros(nbins)
    Force_num = np.zeros(nbins)
    ofv_num = np.zeros(nbins)
    Bias = np.zeros(nbins)
    F_bias = np.zeros(nbins)
    
    # if n_pos not specified, n_pos is the length of the position array
    if n_pos is None or n_pos < 0: n_pos = len(position)
    # if n_pos_per_window not specified, n_pos_per_window is the length of the position array if metad is not active, otherwise it is the length of the position array divided by the number of hills
    if n_pos_per_window is None or n_pos_per_window <= 0: n_pos_per_window = n_pos if HILLS is None else int(round(len(position)/len(HILLS)))
    # n_windows: Forces are calculated in (n) windows of constant bias. Use np.ceil to include incomplete windows (windows with less than n_pos_per_window positions). use min() to avoid n_windows > len(hills).
    if HILLS is not None: n_windows = min(int(np.ceil(n_pos / n_pos_per_window)), len(HILLS))
    else: n_windows = int(np.ceil(n_pos / n_pos_per_window))
    
    # get the width of the metadynamics Gaussian and check if it is the same at the end.
    sigma_meta2 = HILLS[1, 2] ** 2  if HILLS is not None else None # width of Gaussian
    assert HILLS[1, 2] == HILLS[-1, 2], "The width of the Gaussian is not constant across the hills data."

    #Cycle over windows of constant bias
    for i in range(n_windows):
        
        #Get position data of window        
        if HILLS is not None:
            s = HILLS[i, 1]  # centre position of Gaussian
            height_meta = HILLS[i, 3] * Gamma_Factor  # Height of Gaussian
            pos_meta = find_periodic_point(np.array([s]), grid_min, grid_max, periodic, periodic_range) if periodic else np.array([s])
        data = position[i * n_pos_per_window: (i + 1) * n_pos_per_window]  # positons of window of constant bias force.
        pos = find_periodic_point(data, grid_min, grid_max, periodic, periodic_range) if periodic else data
                
        # Find forces of window
        [PD_i, F_PD_i, Bias_i, F_bias_i] = window_forces(grid, pos, pos_meta, sigma_meta2, height_meta, kT, const, bw2, PD_limit)

        # update force terms        
        PD += PD_i # total probability density     
        if HILLS is not None: Bias += Bias_i       
        if HILLS is not None: F_bias += F_bias_i
        Force_i = F_PD_i + F_bias - F_static
        Force_num += np.multiply(PD_i, Force_i)

        # terms for error calculation
        PD2 += np.square(PD_i) 
        ofv_num += np.multiply(PD_i, np.square(Force_i))  
            
    Force = np.divide(Force_num, PD, out=np.zeros_like(Force_num), where=PD > PD_limit)           
    FES = intg_1D(Force, dx) if return_FES else np.zeros_like(Force)
    
    return PD, PD2, Force, ofv_num, F_bias, Bias, FES

def patch_forces(forces, base_forces=None, PD_limit=1E-10):
    
    # check if forces and base_forces are lists. If they are, convert them to numpy arrays
    if isinstance(forces, list): forces = np.array(forces)
    if isinstance(base_forces, list): base_forces = np.array(base_forces)
    
    # Find the length of the grid 
    if len(np.shape(forces)) == 3: nbins = len(forces[0, 0])
    elif len(np.shape(forces)) == 2: nbins = len(forces[0])
    else: raise ValueError("\n *** The forces array has an unexpected shape *** \n")
    
    if np.shape(forces)[-2] == 4: # this if statement activated when the forces have the format: [PD, PD2, F, OFV]
    
        #Initialise arrays
        PD = np.zeros(nbins)
        PD2 = np.zeros(nbins)
        Force = np.zeros(nbins)
        ofv_num = np.zeros(nbins)
        
        # add forces to the total
        if len(np.shape(forces)) == 3: 
            for i in range(len(forces)):
                PD += forces[i, 0]
                PD2 += forces[i, 1]
                Force += np.multiply(forces[i, 2], forces[i, 0])
                ofv_num += forces[i, 3]
        elif len(np.shape(forces)) == 2: 
            PD += forces[0]
            PD2 += forces[1]
            Force += np.multiply(forces[2], forces[0])
            ofv_num += forces[3]
            
        # if base_forces is not None, add base_forces to the total
        if base_forces is not None:
            assert np.shape(base_forces)[-2] == 4, "The base_forces array has an unexpected shape. It should have the format: [PD, PD2, F, OFV]"
            if len(np.shape(base_forces)) == 3:
                for i in range(len(base_forces)):
                    PD += base_forces[i, 0]
                    PD2 += base_forces[i, 1]
                    Force += np.multiply(base_forces[i, 2], base_forces[i, 0])
                    ofv_num += base_forces[i, 3]
            elif len(np.shape(base_forces)) == 2:
                PD += base_forces[0]
                PD2 += base_forces[1]
                Force += np.multiply(base_forces[2], base_forces[0])
                ofv_num += base_forces[3]
    
        # force is divided by the probability density
        Force = np.divide(Force, PD, out=np.zeros_like(Force), where=PD>PD_limit)
        return [PD, PD2, Force, ofv_num]
    
    if np.shape(forces)[-2] == 2: # this if statement activated when the forces have the format: [PD, F]
        
        #Initialise arrays
        PD = np.zeros(nbins)
        Force = np.zeros(nbins)
        
        # add forces to the total
        if len(np.shape(forces)) == 3: 
            for i in range(len(forces)):
                PD += forces[i, 0]
                Force += np.multiply(forces[i, 1], forces[i, 0])
        elif len(np.shape(forces)) == 2: 
            PD += forces[0]
            Force += np.multiply(forces[1], forces[0])
            
        # if base_forces is not None, add base_forces to the total
        if base_forces is not None:
            if np.shape(base_forces)[-2] == 4:
                print("\n *** The base_forces array has different shape than the force_array. It should have the format: [PD, F]. Will use the 0th and 2nd columns of the base_forces array *** \n")
                base_forces = base_forces[:,[0,2]] if len(np.shape(base_forces)) == 3 else base_forces[[0,2]]
            assert np.shape(base_forces)[-2] == 2, "The base_forces array has an unexpected shape. It should have the format: [PD, F]"
            if len(np.shape(base_forces)) == 3:
                for i in range(len(base_forces)):
                    PD += base_forces[i, 0]
                    Force += np.multiply(base_forces[i, 1], base_forces[i, 0])
            elif len(np.shape(base_forces)) == 2:
                PD += base_forces[0]
                Force += np.multiply(base_forces[1], base_forces[0])
    
    # force is divided by the probability density
    Force = np.divide(Force, PD, out=np.zeros_like(Force), where=PD>PD_limit)
    return [PD, Force]        
 
####  ---- MFI functions Numba  ----  ####

@jit
def find_periodic_point_numba(coord_array, min_grid, max_grid, periodic, periodic_range):
    """Finds periodic copies of input coordinates. First checks if systems is periodic. If not, returns input coordinate array. Next, it checks if each coordinate is within the boundary range (grid min/max +/- periodic_range). If it is, periodic copies will be made on the other side of the CV-domain. 

    Args:
        coord_array (array): array of CV-coordinate s
        min_grid (float): minimum value of grid
        max_grid (float): maximum value of grid
        periodic (binary): information if system is periodic. value of 0 corresponds to non-periodic system; function will only return input coordinates. Value of 1 corresponds to periodic system; function will return input coordinates with periodic copies.
        periodic_range (float): how far outside the domain periodic copies are searched for. E.g. if the domain is from -2 to +2 and the grid ext(end) is set to 1, then periodic copies are only found for input_coordiante < -2 + 1 or input_coordiante > +2 - 1.


    Returns:
        np.array: list of input coord and if applicable periodic copies
    """

    if periodic == 0:
        return coord_array
    elif periodic == 1:
        grid_length = max_grid - min_grid
        len_coord_array = len(coord_array)
        for i in range(len_coord_array):
            if coord_array[i] < min_grid+periodic_range:
                coord_array = np.append(coord_array, coord_array[i] + grid_length)
            elif coord_array[i] > max_grid-periodic_range:
                coord_array = np.append(coord_array, coord_array[i] - grid_length)
        return coord_array  
    
@njit
def find_hp_force_numba(hp_centre, hp_kappa, grid, min_grid, max_grid, grid_space, periodic):
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
    # Harmonic = hp_kappa / 2 * (grid - hp_centre)**2
    #Second, if periodic, make harmonic potential periodic
    if periodic == 1:
        grid_length = max_grid - min_grid
        grid_centre = min_grid + grid_length/2
        if hp_centre < grid_centre:
            index_period = index(hp_centre + grid_length / 2, min_grid, grid_space)
            F_harmonic[index_period:] = hp_kappa * (grid[index_period:] - hp_centre - grid_length)
            # Harmonic[index_period:] = hp_kappa / 2 * (grid[index_period:] - hp_centre - grid_length)**2
        elif hp_centre > grid_centre:
            index_period = index(hp_centre - grid_length / 2, min_grid, grid_space)
            F_harmonic[:index_period] = hp_kappa * (grid[:index_period] - hp_centre + grid_length)
            # Harmonic[:index_period] = hp_kappa / 2 * (grid[:index_period] - hp_centre + grid_length)**2

    return F_harmonic

@njit
def find_lw_force_numba(lw_centre, lw_kappa, grid, min_grid, max_grid, grid_space, periodic):
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
def find_uw_force_numba(uw_centre, uw_kappa, grid, min_grid, max_grid, grid_space, periodic):
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
def intg_1D_numba(Force, dx, remove_zeros=True):
    """Integration of 1D gradient using finite difference method (simpson's method).

    Args:
        Force (array): Mean force
        dx (float): grid spacing (i.e. space between consecutive grid entries)

    Returns:
        array: Free energy surface
    """
    
    if remove_zeros:
        first_non_zero = np.argmax(Force!=0) if Force[0] == 0 else 0
        last_non_zero = len(Force) - np.argmax(Force[::-1]!=0) - 1 if Force[-1] == 0 else len(Force) - 1
        non_zero_Force = Force[first_non_zero:last_non_zero+1]
        non_zero_fes = np.zeros_like(non_zero_Force)
    else: non_zero_Force, non_zero_fes = Force, np.zeros_like(Force)
    
    for j in range(len(non_zero_Force)): 
        y = non_zero_Force[:j + 1]
        N = len(y)
        if N % 2 == 0: non_zero_fes[j] = dx/6 * (np.sum(y[: N-3: 2] + 4*y[1: N-3+1: 2] + y[2: N-3+2: 2]) + np.sum(y[1: N-2: 2] + 4*y[1+1: N-1: 2] + y[1+2: N: 2])) + dx/4 * ( y[1] + y[0] + y[-1] + y[-2])
        else: non_zero_fes[j] = dx / 3.0 * np.sum(y[: N-2: 2] + 4*y[1: N-1: 2] + y[2: N: 2])
    
    if remove_zeros:
        fes = np.zeros_like(Force)    
        fes[first_non_zero:last_non_zero+1] = non_zero_fes
        if first_non_zero != 0: fes[:first_non_zero] = non_zero_fes[0]
        if last_non_zero != len(Force) - 1: fes[last_non_zero+1:] = non_zero_fes[-1]
    else: fes = non_zero_fes
    
    return fes - min(fes)

@njit
def window_forces_numba(grid, pos, pos_meta, sigma_meta2, height_meta, kT, const, bw2, PD_limit=1E-10):
    """Takes in two arrays of positions. The periodic_positions are collected from the COLVAR file during a period of constant bias and calculates the force component associated with the probability density. periodic_hills are also positions collected from the HILLS file and calculates the force component resulting from the metadynamics bias. In a periodic system, positions have periodic copies if applicable. 

    Args:
        periodic_positions (array of shape (n,)): This array contains the positions from the COLVAR file, that were collected during a period of constant bias. n is the number of positions including their periodic copies if applicable. 
        periodic_hills (array of shape (n,)): This array contains the position of the metadynamics hill (from the HILLS file), that was deposited at the beginning of the period of constant bias. If applicable the array also contains periodic copies. 
        grid (array of shape (nbins,)): CV-array.
        sigma_meta2 (float): width of metadynamics hill squared.
        height_meta (float): height of metadynamics hill
        kT (float): Temperature of the system.
        const (float): constant factor used for the calculation of the kernel. This constant enables the patching of data with different values of bw and position_pace (cv recorded at every position_pace time_step) . const = position_pace/ bw
        bw2 (float): width of kernel density estimation (for the probability density) squared.
        PD_limit (float, optional): Probability density limit below which data will be set to zero (this is done for numerical stability reasons. For default PD_limit, numerical difference is negligable). Defaults to 1E-10.

    Returns:
        list: [pb_t, Fpbt, Fbias_window]\n
        pb_t (array of shape (nbins,)): Probability density of window\n
        Fpbt (array of shape (nbins,)): Force component associated with the probability density of the time-window. \n
        Fbias_window (array of shape (nbins,)): Force component associated with the metadynamics hill deposited at the beginning of the time-window. 
    """
    
    PD_i = np.zeros(len(grid))
    F_PD_i = np.zeros(len(grid))
    F_bias_i = np.zeros(len(grid))
    Bias_i = np.zeros(len(grid))
    
    for j in range(len(pos_meta)):
        Bias_i += height_meta * np.exp( - np.square(grid - pos_meta[j]) / (2*sigma_meta2) )
        F_bias_i += 1 / sigma_meta2 * np.multiply(Bias_i, (grid - pos_meta[j]))
    
    # Estimate the biased proabability density
    for j in range(len(pos)):
        kernel = const * np.exp(- np.square(grid - pos[j]) / (2 * bw2))  
        PD_i += kernel
        F_PD_i += kT / bw2 * np.multiply(kernel, (grid - pos[j]))

    PD_i = np.where(PD_i > PD_limit, PD_i, 0)  # truncated probability density of window
    F_PD_i = np.where(PD_i > PD_limit, F_PD_i / PD_i, 0)
   
    return [PD_i, F_PD_i, Bias_i, F_bias_i]  

@njit
def find_force_nobias_numba(grid, pos, bw2, kT, const, PD_limit=1E-10):
    
    PD_i = np.zeros(len(grid)) # Probability density
    F_PD_i = np.zeros(len(grid))  # Force component associated with the probability density
    
    # Estimate the biased proabability density
    for j in range(len(pos)):
        kernel = const * np.exp(- np.square(grid - pos[j]) / (2 * bw2))  
        PD_i += kernel
        F_PD_i += kT / bw2 * np.multiply(kernel, (grid - pos[j]))
    
    PD_i = np.where(PD_i > PD_limit, PD_i, 0)  # truncated probability density of window
    F_PD_i = np.where(PD_i > PD_limit, F_PD_i / PD_i, 0)
    
    return PD_i, F_PD_i
   
@njit
def MFI_forces_numba(HILLS, position, n_pos_per_window, const, bw2, kT, grid, Gamma_Factor, F_static, n_pos=-1, periodic=False, PD_limit = 1E-10, return_FES=True):

    #Specify grid variables
    nbins = len(grid)
    grid_min = grid[0]
    grid_max = grid[-1]
    periodic_range = 0.25 * (grid_max-grid_min)
    dx = grid[1] - grid[0]
    
    # initialise force terms
    PD = np.zeros(nbins)
    PD2 = np.zeros(nbins)
    Force_num = np.zeros(nbins)
    ofv_num = np.zeros(nbins)
    Bias = np.zeros(nbins)
    F_bias = np.zeros(nbins)
    
    if HILLS is not None:

        #Cycle over windows of constant bias (for each deposition of a gaussian bias)
        if n_pos_per_window is None or n_pos_per_window <= 0: n_pos_per_window = int(round(len(position)/len(HILLS)))
        if n_pos is None or n_pos < 0: n_windows = len(HILLS)
        else: n_windows = min(int(np.ceil(n_pos / n_pos_per_window)), len(HILLS))
               
        for i in range(n_windows):
            
            #Get position data of window        
            s = HILLS[i, 1]  # centre position of Gaussian
            sigma_meta2 = HILLS[i, 2] ** 2  # width of Gaussian
            height_meta = HILLS[i, 3] * Gamma_Factor  # Height of Gaussian
            data = position[i * n_pos_per_window: (i + 1) * n_pos_per_window]  # positons of window of constant bias force.
            pos_meta = find_periodic_point_numba(np.array([s]), grid_min, grid_max, periodic, periodic_range) if periodic else np.array([s])
            pos = find_periodic_point_numba(data, grid_min, grid_max, periodic, periodic_range) if periodic else data
                    
            # Find forces of window
            [PD_i, F_PD_i, Bias_i, F_bias_i] = window_forces_numba(grid, pos, pos_meta, sigma_meta2, height_meta, kT, const, bw2, PD_limit)

            # update force terms        
            PD += PD_i # total probability density     
            Bias += Bias_i       
            F_bias += F_bias_i
            Force_i = F_PD_i + F_bias - F_static
            Force_num += np.multiply(PD_i, Force_i)

            # terms for error calculation
            PD2 += np.square(PD_i) 
            ofv_num += np.multiply(PD_i, np.square(Force_i))  
            
    else: #MFI without hills. Iterate over time windows to have a consisten error calculation. Can be skippe for marginal increase in speed.
        
        if n_pos is None or n_pos < 0: n_pos = len(position)
        if n_pos_per_window is None or n_pos_per_window <= 0: n_pos_per_window = n_pos
        n_windows = int(round(len(position)/n_pos_per_window, 0))
        
        for i in range(n_windows):
            
            data = position[i * n_pos_per_window: (i + 1) * n_pos_per_window]  # positons of window of constant bias force.
            pos = find_periodic_point_numba(data, grid_min, grid_max, periodic, periodic_range) if periodic else data
        
            # Find forces of window
            [PD_i, F_PD_i] = find_force_nobias_numba(grid, pos, bw2, kT, const, PD_limit) #
            
            # update force terms        
            PD += PD_i
            Force_i = F_PD_i - F_static   ###Fstatic is HP and/or InvFES
            Force_num += np.multiply(PD_i, Force_i)
            
            # terms for error calculation  
            PD2 += np.square(PD_i) 
            ofv_num += np.multiply(PD_i, np.square(Force_i))
        
        # # Find forces of window
        # [PD_i, F_PD_i] = find_force_nobias_numba(grid, position, bw2, kT, const, PD_limit)
        
        # PD += PD_i
        # Force_i = F_PD_i - F_static   ###Fstatic is HP and/or InvFES
        # Force_num += np.multiply(PD_i, Force_i)
        
        # PD2 += np.square(PD_i)
        # ofv_num += np.multiply(PD_i, np.square(Force_i))
            
    Force = np.where(PD > PD_limit, Force_num / PD, 0)            
    if return_FES: FES = intg_1D_numba(Force, dx)
    
    if return_FES:
        end_arr = np.zeros((7, nbins))
        end_arr[0], end_arr[1], end_arr[2], end_arr[3], end_arr[4], end_arr[5], end_arr[6] = PD, PD2, Force, ofv_num, F_bias, Bias, FES
    else: 
        end_arr = np.zeros((6, nbins))
        end_arr[0], end_arr[1], end_arr[2], end_arr[3], end_arr[4], end_arr[5] = PD, PD2, Force, ofv_num, F_bias, Bias

    return end_arr
           
####  ---- Real Time Re-Initialisation functions  ----  ####

def check_US_criteria(error_map, cutoff_map=None, gaussian_sigma=3, check_termination=None, periodic=False):
    
    if cutoff_map is None: cutoff_map = np.ones_like(error_map)
    
    filter_mode = "wrap" if periodic else "reflect"
    error_smooth = gaussian_filter(error_map, gaussian_sigma, mode=filter_mode) * cutoff_map   
    error_avr = np.sum(error_smooth) / np.count_nonzero(error_smooth)
        
    if check_termination is None:
        error_max = error_smooth[np.nanargmax(error_smooth)]
        return error_avr, error_max
        
    else:
        grid, hp_centre = check_termination
        hp_index = np.argmin(np.abs(grid - hp_centre))
        error_centre = error_smooth[hp_index]
        return error_avr, error_centre
    
def find_hp_centre(grid, error_map, cutoff_map=None, gaussian_sigma=3, prev_hp_centre=None, periodic=False):

    if cutoff_map is None: cutoff_map = np.ones_like(error_map)
    dx = grid[1] - grid[0]

    filter_mode = "wrap" if periodic else "reflect"
    error_smooth = gaussian_filter(error_map, sigma=gaussian_sigma, mode=filter_mode) * cutoff_map   

    # find the position of the maximum error
    new_hp_centre = round(grid[np.nanargmax(error_smooth)],6)          
    
    # if the previous simulation was in the US phase (prev hp centre is defined), check if the new hp centre is somewhere else. If centre in the same position, switch to flat phase.
    for prev_hp in prev_hp_centre:
        if prev_hp is not None and abs(new_hp_centre - prev_hp) < dx/10: return None
        
    return new_hp_centre

####  ---- Statistical Analysis of Error progression collections  ----  ####

def bootstrapping_error(grid, force_array, n_bootstrap, periodic=False, FES_cutoff=None, PD_cutoff=None, use_VNORM=False, set_fes_minima=None, print_progress=False):
    """Algorithm to determine bootstrap error

    Args:
        grid (array of shape (nbins,)): CV grid positions
        force_array (list): collection of force terms (n * [PD, Force]).
        n_bootstrap (int): bootstrap iterations

    Returns:
        FES_avr (array of shape (nbins,)): Average of all FES generated during the bootstrap algorithm.
        sd_fes (array of shape (nbins,)): Standard deviation of all FES generated during the bootstrap algorithm.
        sd_fes_prog (array of shape (n_bootstrap,)): Global average of the standard deviation after each bootstrap iteration. When this array converges, enough iterations have been performed. If it does not converge, move iterations are necessary.
    """

    # Get the correct shape of the force array (Needs to be [PD, F], but sometimes it is [PD, PD2, F, OFV])
    if force_array.shape[-2] == 4: force_array = force_array[:,[0,2]]
    if force_array.shape[-2] != 2: raise ValueError("force_array should have shape (n_forces, 3) or (n_forces, 6)")

    #Define constants and initialise arrays
    nbins = len(grid)
    dx = grid[1] - grid[0]
    sd_fes_prog = np.zeros(n_bootstrap)    
    FES_avr = np.zeros(nbins)
    M2 = np.zeros(nbins)
    n_forces = len(force_array)
    
    # Find FES of the complete force array
    [PD, Force] = patch_forces(force_array)
    FES_0 = intg_1D(Force, dx)
    
    # find cutoff array and averaging denominator
    cutoff = np.ones(nbins)
    if FES_cutoff is not None: cutoff = np.where(FES_0 < FES_cutoff, cutoff, 0)
    if PD_cutoff is not None: cutoff = np.where(PD > PD_cutoff, cutoff, 0)
    if use_VNORM: averaging_denominator = (np.sum(cutoff)**2) / nbins
    else: averaging_denominator = np.sum(cutoff) 
        
    for iteration in range(n_bootstrap):
        
        #Randomly choose forward forces and backward forces and save to force array
        force = np.zeros((int(n_forces), 2, nbins)) 
        np.random.seed()
        random_sample_index = np.random.choice(n_forces, size=n_forces)      
        force = force_array[random_sample_index]
     
        #patch forces to find average PD, Force and FES
        [_, Force] = patch_forces(force)
        FES = intg_1D(Force, dx)
        if set_fes_minima == "first_value":  FES = FES - FES[0]
        else: FES = FES - min(FES)
        
        # calculate standard devaition using Welford’s method
        delta = FES - FES_avr
        FES_avr += delta/(iteration+1)
        delta2 = FES - FES_avr
        M2 += delta*delta2
        
        if iteration > 0:
            sd_fes = np.sqrt(M2/iteration)
            if PD_cutoff is not None or FES_cutoff is not None: sd_fes *= cutoff
            sd_fes_prog[iteration] = np.sum(sd_fes) / averaging_denominator 
        else: sd_fes_prog[iteration] = 0
        
        # print progress
        if print_progress and (iteration+1) % 50 == 0: print("Iteration:", iteration+1, "- sd:", round(sd_fes_prog[iteration],5) )
       
    return [FES_0, FES_avr, sd_fes, sd_fes_prog]

def weighted_bootstrapping_error(grid, force_array, n_bootstrap, periodic=False, FES_cutoff=None, PD_cutoff=None, use_VNORM=False, set_fes_minima=None, print_progress=False):
    """Algorithm to determine bootstrap error

    Args:
        grid (array of shape (nbins,)): CV grid positions
        force_array (list): collection of force terms (n * [PD, Force]).
        n_bootstrap (int): bootstrap iterations

    Returns:
        FES_avr (array of shape (nbins,)): Average of all FES generated during the bootstrap algorithm.
        sd_fes (array of shape (nbins,)): Standard deviation of all FES generated during the bootstrap algorithm.
        sd_fes_prog (array of shape (n_bootstrap,)): Global average of the standard deviation after each bootstrap iteration. When this array converges, enough iterations have been performed. If it does not converge, move iterations are necessary.
    """

    # Get the correct shape of the force array (Needs to be [PD, Force], but sometimes it is [PD, PD2, Force, OFV])
    if force_array.shape[-2] == 4: force_array = force_array[:,[0,2]]
    if force_array.shape[-2] != 2: raise ValueError("force_array should have shape (n_forces, 3) or (n_forces, 6)")

    #Define constants and initialise arrays
    dx = grid[1] - grid[0]
    nbins = len(grid)
    n_forces = len(force_array)
    sd_fes_prog = np.zeros(n_bootstrap)    
    FES_sum = np.zeros(nbins)
    sum_var_num = np.zeros(nbins)
    sum_PD = np.zeros(nbins)
    sum_PD2 = np.zeros(nbins)    
    
    # Find FES of the complete force array
    [PD, Force] = patch_forces(force_array)
    FES_0 = intg_1D(Force, dx)
    
    # find cutoff array and averaging denominator
    cutoff = np.ones(nbins)
    if FES_cutoff is not None: cutoff = np.where(FES_0 < FES_cutoff, cutoff, 0)
    if PD_cutoff is not None: cutoff = np.where(PD > PD_cutoff, cutoff, 0)
    if use_VNORM: averaging_denominator = (np.sum(cutoff)**2) / nbins
    else: averaging_denominator = np.sum(cutoff) 
        
    for iteration in range(n_bootstrap):
        
        #Randomly choose forward forces and backward forces and save to force array
        force = np.zeros((int(n_forces), 2, nbins)) 
        np.random.seed()
        random_sample_index = np.random.choice(n_forces, size=n_forces)      
        force = force_array[random_sample_index]
     
        #patch forces to find average PD, Force and FES
        [PD, Force] = patch_forces(force)
        FES = intg_1D(Force, dx)
        if set_fes_minima == "first_value":  FES = FES - FES[0]
        else: FES = FES - min(FES)
        
        # calculate standard devaition using Welford’s method
        FES_sum += FES
        sum_var_num += PD * np.square(FES - FES_0)
        sum_PD += PD
        sum_PD2 += np.square(PD)
        
        if iteration > 0:
            
            sum_PD_squared = np.square(sum_PD)
            diff_PD2 = sum_PD_squared - sum_PD2
            BC = np.divide(sum_PD_squared, diff_PD2, out=np.zeros_like(PD), where=diff_PD2>0)
            var_fes = np.divide(sum_var_num , sum_PD, out=np.zeros_like(PD), where=sum_PD>1) * BC
            sd_fes = np.sqrt(var_fes) * cutoff if PD_cutoff is not None or FES_cutoff is not None else np.sqrt(var_fes)

            sd_fes_prog[iteration] = np.sum(sd_fes) / averaging_denominator 
        else: sd_fes_prog[iteration] = 0
        
        # print progress
        if print_progress and (iteration+1) % 50 == 0: print("Iteration:", iteration+1, "- sd:", round(sd_fes_prog[iteration],5) )
    
    FES_avr = FES_sum / n_bootstrap
       
    return [FES_0, FES_avr, sd_fes, sd_fes_prog]

def bootstrapping_progression(grid, force_array, time_array=None, n_bootstrap=100, block_size=1, periodic=False, FES_cutoff=None, PD_cutoff=None, use_VNORM=False, show_plot=False):
    
    # get the force array and time array in blocks.
    if block_size > 1: 
        new_force_array = make_force_terms_blocks(force_array, block_size)
        
        if time_array is not None:
            if not (len(force_array) == len(time_array) or len(force_array) == len(time_array)-1): raise ValueError("The length of time_array and force_array does not match. Please check the input")
            if len(force_array) == len(time_array): new_time_array = time_array[block_size-1::block_size]
            elif len(force_array) == len(time_array)-1: new_time_array = time_array[block_size::block_size]
            if new_time_array[-1] != time_array[-1]: new_time_array = np.append(new_time_array, time_array[-1])
            if len(new_time_array) != len(new_force_array): raise ValueError("The length of time_array and force_array does not match. Please check the code")
            time_array = new_time_array
        force_array = new_force_array
        
    elif time_array is not None and len(force_array) == len(time_array)-1: time_array = time_array[1:]

    sd_fes_evo = []
    for n_sim in range(1,len(force_array)+1):
    
        if n_sim > 2: 
            FES_0, FES_avr, sd_fes, sd_fes_prog = bootstrapping_error(grid, force_array[:n_sim], int(max(10,n_bootstrap*(n_sim/len(force_array)))), periodic, FES_cutoff, PD_cutoff, use_VNORM)
            sd_fes_evo.append(sd_fes_prog)
        else: sd_fes_evo.append(np.nan)

    if show_plot: plt.plot(sd_fes_evo); plt.xlabel("Number of forces_e"); plt.ylabel("Standard deviation of the FES [kJ/mol]"); plt.title("Bootstrapping progression"); plt.show()
    
    if time_array is None: return FES_0, FES_avr, sd_fes, sd_fes_evo
    else: return FES_0, FES_avr, sd_fes, sd_fes_evo, time_array
    
def make_force_terms_blocks(force_terms, block_size):
    
    n_blocks = int(np.ceil(len(force_terms) / block_size))
    new_force_terms = []
    for i in range(n_blocks):
        
        if i == n_blocks-1: block = patch_forces(force_terms[i*block_size:])
        else: block = patch_forces(force_terms[i*block_size:(i+1)*block_size])
        new_force_terms.append(block)
    
    return np.array(new_force_terms)

def remove_flat_tail(grid, FES, sd):
    """Removes the flat tail of the FES and the same array-elements of the other arrays. This is usefull when the FES has a flat tail a the end (due to the force being zero in that region). 

    Args:
        grid (array of shape (nbins,)): CV grid positions.
        FES (array of shape (nbins,)): Free energy surface.
        sd (array of shape (nbins,)): Standard deviation of the free energy surface. This can also be another arbritary array.

    Returns:
        x (array of shape (nbins,)): CV grid positions with flat tail removed.
        FES (array of shape (nbins,)): Free energy surface with flat tail removed.
        sd (array of shape (nbins,)): Standard deviation of the free energy surface with flat tail removed.
    """
    
    while FES[-1] - FES[-3] < 10E-10:  # move in steps of 2, because sometimes the tail has a "zig-zag" pattern
        FES = FES[:-2]
        grid = grid[:-2]
        sd = sd[:-2]
        
    return [grid, FES, sd]

def patch_and_error_prog_parallel_sim(grid, force_terms_collection, y=None, PD_cutoff=1, FES_cutoff=None, use_ST_ERR=True, use_VNORM=False):
    
    force_terms_collection = np.array(force_terms_collection, dtype=float)
    assert len(np.shape(force_terms_collection)) == 4, "force_terms_collection must be a 4D array with dimensions (n_sim, n_iter, 4_force_terms, nbins)"
    
    n_sim = np.shape(force_terms_collection)[0]
    n_iter = np.shape(force_terms_collection)[1]
    nbins = np.shape(force_terms_collection)[3]
    dx = grid[1] - grid[0]
    
    PD, PD2, Force_num, ofv_num  = np.zeros(nbins), np.zeros(nbins), np.zeros(nbins), np.zeros(nbins)
    ofe_prog, aad_prog = np.zeros(n_iter), np.zeros(n_iter)
    
    for i in range(n_iter):
        for j in range(n_sim):
            PD += force_terms_collection[j, i, 0, :]
            PD2 += force_terms_collection[j, i, 1, :]
            Force_num += force_terms_collection[j, i, 2, :] * force_terms_collection[j, i, 0, :]
            ofv_num += force_terms_collection[j, i, 3, :]
            
        Force = np.divide(Force_num, PD, out=np.zeros_like(Force_num), where=PD>1E-10)
        FES = lib1.intg_1D(Force, dx)
        
        # Find cutoff
        cutoff = np.where(FES > FES_cutoff, 0, 1) if FES_cutoff is not None else np.ones(nbins)
        if PD_cutoff is not None and PD_cutoff > 0: cutoff = np.where(PD < PD_cutoff, 0, cutoff)
        space_explored = np.sum(cutoff)      
        ratio_explored = space_explored / nbins

        # Find error of the mean force
        ofv = np.divide(ofv_num, PD, out=np.zeros_like(ofv_num), where=(PD>1E-10) & (cutoff > 0.5)) - np.square(Force)
        Bessel_corr_num = PD2.copy() if use_ST_ERR == True else np.square(PD)
        ofv *= np.divide(Bessel_corr_num, np.square(PD) - PD2, out=np.zeros_like(Bessel_corr_num), where=((np.square(PD) - PD2 > 0) & (cutoff > 0.5)))
        ofe = np.sqrt(ofv*cutoff)
        Aofe = np.sum(ofe) / space_explored if space_explored > 0 else np.nan
        if use_VNORM: Aofe = Aofe * space_explored / nbins
        
        if y is not None:
            AD = abs(FES - y) * cutoff
            AAD = np.sum(AD) / space_explored if space_explored > 0 else np.nan
            if use_VNORM: AAD = AAD * space_explored / nbins
        else: AAD = None

        ofe_prog[i] = Aofe
        aad_prog[i] = AAD
            
    return ofe_prog, aad_prog

def get_mean_ste_of_n_error_prog(time, error_collection, error_collection_2=None, return_results=True, save_data_path=None, plot=True, save_plot_path=None, plot_log=False, 
                                 plot_title=["Error of Mean Force", "AAD of FES"], line_label="", plot_xlabel=["Time"], plot_ylabel=["Error [kJ/mol]","AAD [kJ/mol]"], ste_alpha=0.3):
    
    time = np.array(time, dtype=float)
    error_collection = np.array(error_collection, dtype=float)
    if error_collection_2 is not None: error_collection_2 = np.array(error_collection_2, dtype=float)
 
    mean = np.mean(error_collection, axis=0)
    ste = np.std(error_collection, axis=0) / np.sqrt(len(error_collection))

    if error_collection_2 is not None:
        mean_2 = np.mean(error_collection_2, axis=0)
        ste_2 = np.std(error_collection_2, axis=0) / np.sqrt(len(error_collection_2))
        
    if save_data_path is not None:
        if error_collection_2 is not None: save_pkl([time, mean, ste, mean_2, ste_2], save_data_path) 
        else: save_pkl([time, mean, ste], save_data_path)
        
        
    if error_collection_2 is not None: 
        plt.figure(figsize=(10,4))  
        plt.subplot(1,2,1)
        plt.plot(time, mean, linewidth=1, color="green", label=line_label)
        plt.fill_between(time, mean - ste, mean + ste, color="green", alpha=ste_alpha)
        plt.ylim(min(mean)*0.9, max(mean)*1.1 )
        if plot_log: plt.yscale("log")
        plt.title(plot_title[0])
        plt.xlabel(plot_xlabel[0])
        plt.ylabel(plot_ylabel[0])
        
        plt.subplot(1,2,2)
        plt.plot(time, mean_2, linewidth=1, color="green", label=line_label)
        plt.fill_between(time, mean_2 - ste_2, mean_2 + ste_2, color="green", alpha=ste_alpha)
        # plt.ylim(min(mean_2)*0.9, max(mean_2)*1.1 )
        if plot_log: plt.yscale("log")
        plt.title(plot_title[-1]); plt.xlabel(plot_xlabel[-1]); plt.ylabel(plot_ylabel[-1])
                    
        if save_plot_path is not None: plt.savefig(save_plot_path)
        if plot: 
            plt.tight_layout()
            plt.show()
    
    else: 
        plt.figure(figsize=(5,4))
        plt.plot(time, mean, linewidth=1, color="red", label=line_label)
        plt.fill_between(time, mean - ste, mean + ste, color="red", alpha=0.3)
        # plt.ylim(min(mean)*0.9, max(mean)*1.1 )
        if plot_log: plt.yscale("log")
        plt.title(plot_title[0]); plt.xlabel(plot_xlabel[0]); plt.ylabel(plot_ylabel[0])

        if save_plot_path is not None: plt.savefig(save_plot_path)
        if plot: plt.show()
        
    if return_results: 
        if error_collection_2 is not None: return [time, mean, ste, mean_2, ste_2]
        else: return [time, mean, ste]

def get_avr_error_prog(path_data=None, n_surf=4, total_campaigns=50, time_budget=100, include_aad=True, simulation_type="", return_avr_prog=False,
                       make_plot=True, show_plot=True, plot_log=True, plot_title=["Error of Mean Force", "AAD of FES"], line_label="", plot_xlabel=["Time"], plot_ylabel=["Error [kJ/mol]","AAD [kJ/mol]"]):
    
    # initialize t min and max, and data lists
    interpolation_time_min, interpolation_time_max = 0, np.inf
    t_data, err_data, err_interp, aad_data, aad_interp = [], [], [], [], []
    
    if simulation_type == "SRTR": sim_folder_prefix = "SRTRcampaign"
    if simulation_type == "PRTR": sim_folder_prefix = "PRTRcampaign"
    if simulation_type == "long": sim_folder_prefix = "simulation"

    # Load error "progression data
    for i in range(1,total_campaigns+1):
        camp_id = f"_{i}_{time_budget}ns"
        if include_aad: [t_i,err_i, aad_i] = load_pkl(f"{path_data}S{n_surf}/{sim_folder_prefix}{camp_id}/error_progression{camp_id}.pkl")
        else: [t_i,err_i] = load_pkl(f"{path_data}S{n_surf}/{sim_folder_prefix}{camp_id}/error_progression{camp_id}.pkl")
        t_data.append(list(t_i)); err_data.append(list(err_i))
        if include_aad: aad_data.append(list(aad_i))
        interpolation_time_min, interpolation_time_max = max(interpolation_time_min, min(t_i)), min(interpolation_time_max, max(t_i))
    
    # Interpolate the data
    time = np.linspace(interpolation_time_min, interpolation_time_max, 250)
    for i in range(total_campaigns):
        err_interp.append(interp1d(t_data[i], err_data[i], kind='cubic')(time))
        if include_aad: aad_interp.append(interp1d(t_data[i], aad_data[i], kind='cubic')(time))
    
    # Get the mean and standard error of the error progressions    
    if include_aad: [time, ofe_mean, ofe_ste, aad_mean, aad_ste] = get_mean_ste_of_n_error_prog(time, err_interp, aad_interp, plot=False, plot_log=plot_log, plot_title=plot_title, line_label=line_label, plot_xlabel=plot_xlabel, plot_ylabel=plot_ylabel)
    else: [time, ofe_mean, ofe_ste] = get_mean_ste_of_n_error_prog(time, err_interp, plot=False, plot_log=plot_log, plot_title=plot_title, line_label=line_label, plot_xlabel=plot_xlabel, plot_ylabel=plot_ylabel)
    print(f"t=[{interpolation_time_min:.2f},{interpolation_time_max:.2f}] | Final avr: AOFE={ofe_mean[-1]:.3f}", end="")
    if include_aad: print(f", AAD={aad_mean[-1]:.3f}")

    # Show the plot of the individual error progressions
    if make_plot:
        for i in range(len(err_interp)):         
            if include_aad: plt.subplot(1,2,2); plt.plot(time, aad_interp[i], linewidth=0.5, alpha=0.2, color="black"); plt.subplot(1,2,1)
            plt.plot(time, err_interp[i], linewidth=0.5, alpha=0.2, color="black")
        plt.tight_layout()
        if show_plot: plt.show()
        
    if return_avr_prog: return time, ofe_mean, ofe_ste, aad_mean, aad_ste

def weighted_average(data, factor=1.5, only_last=False):
    
    len_data = len(data)
    weights = np.array([( factor**(len_data-j))**-1 for j in range(len_data)])
    
    if only_last: 
        return np.sum(data * weights) / np.sum(weights)
    else:
        data_w = np.zeros_like(data)
        for i in range(len_data): data_w[i] = np.sum(data[:i+1] * weights[:i+1]) / np.sum(weights[:i+1])
        return data_w

def exponential_moving_average(data, alpha=0.1):
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for t in range(1, len(data)): ema[t] = alpha * data[t] + (1 - alpha) * ema[t-1]
    return ema

def exponential_weighted_moving_average(data, lambda_=0.1):
    ewma_data = np.zeros_like(data, dtype=float)
    ewma_data[0] = data[0]
    for t in range(1, len(data)): ewma_data[t] = lambda_ * data[t] + (1 - lambda_) * ewma_data[t - 1]
    return ewma_data

def weighted_moving_average(data, weights=None):
    if weights is None: weights = np.arange(1, 10)
    wma_data = np.zeros(len(data) - len(weights) + 1)
    for t in range(len(wma_data)): wma_data[t] = np.dot(data[t:t+len(weights)], weights) / np.sum(weights)
    wma_data = np.concatenate((np.zeros(len(data) - len(wma_data)), wma_data))
    return wma_data

def gaussian_weighted_average(data, window_size=5, sigma=2):
    x = np.linspace(-window_size // 2, window_size // 2, window_size)
    gaussian_weights = np.exp(-x**2 / (2 * sigma**2))
    weights = gaussian_weights / gaussian_weights.sum()
    gwa_data = np.convolve(data, weights, mode='valid')
    gwa_data = np.concatenate((np.zeros(len(data) - len(gwa_data)), gwa_data))
    return gwa_data

####  ---- Saving and Loading Data  ----  ####

for _save_and_load_ in [0]:

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
        
####  ---- Useful functions  ----  ####

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

def gaus_1d(x, *parameters):
    
    n_G = int(len(parameters)/3)
    amp_list = np.asarray(parameters[:n_G])
    mu_list = np.asarray(parameters[n_G:2*n_G])
    sigma_list = np.asarray(parameters[2*n_G:])
     
    return np.sum([amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) for amp, mu, sigma in zip(amp_list, mu_list, sigma_list)], axis=0)
        
def Gauss_fitting_to_CVdata(grid, position, probcutoff=0.0075, min_peak_distance=20, initial_sgima_guess=0.1, periodic=False, print_info=False, show_plot=False):
    
    # make a histogram and find the peaks
    histo, histo_grid = np.histogram(position, bins=grid)
    histo_grid = [ c + (grid[1]-grid[0])/2 for c in histo_grid[:-1] ]
    histo = histo / len(position)
    histo = np.where(histo < probcutoff, 0, histo)
    peaks, _ = find_peaks(histo, distance=min_peak_distance)  
    n_G = len(peaks)
    
    # initial guess for the parameters of the Gaussians
    amp_0 = [ histo[p] for p in peaks ]
    mu_0 = [ histo_grid[p] for p in peaks ]
    sigma_0 = [initial_sgima_guess for _ in peaks ]   

    # bounds for the parameters of the Gaussians
    height_bounds = 0, 1
    position_bounds = min(grid)*1.5, max(grid)*1.5
    sigma_bounds = grid[1] - grid[0], (grid[-1] - grid[0])/4
    bounds= ( np.array([height_bounds[0]]*n_G + [position_bounds[0]]*n_G + [sigma_bounds[0]]*n_G).flatten(), np.array([height_bounds[1]]*n_G + [position_bounds[1]]*n_G + [sigma_bounds[1]]*n_G).flatten() )
    
    # fit the histogram to the sum of Gaussians 
    popt, _ = curve_fit(gaus_1d, histo_grid, histo, p0=np.array([amp_0, mu_0, sigma_0]).flatten(), bounds=bounds)

    # if periodic and the first and last peak are too close (with pbc), delete smaller one.
    if periodic == True:
        if (peaks[0] - (peaks[-1] - len(histo) ) < min_peak_distance):  #if the first and last peak are too close (with pbc), delete smaller one.
            if histo[peaks[0]]>=histo[peaks[-1]]: peaks, popt = np.delete(peaks,-1), np.delete(popt,range(n_G,len(popt),n_G))
            else: peaks, popt=np.delete(peaks,0), np.delete(popt,range(0,len(popt),n_G))
            n_G -= 1

    # remove the parameter of peaks that are outsite of the grid 
    height_list, position_list, sigma_list = popt[:n_G], popt[n_G:2*n_G], popt[2*n_G:]
    valid_indices = [i for i, pos in enumerate(position_list) if min(grid) <= pos <= max(grid)]
    position_list = [position_list[i] for i in valid_indices]
    sigma_list = [sigma_list[i] for i in valid_indices]
    height_list = [height_list[i] for i in valid_indices]
    
    if show_plot:
        plt.figure(figsize=(18,6))
        plt.subplot(1,2,1)
        plt.scatter(position, range(len(position)),s=0.5, alpha=0.2)
        plt.xlabel("CV"); plt.ylabel("count"); plt.title("CV-Trajectory and peaks")
        for p in peaks: plt.axvline(x = grid[p], color = 'r')
        # plt.xlim(min(grid), max(grid))
        
        plt.subplot(1,2,2)
        plt.bar(histo_grid, histo, width=(grid[1]-grid[0]), color='grey', alpha=0.5)
        plt.scatter(grid[peaks], histo[peaks], color='r', s=5)
        for i in range(n_G):
            G_i = gaus_1d(histo_grid,height_list[i],position_list[i],sigma_list[i])            
            plt.plot(histo_grid,np.where( G_i > 1E-5, G_i, np.nan) ,label=str(i)+": h="+str(round(height_list[i],5))+", cv="+str(round(position_list[i],2))+", sigma="+str(round(sigma_list[i], 3)), linewidth=3, alpha=0.5)
        plt.legend()          
        plt.xlabel("CV"); plt.ylabel("relative count"); plt.title("Histogram and fitted Gaussians")	
        # plt.xlim(min(grid), max(grid))
        # plt.yscale('log')
        plt.suptitle("Peaks at CV = " + str(peaks))
        plt.tight_layout()
    
    return height_list, position_list, sigma_list

def Gauss_fitting_to_fes(grid, fes, min_peak_distance=None, initial_sgima_guess=0.1, height_scaling=1, sigma_scaling=1, periodic=False, fix_G_position=False, print_info=False, show_plot=False):
    
    if print_info: start = time.time()
    
    if min_peak_distance is None: min_peak_distance = len(grid)//25
    
    if periodic: grid, fes, grid_length = np.concatenate((grid, grid + (grid[-1] - grid[0]) )), np.concatenate((fes, fes)), max(grid) - min(grid)
    
    # find the peaks and inv_peaks (basins) of the fes
    pos_fes = np.array(fes)
    fes = -fes
    fes = fes - np.min(fes)
    peaks, _ = find_peaks(fes, distance=min_peak_distance)  
    inv_peak, _ = find_peaks(-fes, distance=min_peak_distance)
    inv_peak = np.concatenate(( [0], inv_peak, [len(fes)-1] ))
    n_G = len(peaks)

    # bounds for the parameters of the Gaussians
    height_bounds = 0, np.max(fes) * 1.5
    position_bounds = min(grid)*1.5, max(grid)*1.5
    sigma_bounds = grid[1] - grid[0], (grid[-1] - grid[0])/4
    if fix_G_position: bounds = ( [height_bounds[0], sigma_bounds[0]], [height_bounds[1], sigma_bounds[1]] )
    else: bounds = ( [height_bounds[0], position_bounds[0], sigma_bounds[0]], [height_bounds[1], position_bounds[1], sigma_bounds[1]] )
    height_list , position_list, sigma_list = [], [], []
    
    # Cut the fes at the height of the smaller neighboring inv_peak to aviod fitting the whole FES.
    fes_cut_list, to_fit_list = [], []
    for i, p in enumerate(peaks): 
        
        fes_cut = np.array(fes)
        # find index of neighboring inv_peaks. Use lower inv_peak as cut. 
        for j in range(len(inv_peak)):
            if inv_peak[j] > p: 
                if pos_fes[inv_peak[j]] < pos_fes[inv_peak[j-1]]:   # if the inv_peak on the right is lower
                    cut, cut_i = fes[inv_peak[j]], inv_peak[j]
                    fes_cut[cut_i:] = cut                            # cut the fes at the height of the lower inv_peak ( for indexes > lower inv_peak)
                    for k in range(len(fes_cut[:p])):
                        if fes_cut[p-k] < cut: 
                            fes_cut[:p-k+1] = cut 				   # cut the fes at the height of the lower inv_peak ( for indexes < higher inv_peak)	
                            break
                else: 												# if the inv_peak on the left is lower
                    cut, cut_i = fes[inv_peak[j-1]], inv_peak[j-1]
                    fes_cut[:cut_i] = cut							# cut the fes at the height of the lower inv_peak ( for indexes < lower inv_peak)
                    for k in range(len(fes_cut[p:])):
                        if fes_cut[p+k] < cut: 
                            fes_cut[p+k-1:] = cut					# cut the fes at the height of the lower inv_peak ( for indexes > higher inv_peak)
                            break
                break
        
        fes_cut_list.append(fes_cut)
        to_fit_list.append((fes_cut - min(fes_cut))*height_scaling)
    
    if print_info: print("time start-up: ", round(time.time()-start,6), "s"); time1 = time.time()
    
    # Find optimal parameters for each fes_cut           
    for i, fes_i in enumerate(to_fit_list):
        
        if fix_G_position:  # if fix_G_position = True, use the position of the minima of the basin  as mu for faster curve_fit() 
     
            def Gauss_temp(x, *prms): #prms=amp, sigma
                return prms[0] * np.exp(-((x - x[peaks[i]]) ** 2) / (2 * prms[1] ** 2))   
            
            bounds = ( [height_bounds[0], sigma_bounds[0]], [height_bounds[1], sigma_bounds[1]] )
            initial_guess = np.array([max(fes_i), initial_sgima_guess] )
            prms = curve_fit(Gauss_temp,grid,fes_i,p0=initial_guess,bounds=bounds)[0][:3]
            height_list.append(prms[0])
            position_list.append(grid[peaks[i]])       
            sigma_list.append(prms[1])        
        else:   
            initial_guess = np.array([max(fes_i), grid[np.argmax(fes_i)], initial_sgima_guess] )
            prms = curve_fit(gaus_1d,grid,fes_i,p0=initial_guess,bounds=bounds)[0][:3]
            height_list.append(prms[0])
            position_list.append(prms[1])       
            sigma_list.append(prms[2])
                
    if print_info: print("time optim: ", round(time.time()-time1,6), "s")

    # if periodic and the first and last peak are too close (with pbc), delete smaller one.
    if periodic == True:
        
        # set the parameters of the first Gaussian to the parameters of the first Gaussian of the periodic extension
        n_copy = len(height_list)//2
        height_list[0], sigma_list[0] = height_list[n_copy], sigma_list[n_copy]
        fes_cut_list[0], to_fit_list[0] = np.concatenate((fes_cut_list[n_copy][len(grid)//2:], fes_cut_list[n_copy][:len(grid)//2])), np.concatenate((to_fit_list[n_copy][len(grid)//2:], to_fit_list[n_copy][:len(grid)//2]))
        
        # remove the periodic extension 
        grid, pos_fes, fes = grid[:len(grid)//2], pos_fes[:len(fes)//2], fes[:len(fes)//2]
        fes_cut_list = [fes_cut[:len(fes_cut)//2] for fes_cut in fes_cut_list]
        to_fit_list = [to_fit[:len(to_fit)//2] for to_fit in to_fit_list]
        peaks, _ = find_peaks(fes, distance=min_peak_distance)
        valid_indices = [i for i, pos in enumerate(position_list) if min(grid) <= pos <= max(grid)]

        height_list, position_list, sigma_list, fes_cut_list, n_G = [height_list[i] for i in valid_indices], [position_list[i] for i in valid_indices], [sigma_list[i] for i in valid_indices], [fes_cut_list[i] for i in valid_indices], len(valid_indices)        
        
        if (peaks[0] - (peaks[-1] - len(fes) ) < min_peak_distance):  #if the first and last peak are too close (with pbc), delete smaller one. 
            if fes[peaks[0]]>=fes[peaks[-1]]: peaks, height_list , position_list, sigma_list = np.delete(peaks,-1), height_list[:-1] , position_list[:-1], sigma_list[:-1]
            else: peaks, height_list , position_list, sigma_list = np.delete(peaks,0), height_list[1:] , position_list[1:], sigma_list[1:]
            n_G -= 1
            
    # remove peaks that are too small. 
    valid_indices = [i for i, h in enumerate(height_list) if h > max(pos_fes)/100]
    height_list, position_list, sigma_list, fes_cut_list, n_G = [height_list[i] for i in valid_indices], [position_list[i] for i in valid_indices], [sigma_list[i] for i in valid_indices], [fes_cut_list[i] for i in valid_indices], len(valid_indices)
    
    # # remove the parameter of peaks that are outsite of the grid   # is that necessary? Could there be centre of Gaussians outside of the grid?
    # for _remove_outliers_ in [0]:
    #     valid_indices = [i for i, pos in enumerate(position_list) if min(grid) <= pos <= max(grid)]
    #     height_list, position_list, sigma_list, fes_cut_list, n_G = [height_list[i] for i in valid_indices], [position_list[i] for i in valid_indices], [sigma_list[i] for i in valid_indices], [fes_cut_list[i] for i in valid_indices], len(valid_indices)

    if print_info: print_info("Parameters found.\n" + str(n_G) + " Basins found.") if n_G > 0 else print("No basins found.")
    if print_info: print("Time Total: ", round(time.time()-start,6), "s")      
      
    if show_plot:
        Gauss_list = [ gaus_1d(grid, height_list[i], position_list[i], sigma_list[i]) for i in range(n_G) ]
        area_list = [ np.trapz(G, dx = grid[1]-grid[0]) for G in Gauss_list ]
       
        plt.figure(figsize=(23,6))
        
        # list of 40 colors
        colors = ['g', 'r', 'c', 'm', 'y', 'k', 'orange', 'pink', 'brown', 'purple', 'olive', 'lime', 'teal', 'indigo', 'lavender', 'salmon', 'gold', 'lightblue', 'darkgreen', 'darkred', 'darkblue', 'darkorange', 'darkviolet', 'darkturquoise', 'darkmagenta', 'darkkhaki', 'darkolivegreen', 'darkgoldenrod', 'darkslategray', 'darkseagreen', 'darkslateblue', 'darkorchid', 'darkcyan', 'darkred', 'darkblue', 'darkorange', 'darkviolet', 'darkturquoise', 'darkmagenta']
            
        plt.subplot(1 ,3,1)
        plt.plot(grid, pos_fes, linewidth=5, label="FES", alpha=0.2, color="orange")   
        plt.plot(grid, [np.nan for _ in grid] , label= str(n_G) + " Basins", alpha=0.7, color="grey")
        
        for i, fes_i in enumerate(fes_cut_list):
            i_max = np.argmax(fes_i)
            basin_i = np.where(fes_i < (min(fes_i) + 1E-3), np.nan, -fes_i - (-fes_i[i_max] - pos_fes[i_max]) )
            plt.plot(grid, basin_i, alpha=0.7, color=colors[i])
            
            plt.plot(grid, np.where(to_fit_list[i] > 1E-3, to_fit_list[i] + max(fes), np.nan), color=colors[i], alpha=0.7)        
            plt.plot(grid, np.where(Gauss_list[i] > 1E-3, Gauss_list[i] + max(fes), np.nan), color="pink", linewidth=3, alpha=0.6)
            
            if periodic and i==0 and Gauss_list[0][0] > max(pos_fes)/1000:

                fes_cut_i, max_fes_cut_i = np.array([np.nan for _ in grid]), np.nanmax(basin_i)
                for j in range(1,len(fes_cut_i)+1):
                    if pos_fes[-j] < max_fes_cut_i: fes_cut_i[-j] = pos_fes[-j]
                    else: break
                plt.plot(grid, fes_cut_i, alpha=0.7, color=colors[i])
    
                G_cut_i = np.where(gaus_1d(grid, height_list[i], position_list[i]+grid_length, sigma_list[i]) > 1E-3, gaus_1d(grid, height_list[i], position_list[i]+grid_length, sigma_list[i]) + max(fes), np.nan)
                plt.plot(grid, G_cut_i, color="pink", linewidth=3, alpha=0.6)
                plt.plot(grid, -fes_cut_i*height_scaling + (np.nanmax(G_cut_i)-np.nanmax(-fes_cut_i*height_scaling)), alpha=0.7, color=colors[i])
                   
            if periodic and i==len(fes_cut_list)-1 and Gauss_list[-1][-1] > max(pos_fes)/1000:
                
                fes_cut_i, max_fes_cut_i = np.array([np.nan for _ in grid]), np.nanmax(basin_i)
                for j in range(len(fes_cut_i)):
                    if pos_fes[j] < max_fes_cut_i: fes_cut_i[j] = pos_fes[j]
                    else: break
                plt.plot(grid, fes_cut_i, alpha=0.7, color=colors[i])
                
                G_cut_i = np.where(gaus_1d(grid, height_list[i], position_list[i]-grid_length, sigma_list[i]) > 1E-3, gaus_1d(grid, height_list[i], position_list[i]-grid_length, sigma_list[i]) + max(fes), np.nan)   
                plt.plot(grid, G_cut_i, color="pink", linewidth=3, alpha=0.6)
                plt.plot(grid, -fes_cut_i*height_scaling + (np.nanmax(G_cut_i)-np.nanmax(-fes_cut_i*height_scaling)), alpha=0.7, color=colors[i])
        
            
        plt.plot(grid, [np.nan for _ in grid] , label= str(n_G) + " Gaussians", alpha=0.6, color="pink", linewidth=3)
        plt.title("FES | Basins | Fitted Gaussians"); plt.xlabel("CV"); plt.ylabel("FES | Gaussians [kJ/mol]");
        plt.legend()
            
        plt.subplot(1,3,2)
        # plt.plot(grid, pos_fes, linewidth=5, alpha=0.3, label="FES")   
        for i in range(n_G): 
            plt.fill_between(grid, pos_fes, pos_fes + Gauss_list[i], color=colors[i] , alpha=0.5, label="G" + str(i) + ": h=" + str(round(height_list[i],1)) + " cv=" + str(round(position_list[i],1)) + " sigma=" + str(round(sigma_list[i], 3)))
            if periodic and i==0 and Gauss_list[0][0] > max(pos_fes)/1000: plt.fill_between(grid, pos_fes, pos_fes + gaus_1d(grid, height_list[i], position_list[i] + grid_length, sigma_list[i]), color=colors[i] , alpha=0.5)
            elif periodic and i==n_G-1 and Gauss_list[-1][-1] > max(pos_fes)/1000: plt.fill_between(grid, pos_fes, pos_fes + gaus_1d(grid, height_list[i], position_list[i] - grid_length, sigma_list[i]), color=colors[i] , alpha=0.5)
        plt.legend()
        plt.title("FES and fitted Gaussians added"); plt.xlabel("CV"); plt.ylabel("FES [kJ/mol]")
        
        plt.subplot(1,3,3)
        i_small_area = np.argmin(area_list)  
        for i in range(n_G): 
            G_i = gaus_1d(grid,height_list[i_small_area],position_list[i],sigma_list[i_small_area])
            plt.fill_between(grid , pos_fes, pos_fes + G_i, color=colors[i] , alpha=0.5, label="G" + str(i) + ": h=" + str(round(height_list[i],1)) + " cv=" + str(round(position_list[i],1)) + " sigma=" + str(round(sigma_list[i], 3)))
            if periodic and i==0 and G_i[0] > max(pos_fes)/1000:        plt.fill_between(grid, pos_fes, pos_fes + gaus_1d(grid, height_list[i_small_area], position_list[i] + grid_length, sigma_list[i_small_area]), color=colors[i] , alpha=0.5)
            elif periodic and i==n_G-1 and G_i[-1] > max(pos_fes)/1000: plt.fill_between(grid, pos_fes, pos_fes + gaus_1d(grid, height_list[i_small_area], position_list[i] - grid_length, sigma_list[i_small_area]), color=colors[i] , alpha=0.5)
        plt.legend()
        plt.title(f"FES and smallest fitted Gaussians added (G{i_small_area})"); plt.xlabel("CV"); plt.ylabel("FES [kJ/mol]")
        plt.show()

    return height_list, position_list, sigma_list

####  ---- Print progress  ----  ####

for _print_progress_ in [0]:
    
    def print_progress(start, iteration, total, bar_length=50, variable_name='progress variable' , variable=0, variable_units=""):
        """Function to show a progress bar, that fills up as the iteration number reaches the total. Prints a variable at the end of the progress bar, that can be continiously updated.

        Args:
            iteration (int): Currrent iteration
            total (int): Total iterations
            bar_length (int, optional): Length of the progress bar. Defaults to 50.
            variable_name (str, optional): Name of variable that is being shown at the end of the progress bar. Defaults to 'progress variable'.
            variable (float, optional): Variable that is being shown at the end of the progress bar. Defaults to 0.
        """
        progress = ( (iteration+1) / total)
        arrow = '' + '*' * int(round(bar_length * progress))
        spaces = '' + ' ' * (bar_length - len(arrow))
        
        time_to_now = time.time() - start
        time_to_end = time_to_now / progress - time_to_now
        time_at_end = " at " + str((datetime.now() + timedelta(seconds=time_to_end)).strftime("%H:%M:%S")) if time_to_end < 60 else ""
        
        if progress < 1: print(f'\r|{arrow}{spaces}| {int(progress * 100)}% | {variable_name}: {variable} {variable_units} | Time left: {format_seconds(time_to_end)}', end='', flush=True)
        else: print(f'\r| {int(progress * 100)}% | {variable_name}: {variable} {variable_units} | Total time: {format_seconds(time_to_now)}                                                            ') 
                    
    def live_print_progress(start, iteration, total, bar_length=50, variable_name='progress variable' , variable=0):

        progress = ( (iteration+1) / total)
        arrow = '' + '*' * int(round(bar_length * progress))
        spaces = '' + ' ' * (bar_length - len(arrow))
        
        time_to_now = time.time() - start
        time_to_end = time_to_now / progress - time_to_now
        time_at_end = " at " + str((datetime.now() + timedelta(seconds=time_to_end)).strftime("%H:%M:%S")) if time_to_end < 60 else ""
        
        if progress < 1: print(f'\r|{arrow}{spaces}| {int(progress * 100)}% | {variable_name}: {variable}ns | Time left: {format_seconds(time_to_end)} |{time_at_end}    ', end='', flush=True)
        else: print(f'\r| {int(progress * 100)}% | {variable_name}: {variable}ns | Total time: {format_seconds(time_to_now)} | Finished at {datetime.now().strftime("%H:%M:%S")}                                                         ')

    def format_seconds(total_seconds):
        # Calculate hours, minutes, and seconds
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        # Format as hh:mm:ss or mm:ss based on the length of the duration
        if hours > 0:
            return f"{hours} h : {minutes} min : {seconds:.0f} sec"
        elif minutes > 0:
            return f"{minutes} min : {seconds:.0f} sec"
        else:
            return f"{seconds:.0f} sec"
        
####  ---- plot functions  ----  ####

def plot_FES_Bias_and_Traj(grid, fes, metad_bias=None, static_bias=None, position=None, hills=None, y=None, save_figure_path=None):
     
    plt.figure(figsize=(15,3))
    plt.subplot(1,2,1)
    plt.plot(grid, fes, color="cyan", label="FES")
    if y is not None: plt.plot(grid, y, color="grey", alpha=0.3, label="Analytical")
    else: y = np.array(fes)
    
    if static_bias is not None: plt.fill_between(grid, y, y + static_bias, color="blue", alpha=0.7, label="Static Bias")
    else: static_bias = np.zeros_like(y)
    if metad_bias is not None: plt.fill_between(grid, y + static_bias, y + static_bias + metad_bias, color="red", alpha=0.3, label="MetaD Bias")
    plt.title("FES and Bias"); plt.xlabel("CV"); plt.ylabel("Energy"); plt.legend(fontsize=8)

    plt.subplot(1,2,2)
    if hills is not None: plt.scatter(hills[:,0], hills[:,1], facecolors='none', edgecolors='blue', label="Hills")
    if position is not None: plt.scatter(position[:,0], position[:,1], facecolors='none', edgecolors='red', alpha=0.2, label="Position", s=1)
    plt.title("Trajectory"); plt.xlabel("time"); plt.ylabel("CV")
    plt.tight_layout()
    if save_figure_path is not None: plt.savefig(save_figure_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.show()	

def plot_PRTR_d_error(mfi_PRTR, plot_aad=False):	
    
    if not plot_aad: title_list = ["dAofe", "dAofe/dt", "dtotAofe", "dtotAofe/dt"]
    else: title_list = ["dAAD", "dAAD/dt", "dtotAAD", "dtotAAD/dt"]
    y_label_list = ["Error [kJ/mol]", "Error [kJ/mol/ns]", "Error [kJ/mol]", "Error [kJ/mol/ns]"]
    plt.figure(figsize=(20,4))
 
    for sim in range(len(mfi_PRTR.sim)):
        
        if len(mfi_PRTR.sim[sim].Avr_Error_list[:,0]) < 2: print(f"SIM {sim} too short: len_err={len(mfi_PRTR.sim[sim].Avr_Error_list[:,0])}"); continue
        
        for err_type in range(4):
            plt.subplot(1,4,err_type+1)
            plt.title(title_list[err_type]); plt.xlabel("Time [ns]"); plt.ylabel(y_label_list[err_type])
   
            if sim == 0: 
                t = mfi_PRTR.sim[sim].Avr_Error_list[1:,0]
                error = mfi_PRTR.sim[sim].d_AAD[:, err_type] if plot_aad else mfi_PRTR.sim[sim].d_Aofe[:, err_type]
                plt.plot(t, error, label=f"S {sim}", linewidth=3, alpha=0.2)
            else: 
                t = mfi_PRTR.sim[sim].Avr_Error_list[1:,0]
                t = [t[0] + (t[i] - t[0])*4 for i in range(len(t))]
                error = mfi_PRTR.sim[sim].d_AAD[:, err_type] if plot_aad else mfi_PRTR.sim[sim].d_Aofe[:, err_type]
                plt.plot(t , error, label=f"S {sim}", marker=".")
    
            plt.legend(fontsize=8)

    plt.tight_layout(); plt.show()

def large_error_plot(mfi_PRTR, time_percentage=100, plot_guaranteed_sim_time=False, plot_double_error_dashed=False, plot_also_aad=True):
    
    # set variables and find time limit
    workers = mfi_PRTR.workers
    sim0 = mfi_PRTR.sim[0]
    min_sim_t = sim0.guaranteed_sim_time*workers
    
    if time_percentage >= 100: 
        ts = len(sim0.Avr_Error_list)
        time_limit = sim0.Avr_Error_list[-1,0]
        plot_sim_index = [i for i in range(1, len(mfi_PRTR.sim)) if (len(mfi_PRTR.sim[i].Avr_Error_list) > 0)]
    else: 
        ts = int(len(sim0.Avr_Error_list)// (1/(time_percentage*0.01)))
        time_limit = sim0.Avr_Error_list[ts,0]
        plot_sim_index = [i for i in range(1, len(mfi_PRTR.sim)) if (mfi_PRTR.sim[i].Avr_Error_list[0,0] < time_limit and len(mfi_PRTR.sim[i].Avr_Error_list) > 2)]
            
    t = sim0.Avr_Error_list[:ts,0]
    t_dt = sim0.Avr_Error_list[1:ts+1,0]
    
    def get_t(i, i_start=0):
        t_sim = mfi_PRTR.sim[i].Avr_Error_list[i_start:,0]
        t_sim = [t_sim[0] + (t_i - t_sim[0])*workers for t_i in t_sim if t_sim[0] + (t_i - t_sim[0])*workers < time_limit]
        return t_sim

    plt.figure(figsize=(20,16)) if plot_also_aad else plt.figure(figsize=(20,8))
    
    # plot Aofe
    plt.subplot(4,1,1) if plot_also_aad else plt.subplot(2,1,1)
    plt.plot(t, sim0.Avr_Error_list[:ts, 2] , label=f"S {0}", color="grey", linewidth=3, alpha=0.5)
    for i in plot_sim_index: plt.plot(get_t(i) , mfi_PRTR.sim[i].Avr_Error_list[:len(get_t(i)), 2] , label=f"S {i}", marker=".")
    plt.yscale("symlog", linthresh=0.0001)
    if plot_double_error_dashed: plt.plot(t, 2 * sim0.Avr_Error_list[:ts, 2], color="grey", linewidth=3, alpha=0.2, linestyle="--")
    if plot_guaranteed_sim_time: plt.axvline(min_sim_t, color="yellow", linestyle="--", alpha=0.4)
    if mfi_PRTR.main_error_type == "ST_ERR": plt.axhline(mfi_PRTR.goal, color="green", linestyle="--", alpha=0.4)
    plt.legend(fontsize=8, ncol=3); plt.title("Simulation Aofe"); plt.xlabel("Time [ns]"); plt.ylabel("Error [kJ/mol]")    
    


    # plot dAofe/dt
    plt.subplot(4,1,2) if plot_also_aad else plt.subplot(2,1,2)
    plt.plot(t_dt, sim0.d_Aofe_w[:ts, 1] , label=f"S {0}", color="grey", linewidth=3, alpha=0.5)
    for i in plot_sim_index: plt.plot(get_t(i,1) , mfi_PRTR.sim[i].d_Aofe_w[:len(get_t(i,1)), 1] , label=f"S {i}", marker=".")
    plt.yscale("symlog", linthresh=0.0001)
    if plot_double_error_dashed: plt.plot(t_dt, 2 * sim0.d_Aofe_w[:ts, 1] , color="grey", linewidth=3, alpha=0.2, linestyle="--")
    if plot_guaranteed_sim_time: plt.axvline(min_sim_t, color="yellow", linestyle="--", alpha=0.4)
    plt.legend(fontsize=8, ncol=3); plt.title("weighted dAofe/dt"); plt.xlabel("Time [ns]"); plt.ylabel("Error [kJ/mol/ns]")    
    
        # plot AAD
    if plot_also_aad:
        i_aad = sim0.aad_index
        
        plt.subplot(4,1,3)
        plt.plot(t, sim0.Avr_Error_list[:ts, i_aad] , label=f"S {0}", color="grey", linewidth=3, alpha=0.5)
        for i in plot_sim_index: plt.plot(get_t(i) ,mfi_PRTR.sim[i].Avr_Error_list[:len(get_t(i)), i_aad] , label=f"S {i}", marker=".")
        plt.yscale("symlog", linthresh=0.0001)
        if plot_double_error_dashed: plt.plot(t, 2 * sim0.Avr_Error_list[:ts, i_aad] , color="grey", linewidth=3, alpha=0.2, linestyle="--")
        if plot_guaranteed_sim_time: plt.axvline(min_sim_t, color="yellow", linestyle="--", alpha=0.4)
        if mfi_PRTR.main_error_type == "AAD": plt.axhline(mfi_PRTR.goal, color="green", linestyle="--", alpha=0.4)
        plt.legend(fontsize=8, ncol=3); plt.title("Simulation AAD"); plt.xlabel("Time [ns]"); plt.ylabel("Error [kJ/mol/ns]")    

        # plot dAAD/dt
        plt.subplot(4,1,4)
        plt.plot(t_dt, sim0.d_AAD_w[:ts, 1], label=f"S {0}", color="grey", linewidth=3, alpha=0.5)
        for i in plot_sim_index: plt.plot(get_t(i,1) , mfi_PRTR.sim[i].d_AAD_w[:len(get_t(i,1)), 1] , label=f"S {i}", marker=".")
        plt.yscale("symlog", linthresh=0.0001)
        if plot_double_error_dashed: plt.plot(t_dt, 2 * sim0.d_AAD_w[:ts, 1], color="grey", linewidth=3, alpha=0.2, linestyle="--")
        if plot_guaranteed_sim_time: plt.axvline(min_sim_t, color="yellow", linestyle="--", alpha=0.4)
        plt.legend(fontsize=8, ncol=3); plt.title("weighted dAAD/dt"); plt.xlabel("Time [ns]"); plt.ylabel("Error [kJ/mol/ns]")    

    plt.tight_layout(); plt.show()


