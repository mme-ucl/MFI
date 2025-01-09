
import os
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import signal
import matplotlib.ticker as ticker
from matplotlib import colors
import copy
import dill
import time
from IPython.display import clear_output
from pympler import asizeof
from typing import List, Optional, Dict
from dataclasses import dataclass, field, asdict

import MFI_lib2D as lib2

#Class MFI 2D
@dataclass
class MFI2D:
    
    for init_parameters in [0]:
    
        #Grid parameters
        X: np.ndarray = field(default=None) # 2D grid for the CV, 1st dimension.
        Y: np.ndarray = field(default=None) # 2D grid for the CV, 2nd dimension.
        grid_x: np.ndarray = field(default=None) # 1D grid for the CV, 1st dimension.
        grid_y: np.ndarray = field(default=None) # 1D grid for the CV, 2nd dimension.
        grid_min: list = field(default=None) # minimum values of the grid, [x_min, y_min]
        grid_max: list = field(default=None) # maximum values of the grid, [x_max, y_max]
        grid_n: list = field(default_factory=lambda: [100, 100]) # number of grid points in each dimension, [n_x, n_y]
        periodic: list = field(default_factory=lambda: [False, False]) # periodicity of the grid, [x_periodic, y_periodic]
        periodic_range: list = field(default_factory=lambda: [None, None]) # the periodic is used to find periodic points. If None, the periodic_range is set to 1/4 of the grid length [length_x/4, length_y/4], which will only make periodic copies for points that are within this range from the edge of the grid.

        # plumed grid
        plX: np.ndarray = field(default=None) # 2D grid for the CV, 1st dimension.
        plY: np.ndarray = field(default=None) # 2D grid for the CV, 2nd dimension.
        pl_min: list = field(default=None) # minimum values of the grid, [x_min, y_min]
        pl_max: list = field(default=None) # maximum values of the grid, [x_max, y_max]
        pl_n: list = field(default=None) # number of grid points in each dimension, [n_x, n_y]    
            
        ### Simulation parameters
        simulation_steps: int = None # number of simulation steps.
        time_step: float = 0.005 # time step of the simulation in ps.
        initial_position: list = field(default_factory=lambda: [None, None]) # initial position for the simulation in the units of the collective variable [x, y] ([cv1, cv2]).
        kT: float = 1 # temperature of the simulation in reduced units. (kT = 1: 120K, kT = 2.5: 300K)
        start_sim: bool = True # if True, the simulation is started without needing to call the run_simulation function.
        
        ### Simulation paths and data
        simulation_folder_path: str = None # path to the simulation folder
        ID: str = "" # ID of the files. Will turn the "file_name" into "file_name"+ID
        hills_file: str = "HILLS" # path to the hills file (either already existing or to be using in the plumed input file). ID is added automatically. 
        position_file: str = "position" # path to the position (cv) file (either already existing or to be using in the plumed input file). ID is added automatically.
        hills: np.ndarray = field(default=None) # hills data
        position: np.ndarray = field(default=None) # position (cv) data
        base_forces: np.ndarray = field(default=None)  # [PD, PD2, Force_x, Force_y, ofv_num_x, ofv_num_y] # if provided, the forces are patched to the base_forces before the error calculation.

        ### System (to be simulated) parameters
        System: str = "Langevin2D" # The system that is used to run the simulating. The supported systems are: ["Langevin", "Langevin2D", "gmx", "gromacs", "GMX", "GROMACS"]
        cv_name: list = field(default_factory=lambda: ["p.x", "p.y"]) # name of the collective variable used in the plumed files.
        plumed_dat_text: str = field(default_factory=lambda: "p: DISTANCE ATOMS=1,2 COMPONENTS\nff: MATHEVAL ARG=p.x,p.y FUNC=(1.34549*x^4+1.90211*x^3*y+3.92705*x^2*y^2-6.44246*x^2-1.90211*x*y^3+5.58721*x*y+1.33481*x+1.34549*y^4-5.55754*y^2+0.904586*y+18.5598) PERIODIC=NO\nbb: BIASVALUE ARG=ff\n")
        trajectory_xtc_file_path_list: List[str] = field(default_factory=lambda: []) # list of paths to an existing trajectory files, to find initial strucutres for the simulation
        structure_gro_file_path: str = None # path to an existing structure file.
        mdp_file_path: str = None # path to the mdp file.
        top_file_path: str = None # path to the topology file.
        tpr_file_path: str = None # path to the tpr file.
        pdb_file_path: str = None # path to the pdb file.
        find_sim_init_structure: bool = False # if True, the initial structure for the simulation is found from the trajectory file. Otherwise, the structure from the gro file is used. 
        find_structure_text: str = None # This is the text that will be written to the plumed.dat file. This text should include everything upto the lines containing the UPDADE_IF and DUMPATOMS commands. If None, plumed_dat_text will be used.
        make_tpr_input_file: bool = False # if True, the tpr file is made from the mdp file and the gro file.

        #Biasing parameters
        position_pace: int = 20  # 1 position (cv) recoeded every position_pace * Molecular_Dynamics_time_step  ### important for simulation input and for combining probability densities with differnt position_pace
        n_pos_per_window: float = 10 # number of positions per hill ### important for loading history dependent bias (like MetaD)  ### should be ~ 10 - 20.
        metad_pace: int = 200 # 1 hill every metad_pace * Molecular_Dynamics_time_step ### should be about 10 - 20 times the position_pace.
        metad_width: list = field(default=None) # width of the Gaussian hill for the metadynamics bias in the units of the collective variable [x, y] ([cv1, cv2]).
        metad_height: float = None # height of the Gaussian hill for the metadynamics bias.
        biasfactor: float = None # biasfactor for the metadynamics bias.
        WellTempered: bool = True # if True, the well-tempered metadynamics is used.
        external_bias_file: str = None # path to the external bias file.
        hp_centre_x: float = None; hp_kappa_x:  float = None; hp_centre_y: float = None; hp_kappa_y:  float = None # harmonic potential parameters. Kappa is the force constant, and centre specifies the location of the harmonic potential.
        lw_centre_x: float = None; lw_kappa_x:  float = None; lw_centre_y: float = None; lw_kappa_y:  float = None # lower wall parameters (just the lower half of a harmonic potential). Kappa is the force constant, and centre specifies the location of the lower wall.
        uw_centre_x: float = None; uw_kappa_x:  float = None; uw_centre_y: float = None; uw_kappa_y:  float = None # upper wall parameters (just the upper half of a harmonic potential). Kappa is the force constant, and centre specifies the location of the upper wall.
        Bias_static: np.ndarray = field(default=None) # initial static bias potential
        Force_static_x: np.ndarray = field(default=None) # initial static bias force in the x direction
        Force_static_y: np.ndarray = field(default=None) # initial static bias force in the y direction
        Bias_sf: float = 1 # scaling factor for the external bias. A value of 1 will not do anything. A value of 0.95 will result in a effect simillar to well-tempred metadynamics. A value of 1.05 will result in a bias that encourages the sampling of high energy states.
        gaus_filter_sigma: float = None # sigma for the Gaussian filter used to smooth the external bias.
        bias_type: str = "" # type of the static bias generated. If "", the bias is constructed using the estimate of the FES (resulting in a nearly flat energy surface). If "error", the bias is constructed using the FES and in addition high error (ofe) regions have a lower energy, encouraging sampling of those regions. If "PD", simmilar to "error", encouraging sampling of regions with lower probability density (PD)
        
        ### MFI parameters
        n_pos: int = -1
        bw: list = field(default_factory=lambda: [0.1, 0.1]) # bandwidth of the Gaussian for the calculation of the probability density. [bw_x, bw_y]

        ### Error calculation variables   
        PD_limit: float = 1E-10  # probability density values below this limit are set to zero and are ignored for numerical stability
        PD_cutoff: float = 1 # grid points with probability density values below this limit will not contribuite to the error calculation. This is for the case when the probability density is so low that it is not revelant. 
        FES_cutoff: float = None # grid points with FES values above this limit will not contribuite to the error calculation. This is for the case when the FES is so high that it is not revelant.
        len_error_list: int = 50 # number of hills / position points between error calculations
        i_error_list: np.ndarray = field(default=None)
        calculate_error_change: bool = False # if True, the change in the error is calculated.
        weighted_avr_error_change: bool = True # if True, and calculate_error_change is True, an additional change in error is calculated as an average of past errors with past error having an exponentially decreasing weight.
        use_ST_ERR: bool = True # if True, the standard error of the mean is used in the error calculation. Otherwise, the standard deviation is used.
        use_VNORM: bool = False # if True, the error is normalised by the ratio of the explored space to the total space.
        bootstrap_iter: int = None # number of bootstrap iterations for finding the error of the FES. If None, the error is not calculated.
        calculate_FES_st_dev: bool = False # if True, the variance of the FES is calculated.
        non_exploration_penalty: float = None  # grid points that have not been explored are assigned this penalty. If None, the penalty is not applied.  (what about FES_cutoff?)
        base_time: float = 0 # simulation time in ns before the current simulation used for the error calculation.
        main_error_type: str = None # the main error type used for the error calculation. If None, the main error type is the AAD if y is provided, otherwise the main error type bootstrapped error, if calculated, otherwise the main error type is Aofe (average on the fly error).        

        ### Variables fort the calculation of the absolute deviation to a reference force and FES
        Z: np.ndarray = field(default=None) # reference Free Energy Surface
        dZ_dX: np.ndarray = field(default=None) # reference force in the x direction
        dZ_dY: np.ndarray = field(default=None) # reference force in the y direction

        # ### Variables for real time analysis (SRTR/PRTR campaigns)
        n_pos_analysed: list = field(default_factory = lambda: [0]) # list of the number of positions analysed. The 0th element is the total number of positions analysed, and the rest are the number of positions analysed at each analysis iteration.
        goal: float = 0.01 # goal value for the error calculation. The simulation stops when the error is below this value or the time_budget is reached.
        guaranteed_sim_time: float = None  # guaranteed simulation time in ns. Simulations will be run until this time is reached (exceptions for exploration phase and umbrella sampling simulations).
        max_sim_time: float = None # maximum simulation time in ns. 
        time_budget: float = None # time budget in ns. The simulations campaign will be stoped if this time is reached.

        ### Boolean Variables for saving, printing and plotting
        save_error_progression: bool = True # if True, the error progression is saved as error_progression{ID}.pkl
        save_force_terms: bool = True # if True, the force terms [PD, PD2, Force_x, Force_y, ofv_num_x, ofv_num_y] are saved as force_terms{ID}.pkl
        save_results: bool = False # if True, the results are saved as results{ID}.pkl . The results include (n_pos_analysed,force_terms, Avr_Error_list, Maps (FES, cutoff, ofe, AAD, ...)(if record_maps), intermediate force terms (if record_forces_e))
        save_mfi_instance: bool = False # if True, the MFI instance is saved as MFI_instance{ID}.pkl
        record_maps: bool = False # if True, the FES, cutoff, ofe, (AAD, ABS_error, Avr_FES_st_dev) are saved for each error calculation.
        record_forces_e: bool = False # if True, the [PD, PD2, Force_x, Force_y, ofv_num_x, ofv_num_y] are saved for each error calculation.
        
        print_info: bool = True
        save_figures: bool = False
        # live_plot: bool = False
        # save_video: bool = False
    
    def __post_init__(self):

        # check if system is supported and set up the simulation files path dictionary
        supported_systems = ["Langevin","Langevin2D", "gmx", "gromacs", "GMX", "GROMACS"]
        if self.System not in supported_systems: raise ValueError(f"System {self.System} is not supported. The supported systems are: {supported_systems}")
        self.sim_files_path = {"trajectory_xtc_file_path_list": self.trajectory_xtc_file_path_list, "structure_gro_file_path": self.structure_gro_file_path, 
                               "mdp_file_path": self.mdp_file_path, "top_file_path": self.top_file_path, "tpr_file_path": self.tpr_file_path, "pdb_file_path":self.pdb_file_path}
        
        # Set up the grid
        if self.X is None or self.Y is None: self.find_grid()
        self.grid_x, self.grid_y = self.X[0, :], self.Y[:, 0]
        self.grid_dx, self.grid_dy = self.grid_x[1] - self.grid_x[0], self.grid_y[1] - self.grid_y[0]
        self.grid_min = [self.grid_x[0], self.grid_y[0]]
        self.grid_max = [self.grid_x[-1], self.grid_y[-1]]
        self.grid_n = [len(self.grid_x), len(self.grid_y)]
        self.nbins_yx = [self.grid_n[1], self.grid_n[0]]
        self.grid_length = [self.grid_max[0] - self.grid_min[0], self.grid_max[1] - self.grid_min[1]]
        if self.periodic and all(pr == None for pr in self.periodic_range): self.periodic_range = [self.grid_length[0]/4, self.grid_length[1]/4]
        
        # set up plumed grid 
        [self.plX, self.plY, self.pl_min, self.pl_max, self.pl_n, self.pl_extra] = lib2.get_plumed_grid_2D(self.X, self.Y, periodic=self.periodic)
                
        # Initialise the arrays
        self.Bias = np.zeros(self.nbins_yx)
        self.Force_bias_x = np.zeros(self.nbins_yx)
        self.Force_bias_y = np.zeros(self.nbins_yx)
        self.PD = np.zeros(self.nbins_yx)
        self.PD2 = np.zeros(self.nbins_yx)
        self.Force_num_x = np.zeros(self.nbins_yx)
        self.Force_num_y = np.zeros(self.nbins_yx)
        self.Force_x = np.zeros(self.nbins_yx)
        self.Force_y = np.zeros(self.nbins_yx)
        self.ofv_num_x = np.zeros(self.nbins_yx)  
        self.ofv_num_y = np.zeros(self.nbins_yx)  
        self.FES = np.zeros(self.nbins_yx)
        
        # Calculate static bias potential
        if self.Bias_static is None and (self.Force_static_x is not None and self.Force_static_y is not None): self.Bias_static = lib2.FFT_intg_2D(self.Force_static_x, self.Force_static_y, self.grid_min, self.grid_max, self.periodic)
        elif self.Bias_static is not None and (self.Force_static_x is None and self.Force_static_y is None): self.Force_static_y, self.Force_static_x = np.gradient(self.Bias_static, self.grid_dy, self.grid_dx)
        if self.Bias_static is None: self.Bias_static = np.zeros(self.nbins_yx)
        if self.Force_static_x is None: self.Force_static_x = np.zeros(self.nbins_yx)      
        if self.Force_static_y is None: self.Force_static_y = np.zeros(self.nbins_yx)      
        self.make_harmonic_bias()

        # Initialise lists for saving data:  initialise error_list
        self.Avr_Error_info = ["time", "ratio_explored", "Aofe"]
        if self.Z is not None: 
            self.Avr_Error_info.append("AAD")
            self.aad_index = self.Avr_Error_info.index("AAD")
        if self.dZ_dX is not None and self.dZ_dY is not None: 
            self.Avr_Error_info.append("AAD_Force")
            self.aad_force_index = self.Avr_Error_info.index("AAD_Force")
        if self.bootstrap_iter is not None: 
            self.record_forces_e = True
            self.Avr_Error_info.append("ABS_error")
            self.abs_error_index = self.Avr_Error_info.index("ABS_error")
        if self.calculate_FES_st_dev:
            self.record_maps = True
            self.Avr_Error_info.append("FES_st_dev")
            self.FES_st_dev_index = self.Avr_Error_info.index("FES_st_dev")

        if self.main_error_type is None: 
            if self.Z is not None: self.main_error_type = "AAD"
            else: self.main_error_type = "ST_ERR"
        if self.main_error_type not in ["AAD", "ST_ERR"]: raise ValueError(f"main_error_type must be either 'AAD' or 'ST_ERR'. Please change or add more error_types to code")

        # if calculate_error_change is True, initialise the arrays for saving the error change 

        # initialise arrrays for saving errors
        self.Avr_Error_list = np.empty((0, len(self.Avr_Error_info))) # [time, ratio_explored, Aofe, AAD_FES, AAD_F, ABS_error]
        if self.record_maps: self.Maps_list = np.empty((0, len(self.Avr_Error_info), self.nbins_yx[0], self.nbins_yx[1]))   ### [FES, cutoff, ofe, AD_FES, AD_force, BS_error]
        if self.record_forces_e: self.forces_e_list = np.empty((0,4, self.nbins_yx[0], self.nbins_yx[1]))  # forces_e = [PD_i, PD2_i, Force_i, ofv_num_i ]
        # if calculate_error_change is True, initialise the arrays for saving the error change
        if self.calculate_error_change: 
            self.d_Aofe = np.empty((0,4))
            if self.Z is not None: self.d_AAD = np.empty((0,4))
            if self.weighted_avr_error_change: 
                self.d_Aofe_w = np.empty((0,4))
                if self.Z is not None: self.d_AAD_w = np.empty((0,4))
            else: self.d_Aofe_w, self.d_AAD_w = None, None
        else: self.d_Aofe, self.d_AAD, self.d_Aofe_w, self.d_AAD_w = None, None, None, None
        
        #check if n_pos_per_window is consistent with metad_pace and position_pace
        if self.metad_height is not None and self.metad_height > 0:
            if self.metad_pace is not None and self.position_pace is not None and self.n_pos_per_window is not None: 
                assert self.metad_pace % self.position_pace == 0, "metad_pace must be a multiple of position_pace"
                assert int(round(self.metad_pace / self.position_pace)) == int(self.n_pos_per_window), "metad_pace / position_pace must be equal to n_pos_per_window"
            elif self.metad_pace is not None and self.position_pace is not None:
                self.n_pos_per_window = int(round(self.metad_pace / self.position_pace))
            elif self.n_pos_per_window is not None and self.position_pace is not None:
                self.metad_pace = self.n_pos_per_window * self.position_pace
            elif self.n_pos_per_window is not None and self.metad_pace is not None:
                self.position_pace = self.metad_pace / self.n_pos_per_window
            elif self.metad_pace is not None: self.n_pos_per_window, self.position_pace = 10, self.metad_pace / 10
            else: raise ValueError("metad_pace, position_pace, and n_pos_per_window are not provided. Provide at least metad_pace")
        
        # set up bw (bandwidth) and const (constant) for the Gaussian. (constant allows to compare / combine Probability densities that use different position_pace and/or different bw)
        self.bw2 = [self.bw[0] ** 2, self.bw[1] ** 2, self.bw[0] * self.bw[1]]
        self.const = self.position_pace / (self.bw[0] * self.bw[1])
        
        # set up data file names
        if self.hills_file is not None and self.ID is not None: self.hills_file = self.hills_file + self.ID
        if self.position_file is not None and self.ID is not None: self.position_file = self.position_file + self.ID

        # set up the simulation folder path
        if self.simulation_folder_path is None: self.simulation_folder_path = os.getcwd() + "/"
        if self.simulation_folder_path[-1] != "/": self.simulation_folder_path += "/"
        lib2.set_up_folder(self.simulation_folder_path)

        # set up variables
        self.sim_time = 0
        self.phase = None
        
    def save_instance(self, save_instance_path=None, use_parent_instance=None):
        
        # set up the path to save the instance
        if save_instance_path is None: save_instance_path = self.simulation_folder_path + f"MFI_instance{self.ID}.pkl"
        
        # save the instance
        with open(save_instance_path, 'wb') as file:
            if use_parent_instance is None: dill.dump(self, file)
            else: dill.dump(use_parent_instance, file)

    def load_instance(self, save_instance_path=None):

        # set up the path where the instance is saved
        if save_instance_path is None: save_instance_path = self.simulation_folder_path + f"MFI_instance{self.ID}.pkl"

        # laod the instance
        with open(save_instance_path, 'rb') as file:
            return dill.load(file)
            
    def run_simulation(self, assign_process=False, simulation_path=None, file_extension=None, n_cores_per_simulation=None, sim_files_path=None):     

        # set up the simulation folder path and the file extension
        if simulation_path is None: simulation_path = self.simulation_folder_path
        if simulation_path is not None and simulation_path != "": lib2.set_up_folder(simulation_path)
        if file_extension is None: file_extension = self.ID
        if sim_files_path is None: sim_files_path = self.sim_files_path

        # set up the simulation steps and grid
        if self.simulation_steps is None: raise ValueError("\n ***** The number of steps are not provided ***** \n")
        if self.plX is None or self.plY is None: self.plX, self.plY, self.pl_min, self.pl_max, self.pl_n, self.pl_extra = lib2.get_plumed_grid_2D(self.X, self.Y, self.periodic)
        else: 
            if self.pl_n is None: self.pl_n = [len(self.plX[0, :]), len(self.plY[:, 0])]
            if self.pl_min is None or self.pl_max is None: self.pl_min, self.pl_max = [self.plX[0, 0], self.plY[0, 0]], [self.plX[-1, -1], self.plY[-1, -1]]

        # set up the initial position. If not provided, the initial position is randomly selected from the grid.
        if self.initial_position[0] is None:
            np.random.seed()
            i_pos_x = round(self.grid_min[0] + self.grid_length[0] / 1000 * np.random.randint(0, 1000), 3)
        else: i_pos_x = self.initial_position[0]
        if self.initial_position[1] is None: 
            np.random.seed()
            i_pos_y = round(self.grid_min[1] + self.grid_length[1] / 1000 * np.random.randint(0, 1000), 3)
        else: i_pos_y = self.initial_position[1]

        # if assign_process is True, the simulation is started in a separate process, and the process PID is returned            
        start_sim = False if assign_process else self.start_sim

        # start the simulation instance and start the simulation if start_sim is True (and assign_process is False)
        simulation = lib2.Run_Simulation(# plumed grid
                                        pl_X=self.plX, pl_Y=self.plY, periodic=self.periodic,
                                        # simulation parameters
                                        System=self.System, cv_name=self.cv_name, n_steps=self.simulation_steps, time_step=self.time_step, temperature=self.kT, n_cores_per_simulation=n_cores_per_simulation,
                                        make_tpr_input_file=self.make_tpr_input_file, find_sim_init_structure=self.find_sim_init_structure, initial_position=[i_pos_x, i_pos_y], find_structure_text=self.find_structure_text,
                                        # metadynamics biasing parameters
                                        plumed_dat_text=self.plumed_dat_text, metad_width=self.metad_width, metad_height=self.metad_height, biasfactor=self.biasfactor,
                                        metad_pace=self.metad_pace, position_pace=self.position_pace, n_pos_per_window=self.n_pos_per_window, 
                                        # static biasing parameters
                                        external_bias_file=self.external_bias_file, 
                                        hp_centre_x=self.hp_centre_x, hp_centre_y=self.hp_centre_y, hp_kappa_x=self.hp_kappa_x, hp_kappa_y=self.hp_kappa_y, 
                                        lw_centre_x=self.lw_centre_x, lw_centre_y=self.lw_centre_y, lw_kappa_x=self.lw_kappa_x, lw_kappa_y=self.lw_kappa_y,
                                        uw_centre_x=self.uw_centre_x, uw_centre_y=self.uw_centre_y, uw_kappa_x=self.uw_kappa_x, uw_kappa_y=self.uw_kappa_y, 
                                        # information about the simulation files, wether to start the simulation, and wether to print information
                                        file_extension=file_extension, sim_files_path=sim_files_path, start_sim=start_sim, print_info=self.print_info)

        # if assign_process is True, start the simulation and return the process PID    
        if assign_process: self.process = simulation.start_sim_return_process()

    def make_external_bias(self, FES=None, Bias=None, Bias_sf=None, gaus_filter_sigma=None, FES_cutoff=None, external_bias_file=None, error=None, PD=None, bias_type=None):
        
        if Bias_sf == 0: return
                
        # get the external (static) bias parameters
        if FES is not None: Bias = None
        elif Bias is not None: FES = None
        else: FES, Bias = self.FES, None       
        
        if bias_type is None: bias_type = self.bias_type
        if Bias_sf is None: Bias_sf = self.Bias_sf if (bias_type is None or bias_type == "" or self.phase == "metad") else 1 
        if gaus_filter_sigma is None: gaus_filter_sigma = self.gaus_filter_sigma
        if FES_cutoff is None: FES_cutoff = self.FES_cutoff
        if external_bias_file is None: external_bias_file = self.external_bias_file
        
        # if bias_type is "transition_path", the external bias is created without a Gaussian filter. The Gaussian filter is applied to the creation of the transition path bias.
        if bias_type == "transition_path": 
            gaus_filter_sigma_trans_path = gaus_filter_sigma
            gaus_filter_sigma = None
                
        # make the external (static) bias              
        Bias_static, Force_static_x, Force_static_y, self.external_bias_file = lib2.make_external_bias_2D(self.X, self.Y, FES=FES, Bias=Bias, Bias_sf=Bias_sf, gaus_filter_sigma=gaus_filter_sigma, FES_cutoff=FES_cutoff, pl_min=self.pl_min, pl_max=self.pl_max, periodic=self.periodic, return_array=True, cv_name=self.cv_name)

        # if bias_type is not provided, add the bias without further modification. Otherwise, modify the bias according to the bias_type     
        if bias_type is None or bias_type == "" or self.phase == "metad": 
            self.Force_static_x += Force_static_x
            self.Force_static_y += Force_static_y
            self.Bias_static += Bias_static
            return

        if bias_type == "error":
            if error is None: error = np.array(self.ofe)
            error = gaussian_filter(error, 6) if any(self.periodic) is False else gaussian_filter(error, 6, mode="wrap")
            error = -error - np.min(-error)
            Bias_static += error * 10 / np.max(error)## scale Bias so that the max ~ [1 - 3] kJ/mol
            Bias_static = Bias_static - np.min(Bias_static)
        
        elif bias_type == "PD":
            if PD is None: PD = np.array(self.PD)
            PD = gaussian_filter(PD, 6) if any(self.periodic) is False else gaussian_filter(PD, 6, mode="wrap")
            Bias_static += PD * 3 / np.max(PD)## scale Bias so that the max ~ [1 - 3] kJ/mol
            Bias_static = Bias_static - np.min(Bias_static)
            
        elif bias_type == "transition_path":
            
            ######### temp plot
            # plt.contourf(self.X, self.Y, Bias_static, cmap="coolwarm"); plt.colorbar(); plt.title("Bias0"); plt.show()
            
            if FES is None: FES = self.FES
            ######### temp plot
            ZZZ = lib2.create_valley_surface(self.X, self.Y, FES=FES, FES_gaus_filter=gaus_filter_sigma_trans_path, periodic=self.periodic)
            # plt.contourf(self.X, self.Y, ZZZ, cmap="coolwarm"); plt.colorbar(); plt.title("ZZZ"); plt.show() ######### temp plot
            Bias_static += ZZZ
            Bias_static = Bias_static - np.min(Bias_static)                        
            # plt.contourf(self.X, self.Y, Bias_static, levels=np.linspace(0,50,21), cmap="coolwarm"); plt.colorbar(); plt.title("BiasN"); plt.show() ######### temp plot
            # nZ = self.Z + Bias_static  ######### temp plot
            # nZ = nZ - np.min(nZ)  ########## temp plot
            # plt.contourf(self.X, self.Y, nZ, levels=np.linspace(0,50,21), cmap="coolwarm"); plt.colorbar(); plt.title("inve_pot + BiasN"); plt.show()######### temp plot
            
            FES_cutoff = max(FES_cutoff, np.max(Bias_static))
        
        # creat static bias and external bias file usig the modified Bias_static    
        Bias_static, Force_static_x, Force_static_y, self.external_bias_file = lib2.make_external_bias_2D(self.X, self.Y, Bias=Bias_static, gaus_filter_sigma=gaus_filter_sigma, FES_cutoff=FES_cutoff, pl_min=self.pl_min, pl_max=self.pl_max, periodic=self.periodic, return_array=True, cv_name=self.cv_name)
        # add the modified Bias_static and Force_static to the static bias and force arrays
        self.Force_static_x += Force_static_x
        self.Force_static_y += Force_static_y
        self.Bias_static += Bias_static        
        # self.Bias_static += lib2.FFT_intg_2D(Force_static_x, Force_static_y, self.grid_min, self.grid_max, self.periodic)
        
        # plt.contourf(self.X, self.Y, Bias_static, levels=np.linspace(0,50,21), cmap="coolwarm"); plt.colorbar(); plt.title("BiasFinal"); plt.show() ######### temp plot
        # nZ = self.Z + Bias_static  ######### temp plot
        # nZ = nZ - np.min(nZ)  ########## temp plot
        # plt.contourf(self.X, self.Y, nZ, levels=np.linspace(0,50,21), cmap="coolwarm"); plt.colorbar(); plt.title("inve_pot + BiasFinal"); plt.show()######### temp plot
        
    def make_harmonic_bias(self, hp_centre_x=None, hp_centre_y=None, hp_kappa_x=None, hp_kappa_y=None, lw_centre_x=None, lw_centre_y=None, lw_kappa_x=None, lw_kappa_y=None, uw_centre_x=None, uw_centre_y=None, uw_kappa_x=None, uw_kappa_y=None):
            
        if hp_centre_x is None: hp_centre_x = self.hp_centre_x
        if hp_centre_y is None: hp_centre_y = self.hp_centre_y
        if hp_kappa_x is None: hp_kappa_x = self.hp_kappa_x
        if hp_kappa_y is None: hp_kappa_y = self.hp_kappa_y
        if lw_centre_x is None: lw_centre_x = self.lw_centre_x
        if lw_centre_y is None: lw_centre_y = self.lw_centre_y
        if lw_kappa_x is None: lw_kappa_x = self.lw_kappa_x
        if lw_kappa_y is None: lw_kappa_y = self.lw_kappa_y
        if uw_centre_x is None: uw_centre_x = self.uw_centre_x
        if uw_centre_y is None: uw_centre_y = self.uw_centre_y
        if uw_kappa_x is None: uw_kappa_x = self.uw_kappa_x
        if uw_kappa_y is None: uw_kappa_y = self.uw_kappa_y
        
        if hp_kappa_x is not None or hp_kappa_y is not None or lw_kappa_x is not None or lw_kappa_y is not None or uw_kappa_x is not None or uw_kappa_y is not None:
            
            Force_harmonic_x, Force_harmonic_y = np.zeros(self.nbins_yx), np.zeros(self.nbins_yx)
            
            if hp_kappa_x is not None or hp_kappa_y is not None: 
                f_harm_x, f_harm_y = lib2.find_hp_force(hp_centre_x, hp_centre_y, hp_kappa_x, hp_kappa_y, self.X, self.Y, self.grid_min, self.grid_max, [self.grid_dx, self.grid_dy], self.periodic)
                Force_harmonic_x += f_harm_x
                Force_harmonic_y += f_harm_y
            if lw_kappa_x is not None or lw_kappa_y is not None: 
                f_harm_x, f_harm_y = lib2.find_lw_force(lw_centre_x, lw_centre_y, lw_kappa_x, lw_kappa_y, self.X, self.Y, self.grid_min, self.grid_max, [self.grid_dx, self.grid_dy], self.periodic)
                Force_harmonic_x += f_harm_x
                Force_harmonic_y += f_harm_y
            if uw_kappa_x is not None or uw_kappa_y is not None:
                f_harm_x, f_harm_y = lib2.find_uw_force(uw_centre_x, uw_centre_y, uw_kappa_x, uw_kappa_y, self.X, self.Y, self.grid_min, self.grid_max, [self.grid_dx, self.grid_dy], self.periodic)
                Force_harmonic_x += f_harm_x
                Force_harmonic_y += f_harm_y
                
            self.Force_static_x += Force_harmonic_x
            self.Force_static_y += Force_harmonic_y
            self.Bias_static += lib2.FFT_intg_2D(Force_harmonic_x, Force_harmonic_y, self.grid_min, self.grid_max, self.periodic)
                
    def load_data(self, hills_file=None, position_file=None, n_pos_analysed=None):
        
        if position_file is None and self.position_file is None: self.position_file = lib2.get_file_path(self.position_file, "position")
        if position_file is None: position_file = self.position_file
        
        if self.metad_height is None or self.metad_height <= 0: 
            self.metad_height = None
            hills_file = None
        else: 
            if hills_file is None and self.hills_file is None: self.hills_file = lib2.get_file_path(self.hills_file, "hills")
            if hills_file is None: hills_file = self.hills_file
        
        if n_pos_analysed is None:
            if hasattr(self, "n_pos_analysed"): n_pos_analysed = self.n_pos_analysed[0]
            else: n_pos_analysed = 0
            
        self.hills, self.position = lib2.read_data(hills_path=hills_file, pos_path=position_file, n_pos_analysed=n_pos_analysed, n_pos_per_window=self.n_pos_per_window, metad_h=self.metad_height)
                
        if (self.hills is None and self.metad_height is not None) or self.position is None: 
            print("\n ***** The data files are not found or empty ***** \n")
            return
            
        if self.metad_height is not None:
            self.Gamma_Factor = (self.hills[1, 6] - 1) / (self.hills[1, 6]) if self.WellTempered else 1
            self.sigma_meta2_x = self.hills[1, 3] ** 2 
            self.sigma_meta2_y = self.hills[1, 4] ** 2 
            assert self.sigma_meta2_x == self.hills[-1, 3] ** 2, "the first sigma_meta2_x is not equal to the last sigma_meta2_x"
            assert self.sigma_meta2_y == self.hills[-1, 4] ** 2, "the first sigma_meta2_y is not equal to the last sigma_meta2_y"
            assert len(self.hills) == len(self.position) / self.n_pos_per_window, f"The number of hills and the number of position points are not consistent: {len(self.hills) = }, {len(self.position) = }, {self.n_pos_per_window = }"

    def load_position(self, position_file=None, n_pos_analysed=None):
        
        if position_file is None and self.position_file is None: self.position_file = lib2.get_file_path(self.position_file, "position")
        if position_file is None: position_file = self.position_file
        
        if n_pos_analysed is None:
            if hasattr(self, "n_pos_analysed"): n_pos_analysed = self.n_pos_analysed[0]
            else: n_pos_analysed = 0
                
        self.position = lib2.read_position(position_file, n_pos_analysed)

    def check_if_results_exist(self, save_path, return_results=False):
        if os.path.exists(save_path):
            if self.print_info: print(f"\n ***** The results file already exists: {save_path} ***** \n")
            loaded_data = lib2.load_pkl(save_path)
            if return_results: return loaded_data
            else: return True
        else: return False
            
    def initialise_time_dependent_variables(self):

        # n_pos is the total number of positions that are analysed. If n_pos not specified, all positions are analysed. Otherwise, only n_pos positions are analysed and the rest ignored.
        if self.n_pos is None or self.n_pos == -1: 
            if self.hills is not None: self.n_pos = len(self.hills) * self.n_pos_per_window  # if metadynamics active: n_pos is the total number of hills * n_pos_per_window
            else: self.n_pos = len(self.position) # if metadynamics not active: n_pos is the total number of positions.          
        
        # if metadynamics is inactive and n_pos_per_window == -1 or None: position data is analysed in one window. If metadynamics is active, n_pos_per_window was set in init or post_init              
        if (self.n_pos_per_window is None or self.n_pos_per_window <= 0) and self.hills is None: self.n_pos_per_window = self.n_pos         
        
        # n_windows: Forces are calculated in (n) windows of constant bias. Use np.ceil to include incomplete windows (windows with less than n_pos_per_window positions). use min() to avoid n_windows > len(hills).
        if self.hills is not None: self.n_windows = min(int(np.ceil(self.n_pos / self.n_pos_per_window)), len(self.hills))
        else: self.n_windows = int(np.ceil(self.n_pos / self.n_pos_per_window))
                
        # find the indexes where the error is calculated. If len_error_list is longer than n_windows, the error is calculated in every window. If len_error_list is None or less than 1, the error is calculated only at the end.
        if self.i_error_list is None: 
            if self.len_error_list is None or self.len_error_list <= 0: self.len_error_list = 1
            if self.len_error_list > self.n_windows: self.len_error_list = self.n_windows
            self.i_error_list = np.array([int(self.n_windows - i*(self.n_windows / self.len_error_list)) for i in range(self.len_error_list)])[::-1] # list of iterations when the error is calculated
        else: # if i_error_list is provided, check if the values are within the range of n_windows
            if any([i < 0 or i >= self.n_windows for i in self.i_error_list]): print(f"some values in i_error_list are either below 0 or above n_windows: {min(self.i_error_list) = }, {max(self.i_error_list) = }, {self.n_windows = }")
            if all([i < 0 or i >= self.n_windows for i in self.i_error_list]): raise ValueError(f"all values in i_error_list are either below 0 or above n_windows: {min(self.i_error_list) = }, {max(self.i_error_list) = }, {self.n_windows = }")
                       
    def find_grid(self):
        
        if None in (self.grid_min[0], self.grid_min[1], self.grid_max[0], self.grid_max[1], self.grid_n[0], self.grid_n[1]):
            print("\n ***** Boundaries of the grid not provided ***** \nPlease provide the grid information ...\n")

            self.grid_min = input("(1/5) Enter the lower boundary for the x (CV1) and y (CV2) dimension with format [min_x, min_y] : ")
            self.grid_max = input("(2/5) Enter the upper boundary for the x (CV1) and y (CV2) dimension with format [max_x, max_y] : ")
            self.grid_n = input("(3/5) Enter the number of grid points for the x (CV1) and y (CV2) dimension with format [n_x, n_y] : ")
            periodic_x = True if input("(4/5) Is the grid periodic in the x-dimension (CV1)? Press 'y' and then 'Enter' if yes, otherwise Enter anything else: ").lower() == "y" else False
            periodic_y = True if input("(5/5) Is the grid periodic in the y-dimension (CV2)? Press 'y' and then 'Enter' if yes, otherwise Enter anything else: ").lower() == "y" else False
            self.periodic = [periodic_x, periodic_y]
            
            self.grid_min = self.grid_min.replace("[", "").replace("]", "").split(",")
            self.grid_max = self.grid_max.replace("[", "").replace("]", "").split(",")
            self.grid_n = self.grid_n.replace("[", "").replace("]", "").split(",")
            
            self.grid_min[0], self.grid_min[1] = float(self.grid_min[0]), float(self.grid_min[1])
            self.grid_max[0], self.grid_max[1] = float(self.grid_max[0]), float(self.grid_max[1])
            self.grid_n[0], self.grid_n[1] = int(self.grid_n[0]), int(self.grid_n[1])
                        
        # If grid is not provided but grid_min, grid_max, and grid_n are, calculate the grid
        if (self.grid_x is None or self.grid_y is None) and (None not in (self.grid_min[0], self.grid_min[1], self.grid_max[0], self.grid_max[1], self.grid_n[0], self.grid_n[1])):
            self.grid_x = np.linspace(self.grid_min[0], self.grid_max[0], self.grid_n[0])
            self.grid_y = np.linspace(self.grid_min[1], self.grid_max[1], self.grid_n[1])
            self.X, self.Y = np.meshgrid(self.grid_x, self.grid_y)
        else: raise ValueError("The grid cannot be calculated. Please provide the grid or the grid_min, grid_max, and grid_n when initialising the class.")
 
    def get_data_i(self):
        #Get position data of window
        if self.hills is not None: 
            pos_meta_x, pos_meta_y, height_meta = self.hills[self.i, 1], self.hills[self.i, 2], self.hills[self.i, 5] * self.Gamma_Factor # centre position (x and y) of Gaussian, and height of Gaussian
            pos_meta = lib2.find_periodic_points([pos_meta_x], [pos_meta_y], self.grid_min, self.grid_max, self.periodic, self.grid_length, self.periodic_range) if (self.periodic[0] or self.periodic[1])  else [[xi, yi] for xi, yi in zip([pos_meta_x], [pos_meta_y])]
        else: pos_meta, height_meta = None, None
        
        pos_x, pos_y = self.position[self.i * self.n_pos_per_window: (self.i+1) * self.n_pos_per_window, 1], self.position[self.i * self.n_pos_per_window: (self.i+1) * self.n_pos_per_window, 2]# positons of window of constant bias force.
        pos = lib2.find_periodic_points(pos_x, pos_y, self.grid_min, self.grid_max, self.periodic, self.grid_length, self.periodic_range) if (self.periodic[0] or self.periodic[1]) else [[xi, yi] for xi, yi in zip(pos_x, pos_y)]
        time_i = self.position[(self.i+1) * self.n_pos_per_window - 1, 0] 
                        
        return time_i, pos, pos_meta, height_meta
            
    def calculate_errors(self, force_terms_tot=None):
        
        #If applicable, find FES and cutoff and explored space. Explored space is the space not cut off by the cutoff.
        self.cutoff = np.ones(self.nbins_yx)
        
        if force_terms_tot is not None: PD_tot, PD2_tot, Force_x_tot, Force_y_tot, ofv_num_x_tot, ofv_num_y_tot = force_terms_tot
        else:
            if self.base_forces is not None: [PD_tot, PD2_tot, Force_x_tot, Force_y_tot, ofv_num_x_tot, ofv_num_y_tot] = lib2.patch_forces(np.array([self.PD, self.PD2, self.Force_x, self.Force_y, self.ofv_num_x, self.ofv_num_y]), self.base_forces)
            else: PD_tot, PD2_tot, Force_x_tot, Force_y_tot, ofv_num_x_tot, ofv_num_y_tot = self.PD, self.PD2, self.Force_x, self.Force_y, self.ofv_num_x, self.ofv_num_y
        
        if self.FES_cutoff is not None or self.save_maps is True: self.FES = lib2.FFT_intg_2D(Force_x_tot, Force_y_tot, self.grid_min, self.grid_max, self.periodic)        
        if self.FES_cutoff is not None: self.cutoff = np.where(self.FES < self.FES_cutoff, self.cutoff, 0)
        if self.PD_cutoff is not None: self.cutoff = np.where(PD_tot > self.PD_cutoff, self.cutoff, 0)
        self.space_explored = np.sum(self.cutoff)
        self.ratio_explored = self.space_explored / np.prod(self.nbins_yx)
               
        # calculate error
        ofv_x = np.divide(ofv_num_x_tot, PD_tot, out=np.zeros_like(ofv_num_x_tot), where=PD_tot > self.PD_limit) - np.square(Force_x_tot)
        ofv_y = np.divide(ofv_num_y_tot, PD_tot, out=np.zeros_like(ofv_num_y_tot), where=PD_tot > self.PD_limit) - np.square(Force_y_tot)
        PD_sq = np.square(PD_tot)
        PD_diff = PD_sq - PD2_tot
        Bessel_corr_num = PD2_tot.copy() if self.use_ST_ERR == True else PD_sq
        Bessel_corr = np.divide(Bessel_corr_num, PD_diff, out=np.zeros_like(Bessel_corr_num), where=(PD_diff > 0)) * self.cutoff
        ofv_x *= Bessel_corr * self.cutoff
        ofv_y *= Bessel_corr * self.cutoff
        if self.non_exploration_penalty is not None: ofv_x, ofv_y = np.where(self.cutoff > 0.5, ofv_x, self.non_exploration_penalty**2), np.where(self.cutoff > 0.5, ofv_y, self.non_exploration_penalty**2) 
          
        self.ofe = np.sqrt(np.absolute(ofv_x) + np.absolute(ofv_y))
        self.Aofe = np.sum(self.ofe) / self.space_explored if self.space_explored > 0 else np.nan
        if self.use_VNORM and self.space_explored > 0: self.Aofe /= self.ratio_explored

        # if specified, calculate Average Absolute Deviation (AAD) of the FES
        if self.Z is not None:
            self.AD = abs(self.FES - self.Z)*self.cutoff
            self.AAD = np.sum(self.AD) / self.space_explored if self.space_explored > 0 else np.nan
            if self.use_VNORM and self.space_explored > 0: self.AAD /= self.ratio_explored
                              
        # if specified, calculate Average Absolute Deviation (AAD) of the force
        if (self.dZ_dX is not None) and (self.dZ_dY is not None): 
            self.AD_Force_x = abs(Force_x_tot - self.dZ_dX)*self.cutoff
            self.AD_Force_y = abs(Force_y_tot - self.dZ_dY)*self.cutoff
            self.AD_Force = np.sqrt(np.square(self.AD_Force_x) + np.square(self.AD_Force_y))
            self.AAD_Force = np.sum(self.AD_Force) / self.space_explored if self.space_explored > 0 else np.nan
            if self.use_VNORM and self.space_explored > 0: self.AAD_Force /= self.ratio_explored
        
        
        if self.record_forces_e:
            if all(hasattr(self, attr) for attr in ["PD_e", "PD2_e", "Force_x_e", "Force_y_e", "ofv_num_x_e", "ofv_num_y_e"]):
                if len(self.forces_e_list) == 0: self.forces_e_list = np.array([[self.PD_e, self.PD2_e, self.Force_x_e, self.Force_y_e, self.ofv_num_x_e, self.ofv_num_y_e]])
                else: self.forces_e_list = np.concatenate((self.forces_e_list, [[self.PD_e, self.PD2_e, self.Force_x_e, self.Force_y_e, self.ofv_num_x_e, self.ofv_num_y_e]]), axis=0)
            else:
                if len(self.forces_e_list) == 0: self.forces_e_list = np.array([[self.PD, self.PD2, self.Force_x, self.Force_y, self.ofv_num_x, self.ofv_num_y]])
                else: 
                    PD_e = self.PD - self.old_forces[0]
                    PD2_e = self.PD2 - self.old_forces[1]
                    Force_x_e = np.divide( self.Force_num_x - self.old_forces[2], PD_e, out=np.zeros_like(PD_e), where=PD_e > self.PD_limit)
                    Force_y_e = np.divide( self.Force_num_y - self.old_forces[3], PD_e, out=np.zeros_like(PD_e), where=PD_e > self.PD_limit)
                    ofv_num_x_e = self.ofv_num_x - self.old_forces[4]
                    ofv_num_y_e = self.ofv_num_y - self.old_forces[5]
                    self.forces_e_list = np.concatenate((self.forces_e_list, [[PD_e, PD2_e, Force_x_e, Force_y_e, ofv_num_x_e, ofv_num_y_e]]), axis=0)             
                self.old_forces = np.array([self.PD, self.PD2, self.Force_num_x, self.Force_num_y, self.ofv_num_x, self.ofv_num_y])
           
            if self.bootstrap_iter is not None and self.bootstrap_iter > 0:
                if self.forces_e_list.shape[0] > 2:
                    _, _, self.BS_error, _ = lib2.bootstrapping_error(X=self.X, Y=self.Y, force_array=self.forces_e_list, n_bootstrap=self.bootstrap_iter, periodic=self.periodic, FES_cutoff=self.FES_cutoff, PD_cutoff=self.PD_cutoff, PD_limit=self.PD_limit, use_VNORM=self.use_VNORM, get_progression=False, print_progress=False)
                    self.ABS_error = np.sum(self.BS_error) / self.space_explored if self.space_explored > 0 else np.nan
                    if self.use_VNORM and self.space_explored > 0: self.ABS_error /= self.ratio_explored
                else: self.BS_error, self.ABS_error = np.zeros(self.nbins_yx), np.nan

        if self.calculate_FES_st_dev:
            if self.Maps_list.shape[0] > 2:
                FES_maps_list = np.array(self.Maps_list)[-30:,0]
                FES_maps_list = np.concatenate((FES_maps_list, [self.FES]), axis=0)
                FES_maps_list = FES_maps_list * self.cutoff[np.newaxis, :, :]
                FES_st_dev = np.var(FES_maps_list, axis=0)
                self.FES_st_dev = np.sqrt(FES_st_dev) * (len(FES_maps_list) / (len(FES_maps_list) - 1))
                self.Avr_FES_st_dev = np.sum(self.FES_st_dev) / self.space_explored if self.space_explored > 0 else np.nan
                if self.use_VNORM and self.space_explored > 0: self.Avr_FES_st_dev /= self.ratio_explored
            else: self.FES_st_dev, self.Avr_FES_st_dev = np.zeros(self.nbins_yx), np.nan
            
        #save global error evolution [time, space_ratio, Aofe, AAD, AAD_Force, ABS_error]
        Avr_Error_e = [self.sim_time+self.base_time, self.ratio_explored, self.Aofe]
        if self.Z is not None: Avr_Error_e.append(self.AAD)
        if self.dZ_dX is not None and self.dZ_dY is not None: Avr_Error_e.append(self.AAD_Force)
        if self.bootstrap_iter is not None: Avr_Error_e.append(self.ABS_error)
        if self.calculate_FES_st_dev: Avr_Error_e.append(self.Avr_FES_st_dev)
        self.Avr_Error_list = np.concatenate((self.Avr_Error_list, [Avr_Error_e]), axis=0)

        # save Maps (i.e. quatities defined on grid) [ cutoff, ofe, BS_error, AD_F, AD_FES, FES]        
        if self.record_maps: 
            Maps_e = [self.FES, self.cutoff, self.ofe] 
            if self.Z is not None: Maps_e.append(self.AD)
            if (self.dZ_dX is not None) and (self.dZ_dY is not None): Maps_e.append(self.AD_Force)
            if self.bootstrap_iter is not None: Maps_e.append(self.BS_error)    
            if self.calculate_FES_st_dev: Maps_e.append(self.FES_st_dev)        
            self.Maps_list = np.concatenate((self.Maps_list, [Maps_e]), axis=0)       
 
    def calculate_difference_in_error(self):
        
        if self.Avr_Error_list.shape[0] < 2: return
        
        # change in time
        dt = self.Avr_Error_list[-1][0] - self.Avr_Error_list[-2][0]
        dt_tot = self.Avr_Error_list[-1][0] - self.base_time        
        
        #change in Aofe
        d_Aofe = self.Avr_Error_list[-1][2] - self.Avr_Error_list[-2][2]
        d_Aofe_dt = d_Aofe / dt if dt > 0 else 0
        dtot_Aofe = self.Avr_Error_list[-1][2] - self.Aofe_0
        dtot_Aofe_dt = dtot_Aofe / dt_tot if dt_tot > 0 else 0
        self.d_Aofe = np.concatenate((self.d_Aofe, [[d_Aofe, d_Aofe_dt, dtot_Aofe, dtot_Aofe_dt]]))
        
        if self.Z is not None: # change in AAD
            d_AAD = self.Avr_Error_list[-1][self.aad_index] - self.Avr_Error_list[-2][self.aad_index]
            d_AAD_dt = d_AAD / dt if dt > 0 else 0
            dtot_AAD = self.Avr_Error_list[-1][self.aad_index] - self.AAD_0
            dtot_AAD_dt = dtot_AAD / dt_tot if dt_tot > 0 else 0
            self.d_AAD = np.concatenate((self.d_AAD, [[d_AAD, d_AAD_dt, dtot_AAD, dtot_AAD_dt]]))
        
        # if find weighted average, calculate the weighted average of the d_error lists, where the last element has a weight of 1/(1.5**0), the second last has a weight of 1/(1.5**1), ... 
        if self.weighted_avr_error_change:
            
            # find weighted average of Aofe
            d_Aofe_w_new = [lib2.weighted_average(self.d_Aofe[:,err_type], factor=1.5, only_last=True) for err_type in range(4)]
            self.d_Aofe_w = np.concatenate((self.d_Aofe_w, [d_Aofe_w_new]))
            
            # if y is not None, find weighted average of AAD
            if self.Z is not None:
                d_AAD_w_new = [lib2.weighted_average(self.d_AAD[:,err_type], factor=1.5, only_last=True) for err_type in range(4)]
                self.d_AAD_w = np.concatenate((self.d_AAD_w, [d_AAD_w_new]))
                
            # set d_error the the weighted average of the goal type
            if self.main_error_type == "ST_ERR": self.d_error = self.d_Aofe_w
            elif self.main_error_type == "AAD": self.d_error = self.d_AAD_w
            else: raise ValueError("main_error_type not recognised")
        
        else:
            if self.main_error_type == "ST_ERR": self.d_error = self.d_Aofe
            elif self.main_error_type == "AAD": self.d_error = self.d_AAD
            else: raise ValueError("main_error_type not recognised")

    def plot_results(self, save_path="", show=True, more_aofe=None, more_aad=None, more_vol=None, t_compare=None, v_compare=None, aofe_compare=None, aad_compare=None, aofe_lim=None, aad_lim=None, error_map_gaussian_filter=None, min_PD=None, max_PD=None, min_FES=None, max_FES=None, min_ofe=None, max_ofe=None, min_AD=None, max_AD=None):
        
        if error_map_gaussian_filter is None: 
            ofe_plot = self.ofe*self.cutoff
            if self.Z is not None: ad_plot = self.AD*self.cutoff
        else:
            ofe_plot = gaussian_filter(self.ofe, error_map_gaussian_filter)*self.cutoff if any(self.periodic) is False else gaussian_filter(self.ofe, error_map_gaussian_filter, mode="wrap")*self.cutoff
            if self.Z is not None: ad_plot = gaussian_filter(self.AD, error_map_gaussian_filter)*self.cutoff if any(self.periodic) is False else gaussian_filter(self.AD, error_map_gaussian_filter, mode="wrap")*self.cutoff
        
        min_PD = 10_000 if min_PD is None else min_PD
        if max_ofe is None: max_ofe = np.percentile(np.array(ofe_plot)[np.nonzero(ofe_plot)], 99.0)
        if min_ofe is None: min_ofe = np.percentile(np.array(ofe_plot)[np.nonzero(ofe_plot)], 1.0)
        if max_AD is None and self.Z is not None: max_AD = np.percentile(np.array(ad_plot)[np.nonzero(ad_plot)], 99.9)
        if min_AD is None and self.Z is not None: min_AD = np.percentile(np.array(ad_plot)[np.nonzero(ad_plot)], 0.1)
        
        level_PD = lib2.find_contour_levels(self.PD, min_level=min_PD, max_level=max_PD, error_name="PD")
        level_fes = lib2.find_contour_levels(self.FES*self.cutoff, min_level=min_FES, max_level=max_FES, error_name="FES")
        level_ofe = lib2.find_contour_levels(self.ofe*self.cutoff, min_level=min_ofe, max_level=max_ofe, error_name="ofe")
        if self.Z is not None: level_aad = lib2.find_contour_levels(self.AD*self.cutoff, min_level=min_AD, max_level=max_AD, error_name="AD_FES")
 
        plt.figure(figsize=(15, 8))
        
        plt.subplot(2,3, 1)
        plt.contourf(self.X,self.Y, np.where(self.PD < level_PD[0], np.nan, self.PD), levels=level_PD, cmap='coolwarm', norm=colors.LogNorm(vmin=min(level_PD), vmax=max(level_PD)))
        plt.rcParams['ytick.minor.visible'] = False; plt.colorbar(label="PD [-]"); plt.rcParams['ytick.minor.visible'] = True; 
        plt.title("Probability Density", fontsize=20); plt.xlabel("x"); plt.ylabel("y")

        plt.subplot(2,3, 2)
        plt.contourf(self.X,self.Y, lib2.zero_to_nan(ofe_plot), levels=level_ofe, cmap='coolwarm')
        title_text = "On-The-Fly Error of the Mean Force" if error_map_gaussian_filter is None else f"On-The-Fly Error of the Mean Force\n(Gaussian Filter: {error_map_gaussian_filter})"
        plt.colorbar(label="OFE [kJ mol$^{-1}$]"); plt.title(title_text, fontsize=20); plt.xlabel("x"); plt.ylabel("y")

        
        ax4 = plt.subplot(2,3,3)
        ax4.plot(self.Avr_Error_list[:, 0], self.Avr_Error_list[:, 2], color="black"); plt.ylabel("Avr. ST ERR [kJ/mol]")
        for j in range(1,len(self.n_pos_analysed)): ax4.axvline(np.sum(self.n_pos_analysed[1:int(j+1)]) * self.time_step * self.position_pace / 1000, color="yellow", linestyle="--", alpha=0.7) 

        plt.title("Error Evolution", fontsize=20); plt.xlabel("Time [ns]")
        
        if aofe_compare is not None: 
            t_index = np.argmin(np.abs(t_compare - self.Avr_Error_list[-1, 0]))
            ax4.plot(t_compare[:t_index], aofe_compare[:t_index], color="black", linestyle="--", alpha=0.5, label="Compare Sim")
        
        if more_aofe is not None: 
            more_aofe = np.array(more_aofe) 
            if len(np.shape(more_aofe)) == 2: ax4.plot(more_aofe[0], more_aofe[1], color="black", alpha=0.5, label="Base Sim")
            elif len(np.shape(more_aofe)) == 3: [ax4.plot(more_aofe[i][0], more_aofe[i][1], color="black", alpha=0.3 + 0.4*(i/len(more_aofe)), label=f"Base Sim {i}") for i in range(len(more_aofe)) ]
            else: raise ValueError("more_aofe has wrong shape")
            if aofe_lim is not None: 
                if isinstance(aofe_lim, (int, float)): ax4.set_ylim(0,aofe_lim)
                elif isinstance(aofe_lim, list) and len(aofe_lim) == 2: ax4.set_ylim(aofe_lim)
                else: raise ValueError("aofe_lim is supposed to be a number (upper limit) of a list [lower limit, upper limit]")
                ax4.legend(loc="upper left", framealpha=0.3, fontsize=10 )
        ax4.set_yscale('log')

        if self.Z is not None:
            ax4_2 = ax4.twinx()
            ax4_2.plot(self.Avr_Error_list[:, 0], self.Avr_Error_list[:, self.aad_index], color="red"); plt.ylabel("Avr.Abs.Dev. FES [kJ/mol]", color="red")
            ax4_2.tick_params(axis='y', labelcolor="red"); ax4_2.spines['right'].set_color('red')  
            
            if aad_compare is not None:
                t_index = np.argmin(np.abs(t_compare - self.Avr_Error_list[-1, 0]))
                ax4_2.plot(t_compare[:t_index], aad_compare[:t_index], color="red", linestyle="--", alpha=0.5, label="Compare Sim")
            
            if more_aad is not None:
                more_aad = np.array(more_aad)
                if len(np.shape(more_aad)) == 2: ax4_2.plot(more_aad[0], more_aad[1], color="red", alpha=0.5, label="Base Sim")
                elif len(np.shape(more_aad)) == 3: [ax4_2.plot(more_aad[i][0], more_aad[i][1], color="red", alpha=0.3 + 0.4*(i/len(more_aad)), label=f"Base Sim {i}") for i in range(len(more_aad)) ]
                else: raise ValueError("more_aad has wrong shape")
                if aad_lim is not None:
                    if isinstance(aad_lim, (int, float)): ax4_2.set_ylim(0,aad_lim)
                    elif isinstance(aad_lim, list) and len(aad_lim) == 2: ax4_2.set_ylim(aad_lim)
                    else: raise ValueError("aad_lim is supposed to be a number (upper limit) of a list [lower limit, upper limit]")
                    ax4_2.legend(loc="upper right", framealpha=0.3, fontsize=10 )
            ax4_2.set_yscale('log')        
        
        plt.subplot(2,3,4)
        plt.contourf(self.X,self.Y, lib2.zero_to_nan(self.FES*self.cutoff), levels=level_fes, cmap='coolwarm')
        plt.colorbar(label="FES [kJ mol$^{-1}$]"); plt.title("Estimated FES", fontsize=20); plt.xlabel("x"); plt.ylabel("y")
        
        if self.Z is not None:         
            plt.subplot(2,3,5)
            plt.contourf(self.X,self.Y, lib2.zero_to_nan(ad_plot), levels=level_aad, cmap='coolwarm')
            title_text = "Absolute Deviation of FES" if error_map_gaussian_filter is None else f"Absolute Deviation of FES\n(Gaussian Filter: {error_map_gaussian_filter})"
            plt.colorbar(label="AD [kJ mol$^{-1}$]"); plt.title(title_text, fontsize=20); plt.xlabel("x"); plt.ylabel("y")
        
        plt.subplot(2,3,6)
        plt.plot(self.Avr_Error_list[:, 0], self.Avr_Error_list[:, 1], color="black"); plt.ylabel("Explored Volume Ratio [-]")        
        for j in range(1,len(self.n_pos_analysed)): plt.axvline(np.sum(self.n_pos_analysed[1:int(j+1)]) * self.time_step * self.position_pace / 1000, color="yellow", linestyle="--", alpha=0.7) 
        plt.title("Progression of Explored Volume", fontsize=20); plt.xlabel("Time [ns]")
        
        if v_compare is not None:
            t_index = np.argmin(np.abs(t_compare - self.Avr_Error_list[-1, 0]))
            plt.plot(t_compare[:t_index], v_compare[:t_index], color="grey", linestyle="--", alpha=0.3, label="Compare Sim")
        
        if more_vol is not None: 
            more_vol = np.array(more_vol) 
            if len(np.shape(more_vol)) == 2: plt.plot(more_vol[0], more_vol[1], color="black", alpha=0.5, label="Base Sim")
            elif len(np.shape(more_vol)) == 3: [plt.plot(more_vol[i][0], more_vol[i][1], color="black", alpha=0.3 + 0.4*(i/len(more_vol)), label=f"Base Sim {i}") for i in range(len(more_vol)) ]
            else: raise ValueError("more_vol has wrong shape")
                
        plt.tight_layout()
        if save_path != "": plt.savefig(save_path)
        if show: plt.show()

    def plot_errors(self, error_type=["Aofe", "AAD_Force", "AAD", "ABS_error"], save_path="", error_force_log=True, error_fes_log=True, show=True, more_aofe=None, more_aad=None, t_compare=None, aofe_compare=None, aad_compare=None, aofe_lim=None, aad_lim=None):
        
        plt.figure(figsize=(10, 4))
        ax = plt.subplot(1, 1, 1)
        err_arr = np.array(self.Avr_Error_list)

        # plot Average on-the-fly error (Aofe)
        ax.plot(err_arr[:, 0], err_arr[:, 2], color="black", label="Aofe"); 
        
        # if available, plot average absolute deviation of the Force (AAD_Force)
        if "AAD_Force" in error_type and "AAD_Force" in self.Avr_Error_info:
            ax.plot(err_arr[:, 0], err_arr[:, self.aad_force_index], color="grey", alpha=0.7, label="AAD Force")
        
        # if available, plot Aofe for comparison
        if aofe_compare is not None and t_compare is not None: 
            t_index = np.argmin(np.abs(t_compare - err_arr[-1, 0]))
            ax.plot(t_compare[:t_index], aofe_compare[:t_index], color="black", linestyle="--", alpha=0.5, label="Compare Sim")
        ax.set_ylabel("Error of Force [kJ/mol]"); ax.set_xlabel("Time [ns]"); ax.set_title("Error Evolution")
        
        # if available, plot additional Aofe (more_aofe) (e.g. from other simulations)
        if more_aofe is not None: 
            more_aofe = np.array(more_aofe) 
            if len(np.shape(more_aofe)) == 2: ax.plot(more_aofe[0], more_aofe[1], color="black", alpha=0.5, label="Base Sim")
            elif len(np.shape(more_aofe)) == 3: [ax.plot(more_aofe[i][0], more_aofe[i][1], color="black", alpha=0.3 + 0.4*(i/len(more_aofe)), label=f"Base Sim {i}") for i in range(len(more_aofe)) ]
            else: raise ValueError("more_aofe has wrong shape")
            if aofe_lim is not None: 
                if isinstance(aofe_lim, (int, float)): ax.set_ylim(0,aofe_lim)
                elif isinstance(aofe_lim, list) and len(aofe_lim) == 2: ax.set_ylim(aofe_lim)
                else: raise ValueError("aofe_lim is supposed to be a number (upper limit) of a list [lower limit, upper limit]")
                
        ax.legend(loc="upper center", framealpha=0.3, fontsize=10 )
        if error_force_log: ax.set_yscale('log')

        # if available, plot error of the FES (AAD / ABS_error / FES_st_dev)
        if ("AAD" in error_type and "AAD" in self.Avr_Error_info) or ("ABS_error" in error_type and "ABS_error" in self.Avr_Error_info) or ("FES_st_dev" in error_type and "FES_st_dev" in self.Avr_Error_info):
            ax_2 = ax.twinx()
 
        
            # if available, plot Average Absolute Deviation (AAD) of the FES
            if "AAD" in error_type and "AAD" in self.Avr_Error_info: 
                
                # plot AAD
                ax_2.plot(err_arr[:, 0], err_arr[:, self.aad_index], color="orange", alpha=0.7, label="AAD"); 
                
                # if available, plot AAD for comparison
                if aad_compare is not None and t_compare is not None:
                    t_index = np.argmin(np.abs(t_compare - err_arr[-1, 0]))
                    ax_2.plot(t_compare[:t_index], aad_compare[:t_index], color="orange", linestyle="--", alpha=0.5, label="Compare Sim")
                
                # if available, plot additional AAD (more_aad) (e.g. from other simulations)
                if more_aad is not None:
                    more_aad = np.array(more_aad)
                    if len(np.shape(more_aad)) == 2: ax_2.plot(more_aad[0], more_aad[1], color="orange", alpha=0.5, label="Base Sim")
                    elif len(np.shape(more_aad)) == 3: [ax_2.plot(more_aad[i][0], more_aad[i][1], color="red", alpha=0.3 + 0.4*(i/len(more_aad)), label=f"Base Sim {i}") for i in range(len(more_aad)) ]
                    else: raise ValueError("more_aad has wrong shape")
                    if aad_lim is not None:
                        if isinstance(aad_lim, (int, float)): ax_2.set_ylim(0,aad_lim)
                        elif isinstance(aad_lim, list) and len(aad_lim) == 2: ax_2.set_ylim(aad_lim)
                        else: raise ValueError("aad_lim is supposed to be a number (upper limit) of a list [lower limit, upper limit]")
            
            # if available, plot Average Bootstrap Error (ABS_error)            
            if "ABS_error" in error_type and "ABS_error" in self.Avr_Error_info:
                ax_2.plot(err_arr[:, 0], err_arr[:, self.abs_error_index], color="red", label="ABS Error")
            
            # if available, plot Average FES Standard Deviation (FES_st_dev)    
            if "FES_st_dev" in error_type and "FES_st_dev" in self.Avr_Error_info:
                ax_2.plot(err_arr[:, 0], err_arr[:, self.FES_st_dev_index], color="purple", label="FES st. dev.")
           
            ax_2.set_ylabel("Error of FES [kJ/mol]", color="red"); ax_2.tick_params(axis='y', labelcolor="red"); ax_2.spines['right'].set_color('red')  
            if error_fes_log: ax_2.set_yscale('log')
            ax_2.legend(loc="upper right", framealpha=0.3, fontsize=10 )
            
        plt.tight_layout()
        if save_path != "": plt.savefig(save_path)
        if show: plt.show()

    def save_data(self, save_data_path="", use_parent_instance=None):
        
        if save_data_path != "" and save_data_path[-1] != "/": save_data_path += "/"
        
        if self.save_force_terms: lib2.save_pkl(self.force_terms, f"{save_data_path}force_terms{self.ID}.pkl")
        
        if self.save_error_progression:
            
            if self.Z is not None: err_prog, txt = self.Avr_Error_list[:, [0, 2, self.aad_index]], "" 
            elif self.bootstrap_iter is not None: err_prog, txt = self.Avr_Error_list[:, [0, 2, self.abs_error_index]], "_ABS"
            elif self.calculate_FES_st_dev: err_prog, txt = self.Avr_Error_list[:, [0, 2, self.FES_st_dev_index]] , "_FESstdev"
            else: err_prog, txt = self.Avr_Error_list[:, [0, 2]], "Aofe"
            lib2.save_pkl(err_prog.T, f"{save_data_path}error_progression{self.ID}{txt}.pkl")
            
        if self.save_results: 
            info_text = ["INFO_TEXT", "force_terms:[PD, PD2, Force_x, Force_y, ofv_num_x, ofv_num_y]", f"Avr_Error_list: {self.Avr_Error_info}"]
            save_list = [np.array(self.force_terms), np.array(self.Avr_Error_list)]
            
            if self.calculate_error_change:
                new_txt = "Error_change: [d_Aofe"
                new_list = [self.d_Aofe]
                if self.Z is not None: 
                    new_txt += ", d_AAD"
                    new_list += [self.d_AAD]
                if self.weighted_avr_error_change:
                    new_txt += ", d_Aofe_w"
                    new_list += [self.d_Aofe_w]
                    if self.Z is not None: 
                        new_txt += ", d_AAD_w"
                        new_list += [self.d_AAD_w]
                info_text += [f"{new_txt}]"]
                save_list += [np.array(new_list)]
            
            
            if self.record_maps: 
                new_txt = "Maps: [FES, cutoff, ofe"
                if self.Z is not None: new_txt += ", AD"
                if self.bootstrap_iter is not None: new_txt += ", BS_error"
                if self.calculate_FES_st_dev: new_txt += ", FES_st_dev"
                new_txt += "]"
                info_text += [new_txt]
                save_list += [self.Maps_list]
            
            
            if self.record_forces_e: 
                info_text += ["forces_e_list: [PD_e, PD2_e, Force_x_e, Force_y_e, ofv_num_x_e, ofv_num_y_e]"]
                save_list += [np.array(self.forces_e_list)]
                
            save_list = [info_text] + [save_list]
            lib2.save_pkl(save_list, f"{save_data_path}MFI_results{self.ID}.pkl")
            
        if self.save_mfi_instance: 
            if use_parent_instance is None: self.save_instance(save_data_path + f"MFI_instance{self.ID}.pkl")
            else: self.save_instance(save_data_path + f"MFI_instance{self.ID}.pkl", use_parent_instance)

    def analyse_data(self, print_analysis=True):
        
        if self.simulation_folder_path is not None: lib2.set_up_folder(self.simulation_folder_path) 
        
        # get data
        self.load_data()
        
        # initialise time dependent variables (n_windows, i_error_list, ...)
        self.initialise_time_dependent_variables()
        
        #iterate over hills
        if self.print_info: start = time.time()
        for self.i in range(self.n_windows):
            
            # get data_i for a window of constant bias potential
            time_i, pos, pos_meta, height_meta = self.get_data_i()
            
            ###--- Updated Bias and Bias_force from Metadyanmics of new window      
            if pos_meta is not None:
                for pos_m in pos_meta:
                    kernelmeta_x = np.exp( - np.square(self.grid_x - pos_m[0]) / (2 * self.sigma_meta2_x)) * height_meta
                    kernelmeta_y = np.exp( - np.square(self.grid_y - pos_m[1]) / (2 * self.sigma_meta2_y))
                    self.Bias += np.outer(kernelmeta_y, kernelmeta_x)
                    self.Force_bias_x += np.outer(kernelmeta_y, np.multiply(kernelmeta_x, (self.grid_x - pos_m[0])) / self.sigma_meta2_x )
                    self.Force_bias_y += np.outer(np.multiply(kernelmeta_y, (self.grid_y - pos_m[1])) / self.sigma_meta2_y, kernelmeta_x )
                
            ###--- Get PD (Probability Density) and PD_force from sampling data        
            PD_i, F_PD_x_i, F_PD_y_i = np.zeros(self.nbins_yx), np.zeros(self.nbins_yx), np.zeros(self.nbins_yx)
            for p in pos: 
                kernel_x = np.exp( - np.square(self.grid_x - p[0]) / (2 * self.bw2[0])) * self.const #add constant here for less computations
                kernel_y = np.exp( - np.square(self.grid_y - p[1]) / (2 * self.bw2[1]))
                PD_i += np.outer(kernel_y, kernel_x)
                kernel_x *= self.kT / self.bw2[2] #add constant here for less computations
                F_PD_x_i += np.outer(kernel_y, np.multiply(kernel_x, (self.grid_x - p[0])) )
                F_PD_y_i += np.outer(np.multiply(kernel_y, (self.grid_y - p[1])) , kernel_x )

            PD_i = np.where(PD_i > self.PD_limit, PD_i, 0)  # truncated probability density of window
            F_PD_x_i = np.divide(F_PD_x_i, PD_i, out=np.zeros_like(F_PD_x_i), where=PD_i > self.PD_limit)
            F_PD_y_i = np.divide(F_PD_y_i, PD_i, out=np.zeros_like(F_PD_y_i), where=PD_i > self.PD_limit)
        
        
            # Add the results to the total      
            self.PD += PD_i
            Force_x_i = F_PD_x_i + self.Force_bias_x - self.Force_static_x
            Force_y_i = F_PD_y_i + self.Force_bias_y - self.Force_static_y
            self.Force_num_x += np.multiply(Force_x_i, PD_i)
            self.Force_num_y += np.multiply(Force_y_i, PD_i)
            self.PD2 += np.square(PD_i)
            self.ofv_num_x += np.multiply(np.square(Force_x_i), PD_i)
            self.ofv_num_y += np.multiply(np.square(Force_y_i), PD_i)
            
            #calculate error
            if (self.i+1) in self.i_error_list or (self.i+1) == self.n_windows:
                self.Force_x = np.divide(self.Force_num_x, self.PD, out=np.zeros_like(self.Force_num_x), where=self.PD > self.PD_limit)
                self.Force_y = np.divide(self.Force_num_y, self.PD, out=np.zeros_like(self.Force_num_y), where=self.PD > self.PD_limit)
                self.sim_time = time_i / 1000
                self.calculate_errors()
                if self.calculate_error_change: self.calculate_difference_in_error()
                
                if self.print_info and print_analysis: 
                    if self.Z is not None: lib2.print_progress(start, self.i+1, self.n_windows, bar_length=50, variable_name='AAD', variable=round(self.AAD,3), variable_units="[kJ/mol]")
                    else: lib2.print_progress(start, self.i+1, self.n_windows, bar_length=50, variable_name='ST ERR', variable=round(self.Aofe,2), variable_units="[kJ/mol]")
                    
                if self.save_figures: print(f"{self.i = }", end=" | NEED TO IMPLEMENT PLOTTING")
                
        self.Force_x = np.divide(self.Force_num_x, self.PD, out=np.zeros_like(self.Force_num_x), where=self.PD > self.PD_limit)
        self.Force_y = np.divide(self.Force_num_y, self.PD, out=np.zeros_like(self.Force_num_y), where=self.PD > self.PD_limit)
        self.force_terms = [self.PD, self.PD2, self.Force_x, self.Force_y, self.ofv_num_x, self.ofv_num_y]
        
        self.save_data(save_data_path=self.simulation_folder_path)
        
##### ~~~~~ SRTR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #####

    def print_SRTR_info(self, info):
     
        if info == "sim_start":
      
                print(f"~~ S{len(self.force_terms)-1}  START ~~~")# PLUMED Startup time: ", round(time.time() - time_start_of_sim,2), " ...  to deposit "+str(n_pos)+" positions" )
                print(f"~~ S{len(self.force_terms)-1}  PHASE: {self.phase} ~~~ ", end="")
                if self.phase == "flat": print(f"Bias Type: \" {self.bias_type} \" ~~~ ", end="")
                if self.metad_height is not None and self.metad_height > 0: print(f" | MetaD h= {self.metad_height} , w= [{self.metad_width[0]},{self.metad_width[1]}] , bf= {self.biasfactor}", end="")
                if self.phase != "exploration": print(f" | InvF Bias sf= {self.Bias_sf}, guas sigma = {self.gaus_filter_sigma}", end="")
                if self.phase == "us": print(f" | hp_centre= [{self.hp_centre_x},{self.hp_centre_y}], hp_kappa= [{self.hp_kappa_x},{self.hp_kappa_y}]", end="")
                print(" | ")
                
        elif info == "sim_end":
            
            print(f"S{len(self.force_terms)-1:2} END | t={self.sim_time:5.2f}ns | t_tot={self.base_time+self.sim_time:5.2f}ns | n_pos: {len(self.position):4}/{self.n_pos_analysed[0]:5} : Aofe: {self.Aofe:5.3f} | ", end="")
            if self.Z is not None: print(f"AAD: {self.AAD:5.3f} | ", end="")
            if self.bootstrap_iter is not None and self.bootstrap_iter > 0:  print(f"ABS: {self.ABS_error:5.3f} | ", end="") 
            print(f"Reason for termination: {self.reason_for_termination}")

        elif info == "update_progress":
            
            if self.Avr_Error_list.shape[0] > 1: 
                print(f"S{len(self.force_terms)-1:2} | t={self.sim_time:5.2f}ns | t_tot={self.base_time+self.sim_time:5.2f}ns | n_pos: {len(self.position):4}/{self.n_pos_analysed[0]:5} : Aofe: {self.Aofe:5.3f} | ", end="")
                if self.Z is not None: print(f"AAD: {self.AAD:5.3f} | ", end="") 
                if self.bootstrap_iter is not None and self.bootstrap_iter > 0: print(f"ABS: {self.ABS_error:5.3f} ", end="") 
                print("")
            else: print(f"S{len(self.force_terms)-1:2} | t={self.sim_time:5.2f}ns | t_tot={self.base_time+self.sim_time:5.2f}ns | n_pos: {len(self.position):4}/{self.n_pos_analysed[0]:5} : Aofe: {self.Aofe:4.2f} | AAD: {self.AAD:4.2f}")
        
        elif info == "restart_SRTR":
        
            print(f"\nRestarted SRTR campaign with {len(self.force_terms)-1} existing simulations and {self.Avr_Error_list[-1][0]:.4f} ns existing simulation time.")
            print(f"Aofe = {self.Aofe:.3f} | ", end="")
            if self.Z is not None: print(f"AAD = {self.AAD:.3f} | ", end="")
            if self.bootstrap_iter is not None and self.bootstrap_iter > 0: print(f"ABS = {self.ABS_error:.3f} | ", end="")
            print(f"\nContinuing with phase: {self.phase}\n")
                
    def initialise_SRTR(self, ID, goal, main_error_type, time_budget, guaranteed_sim_time, max_sim_time, us_criteria_max_avr_ratio, restart_SRTR):

        # if ID is specified, set self.ID to ID
        if ID != "": self.ID = ID
        
        # Move into SRTRcampaign. If it does not exist, create it
        self.campaign_path = f"{self.simulation_folder_path}SRTRcampaign{ID}/"
        if restart_SRTR and os.path.exists(self.campaign_path) is False: 
            print("\nSimulation folder does not exist. \nCannot restart SRTR campaign\nContiniue with new campaign\n")
            restart_SRTR = False     
        self.current_path = lib2.set_up_folder(self.campaign_path, remove_folder=(not restart_SRTR))
        
        # initialise lists and vaiables
        self.force_terms = np.array([np.zeros((6, self.nbins_yx[0], self.nbins_yx[1]))])
        self.goal_reached, self.print_info = False, False

        # lists for calculating the change in the error
        if self.calculate_error_change == False:
            self.calculate_error_change = True
            self.d_Aofe = np.empty((0,4))
            if self.Z is not None: self.d_AAD = np.empty((0,4))
            if self.weighted_avr_error_change: 
                self.d_Aofe_w = np.empty((0,4))
                if self.Z is not None: self.d_AAD_w = np.empty((0,4))

        # list for storing the position analysed in each simulation. n_pos_analysed[0] is the total number of positions analysed in all simulations.          
        self.n_pos_analysed = [0]

        # set simulation times
        if time_budget is not None: self.time_budget = time_budget
        if self.time_budget is None: 
            self.time_budget = 100
            print(f"\nTime budget not set. Will use default time budget of {self.time_budget} ns\n")
        if guaranteed_sim_time is not None: self.guaranteed_sim_time = guaranteed_sim_time
        if self.guaranteed_sim_time is None: 
            self.guaranteed_sim_time = self.time_budget / 20
            print(f"\nGuaranteed simulation time not set. Will use guaranteed simulation time of {self.guaranteed_sim_time} ns \n")
        if max_sim_time is not None: self.max_sim_time = max_sim_time
        if self.max_sim_time is None:
            self.max_sim_time = self.time_budget/2
            print(f"\nMax simulation time not set. Will use max simulation time of {self.max_sim_time} ns \n")
        
        # calculate number of simulation steps used for simulation input file
        self.simulation_steps = int(1.1*(self.max_sim_time / self.time_step * 1000))
        
        # set SRTR parameters
        if main_error_type is not None: self.main_error_type = main_error_type
        self.goal = goal
        self.original_Bias_sf = self.Bias_sf
        self.us_criteria_max_avr_ratio = us_criteria_max_avr_ratio     
        
        # start in the exploration phase.
        self.phase = "exploration" 
        
        # estimate metad_height for the exploration phase         
        self.metad_width_exploration = self.metad_width if self.metad_width is not None else [round((self.grid_max[0] - self.grid_min[0]) / 40,5), round((self.grid_max[1] - self.grid_min[1]) / 40,5)]
        if self.metad_height is not None and self.metad_height > 0: self.metad_height_exploration = self.metad_height
        else:
            expected_time_ps = self.time_budget / 10 * 1000
            metad_pace_ps = self.metad_pace * self.time_step
            n_hills = expected_time_ps / metad_pace_ps
            area_to_fill = self.FES_cutoff * (self.grid_max[0] - self.grid_min[0]) * (self.grid_max[1] - self.grid_min[1]) / 3
            self.metad_height_exploration = area_to_fill / (6.2831853 * n_hills * self.metad_width_exploration[0] * self.metad_width_exploration[1])
            self.metad_height_exploration = round(max(self.metad_height_exploration , self.FES_cutoff/5),2) # to avoid self.metad_height_exploration to be too small, set it to at elast 1/5 of the FES_cutoff.
        
        if restart_SRTR:
                        
            # Find existing simulation folders
            simulation_folder_prefix = "simulation" + ID + "_"
            existing_simulation_folders = [f for f in os.listdir() if simulation_folder_prefix in f] 
            existing_simulation_folders = sorted(existing_simulation_folders, key=lambda x: int(x.split('_')[-1])) 
            
            if len(existing_simulation_folders) == 0: 
                print("\nNo existing simulation folders found. \nCannot restart SRTR campaign\nContiniue with new campaign\n")
                return
            
            existing_bias = np.zeros(self.nbins_yx)
                        
            for sim_folder in existing_simulation_folders:
                
                self.current_path = lib2.set_up_folder(self.campaign_path+sim_folder, remove_folder=False)
                SIM_ID = self.ID + "_" + sim_folder.split("_")[-1]
                self.hills_file, self.position_file = f"HILLS{SIM_ID}", f"position{SIM_ID}"

                if os.path.exists(self.position_file) is False:
                    print(f"\nCannot find position file with name: \"{self.position_file}\" in the folder \"{sim_folder}\"\nFolder will be skipped")
                    os.chdir("..")
                    continue
                
                # initialise variables for new simulation
                self.force_terms = np.concatenate((self.force_terms, [np.zeros((6, self.nbins_yx[0], self.nbins_yx[1]))]), axis=0)
                if len(self.Avr_Error_list) == 0: 
                    self.Aofe_0 = 50
                    if self.Z is not None: self.AAD_0 = 10
                else: 
                    [self.base_time, self.Aofe_0] = [self.Avr_Error_list[-1][0], self.Avr_Error_list[-1][2]]
                    if self.Z is not None: self.AAD_0 = self.Avr_Error_list[-1][self.aad_index]
                    
                # if file exists with name "self.hills_file", self.metad_height = 1 else self.metad_height = None
                if os.path.exists(self.hills_file): self.metad_height = 1
                print(f"Loading simulation folder: {sim_folder}")#, {self.hills = }, {self.position.shape = }, {self.hills_file = }, {self.position_file = }, {os.getcwd() = }")
                self.load_data(n_pos_analysed=0)
                
                # if force terms exist, load them
                if os.path.exists(f"force_terms{SIM_ID}.pkl"): PD_sim, PD2_sim, Force_x_sim, Force_y_sim, ofv_num_x_sim, ofv_num_y_sim = lib2.load_pkl(f"force_terms{SIM_ID}.pkl")                
                else:
                    print(f"force terms file not found. Calculating force terms ", end="")
                    self.const, self.bw2 = self.position_pace / (self.bw[0] * self.bw[1]) , [self.bw[0] ** 2, self.bw[1] ** 2, self.bw[0] * self.bw[1]]  
                    if os.path.exists("external_bias.dat"):
                        print("with external bias. ")
                        plx, ply, plBias, plfbiasx, plfbiasy = lib2.read_plumed_grid_file("external_bias.dat")    
                        Force_static_x, Force_static_y = plfbiasx[self.pl_extra[0]:-self.pl_extra[1], self.pl_extra[2]:-self.pl_extra[3]], plfbiasy[self.pl_extra[0]:-self.pl_extra[1], self.pl_extra[2]:-self.pl_extra[3]]
                    else: 
                        print("without external bias. ")
                        Force_static_x, Force_static_y = np.zeros(self.nbins_yx), np.zeros(self.nbins_yx)
                    PD_sim, PD2_sim, Force_x_sim, Force_y_sim, ofv_num_x_sim, ofv_num_y_sim, _, _, _, _ = lib2.MFI_forces(self.hills, self.position[:,1], self.position[:,2], self.const, self.bw2, self.kT, self.X, self.Y, Force_static_x, Force_static_y, self.n_pos_per_window, self.Gamma_Factor, periodic=self.periodic, PD_limit = self.PD_limit, return_FES=False)       
                    
                self.patch_and_find_error_SRTR(np.array([PD_sim, PD2_sim, Force_x_sim, Force_y_sim, ofv_num_x_sim, ofv_num_y_sim]), np.array([np.zeros(self.nbins_yx), np.zeros(self.nbins_yx), np.zeros(self.nbins_yx)])) 
                
                if self.System != "Langevin": 
                    new_traj_path = lib2.get_trajectory_xtc_file_path(self.current_path, System=self.System)
                    if new_traj_path is not None: self.sim_files_path["trajectory_xtc_file_path_list"] = [new_traj_path] + self.sim_files_path["trajectory_xtc_file_path_list"].copy()
                
                if self.phase == "exploration": 
                    existing_bias += lib2.find_total_bias_from_hills(self.X, self.Y, self.hills, periodic=self.periodic)
                    fes_with_bias = self.FES + existing_bias
                    if (fes_with_bias > self.FES_cutoff).all(): self.phase = "metad"
                
                # move back to simulation_folder
                os.chdir("..")    
                    
            # check if goal is reached
            if self.main_error_type == "ST_ERR": self.goal_reached = self.Aofe < self.goal
            elif self.main_error_type == "AAD": self.goal_reached = self.AAD < self.goal
            else: raise ValueError("main_error_type not recognised")
            if self.goal_reached:
                print(f"\nGoal reached in previous simulations. \nNo new simulations will be started.\n")
                return
            
            # check if time_budget is reached                            
            if self.time_budget is not None: self.goal_reached = self.Avr_Error_list[-1][0] > self.time_budget
            if self.goal_reached: 
                print(f"\nTime budget reached in previous simulations. \nNo new simulations will be started.\n")
                return  
                        
            # # Decide between metad and flat phase          
            # if self.phase == "metad":
                
            #     if self.bootstrap_iter is not None and self.bootstrap_iter > 0 and not np.isnan(self.Avr_Error_list[-1, self.abs_error_index]):
            #         if self.Avr_Error_list[-1, self.abs_error_index] < self.FES_cutoff / 20: self.phase = "flat"
                
            #     elif self.calculate_FES_st_dev:
            #         if self.Avr_Error_list[-1, self.FES_st_dev_index] < self.FES_cutoff / 20: self.phase = "flat"
                
            #     elif self.record_maps:
            #         st_dev = np.sum(np.sqrt(np.var(np.array(self.Maps_list)[-30:,0], axis=0) * self.Maps_list[-1][1])) / np.count_nonzero(self.Maps_list[-1][1])
            #         if st_dev < self.FES_cutoff/20: self.phase = "flat"
                    
            #     else:
            #         vol_last, min_vol_last_5 = self.Avr_Error_list[-1,1], np.min(self.Avr_Error_list[-5:,1])
            #         max_vol_diff = (vol_last - min_vol_last_5) / min_vol_last_5 * 100
            #         if max_vol_diff < 5: self.phase = "flat"
            
            self.print_SRTR_info("restart_SRTR")    
        
    def start_new_sim_SRTR(self):
        
        # Move into folder for current simulation 
        self.SIM_ID = f"{self.ID}_{len(self.force_terms)}"
        self.current_path = lib2.set_up_folder(f"{self.campaign_path}simulation{self.SIM_ID}", remove_folder=True)
        self.hills_file, self.position_file = f"HILLS{self.SIM_ID}", f"position{self.SIM_ID}"
        
        #initialise variables
        self.n_pos_analysed.append(0)
        self.error_strike = 0
        self.force_terms = np.concatenate((self.force_terms, [np.zeros((6, self.nbins_yx[0], self.nbins_yx[1]))]), axis=0)
        
        # define base_time, Aofe_0, AAD_0 at start of simulation
        if len(self.Avr_Error_list) == 0: 
            self.Aofe_0 = 50
            if self.Z is not None: self.AAD_0 = 10
        else: 
            [self.base_time, self.Aofe_0] = [self.Avr_Error_list[-1][0], self.Avr_Error_list[-1][2]]
            if self.Z is not None: self.AAD_0 = self.Avr_Error_list[-1][self.aad_index]
        
        # Get new simulation parameters (will depend on phase)
        self.get_new_simulation_parameters_SRTR()
        self.const, self.bw2 = self.position_pace / (self.bw[0] * self.bw[1]) , [self.bw[0] ** 2, self.bw[1] ** 2, self.bw[0] * self.bw[1]]           
        
        # if applicable, make external bias
        self.Bias_static, self.Force_static_x, self.Force_static_y = np.zeros(self.nbins_yx), np.zeros(self.nbins_yx), np.zeros(self.nbins_yx)
        if self.Bias_sf > 0 and len(self.Avr_Error_list) > 0: 
            if self.phase == "flat":                   
                self.bias_type, self.Bias_sf = "", self.original_Bias_sf 
                # if len(self.force_terms)-1 < 3: self.bias_type, self.Bias_sf = "", self.original_Bias_sf 
                # elif len(self.force_terms)-1 % 3 == 0: self.bias_type, self.Bias_sf = "", self.original_Bias_sf 
                # elif len(self.force_terms)-1 % 3 == 1: self.bias_type, self.Bias_sf = "error", 1
                # else: self.bias_type, self.Bias_sf = "PD", 1
            elif self.phase == "us": self.bias_type, self.Bias_sf = "", 1
            else: self.Bias_sf = self.Bias_sf
            
            self.make_external_bias()
        
        # if applicable, add harmonic potential to Force_static_x , Force_static_y and Bias_static
        self.make_harmonic_bias()
        
        # set bias and bias_force to zero            
        self.Bias, self.Force_bias_x, self.Force_bias_y = np.zeros(self.nbins_yx), np.zeros(self.nbins_yx), np.zeros(self.nbins_yx)
        
        # Start New Langevin Simulation ~~~~~ ###
        self.run_simulation(assign_process=True, simulation_path="", file_extension=self.SIM_ID)   

        # wait for at least 100 new positions, then continue
        n_pos = lib2.wait_for_positions(10, sleep_between_checks=0.1, position_path=self.position_file)
        
        self.print_SRTR_info("sim_start")
        
        if self.System != "Langevin": 
            new_traj_path = lib2.get_trajectory_xtc_file_path(self.current_path, System=self.System)
            if new_traj_path is not None: self.sim_files_path["trajectory_xtc_file_path_list"] = [new_traj_path] + self.sim_files_path["trajectory_xtc_file_path_list"].copy()

    def get_new_simulation_parameters_SRTR(self):
        
        if self.phase == "exploration":
            # if not specified, set biasfactor to FES_cutoff. This should be high enough, so that there wont be a significant damping of the Bias hight.
            if self.biasfactor is None: self.biasfactor = int(self.FES_cutoff)
            # if not specified, estimate metad_width as length_of_grid / 50 (x_min=-5, x_max=5 -> metad_width=0.2)
            if self.metad_width is None: self.metad_width = self.metad_width_exploration
            # if not specified, set metad_height to metad_height_exploration
            if self.metad_height is None: self.metad_height = self.metad_height_exploration
            return

        if self.phase == "metad":
            
            # set biasfactor to 1 + FES_cutoff/5.
            self.biasfactor = int(2 + self.FES_cutoff/5) + 1  
            
            # estimate the metad_width by fitting the minima with Gaussians and taking the smallest sigma.
            # self.metad_width = [round((self.grid_max[0] - self.grid_min[0]) / 50,5), round((self.grid_max[1] - self.grid_min[1]) / 50,5)]
            _, _, parameter_list = lib2.Gaus_fitting_to_fes_2D(self.X, self.Y, np.where(self.cutoff > 0.5, self.FES, self.FES_cutoff))
            self.metad_width = [round(min([p[1] for p in parameter_list])/4,5), round(min([p[2] for p in parameter_list])/4,5)]
            
            # set the metad_height to FES_cutoff / 30
            self.metad_height = round(min(self.metad_height_exploration/3, self.FES_cutoff / 10),2) + 1
            
            return
        
        # check if us_criteria_max_avr_ratio is defined. If it is, check if the max error is above the (avr error)*us_criteria_max_avr_ratio.
        if self.us_criteria_max_avr_ratio is not None: 
            smooth_error_avr, smooth_error_max = lib2.check_US_criteria(error_map=self.ofe, cutoff_map=self.cutoff, gaussian_sigma=3)
            if smooth_error_max > self.us_criteria_max_avr_ratio * smooth_error_avr:
                
                print(f" ******* {smooth_error_avr=:.4f} < {smooth_error_max=:.4f} *******")
                
                # Find position of max error for hp_centre. if the previous simulation was in the US phase, check if the new hp centre is somewhere else. If centre in the same position, switch to flat phase.
                self.hp_centre_x, self.hp_centre_y = lib2.find_hp_centre(self.X, self.Y, self.ofe, self.cutoff, gaussian_sigma=3, prev_hp_centre_x=[self.hp_centre_x], prev_hp_centre_y=[self.hp_centre_y])
                
                if self.hp_centre_x is not None or self.hp_centre_y is not None:
                    self.phase = "us"
                    self.hp_kappa_x, self.hp_kappa_y = 10/self.grid_length[0], 10/self.grid_length[1]
                    self.metad_height, self.lw_kappa_x, self.lw_kappa_y, self.uw_kappa_x, self.uw_kappa_y = None, None, None, None, None
                    return
        
        self.phase = "flat"
           
        # set biasfactor to 1 + FES_cutoff/40.
        self.biasfactor = round(2 + self.FES_cutoff/10,3)  
        
        # estimate the metad_width by fitting the minima with Gaussians and taking the smallest sigma.
        # self.metad_width = [round((self.grid_max[0] - self.grid_min[0]) / 50,5), round((self.grid_max[1] - self.grid_min[1]) / 50,5)]
        _, _, parameter_list = lib2.Gaus_fitting_to_fes_2D(self.X, self.Y, np.where(self.cutoff > 0.5, self.FES, self.FES_cutoff))
        self.metad_width = [round(min([p[1] for p in parameter_list])/6,5), round(min([p[2] for p in parameter_list])/6,5)]
        
        # set the metad_height to FES_cutoff / 30
        self.metad_height = round(min(self.metad_height_exploration/10, self.FES_cutoff / 20),2) + 0.1
        
        self.hp_kappa_x, self.hp_kappa_y = None, None 
        self.lw_kappa_x, self.lw_kappa_y, self.uw_kappa_x, self.uw_kappa_y = None, None, None, None
        
        # adjust position pace ?
        
        return

    def patch_and_find_error_SRTR(self, force_terms_e, bias_terms_e, patch_realtime_forces=True):
        
        # if patch_realtime_forces is True, patch the force terms of simulation i with the (new) force_terms_e. If False, force_terms_e is the new force terms of the entire simulation.
        if patch_realtime_forces: 
            
            # record bias potential and bias force
            self.Force_bias_x += bias_terms_e[0]
            self.Force_bias_y += bias_terms_e[1]
            self.Bias += bias_terms_e[2]            
            
            # record last force terms (_e) of latest simulation  
            if self.record_forces_e: [self.PD_e, self.PD2_e, self.Force_x_e, self.Force_y_e, self.ofv_num_x_e, self.ofv_num_y_e] = force_terms_e
            # Patch forces of latest simulation with the (new) force_terms_e
            self.force_terms[-1] = lib2.patch_forces(force_terms_e, self.force_terms[-1], PD_limit=self.PD_limit)
            # Patch forces of all simulations
            self.force_terms[0] = lib2.patch_forces(force_terms_e, self.force_terms[0], PD_limit=self.PD_limit)
            # record n_pos_analysed of latest simulation
            self.n_pos_analysed[-1] += len(self.position)
        
        else: 
            
            # record bias potential and bias force
            self.Force_bias_x = bias_terms_e[0]
            self.Force_bias_y = bias_terms_e[1]
            self.Bias = bias_terms_e[2]               
            
            if self.record_forces_e: 
                PD_sim, PD2_sim, Force_x_sim, Force_y_sim, ofv_num_x_sim, ofv_num_y_sim = force_terms_e
                PD_last, PD2_last, Force_x_last, Force_y_last, ofv_num_x_last, ofv_num_y_last = self.force_terms[-1]
                [self.PD_e, self.PD2_e, self.ofv_num_x_e, self.ofv_num_y_e] = [PD_sim - PD_last, PD2_sim - PD2_last, ofv_num_x_sim - ofv_num_x_last, ofv_num_y_sim - ofv_num_y_last]
                self.Force_x_e = np.divide(np.multiply(Force_x_sim, PD_sim) - np.multiply(Force_x_last, PD_last), self.PD_e, out=np.zeros_like(self.PD_e), where=self.PD_e > self.PD_limit)
                self.Force_y_e = np.divide(np.multiply(Force_y_sim, PD_sim) - np.multiply(Force_y_last, PD_last), self.PD_e, out=np.zeros_like(self.PD_e), where=self.PD_e > self.PD_limit)                
            # replace force terms with the force terms of the simulation analysed in one go
            self.force_terms[-1] = force_terms_e
            # Patch forces of all simulations
            self.force_terms[0] = lib2.patch_forces(self.force_terms[1:], PD_limit=self.PD_limit)
            # record n_pos_analysed of latest simulation
            self.n_pos_analysed[-1] = len(self.position)
        
        # record n_pos_analysed and sim_time    
        self.n_pos_analysed[0] = sum(self.n_pos_analysed[1:])
        self.sim_time = self.position[-1, 0] / 1000
        
        # get new error and difference in error
        self.PD, self.PD2, self.Force_x, self.Force_y, self.ofv_num_x, self.ofv_num_y = self.force_terms[0]
        self.calculate_errors()
        self.calculate_difference_in_error()        

        
    def check_termination_criteria_SRTR(self, ReInit_criteria, Strike_factor=100):
        
        if self.Avr_Error_list.shape[0] < 2: return False
                     
        # check if goal is reached                
        if self.main_error_type == "ST_ERR": self.goal_reached = self.Aofe < self.goal
        elif self.main_error_type == "AAD": self.goal_reached = self.AAD < self.goal
        else: raise ValueError("main_error_type not recognised")
        if self.goal_reached: 
            self.reason_for_termination = "GOAL REACHED"
            return True
        
        # check if time_budget is reached                            
        if self.time_budget is not None: self.goal_reached = self.Avr_Error_list[-1][0] > self.time_budget
        if self.goal_reached: 
            self.reason_for_termination = "TIME BUDGET REACHED"
            return True       
         
        # check if simulation is above max_sim_time
        if self.sim_time > self.max_sim_time: 
            self.reason_for_termination = "Max simulation time reached"
            return True 
        
        if self.phase == "exploration": 
            
            # check if FES + Bias > FES_cutoff
            fes_with_bias = np.where(self.PD < self.PD_cutoff, self.FES_cutoff+1, self.FES + self.Bias)
            
            if (fes_with_bias > self.FES_cutoff).all() or self.sim_time > self.time_budget/2: 
                self.phase = "metad"
                self.reason_for_termination = "Exploration stage completed"
                return True
            else: return False   
            
        if self.phase == "metad" and self.sim_time > self.guaranteed_sim_time/2:
            
            # if self.bootstrap_iter is not None and self.bootstrap_iter > 0 and not np.isnan(self.Avr_Error_list[-1, self.abs_error_index]):
            #     if self.Avr_Error_list[-1, self.abs_error_index] < self.FES_cutoff / 20: 
            #         self.phase = "flat"
            #         self.reason_for_termination = f"Metad phase completed (abs_error={self.Avr_Error_list[-1, self.abs_error_index]:.4f})"
            #         return True
            
            # elif self.calculate_FES_st_dev:
            #     if self.Avr_Error_list[-1, self.FES_st_dev_index] < self.FES_cutoff / 20: 
            #         self.phase = "flat"
            #         self.reason_for_termination = f"Metad phase completed (var_fes={self.Avr_Error_list[-1, self.FES_st_dev_index]:.4f})"
            #         return True
            
            # elif self.record_maps:
            #     # flat / US phase will be start if var of the last 10 FES maps is small enough. If not, continue metad.    
            #     st_dev = np.sum(np.sqrt(np.var(np.array(self.Maps_list)[-30:,0], axis=0) * self.Maps_list[-1][1])) / np.count_nonzero(self.Maps_list[-1][1])
            #     if st_dev < self.FES_cutoff/20: 
            #         self.phase = "flat"
            #         self.reason_for_termination = f"Metad phase completed (var_fes={st_dev:.4f})"
            #         return True    
                
            # else: 
            #     vol_last, min_vol_last_5 = self.Avr_Error_list[-1,1], np.min(self.Avr_Error_list[-5:,1])
            #     max_vol_diff = (vol_last - min_vol_last_5) / min_vol_last_5 * 100
            #     if max_vol_diff < 5: 
            #         self.phase = "flat"
            #         self.reason_for_termination = f"Metad phase completed (max_vol_diff={max_vol_diff:.2f}%)"
            #         return True
            
            pass
                
        if self.phase == "us":
                    
            smooth_error_avr, smooth_error_hp_centre = lib2.check_US_criteria(self.ofe, self.cutoff, gaussian_sigma=3, check_termination=[self.X, self.Y, self.hp_centre_x, self.hp_centre_y])
            if smooth_error_hp_centre < smooth_error_avr: 
                self.reason_for_termination = f"US criteria reached ({smooth_error_hp_centre=:.4f} < {smooth_error_avr=:.4f})"
                return True    
        
            if self.sim_time > self.guaranteed_sim_time/2: return True

        # get the change in error [change error wrt last error, change error wrt last error / dt, total change error, total change error / dt_tot]
        [self.d_err, self.d_err_dt, self.dtot_err, self.dtot_err_dt] = self.d_error[-1]
            
        # check the change in error. If the change in error is negative enough, simulation continiues
        if self.d_err_dt > ReInit_criteria/Strike_factor: self.error_strike += 1
        if self.dtot_err_dt < ReInit_criteria and self.error_strike < 4: return False 
        else:
            if self.error_strike > 3: self.reason_for_termination = f"Max error strike reached (n_strike={self.error_strike})"
            else: self.reason_for_termination = f"Change in error above ReInit_criteria dtot_err_dt={self.dtot_err_dt:.4f} > RIC={ReInit_criteria:.4f}"                       
        

        if (self.phase == "flat" or self.phase == "metad") and self.sim_time < self.guaranteed_sim_time: return False
        if self.phase == "us" and self.sim_time < self.guaranteed_sim_time/10: return False
        return True  

    def re_analyse_SRTR(self):
        # Re-analyse all the data again in one go. This is done to increase the numerical accuray and to check if the results are consistent. (minor inconsistencies might arrise due to additional data egnerated since the last check)
        # error_old = self.AAD if self.Z is not None else self.Aofe
        # prev_n_pos_analysed = self.n_pos_analysed[-1]
    
        # load entire simulation data and analyse in one go
        self.load_data(n_pos_analysed=0)
        PD_sim, PD2_sim, Force_x_sim, Force_y_sim, ofv_num_x_sim, ofv_num_y_sim, Force_bias_x_sim, Force_bias_y_sim, Bias_sim, _ = lib2.MFI_forces(self.hills, self.position[:,1], self.position[:,2], self.const, self.bw2, self.kT, self.X, self.Y, self.Force_static_x, self.Force_static_y, self.n_pos_per_window, self.Gamma_Factor, periodic=self.periodic, PD_limit = self.PD_limit, return_FES=False)

        # patch results with existing results and calculate error
        self.patch_and_find_error_SRTR(np.array([PD_sim, PD2_sim, Force_x_sim, Force_y_sim, ofv_num_x_sim, ofv_num_y_sim]), np.array([Force_bias_x_sim, Force_bias_y_sim, Bias_sim]), patch_realtime_forces=False) 

        # # check if aad for all data patched at once is the same as the one calculated in real time
        # error_new = self.AAD if self.Z is not None else self.Aofe
        # if abs(error_new - error_old)/error_new > 0.1: 
        #     print(f"\n*** ATTENTION ***: \nSignificant difference in Re-Analysis: {error_old = :.4f} -> {error_new = :.4f}, has a {abs(error_new - error_old)/error_new:.2%} difference")
        #     print(f"n_pos_analysed: {prev_n_pos_analysed} -> {len(self.position)}")
            
    def stop_simulation_SRTR(self):
        
        # kill simulation process. Wait to confirm it shut down
        os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)                        
        
        # take hills and position files and analyse in one go                      
        self.re_analyse_SRTR()
        
        # print info
        self.print_SRTR_info("sim_end")
        
        # save simulation results
        if self.save_force_terms: lib2.save_pkl(self.force_terms[-1], f"force_terms{self.SIM_ID}.pkl")  
        
        # if goal reached (end of campaign), save results    
        if self.goal_reached:
            if np.sum(self.force_terms[-1,2]) == 0: raise ValueError("No forces were calculated. Simulation was not successful")
            
            # save campaign results if goal reached
            self.save_data(save_data_path=self.campaign_path)         
                    
        # check if simulation was terminated successfully            
        if self.process.poll() is None: print("\n\nFailed to terminate the process")

    def MFI_real_time_ReInit(self, ID="", goal=0.25, main_error_type=None, ReInit_criteria=-1, n_pos_before_analysis=500, time_budget=100, guaranteed_sim_time=None, max_sim_time=None, us_criteria_max_avr_ratio=None, restart_SRTR=False):
        
        # Initialisation
        self.initialise_SRTR(ID=ID, goal=goal, main_error_type=main_error_type, time_budget=time_budget, guaranteed_sim_time=guaranteed_sim_time, max_sim_time=max_sim_time, us_criteria_max_avr_ratio=us_criteria_max_avr_ratio, restart_SRTR=restart_SRTR)
                
        # Itterate over simulations            
        while self.goal_reached == False:

            # Set-up and start new simulation
            self.start_new_sim_SRTR()
            
            while self.process.poll() is None and self.goal_reached == False:
            
                #wait for at least _x_ new positions
                lib2.wait_for_positions(n_pos_before_analysis, self.n_pos_analysed[-1], return_n_pos=False, position_path=self.position_file)  
                
                # load and analyse the new data 
                self.load_data(n_pos_analysed=self.n_pos_analysed[-1])
                PD_i, PD2_i, Force_x_i, Force_y_i, ofv_num_x_i, ofv_num_y_i, F_bias_x_i, F_bias_y_i, Bias_i, _ = lib2.MFI_forces(self.hills, self.position[:,1], self.position[:,2], self.const, self.bw2, self.kT, self.X, self.Y, self.Force_static_x - self.Force_bias_x, self.Force_static_y - self.Force_bias_y, self.n_pos_per_window, self.Gamma_Factor, periodic=self.periodic, PD_limit = self.PD_limit, return_FES=False)       
                self.patch_and_find_error_SRTR(np.array([PD_i, PD2_i, Force_x_i, Force_y_i, ofv_num_x_i, ofv_num_y_i]), np.array([F_bias_x_i, F_bias_y_i, Bias_i]))
                                
                # check termination criteria
                stop_sim = self.check_termination_criteria_SRTR(ReInit_criteria)
                
                # terminate simulation if stop_sim or goal_reached. Otherwise, print progress
                if (stop_sim or self.goal_reached): self.stop_simulation_SRTR()
                else: self.print_SRTR_info("update_progress")

##### ~~~~~ Parallel ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #####
                        
    @dataclass
    class MFI_parallel:
        parent: 'MFI2D'  # Reference to the enclosing MFI2D instance
        workers: int
        n_cores_per_simulation: int = None
        
        simulation_folder_path_list: Optional[List[str]] = None        
        simulation_steps_list: Optional[List[int]] = None
        initial_position_list: Optional[list] = None
        bw_list: Optional[List[list]] = None
        biasfactor_list: Optional[List[float]] = None
        metad_width_list: Optional[List[list]] = None
        metad_height_list: Optional[List[float]] = None
        metad_pace_list: Optional[List[int]] = None
        position_pace_list: Optional[List[int]] = None
        
        sim: List['MFI2D'] = field(default_factory=list)
        
        
        def __post_init__(self):
            
            self.patent.record_forces_e = True
            
            parent_params = [asdict(self.parent) for _ in range(self.workers)]
            
            if self.simulation_folder_path_list is not None:
                for i, path in enumerate(self.simulation_folder_path_list): parent_params[i]['simulation_folder_path'] = path
            if self.initial_position_list is not None:
                for i, pos in enumerate(self.initial_position_list): parent_params[i]['initial_position'] = pos 
            if self.simulation_steps_list is not None:
                for i, steps in enumerate(self.simulation_steps_list): parent_params[i]['simulation_steps'] = steps
            if self.metad_height_list is not None:
                for i, height in enumerate(self.metad_height_list): parent_params[i]['metad_height'] = height
            if self.metad_width_list is not None:
                for i, width in enumerate(self.metad_width_list): parent_params[i]['metad_width'] = width
            if self.biasfactor_list is not None:
                for i, biasfactor in enumerate(self.biasfactor_list): parent_params[i]['biasfactor'] = biasfactor
            if self.metad_pace_list is not None:
                for i, pace in enumerate(self.metad_pace_list): parent_params[i]['metad_pace'] = pace
            if self.position_pace_list is not None:
                for i, pace in enumerate(self.position_pace_list): parent_params[i]['position_pace'] = pace
            if self.bw_list is not None:
                for i, bw in enumerate(self.bw_list): parent_params[i]['bw'] = bw
            
            for i in range(len(parent_params)): parent_params[i]['print_info'] = False    
            if self.parent.print_info: parent_params[0]['print_info'] = True
                
            self.sim = [self.parent.__class__(**parent_params[i]) for i in range(self.workers)]

        def run_parallel_sim(self):
            self.simulation_process = []
            for sim_i in self.sim: 
                
                p = multiprocessing.Process(target=sim_i.run_simulation, args=())
                p.start()
                self.simulation_process.append(p)
                
            for p in self.simulation_process: p.join()
            print("All simulations finished")
                        
        def analyse_parallel(self):
            for i, sim_i in enumerate(self.sim): 
                print(f"Analysing simulation {i} / {len(self.sim)}")
                sim_i.analyse_data()
            print("All simulations analysed")
                
        def patch_simulations(self):
            
            print("Patching simulations")
            
            force_terms_collection = []
            for sim_i in self.sim: force_terms_collection.append(sim_i.forces_e_list)
            force_terms_collection = np.array(force_terms_collection, dtype=float)
            assert len(np.shape(force_terms_collection)) == 5, "force_terms_collection must be a 4D array with dimensions (n_sim, n_iter, 4_force_terms, nbins_y, nbins_x)"
            
            n_iter = np.shape(force_terms_collection)[1]
                       
            for i in range(n_iter):
                for j in range(len(self.sim)):
                    self.parent.PD += force_terms_collection[j, i, 0, :, :]
                    self.parent.PD2 += force_terms_collection[j, i, 1, :, :]
                    self.parent.Force_num_x += force_terms_collection[j, i, 2, :, :] * force_terms_collection[j, i, 0, :, :]
                    self.parent.Force_num_y += force_terms_collection[j, i, 3, :, :] * force_terms_collection[j, i, 0, :, :]
                    self.parent.ofv_num_x += force_terms_collection[j, i, 4, :, :]
                    self.parent.ofv_num_y += force_terms_collection[j, i, 5, :, :]
                
                self.parent.Force_x = np.divide(self.parent.Force_num_x, self.parent.PD, out=np.zeros_like(self.parent.Force_num_x), where=self.parent.PD>self.parent.PD_limit)
                self.parent.Force_y = np.divide(self.parent.Force_num_y, self.parent.PD, out=np.zeros_like(self.parent.Force_num_y), where=self.parent.PD>self.parent.PD_limit)
                self.parent.sim_time = sum([self.sim[ii].Avr_Error_list[i][0] for ii in range(len(self.sim))])
                self.parent.calculate_errors(force_terms_tot=[self.parent.PD, self.parent.PD2, self.parent.Force_x, self.parent.Force_y, self.parent.ofv_num_x, self.parent.ofv_num_y])
                
        def plot_parallel_results(self):
            
            plt.figure(figsize=(10,4))
            plt.subplot(1,2,1)
            
            plt.plot(np.array(self.parent.Avr_Error_list)[:,0], np.array(self.parent.Avr_Error_list)[:,2], color="black", linewidth=2, alpha=0.7, label=f"Patched Sim") 
            for i in range(len(self.sim)): plt.plot(np.array(self.sim[i].Avr_Error_list)[:,0], np.array(self.sim[i].Avr_Error_list)[:,2], label=f"Sim {i}", linewidth=0.5)
            plt.title("Progression of Error of the mean force"); plt.xlabel("Time [ns]"); plt.ylabel("Mean Force Error [kJ/mol]"); plt.yscale("log")

            if self.parent.Z is not None:
                plt.subplot(1,2,2)
                plt.plot(np.array(self.parent.Avr_Error_list)[:,0], np.array(self.parent.Avr_Error_list)[:,self.parent.aad_index], color="black", linewidth=2, alpha=0.7, label=f"Patched Sim")
                for i in range(len(self.sim)): plt.plot(np.array(self.sim[i].Avr_Error_list)[:,0], np.array(self.sim[i].Avr_Error_list)[:,self.parent.aad_index], label=f"Sim {i}", linewidth=0.5)
                plt.title("Progression of Avr. Abs. Deviation of the FES"); plt.xlabel("Time [ns]"); plt.ylabel("AAD [kJ/mol]"); plt.yscale("log")

            plt.legend()
            plt.tight_layout(); plt.show()

##### ~~~~~ PRTR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #####

    @dataclass
    class MFI_parallel_RTR:
        parent: 'MFI2D'  # Reference to the enclosing MFI1D instance
        workers: int
        n_cores_per_simulation: int = None
        ID: str = ""
        goal: float = 0.25
        main_error_type: str = "AAD"
        restart_PRTR: bool = False
        
        guaranteed_sim_time: float = 2.0
        max_sim_time: float = 5.0
        time_budget: float = 30.0
        n_pos_before_analysis: int = 3000
        
        initial_position_list: Optional[List] = None
        position_pace_list: Optional[List[int]] = None
        metad_pace_list: Optional[List[int]] = None
        bw_list: Optional[List[list]] = None
        biasfactor_list: Optional[List[float]] = None
        metad_width_list: Optional[List[list]] = None
        metad_height_list: Optional[List[float]] = None
        
        sim: Dict[int, 'MFI2D'] = field(default_factory=dict)
        active_sim: list = field(default_factory=list)
        goal_reached: bool = False
        
        dynamic_print: bool = False
        printed_lines: list = field(default_factory=list)
        updating_lines: Dict[int, list] = field(default_factory=dict)
                
        live_plot: bool = False
        make_movie: bool = False
        save_media: bool = False
        dpi: int = 100
        
        us_criteria_max_avr_ratio: float = None
        failed_position_checks_before_waring: int = 10_000
        stop_sim: bool = False
        reason_for_termination: str = "(Just started)"
        
        save_comp_cost: bool = False

        def __post_init__(self):
            
            # if ID is specified, set self.ID to ID
            if self.ID != "": self.parent.ID = self.ID
            self.ID = self.parent.ID
            
            # copy the parent sim_files_path to add new trajectory file paths to the sim_files_path["trajectory_xtc_file_path_list"].
            self.sim_files_path = self.parent.sim_files_path.copy()
            
            # set simulation times (total budget, guaranteed simulation time, max simulation time)
            if self.time_budget is not None: self.parent.time_budget = self.time_budget
            if self.parent.time_budget is None: 
                self.parent.time_budget = 100
                print(f"\nTime budget not set. Will use default time budget of {self.parent.time_budget} ns\n")
            self.time_budget = self.parent.time_budget
            if self.guaranteed_sim_time is not None: self.parent.guaranteed_sim_time = self.guaranteed_sim_time
            if self.parent.guaranteed_sim_time is None: 
                self.parent.guaranteed_sim_time = self.parent.time_budget / 20
                print(f"\nGuaranteed simulation time not set. Will use guaranteed simulation time of {self.guaranteed_sim_time} ns \n")
            self.guaranteed_sim_time = self.parent.guaranteed_sim_time    
            if self.max_sim_time is not None: self.parent.max_sim_time = self.max_sim_time
            if self.parent.max_sim_time is None:
                self.parent.max_sim_time = self.parent.time_budget/2
                print(f"\nMax simulation time not set. Will use max simulation time of {self.parent.max_sim_time} ns \n")
            self.max_sim_time = self.parent.max_sim_time
            # Set the number of simulation steps for input files/command line
            self.parent.simulation_steps = int(1.1*(self.parent.max_sim_time / self.parent.time_step * 1000))
            
            # parameters that will be copied and used for all simulations (unless changed later on)
            if self.parent.plX is None or self.parent.plY is None: self.parent.plX, self.parent.plY, self.parent.pl_min, self.parent.pl_max, self.parent.pl_n, self.parent.pl_extra = lib2.get_plumed_grid_2D(self.parent.X, self.parent.Y)
            self.parent.print_info = False
            self.parent.record_maps = False
            self.parent.record_forces_e = False
            if self.main_error_type is not None: self.parent.main_error_type = self.main_error_type
            self.main_error_type = self.parent.main_error_type
            
            # lists for calculating the change in the error
            if self.parent.calculate_error_change == False:
                self.parent.calculate_error_change = True
                self.parent.d_Aofe = np.empty((0,4))
                if self.parent.Z is not None: self.parent.d_AAD = np.empty((0,4))
                if self.parent.weighted_avr_error_change: 
                    self.parent.d_Aofe_w = np.empty((0,4))
                    if self.parent.Z is not None: self.parent.d_AAD_w = np.empty((0,4))
                        
            # parameters for MFI and metadynamics
            if self.bw_list is not None: self.parent.bw = np.mean(self.bw_list, axis=0)
            if self.metad_width_list is not None: self.metad_width_exploration = np.mean(self.metad_width_list, axis=0)
            else: self.metad_width_exploration = np.array([round((self.parent.grid_max[0] - self.parent.grid_min[0]) / 50,5), round((self.parent.grid_max[1] - self.parent.grid_min[1]) / 50,5)])
            if self.metad_height_list is not None: self.metad_height_exploration = np.mean(self.metad_height_list)
            else:
                expected_time_ps = self.time_budget / 10 / self.workers * 1000
                metad_pace_ps = self.parent.metad_pace * self.parent.time_step
                n_hills = expected_time_ps / metad_pace_ps
                area_to_fill = self.parent.FES_cutoff * (self.parent.grid_max[0] - self.parent.grid_min[0]) * (self.parent.grid_max[1] - self.parent.grid_min[1]) / 3
                self.metad_height_exploration = area_to_fill / (6.2831853 * n_hills * self.metad_width_exploration[0] * self.metad_width_exploration[1])
                self.metad_height_exploration = round(max(self.metad_height_exploration , self.parent.FES_cutoff/5),2) # to avoid self.metad_height_exploration to be too small, set it to at elast 1/5 of the FES_cutoff.

            # set path of the campaign
            self.campaign_path = f"{self.parent.simulation_folder_path}PRTRcampaign{self.ID}/"
                 
            if self.restart_PRTR and os.path.exists(self.campaign_path) is False: 
                print("\nSimulation folder does not exist. \nCannot restart PRTR campaign\nContiniue with new campaign\n")
                self.restart_PRTR = False             
            
            # initialise simulation parent folder
            _ = lib2.set_up_folder(self.campaign_path, remove_folder=(not self.restart_PRTR))
            
            # Parent parameters will be used for all simulations, unless changed later on
            self.parent_params = asdict(self.parent)
            
            # Initialise master simulation instance. sim[0] will be used to collect the results from all simulations, but will not run simulations itself.
            self.set_up_new_sim(0)   
            os.chdir(self.campaign_path) # move back to campaign folder
                  
            if self.restart_PRTR:                
                # Find existing simulation folders
                simulation_folder_prefix = f"simulation{self.ID}"
                existing_simulation_folders = [f for f in os.listdir() if simulation_folder_prefix in f]  
                existing_simulation_folders = sorted(existing_simulation_folders, key=lambda x: int(x.split('_')[-1]))
                
                if len(existing_simulation_folders) == 0: 
                    print("\nNo existing simulation folders found. \nCannot restart PRTR campaign\nContiniue with new campaign\n")
                    self.restart_PRTR = False
                    
            if self.restart_PRTR:    
                                
                # loop over existing folders and load the data
                for sim_folder in existing_simulation_folders:
                    
                    # identify the simulation number and move into simulation folder
                    i = int(sim_folder.split("_")[-1])
                    SIM_ID = f"{self.ID}_{i}"                   
                    assert i == len(self.sim), f"Problem reading simulation folders. Please check that the folders have consecutive numbers. ({i = } != {len(self.sim) = })"
                    self.current_path = lib2.set_up_folder(sim_folder, remove_folder=False)
                    
                    # check if position file exists. If not, skip the folder
                    if os.path.exists(f"position{SIM_ID}") is False:
                        print(f"\nCannot find position file with name: \"position{SIM_ID}\" in the folder \"{sim_folder}\"\nFolder will be skipped")
                        os.chdir(self.campaign_path)
                        continue
                    
                    # set up new simulation instance and relevant parameters   
                    self.sim[i] = self.parent.__class__(**copy.deepcopy(self.parent_params))   
                    self.active_sim.append(0)
                    self.sim[0].n_pos_analysed.append(0)
                    self.sim[0].force_terms = np.concatenate((self.sim[0].force_terms, np.zeros((1,6, self.sim[i].nbins_yx[0], self.sim[i].nbins_yx[1])) ), axis=0)  
                    self.sim[i].Avr_Error_list = self.sim[0].Avr_Error_list[-1:, :int(3 + (self.sim[0].Z is not None) + (self.sim[i].dZ_dX is not None and self.sim[0].dZ_dY is not None))]  # this works because 1 + (True) = 2 and 1 + (False) = 1.
                    self.sim[i].base_time = self.sim[i].Avr_Error_list[0, 0]
                    self.sim[i].Aofe_0 = self.sim[i].Avr_Error_list[0, 2]
                    if self.sim[i].Z is not None: self.sim[i].AAD_0 = self.sim[i].Avr_Error_list[0, self.parent.aad_index]   
                    self.sim[i].simulation_folder_path = self.campaign_path + f"simulation{SIM_ID}/"
                    self.sim[i].hills_file = self.sim[i].simulation_folder_path + f"HILLS{SIM_ID}"
                    self.sim[i].position_file = self.sim[i].simulation_folder_path + f"position{SIM_ID}"                  
                   
                    # check if HILLS file exists
                    if os.path.exists(self.sim[i].hills_file): self.sim[i].metad_height = 1

                    # laod data
                    print(f"\nLoading simulation folder: {sim_folder}")
                    self.sim[i].load_data(n_pos_analysed=0)
                    
                    # find bias from hills
                    if self.sim[i].metad_height is not None and self.sim[i].metad_height > 0: 
                        Bias_sim = lib2.find_total_bias_from_hills(self.sim[i].X, self.sim[i].Y, self.sim[i].hills, periodic=self.sim[i].periodic)
                        Force_bias_y_sim, Force_bias_x_sim = np.gradient(Bias_sim, self.sim[i].Y[1,0] - self.sim[i].Y[0,0], self.sim[i].X[0,1] - self.sim[i].X[0,0])
                    else: [Force_bias_x_sim, Force_bias_y_sim, Bias_sim] = [np.zeros(self.sim[i].nbins_yx) for _ in range(3)]
                                        
                    # load force terms
                    os.chdir(self.sim[i].simulation_folder_path)
                    if os.path.exists(f"force_terms{SIM_ID}.pkl"): PD_sim, PD2_sim, Force_x_sim, Force_y_sim, ofv_num_x_sim, ofv_num_y_sim = lib2.load_pkl(f"force_terms{SIM_ID}.pkl")
                    else:
                        print(f"force terms file not found. Calculating force terms ", end="")
                        self.sim[i].const, self.sim[i].bw2 = self.sim[i].position_pace / (self.sim[i].bw[0] * self.sim[i].bw[1]) , [self.sim[i].bw[0] ** 2, self.sim[i].bw[1] ** 2, self.sim[i].bw[0] * self.sim[i].bw[1]]
                        if os.path.exists("external_bias.dat"):
                            print("with external bias. ")
                            plx, ply, plBias, plfbiasx, plfbiasy = lib2.read_plumed_grid_file("external_bias.dat")    
                            Force_static_x, Force_static_y = plfbiasx[self.sim[i].pl_extra[0]:-self.sim[i].pl_extra[1], self.sim[i].pl_extra[2]:-self.sim[i].pl_extra[3]], plfbiasy[self.sim[i].pl_extra[0]:-self.sim[i].pl_extra[1], self.sim[i].pl_extra[2]:-self.sim[i].pl_extra[3]]
                        else: 
                            print("without external bias. ")
                            Force_static_x, Force_static_y = np.zeros(self.sim[i].nbins_yx), np.zeros(self.sim[i].nbins_yx)
                        PD_sim, PD2_sim, Force_x_sim, Force_y_sim, ofv_num_x_sim, ofv_num_y_sim, _, _, _, _ = lib2.MFI_forces(self.sim[i].hills, self.sim[i].position[:,1], self.sim[i].position[:,2], self.sim[i].const, self.sim[i].bw2, self.sim[i].kT, self.sim[i].X, self.sim[i].Y, Force_static_x, Force_static_y, self.sim[i].n_pos_per_window, self.sim[i].Gamma_Factor, periodic=self.sim[i].periodic, PD_limit = self.sim[i].PD_limit, return_FES=False)       

                    # patch forces and find error 
                    self.patch_and_find_error(i, np.array([PD_sim, PD2_sim, Force_x_sim, Force_y_sim, ofv_num_x_sim, ofv_num_y_sim]), np.array([Force_bias_x_sim, Force_bias_y_sim, Bias_sim]) )

                    # if trajectory files is in folder, add the path to the list of trajectory files
                    if self.parent.System != "Langevin": 
                        new_traj_path = lib2.get_trajectory_xtc_file_path(self.current_path, System=self.parent.System)
                        if new_traj_path is not None: self.sim_files_path["trajectory_xtc_file_path_list"] = [new_traj_path] + self.sim_files_path["trajectory_xtc_file_path_list"].copy()    

                    # check if exploration phase is completed
                    if self.sim[0].phase == "exploration":
                        # check if FES + Bias > FES_cutoff and decide if the phase should be changed
                        fes_with_bias = self.sim[0].FES + Bias_sim
                        if (fes_with_bias > self.sim[0].FES_cutoff).all(): self.sim[0].phase = "metad"

                    # go back to simulation folder
                    os.chdir(self.campaign_path)
                    
                # check if goal is reached
                if self.main_error_type == "ST_ERR": self.goal_reached = self.sim[0].Aofe < self.goal
                elif self.main_error_type == "AAD": self.goal_reached = self.sim[0].AAD < self.goal
                else: raise ValueError("main_error_type not recognised")
                if self.goal_reached: 
                    print(f"\nGoal reached in previous simulations. \nNo new simulations will be started.\n")
                    return
                
                # check if time_budget is reached
                if self.time_budget is not None: self.goal_reached = self.sim[0].sim_time > self.time_budget
                if self.goal_reached: 
                    print(f"\nTime budget reached in previous simulations. \nNo new simulations will be started.\n")
                    return True
                
                # Decide between metad and flat phase    
                # find the biasing phase
                if self.sim[0].phase == "metad":
                    
                    if self.sim[0].bootstrap_iter is not None and self.sim[0].bootstrap_iter > 0 and not np.isnan(self.sim[0].Avr_Error_list[-1, self.sim[0].abs_error_index]):
                        if self.sim[0].Avr_Error_list[-1, self.sim[0].abs_error_index] < self.sim[0].FES_cutoff / 20: self.sim[0].phase = "flat"
                    
                    elif self.sim[0].calculate_FES_st_dev:
                        if self.sim[0].Avr_Error_list[-1, self.sim[0].FES_st_dev_index] < self.sim[0].FES_cutoff / 20: self.sim[0].phase = "flat"
                    
                    elif self.sim[0].record_maps:
                        # flat / US phase will be start if var of the last 10 FES maps is small enough. If not, continue metad.    
                        st_dev = np.sum(np.sqrt(np.var(np.array(self.sim[0].Maps_list)[-30:,0], axis=0) * self.sim[0].Maps_list[-1][1])) / np.count_nonzero(self.sim[0].Maps_list[-1][1])
                        if st_dev < self.sim[0].FES_cutoff/20: self.sim[0].phase = "flat"
                        
                    else: 
                        vol_last, min_vol_last_5 = self.sim[0].Avr_Error_list[-1,1], np.min(self.sim[0].Avr_Error_list[-5:,1])
                        max_vol_diff = (vol_last - min_vol_last_5) / min_vol_last_5 * 100
                        if max_vol_diff < 5: self.sim[0].phase = "flat"
                        
                self.print_live("restart_PRTR")
                                  
            # Initialise starting simulation instances. sim[i] will be the instance for simulation i.
            n_existing_simulations = len(self.sim[0].force_terms) - 1
            for i in range(n_existing_simulations + 1, n_existing_simulations + 1 + self.workers): self.set_up_new_sim(i)
                                       
            if self.save_media: 
                self.media_path = self.campaign_path + "media/"
                if os.path.exists(self.media_path): raise ValueError("The media folder already exists. Please remove it or change the save_media parameter")
                lib2.set_up_folder(self.media_path)
                
                self.media_counter = 0
            
            if self.live_plot:
                plt.ion()
                self.fig = plt.figure(figsize=(20,8))

        def print_memory_usage(self):
            memory_bytes = asizeof.asizeof(self) + asizeof.asizeof(self.parent) + sum([asizeof.asizeof(self.sim[i]) for i in range(self.workers+1)])
            memory_mb = memory_bytes / (1024 * 1024)
            return f"Memory: {memory_mb:.2f} MB" 
    
        def print_live(self, text=None, i=None, time=None, new_lines=None):
            
            if hasattr(self, "jupyter_notebook") == False:
                
                # the first time something is printed, the previous lines are not removed
                self.remove_text = False
                self.len_printed_lines = 0
                
                # set up the updating lines
                self.updating_lines[0] = f"please wait"
                for ii in range(1, self.workers+1): self.updating_lines[ii] = f"S{ii} is initialising ..."
                
                #Check if in a Jupyter notebook or not. Will execute different prints for jupyter notebook (.ipynb file) and terminal (.py file)
                try:
                    from IPython import get_ipython
                    if 'IPKernelApp' not in get_ipython().config:  return False  # Not in a notebook
                    else: self.jupyter_notebook = True  # In a Jupyter notebook
                except: self.jupyter_notebook = False  # Not in a notebook
                            
            if self.dynamic_print == False:
                if text is not None: 
                    
                    if text == "": print(""); return
                    
                    if text == "start_simulation":
                        text = f"S{i:2}  START in {self.sim[0].phase} phase | "
                        if self.sim[i].metad_height is not None and self.sim[i].metad_height > 0: text += f"MetaD_H={self.sim[i].metad_height} | MetaD_W={self.sim[i].metad_width} | BF={self.sim[i].biasfactor} | "
                        if self.sim[0].phase != "exploration": text += f"|| Bias_type={self.sim[i].bias_type} | Bias_sf={self.sim[i].Bias_sf} | "
                        if self.sim[i].hp_kappa_y is not None and self.sim[i].hp_kappa_y > 0: text += f"hp_centre= [{self.sim[i].hp_centre_x},{self.sim[i].hp_centre_y}], hp_kappa= [{self.sim[i].hp_kappa_x},{self.sim[i].hp_kappa_y}]"
                        if time is not None: text += f"Startup: {time} sec"
                                      
                    if text == "end_simulation":
                        
                        text = f"S{i:2} END t={self.sim[i].sim_time:5.2f}ns | nPos: {self.sim[i].n_pos_analysed[0]:5}: Aofe: {self.sim[i].Aofe:.2f} | "
                        if self.sim[0].Z is not None: text += f"AAD: {self.sim[i].AAD:.2f} | "
                        # text += f"dErr_dt: {self.sim[i].d_error[-1][1]:.4f} < derr_dt_all: {self.sim[0].d_error[-1][1]:.4f} | "
                        # | derr_dt: {self.sim[i].d_error[-1][1]:.4f} < derr_dt_all: {self.sim[0].d_error[-1][1]:.4f} | Reason: {self.reason_for_termination}")
                        text += f"Aofe_all: {self.sim[0].Aofe:.2f} | "
                        if self.sim[0].Z is not None: text += f"AAD_all: {self.sim[0].AAD:.2f} | "
                        if self.sim[0].bootstrap_iter is not None and self.sim[0].bootstrap_iter > 0 and self.sim[0].ABS_error is not None: text += f"ABS_all: {self.sim[0].ABS_error:.2f} | "
                        text += f"Reason: {self.reason_for_termination}"
                        
                    if text == "error_all": 
                        
                        text = f"ALL | t={self.sim[0].sim_time:5.2f}ns : "
                        if self.goal_reached: 
                            text += f"Aofe_all: {self.sim[0].Aofe:.2f}  |  "
                            if self.sim[0].Z is not None: text += f"AAD_all: {self.sim[0].AAD:.2f} | "
                            if self.sim[0].bootstrap_iter is not None and self.sim[0].bootstrap_iter > 0 and self.sim[0].ABS_error is not None: text += f"ABS_all: {self.sim[0].ABS_error:.2f} | "
                        text += f"  ->->->-> {self.print_memory_usage()}\n"

                    if text == "Loading_data_failed": 
                        text = f"\n ***** Loading data of sim {i} failed (position=None): {new_lines = }, total_pos_analysed = {self.sim[i].n_pos_analysed[0]} *****\n"

                    if text == "goal_reached": 
                        text = f"\n   +++++  Goal reached in sim_time {self.sim[0].sim_time:5.2f} ns  +++++   \n" if self.sim[0].sim_time < self.time_budget else f"\n   ------ The budget of {self.time_budget} ns is reached: Tot sim time: {self.sim[0].sim_time:5.2f} ns ------   \n"

                    if text == "print_progress":
                        
                        if len(self.sim[i].n_pos_analysed) < 2 and i <= self.workers: 
                            text = f"    S{i:2} | t={self.sim[i].sim_time:5.2f}ns | nPos: {self.sim[i].n_pos_analysed[-1]:4}/{self.sim[i].n_pos_analysed[0]:5}: Aofe: {self.sim[i].Aofe:2.2f} | "
                            if self.sim[i].Z is not None: text += f"AAD: {self.sim[i].AAD:2.2f} | "
                            if self.sim[0].Z is not None: text += f"AAD_all: {self.sim[0].AAD:.2f} | "
                            if self.sim[0].bootstrap_iter is not None and self.sim[0].bootstrap_iter > 0 and self.sim[0].ABS_error is not None: text += f"ABS_all: {self.sim[0].ABS_error:.2f} | "
                            # "derr_dt: {self.sim[i].d_error[-1][1]:.4f} < derr_dt_all: {self.sim[0].d_error[-1][1]:.4f}"

                        else: 
                            text = f"    S{i:2} | t={self.sim[i].sim_time:5.2f}ns | nPos: {self.sim[i].n_pos_analysed[-1]:4}/{self.sim[i].n_pos_analysed[0]:5}: Aofe: {self.sim[i].Aofe:2.2f} | "
                            if self.sim[i].Z is not None: text += f"AAD: {self.sim[i].AAD:2.2f} | "
                            text += f"Aofe_all: {self.sim[0].Aofe:.2f}  |  "
                            if self.sim[0].Z is not None: text += f"AAD_all: {self.sim[0].AAD:.2f} | "
                            if self.sim[0].bootstrap_iter is not None and self.sim[0].bootstrap_iter > 0 and self.sim[0].ABS_error is not None: text += f"ABS_all: {self.sim[0].ABS_error:.2f} | "
                            # "derr_dt: {self.sim[i].d_error[-1][1]:.4f} < derr_dt_all: {self.sim[0].d_error[-1][1]:.4f}"
                                          
                    if text == "restart_PRTR":
                        text = f"\nRestarted SRTR campaign with {len(self.sim[0].force_terms)-1} existing simulations and {self.sim[0].sim_time:.4f} ns existing simulation time."
                        text += f"\nAofe = {self.sim[0].Aofe:.3f} | "
                        if self.sim[0].Z is not None: text += f"AAD = {self.sim[0].AAD:.3f} | "
                        if self.sim[0].bootstrap_iter is not None and self.sim[0].bootstrap_iter > 0: text += f"ABS = {self.sim[0].ABS_error:.3f} | "
                        text += f"\nContinuing with phase: {self.sim[0].phase}\n"
                        
                    
                    print(text)
                    return         
            
            if i is not None and text is not None: self.updating_lines[i] = text
            elif text is not None: self.printed_lines.append(text)
                        
            lines = [e for e in [self.updating_lines[k] for k in sorted(self.updating_lines) if k != 0] + [self.updating_lines[0]] if e is not None]
                
            if self.jupyter_notebook:

                content = "\n".join(self.printed_lines + [""] + list(lines))
                if self.remove_text: clear_output(wait=True)
                print(content)
                
            else: 
                if self.remove_text == True:
                    print('\033[F\033[K' * (len(lines) + 1 + self.len_printed_lines), end='')  # remove the previous lines and the printed lines. \033[F is moving up one line, \033[K is clearing the line
                    
                    if len(self.printed_lines) > 0: 
                        for line in (self.printed_lines + [""]): print(f'{line}\033[K', flush=True)  # print the printed lines again
                    else: print(f'\033[K', flush=True) 

                else: 
                    print("")
                    self.printed_lines = []    
                    
                for line in lines: print(f'{line}\033[K', flush=True)  # '\033[K' clears to the end of line in case of shorter updates
                
                self.len_printed_lines = len(self.printed_lines) 
            self.remove_text = True

        def plot_live(self):
                      
            # # while not self.plotter_stop_event.is_set():
            
            # plt.clf()
             
            # ax1 = ax2 = ax3 = ax4 = None   
            # ax12 = ax22 = ax32 = ax42 = None
            # ax = [ax1, ax2, ax3, ax4]
            # ax_2 = [ax12, ax22, ax32, ax42]
            
            # error_all = np.array(self.sim[0].Avr_Error_list)
            
            # active_sim_indexes = [i+1 for i, value in enumerate(self.active_sim) if value == 1]
                
            # for ii in range(len(active_sim_indexes)):
                
            #     i = active_sim_indexes[ii]
                                    

            #     # plot FES1
            #     ax[ii] = plt.subplot(2,4,ii+1)
                
            #     ax[ii].plot(self.sim[0].grid, self.sim[0].Z, label="ref", alpha=0.5, color="grey")
            #     ax[ii].plot(self.sim[0].grid, self.sim[i].FES, label="FES", color="blue")   
            #     ax[ii].set_xlabel("CV"); ax[ii].set_ylabel("FES", color="blue"); ax[ii].set_title(f"FES {i}")
            #     ax[ii].text(0.5, 0.8, str(i), fontsize=50, color="blue", weight="bold", alpha=0.2, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            #     ax[ii].legend()
                
            #     # ax1.tick_params(axis='y', colors='blue')  
            #     # ax_2[ii] = ax[ii].twinx()
            #     # ax_2[ii].fill_between(self.sim[0].grid, np.zeros_like(self.sim[0].grid)+1E-9, self.sim[i].PD , color="red", alpha=0.2)
            #     # ax_2[ii].set_ylabel('Probability density', color="red")  
            #     # ax_2[ii].set_yscale("log")
            #     # ax_2[ii].set_ylim(1E-2, 1E4)
            #     # ax_2[ii].tick_params(axis='y', colors='red')
            #     # ax_2[ii].spines['right'].set_color('red')
            #     # ax_2[ii].spines['left'].set_color('blue')
                

            # # plot FES_TOT
            # ax5 = plt.subplot(2,4,5)
            
            # ax5.plot(self.sim[0].grid, self.sim[0].Z, label="ref", alpha=0.5, color="grey")
            # ax5.plot(self.sim[0].grid, self.sim[0].FES, label="FES", color="blue")   
            # ax5.set_xlabel("CV"); ax5.set_ylabel("FES", color="blue") ; ax5.set_title("FES ALL") 
            # ax5.text(0.5, 0.8, "ALL", fontsize=50, color="blue", weight="bold", alpha=0.2, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            # ax5.legend()   
            # ax5.tick_params(axis='y', colors='blue')  
            
            # # ax52 = ax5.twinx()
            # # ax52.fill_between(self.sim[0].grid, np.zeros_like(self.sim[0].grid)+1E-9, self.sim[0].PD , color="red", alpha=0.2)
            # # ax52.set_ylabel('Probability density', color="red")  
            # # ax52.set_yscale("log")
            # # ax52.set_ylim(1E-2, 1E4)
            # # ax52.tick_params(axis='y', colors='red')
            # # ax52.spines['right'].set_color('red')
            # # ax52.spines['left'].set_color('blue')    
            
                
            # #plot Force
            # plt.subplot(2,4,6)
            
            # plt.plot(self.sim[0].grid, self.sim[0].dy, label="ref", alpha=0.5, color="grey")
            # plt.plot(self.sim[0].grid, self.sim[0].Force, label="Force")   
            # plt.xlabel("CV"); plt.ylabel("Mean Force") ; plt.title("Mean Force ALL") 
            # plt.legend()        
            
            # line_width = 0.5
            
            # # plt AD and OFE
            # ax7 = plt.subplot(2,4,7)

            # if self.sim[0].use_VNORM: pass
            # ax7.plot(self.sim[0].grid, self.sim[0].AD_FES, color="blue",linewidth=line_width)
            # ax7.set_ylabel('Absolute Deviation', color="blue")
            # ax7.set_xlabel('CV')
            # # ax7.set_ylim(0,1)
            # ax7.tick_params(axis='y', colors='blue')  
            
            # ax72 = ax7.twinx()
            # ax72.plot(self.sim[0].grid, self.sim[0].ofe, color="red",linewidth=line_width, alpha=0.7)
            # ax72.set_ylabel('Error of Mean Force', color="red")  
            # # ax72.set_ylim(0,20)
            # ax72.tick_params(axis='y', colors='red')
            # ax72.spines['right'].set_color('red')
            # ax72.spines['left'].set_color('blue')
            
            # # plot AAD_progression and OFE_progression   
            # ax8 = plt.subplot(2,4,8)
            
            
            # if self.sim[0].use_VNORM: ax8.plot(error_all[:,0], error_all[:,5], color="blue",linewidth=line_width)
            # else: ax8.plot(error_all[:,0], error_all[:,4], color="blue",linewidth=line_width)
            # if self.main_error_type == "AAD": ax8.plot([error_all[0,0], error_all[-1,0]], [self.goal, self.goal], color="grey",linewidth=line_width, alpha=0.5)
            
            # ax8.set_ylabel('Absolute Deviation', color="blue")
            # ax8.set_xlabel('CV')
            # # ax8.set_ylim(0,2)
            # ax8.tick_params(axis='y', colors='blue')  

            
            # ax82 = ax8.twinx()
            # if self.sim[0].use_VNORM: ax82.plot(error_all[:,0], error_all[:,3], color="red",linewidth=line_width, alpha=0.7)
            # else: ax82.plot(error_all[:,0], error_all[:,2], color="red",linewidth=line_width, alpha=0.7)
            # if self.main_error_type == "Aofe": ax82.plot([error_all[0,0], error_all[-1,0]], [self.goal, self.goal], color="grey",linewidth=line_width, alpha=0.5)
            # ax82.set_ylabel('Error of Mean Force', color="red")  
            # # ax82.set_ylim(0,30)
            # ax82.tick_params(axis='y', colors='red')
            # ax82.spines['right'].set_color('red')
            # ax82.spines['left'].set_color('blue')
            
            
            
            
            # if self.save_media: plt.savefig(self.media_path + f"figure_{self.media_counter:04}", dpi=self.dpi)


            # plt.tight_layout()
            # # plt.sdraw(); 
            # self.fig.canvas.draw(); 
            # self.fig.canvas.flush_events()
            
            # # self.close_plot_event.wait()
            # # plt.close(fig)
            
            raise ValueError("The plot live function is not yet implemented")
        
        def get_new_simulation_parameters(self, i):
            # set up parameters of starting simulations. Use the input parameters lists from MFI_parallel_RTR if available, otherwise use default values.
            if i <= self.workers: self.sim[i].phase = "exploration"
            else: self.sim[i].phase = self.sim[0].phase
                            
            if self.sim[i].phase == "exploration": 
                
                if self.initial_position_list is not None: self.sim[i].initial_position = self.initial_position_list[i-1] 
                if self.position_pace_list is not None: self.sim[i].position_pace = self.position_pace_list[i-1]
                if self.metad_pace_list is not None: self.sim[i].metad_pace = self.metad_pace_list[i-1]
                self.sim[i].n_pos_per_window = int(self.sim[i].metad_pace / self.sim[i].position_pace)                

                if self.bw_list is not None: self.sim[i].bw = self.bw_list[i-1]
                
                # set biasfactor if available, otherwise set it to the FES_cutoff
                if self.biasfactor_list is not None: self.sim[i].biasfactor = self.biasfactor_list[i-1]
                else: self.sim[i].biasfactor = self.sim[i].FES_cutoff
                
                # set metad_width if available, otherwise set it to 1/50 of the grid size
                if self.metad_width_list is not None: self.sim[i].metad_width = self.metad_width_list[i-1]
                else: 
                    scaing_list = np.linspace(0.8, 1.2, self.workers)
                    self.sim[i].metad_width = self.metad_width_exploration * scaing_list[i-1]
                
                # set metad_height if available, otherwise estimate metad_height so that the metad_hills fill the FES upto the FES_cutoff in appoximately (1/20/workers) of the simulation budget.
                if self.metad_height_list is not None: self.sim[i].metad_height = self.metad_height_list[i-1] 
                else: self.sim[i].metad_height = self.metad_height_exploration
                return
                                                        
            if self.sim[i].phase == "metad": 
                               
                self.sim[i].biasfactor = round(1 + self.sim[i].FES_cutoff / 4,3)
                
                # self.sim[i].metad_width = self.metad_width_exploration #/ 1.5
                _, _, parameter_list = lib2.Gaus_fitting_to_fes_2D(self.sim[0].X, self.sim[0].Y, np.where(self.sim[0].cutoff > 0.5, self.sim[0].FES, self.sim[0].FES_cutoff))
                self.sim[i].metad_width = [round(min([p[1] for p in parameter_list])/2,5), round(min([p[2] for p in parameter_list])/2,5)]
                
                self.sim[i].metad_height = round(min(self.metad_height_exploration/3, self.sim[i].FES_cutoff / 7),2) 
                
                self.sim[i].Bias_sf = 1.05
                self.sim[i].gaus_filter_sigma = 3
                
                return

            # check if us_criteria_max_avr_ratio is defined. If it is, check if the max error is above the (avr error)*us_criteria_max_avr_ratio.
            if self.us_criteria_max_avr_ratio is not None: 
                smooth_error_avr, smooth_error_max = lib2.check_US_criteria(error_map=self.sim[0].ofe, cutoff_map=self.sim[0].cutoff, gaussian_sigma=3)
                if smooth_error_max > self.us_criteria_max_avr_ratio * smooth_error_avr: 
                    
                    # find hp_centre of active sim, if the are in the US phase.
                    active_sim_indexes = [i+1 for i, value in enumerate(self.active_sim) if value == 1]
                    us_sim_indexes = [i for i in active_sim_indexes if self.sim[i].phase == "us"]
                    if len(us_sim_indexes) > 0: prev_hp_centre_x , prev_hp_centre_y = [self.sim[i].hp_centre_x for i in us_sim_indexes], [self.sim[i].hp_centre_y for i in us_sim_indexes]
                    
                    # Find position of max error for hp_centre. if the active simulation ares in the US phase, check if the new hp centre is somewhere else. If centre in the same position, keep flat phase (from sim[0].phase).
                    self.sim[i].hp_centre_x, self.sim[i].hp_centre_y = lib2.find_hp_centre(self.sim[i].X, self.sim[i].Y, self.sim[0].ofe, self.sim[0].cutoff, gaussian_sigma=3, prev_hp_centre_x=prev_hp_centre_x, prev_hp_centre_y=prev_hp_centre_y)
                    
                    if self.hp_centre_x is not None or self.hp_centre_y is not None:
                        self.sim[i].phase = "us"
                        self.sim[i].hp_kappa_x, self.sim[i].hp_kappa_y = 10/self.sim[i].grid_length[0], 10/self.sim[i].grid_length[1]
                        self.sim[i].metad_height = None
                        
                        return
                                    
            if self.sim[i].phase == "flat": 
                
                self.sim[i].biasfactor = round(1 + self.sim[i].FES_cutoff / 20,3)
                
                # self.sim[i].metad_width = [round((self.sim[i].grid_max[0] - self.sim[i].grid_min[0]) / 100,5), round((self.sim[i].grid_max[1] - self.sim[i].grid_min[1]) / 100,5)]
                # self.sim[i].metad_width = self.metad_width_exploration
                _, _, parameter_list = lib2.Gaus_fitting_to_fes_2D(self.sim[0].X, self.sim[0].Y, np.where(self.sim[0].cutoff > 0.5, self.sim[0].FES, self.sim[0].FES_cutoff))
                self.sim[i].metad_width = [round(min([p[1] for p in parameter_list])/10,5), round(min([p[2] for p in parameter_list])/10,5)]
                print(f"Sim {i}: {self.sim[i].metad_width}")
                
                self.sim[i].metad_height = round(min(self.metad_height_exploration/30, self.sim[i].FES_cutoff / 60),2)
                
                if len(self.active_sim) < 3: self.sim[i].bias_type = ""
                elif len(self.active_sim) % 3 == 0: self.sim[i].bias_type = "error"
                elif len(self.active_sim) % 3 == 1: self.sim[i].bias_type = ""
                else: self.sim[i].bias_type = "PD"              
                     
                self.sim[i].hp_kappa_x, self.sim[i].hp_kappa_y = None, None 
                self.sim[i].lw_kappa_x, self.sim[i].lw_kappa_y, self.sim[i].uw_kappa_x, self.sim[i].uw_kappa_y = None, None, None, None
                
                # adjust position/metad pace ?
          
        def set_up_new_sim(self,i):
                        
            if i == 0: 
                sim0_params = copy.deepcopy(self.parent_params)
                sim0_params['record_maps'] = True
                self.parent_params['record_forces_e'] = False  # only save forcs_e for sim[0].
                self.parent_params['bootstrap_iter'] = None
                self.parent_params['calculate_FES_st_dev'] = False  # could also be removed?
                self.sim[i] = self.parent.__class__(**sim0_params)  
                self.sim[0].force_terms = np.zeros((1,6, self.sim[i].nbins_yx[0], self.sim[i].nbins_yx[1])) # force_terms of sim[0] will have the total force terms of each simulation
                self.sim[0].Aofe_0 = 50
                self.sim[0].Avr_Error_list = [0, 0, self.sim[0].Aofe_0]
                if self.sim[0].Z is not None: self.sim[0].AAD_0 = 10
                if self.sim[0].Z is not None: self.sim[0].Avr_Error_list.append(self.sim[0].AAD_0)
                if self.sim[0].dZ_dX is not None and self.sim[0].dZ_dY is not None: self.sim[0].Avr_Error_list.append(self.sim[0].Aofe_0)
                if self.sim[0].bootstrap_iter is not None and self.sim[0].bootstrap_iter > 0: self.sim[0].Avr_Error_list.append(self.sim[0].AAD_0)
                if self.sim[0].calculate_FES_st_dev: self.sim[0].Avr_Error_list.append(self.sim[0].AAD_0)
                self.sim[0].Avr_Error_list = np.array([self.sim[0].Avr_Error_list])
                self.sim[0].phase = "exploration"
                return
            
            # create instances of the simulation class
            self.sim[i] = self.parent.__class__(**copy.deepcopy(self.parent_params))      
                               
            self.sim[0].n_pos_analysed.append(0)
            self.sim[0].force_terms = np.concatenate((self.sim[0].force_terms, np.zeros((1,6, self.sim[i].nbins_yx[0], self.sim[i].nbins_yx[1])) ), axis=0) 
    
            # specify the path for simulation folder, hills_file, and position_file
            self.sim[i].SIM_ID = f"{self.ID}_{i}"
            self.sim[i].simulation_folder_path = self.campaign_path + f"simulation{self.sim[i].SIM_ID}/"
            self.sim[i].hills_file = self.sim[i].simulation_folder_path + f"HILLS{self.sim[i].SIM_ID}"
            self.sim[i].position_file = self.sim[i].simulation_folder_path + f"position{self.sim[i].SIM_ID}"                          

            # Move into new folder for simulation create the simulation folder
            lib2.set_up_folder(self.sim[i].simulation_folder_path, remove_folder=True)

            ### --- Find new parameters for new simulations --- ###
            self.get_new_simulation_parameters(i)
            self.sim[i].const, self.sim[i].bw2 = self.sim[i].position_pace / (self.sim[i].bw[0] * self.sim[i].bw[1]) , [self.sim[i].bw[0] ** 2, self.sim[i].bw[1] ** 2, self.sim[i].bw[0] * self.sim[i].bw[1]]           
            
            # Set the initial variables (base time, error, and forces) for calculating the error and change in the error. For simulation i <= workers these are set. For simulation i > workers, initial values are taken from the patch.                   
            self.sim[i].Avr_Error_list = self.sim[0].Avr_Error_list[-1:, :int(3 + (self.sim[0].Z is not None) + (self.sim[i].dZ_dX is not None and self.sim[0].dZ_dY is not None))]  # this works because 1 + (True) = 2 and 1 + (False) = 1.
            self.sim[i].base_time = self.sim[i].Avr_Error_list[0][0]
            self.sim[i].Aofe_0 = self.sim[i].Avr_Error_list[0][2]
            if self.sim[i].Z is not None: self.sim[i].AAD_0 = self.sim[i].Avr_Error_list[0,self.parent.aad_index]

            if i <= self.workers: return
                
            # Set up the base_forces and static bias
            self.sim[i].base_forces = np.array(self.sim[0].force_terms[0])
            self.sim[i].make_external_bias(FES=self.sim[0].FES, error=self.sim[0].ofe, PD=self.sim[0].PD, bias_type=self.sim[i].bias_type )
            self.sim[i].make_harmonic_bias()
            
        def start_simulation(self, i):
            
            time_start = time.time()
            
            #initialise new simulation variables and decide on MetaD parameters
            if i > self.workers: self.set_up_new_sim(i)
                            
            # run simulation                
            self.sim[i].run_simulation(assign_process=True, file_extension=self.sim[i].SIM_ID, n_cores_per_simulation=self.n_cores_per_simulation, sim_files_path=self.sim_files_path)
            self.active_sim.append(0)
            
            # wait for simulations to have at elast X hills   # this should be changed to wait for positions. Not urgent as initial simulations start in exploration stage
            nPos_0 = 10
            nPos = lib2.wait_for_positions(nPos_0, position_path=self.sim[i].position_file, return_n_pos=True, sleep_between_checks=0.05)
            if nPos > nPos_0*0.9: self.active_sim[-1] = 1  
            else: raise ValueError(f"Simulation {i} did not start correctly. Please check the simulation")         
            if i > self.workers: self.updating_lines[i] = None
            self.print_live("start_simulation", i, time=str(round(time.time() - time_start,2)))
            
            # Add the path of the trajectory file to the list of trajectory files
            if self.parent.System != "Langevin": 
                new_traj_path = lib2.get_trajectory_xtc_file_path(self.sim[i].simulation_folder_path, System=self.parent.System)                
                if new_traj_path is not None: self.sim_files_path["trajectory_xtc_file_path_list"] = [new_traj_path] + self.sim_files_path["trajectory_xtc_file_path_list"].copy()    
            
            # Move back to parent folder
            os.chdir(self.campaign_path)

        def patch_and_find_error(self, i, force_terms_e, bias_terms_e, patch_realtime_forces=True):

            # if patch_realtime_forces is True, patch the force terms of simulation i with the (new) force_terms_e. If False, force_terms_e is the new force terms of the entire simulation.
            if patch_realtime_forces:
                
                # record bias potential and bias force 
                self.sim[i].Force_bias_x += bias_terms_e[0]
                self.sim[i].Force_bias_y += bias_terms_e[1]
                self.sim[i].Bias += bias_terms_e[2]                
                
                # record last force terms (_e) of simulation i
                if self.sim[0].record_forces_e: [self.sim[0].PD_e, self.sim[0].PD2_e, self.sim[0].Force_x_e, self.sim[0].Force_y_e, self.sim[0].ofv_num_x_e, self.sim[0].ofv_num_y_e] = force_terms_e
                #Patch forces of Simulation
                self.sim[0].force_terms[i] = lib2.patch_forces(self.sim[0].force_terms[i], force_terms_e, PD_limit=self.sim[i].PD_limit) 
                # record n_pos_analysed of simulation i
                self.sim[i].n_pos_analysed.append(len(self.sim[i].position))
                self.sim[i].n_pos_analysed[0] = sum(self.sim[i].n_pos_analysed[1:])
            
            else: 
                
                # record bias potential and bias force 
                self.sim[i].Force_bias_x += bias_terms_e[0]
                self.sim[i].Force_bias_y += bias_terms_e[1]
                self.sim[i].Bias += bias_terms_e[2]                
                
                # record force terms of entire simulation. if record_forces_e is True, calculate the difference between the new (whole) force terms and the previous force terms.
                if self.sim[0].record_forces_e: 
                    PD_sim, PD2_sim, Force_x_sim, Force_y_sim, ofv_num_x_sim, ofv_num_y_sim = force_terms_e
                    [self.sim[0].PD_e, self.sim[0].PD2_e, self.sim[0].ofv_num_x_e, self.sim[0].ofv_num_y_e] = [PD_sim - self.sim[i].PD, PD2_sim - self.sim[i].PD2, ofv_num_x_sim - self.sim[i].ofv_num_x, ofv_num_y_sim - self.sim[i].ofv_num_y]
                    self.sim[0].Force_x_e = np.divide(np.multiply(Force_x_sim, PD_sim) - np.multiply(self.sim[i].Force_x, self.sim[i].PD), self.sim[0].PD_e, out=np.zeros_like(self.sim[0].PD_e), where=self.sim[0].PD_e > self.sim[0].PD_limit)
                    self.sim[0].Force_y_e = np.divide(np.multiply(Force_y_sim, PD_sim) - np.multiply(self.sim[i].Force_y, self.sim[i].PD), self.sim[0].PD_e, out=np.zeros_like(self.sim[0].PD_e), where=self.sim[0].PD_e > self.sim[0].PD_limit)
                self.sim[0].force_terms[i] = force_terms_e
                # record n_pos_analysed of simulation i
                self.sim[i].n_pos_analysed.append(len(self.sim[i].position)-self.sim[i].n_pos_analysed[0])
                self.sim[i].n_pos_analysed[0] = len(self.sim[i].position)
                
            # record n_pos_analysed and sim_time 
            self.sim[0].n_pos_analysed[i] = self.sim[i].n_pos_analysed[0]
            self.sim[0].n_pos_analysed[0] = sum(self.sim[0].n_pos_analysed[1:]) # this can be replaced with += len(self.sim[i].position)
            new_sim_time = self.sim[i].position[-1, 0] / 1000 - self.sim[i].sim_time
            self.sim[i].sim_time += new_sim_time           
            self.sim[0].sim_time += new_sim_time
            if len(self.sim[i].n_pos_analysed) > 2 and self.sim[i].sim_time < 0.01: raise ValueError(f"Simulation {i} has unexpected low sim_time = {self.sim[i].sim_time}, n_pos_analysed = {self.sim[i].n_pos_analysed}. Please check if all simulations are running correctly")
            
            # calculate errors of simulation i
            self.sim[i].PD, self.sim[i].PD2, self.sim[i].Force_x, self.sim[i].Force_y, self.sim[i].ofv_num_x, self.sim[i].ofv_num_y = self.sim[0].force_terms[i]        
            self.sim[i].calculate_errors()  
            
            # Patch sim i results with all other results and calculate error of all simulations        
            self.sim[0].force_terms[0] = lib2.patch_forces(self.sim[0].force_terms[1:], PD_limit=self.sim[i].PD_limit)
            self.sim[0].PD, self.sim[0].PD2, self.sim[0].Force_x, self.sim[0].Force_y, self.sim[0].ofv_num_x, self.sim[0].ofv_num_y = self.sim[0].force_terms[0]
            self.sim[0].calculate_errors()
            
            #calculate difference in error for simulation and combined results
            if len(self.sim[i].Avr_Error_list) > 1 or i > self.workers: self.sim[i].calculate_difference_in_error()
            if len(self.sim[0].Avr_Error_list) > 1: self.sim[0].calculate_difference_in_error()

        def check_termination_criteria_PRTR(self, i):
            
            # check if there are enough results to calculate the change in error
            if len(self.sim[0].Avr_Error_list) < 2: return False
            
            # check if goal is reached                
            if self.goal_reached: return True
            
            if self.main_error_type == "ST_ERR": self.goal_reached = self.sim[0].Aofe < self.goal
            elif self.main_error_type == "AAD": self.goal_reached = self.sim[0].AAD < self.goal
            else: raise ValueError("main_error_type not recognised")
            if self.goal_reached: 
                self.reason_for_termination = "GOAL REACHED"
                return True
            
            # check if time_budget is reached                            
            if self.time_budget is not None: self.goal_reached = self.sim[0].sim_time > self.time_budget
            if self.goal_reached: 
                self.reason_for_termination = "TIME BUDGET REACHED"
                return True
            
            if i == 0: return False

            # check if max_sim_time is reached
            if self.sim[i].sim_time > self.sim[i].max_sim_time:
                self.reason_for_termination = "MAX SIM TIME REACHED"
                return True     
            
            # check exploration termination criteria
            if i <= self.workers: # check if FES + Bias > FES_cutoff
                
                if self.sim[0].phase == "metad": 
                    self.reason_for_termination = "Exploration stage completed"
                    return True
                    
                fes_with_bias = np.where(self.sim[0].PD < self.sim[i].PD_cutoff, self.sim[0].FES_cutoff+1 , self.sim[0].FES + self.sim[i].Bias)
                if (fes_with_bias > self.sim[0].FES_cutoff).all(): 
                    self.sim[0].phase = "metad"
                    self.reason_for_termination = "Exploration stage completed"
                    return True
                else: return False
                
            if self.sim[i].phase == "metad" and self.sim[i].sim_time > self.guaranteed_sim_time:
                
                if self.sim[0].phase == "flat":
                    self.reason_for_termination = "Metad phase completed (with other sim)"
                    return True
                
                # if self.sim[0].bootstrap_iter is not None and self.sim[0].bootstrap_iter > 0 and not np.isnan(self.sim[0].Avr_Error_list[-1, self.sim[0].abs_error_index]):
                                        
                #     if self.sim[0].Avr_Error_list[-1, self.sim[0].abs_error_index] < self.sim[0].FES_cutoff / 20: 
                #         self.sim[0].phase = "flat"
                #         self.reason_for_termination = f"Metad phase completed (abs_error={self.sim[0].Avr_Error_list[-1, self.sim[0].abs_error_index]:.4f})"
                #         return True
                
                # elif self.sim[0].calculate_FES_st_dev:
                #     if self.sim[0].Avr_Error_list[-1, self.sim[0].FES_st_dev_index] < self.sim[0].FES_cutoff / 20: 
                #         self.sim[0].phase = "flat"
                #         self.reason_for_termination = f"Metad phase completed (var_fes={self.sim[0].Avr_Error_list[-1, self.sim[0].FES_st_dev_index]:.4f})"
                #         return True
                
                # elif self.sim[0].record_maps:
                #     # flat / US phase will be start if var of the last 10 FES maps is small enough. If not, continue metad.    
                #     st_dev = np.sum(np.sqrt(np.var(np.array(self.sim[0].Maps_list)[-30:,0], axis=0) * self.sim[0].Maps_list[-1][1])) / np.count_nonzero(self.sim[0].Maps_list[-1][1])
                #     if st_dev < self.sim[0].FES_cutoff/20: 
                #         self.sim[0].phase = "flat"
                #         self.reason_for_termination = f"Metad phase completed (var_fes={st_dev:.4f})"
                #         return True    
                    
                # else: 
                #     vol_last, min_vol_last_5 = self.sim[0].Avr_Error_list[-1,1], np.min(self.sim[0].Avr_Error_list[-5:,1])
                #     max_vol_diff = (vol_last - min_vol_last_5) / min_vol_last_5 * 100
                #     if max_vol_diff < 5: 
                #         self.sim[0].phase = "flat"
                #         self.reason_for_termination = f"Metad phase completed (max_vol_diff={max_vol_diff:.2f}%)"
                #         return True      
                
            if (self.sim[i].phase == "metad" or self.sim[i].phase == "flat") and self.sim[i].sim_time < self.guaranteed_sim_time: return False
                
            if self.sim[i].phase == "us" and self.sim[i].sim_time > self.guaranteed_sim_time/10:
                
                if self.sim[i].sim_time > self.guaranteed_sim_time: 
                    self.reason_for_termination = "US phase reached guaranteed sim time"
                    return True
                
                smooth_error_avr, smooth_error_hp_centre = lib2.check_US_criteria(self.sim[0].ofe, self.sim[0].cutoff, gaussian_sigma=3, check_termination=[self.X, self.Y, self.sim[i].hp_centre_x, self.sim[i].hp_centre_y])
                if smooth_error_hp_centre < smooth_error_avr: 
                    self.reason_for_termination = f"US criteria reached ({smooth_error_hp_centre=:.4f} < {smooth_error_avr=:.4f})"
                    return True
                                
                                
            # check change in error for simulation i is negative enough
            [d_err_i, d_err_dt_i, dtot_err_i, dtot_err_d_i] = self.sim[i].d_error[-1]
            [d_err_all, d_err_dt_all, dtot_err_all, dtot_err_d_all] = self.sim[0].d_error[-1]            
                                                    
            # Compare the change in error (d_error_dt) of sim i with that of the the combined results. If d_error_dt of sim i not negative enough, stop sim i. 
            error_scaling = 2  # Use error_scaling to set the difference in error that is allowed. # error_scaling=2: this will require the d_error_dt of sim[i] to be half d_error_dt of sim[0] or more negative than that. Keep in mind that the d_error_dt should be as negative as possible.
            if d_err_dt_i > d_err_dt_all * error_scaling: 
                self.reason_for_termination = "d_error_dt not negative enough"
                return True
                                            
            return False
             
        def re_analyse_data(self, i):
            
            # # Re-analyse all the data again in one go. This is done to increase the numerical accuray and to check if the results are consistent. (minor inconsistencies might arrise due to additional data egnerated since the last check)
            # if not (self.reason_for_termination == "Exploration stage completed" or self.reason_for_termination.startswith("Metad phase completed") or self.goal_reached):
            #     error_old = self.sim[0].AAD if self.sim[0].Z is not None else self.sim[0].Aofe
            #     error_sim_i_old = self.sim[i].AAD if self.sim[i].Z is not None else self.sim[i].Aofe
            #     prev_n_pos_analysed = self.sim[i].n_pos_analysed[0]
            
            # load the entire simulation data and analyse in one go
            self.sim[i].load_data(n_pos_analysed=0)
            PD_sim, PD2_sim, Force_x_sim, Force_y_sim, ofv_num_x_sim, ofv_num_y_sim, Force_bias_x_sim, Force_bias_y_sim, Bias_sim, _ = lib2.MFI_forces(self.sim[i].hills, self.sim[i].position[:,1], self.sim[i].position[:,2], self.sim[i].const, self.sim[i].bw2, self.sim[i].kT, self.sim[i].X, self.sim[i].Y, self.sim[i].Force_static_x, self.sim[i].Force_static_y, self.sim[i].n_pos_per_window, self.sim[i].Gamma_Factor, periodic=self.sim[i].periodic, PD_limit = self.sim[i].PD_limit, return_FES=False )
            
            # patch results with existing results and calculate error
            self.patch_and_find_error(i, np.array([PD_sim, PD2_sim, Force_x_sim, Force_y_sim, ofv_num_x_sim, ofv_num_y_sim]), np.array([Force_bias_x_sim, Force_bias_y_sim, Bias_sim]), patch_realtime_forces=False)
            
            # self.sim[i].n_pos_analysed.append(len(self.sim[i].position)-self.sim[i].n_pos_analysed[0])
            # self.sim[i].n_pos_analysed[0] = len(self.sim[i].position)
            # self.sim[0].n_pos_analysed[i] = self.sim[i].n_pos_analysed[0]
            # self.sim[0].n_pos_analysed[0] = sum(self.sim[0].n_pos_analysed[1:]) 
            # new_sim_time = self.sim[i].position[-1, 0] / 1000 - self.sim[i].sim_time
            # self.sim[i].sim_time += new_sim_time
            # self.sim[0].sim_time += new_sim_time 
            
            # # check if aad for all data patched at once is the same as the one calculated in real time
            # if not (self.reason_for_termination == "Exploration stage completed" or self.reason_for_termination.startswith("Metad phase completed") or self.goal_reached):
            #     error_new = self.sim[0].AAD if self.sim[0].Z is not None else self.sim[0].Aofe
            #     if abs(error_new - error_old)/error_new > 0.1: print(f"\n*** ATTENTION ***: \nSignificant difference in Re-Analysis of ALL SIM: {error_old = :.4f} -> {error_new = :.4f}, has a {abs(error_new - error_old)/error_new:.2%} difference")
            #     error_sim_i_new = self.sim[i].AAD if self.sim[i].Z is not None else self.sim[i].Aofe
            #     if abs(error_sim_i_new - error_sim_i_old)/error_sim_i_new > 0.1: print(f"*** ATTENTION ***: \nSignificant difference in Re-Analysis of SIM {i}: {error_sim_i_old = :.4f} -> {error_sim_i_new = :.4f}, has a {abs(error_sim_i_new - error_sim_i_old)/error_sim_i_new:.2%} difference")
            #     print(f"n_pos_analysed: {prev_n_pos_analysed} -> {len(self.sim[i].position)}")

        def terminate_simulation(self, i):
            
            # kill simulation process. Wait to confirm it shut down
            if self.active_sim[i-1] == 1: os.killpg(os.getpgid(self.sim[i].process.pid), signal.SIGTERM)

            # analyse data. if save_comp_cost == false, analyse whole sim again. If save_comp_cost == True, patch latest data with existing results.            
            if self.save_comp_cost:
                self.sim[i].load_data()                    
                if self.sim[i].position is not None: 
                    PD_e, PD2_e, Force_x_e, Force_y_e, ofv_num_x_e, ofv_num_y_e, Force_bias_x_e, Force_bias_y_e, Bias_e, _ = lib2.MFI_forces(self.sim[i].hills, self.sim[i].position[:,1], self.sim[i].position[:,2], self.sim[i].const, self.sim[i].bw2, self.sim[i].kT, self.sim[i].X, self.sim[i].Y, self.sim[i].Force_static_x-self.sim[i].Force_bias_x, self.sim[i].Force_static_y-self.sim[i].Force_bias_y, self.sim[i].n_pos_per_window, self.sim[i].Gamma_Factor, periodic=self.sim[i].periodic, PD_limit = self.sim[i].PD_limit, return_FES=False)
                    self.patch_and_find_error(i, np.array([PD_e, PD2_e, Force_x_e, Force_y_e, ofv_num_x_e, ofv_num_y_e]), np.array([Force_bias_x_e, Force_bias_y_e, Bias_e]) )
                    _ = self.check_termination_criteria_PRTR(i)
            else: self.re_analyse_data(i)
                        
            # save force terms
            if self.sim[0].save_force_terms: lib2.save_pkl(self.sim[0].force_terms[i], self.sim[i].simulation_folder_path  + f"force_terms{self.sim[i].SIM_ID}.pkl")
            
            # check if simulation was terminated successfully    
            if self.sim[i].process.poll() is None: self.print_live(""); self.print_live(""); self.print_live("\n\n******* Failed to terminate the process *******"); self.print_live(""); self.print_live(""); self.remove_text = False
            else: self.active_sim[i-1] = 0
            if len(self.sim[i].Avr_Error_list) > 1: self.print_live("end_simulation", i)
            
            # delte simulation instance
            # if self.save_comp_cost: self.sim[i] = None
                            
        def terminate_all_processes(self, Reinitialise=False):
            
            if Reinitialise: sim_indexes_to_terminate = [i+1 for i, value in enumerate(self.active_sim) if value == 1]
            else: sim_indexes_to_terminate = [i+1 for i, value in enumerate(self.active_sim)]
                
            for i in sim_indexes_to_terminate:
                if self.sim[i] is not None and hasattr(self.sim[i], "process") and self.sim[i].process.poll() is None: os.kill(os.getpgid(self.sim[i].process.pid), signal.SIGTERM)            
            
            if Reinitialise: 
                for i_terminate in sim_indexes_to_terminate: self.terminate_simulation(i_terminate)
                _ = self.check_termination_criteria_PRTR(0)
                if not self.goal_reached:              
                    for i_start in range(len(self.active_sim)+1, len(self.active_sim)+1+self.workers): self.start_simulation(i_start)
            else: time.sleep(0.1)    
            
            if self.goal_reached: self.sim[0].save_data(save_data_path=self.campaign_path, use_parent_instance=self)
            
            for i in sim_indexes_to_terminate:
                if self.sim[i] is not None and hasattr(self.sim[i], "process") and self.sim[i].process.poll() is None: print(f"\n\nFailed to terminate the process {i}")             
                          
        def run(self):
            
            # start initial simulations
            for i in range(len(self.sim)-self.workers, len(self.sim)): self.start_simulation(i)
                        
            while self.goal_reached == False:
                
                # Loop over active simulations. If enough data -> Analyse and patch results -> Check if termination criteria / Goal True -> Stop (and ReInit)     
                active_sim_indexes = [i+1 for i, value in enumerate(self.active_sim) if value == 1]
                no_new_lines_counter = 0
                
                for i in active_sim_indexes:
                    
                    # Count new lines in position file. If not enough new lines, skip analysis and move to next simulation
                    new_lines = lib2.count_lines(self.sim[i].position_file) - self.sim[i].n_pos_analysed[0] 
                    if new_lines < self.n_pos_before_analysis: no_new_lines_counter += 1; continue    
                    else: no_new_lines_counter = 0
                    
                    # Load data
                    self.sim[i].load_data()
                    if self.sim[i].position is None: self.print_live(f"Loading_data_failed", i, new_lines=new_lines); continue
                    
                    # Analyse new data                     
                    PD_e, PD2_e, Force_x_e, Force_y_e, ofv_num_x_e, ofv_num_y_e, Force_bias_x_e, Force_bias_y_e, Bias_e, _ = lib2.MFI_forces(self.sim[i].hills, self.sim[i].position[:,1], self.sim[i].position[:,2], self.sim[i].const, self.sim[i].bw2, self.sim[i].kT, self.sim[i].X, self.sim[i].Y, self.sim[i].Force_static_x-self.sim[i].Force_bias_x, self.sim[i].Force_static_y-self.sim[i].Force_bias_y, self.sim[i].n_pos_per_window, self.sim[i].Gamma_Factor, periodic=self.sim[i].periodic, PD_limit = self.sim[i].PD_limit, return_FES=False)
                    self.patch_and_find_error(i, np.array([PD_e, PD2_e, Force_x_e, Force_y_e, ofv_num_x_e, ofv_num_y_e]), np.array([Force_bias_x_e, Force_bias_y_e, Bias_e]) )
                    
                    #Check Termination criteria. If met, stop simulation and start new one if time budget / goal not reached. If not met, print progress and continue 
                    stop_sim = self.check_termination_criteria_PRTR(i)
                    
                    if self.goal_reached: # terminate all simulations
                        for ii, value in enumerate(self.active_sim): self.terminate_simulation(ii+1) if value == 1 else None
                        break 

                    elif stop_sim and (self.reason_for_termination == "Exploration stage completed" or self.reason_for_termination.startswith("Metad phase completed")): 
                        self.terminate_all_processes(Reinitialise=True)
                        if self.goal_reached: break                   

                    elif stop_sim: # terminate simulation and start new
                        self.terminate_simulation(i)
                        if sum(self.active_sim) < self.workers: self.start_simulation(len(self.active_sim)+1)   
                            
                    elif (new_lines > self.n_pos_before_analysis): self.print_live("print_progress", i)
                                                                            
                self.print_live("error_all") if no_new_lines_counter < self.workers else None
                if no_new_lines_counter >= self.workers * self.failed_position_checks_before_waring: 
                    if no_new_lines_counter > self.workers * self.failed_position_checks_before_waring*10: raise ValueError(f"No new lines in position files after {self.failed_position_checks_before_waring*10} cheks for workers. Make sure simulations are running correctly and consider increasing \"failed_position_checks_before_waring\" variable.") 
                    else: print(f"No new lines in position files after {self.failed_position_checks_before_waring} cheks for workers. Make sure simulations are running correctly and consider increasing \"failed_position_checks_before_waring\" variable.")
                
                if self.live_plot: self.plot_live()    

            # after goal_reached, terminate all processes:
            if self.goal_reached: self.print_live("goal_reached")
            self.terminate_all_processes()
                        
            #make video
            

def load_instance(save_instance_path=None):

    # set up the path where the instance is saved
    if save_instance_path is None: save_instance_path = os.getcwd() + "/MFI_instance.pkl"
    if not os.path.exists(save_instance_path): raise ValueError(f"The instance file is not found: {save_instance_path}\n\nPlease provide the correct path to the instance file.")

    # laod the instance
    with open(save_instance_path, 'rb') as file: 
        return dill.load(file)           

                
            
                
                
                
                