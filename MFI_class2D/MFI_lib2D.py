import os
import sys
import subprocess
import shutil
import glob

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pickle

import random
import time
from datetime import datetime, timedelta 
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit
from dataclasses import dataclass, field
from typing import Union, List, Optional

####  ---- Run Simulation with 2 CV (2D)  ----  ####

@dataclass
class Run_Simulation:
    """Class for running simulations."""
    
    for init_parameters in [0]:
        
        # Variables for the plumed grid
        pl_X: np.ndarray = field(default = None)
        pl_Y: np.ndarray = field(default = None)
        pl_min: list = field(default_factory=lambda: [-3.0, -3.0])
        pl_max: list = field(default_factory=lambda: [3.0, 3.0])
        pl_n: list = field(default_factory=lambda: [200, 200]) #what grid does plumed create? should this be 200 to be compatible with np.linspace(-3,3,201)?
        periodic: list = field(default_factory=lambda: [False, False])

        # Variables for the simulation
        System: str = None
        cv_name: list = field(default_factory=lambda: [None, None])
        n_steps: int = 1_000_000
        time_step: float = 0.005
        temperature: float = 1
        find_sim_init_structure: bool = False
        initial_position: list = field(default_factory=lambda: [None, None])
        initial_position_accuracy: list = field(default_factory=lambda: [None, None])              
        make_tpr_input_file: bool = False
        n_cores_per_simulation: int = None
        start_sim: bool = True
        friction: float = 1

        # Variables for the metadynamics
        plumed_dat_text: str = None  # This is the text that will be written to the plumed.dat file. This text should include everything upto the lines specifying the Bias (metad adn/or static bias)
        metad_width: list = field(default_factory=lambda: [0.1, 0.1])
        metad_height: float = 5
        biasfactor: float = 10.0
        metad_pace: int = 500
        position_pace: int = 50
        n_pos_per_window: int = 10

        # Variables for static bias
        hp_centre_x: float = None; hp_centre_y: float = None; hp_kappa_x: float = None; hp_kappa_y: float = None # harmonic potential
        lw_centre_x: float = None; lw_centre_y: float = None; lw_kappa_x: float = None; lw_kappa_y: float = None # lower wall
        uw_centre_x: float = None; uw_centre_y: float = None; uw_kappa_x: float = None; uw_kappa_y: float = None # upper wall
        external_bias_file: str = None
        
        # Variables specifying file paths and file extensions
        sim_files_path: dict = field(default_factory=lambda: {"trajectory_xtc_file_path_list":["traj_0.xtc"], "structure_gro_file_path":"structure.gro", "mdp_file_path":"gromppvac.mdp", "top_file_path":"topology.top", "tpr_file_path":"input.tpr", "pdb_file_path":"reference.pdb"})
        file_extension: str = ""    
        save_simulation_data_file: str = None
        
        # Variables for printing information about the simualtion
        print_info: bool = True
        info: str = None
        
        # # This is the text that will be written to the plumed.dat file. This text should include everything upto the lines containing the UPDADE_IF and DUMPATOMS commands. If None, plumed_dat_text will be used.
        find_structure_text: str = None # This is the text that will be written to the plumed_traj.dat file. If the text is the same as plumed_dat_text (defined above), this field can be left to None, and plumed_dat_text will be used.
    
    def __post_init__(self):

        # Define the plumed grid
        if (self.pl_X is None or self.pl_Y is None) and (self.pl_min is None or self.pl_max is None or self.pl_n is None): print("\n ***** Please either plumed grid or plumed min, max, and nbins. ***** \n")
        elif self.pl_X is None or self.pl_Y is None: 
            self.pl_x = np.linspace(self.pl_min[0], self.pl_max[0], self.pl_n[0])
            self.pl_y = np.linspace(self.pl_min[1], self.pl_max[1], self.pl_n[1])
            self.pl_X, self.pl_Y = np.meshgrid(self.pl_x, self.pl_y)
        else: self.pl_min, self.pl_max, self.pl_n = [np.min(self.pl_X), np.min(self.pl_Y)], [np.max(self.pl_X), np.max(self.pl_Y)], [self.pl_X.shape[1], self.pl_X.shape[0]]
        self.pl_l = [self.pl_max[0] - self.pl_min[0], self.pl_max[1] - self.pl_min[1]]
        
        # Set the metadynamics parameters and initial position    
        if self.position_pace is None: self.position_pace = int(round(self.metad_pace / self.n_pos_per_window))
        assert round(self.metad_pace / self.position_pace,4) == round(self.n_pos_per_window,4), "The metad_pace, position_pace, and n_pos_per_window do not match."
        # self.periodic_boundaries = "NO" if (self.periodic[0] is False) and (self.periodic[1] is False) else str(self.pl_min) + "," + str(self.pl_max)
        pl_l = self.pl_l if self.periodic[0] is False and self.periodic[1] is False else [0.0, 0.0]
        if self.initial_position[0] is None: self.initial_position[0] = get_random_move(self.pl_min[0]+pl_l[0]/10, self.pl_max[0]-pl_l[0]/10)
        if self.initial_position[1] is None: self.initial_position[1] = get_random_move(self.pl_min[1]+pl_l[1]/10, self.pl_max[1]-pl_l[1]/10)

        # If specified, find the initial structure for the simulation
        if self.find_sim_init_structure: self.find_init_structure(new_structure_gro_file_path="structure_new.gro")

        # If specified, create the tpr file for the simulation
        if self.make_tpr_input_file: self.make_tpr_input(new_input_tpr_file_path="input_new.tpr")
        # check if the input.tpr file exists
        if self.System in ["gmx", "gromacs", "GMX", "GROMACS"] and not os.path.exists(self.sim_files_path["tpr_file_path"]): raise FileNotFoundError(f"tpr file not found: {self.sim_files_path['tpr_file_path']}")
        
        # if grid is periodic, set the min and max to "-pi" and "pi"        
        pl_min, pl_max = self.pl_min, self.pl_max
        if self.periodic[0]: pl_min[0], pl_max[0] = "-pi", "pi"
        if self.periodic[1]: pl_min[1], pl_max[1] = "-pi", "pi"
        
        # make the input command for the simulation. Set the number of cores per simulation if specified.
        if self.System in ["gmx", "gromacs", "GMX", "GROMACS"]:
            if self.n_cores_per_simulation is None: self.terminal_input = f"gmx mdrun -s {self.sim_files_path['tpr_file_path']} -nsteps {int(self.n_steps)} -plumed plumed.dat >/dev/null 2>&1"    
            elif self.n_cores_per_simulation > 0: self.terminal_input = f"gmx mdrun -s {self.sim_files_path['tpr_file_path']} -nsteps {int(self.n_steps)} -plumed plumed.dat -ntomp {self.n_cores_per_simulation} >/dev/null 2>&1"    
            else: raise ValueError(f"Number of cores per simulation must be a positive integer. Current n_cores_per_simulation = {self.n_cores_per_simulation}.")
        if self.System in ["Langevin", "Langevin2D"]:
            self.terminal_input = "plumed pesmd < input >/dev/null 2>&1" 
            # Write the input file or command
            input_file_text = f"nstep {self.n_steps}\nipos {self.initial_position[0]},{self.initial_position[1]}\ntemperature {self.temperature}\ntstep {self.time_step}\nfriction {self.friction}\ndimension 2"
            input_file_text += f"\nperiodic on min {self.pl_min} max {self.pl_max}" if (self.periodic[0]) or (self.periodic[1]) else "\nperiodic false"                   
            with open("input" ,"w") as f: print( input_file_text,file=f)        
              
        # Write the plumed file
        cv_str = f"{self.cv_name[0]},{self.cv_name[1]}"
        plumed_file_text = str(self.plumed_dat_text)
        # Metadynamics bias. To activate, the height needs to be a positive number
        if self.metad_height is not None and self.metad_height > 0: plumed_file_text += f"METAD ARG={cv_str} PACE={self.metad_pace} SIGMA={self.metad_width[0]},{self.metad_width[1]} HEIGHT={self.metad_height} BIASFACTOR={self.biasfactor} GRID_MIN={pl_min[0]},{pl_min[1]} GRID_MAX={pl_max[0]},{pl_max[1]} GRID_BIN={self.pl_n[0]-1},{self.pl_n[1]-1} TEMP={self.temperature*120} FILE=HILLS{self.file_extension}\n"
        # Harmonic potential bias. To activate, the force constant (kappa) needs to be a positive number
        if self.hp_kappa_x is not None or self.hp_kappa_y is not None: plumed_file_text += f"RESTRAINT ARG={cv_str} KAPPA={self.hp_kappa_x},{self.hp_kappa_y} AT={self.hp_centre_x},{self.hp_centre_y} LABEL=restraint\n"
        # Lower wall bias. To activate, the force constant (kappa) needs to be a positive number
        if self.lw_kappa_x is not None or self.lw_kappa_y is not None: plumed_file_text += f"LOWER_WALLS ARG={cv_str} AT={self.lw_centre_x},{self.lw_centre_y} KAPPA={self.lw_kappa_x},{self.lw_kappa_y} LABEL=lowerwall\n"
        # Upper wall bias. To activate, the force constant (kappa) needs to be a positive number
        if self.uw_kappa_x is not None or self.uw_kappa_y is not None: plumed_file_text += f"UPPER_WALLS ARG={cv_str} AT={self.uw_centre_x},{self.uw_centre_y} KAPPA={self.uw_kappa_x},{self.uw_kappa_y} LABEL=upperwall\n"
        # External bias. To activate, the file name needs to be given
        if (self.external_bias_file is not None) and (self.external_bias_file != ""): plumed_file_text += f"EXTERNAL ARG={cv_str} FILE={self.external_bias_file} LABEL=external\n"
        # Print position of system. 
        plumed_file_text += f"PRINT FILE=position{self.file_extension} ARG={cv_str} STRIDE={self.position_pace}"
        # Write the plumed file
        with open("plumed.dat" ,"w") as f: print(plumed_file_text ,file=f)

        # Start the simulation
        if self.start_sim == True: self.start_simulation()

    def find_init_structure(self, new_structure_gro_file_path):#, only_first_structure=None):
        
        # check if the trajectory file and pdb file exists
        if any([os.path.exists(traj_path) for traj_path in self.sim_files_path["trajectory_xtc_file_path_list"]]) == False: print(f"\n******* No Trajectory files were found: {self.sim_files_path['trajectory_xtc_file_path_list']} ******* \n\n The next simulation will start from the input.tpr file (if available).")
        # if not os.path.exists(self.sim_files_path["trajectory_xtc_file_path_list"]): raise FileNotFoundError(f"Trajectory file not found: {self.sim_files_path['trajectory_xtc_file_path_list']}")
        if not os.path.exists(self.sim_files_path["pdb_file_path"]): raise FileNotFoundError(f"pdb file not found: {self.sim_files_path['pdb_file_path']}")
        
        if self.find_structure_text is None: find_structure_text = str(self.plumed_dat_text)
        else: find_structure_text = str(self.find_structure_text)
        
        # set the distance for the initial position where the structure is searched. If not found, distance is doubled until a structure is found or 10 attempts are made. 
        d_pos_x = self.initial_position_accuracy[0] if self.initial_position_accuracy[0] is not None else (self.pl_max[0] - self.pl_min[0])/100
        d_pos_y = self.initial_position_accuracy[1] if self.initial_position_accuracy[1] is not None else (self.pl_max[1] - self.pl_min[1])/100  
                
        total_structures, attempts = 0, 0
        while total_structures < 1 and attempts < 11:
            
            # iterate through existing trajectory files
            for traj_path in self.sim_files_path["trajectory_xtc_file_path_list"]:
            
                # get the initial position
                if self.initial_position[0] is None or self.initial_position[1] is None: np.random.seed()
                i_pos_x = self.initial_position[0] if self.initial_position[0] is not None else np.random.uniform(self.pl_min[0], self.pl_max[0])
                i_pos_y = self.initial_position[1] if self.initial_position[1] is not None else np.random.uniform(self.pl_min[1], self.pl_max[1])

                # write plumed file to get the initial structure   
                find_structure_text += f"UPDATE_IF ARG={self.cv_name[0]},{self.cv_name[1]} MORE_THAN={i_pos_x-d_pos_x},{i_pos_y-d_pos_y} LESS_THAN={i_pos_x+d_pos_x},{i_pos_y+d_pos_y}\n"
                find_structure_text += f"DUMPATOMS FILE={new_structure_gro_file_path} ATOMS=1-22\n"        
                find_structure_text += f"UPDATE_IF ARG={self.cv_name[0]},{self.cv_name[1]} END\n"
                with open("plumed_traj.dat", "w") as f: print(find_structure_text, file=f)
                
                # run plumed driver to get the initial structure
                os.system(f"plumed driver --plumed plumed_traj.dat --mf_xtc {traj_path} > /dev/null")

                # count the structures in the new structure file
                with open(new_structure_gro_file_path, 'r') as file: total_structures = sum(1 for line in file if line.startswith("Made with PLUMED"))
                
                # if at least one structure is found, break the loop and continue
                if total_structures > 0: break
            
            # if there is a file with "bck.*.structure_new.gro" remove it
            if os.path.exists(f"bck.*.{new_structure_gro_file_path}"): os.system(f"rm bck.*.{new_structure_gro_file_path}")

            # if no structure is found, double the distance and try again
            attempts += 1
            d_pos_x *= 2
            d_pos_y *= 2
        if total_structures < 1: raise FileNotFoundError(f"Initial structure not found in {attempts} attempts.")

        # find the lines per structure in the structure file
        total_n_lines = int(os.popen(f"wc -l {new_structure_gro_file_path}").read().split()[0])
        lines_per_structure = int(total_n_lines / total_structures)

        # if total_structures > 1, remove a random number of structures from the start of the file
        if total_structures > 1: 
            np.random.seed()
            del_structure_lines = int(lines_per_structure * (total_structures-np.random.randint(1, total_structures)) )
            os.system(f"sed -i -e '1,{del_structure_lines}d' {new_structure_gro_file_path}")
            
        # # if total_structures > 1, remove all but the first structure from the file
        if total_structures > 1: os.system(f"sed -i -e '1,{lines_per_structure}!d' {new_structure_gro_file_path}")
        
        # update the path to the new structure file     
        self.sim_files_path["structure_gro_file_path"] = os.getcwd() + "/" + new_structure_gro_file_path
        
    def make_tpr_input(self, new_input_tpr_file_path):
        
        if not os.path.exists(self.sim_files_path["structure_gro_file_path"]): raise FileNotFoundError(f"Stucture(.gro) file not found: {self.sim_files_path['structure_gro_file_path']}")
        if not os.path.exists(self.sim_files_path["mdp_file_path"]): raise FileNotFoundError(f"mdp file not found: {self.sim_files_path['mdp_file_path']}")
        if not os.path.exists(self.sim_files_path["top_file_path"]): raise FileNotFoundError(f"Topology(.top) file not found: {self.sim_files_path['top_file_path']}")
        
        # print("Make TPR file command: ", "gmx", "grompp", "-f", self.sim_files_path["mdp_file_path"], "-c", self.sim_files_path["structure_gro_file_path"], "-p", self.sim_files_path["top_file_path"], "-o", new_input_tpr_file_path)
        
        #Prepare new input file<- input structure.gro, topolvac.top, gromppvac.mdp -> input(n).tpr
        find_input_structure = subprocess.Popen(["gmx", "grompp", "-f", self.sim_files_path["mdp_file_path"], "-c", self.sim_files_path["structure_gro_file_path"], "-p", self.sim_files_path["top_file_path"], "-o", new_input_tpr_file_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)#, text=True)
        find_input_structure.wait()
        
        output_find_input_structure, errors_find_input_structure = find_input_structure.communicate()
        if "Error" in errors_find_input_structure.decode('utf-8'):
            print("\n*****There is an error message when attempting to create a tpr input file:*****\n")
            # change the type of errors_find_input_structure to string
            errors_find_input_structure = errors_find_input_structure.decode('utf-8')
            # remove the gromacs header from the error message to make it more readable
            remove_text = "                :-) GROMACS - gmx grompp, 2021-plumed-2.7.5 (-:\n\n                            GROMACS is written by:\n     Andrey Alekseenko              Emile Apol              Rossen Apostolov     \n         Paul Bauer           Herman J.C. Berendsen           Par Bjelkmar       \n       Christian Blau           Viacheslav Bolnykh             Kevin Boyd        \n     Aldert van Buuren           Rudi van Drunen             Anton Feenstra      \n    Gilles Gouaillardet             Alan Gray               Gerrit Groenhof      \n       Anca Hamuraru            Vincent Hindriksen          M. Eric Irrgang      \n      Aleksei Iupinov           Christoph Junghans             Joe Jordan        \n    Dimitrios Karkoulis            Peter Kasson                Jiri Kraus        \n      Carsten Kutzner              Per Larsson              Justin A. Lemkul     \n       Viveca Lindahl            Magnus Lundborg             Erik Marklund       \n        Pascal Merz             Pieter Meulenhoff            Teemu Murtola       \n        Szilard Pall               Sander Pronk              Roland Schulz       \n       Michael Shirts            Alexey Shvetsov             Alfons Sijbers      \n       Peter Tieleman              Jon Vincent              Teemu Virolainen     \n     Christian Wennberg            Maarten Wolf              Artem Zhmurov       \n                           and the project leaders:\n        Mark Abraham, Berk Hess, Erik Lindahl, and David van der Spoel\n\nCopyright (c) 1991-2000, University of Groningen, The Netherlands.\nCopyright (c) 2001-2019, The GROMACS development team at\nUppsala University, Stockholm University and\nthe Royal Institute of Technology, Sweden.\ncheck out http://www.gromacs.org for more information.\n\nGROMACS is free software; you can redistribute it and/or modify it\nunder the terms of the GNU Lesser General Public License\nas published by the Free Software Foundation; either version 2.1\nof the License, or (at your option) any later version.\n\n"
            remove_line_start = ["GROMACS:      gmx", "Executable:", "Data prefix:", "Program:     gmx", "Source file:", "Function:", "For more information", "website at", "------------------"]
            if remove_text in errors_find_input_structure: errors_find_input_structure = errors_find_input_structure.replace(remove_text, "")
            errors_find_input_structure = errors_find_input_structure.split("\n")
            for remove_line in remove_line_start: errors_find_input_structure = [line for line in errors_find_input_structure if not line.startswith(remove_line)] 
            # remove the empty lines from the error message
            errors_find_input_structure = [line for line in errors_find_input_structure if line != ""]
            # change the list of strings to a single string, separated by new lines
            errors_find_input_structure = "\n".join(errors_find_input_structure)
            print(errors_find_input_structure, "\n   ******************************\n")
        
        if not os.path.exists(new_input_tpr_file_path): raise FileNotFoundError(f"tpr file not found: {new_input_tpr_file_path = }")
        else: self.sim_files_path["tpr_file_path"] = new_input_tpr_file_path
                
    def start_simulation(self):

        process = subprocess.Popen(self.terminal_input, shell=True, preexec_fn=os.setsid)
        if self.print_info: print("Simulation started with Terminal input:", self.terminal_input) 

        # Write simulation info text. If self.print_info, print the information about the simulation
        if self.print_info: start = time.time()
        if self.print_info or self.save_simulation_data_file is not None:
            info = f"\nRunning simulation with System = {self.System}: n_steps={self.n_steps:,}, ipos={self.initial_position[0]},{self.initial_position[1]}, Pos_t={self.position_pace}, T={self.temperature}, t_Tot={self.n_steps*self.time_step/1000:,.2f}ns"
            if self.metad_height is not None: info += f"\nsigma=[{self.metad_width[0]},{self.metad_width[1]}], h={self.metad_height}, bf={self.biasfactor}, Gaus_t={self.metad_pace}"
            if self.hp_kappa_x is not None or self.hp_kappa_y is not None: info += f"\nHarmonic potential: centre={self.hp_centre_x},{self.hp_centre_y}, kappa={self.hp_kappa_x},{self.hp_kappa_y}"
            if self.lw_kappa_x is not None or self.lw_kappa_y is not None: info += f"\nLower wall: centre={self.lw_centre_x},{self.lw_centre_y}, kappa={self.lw_kappa_x},{self.lw_kappa_y}"
            if self.uw_kappa_x is not None or self.uw_kappa_y is not None: info += f"\nUpper wall: centre={self.uw_centre_x},{self.uw_centre_y}, kappa={self.uw_kappa_x},{self.uw_kappa_y}"
            if self.external_bias_file is not None: info += f"\nStatic bias used: {self.external_bias_file}"
            if self.print_info: print(info, "\n") 
        if self.print_info:
            tot_pos = self.n_steps / self.position_pace               
            while process.poll() is None:
                time.sleep(1)
                n_lines = count_lines("position"+self.file_extension)
                live_print_progress(start, n_lines, tot_pos, bar_length=50, variable_name='Simulated time', variable=n_lines*self.time_step*self.position_pace/1_000)
        else: process.wait()
        if self.print_info: print(f"\n{self.System} simulation finished in {format_seconds(time.time()-start)}.")

        # if self.save_simulation_data_file is given save the data
        if self.save_simulation_data_file is not None:
            if info is None: info = f"\nRunning {self.System} simulation: n_steps={self.n_steps:,}, ipos={self.initial_position[0]},{self.initial_position[1]}, Pos_t={self.position_pace}, T={self.temperature}, t_Tot={self.n_steps*self.time_step/1000:,.2f}ns"
            [HILLS, pos] = read_data(ext=self.file_extension)
            external_bias = read_plumed_grid_file(self.external_bias_file) if self.external_bias_file != "" else [None]
            save_pkl([HILLS, pos, external_bias, info], self.save_simulation_data_file)        
            
    def start_sim_return_process(self, print_info=False):
        # Start the simulation and return the process so that it can be interupted later.
        if self.print_info: print("Simulation started with Terminal input:", self.terminal_input) 
        process = subprocess.Popen(self.terminal_input, shell=True, preexec_fn=os.setsid)
        if print_info: print("Simulation started with pid:", process.pid, os.getpgid(process.pid))
        return process
    
def get_trajectory_xtc_file_path(simulation_path=None, System="gromacs"):
    
    if simulation_path is not None: os.chdir(simulation_path)
    current_path = os.getcwd() + "/"

    if System in ["gmx", "gromacs", "GMX", "GROMACS"]:
        traj_files = sorted(glob.glob("*traj*.xtc"), key=os.path.getctime)
        if len(traj_files) == 0: 
            print(f"\n******* No trajectory files found in {simulation_path} ******* ")
            return None
        return current_path + traj_files[-1]
    
    if System in ["Langevin", "Langevin2D"]:
        print(f"\n******* No trajectory files are generated in Langevin simulations ******* ")
        return None
         
def wait_for_HILLS(new_hills, hills_analysed=0, hills_path="HILLS", return_nhills=False, lines_with_coments="default", periodic=[False,False], sleep_between_checks=0.1):
    wait = True
    counter = 1
    if lines_with_coments == "default": lines_with_coments = 3 if periodic == [False,False] else 7
    while wait == True:
        
        result = subprocess.run(['wc', '-l', hills_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # The output will be in the format 'number filename', so split on whitespace and take the first component
        if result.returncode == 0: 
            line_count = int(result.stdout.split()[0])   # Check if the command was successful
            if (line_count - hills_analysed - lines_with_coments) > new_hills: wait = False 
            else: time.sleep(sleep_between_checks)        
        else: 
            counter += 1
            if counter > 3: raise Exception(f"Error counting lines in {hills_path}: {result.stderr} (after 3 attempts)")
            else: time.sleep(sleep_between_checks*5)

    if return_nhills: return line_count - hills_analysed - lines_with_coments
    else: return

def wait_for_positions(new_positions, n_pos_analysed=0, position_path="position", return_n_pos=False, lines_with_coments="default", periodic=[False,False], sleep_between_checks=0.05):
    wait = True
    counter = 1
    if lines_with_coments == "default": lines_with_coments = 1 if periodic == [False,False] else 5
    while wait == True:
        
        result = subprocess.run(['wc', '-l', position_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Check if the command was successful. The output will be in the format 'number filename', so split on whitespace and take the first component
        if result.returncode == 0: 
            line_count = int(result.stdout.split()[0]) 
            if (line_count - n_pos_analysed - lines_with_coments) > new_positions: wait = False
            else: time.sleep(sleep_between_checks)
        else: 
            counter += 1
            if counter > 3: raise Exception(f"Error counting lines in {position_path}: {result.stderr} (after 3 attempts). ")
            time.sleep(sleep_between_checks*5)

    if return_n_pos: return line_count - n_pos_analysed - lines_with_coments
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

def get_plumed_grid_2D(X, Y, pl_min=None, pl_max=None, print_info=False, periodic=[False,False]):

    # get grid min, max, number of bins and grid spacing
    x_min, x_max, nx = np.min(X), np.max(X), len(X[0])
    y_min, y_max, ny = np.min(Y), np.max(Y), len(Y)
    dx, dy = X[0,1] - X[0,0], Y[1,0] - Y[0,0]
    
    # if the system is periodic, the plumed grid will be the same as the MFI grid
    if periodic[0] and periodic[1]: 
        if print_info: print("The system is periodic. The grid will be the same as the MFI grid.")
        return [X, Y, [x_min, y_min], [x_max, y_max], [nx, ny], [0, -ny, 0, -nx]]
    
    if periodic[0] or periodic[1]: print("\n Warning \n The system is periodic in one direction, but treated as non-periodic in both directions. Please check the code in the get_plumed_grid_2D function. \n") 
  
    # if pl_min and pl_max are not given, the plumed max and min will be the MFI max and min +/- 1
    if pl_min is None: pl_min = [x_min - 1, y_min - 1]
    if pl_max is None: pl_max = [x_max + 1, y_max + 1]
     
    # find the extra bins needed to reach the plumed min and max    
    diff_x_low = x_min - pl_min[0]
    diff_x_up = pl_max[0] - x_max
    diff_y_low = y_min - pl_min[1]
    diff_y_up = pl_max[1] - y_max
    
    nx_low_extra = int(np.ceil(diff_x_low / dx))
    nx_up_extra = int(np.ceil(diff_x_up / dx))
    ny_low_extra = int(np.ceil(diff_y_low / dy))
    ny_up_extra = int(np.ceil(diff_y_up / dy))
    
    # Find the new plumed min and max that corresponds to the extra bins (nx_low_extra, nx_up_extra, ny_low_extra, ny_up_extra)    
    pl_x_min_new = x_min - nx_low_extra * dx
    pl_x_max_new = x_max + nx_up_extra * dx
    pl_y_min_new = y_min - ny_low_extra * dy
    pl_y_max_new = y_max + ny_up_extra * dy
    
    # find the number of bins of the new plumed grid
    pl_nx = nx + nx_low_extra + nx_up_extra
    pl_ny = ny + ny_low_extra + ny_up_extra
    
    # create the new plumed grid
    pl_x = np.linspace(pl_x_min_new, pl_x_max_new, pl_nx)
    pl_y = np.linspace(pl_y_min_new, pl_y_max_new, pl_ny)
    plX, plY = np.meshgrid(pl_x, pl_y)
    
    # check if the plumed grid matches the MFI grid
    assert np.mean(abs(plX[ny_low_extra:-ny_up_extra, nx_low_extra:-nx_up_extra] - X)) < 0.1, "The plumed grid does not match the MFI grid. Please check the calculation of the plumed grid."
    
    # print the grid information (if applicable) and return the plumed grid
    if print_info: print("The MFI grid was: \nxmin=", round(x_min,2), " xmax=", round(x_max,2), " nx=", nx, "\nymin=", round(y_min,2), " ymax=", round(y_max,2), " ny=", ny)
    if print_info: print("\nThe new PLUMED grid is: \npl_xmin=", round(pl_x_min_new,2), " pl_xmax=", round(pl_x_max_new,2), " pl_nx=", pl_nx, "\npl_ymin=", round(pl_y_min_new,2), " pl_ymax=", round(pl_y_max_new,2), " pl_ny=", pl_ny) 
    return [plX, plY, [pl_x_min_new, pl_y_min_new], [pl_x_max_new, pl_y_max_new], [pl_nx, pl_ny], [ny_low_extra, ny_up_extra, nx_low_extra, nx_up_extra]]

def make_external_bias_2D(X, Y, FES=None, Bias=None, Bias_sf=1, gaus_filter_sigma=None, FES_cutoff=None, pl_min=None, pl_max=None, periodic=[False, False], file_name_extension="", return_array=None, cv_name=["p.x", "p.y"]):
    
    #Get plumed grid
    [plX, plY, pl_min, pl_max, pl_n, pl_ext] = get_plumed_grid_2D(X, Y, pl_min, pl_max, print_info=False, periodic=periodic)
    assert np.sum(plX[pl_ext[0]:-pl_ext[1], pl_ext[2]:-pl_ext[3]] - X) / (np.shape(X)[1] * np.shape(X)[0]) < 0.01, "The plumed grid does not match the MFI grid. Please check the grid creation."
    assert np.sum(plY[pl_ext[0]:-pl_ext[1], pl_ext[2]:-pl_ext[3]] - Y) / (np.shape(X)[1] * np.shape(X)[0]) < 0.01, "The plumed grid does not match the MFI grid. Please check the grid creation."
    
    #Find plumed Bias
    pl_Bias = np.zeros(pl_n[::-1])
    if FES is not None: 
        if FES_cutoff is not None: FES = np.where(FES < FES_cutoff, FES, FES_cutoff)
        pl_Bias[pl_ext[0]:-pl_ext[1], pl_ext[2]:-pl_ext[3]] = -FES - np.min(-FES)
    elif Bias is not None: 
        if FES_cutoff is not None:
            max_Bias = np.max(Bias)
            if max_Bias > FES_cutoff: Bias = np.where(Bias < (max_Bias - FES_cutoff), 0, Bias - (max_Bias - FES_cutoff))
        pl_Bias[pl_ext[0]:-pl_ext[1], pl_ext[2]:-pl_ext[3]] = Bias
    
    # Modify Bias by scaling factor and/or gaussian filter    
    if gaus_filter_sigma is not None: 
        if periodic[0] and periodic[1]: pl_Bias = gaussian_filter(pl_Bias, sigma=gaus_filter_sigma, mode='wrap')
        else: pl_Bias = gaussian_filter(pl_Bias, sigma=gaus_filter_sigma)
    if Bias_sf > 0 and Bias_sf != 1: pl_Bias *= Bias_sf
    
    #Find gradient of Bias
    pl_F_bias_y, pl_F_bias_x = np.gradient(pl_Bias, plY[1,0]-plY[0,0], plX[0,1]-plX[0,0])
    
    # get format to save in file "external_bias2.dat"
    if periodic[0] and periodic[1]: 
        plX_flat = np.copy(plX)[:-1,:-1].flatten()
        plY_flat = np.copy(plY)[:-1,:-1].flatten()
        pl_bias_flat = np.copy(pl_Bias)[:-1,:-1].flatten()
        pl_Fbias_x_flat = np.copy(pl_F_bias_x)[:-1,:-1].flatten()
        pl_Fbias_y_flat = np.copy(pl_F_bias_y)[:-1,:-1].flatten()
        if round(pl_min[0] + np.pi, 1) == 0.0 and round(pl_min[1] + np.pi, 1) == 0.0: pl_min[0], pl_max[0], pl_min[1], pl_max[1] = "-pi", "pi", "-pi", "pi"
        periodic_0, periodic_1 = "true", "true"  
    else:
        plX_flat = plX.flatten()
        plY_flat = plY.flatten()
        pl_bias_flat = pl_Bias.flatten()
        pl_Fbias_x_flat = pl_F_bias_x.flatten()
        pl_Fbias_y_flat = pl_F_bias_y.flatten()
        periodic_0, periodic_1 = "false", "false"
        
    external_bias_vector = np.array([plX_flat, plY_flat, pl_bias_flat, pl_Fbias_x_flat, pl_Fbias_y_flat]).T  
    
    # change the plumed nbins to the correct format (irrespective of periodicity)
    pl_n = [pl_n[0]-1, pl_n[1]-1]
    
    with open(f"external_bias{file_name_extension}.dat", "w") as f:
        f.write(f"#! FIELDS {cv_name[0]} {cv_name[1]} external.bias der_{cv_name[0]} der_{cv_name[1]}\n")
        f.write(f"#! SET min_{cv_name[0]} {pl_min[0]}\n")
        f.write(f"#! SET max_{cv_name[0]} {pl_max[0]}\n")
        f.write(f"#! SET nbins_{cv_name[0]} {pl_n[0]}\n")
        f.write(f"#! SET periodic_{cv_name[0]} {periodic_0}\n")
        f.write(f"#! SET min_{cv_name[1]} {pl_min[1]}\n")
        f.write(f"#! SET max_{cv_name[1]} {pl_max[1]}\n")
        f.write(f"#! SET nbins_{cv_name[1]} {pl_n[1]}\n")
        f.write(f"#! SET periodic_{cv_name[1]} {periodic_1}\n")
        add_extra_lines = 1 if periodic[0] is False else 0
        for i in range(0, len(external_bias_vector), pl_n[0]+add_extra_lines):
            np.savetxt(f, external_bias_vector[i:i+pl_n[0]+add_extra_lines], fmt='%.9f')
            f.write("\n")

    if return_array != None: return [pl_Bias[pl_ext[0]:-pl_ext[1], pl_ext[2]:-pl_ext[3]], pl_F_bias_x[pl_ext[0]:-pl_ext[1], pl_ext[2]:-pl_ext[3]], pl_F_bias_y[pl_ext[0]:-pl_ext[1], pl_ext[2]:-pl_ext[3]], f"{os.getcwd()}/external_bias{file_name_extension}.dat"]

def find_total_bias_from_hills(X, Y, HILLS, nhills=-1, periodic=[False, False]):
    #Specify grid variables
    gridx, gridy = X[0, :], Y[:, 0]
    nbins_yx = np.shape(X)
    grid_min = [np.min(X), np.min(Y)]
    grid_max = [np.max(X), np.max(Y)]
    grid_length = [grid_max[0] - grid_min[0], grid_max[1] - grid_min[1]]
    periodic_range = [0.25*grid_length[0], 0.25*grid_length[1]]

    # initialise Bias 
    Bias = np.zeros(nbins_yx)

    # Specify Gaussian parameters
    sigma_meta2_x, sigma_meta2_y = HILLS[1, 3]**2, HILLS[1, 4] ** 2
    Gamma_Factor = (HILLS[1, 6] - 1) / HILLS[1, 6]
    assert HILLS[2, 6] == HILLS[1, 6], "The Gamma_Factor is not the same for all hills. Please check the code (account for non_WT MetaD)."
    if nhills == -1: nhills = len(HILLS)
    
    #Cycle over hills
    for i in range(nhills):
        
        # get position and height of Gaussian. Add the Gaussian to the Bias
        pos_meta_x, pos_meta_y, height_meta = HILLS[i, 1], HILLS[i, 2], HILLS[i, 5] * Gamma_Factor # centre position (x and y) of Gaussian, and height of Gaussian
        pos_meta = find_periodic_points([pos_meta_x], [pos_meta_y], grid_min, grid_max, periodic, grid_length, periodic_range) if (periodic[0] or periodic[1]) else [[xi, yi] for xi, yi in zip([pos_meta_x], [pos_meta_y])]
        for pos_m in pos_meta:
            kernelmeta_x = np.exp( - np.square(gridx - pos_m[0]) / (2 * sigma_meta2_x)) * height_meta
            kernelmeta_y = np.exp( - np.square(gridy - pos_m[1]) / (2 * sigma_meta2_y))
            Bias += np.outer(kernelmeta_y, kernelmeta_x)
   
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

####  ---- Read simulation Data  --------------------------------------------------------  ####

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
        if os.path.exists(pos_path_cp): os.system(f"rm {pos_path_cp}")
                
        if metad_h is not None and metad_h > 0: 
            hills_path_cp = hills_path + "_cp"
            if os.path.exists(hills_path_cp): os.system(f"rm {hills_path_cp}")
        
        os.system(f"cp {pos_path} {pos_path_cp}")        
        if metad_h is not None and metad_h > 0: os.system(f"cp {hills_path} {hills_path_cp}")
        
        #### ~~~~~ Load position Data ~~~~~ ####

        with open(pos_path_cp, 'r') as file: pos = [line.split() for line in file if not line.startswith("#")]
        
        # check if the position data is empty
        assert len(pos) > n_pos_per_window, f"The position data: {pos = } is empty ({len(pos) = }) (ec1)"
        
        # the first position is at time 0 is removed. This is done because the position is preset and to have the number of positions being a const multiple (=n_pos_per_window) of the number of hills.
        # n_pos_per_window is the same for every hill, but at the start there are usually n_pos_per_window+1 positions. 
        pos_time_00, pos_x_00, pos_y_00 = pos[0][0], pos[0][1], pos[0][2]
        pos = pos[1:]

        # remove the last line if it is not complete
        if len(pos[-1]) < len(pos[-2]): pos = pos[:-1]  
        last_index_pos_line = len(pos[-2])-1
        if len(pos[-1][last_index_pos_line]) < len(pos[-2][last_index_pos_line]) / 2: pos = pos[:-1]
        assert len(pos) > n_pos_per_window, f"The position data: {pos = } is empty ({len(pos) = }) (ec2)"

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
            if n_pos_analysed == 0: hills = [[pos_time_00, pos_x_00, pos_y_00, hills[0][3], hills[0][4], 0.0, hills[0][6]]] + hills
            
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
                t_index = np.where(t_hills_slided == 0.0)
                assert len(t_index[0]) == 1, f"Could not find the (first) hill corresponding to the first position: t_pos_0 - dt_pos = {t_pos_0} - {dt_pos} = {t_hills_0 = } in the hills file: {hills_path} \n{hills[:5] = } \n{hills[-5:] = }\n{t_hills_slided = }\n{round(hills[1,0] - t_hills_0, 4) = }"
            
                # cut hills data, removing the hills before the first hill corresponding to the first position
                hills = hills[t_index[0][0]:]
            
            # check if t_pos_0 - dt_pos == t_hills_0
            assert round(pos[0][0] - dt_pos, 4) == round(hills[0][0], 4), f"The first time in the position data and the first time in the hills data don't match: pos[0][0] - dt_pos = {pos[0][0]} - {dt_pos} = {pos[0][0] - dt_pos} != {hills[0][0] = }"

            #### ~~~~~ Cut from the end to make len(hills) * n_pos_per_window = len(pos) ~~~~~ ####        

            len_hills, len_pos = len(hills), len(pos)
            if len_hills * n_pos_per_window != len_pos:

                # if there are too many hills lines, remove the extra hills lines
                if len_hills * n_pos_per_window > len_pos: 
                    extra_hills = int(np.ceil((len_hills * n_pos_per_window - len_pos) / n_pos_per_window))
                    hills = hills[:-extra_hills] # this will require removing some positions with the next if statement
                    len_hills = len(hills)
                
                # if there are too many position lines, remove the extra position lines
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
        filtered_lines = [line for line in external_bias.strip().split('\n') if (not line.startswith("#"))]
        data_array = np.array([list(map(float, line.split())) for line in filtered_lines if line.strip() != ""])
        
        # find numbor of bins for each dimension from the comment lines
        nbins = [ int(cl.split()[-1]) for cl in comment_lines if "nbins" in cl]

        periodic = [ cl.split()[-1] for cl in comment_lines if "periodic" in cl]
        
        nbins_new = [nbins[i] + 1 if periodic[i] == "false" else nbins[i] for i in range(len(nbins))]
            
        data_periodic_wrap = []
        # if periodic[0] is True, one needs to copy over the first line of the block to the end of the block. This is done because plumed works with periodic boundaries where the walls only appear at the lower end, however, MFI works with periodic boundaries that appear on both sides. 
        if periodic[0] == "true":
            for i in range(0, len(data_array), nbins[0]):
                first_line = data_array[i:i + nbins[0]][0].copy()
                first_line[0] = abs(first_line[0])
                block = np.vstack([data_array[i:i + nbins[0]], first_line])
                data_periodic_wrap.append(block)
            data_periodic_wrap = np.vstack(data_periodic_wrap)
            nbins[0] += 1
        else: data_periodic_wrap = data_array
            
        # if periodic[1] is True, one needs to copy over the first block to the end of the file. This is done because plumed works with periodic boundaries where the walls only appear at the lower end, however, MFI works with periodic boundaries that appear on both sides. 
        if len(nbins)>1 and periodic[1] == "true":
            first_block = data_periodic_wrap[:nbins[0]].copy()
            first_block[:, 1] = abs(first_block[:, 1])
            data_periodic_wrap = np.vstack([data_periodic_wrap, first_block])
            nbins[1] += 1
        
        # for 1 or 2 dimensions return the data reshaped to a meshgrid format
        if len(nbins) == 1 or len(nbins) == 2: 
            return [data_periodic_wrap[:, i].reshape(nbins) for i in range(len(data_periodic_wrap[0]))]
        
        # for other dimensions, raise an error.              
        else: raise ValueError("Unrecognized format. Only 1D and 2D grids are supported")
    
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

####  ---- MFI functions  --------------------------------------------------------  ####

for _MFI_functions_ in [1]:

    def find_periodic_points(x_coord, y_coord, min_grid, max_grid, periodic, grid_length=None, periodic_range=None):
        """Finds periodic copies of input coordinates. First checks if systems is periodic. If not, returns input coordinate array. Next, it checks if each coordinate is within the boundary range (grid min/max +/- periodic_range). If it is, periodic copies will be made on the other side of the CV-domain. 
        
        Args:
            x_coord (float): CV1-coordinate
            y_coord (float): CV2-coordinate
            min_grid (list): list of CV1-minimum value of grid and CV2-minimum value of grid
            max_grid (list): list of CV1-maximum value of grid and CV2-maximum value of grid
            periodic (list or array of shape (2,)): Specifies if system is periodic. First entry sepcifies if CV1 is periodic. Second entry sepcifies if CV2 is periodic. value of 0 corresponds to non-periodic CV1. Value of 1 corresponds to periodic CV1.
        Returns:
            list: list of [x-coord, y-coord] pairs (i.e. [[x1,y1], [x2,y2], ..., [xn,yn]])
        """
        if periodic[0] == 0 and periodic[1] == 0: return [ [xi , yi] for xi, yi in zip(x_coord, y_coord)]
        else:
            # Use periodic extension for defining PBC
            if grid_length is None: grid_length = [max_grid[0] - min_grid[0], max_grid[1] - min_grid[1]]
            if periodic_range is None: periodic_range = [grid_length[0]/4, grid_length[1]/4]        
            coord_list = []
            
            for xi, yi in zip(x_coord, y_coord):
                coord_list.append([xi, yi])

                # There are potentially 4 points, 1 original and 3 periodic copies, or less.
                copy_record = [0, 0, 0, 0]
                
                if periodic[0]: # check for x-copy
                    if xi < min_grid[0] + periodic_range[0]:
                        coord_list.append([xi + grid_length[0], yi])
                        copy_record[0] = 1
                    elif xi > max_grid[0] - periodic_range[0]:
                        coord_list.append([xi - grid_length[0], yi])
                        copy_record[1] = 1
                if periodic[1]: # check for y-copy
                    if yi < min_grid[1] + periodic_range[1]:
                        coord_list.append([xi, yi + grid_length[1]])
                        copy_record[2] = 1
                    elif yi > max_grid[1] - periodic_range[1]:
                        coord_list.append([xi, yi - grid_length[1]])
                        copy_record[3] = 1
                        
                # check for xy-copy
                if sum(copy_record) == 2:
                    if copy_record[0] == 1 and copy_record[2] == 1: coord_list.append([xi + grid_length[0], yi + grid_length[1]])
                    elif copy_record[1] == 1 and copy_record[2] == 1: coord_list.append([xi - grid_length[0], yi + grid_length[1]])
                    elif copy_record[0] == 1 and copy_record[3] == 1: coord_list.append([xi + grid_length[0], yi - grid_length[1]])
                    elif copy_record[1] == 1 and copy_record[3] == 1: coord_list.append([xi - grid_length[0], yi - grid_length[1]])       
        
        return coord_list

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
        if hp_kappa_x is None: hp_kappa_x = 0
        if hp_kappa_y is None: hp_kappa_y = 0
        
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
        grid_spacex = (max_grid[0] - min_grid[0]) / (nbins_yx[1] - 1)
        grid_spacey = (max_grid[1] - min_grid[1]) / (nbins_yx[0] - 1)	
        # gridx = np.linspace(min_grid[0], max_grid[0], nbins_yx[1])
        # gridy = np.linspace(min_grid[1], max_grid[1], nbins_yx[0])
        # X, Y = np.meshgrid(gridx, gridy)

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
        freq_1dx = np.fft.fftfreq(nbins_yx[1], grid_spacex)  
        freq_1dy = np.fft.fftfreq(nbins_yx[0], grid_spacey)
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
        return fes

    def MFI_forces(HILLS, position_x, position_y, const, bw2, kT, X, Y, F_static_x, F_static_y, n_pos_per_window=10, Gamma_Factor=None, n_pos=-1, periodic=[False, False], PD_limit = 1E-10, return_FES=False):

        #Specify grid variables
        gridx, gridy = X[0, :], Y[:, 0]
        nbins_yx = np.shape(X)
        grid_min = [np.min(X), np.min(Y)]
        grid_max = [np.max(X), np.max(Y)]
        grid_length = [grid_max[0] - grid_min[0], grid_max[1] - grid_min[1]]
        periodic_range = [0.25*grid_length[0], 0.25*grid_length[1]]

        # initialise force terms
        PD = np.zeros(nbins_yx)
        PD2 = np.zeros(nbins_yx)
        Force_num_x = np.zeros(nbins_yx)
        Force_num_y = np.zeros(nbins_yx)
        ofv_num_x = np.zeros(nbins_yx)
        ofv_num_y = np.zeros(nbins_yx)
        Bias = np.zeros(nbins_yx)
        F_bias_x = np.zeros(nbins_yx)
        F_bias_y = np.zeros(nbins_yx)
        
        # if n_pos not specified, n_pos is the length of the position array
        if n_pos is None or n_pos < 0: n_pos = len(position_x)
        # if n_pos_per_window not specified, n_pos_per_window is the length of the position array if metad is not active, otherwise it is the length of the position array divided by the number of hills
        if n_pos_per_window is None or n_pos_per_window <= 0: n_pos_per_window = n_pos if HILLS is None else int(round(len(position)/len(HILLS)))
        # n_windows: Forces are calculated in (n) windows of constant bias. Use np.ceil to include incomplete windows (windows with less than n_pos_per_window positions). use min() to avoid n_windows > len(hills).
        if HILLS is not None: n_windows = min(int(np.ceil(n_pos / n_pos_per_window)), len(HILLS))
        else: n_windows = int(np.ceil(n_pos / n_pos_per_window))
        
        # get the width of the metadynamics Gaussian and check if it is the same at the end.
        sigma_meta2_x, sigma_meta2_y = HILLS[1, 3]**2, HILLS[1, 4] ** 2  if HILLS is not None else None # width of Gaussian
        assert (HILLS[1, 3] == HILLS[-1, 3] and HILLS[1, 4] == HILLS[-1, 4]), "The width of the Gaussian is not constant across the hills data."

        #Cycle over windows of constant bias (for each deposition of a gaussian bias)
        for i in range(n_windows):
            
            ###--- Updated Bias and Bias_force from Metadyanmics of new window      
            if HILLS is not None:
                pos_meta_x, pos_meta_y, height_meta = HILLS[i, 1], HILLS[i, 2], HILLS[i, 5] * Gamma_Factor # centre position (x and y) of Gaussian, and height of Gaussian
                pos_meta = find_periodic_points([pos_meta_x], [pos_meta_y], grid_min, grid_max, periodic, grid_length, periodic_range) if (periodic[0] or periodic[1]) else [[xi, yi] for xi, yi in zip([pos_meta_x], [pos_meta_y])]
                for pos_m in pos_meta:
                    kernelmeta_x = np.exp( - np.square(gridx - pos_m[0]) / (2 * sigma_meta2_x)) * height_meta
                    kernelmeta_y = np.exp( - np.square(gridy - pos_m[1]) / (2 * sigma_meta2_y))
                    Bias += np.outer(kernelmeta_y, kernelmeta_x)
                    F_bias_x += np.outer(kernelmeta_y, np.multiply(kernelmeta_x, (gridx - pos_m[0])) / sigma_meta2_x )
                    F_bias_y += np.outer(np.multiply(kernelmeta_y, (gridy - pos_m[1])) / sigma_meta2_y, kernelmeta_x )
                
            ###--- Get PD (Probability Density) and PD_force from sampling data        
            PD_i, F_PD_x_i, F_PD_y_i = np.zeros(nbins_yx), np.zeros(nbins_yx), np.zeros(nbins_yx)
            pos_x, pos_y = position_x[i*n_pos_per_window : (i+1)*n_pos_per_window], position_y[i*n_pos_per_window : (i+1)*n_pos_per_window]  # positons of window of constant bias force.
            pos = find_periodic_points(pos_x, pos_y, grid_min, grid_max, periodic, grid_length, periodic_range) if (periodic[0] or periodic[1]) else [[xi, yi] for xi, yi in zip(pos_x, pos_y)]

            for p in pos: 
                kernel_x = np.exp( - np.square(gridx - p[0]) / (2 * bw2[0])) * const #add constant here for less computations
                kernel_y = np.exp( - np.square(gridy - p[1]) / (2 * bw2[1]))
                PD_i += np.outer(kernel_y, kernel_x)
                kernel_x *= kT / bw2[2] #add constant here for less computations
                F_PD_x_i += np.outer(kernel_y, np.multiply(kernel_x, (gridx - p[0])) )
                F_PD_y_i += np.outer(np.multiply(kernel_y, (gridy - p[1])) , kernel_x )

            PD_i = np.where(PD_i > PD_limit, PD_i, 0)  # truncated probability density of window
            F_PD_x_i = np.divide(F_PD_x_i, PD_i, out=np.zeros_like(F_PD_x_i), where=PD_i > PD_limit)
            F_PD_y_i = np.divide(F_PD_y_i, PD_i, out=np.zeros_like(F_PD_y_i), where=PD_i > PD_limit)
            
            
            ###--- Update force terms        
            PD += PD_i # total probability density     
            Force_x_i = F_PD_x_i + F_bias_x - F_static_x
            Force_y_i = F_PD_y_i + F_bias_y - F_static_y
            Force_num_x += np.multiply(PD_i, Force_x_i)
            Force_num_y += np.multiply(PD_i, Force_y_i)
            # terms for error calculation
            PD2 += np.square(PD_i) 
            ofv_num_x += np.multiply(PD_i, np.square(Force_x_i))  
            ofv_num_y += np.multiply(PD_i, np.square(Force_y_i))  
        
        ###--- Calculate Force and FES before returning  
        Force_x = np.divide(Force_num_x, PD, out=np.zeros_like(Force_num_x), where=PD > PD_limit)
        Force_y = np.divide(Force_num_y, PD, out=np.zeros_like(Force_num_y), where=PD > PD_limit) 
        
        if return_FES: FES = FFT_intg_2D(Force_x, Force_y, grid_min, grid_max, periodic)
        else: FES = np.zeros(nbins_yx)
        
        return PD, PD2, Force_x, Force_y, ofv_num_x, ofv_num_y, F_bias_x, F_bias_y, Bias, FES

    def MFI_forces_static(position_x, position_y, const, bw2, kT, X, Y, F_static_x, F_static_y, n_pos_per_window=10, npos=-1, periodic=[False, False], PD_limit = 1E-10):

        #Specify grid variables
        gridx, gridy = X[0, :], Y[:, 0]
        nbins_yx = np.shape(X)
        grid_min = [np.min(X), np.min(Y)]
        grid_max = [np.max(X), np.max(Y)]
        grid_length = [grid_max[0] - grid_min[0], grid_max[1] - grid_min[1]]
        periodic_range = [0.25*grid_length[0], 0.25*grid_length[1]]

        # initialise force terms
        PD = np.zeros(nbins_yx)
        PD2 = np.zeros(nbins_yx)
        Force_num_x = np.zeros(nbins_yx)
        Force_num_y = np.zeros(nbins_yx)
        ofv_num_x = np.zeros(nbins_yx)
        ofv_num_y = np.zeros(nbins_yx)
        
        if npos == -1: npos = len(position_x)    
        if n_pos_per_window == -1: n_windows, n_pos_per_window = 1, npos
        else: n_windows = int(np.ceil(npos / n_pos_per_window))
        
        #Cycle over windows of constant bias (for each deposition of a gaussian bias)
        for i in range(n_windows):
            
            ###--- Get PD (Probability Density) and PD_force from sampling data        
            PD_i, F_PD_x_i, F_PD_y_i = np.zeros(nbins_yx), np.zeros(nbins_yx), np.zeros(nbins_yx)
            pos_x, pos_y = position_x[i*n_pos_per_window : (i+1)*n_pos_per_window], position_y[i*n_pos_per_window : (i+1)*n_pos_per_window]  # positons of window of constant bias force.
            pos = find_periodic_points(pos_x, pos_y, grid_min, grid_max, periodic, grid_length, periodic_range) if (periodic[0] or periodic[1]) else [[xi, yi] for xi, yi in zip(pos_x, pos_y)]

            for p in pos: 
                kernel_x = np.exp( - np.square(gridx - p[0]) / (2 * bw2[0])) * const #add constant here for less computations
                kernel_y = np.exp( - np.square(gridy - p[1]) / (2 * bw2[1]))
                PD_i += np.outer(kernel_y, kernel_x)
                kernel_x *= kT / bw2[2] #add constant here for less computations
                F_PD_x_i += np.outer(kernel_y, np.multiply(kernel_x, (gridx - p[0])) )
                F_PD_y_i += np.outer(np.multiply(kernel_y, (gridy - p[1])) , kernel_x )

            PD_i = np.where(PD_i > PD_limit, PD_i, 0)  # truncated probability density of window
            F_PD_x_i = np.divide(F_PD_x_i, PD_i, out=np.zeros_like(F_PD_x_i), where=PD_i > PD_limit)
            F_PD_y_i = np.divide(F_PD_y_i, PD_i, out=np.zeros_like(F_PD_y_i), where=PD_i > PD_limit)
            
            
            ###--- Update force terms        
            PD += PD_i # total probability density     
            Force_x_i = F_PD_x_i - F_static_x
            Force_y_i = F_PD_y_i - F_static_y
            Force_num_x += np.multiply(PD_i, Force_x_i)
            Force_num_y += np.multiply(PD_i, Force_y_i)
            # terms for error calculation
            PD2 += np.square(PD_i) 
            ofv_num_x += np.multiply(PD_i, np.square(Force_x_i))  
            ofv_num_y += np.multiply(PD_i, np.square(Force_y_i))  
        
        ###--- Calculate Force and FES before returning  
        Force_x = np.divide(Force_num_x, PD, out=np.zeros_like(Force_num_x), where=PD > PD_limit)
        Force_y = np.divide(Force_num_y, PD, out=np.zeros_like(Force_num_y), where=PD > PD_limit)                             
        FES = FFT_intg_2D(Force_x, Force_y, grid_min, grid_max, periodic)
        
        return PD, PD2, Force_x, Force_y, ofv_num_x, ofv_num_y, FES

    def patch_forces(forces, base_forces=None, PD_limit=1E-10):
    
        # check if forces and base_forces are lists. If they are, convert them to numpy arrays
        if isinstance(forces, list): forces = np.array(forces)
        if isinstance(base_forces, list): base_forces = np.array(base_forces)
        
        # Find the length of the grid 
        if len(np.shape(forces)) == 4: nbins_yx = np.shape(forces[0, 0])
        elif len(np.shape(forces)) == 3: nbins_yx = np.shape(forces[0])
        else: raise ValueError("\n *** The forces array has an unexpected shape *** \n")
        
        if np.shape(forces)[-3] == 6:
            
            #Initialise arrays
            PD = np.zeros(nbins_yx)
            PD2 = np.zeros(nbins_yx)
            Force_x = np.zeros(nbins_yx)
            Force_y = np.zeros(nbins_yx)
            ofv_num_x = np.zeros(nbins_yx)
            ofv_num_y = np.zeros(nbins_yx)
            
            # add forces to the total
            if len(np.shape(forces)) == 4: 
                for i in range(len(forces)):
                    PD += forces[i, 0]
                    PD2 += forces[i, 1]
                    Force_x += np.multiply(forces[i, 2], forces[i, 0])
                    Force_y += np.multiply(forces[i, 3], forces[i, 0])
                    ofv_num_x += forces[i, 4]
                    ofv_num_y += forces[i, 5]
            elif len(np.shape(forces)) == 3: 
                PD += forces[0]
                PD2 += forces[1]
                Force_x += np.multiply(forces[2], forces[0])
                Force_y += np.multiply(forces[3], forces[0])
                ofv_num_x += forces[4]
                ofv_num_y += forces[5]
                
            # if base_forces is not None, add base_forces to the total
            if base_forces is not None:
                assert np.shape(base_forces)[-3] == 6, "The base_forces array has an unexpected shape"
                if len(np.shape(base_forces)) == 4:
                    for i in range(len(base_forces)):
                        PD += base_forces[i, 0]
                        PD2 += base_forces[i, 1]
                        Force_x += np.multiply(base_forces[i, 2], base_forces[i, 0])
                        Force_y += np.multiply(base_forces[i, 3], base_forces[i, 0])
                        ofv_num_x += base_forces[i, 4]
                        ofv_num_y += base_forces[i, 5]
                elif len(np.shape(base_forces)) == 3:
                    PD += base_forces[0]
                    PD2 += base_forces[1]
                    Force_x += np.multiply(base_forces[2], base_forces[0])
                    Force_y += np.multiply(base_forces[3], base_forces[0])
                    ofv_num_x += base_forces[4]
                    ofv_num_y += base_forces[5]
            
            # force is divided by the probability density
            Force_x = np.divide(Force_x, PD, out=np.zeros_like(Force_x), where=PD>PD_limit)
            Force_y = np.divide(Force_y, PD, out=np.zeros_like(Force_y), where=PD>PD_limit)
                    
            return [PD, PD2, Force_x, Force_y, ofv_num_x, ofv_num_y]
        
        elif np.shape(forces)[-3] == 3:
            
            #Initialise arrays
            PD = np.zeros(nbins_yx)
            Force_x = np.zeros(nbins_yx)
            Force_y = np.zeros(nbins_yx)
            
            # add forces to the total
            if len(np.shape(forces)) == 4: 
                for i in range(len(forces)):
                    PD += forces[i, 0]
                    Force_x += np.multiply(forces[i, 1], forces[i, 0])
                    Force_y += np.multiply(forces[i, 2], forces[i, 0])
            elif len(np.shape(forces)) == 3: 
                PD += forces[0]
                Force_x += np.multiply(forces[1], forces[0])
                Force_y += np.multiply(forces[2], forces[0])
                
            # if base_forces is not None, add base_forces to the total
            if base_forces is not None:
                assert np.shape(base_forces)[-3] == 3, "The base_forces array has an unexpected shape"
                if len(np.shape(base_forces)) == 4:
                    for i in range(len(base_forces)):
                        PD += base_forces[i, 0]
                        Force_x += np.multiply(base_forces[i, 1], base_forces[i, 0])
                        Force_y += np.multiply(base_forces[i, 2], base_forces[i, 0])
                elif len(np.shape(base_forces)) == 3:
                    PD += base_forces[0]
                    Force_x += np.multiply(base_forces[1], base_forces[0])
                    Force_y += np.multiply(base_forces[2], base_forces[0])
            
            # force is divided by the probability density
            Force_x = np.divide(Force_x, PD, out=np.zeros_like(Force_x), where=PD>PD_limit)
            Force_y = np.divide(Force_y, PD, out=np.zeros_like(Force_y), where=PD>PD_limit)
                    
            return [PD, Force_x, Force_y]
        
        else: raise ValueError("\n *** The forces array has an unexpected shape! np.shape(forces)[-3] should be 3 or 6. *** \n")
        
####  ---- Real Time Re-Initialisation functions  ----  ####

    def check_US_criteria(error_map, cutoff_map=None, gaussian_sigma=3, check_termination=None, periodic=[False, False]):
        
        if cutoff_map is None: cutoff_map = np.ones_like(error_map)
        
        filter_mode = "wrap" if (periodic[0] or periodic[1]) else "reflect"
        error_smooth = gaussian_filter(error_map, gaussian_sigma, mode=filter_mode) * cutoff_map   
        error_avr = np.sum(error_smooth) / np.count_nonzero(error_smooth)
            
        if check_termination is None:
            index_max = np.unravel_index(np.nanargmax(error_smooth), error_smooth.shape)
            error_max = error_smooth[index_max]
            return error_avr, error_max
            
        else:
            X, Y, hp_centre_x, hp_centre_y = check_termination
            hp_index_x = np.argmin(np.abs(X[0, :] - hp_centre_x))
            hp_index_y = np.argmin(np.abs(Y[:, 0] - hp_centre_y))
            error_centre = error_smooth[hp_index_y, hp_index_x]
            return error_avr, error_centre
        
    def find_hp_centre(X, Y, error_map, cutoff_map=None, gaussian_sigma=3, prev_hp_centre_x=None, prev_hp_centre_y=None, periodic=[False, False]):

        if cutoff_map is None: cutoff_map = np.ones_like(error_map)

        filter_mode = "wrap" if (periodic[0] or periodic[1]) else "reflect"
        error_smooth = gaussian_filter(error_map, sigma=gaussian_sigma, mode=filter_mode) * cutoff_map   

        # find the position of the maximum error
        index_max = np.unravel_index(np.nanargmax(error_smooth), error_smooth.shape)
        new_hp_centre_x, new_hp_centre_y = round(X[index_max],6), round(Y[index_max],6)                
        
        # if the previous simulation was in the US phase (hp centre is defined), check if the new hp centre is somewhere else. If centre in the same position, switch to flat phase.
        for prev_hp_x, prev_hp_y in zip(prev_hp_centre_x, prev_hp_centre_y):
            if prev_hp_x is not None and prev_hp_y is not None: 
                d_hp_centre_x, d_hp_centre_y = new_hp_centre_x - prev_hp_x, new_hp_centre_y - prev_hp_y
                if abs(d_hp_centre_x) < (X[0, -1] - X[0, 0])/10 and abs(d_hp_centre_y) < (Y[-1, 0] - Y[0, 0])/10: return None, None
            
        return new_hp_centre_x, new_hp_centre_y

####  ---- Statistical Analysis of Error progression collections  ----  ####

for _stat_analy_ in [1]:

    def bootstrapping_error(X, Y, force_array, n_bootstrap, base_force_array=None, periodic=[False, False], FES_cutoff=None, PD_cutoff=None, PD_limit=1E-10, use_VNORM=False, get_progression=False, print_progress=False):
        
        # Get grid variables
        nbins_yx = X.shape
        min_grid = [np.min(X), np.min(Y)]
        max_grid = [np.max(X), np.max(Y)]
        
        # Get the correct shape of the force array (Needs to be [PD, FX, FY], but sometimes it is [PD, PD2, FX, FY, OFV_X, OFV_Y])
        if force_array.shape[-3] == 6: force_array = force_array[:,[0,2,3]]
        if force_array.shape[-3] != 3: raise ValueError("force_array should have shape (n_forces, 3) or (n_forces, 6)")
    
        if base_force_array is not None:
            if base_force_array.shape[-3] == 6: base_force_array = base_force_array[:,[0,2,3]]
            if base_force_array.shape[-3] != 3: raise ValueError("base_force_array should have shape (n_forces, 3) or (n_forces, 6)")
            
            # combine base_force_array and force_array
            force_array = np.concatenate((force_array, base_force_array), axis=0)
            
        if len(force_array) > 20: 
            
            block_size = int(len(force_array) // 20 + 1)
            # print(f"The number of force terms ({len(force_array)}) is larger than 25. Will use blocks of {block_size} to have ", end="")
            force_array = make_force_terms_blocks(force_array, block_size)
            # print(f"{len(force_array)} force terms.")

        #Define constants and initialise arrays
        n_forces = force_array.shape[0]
        avr_sd_fes_prog = np.zeros(n_bootstrap) if get_progression else None
        FES_avr = np.zeros(nbins_yx)
        M2 = np.zeros(nbins_yx)
        
        # Find FES of the complete force array
        [PD, Force_x, Force_y] = patch_forces(force_array)
        FES_0 = FFT_intg_2D(Force_x, Force_y, min_grid=min_grid, max_grid=max_grid, periodic=periodic)
            
        # find cutoff array and averaging denominator
        cutoff = np.ones(nbins_yx)
        if PD_cutoff is not None: cutoff = np.where(PD > PD_cutoff, cutoff, 0)
        if FES_cutoff is not None: cutoff = np.where(FES_0 < FES_cutoff, cutoff, 0) 
        if use_VNORM: averaging_denominator = (np.sum(cutoff)**2) /np.prod(nbins_yx)
        else: averaging_denominator = np.sum(cutoff) 

        for iteration in range(n_bootstrap):
                
            #Randomly choose forward forces and backward forces and save to force array
            np.random.seed()
            random_sample_index = np.random.choice(n_forces, size=n_forces)      
            force = list(force_array[random_sample_index])    
        
            #Patch forces
            [PD, Force_x, Force_y] = patch_forces(force)
    
            #Calculate FES. if there is a FES_cutoff, find cutoff. 
            FES = FFT_intg_2D(Force_x, Force_y, min_grid=min_grid, max_grid=max_grid, periodic=periodic)
    
            # calculate standard devaition using Welford’s method
            delta = FES - FES_avr
            FES_avr += delta/(iteration+1)
            delta2 = FES - FES_avr
            M2 += delta*delta2
            
            if iteration > 0:
                sd_fes = np.sqrt(M2 /iteration)  # Bessel correction is usually (iteration-1) but we start from 0
                
                if PD_cutoff is not None or FES_cutoff is not None: sd_fes *= cutoff
                
                if get_progression or print_progress: avr_sd_fes = np.sum(sd_fes)/averaging_denominator
                
                if get_progression: avr_sd_fes_prog[iteration] = avr_sd_fes
            
                #print progress
                if print_progress and (iteration+1) % (n_bootstrap/10) == 0: print(f"i={iteration+1}/{n_bootstrap} : St. dev. = {round(avr_sd_fes,3)}")    
                
        if not get_progression and not print_progress: avr_sd_fes_prog = np.sum(sd_fes)/averaging_denominator
                
        return FES_0, FES_avr, sd_fes, avr_sd_fes_prog

    def weighted_bootstrapping_error(X, Y, force_array, n_bootstrap, periodic=[False, False], FES_cutoff=None, PD_cutoff=None, PD_limit=1E-10, use_VNORM=False, get_progression=False, print_progress=False):
        
        # Get grid variables
        nbins_yx = X.shape
        min_grid = [np.min(X), np.min(Y)]
        max_grid = [np.max(X), np.max(Y)]

        # Get the correct shape of the force array (Needs to be [PD, FX, FY], but sometimes it is [PD, PD2, FX, FY, OFV_X, OFV_Y])
        if force_array.shape[-3] == 6: force_array = force_array[:,[0,2,3]]
        if force_array.shape[-3] != 3: raise ValueError("force_array should have shape (n_forces, 3) or (n_forces, 6)")
        
        # Initialise     
        n_forces = force_array.shape[0]
        sd_fes_prog = np.zeros(n_bootstrap) if get_progression else None
        
        # Initialise arrays used for sums
        FES_sum = np.zeros(nbins_yx)
        sum_var_num = np.zeros(nbins_yx)
        sum_PD = np.zeros(nbins_yx)
        sum_PD2 = np.zeros(nbins_yx)
        
        # Patch and find FES without random sampling
        [PD, Force_x, Force_y] = patch_forces(force_array)
        FES_0 = FFT_intg_2D(Force_x, Force_y, min_grid=min_grid, max_grid=max_grid, periodic=periodic)
        
        # Find cutoff and averaging_denominator for error calculation    
        cutoff = np.ones(nbins_yx)
        if PD_cutoff is not None: cutoff = np.where(PD > PD_cutoff, cutoff, 0)
        if FES_cutoff is not None: cutoff = np.where(FES_0 < FES_cutoff, cutoff, 0) 
        if use_VNORM: averaging_denominator = (np.sum(cutoff)**2) /np.prod(nbins_yx)
        else: averaging_denominator = np.sum(cutoff) 

        for iteration in range(n_bootstrap):
                
            #Randomly choose forward forces and backward forces and save to force array
            np.random.seed()
            random_sample_index =  np.random.choice(n_forces, size=n_forces)      
            force = list(force_array[random_sample_index])    
        
            #Patch forces
            [PD, Force_x, Force_y] = patch_forces(force)
    
            #Calculate FES. if there is a FES_cutoff, find cutoff. 
            FES = FFT_intg_2D(Force_x, Force_y, min_grid=min_grid, max_grid=max_grid, periodic=periodic)
    
            # calculate standard devaition using Welford’s method
            FES_sum += FES
            sum_var_num += PD * np.square(FES - FES_0)
            sum_PD += PD
            sum_PD2 += np.square(PD)
            
            if iteration > 0:
                # calculate standard devaition 
                sum_PD_squared = np.square(sum_PD)
                diff_PD2 = sum_PD_squared - sum_PD2
                BC = np.divide(sum_PD_squared, diff_PD2, out=np.zeros_like(PD), where=diff_PD2>0)
                var_fes = np.divide(sum_var_num , sum_PD, out=np.zeros_like(PD), where=sum_PD>1) * BC
                sd_fes = np.sqrt(var_fes) * cutoff if PD_cutoff is not None or FES_cutoff is not None else np.sqrt(var_fes)
                
                if get_progression or print_progress: avr_sd_fes = np.sum(sd_fes)/averaging_denominator
                if get_progression: sd_fes_prog[iteration] = avr_sd_fes
            
                #print progress
                if print_progress and (iteration+1) % (n_bootstrap/10) == 0: print(f"i={iteration+1}/{n_bootstrap} : St. dev. = {round(avr_sd_fes,3)}")  
                
        FES_avr = FES_sum / n_bootstrap
        return FES_0, FES_avr, sd_fes, sd_fes_prog

    def bootstrapping_progression(X, Y, force_array, time_array=None, n_bootstrap=100, block_size=1, periodic=[False, False], FES_cutoff=None, PD_cutoff=None, PD_limit=1E-10, use_VNORM=False, show_plot=False):
        
        # get the force array and time array in blocks.
        if block_size > 1: 
            new_force_array = make_force_terms_blocks(force_array, block_size)
            
            if time_array is not None:
                                
                if not (len(force_array) == len(time_array) or len(force_array) == len(time_array)-1): 
                    raise ValueError("The length of time_array and force_array does not match. Please check the input")
                
                if len(force_array) == len(time_array): new_time_array = time_array[block_size-1::block_size]
                elif len(force_array) == len(time_array)-1: new_time_array = time_array[block_size::block_size]
                if new_time_array[-1] != time_array[-1]: new_time_array = np.append(new_time_array, time_array[-1])
                
                if len(new_time_array) != len(new_force_array): 
                    raise ValueError("The length of time_array and force_array does not match. Please check the code")
                
                time_array = new_time_array
            force_array = new_force_array
            
        elif time_array is not None and len(force_array) == len(time_array)-1: time_array = time_array[1:]

        sd_fes_evo = []
        for n_sim in range(1,len(force_array)+1):
        
            if n_sim > 2: 
                FES_0, FES_avr, sd_fes, sd_fes_prog = bootstrapping_error(X, Y, force_array[:n_sim], int(max(10,n_bootstrap*(n_sim/len(force_array)))), periodic, FES_cutoff, PD_cutoff, PD_limit, use_VNORM)
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
            plt.plot(time, mean, linewidth=1, color="red")
            plt.fill_between(time, mean - ste, mean + ste, color="red", alpha=0.3)
            # plt.ylim(min(mean)*0.9, max(mean)*1.1 )
            if plot_log: plt.yscale("log")
            plt.title(plot_title[0]); plt.xlabel(plot_xlabel[0]); plt.ylabel(plot_ylabel[0])

            if save_plot_path is not None: plt.savefig(save_plot_path)
            if plot: plt.show()
            
        if return_results: 
            if error_collection_2 is not None: return [time, mean, ste, mean_2, ste_2]
            else: return [time, mean, ste]

    def get_avr_error_prog(path_data=None, total_campaigns=50, time_budget=100, include_aad=True, simulation_type="", return_avr_prog=False,
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
            if include_aad: [t_i,err_i, aad_i] = load_pkl(f"{path_data}{sim_folder_prefix}{camp_id}/error_progression{camp_id}.pkl")
            else: [t_i,err_i] = load_pkl(f"{path_data}{sim_folder_prefix}{camp_id}/error_progression{camp_id}.pkl")
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

####  ---- Saving and Loading Data  --------------------------------------------------------  ####

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

for _functions_ in [1]:

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
                if input_array[ii][jj] == 0: output_array[ii][jj] = np.nan
                else: output_array[ii][jj] = input_array[ii][jj]
        return output_array

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
    
    def Gaus_fitting_to_fes_2D(X, Y, surface, max_filter_size=None, smoothening_filter=3, periodic=np.array([False, False])):
        
        """
        Fits a 2D Gaussian function to the given surface and returns the basin indices, basin surfaces, and parameter list.

        Parameters:
        - X (ndarray): The X-coordinates of the surface grid.
        - Y (ndarray): The Y-coordinates of the surface grid.
        - surface (ndarray): The 2D surface to fit the Gaussian to.
        - max_filter_size (int, optional): The size of the maximum filter. Defaults to None -> max_filter_size = min(surface.shape)//10.
        - smoothening_filter (int, optional): The size of the smoothening filter. Defaults to 3.
        - periodic (ndarray, optional): An array indicating whether the surface is periodic in the X and Y directions. Defaults to np.array([False, False]).

        Returns:
        - basin_indices (ndarray): The indices of the basins in the surface.
        - all_basin_surfaces (list): A list of basin surfaces defined on X, Y.
        - parameter_list (list): A list of parameters for each fitten Gaussian (height, centre_x, centre_y).

        """   

        # if periodic, add periodic extension to the surface
        if periodic[0] or periodic[1]: 
            height_org, width_org = surface.shape
            x_org, y_org = np.array(X[0,:]), np.array(Y[:,0])
            lx, ly = X[0,-1]-X[0,0], Y[-1,0]-Y[0,0]
        
            if periodic[0]:
                X = np.concatenate([X[:, width_org//2:]-lx, X, X[:, :width_org//2]+lx], axis=1)
                Y = np.concatenate([Y[:, width_org//2:]   , Y, Y[:, :width_org//2]]   , axis=1)
                surface = np.concatenate([surface[:, width_org//2:], surface, surface[:, :width_org//2]], axis=1)
                
            if periodic[1]:
                X = np.concatenate([X[height_org//2:, :]   , X, X[:height_org//2]   ], axis=0)
                Y = np.concatenate([Y[height_org//2:, :]-ly, Y, Y[:height_org//2]+ly], axis=0)
                surface = np.concatenate([surface[height_org//2:, :], surface, surface[:height_org//2, :]], axis=0)
        
        # set parameters
        if max_filter_size is None: max_filter_size = (min(surface.shape)//10)
        # smoothen the surface to aviod detecting small basins
        if smoothening_filter is not None and smoothening_filter > 0: surface = gaussian_filter(surface, smoothening_filter)
        
        # find the grid for x- and y-direction 
        x, y, xy = X[0,:], Y[:,0], (X[0,:], Y[:,0])
        lx, ly, dx, dy = x[-1]-x[0], y[-1]-y[0], x[1]-x[0], y[1]-y[0]    
        height, width = surface.shape
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]#, (-1,-1), (-1,1), (1,-1), (1,1)]# Directions for moving in the grid (N, S, E, W)
        
        ### --- indentify the minima --- ###
        
        # Invert the surface to find basins as peaks
        inverted_surface = surface.max() - surface
        
        # Use maximum filter to find local minima (or maxima in the inverted surface)
        max_filtered = maximum_filter(inverted_surface, size=max_filter_size)
        minima = (inverted_surface == max_filtered)
        
        # Remove minima on the border
        minima[0, :] = minima[-1, :] = False
        minima[:, 0] = minima[:, -1] = False
        
        # Remove minima that are global above 95% of golbal maxima
        minima[surface > np.max(surface)*0.95] = False
        
        # if periodic, remove minima that are in the periodic extension (below min_x, min_y or above max_x, max_y)
        if periodic[0]: minima[X < x_org[0]] = False; minima[X > x_org[-1]] = False
        if periodic[1]: minima[Y < y_org[0]] = False; minima[Y > y_org[-1]] = False
            
        # get the indices of the minima (which elements are True)
        basin_indices = np.argwhere(minima)
        # print(f"END OF BASIN DETECTION\n{len(basin_indices)=}, {basin_indices = }\nContinuing with the basin area detection")
        
        ### --- indentify the basin area of each basin--- ###
        
        all_basin_surfaces = []
        for i_basin, basin_index in enumerate(basin_indices):
            
            # get the center of the basin, and create a queue with the center
            y_center, x_center = basin_index
            queue = [(y_center, x_center)]# Queue for flood fill algorithm (starting with the center)
            # make array of visited points of the grid and set the center as visited
            visited = np.zeros_like(surface, dtype=bool) # To keep track of visited points
            visited[y_center, x_center] = True
            # create a basin surface with np.nan and set the center to the value of the surface. Accepted points will be set to the value of the surface, the rest will be np.nan.
            basin_surface = np.full_like(surface, np.nan) # Create a result surface initialized with np.nan
            basin_surface[y_center, x_center] = surface[y_center, x_center]
    
            # flood fill algorithm: loop through the queue to check points: go through the neighbors and add the neighbors to [basin_surface, queue, and visited] if neighbor_value > current_value.
            while queue:
                y_center_i, x_center_i = queue.pop(0)
                current_value = surface[y_center_i, x_center_i] # get the value of the current point. Then check the neighbors and accept those if they are larger than the current value.

                for d_y, d_x in directions:
                    ny, nx = y_center_i + d_y, x_center_i + d_x
                    if ny < 0 or ny >= height or nx < 0 or nx >= width or visited[ny, nx]: continue # Skip if out of bounds or already visited

                    neighbor_value = surface[ny, nx]
                    if neighbor_value > current_value:
                        queue.append((ny, nx))
                        basin_surface[ny, nx] = surface[ny, nx]
                        visited[ny, nx] = True
            
            # flip around the basin surface and set minima to zero (and basin center to the maximum value)
            basin_surface = np.nanmax(basin_surface) - basin_surface            
            # make all nan values zero
            basin_surface[np.isnan(basin_surface)] = 0.0
            
            # check if basin area is at least 1% of the total area. If not, remove the basin    
            basin_area = np.count_nonzero(basin_surface) # count all non_zero values in the basin surface
            if basin_area > width*height/100: all_basin_surfaces.append(basin_surface) # append the basin surface to the list of all basin surfaces
            else: basin_indices = np.delete(basin_indices, i_basin, axis=0)
        
        # print(f"\nEND OF BASIN AREA DETECTION\n{len(basin_indices)=}, {len(all_basin_surfaces) = }\nContinuing with the gaussian fitting")
            
        #### --- fitting the gaussian to the basin surface --- ####    
        assert len(basin_indices) == len(all_basin_surfaces), "The number of basin indices and basin surfaces should be the same. Please check the code."
        
        parameter_list = []
            
        for i_basin in range(len(basin_indices)):
            
            # get current basin        
            basin_surface, (y_center, x_center) = all_basin_surfaces[i_basin], basin_indices[i_basin]
            basin_surface_along_xy = np.array([basin_surface[y_center, :], basin_surface[:, x_center]]).ravel()
            
            # def the 2D gaussian function() with the center in the middle of the basin (x_center, y_center)
            def gauss_along_xy_centered(xy, height, sigma_x, sigma_y): return np.array([height * np.exp(-((xy[0] - x[x_center]) ** 2) / (2 * sigma_x ** 2)), height * np.exp(-((xy[1] - y[y_center]) ** 2) / (2 * sigma_y ** 2))]).ravel()   
            
            # bounds / init[      height              ,sigma_x, sigma_y] # bound and initial guess for the fitting
            lower_bound  = [0                         , dx    , dy     ]
            upper_bound  = [np.nanmax(basin_surface)  , lx/4  , ly/4   ]
            initial_guess= (np.nanmax(basin_surface)/2, lx/25 , ly/25  )  
            
            # fit the gaussian function() paremeters to the basin surface
            parameter_list.append(curve_fit(gauss_along_xy_centered, xy, basin_surface_along_xy, p0=initial_guess, bounds=(lower_bound, upper_bound))[0]) 
            
            # reshape basin surface to the original shape
            all_basin_surfaces[i_basin] = basin_surface.reshape(height, width)       
            
            # if periodic, correct the basin indices (center of the basin) and the basin surface
            if periodic[0]: basin_indices[i_basin][1], all_basin_surfaces[i_basin] = np.where(x_org == x[x_center])[0][0], all_basin_surfaces[i_basin][:, -(width_org+width_org//2):-(width_org//2)]
            if periodic[1]: basin_indices[i_basin][0], all_basin_surfaces[i_basin] = np.where(y_org == y[y_center])[0][0], all_basin_surfaces[i_basin][-(height_org+height_org//2):-(height_org//2), :]
                
        return basin_indices, all_basin_surfaces, parameter_list
    
    def Find_basins_of_fes_2D(X, Y, surface, max_filter_size=None, smoothening_filter=3, periodic=np.array([False, False])):
        
        """
        Find the basins of a given surface and returns the basin indices, basin surfaces.

        Parameters:
        - X (ndarray): The X-coordinates of the surface grid.
        - Y (ndarray): The Y-coordinates of the surface grid.
        - surface (ndarray): The 2D surface to fit the Gaussian to.
        - max_filter_size (int, optional): The size of the maximum filter. Defaults to None -> max_filter_size = min(surface.shape)//10.
        - smoothening_filter (int, optional): The size of the smoothening filter. Defaults to 3.
        - periodic (ndarray, optional): An array indicating whether the surface is periodic in the X and Y directions. Defaults to np.array([False, False]).

        Returns:
        - basin_indices (ndarray): The indices of the basins in the surface.
        - all_basin_surfaces (list): A list of basin surfaces defined on X, Y.

        """   

        # if periodic, add periodic extension to the surface
        if periodic[0] or periodic[1]: 
            height_org, width_org = surface.shape
            x_org, y_org = np.array(X[0,:]), np.array(Y[:,0])
            lx, ly = X[0,-1]-X[0,0], Y[-1,0]-Y[0,0]
        
            if periodic[0]:
                X = np.concatenate([X[:, width_org//2:]-lx, X, X[:, :width_org//2]+lx], axis=1)
                Y = np.concatenate([Y[:, width_org//2:]   , Y, Y[:, :width_org//2]]   , axis=1)
                surface = np.concatenate([surface[:, width_org//2:], surface, surface[:, :width_org//2]], axis=1)
                
            if periodic[1]:
                X = np.concatenate([X[height_org//2:, :]   , X, X[:height_org//2]   ], axis=0)
                Y = np.concatenate([Y[height_org//2:, :]-ly, Y, Y[:height_org//2]+ly], axis=0)
                surface = np.concatenate([surface[height_org//2:, :], surface, surface[:height_org//2, :]], axis=0)
        
        # set parameters
        if max_filter_size is None: max_filter_size = (min(surface.shape)//10)
        # smoothen the surface to aviod detecting small basins
        if smoothening_filter is not None and smoothening_filter > 0: surface = gaussian_filter(surface, smoothening_filter)
        
        # find the grid for x- and y-direction 
        x, y, xy = X[0,:], Y[:,0], (X[0,:], Y[:,0])
        lx, ly, dx, dy = x[-1]-x[0], y[-1]-y[0], x[1]-x[0], y[1]-y[0]    
        height, width = surface.shape
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]#, (-1,-1), (-1,1), (1,-1), (1,1)]# Directions for moving in the grid (N, S, E, W)
        
        ### --- indentify the minima --- ###
        
        # Invert the surface to find basins as peaks
        inverted_surface = surface.max() - surface
        
        # Use maximum filter to find local minima (or maxima in the inverted surface)
        max_filtered = maximum_filter(inverted_surface, size=max_filter_size)
        minima = (inverted_surface == max_filtered)
        
        # Remove minima on the border
        minima[0, :] = minima[-1, :] = False
        minima[:, 0] = minima[:, -1] = False
        
        # Remove minima that are global above 95% of golbal maxima
        minima[surface > np.max(surface)*0.95] = False
        
        # if periodic, remove minima that are in the periodic extension (below min_x, min_y or above max_x, max_y)
        if periodic[0]: minima[X < x_org[0]] = False; minima[X > x_org[-1]] = False
        if periodic[1]: minima[Y < y_org[0]] = False; minima[Y > y_org[-1]] = False
            
        # get the indices of the minima (which elements are True)
        basin_indices = np.argwhere(minima)
        # print(f"END OF BASIN DETECTION\n{len(basin_indices)=}, {basin_indices = }\nContinuing with the basin area detection")
        
        ### --- indentify the basin area of each basin--- ###
        
        all_basin_surfaces = []
        for i_basin, basin_index in enumerate(basin_indices):
            
            # get the center of the basin, and create a queue with the center
            y_center, x_center = basin_index
            queue = [(y_center, x_center)]# Queue for flood fill algorithm (starting with the center)
            # make array of visited points of the grid and set the center as visited
            visited = np.zeros_like(surface, dtype=bool) # To keep track of visited points
            visited[y_center, x_center] = True
            # create a basin surface with np.nan and set the center to the value of the surface. Accepted points will be set to the value of the surface, the rest will be np.nan.
            basin_surface = np.full_like(surface, np.nan) # Create a result surface initialized with np.nan
            basin_surface[y_center, x_center] = surface[y_center, x_center]
    
            # flood fill algorithm: loop through the queue to check points: go through the neighbors and add the neighbors to [basin_surface, queue, and visited] if neighbor_value > current_value.
            while queue:
                y_center_i, x_center_i = queue.pop(0)
                current_value = surface[y_center_i, x_center_i] # get the value of the current point. Then check the neighbors and accept those if they are larger than the current value.

                for d_y, d_x in directions:
                    ny, nx = y_center_i + d_y, x_center_i + d_x
                    if ny < 0 or ny >= height or nx < 0 or nx >= width or visited[ny, nx]: continue # Skip if out of bounds or already visited

                    neighbor_value = surface[ny, nx]
                    if neighbor_value > current_value:
                        queue.append((ny, nx))
                        basin_surface[ny, nx] = surface[ny, nx]
                        visited[ny, nx] = True
            
            # flip around the basin surface and set minima to zero (and basin center to the maximum value)
            basin_surface = np.nanmax(basin_surface) - basin_surface            
            # make all nan values zero
            basin_surface[np.isnan(basin_surface)] = 0.0
            
            # check if basin area is at least 1% of the total area. If not, remove the basin    
            basin_area = np.count_nonzero(basin_surface) # count all non_zero values in the basin surface
            if basin_area > width*height/100: all_basin_surfaces.append(basin_surface) # append the basin surface to the list of all basin surfaces
            else: basin_indices = np.delete(basin_indices, i_basin, axis=0)
        
        # print(f"\nEND OF BASIN AREA DETECTION\n{len(basin_indices)=}, {len(all_basin_surfaces) = }\nContinuing with the gaussian fitting")
            
        #### --- reshape basin surface --- ####    
        assert len(basin_indices) == len(all_basin_surfaces), "The number of basin indices and basin surfaces should be the same. Please check the code."
        
            
        for i_basin in range(len(basin_indices)):
            
            # get current basin        
            basin_surface, (y_center, x_center) = all_basin_surfaces[i_basin], basin_indices[i_basin]
           
            # reshape basin surface to the original shape
            all_basin_surfaces[i_basin] = basin_surface.reshape(height, width)       
            
            # if periodic, correct the basin indices (center of the basin) and the basin surface
            if periodic[0]: basin_indices[i_basin][1], all_basin_surfaces[i_basin] = np.where(x_org == x[x_center])[0][0], all_basin_surfaces[i_basin][:, -(width_org+width_org//2):-(width_org//2)]
            if periodic[1]: basin_indices[i_basin][0], all_basin_surfaces[i_basin] = np.where(y_org == y[y_center])[0][0], all_basin_surfaces[i_basin][-(height_org+height_org//2):-(height_org//2), :]
                
        return basin_indices, all_basin_surfaces

    def gaus_2d_from_xy(xy, height, center_x, center_y, width_x, width_y):
        gaus_x = np.exp( - np.square(xy[0] - center_x) / (2 * width_x**2)) * height
        gaus_y = np.exp( - np.square(xy[1] - center_y) / (2 * width_y**2))
        return np.outer(gaus_y, gaus_x)

    def find_linear_transition_path(initial_positions, final_positions, n_images=10):
        return np.linspace(initial_positions, final_positions, n_images)

    def spaceout_transiton_path(path):
        # Compute the cumulative distance along the path and the target distances (assuming evenly spaced points)
        distances = np.insert(np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)), 0, 0)
        target_distances = np.linspace(0, distances[-1], len(path))
        
        # Create a new path by interpolating points at the target distances
        new_path = np.zeros_like(path)
        new_path[0], new_path[-1] = path[0], path[-1]
        
        for i in range(1, len(path)-1):
            
            # Find the segment where the target distance lies
            target_distance = target_distances[i]
            idx = np.searchsorted(distances, target_distance)
            
            # Interpolate between the two points to find the new point. If the target distance is outside the path, use the closest point
            if idx == 0: new_path[i] = path[0]
            elif idx >= len(path): new_path[i] = path[-1]
            else: new_path[i] = path[idx-1] + (path[idx] - path[idx-1]) * ((target_distance - distances[idx-1]) / (distances[idx] - distances[idx-1]))

        return new_path
        
    def find_transition_path(FES, x, y, initial_positions, final_positions, n_images=15, force_convergence_goal=0.2, max_steps=1000, step_size=0.01, print_info=False):

        # Calculate the gradients of the FES
        grad_y, grad_x = np.gradient(FES, y[1]-y[0], x[1]-x[0])

        # Initialize the path as a linear interpolation between initial and final positions
        path = np.linspace(initial_positions, final_positions, n_images)

        for step in range(max_steps):
            max_force = 0
            new_path = np.copy(path)

            # Update each image (except the first and last)
            for i in range(1, n_images - 1):

                # Find the forces at the current position
                ix, iy = np.abs(x - path[i][0]).argmin(), np.abs(y - path[i][1]).argmin()
                forces = - np.array([grad_x[iy, ix], grad_y[iy, ix]])

                # Get the tangential component of the force (only allows the point to move perpendicular to the path direction)
                tangent = (path[i+1] - path[i-1]) / np.linalg.norm(path[i+1] - path[i-1])
                forces -= np.dot(forces, tangent) * tangent

                # Update the position and max force
                new_path[i] = path[i] + forces * step_size  # Step size can be adjusted
                max_force = max(max_force, np.linalg.norm(forces))

            # Reparameterize to maintain even spacing
            if (step+1) % 10 == 0: path = spaceout_transiton_path(new_path)
            else: path = new_path

            # Check for convergence
            if max_force < force_convergence_goal: 
                if print_info: print(f'Converged in {step+1} steps.')
                return path

        if print_info: print(f'Reached {step+1} steps. ({max_force = } > {force_convergence_goal = })')
        return path

    def interpolate_path(path, num_points=1000):
        distances = np.insert(np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)), 0, 0)
        interpolator = interp1d(distances, path, axis=0, kind='linear')
        return interpolator(np.linspace(0, distances[-1], num_points))
    
    def create_valley_surface(X, Y, path=None, FES=None, FES_gaus_filter=None, periodic=[False, False], path_type="transition", valley_depth=15, valley_width=None, sharpness_exponent=6, n_images=15, n_points_interp=100, force_convergence_goal=0.2, max_steps=1000, step_size=None, print_info=False, return_path=False):
        
        # Set default values if not specified, and print info
        if valley_width is None: valley_width = ((X[0, -1] - X[0, 0]) / 10 + (Y[-1, 0] - Y[0, 0]) / 10 ) / 2  # 
        if step_size is None: step_size = valley_width / 20
        if print_info: print(f"{valley_width = }, {step_size = }")
        
        # if path is not provided, use the FES to find a path between the energy minima
        if path is None: 
            if FES is None: raise ValueError("Either path or FES must be provided.")
            
            filter_mode = "wrap" if (periodic[0] or periodic[1]) else "reflect"
            if FES_gaus_filter is not None: FES = gaussian_filter(FES, sigma=FES_gaus_filter, mode=filter_mode)
                        
            # Find the basins of the FES
            basin_indices, _ = Find_basins_of_fes_2D(X, Y, FES)
            if len(basin_indices) < 2: raise ValueError("Could not find multiple basins in the FES.")
            
            i_pos = [X[basin_indices[0][0], basin_indices[0][1]], Y[basin_indices[0][0], basin_indices[0][1]]]
            f_pos = [X[basin_indices[1][0], basin_indices[1][1]], Y[basin_indices[1][0], basin_indices[1][1]]]
            
            if len(basin_indices) > 2: print(f"Warning: More than two basins found in the FES. Using the first two basins: {i_pos = }, {f_pos = }")
            
            # find the path between the basins
            if path_type == "transition": path = find_transition_path(FES, X[0], Y[:,0], i_pos, f_pos, n_images=n_images, force_convergence_goal=force_convergence_goal, max_steps=max_steps, step_size=step_size, print_info=print_info)
            elif path_type == "linear": path = find_linear_transition_path(i_pos, f_pos)
            else: raise ValueError("Unknown path type. Use 'transition' or 'linear'.")
        
        # make sure path is a numpy array
        path = np.array(path)
        
        #### ---- Create the valley surface from the path
        # Interpolate the path to get more points
        if n_points_interp is not None and n_points_interp > 0 and len(path) < n_points_interp: path = interpolate_path(path, num_points=n_points_interp)
        # Compute arrray of all grid points (shape = ( n_grid_points, 2))   
        points = np.column_stack((X.ravel(), Y.ravel()))
        # Compute distances from each grid point to each point if the path (shape = (n_grid_points, n_path_points))
        distances = cdist(points, path)
        # Find the distance of each grid point to the closest point on the path
        min_distances = np.min(distances, axis=1)
        
        # Define a function that creates a valley. If min_distances is small, the drid point is close to the path, corresponding to a higher depth. Grid points far from the path have a high min distance and a low/zero depth.
        valley_function = lambda d: - valley_depth * np.exp(-(d / valley_width)**sharpness_exponent)
        
        # Find the valley depth as function of the distance to the path and reshape it to the shape of the grid
        if return_path: return valley_function(min_distances).reshape(X.shape), path
        return valley_function(min_distances).reshape(X.shape)

####  ---- Print progress  --------------------------------------------------------  ####

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
        progress = ( iteration / total)
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
        
        if progress < 1: print(f'\r|{arrow}{spaces}| {int(progress * 100)}% | {variable_name}: {variable:.4}ns | Time left: {format_seconds(time_to_end)} |{time_at_end}    ', end='', flush=True)
        else: print(f'\r| {int(progress * 100)}% | {variable_name}: {variable:.4}ns | Total time: {format_seconds(time_to_now)} | Finished at {datetime.now().strftime("%H:%M:%S")}                                                         ')

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
        
####  ---- Functions for plotting  --------------------------------------------------------  ####

for _plot_functions_ in [0]:

    def find_contour_levels(plot_array, min_level=None, max_level=None, error_name=None, print_info=False):
        
        if print_info and error_name is not None: print("\n ~~~ ", error_name, " ~~~ ")
        if error_name == "FES":
            
            lowest_value = min_level if min_level is not None else np.min(plot_array[np.nonzero(plot_array)]) // 1
            heighest_value = max_level if max_level is not None else np.max(plot_array)
            if heighest_value < 13: n_level_fes = heighest_value + 1
            else: 
                n_level_fes = np.ceil(heighest_value/10) 
                if (heighest_value % n_level_fes) != 0: heighest_value = heighest_value + (n_level_fes - heighest_value % n_level_fes)
                n_level_fes = heighest_value/n_level_fes + 1
            levels = np.linspace(0, heighest_value, int(n_level_fes))
            if print_info: print("lowest_value: ", lowest_value, " || heighest_value: ", heighest_value, "\nlevels: ", levels)
            return levels
        
        elif error_name == "PD":
            lowest_value = min_level if min_level is not None else np.min(plot_array[np.nonzero(plot_array)])
            heighest_value = max_level if max_level is not None else np.max(plot_array)
            levels = np.logspace(np.log10(lowest_value), np.log10(heighest_value), 10)
            if print_info: print("lowest_value: ", lowest_value, " || heighest_value: ", heighest_value, "\nlevels: ", levels)
            return levels
        
        else:
            lowest_value = min_level if min_level is not None else np.min(plot_array[np.nonzero(plot_array)])
            heighest_value = max_level if max_level is not None else np.max(plot_array)
            levels = np.linspace(lowest_value, heighest_value, 10)
            if print_info: print("lowest_value: ", lowest_value, " || heighest_value: ", heighest_value ,"\nlevels: ", levels)
            return levels

    def plot_hills_trajectory(hills, nhills=-1):
            Gamma = (hills[1,6] -1)/hills[1,6] if len(hills) == 7 else 1
            plt.figure(figsize=(14, 3))
            plt.subplot(1, 3, 1); plt.scatter(hills[:nhills,1], hills[:nhills,2], c=hills[:nhills,5]*Gamma, cmap='coolwarm', s=0.5, alpha=0.5); plt.colorbar(label="Height [kJ/mol]"); plt.xlabel("x"); plt.ylabel("y"); plt.title("Hills in x-y"); 
            plt.subplot(1, 3, 2); plt.scatter(hills[:nhills,0], hills[:nhills,1], c=hills[:nhills,5]*Gamma, cmap='coolwarm', s=0.5, alpha=0.5); plt.colorbar(label="Height [kJ/mol]"); plt.xlabel("time [ps]"); plt.ylabel("x"); plt.title("Hills in x");
            plt.subplot(1, 3, 3); plt.scatter(hills[:nhills,0], hills[:nhills,2], c=hills[:nhills,5]*Gamma, cmap='coolwarm', s=0.5, alpha=0.5); plt.colorbar(label="Height [kJ/mol]"); plt.xlabel("time [ps]"); plt.ylabel("y"); plt.title("Hills in y");
            plt.tight_layout(); plt.show()
   
    def plot_position_trajectory(positions, npos=-1, plot_stride=10):
        plt.figure(figsize=(14, 3))
        plt.subplot(1, 3, 1); plt.scatter(positions[:npos][::plot_stride,1], positions[:npos][::plot_stride,2], c='b', s=3, alpha=0.1); plt.xlabel("x"); plt.ylabel("y"); plt.title("Positions in x-y"); 
        plt.subplot(1, 3, 2); plt.scatter(positions[:npos][::plot_stride,0], positions[:npos][::plot_stride,1], c='b', s=3, alpha=0.1); plt.xlabel("Nr. Position"); plt.ylabel("x"); plt.title("Positions in x");
        plt.subplot(1, 3, 3); plt.scatter(positions[:npos][::plot_stride,0], positions[:npos][::plot_stride,2], c='b', s=3, alpha=0.1); plt.xlabel("Nr. Position"); plt.ylabel("y"); plt.title("Positions in y");
        plt.tight_layout(); plt.show()
        
    def plot_contour_periodic_x9(X, Y, Z, levels=None, cmap='coolwarm', title="", cbar_label="", x_label="", y_label=""):
        
        # make xx array that has periodic extensions of the grid in all directions
        xx = np.concatenate([X-2*np.pi, X, X+2*np.pi], axis=1)
        xx = np.concatenate([xx, xx, xx], axis=0)
        
        yy = np.concatenate([Y, Y, Y], axis=1)
        yy = np.concatenate([yy-2*np.pi, yy, yy+2*np.pi], axis=0)
        
        zz = np.concatenate([Z, Z, Z], axis=1)
        zz = np.concatenate([zz, zz, zz], axis=0)
            
        plt.contourf(xx, yy, zz, cmap=cmap, levels=levels); plt.colorbar(label=cbar_label); plt.xlabel(x_label); plt.ylabel(y_label); plt.title(title); plt.show()
        
    def plot_3D_plotly(X,Y,Z, range_min=None, range_max=None, width=800, height=400, color=None, opacity=0.5, 
                    title=None, x_label="x", y_label="y", z_label="Energy [kJ/mol]", colorbar_label="Free Energy [kJ/mol]"):

        Z1 = np.where(Z < range_max, Z, range_max) if range_max is not None else np.array(Z)
        Z1 = np.where(Z1 > range_min, Z1, range_min) if range_min is not None else np.array(Z1)
        if range_min is None: range_min = np.nanmin(Z)
        if range_max is None: range_max = np.nanmax(Z1)
        # if color is None: color = "viridis"

        fig = make_subplots( rows=1, cols=1, specs=[[{'type': 'surface'}]])
        fig.add_trace(go.Surface(x=X, y=Y, z=Z1, colorscale=color, opacity=opacity, colorbar=dict(title=colorbar_label)), row=1, col=1)
        fig.update_traces(contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True))
        fig.update_layout(title=title, width=width, height=height, scene=dict(xaxis_title=x_label, yaxis_title=y_label, zaxis_title=z_label), margin=dict(l=0, r=0, b=0, t=0) )
        fig.update_scenes(aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.5), zaxis=dict(range=[range_min, range_max]), row=1, col=1)
        fig.update_layout(font=dict(family="Helvetica, Arial, sans-serif",  size=12, color="black" ) )
        print("ready to show")
        fig.show()

    def plot_3D_plotly_x(X,Y,Z, range_min=None, range_max=None, width=None, height=None, opacity=None, 
                        title=None, subtitle=None, x_label=None, y_label=None, z_label=None, colorbar_label=None):

        for _get_params_ready_ in [1]:
            if type(Z) is list: n_plots = len(Z)
            elif len(Z.shape) == 3: n_plots = Z.shape[0]
            elif len(Z.shape) == 2: n_plots = 1
            else: raise ValueError("Z must be either a list of 2D arrays or an array of 2D arrays or just a 2D array")    

            if type(X) is list: assert len(X) == n_plots
            elif len(X.shape) == 3: assert X.shape[0] == n_plots
            elif len(X.shape) == 2: X = [X for _ in range(n_plots)]
            else: raise ValueError("X shape not recognized")		
        
            if type(Y) is list: assert len(Y) == n_plots
            elif len(Y.shape) == 3: assert Y.shape[0] == n_plots
            elif len(Y.shape) == 2: Y = [Y for _ in range(n_plots)]
            else: raise ValueError("Y shape not recognized")
        
            if range_min is None: range_min = [np.nanmin(Z[i]) for i in range(n_plots)]
            elif type(range_min) is not list and type(range_min) is float: range_min = [range_min for _ in range(n_plots)]
            assert len(range_min) == n_plots
        
            if range_max is None: range_max = [np.nanmax(Z[i]) for i in range(n_plots)]
            elif type(range_max) is not list and type(range_max) is float: range_max = [range_max for _ in range(n_plots)]
            assert len(range_max) == n_plots
            
            Z_plot = []
            for i in range(n_plots): 
                if type(Z[i]) is list: 
                    Z_i_plot = []
                    for Z_j in Z[i]:
                        if range_max[i] is not None: Z_i_plot.append(np.where(Z_j < range_max[i], Z_j, range_max[i]))
                        else: Z_i_plot.append(np.array(Z_j))
                    Z_plot.append(Z_i_plot)
                else:
                    if range_max[i] is not None: Z_plot.append(np.where(Z[i] < range_max[i], Z[i], range_max[i]))
                    else: Z_plot.append(np.array(Z[i]))
            
            if width is None: width = 500 * n_plots
            if height is None: height = 500
        
            if opacity is None: opacity = [None for _ in range(n_plots)]
            elif type(opacity) is not list and type(opacity) is float: opacity = [opacity for _ in range(n_plots)]
            
            if subtitle is None: subtitle = [None for _ in range(n_plots)]
            elif type(subtitle) is not list and type(subtitle) is str: subtitle = [subtitle for _ in range(n_plots)]
            assert len(subtitle) == n_plots
            
            if x_label is None: x_label = [None for _ in range(n_plots)]
            elif type(x_label) is not list and type(x_label) is str: x_label = [x_label for _ in range(n_plots)]
            assert len(x_label) == n_plots
            
            if y_label is None: y_label = [None for _ in range(n_plots)]
            elif type(y_label) is not list and type(y_label) is str: y_label = [y_label for _ in range(n_plots)]
            assert len(y_label) == n_plots
            
            if z_label is None: z_label = [None for _ in range(n_plots)]
            elif type(z_label) is not list and type(z_label) is str: z_label = [z_label for _ in range(n_plots)]
            assert len(z_label) == n_plots
            
            if colorbar_label is None: colorbar_label = [None for _ in range(n_plots)]
            elif type(colorbar_label) is not list and type(colorbar_label) is str: colorbar_label = [colorbar_label for _ in range(n_plots)]
            assert len(colorbar_label) == n_plots
            
            fig_specs = [[{'type': 'surface'} for _ in range(n_plots)]]
            
        fig = make_subplots( rows=1, cols=n_plots, specs=fig_specs , subplot_titles=subtitle  )
        for i in range(n_plots): 
            # if type(Z_plot[i]) is list: 
            #     j=0
            #     fig.add_trace(go.Surface(x=X[i], y=Y[i], z=Z_plot[i][j], opacity=opacity[i], colorbar=dict(title=colorbar_label[i], orientation='h', xanchor='center', x=(i*2+1)/(2*n_plots), y=-0.2, len=1/(n_plots+1), lenmode="fraction")), row=1, col=i+1)
            #     if len(Z_plot[i]) > 1: [fig.add_trace(go.Surface(x=X[i], y=Y[i], z=Z_plot[i][j], colorscale=[[0, 'grey'], [1, 'grey']], opacity=opacity[i]*0.75, showscale= i % 2 == 0, colorbar=dict()), row=1, col=i+1) for j in range(1,len(Z_plot[i])) ]
            # else:
                # fig.add_trace(go.Surface(x=X[i], y=Y[i], z=Z_plot[i], opacity=opacity[i], colorbar=dict(title=colorbar_label[i], orientation='h', xanchor='center', x=(i*2+1)/(2*n_plots), y=-0.2, len=1/(n_plots+1), lenmode="fraction")), row=1, col=i+1)
            fig.add_trace(go.Surface(x=X[i], y=Y[i], z=Z_plot[i], opacity=opacity[i], colorbar=dict(title=colorbar_label[i], orientation='h', xanchor='center', x=(i*2+1)/(2*n_plots), y=-0.2, len=1/(n_plots+1), lenmode="fraction")), row=1, col=i+1)
            fig.update_layout({ f"scene{i+1}": dict(xaxis_title=x_label[i], yaxis_title=y_label[i], zaxis_title=z_label[i], aspectmode="manual", 
                                                    aspectratio=dict(x=1, y=1, z=0.5), zaxis=dict(range=[range_min[i], range_max[i]]) ) })
            
        fig.update_traces(contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True))
        fig.update_layout(title=title, width=1500, height=500)#, margin=dict(t=50, b=100, l=50, r=50))
        fig.update_layout(font=dict(family="Helvetica, Arial, sans-serif",  size=12, color="black" ) )
        fig.show()

    def plot_multiple_error_prog_and_maps(MFI, error_types=["AAD", "ABS", "FES_st_dev"], include_Aofe_prog=True):
        
        # set up the error types to plot
        plt_index, plt_labels, plt_titles = [], [], []
        for error_type in error_types:
            if error_type not in ["Aofe", "AAD", "ABS", "FES_st_dev"]:
                print(f"Error type {error_type} not available. Only the following error types are available: \"Aofe\", \"AAD\", \"ABS\", \"FES_st_dev\"")
                continue
                
            if error_type == "Aofe": 
                plt_index.append(2)
                plt_labels.append("Aofe")
                plt_titles.append("Avr. on-the-fly error of Force")
            elif error_type == "AAD": 
                if hasattr(MFI, "aad_index"): 
                    plt_index.append(MFI.aad_index)
                    plt_labels.append("AAD")
                    plt_titles.append("Avr. abs. deviation of FES")
                else: print("AAD not calculated")
            elif error_type == "ABS":
                if hasattr(MFI, "abs_error_index"): 
                    plt_index.append(MFI.abs_error_index)
                    plt_labels.append("ABS")
                    plt_titles.append("Avr. bootstrapped error")
                else: print("ABS not calculated")
            elif error_type == "FES_st_dev": 
                if hasattr(MFI, "FES_st_dev_index"): 
                    plt_index.append(MFI.FES_st_dev_index)
                    plt_labels.append("FES_st_dev")
                    plt_titles.append("Avr. FES standard deviation")
                else: print("FES_st_dev not calculated")
                    
        if len(plt_index) == 0: 
            print("No error type available")
            return
        assert len(plt_index) == len(plt_labels), "The number of error types and labels do not match"
                    
        n_e = int(len(plt_index) + 1)
        plt.figure(figsize=(int(5*n_e),4))
        
        plt.subplot(1,n_e,1)
        if include_Aofe_prog and "Aofe" not in plt_labels: plt.plot(MFI.Avr_Error_list[:,0], MFI.Avr_Error_list[:,2], label="Aofe", alpha=0.3)
        for i in range(n_e-1): plt.plot(MFI.Avr_Error_list[:,0], MFI.Avr_Error_list[:,plt_index[i]], label=plt_labels[i])
        for j in range(1,len(MFI.n_pos_analysed)): plt.axvline(np.sum(MFI.n_pos_analysed[1:int(j+1)]) * MFI.time_step * MFI.position_pace / 1000, color="yellow", linestyle="--", alpha=0.5) 
        
        plt.yscale("log"); plt.legend(); plt.xlabel("Time [ns]"); plt.ylabel("Error"); plt.title("Error Progression")
            
        for i in range(n_e-1): 
            error_max = np.percentile(MFI.Maps_list[-1, plt_index[i]][np.nonzero(MFI.Maps_list[-1, plt_index[i]])], 99.0)
            plt.subplot(1,n_e,i+2); plt.contourf(MFI.X, MFI.Y, zero_to_nan(MFI.Maps_list[-1, plt_index[i]]), levels=np.linspace(0, error_max, 11), cmap='coolwarm'); plt.colorbar(label=plt_labels[i]); plt.xlabel("x"); plt.ylabel("y"); plt.title(plt_titles[i])

        plt.tight_layout(); plt.show()

    def plot_all_change_in_error(MFI, plot_error_log_scale=False):
        
        # check if aad is available
        if hasattr(MFI, "aad_index"): plot_aad = True
        else: plot_aad = False
        
        # set up the titles and axis lables
        title_list = ["Error", "Explored Volume", "d(Error)", "d(Error)/dt", "D(Error)", "D(Error)/Dt"]
        yl_label_list = ["Aofe [kJ/mol]", "volume ratio [-]", "d(Aofe) [kJ/mol]", "d(Aofe)/dt [kJ/mol/ns]", "D(Aofe) [kJ/mol]", "D(Aofe)/Dt [kJ/mol/ns]"]
        if plot_aad: yr_label_list = ["AAD [kJ/mol]", "", "d(AAD) [kJ/mol]", "d(AAD)/dt [kJ/mol/ns]", "D(AAD) [kJ/mol]", "D(AAD)/Dt [kJ/mol/ns]"]

        # set up the figure and its axes
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2, figsize=(20,15))
        ax = [ax1, ax2, ax3, ax4, ax5, ax6]
        if plot_aad: ax_twin = [ax1.twinx(), None, ax3.twinx(), ax4.twinx(), ax5.twinx(), ax6.twinx()]

        # set up the time and error arrays
        t, t_1, vol, Aofe = MFI.Avr_Error_list[:,0], MFI.Avr_Error_list[1:,0], MFI.Avr_Error_list[:,1], MFI.Avr_Error_list[:,2]
        if plot_aad: AAD = MFI.Avr_Error_list[:,MFI.aad_index]
                
        # iterate over each of the 6 subplots. If aad available, plot the aad on the right y-axis (axr).
        for i in range(6):
            axl = ax[i]
            if plot_aad: axr = ax_twin[i]
            
            # plot the error progression 
            if i == 0: 
                axl.plot(t, Aofe, color="red")
                if plot_aad: axr.plot(t, AAD, color="blue")
                if plot_error_log_scale: axl.set_yscale("log"); axr.set_yscale("log")
            # plot the volume progression
            elif i == 1: axl.plot(t, vol, c="black")
            
            # plot the change in error: d(error), d(error)/dt, D(error), D(error)/dt. small d represents the change wrt. the last error calculation. Big D represents the change wrt. the first error calculation.
            else: 
                axl.plot(t_1, MFI.d_Aofe[:, i-2], color="red", alpha=0.5, linewidth=0.5)
                if plot_aad: axr.plot(t_1, MFI.d_AAD[:, i-2], color="blue", alpha=0.5, linewidth=0.5)
                axl.plot(t_1, MFI.d_Aofe_w[:, i-2], color="red", linewidth=5, alpha=0.3)
                if plot_aad: axr.plot(t_1, MFI.d_AAD_w[:, i-2], color="blue", linewidth=5, alpha=0.3)
                axl.axhline(0, color="grey", alpha=0.3, linestyle="--")
                axl.set_yscale("symlog", linthresh=0.001); axr.set_yscale("symlog", linthresh=0.001)
                    
            
            axl.set_title(title_list[i]); axl.set_xlabel("Time [ns]"); axl.set_ylabel(yl_label_list[i], color="red"); 
            for j in range(1,len(MFI.n_pos_analysed)): 
                t_end = np.sum(MFI.n_pos_analysed[1:int(j+1)]) * MFI.time_step * MFI.position_pace / 1000
                y_text = (axl.get_ylim()[0] + axl.get_ylim()[1]) / 2
                axl.axvline(t_end, color="lime", linestyle="--", alpha=0.5)    
                axl.text(t_end, y_text, "End of Sim " + str(j), rotation=90, verticalalignment='bottom', horizontalalignment='right', color="green", fontsize=10, zorder=0)
            if i != 1 and plot_aad: axr.set_ylabel(yr_label_list[i], color="blue"); axl.tick_params('y', colors='red'); axl.spines['left'].set_color('red'); axr.tick_params('y', colors='blue'); axr.spines['right'].set_color('blue')
            
        plt.tight_layout(); plt.show()

