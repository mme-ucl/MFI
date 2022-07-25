import os
import subprocess
from subprocess import PIPE
from random import randint


def run_langevin1D(simulation_steps,
                   analytical_function = "7*x^4-23*x^2", periodic_boundaries= "NO",
                   initial_position=0.0, temperature=1, time_step=0.005,
                   grid_min=-3.0, grid_max=3.0, grid_bin=200,
                   gaus_width=0.1, gaus_height=1, biasfactor=10, gaus_pace=100, position_pace=0,
                   hp_center=0.0, hp_kappa=0,
                   lw_center=0.0, lw_kappa=0,
                   uw_center=0.0, uw_kappa=0):
    """Function to run a langevin simulation in 1 dimension. Default analytical potential: y = 7*x^4-23*x^2.

    Args:
        simulation_steps (int): Number of steps in simulation
        analytical_function (str, optional): The analytical function to be analysed. Defaults to "7*x^4-23*x^2".
        periodic_boundaries (str, optional): Information wheather boundary conditions are periodic ("ON") or not ("NO"). Defaults to "NO".
        initial_position (float, optional): Initial position of simulation. Defaults to 0.0.
        temperature (int, optional): Temperature of simulation (units in kT). Defaults to 1.
        time_step (float, optional): Length of one time step (units in ps). Defaults to 0.005.
        grid_min (float, optional): Minimum value of grid where the bias is stored. Defaults to -3.0.
        grid_max (float, optional): Maximum value of grid where the bias is stored. Defaults to 3.0.
        grid_bin (int, optional): Number of distinct bins in grid. Defaults to 200.
        gaus_width (float, optional): Gaussian width (sigma) of metadynamics bias. Defaults to 0.1.
        gaus_height (int, optional): Gaussian height of metadynamics bias. Defaults to 1.
        biasfactor (int, optional): Biasfactor of metadynamics bias. Defaults to 10.
        gaus_pace (int, optional): Pace of deposition of metadynamics hills. Defaults to 100.
        position_pace (int, optional): Pace of recording the CV in the position file. When position_pace=0, position_pace = gaus_pace/10. Defaults to 0.
        hp_center (float, optional): position of harmonic potential. Defaults to 0.0.
        hp_kappa (int, optional): force_constant of harmonic potential. Defaults to 0.
        lw_center (float, optional): position of lower wall potential. Defaults to 0.0.
        lw_kappa (int, optional): force_constant of lower wall potential. Defaults to 0.
        uw_center (float, optional): position of upper wall potential. Defaults to 0.0.
        uw_kappa (int, optional): force_constant of upper wall potential. Defaults to 0.
    """

    with open("input" ,"w") as f:
        print("""temperature {}
tstep {}
friction 1
dimension 1
nstep {}
ipos {}""".format(temperature, time_step, simulation_steps,  initial_position) ,file=f)

        if periodic_boundaries == "NO":
            f.write("periodic false")
        else:
            f.write("periodic on\n")
            f.write("min {}\n".format(periodic_boundaries.split(",")[0]))
            f.write("max {}".format(periodic_boundaries.split(",")[1]))


    with open("plumed.dat" ,"w") as f:
        print("""p: DISTANCE ATOMS=1,2 COMPONENTS
ff: MATHEVAL ARG=p.x FUNC=({}) PERIODIC={}
bb: BIASVALUE ARG=ff""".format(analytical_function, periodic_boundaries) ,file=f)

    with open("plumed.dat" ,"a") as f:
        # Metadynamics bias. To activate, the height of the bias needs to be a positive number.
        if gaus_height > 0:
            f.write("METAD ARG=p.x SIGMA={} HEIGHT={} BIASFACTOR={} GRID_MIN={} GRID_MAX={} GRID_BIN={} PACE={} \
TEMP={} \n".format(gaus_width, gaus_height, biasfactor, grid_min, grid_max, grid_bin, gaus_pace, temperature * 120))

        # Harmonic potential bias. To activate, the force constant (kappa) needs to be a positive number
        if hp_kappa > 0:
            f.write("RESTRAINT ARG=p.x AT={} KAPPA={} LABEL=restraint \n".format(hp_center, hp_kappa))

        # Lower wall bias. To activate, the force constant (kappa) needs to be a positive number
        if lw_kappa > 0:
            f.write("LOWER_WALLS ARG=p.x AT={} KAPPA={} LABEL=lowerwall \n".format(lw_center, lw_kappa))

        # Upper wall bias. To activate, the force constant (kappa) needs to be a positive number
        if uw_kappa > 0:
            f.write("UPPER_WALLS ARG=p.x AT={} KAPPA={} LABEL=upperwall \n".format(uw_center, uw_kappa))

        # Print position of system. If position_pace = 0, it will be position_pace = gaus_pace/10
        if position_pace == 0: position_pace = int(gaus_pace / 10)
        f.write("PRINT FILE=position ARG=p.x STRIDE={}".format(position_pace))


    os.system("plumed pesmd < input")

def run_langevin2D(simulation_steps,
                   analytical_function="7*x^4-23*x^2+7*y^4-23*y^2", periodic_f="NO",
                   initial_position_x=0.0, initial_position_y=0.0, temperature=1, time_step=0.005,
                   grid_min_x=-3.0, grid_max_x=3.0, grid_min_y=-3.0, grid_max_y=3.0, grid_bin_x=200,
                   grid_bin_y=200,
                   gaus_width_x=0.1, gaus_width_y=0.1, gaus_height=1, biasfactor=10, gaus_pace=100,
                   hp_center_x=0.0, hp_center_y=0.0, hp_kappa_x=0, hp_kappa_y=0,
                   lw_center_x=0.0, lw_center_y=0.0, lw_kappa_x=0, lw_kappa_y=0,
                   uw_center_x=0.0, uw_center_y=0.0, uw_kappa_x=0, uw_kappa_y=0,
                   position_pace=0, file_extension=""):
    """Function to run a langevin simulation in 2 dimension. Default analytical potential: z = 7*x^4-23*x^2+7*y^4-23*y^2.

    Args:
        simulation_steps (int): Number of steps in simulation
        analytical_function (str, optional): The analytical function to be analysed. Defaults to "7*x^4-23*x^2+7*y^4-23*y^2".
        periodic_f (str, optional): Information wheather boundary conditions are periodic ("ON") or not ("NO"). Defaults to "NO".
        initial_position_x (float, optional): x-value of initial position of simulation. Defaults to 0.0.
        initial_position_y (float, optional): y-value of initial position of simulation. Defaults to 0.0.
        temperature (int, optional): Temperature of simulation (units in kT). Defaults to 1.
        time_step (float, optional): Length of one time step (units in ps). Defaults to 0.005.
        grid_min_x (float, optional): x-value of minimum value of grid where the bias is stored. Defaults to -3.0.
        grid_max_x (float, optional): x-value of maximum value of grid where the bias is stored. Defaults to 3.0.
        grid_min_y (float, optional): y-value of minimum value of grid where the bias is stored. Defaults to -3.0.
        grid_max_y (float, optional): y-value of maximum value of grid where the bias is stored. Defaults to 3.0.
        grid_bin_x (int, optional): Number of distinct bins in grid on x-axis. Defaults to 200.
        grid_bin_y (int, optional): Number of distinct bins in grid on y-axis. Defaults to 200.
        gaus_width_x (float, optional): Gaussian width in x-direction (sigma) of metadynamics bias. Defaults to 0.1.
        gaus_width_y (float, optional): Gaussian width in y-direction (sigma) of metadynamics bias. Defaults to 0.1.
        gaus_height (int, optional): Gaussian height of metadynamics bias. Defaults to 1.
        biasfactor (int, optional): Biasfactor of metadynamics bias. Defaults to 10.
        gaus_pace (int, optional): Pace of deposition of metadynamics hills. Defaults to 100.
        hp_centre_x (float, optional): x-position of harmonic potential. Defaults to 0.0.
        hp_centre_y (float, optional): y-position of harmonic potential. Defaults to 0.0.
        hp_kappa_x (int, optional): x-force_constant of harmonic potential. Defaults to 0.
        hp_kappa_y (int, optional): y-force_constant of harmonic potential. Defaults to 0.
        lw_centre_x (float, optional): x-position of lower wall potential. Defaults to 0.0.
        lw_centre_y (float, optional): y-position of lower wall potential. Defaults to 0.0.
        lw_kappa_x (int, optional): x-force_constant of lower wall potential. Defaults to 0.
        lw_kappa_y (int, optional): y-force_constant of lower wall potential. Defaults to 0.
        uw_centre_x (float, optional): x-position of upper wall potential. Defaults to 0.0.
        uw_centre_y (float, optional): y-position of upper wall potential. Defaults to 0.0.
        uw_kappa_x (int, optional): x-force_constant of upper wall potential. Defaults to 0.
        uw_kappa_y (int, optional): y-force_constant of upper wall potential. Defaults to 0.
        position_pace (int, optional): Pace of recording the CV in the position file. When position_pace=0, position_pace = gaus_pace/10. Defaults to 0.
        file_extension (str, optional): Adds an extension the the position and HILLS file. E.g. file_extension="_1" -> position_file="position_1". Defaults to "".
    """    

    with open("input", "w") as f:
        print("""temperature {}
tstep {}
friction 1
dimension 2
nstep {}
ipos {},{}
periodic false""".format(temperature, time_step, simulation_steps, initial_position_x, initial_position_y), file=f)

    with open("plumed.dat", "w") as f:
        print("""p: DISTANCE ATOMS=1,2 COMPONENTS
ff: MATHEVAL ARG=p.x,p.y PERIODIC={} FUNC=({})
bb: BIASVALUE ARG=ff""".format(periodic_f, analytical_function), file=f)

    with open("plumed.dat", "a") as f:
        # Metadynamics bias. To activate, the height of the bias needs to be a positive number.
        if gaus_height > 0:
            f.write("METAD ARG=p.x,p.y SIGMA={},{} HEIGHT={} BIASFACTOR={} GRID_MIN={},{} GRID_MAX={},{} GRID_BIN={},{} PACE={} \
TEMP={} FILE=HILLS{}\n".format(gaus_width_x, gaus_width_y, gaus_height, biasfactor, grid_min_x, grid_min_y, grid_max_x,
               grid_max_y, grid_bin_x, grid_bin_y, gaus_pace, temperature * 120, file_extension))

        # Harmonic potential bias. To activate, the force constant (kappa) needs to be a positive number
        if hp_kappa_x > 0 or hp_kappa_y > 0:
            f.write(
                "RESTRAINT ARG=p.x,p.y AT={},{} KAPPA={},{} LABEL=restraint \n".format(hp_center_x, hp_center_y, hp_kappa_x, hp_kappa_y))

        # Lower wall bias. To activate, the force constant (kappa) needs to be a positive number
        if lw_kappa_x > 0 or lw_kappa_y > 0:
            f.write("LOWER_WALLS ARG=p.x,p.y AT={},{} KAPPA={},{} LABEL=lowerwall \n".format(lw_center_x, lw_center_y, lw_kappa_x,lw_kappa_y))

        # Upper wall bias. To activate, the force constant (kappa) needs to be a positive number
        if uw_kappa_x > 0 or uw_kappa_y > 0:
            f.write("UPPER_WALLS ARG=p.x,p.y AT={},{} KAPPA={},{} LABEL=upperwall \n".format(uw_center_x, uw_center_y, uw_kappa_x, uw_kappa_y))

        # Print position of system. If position_pace = 0, it will be position_pace = gaus_pace/10
        if position_pace == 0: position_pace = int(gaus_pace / 10)
        f.write("PRINT FILE=position{} ARG=p.x,p.y STRIDE={}".format(file_extension , position_pace))
     
    print("starting simulation...")
    os.system("plumed pesmd < input")

    # process_run_simulation = subprocess.Popen(["plumed", "pesmd", "<", "input"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # process_run_simulation.wait()
    # output_process_run_simulation, errors_process_run_simulation = process_run_simulation.communicate()

    # if "Error" in errors_process_run_simulation:
    #     print("There is an error message")
    #     print(errors_process_run_simulation)


def run_2D_Invernizzi(simulation_steps=100000, sigma=0.1, height=0.5, biasfactor=10, initial_position_x=0.0, initial_position_y=0.0,
                   hp_center_x=0.0, hp_center_y=0.0, hp_kappa_x=0, hp_kappa_y=0,
                   lw_center_x=0.0, lw_center_y=0.0, lw_kappa_x=0, lw_kappa_y=0,
                   uw_center_x=0.0, uw_center_y=0.0, uw_kappa_x=0, uw_kappa_y=0,
                   gaus_pace=200, position_pace=0, file_extension=""):
    """Function to run a langevin simulation (in 2D) on the Invernizzi potential. Analytical form is approx.: 1.35*x^4+1.90*x^3*y+3.93*x^2*y^2-6.44*x^2-1.90*x*y^3+5.59*x*y+1.33*x+1.35*y^4-5.56*y^2+0.90*y+18.56.

    Args:
        simulation_steps (int, optional): Number of steps in simulation. Defaults to 100000.
        sigma (float, optional): Gaussian width (sigma) in x-direction and y-direction of metadynamics bias. Defaults to 0.1.
        height (float, optional): aussian height of metadynamics bias. Defaults to 0.5.
        biasfactor (int, optional): Biasfactor of metadynamics bias. Defaults to 10.
        initial_position_x (float, optional): x-value of initial position of simulation. Defaults to 0.0.
        initial_position_y (float, optional): y-value of initial position of simulation. Defaults to 0.0.
        hp_centre_x (float, optional): x-position of harmonic potential. Defaults to 0.0.
        hp_centre_y (float, optional): y-position of harmonic potential. Defaults to 0.0.
        hp_kappa_x (int, optional): x-force_constant of harmonic potential. Defaults to 0.
        hp_kappa_y (int, optional): y-force_constant of harmonic potential. Defaults to 0.
        lw_centre_x (float, optional): x-position of lower wall potential. Defaults to 0.0.
        lw_centre_y (float, optional): y-position of lower wall potential. Defaults to 0.0.
        lw_kappa_x (int, optional): x-force_constant of lower wall potential. Defaults to 0.
        lw_kappa_y (int, optional): y-force_constant of lower wall potential. Defaults to 0.
        uw_centre_x (float, optional): x-position of upper wall potential. Defaults to 0.0.
        uw_centre_y (float, optional): y-position of upper wall potential. Defaults to 0.0.
        uw_kappa_x (int, optional): x-force_constant of upper wall potential. Defaults to 0.
        uw_kappa_y (int, optional): y-force_constant of upper wall potential. Defaults to 0.
        gaus_pace (int, optional): Pace of deposition of metadynamics hills. Defaults to 200.
        position_pace (int, optional): Pace of recording the CV in the position file. When position_pace=0, position_pace = gaus_pace/10. Defaults to 0.
        file_extension (str, optional): Adds an extension the the position and HILLS file. E.g. file_extension="_1" -> position_file="position_1". Defaults to "".
    """    
    with open("plumed.dat","w") as f:
        print("""p: DISTANCE ATOMS=1,2 COMPONENTS
ff: MATHEVAL ARG=p.x,p.y PERIODIC=NO FUNC=(1.34549*x^4+1.90211*x^3*y+3.92705*x^2*y^2-6.44246*x^2-1.90211*x*y^3+5.58721*x*y+1.33481*x+1.34549*y^4-5.55754*y^2+0.904586*y+18.5598)
bb: BIASVALUE ARG=ff
METAD ARG=p.x,p.y PACE={} SIGMA={},{} HEIGHT={} GRID_MIN=-3,-3 GRID_MAX=3,3 GRID_BIN=300,300 BIASFACTOR={} TEMP=120 FILE=HILLSinve_{}""".format(gaus_pace, sigma, sigma, height, biasfactor,file_extension),file=f)

    with open("plumed.dat", "a") as f:
        # Harmonic potential bias. To activate, the force constant (kappa) needs to be a positive number
        if hp_kappa_x > 0 or hp_kappa_y > 0:
            f.write(
                "RESTRAINT ARG=p.x,p.y AT={},{} KAPPA={},{} LABEL=restraint \n".format(hp_center_x, hp_center_y, hp_kappa_x, hp_kappa_y))

        # Lower wall bias. To activate, the force constant (kappa) needs to be a positive number
        if lw_kappa_x > 0 or lw_kappa_y > 0:
            f.write("LOWER_WALLS ARG=p.x,p.y AT={},{} KAPPA={},{} LABEL=lowerwall \n".format(lw_center_x, lw_center_y, lw_kappa_x,lw_kappa_y))

        # Upper wall bias. To activate, the force constant (kappa) needs to be a positive number
        if uw_kappa_x > 0 or uw_kappa_y > 0:
            f.write("UPPER_WALLS ARG=p.x,p.y AT={},{} KAPPA={},{} LABEL=upperwall \n".format(uw_center_x, uw_center_y, uw_kappa_x, uw_kappa_y))

        # Print position of system. If position_pace = 0, it will be position_pace = gaus_pace/10
        if position_pace == 0: position_pace = int(gaus_pace / 10)
        f.write("PRINT FILE=positioninve_{} ARG=p.x,p.y STRIDE={}".format(file_extension , position_pace))
     
    with open("input","w") as f:
        print("""temperature 1
tstep 0.005
friction 10
dimension 2
nstep {}
ipos {},{}
periodic false""".format(simulation_steps,initial_position_x,initial_position_y),file=f)

    #Start simulation
    print("Running simulation...")
    os.system("plumed pesmd < input")

# #Exaple execution (run simulation in folder: test with location <path>:
# path = "/home/antoniu/Desktop/Public_Notebooks/"
# os.chdir(path + "test")
# 
# run_langevin2D(100000, lw_center_x=1.5, lw_center_y=1.0, lw_kappa_x=30, lw_kappa_y=50, uw_center_x=2.0, uw_center_y=2.5, uw_kappa_x=70, uw_kappa_y=90)



def find_alanine_dipeptide_input(initial_position_x=0.0, initial_position_y=0.0, file_extension=""):
    """Prepares the input file for a set of initial positions. 
    Requires traj_comp.xtc file or similar (trajecory file of a simulation that already explored specified initial positions).
    1 step: Find structures that are +- 0.5nm away from intial position.
    2 step: Randomly choose one of the structures.
    3 step: Produce input file for alanine dipeptide simulation.

    Args:
        initial_position_x (float, optional): x-value of initial position of simulation. Defaults to 0.0.
        initial_position_y (float, optional): y-value of initial position of simulation. Defaults to 0.0.
        file_extension (str, optional): Adds an extension the the structure.gro and input.tpr file. E.g. file_extension="_1" -> structure_file="structure_1.gro". Defaults to "".. Defaults to "".
    """    

    #Analyse trajectorry "0traj_comp.xtc" -> output new structure(n).gro
    start_region = [str( initial_position_x - 0.5), str( initial_position_x + 0.5), str( initial_position_y - 0.5), str( initial_position_y + 0.5)]
    print("Preparing new input files ...")
    with open("plumed_traj.dat") as f:
        lines = f.readlines()
    lines[3] = "UPDATE_IF ARG=phi,psi MORE_THAN=" + start_region[0] + "," + start_region[2] + " LESS_THAN=" + start_region[1] + "," + start_region[3] + "\n"
    lines[4] = "DUMPATOMS FILE=structure" + str(file_extension) + ".gro ATOMS=1-22\n"
    with open("plumed_traj.dat", "w") as f:
        f.writelines(lines)

    os.system("plumed driver --plumed plumed_traj.dat --mf_xtc 0traj_comp.xtc > /dev/null")

    # Remove a random number of structures, to avoid using always the same structure.
    total_n_lines = int(os.popen("wc -l structure" + str(file_extension) + ".gro").read().split()[0])
    del_structure_lines = int(25 * randint(1, (int(total_n_lines / 25) - 1)))
    os.system("sed -i -e '1," + str(del_structure_lines) + "d' structure" + str(file_extension) + ".gro")

    #Prepare new input file<- input structure.gro, topolvac.top, gromppvac.mdp -> input(n).tpr
    find_input_structure = subprocess.Popen(["gmx", "grompp", "-f", "gromppvac.mdp", "-c", "structure" + str(file_extension)+".gro", "-p", "topology.top", "-o", "input" + str(file_extension)+ ".tpr"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)#, text=True)
    find_input_structure.wait()
    output_find_input_structure, errors_find_input_structure = find_input_structure.communicate()
    # if "Error" in errors_find_input_structure:
    #     print("*****There is an error message:*****\n\n")
    #     print(errors_find_input_structure)
    #####<<<Prepare new input file input(n).tpr<<<##############################



def run_alanine_dipeptide(simulation_steps, temperature=2.49,
                   grid_min_x="-pi", grid_max_x="pi", grid_min_y="-pi", grid_max_y="pi", grid_bin_x=201, grid_bin_y=201,
                   gaus_width_x=0.1, gaus_width_y=0.1, gaus_height=1, biasfactor=10, gaus_pace=100, position_pace=0, 
                   hp_centre_x=0.0, hp_centre_y=0.0, hp_kappa_x=0, hp_kappa_y=0,
                   lw_centre_x=0.0, lw_centre_y=0.0, lw_kappa_x=0, lw_kappa_y=0,
                   uw_centre_x=0.0, uw_centre_y=0.0, uw_kappa_x=0, uw_kappa_y=0,
                   print_bias = 0, file_extension=""):
    """Function to run molecular simulation on alanine dipeptide. Requires a reference.pdb and input.tpr file.

    Args:
        simulation_steps (int): Number of steps in simulation
        temperature (float, optional): Temperature of simulation (units in kT). Defaults to 2.49.
        grid_min_x (str, optional): phi-value of minimum value of grid where the bias is stored. Defaults to "-pi".
        grid_max_x (str, optional): phi-value of maximum value of grid where the bias is stored. Defaults to "pi".
        grid_min_y (str, optional): psi-value of minimum value of grid where the bias is stored. Defaults to "-pi".
        grid_max_y (str, optional): psi-value of maximum value of grid where the bias is stored. Defaults to "pi".
        grid_bin_x (int, optional): Number of distinct bins in grid on phi-axis. Defaults to 201.
        grid_bin_y (int, optional): Number of distinct bins in grid on psi-axis. Defaults to 201.
        gaus_width_x (float, optional): Gaussian width in phi-direction (sigma) of metadynamics bias. Defaults to 0.1.
        gaus_width_y (float, optional): Gaussian width in psi-direction (sigma) of metadynamics bias. Defaults to 0.1.
        gaus_height (int, optional): Gaussian height of metadynamics bias. Defaults to 1.
        biasfactor (int, optional): Biasfactor of metadynamics bias. Defaults to 10.
        gaus_pace (int, optional): Pace of deposition of metadynamics hills. Defaults to 100.
        position_pace (int, optional): Pace of recording the CV in the position file. When position_pace=0, position_pace = gaus_pace/10. Defaults to 0.
        hp_centre_x (float, optional): phi-position of harmonic potential. Defaults to 0.0.
        hp_centre_y (float, optional): psi-position of harmonic potential. Defaults to 0.0.
        hp_kappa_x (int, optional): phi-force_constant of harmonic potential. Defaults to 0.
        hp_kappa_y (int, optional): psi-force_constant of harmonic potential. Defaults to 0.
        lw_centre_x (float, optional): phi-position of lower wall potential. Defaults to 0.0.
        lw_centre_y (float, optional): psi-position of lower wall potential. Defaults to 0.0.
        lw_kappa_x (int, optional): phi-force_constant of lower wall potential. Defaults to 0.
        lw_kappa_y (int, optional): psi-force_constant of lower wall potential. Defaults to 0.
        uw_centre_x (float, optional): phi-position of upper wall potential. Defaults to 0.0.
        uw_centre_y (float, optional): psi-position of upper wall potential. Defaults to 0.0.
        uw_kappa_x (int, optional): phi-force_constant of upper wall potential. Defaults to 0.
        uw_kappa_y (int, optional): psi-force_constant of upper wall potential. Defaults to 0.
        print_bias (int, optional): When print_bias=1, the experienced bias potential and the bias force squared is printed every 100 steps in a file called "restraint". Defaults to 0.
        file_extension (str, optional):Adds an extension the the position and HILLS file. E.g. file_extension="_1" -> position_file="position_1". Defaults to "".
    """    
    with open("plumed.dat", "w") as f:
        print("""MOLINFO STRUCTURE=reference.pdb
phi: TORSION ATOMS=@phi-2
psi: TORSION ATOMS=@psi-2""", file=f)

    with open("plumed.dat", "a") as f:
        # Metadynamics bias. To activate, the height of the bias needs to be a positive number.
        if gaus_height > 0:
            f.write("METAD ARG=phi,psi SIGMA={},{} HEIGHT={} BIASFACTOR={} GRID_MIN={},{} GRID_MAX={},{} GRID_BIN={},{} PACE={} \
TEMP={} FILE=HILLS{}\n".format(gaus_width_x, gaus_width_y, gaus_height, biasfactor, grid_min_x, grid_min_y, grid_max_x,
               grid_max_y, grid_bin_x, grid_bin_y, gaus_pace, temperature * 120, file_extension))

        # Harmonic potential bias. To activate, the force constant (kappa) needs to be a positive number
        if hp_kappa_x > 0 or hp_kappa_y > 0:
            f.write("RESTRAINT ARG=phi,psi AT={},{} KAPPA={},{} LABEL=restraint \n".format(hp_center_x, hp_center_y, hp_kappa_x, hp_kappa_y))
            if print_bias == 1:
                f.write("PRINT FILE=restraint ARG=phi,psi,restraint.bias,restraint.force2 STRIDE=100")


        # Lower wall bias. To activate, the force constant (kappa) needs to be a positive number
        if lw_kappa_x > 0 or lw_kappa_y > 0:
            f.write("LOWER_WALLS ARG=phi,psi AT={},{} KAPPA={},{} LABEL=lowerwall \n".format(lw_center_x, lw_center_y, lw_kappa_x,lw_kappa_y))

        # Upper wall bias. To activate, the force constant (kappa) needs to be a positive number
        if uw_kappa_x > 0 or uw_kappa_y > 0:
            f.write("UPPER_WALLS ARG=phi,psi AT={},{} KAPPA={},{} LABEL=upperwall \n".format(uw_center_x, uw_center_y, uw_kappa_x, uw_kappa_y))

        # Print position of system. If position_pace = 0, it will be position_pace = gaus_pace/10
        if position_pace == 0: position_pace = int(gaus_pace / 10)
        f.write("PRINT FILE=position{} ARG=phi,psi STRIDE={}".format(file_extension, position_pace))

        if print_bias == 1:
            f.write("PRINT FILE=restraint ARG=phi,psi,restraint.bias,restraint.force2 STRIDE=100")


    """#gmx mdrun -s topolA.tpr -nsteps 1000000 -plumed plumed_first.dat -v"""

    #####>>>run simulations>>>##############################
    print("Running Alanine Dipeptide simulation")
    # os.system("gmx mdrun -s input" + str(simulation_count) + ".tpr -nsteps " + str(int(nsteps)) + " -plumed plumed_restr.dat -v") # > /dev/null
    find_input_file = subprocess.Popen(["gmx", "mdrun", "-s", "input" + str(file_extension) + ".tpr", "-nsteps", str(int(simulation_steps)), "-plumed",
         "plumed.dat"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)#, text=True)
    find_input_file.wait()
    output_find_input_file, errors_find_input_file = find_input_file.communicate()

    # if "Error" in errors_find_input_file:
    #     print("There is an error message")
    #     print(errors_find_input_file)

    print("... Simulation finished.\n")





def run_langevin1D_plumed_fes(length, initial_position=-1.0, sigma=0.1, height=0.1, biasfactor=10, fes_stride = 0, grid_min = -3, grid_max = 3, grid_bin = 301):
    """Function to run a langevin simulation in 1 dimension on analytical potential: y = 7*x^4-23*x^2, while also calculating the FES through plumed.

    Args:
        length (int): Number of steps in simulation
        sigma (float, optional): Gaussian width of metadynamics bias. Defaults to 0.1.
        height (float, optional): Gaussian height of metadynamics bias. Defaults to 0.1.
        biasfactor (int, optional): Biasfactor of metadynamics bias. Defaults to 10.
        fes_stride (int, optional): Number of times the intermediate fes is calculated. Defaults to 0.
        grid_min (int, optional): Minimum value of grid where the bias is stored. Defaults to -3.
        grid_max (int, optional): Maximum value of grid where the bias is stored. Defaults to 3.
        grid_bin (int, optional): Number of distinct bins in grid. Defaults to 301.
    """    
    with open("plumed.dat","w") as f:
        print("""#Define system as distance between two atoms
p: DISTANCE ATOMS=1,2 COMPONENTS
#Define Force field
ff: MATHEVAL ARG=p.x PERIODIC=NO FUNC=(7*x^4-23*x^2)
bb: BIASVALUE ARG=ff
#Define Metadynamics potential
METAD ARG=p.x PACE=100 SIGMA={} HEIGHT={} GRID_MIN={} GRID_MAX={} GRID_BIN={} BIASFACTOR={} TEMP=120 CALC_RCT
#Reweight Bias
bias: REWEIGHT_METAD TEMP=120
#Make Histogram
hh: HISTOGRAM ARG=p.x GRID_MIN={} GRID_MAX={} GRID_BIN={} BANDWIDTH=0.01 LOGWEIGHTS=bias
#Convert Histogram to FES
fes: CONVERT_TO_FES GRID=hh TEMP=120
#Save Histogram and FES at the end. Save position every 10 time-steps    
DUMPGRID GRID=fes FILE=fes.dat STRIDE={}
PRINT FILE=position ARG=p.x STRIDE=10""".format(sigma, height, grid_min, grid_max, grid_bin, biasfactor, grid_min, grid_max, grid_bin, fes_stride),file=f)

    with open("input","w") as f:
        print("""temperature 1
tstep 0.005
friction 1
dimension 1
nstep {}
ipos {}
periodic false""".format(length, initial_position),file=f)

    #Start WT-Metadynamic simulation
    print("Running simulation")
    os.system("plumed pesmd < input")
