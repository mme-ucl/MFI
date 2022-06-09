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




# #Exaple execution (run simulation in folder: test with location <path>:
# path = "/home/antoniu/Desktop/Public_Notebooks/"
# os.chdir(path + "test")
# 
# run_langevin2D(100000, lw_center_x=1.5, lw_center_y=1.0, lw_kappa_x=30, lw_kappa_y=50, uw_center_x=2.0, uw_center_y=2.5, uw_kappa_x=70, uw_kappa_y=90)



def find_alanine_dipeptide_input(initial_position_x=0.0, initial_position_y=0.0, file_extension=""):

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



def run_alanine_dipeptide(simulation_steps, temperature=1,
                   grid_min_x="-pi", grid_max_x="pi", grid_min_y="-pi", grid_max_y="pi", grid_bin_x=201, grid_bin_y=201,
                   gaus_width_x=0.1, gaus_width_y=0.1, gaus_height=1, biasfactor=10, gaus_pace=100, position_pace=0, 
                   hp_center_x=0.0, hp_center_y=0.0, hp_kappa_x=0, hp_kappa_y=0,
                   lw_center_x=0.0, lw_center_y=0.0, lw_kappa_x=0, lw_kappa_y=0,
                   uw_center_x=0.0, uw_center_y=0.0, uw_kappa_x=0, uw_kappa_y=0,
                   print_bias = 0, file_extension=""):
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





