import os

def run_langevin1D(simulation_steps,
                   analytical_function = "7*x^4-23*x^2", periodic_boundaries= "NO",
                   initial_position=0.0, temperature=1, time_step=0.005,
                   grid_min=-3.0, grid_max=3.0, grid_bin=200,
                   gaus_width=0.1, gaus_height=1, biasfactor=10, gaus_pace=100,
                   hp_center=0.0, hp_kappa=0,
                   lw_center=0.0, lw_kappa=0,
                   uw_center=0.0, uw_kappa=0,
                   position_pace=0):

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
        f.write("PRINT FILE=position ARG=p.x STRIDE=10")


    os.system("plumed pesmd < input")

def run_langevin2D(simulation_steps,
                   analytical_function="7*x^4-23*x^2+7*y^4-23*y^2", periodic_f="NO",
                   initial_position_x=0.0, initial_position_y=0.0, temperature=1, time_step=0.005,
                   grid_min_x=-3.0, grid_max_x=3.0, grid_min_y=-3.0, grid_max_y=3.0, grid_bin_x=200,
                   grid_bin_y=200,
                   gaus_width_x=0.1, gaus_width_y=0.1, gaus_height=1, biasfactor=10, gaus_pace=100,
                   hp_center_x=0.0, hp_center_y=0.0, hp_kappa_x=100, hp_kappa_y=100,
                   lw_center_x=0.0, lw_center_y=0.0, lw_kappa_x=0, lw_kappa_y=0,
                   uw_center_x=0.0, uw_center_y=0.0, uw_kappa_x=0, uw_kappa_y=0,
                   position_pace=0):

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
TEMP={} \n".format(gaus_width_x, gaus_width_y, gaus_height, biasfactor, grid_min_x, grid_min_y, grid_max_x,
               grid_max_y, grid_bin_x, grid_bin_x, gaus_pace, temperature * 120))

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
        f.write("PRINT FILE=position ARG=p.x,p.y STRIDE={}".format(position_pace))

    os.system("plumed pesmd < input")


# #Exaple execution (run simulation in folder: test with location <path>:
# path = "/home/antoniu/Desktop/Public_Notebooks/"
# os.chdir(path + "test")
# 
# run_langevin2D(100000, lw_center_x=1.5, lw_center_y=1.0, lw_kappa_x=30, lw_kappa_y=50, uw_center_x=2.0, uw_center_y=2.5, uw_kappa_x=70, uw_kappa_y=90)
