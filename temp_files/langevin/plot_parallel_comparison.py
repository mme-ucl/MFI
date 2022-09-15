import os
import numpy as np
import sys
import time
import multiprocessing as mp
from shutil import rmtree
print("Number of processors: ", mp.cpu_count())

sys.path.insert(0, "/home/antoniu/Desktop/MFI_git/MFI_master/MFI/")
from pyMFI import MFI
from pyMFI import run_plumed
import pickle
import matplotlib.pyplot as plt
from multiprocess import Pool


os.chdir("/home/antoniu/Desktop/MFI_git/MFI_master/MFI/temp_files/langevin")
path = os.getcwd()
print("The path to this notebook is:", path)

for plot_info in range(1):
    from matplotlib import rc
    plt.rcParams.update({ "text.usetex": True, "font.family": "serif", "font.serif": ["computer modern roman"], "font.size": 16})
    plw = 0.6
    pcs = 3
    pms = 3
    bfillc = [0.9,0.9,0.9]
    plt.rcParams['axes.linewidth'] = plw
    plt.rcParams['xtick.top'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['xtick.major.width'] = plw
    plt.rcParams['xtick.minor.width'] = plw
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['xtick.major.size'] = 4.5
    plt.rcParams['ytick.right'] = True
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['ytick.major.width'] = plw
    plt.rcParams['ytick.minor.width'] = plw
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['ytick.major.size'] = 5
    plt.rcParams["figure.figsize"] = (5,4)

    times = r"$\times$"
    pwr_neg = r"$^{-1}$"

for do_analytical_fes in range(1):
    min_grid=np.array((-2, -2))
    max_grid=np.array((2, 2))
    nbins=np.array((200, 200))

    gridx = np.linspace(min_grid[0], max_grid[0], nbins[0])
    gridy = np.linspace(min_grid[1], max_grid[1], nbins[1])
    dx = abs(gridx[1] - gridx[0])
    dy = abs(gridy[1] - gridy[0])
    X, Y = np.meshgrid(gridx, gridy)
    Z = 7*X**4-23*X**2+7*Y**4-23*Y**2
    Z = Z - np.min(Z)
    

initial_pos = [[1,1], [1,-1], [-1,1], [-1,-1], [1,1], [1,-1], [-1,1], [-1,-1], [1,1], [1,-1], [-1,1], [-1,-1]]
n_process = 4

try: os.mkdir("folder_" + str(n_process))
except: 
    rmtree("folder_" + str(n_process))
    os.mkdir("folder_" + str(n_process))
os.chdir("folder_" + str(n_process))


def run_n_simulations(n_sim, i_pos):
    
    extension = "_" + str(n_sim)
    os.mkdir("simulation_" + str(n_sim))
    os.chdir("simulation_" + str(n_sim))
    
    n_steps = 1000000
    
    #Run simulation
    run_plumed.run_langevin2D(int(n_steps), initial_position_x=i_pos[0], initial_position_y=i_pos[1], gaus_width_x=0.1, gaus_width_y=0.1, gaus_height=2, biasfactor=20, gaus_pace=100, file_extension=extension)

def analyse_n_simulations(n_sim):
    extension = "_" + str(n_sim)
    
    print(os.getcwd(), "ext=", extension)
    
    os.chdir("simulation_" + str(n_sim))
    
    #Read the HILLS file
    HILLS=MFI.load_HILLS_2D(hills_name="HILLS" + extension)
    
    print(len(HILLS), "ext=", extension)

    #Read the Colvar File
    [position_x, position_y] = MFI.load_position_2D(position_name="position" + extension)

    #Compute the time-independent mean force
    [X, Y, Ftot_den, Ftot_x, Ftot_y, ofv, ofe, cutoff, ofv_history, ofe_history, time_history, Ftot_den2, ofv_num_x, ofv_num_y] = MFI.MFI_2D(HILLS = HILLS, position_x = position_x, position_y = position_y, bw = 0.1, kT = 1, min_grid=np.array((-2, -2)), max_grid=np.array((2, 2)), nbins=np.array((200, 200)), log_pace = 1, error_pace = 1, nhills=-1, periodic=0, FES_cutoff=38, FFT_integration=1)
    
    print(len(HILLS), "ext=", extension)

    return [X,Y, Ftot_den, Ftot_x, Ftot_y]

master = []
def collect_result(result):
    global master
    master.append(result)


pool = Pool()
start = time.time()


for i in range(n_process):
    print("simulation:" , i)
    pool.apply_async(run_n_simulations, args=(i+1, initial_pos[i]))

while time.time() - start < 5:
    time.sleep(2)
    print("time:" , round(time.time() - start,1))
    
print("\nsimulation time over\n")
    
pool.terminate()
pool.join()

print("\nnext operation\n")


pool = Pool()

for i in range(n_process):
    print("analysis:" , i)
    pool.apply_async(analyse_n_simulations, args=(i+1), callback=collect_result)

pool.close()
pool.join()

[FP, FX, FY] = MFI.patch_2D_simple(master)

#integration on a periodic domain
[X, Y, FES] = MFI.intgrad2(FX, FY, min_grid=np.array((-2, -2)), max_grid=np.array((2, 2)))

print("shape of FES, " , np.shape(FES))
print("shape of Z, " , np.shape(Z))

AD = abs(FES - Z)

print("shape of AD, " , np.shape(AD))

AAD = sum(sum(AD)) / (nbins**2)
print("shape of AAD, " , np.shape(AAD))

print("end time:", round(time.time() - start,1), "     and AAD:", AAD)