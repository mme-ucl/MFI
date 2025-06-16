import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import unittest

# Find the path of the MFI folder
current_path = os.getcwd()
while os.path.basename(current_path) != "MFI_git" and os.path.basename(current_path) != "MFI":
    parent_path = os.path.dirname(current_path)
    if parent_path == current_path: raise Exception("MFI folder not found in parent directories. Please start the script from the MFI_git folder or its subfolders.")
    current_path = parent_path

path_MFI = current_path
path_pyMFI = os.path.join(path_MFI, "pyMFI")
path_tests = os.path.join(path_MFI, "tests_pyMFI")
path_data = os.path.join(path_tests, "data")
print(f"Path of MFI folder: {path_MFI = }")
print(f"Path of pyMFI folder: {path_pyMFI = }")
print(f"Path of tests folder: {path_tests = }")
print(f"Path of data folder: {path_data = }")


sys.path.append(path_MFI)
from pyMFI import MFI
from pyMFI import MFI1D
from pyMFI import run_plumed

class TestMFI(unittest.TestCase):
    
    """
    TestMFI is a unittest.TestCase class that contains various test methods
    for validating the functionality of the MFI (Multidimensional Free Energy)
    simulations and analyses in both 1D and 2D.

    The tests include:
    - Langevin simulations in 1D and 2D
    - MFI calculations in 1D and 2D
    - Patch simulations in 1D and 2D
    - Bootstrapping methods in 1D and 2D
    - Integration tests for both 1D and 2D analytical data

    To run the tests, use the following command in the terminal:
    python -m unittest /home/ucecabj/Desktop/MFI_git/tests_pyMFI/run_pyMFI_tests.py
    """    
    
    print("\n\n")
        
    def test_langevin_simulation_1D(self):
        
        print("--> Running 1D langevin simulation test")
        
        os.chdir(path_data)
        run_plumed.run_langevin1D(simulation_steps = 100_000, initial_position=1.7, gaus_width=0.1, gaus_height=0.5, biasfactor=10, gaus_pace=100)
        
        HILLS=MFI1D.load_HILLS(hills_name=path_data+"/HILLS")
        position = MFI1D.load_position(position_name=path_data+"/position")
        
        self.assertTrue(HILLS.shape == (1000, 5))
        self.assertTrue(position.shape == (10000,))
        self.assertTrue(HILLS[-5,0] > 490)
        self.assertTrue(abs(HILLS[-5,1]) < 10)
        self.assertTrue(HILLS[-5,3] < 1 and HILLS[-5,3] > 0)
        self.assertTrue(abs(position[-5]) < 10)
        
        os.system("rm input plumed.dat HILLS position stats.out")
    
    def test_MFI1D(self):
        
        print("--> Running 1D MFI test (requires compilation of numba functions)")
        
        # load data
        HILLS=MFI1D.load_HILLS(hills_name=path_data+"/HILLS_1D")
        position = MFI1D.load_position(position_name=path_data+"/position_1D")
        
        # analyse data
        results = MFI1D.MFI_1D(HILLS = HILLS, position = position, bw = 0.02, kT = 1, min_grid=-2.0, max_grid=2.0, nhills=-1, error_pace=int(len(HILLS)/200), log_pace=len(HILLS), use_weighted_st_dev=False)
        X, Ftot_den, Ftot_den2, Ftot, ofv_num, FES, ofv, ofe, cutoff, error_evol, fes_error_cutoff_evol = results

        # compare results
        y = 7*X**4 - 23*X**2
        y = y - min(y)
        AD = abs(FES - y)
        AAD = np.mean(AD)
        self.assertTrue(AAD < 0.1 and AAD > 1E-10)
        
        
    def test_patch_simulations_1D(self):
        
        print("--> Running 1D patch test ")
        
        all_forceterms = []
        min_grid=0; max_grid=275; nbins=2201; X = np.linspace(min_grid, max_grid, nbins)

        for i in range(1,6):
            all_forceterms.append(MFI1D.load_pkl(path_data + "/forces_forw_"+str(i)+"_1D") )
            all_forceterms.append(MFI1D.load_pkl(path_data + "/forces_back_"+str(i)+"_1D") )

        Ftot_den, Ftot_den2, Ftot, ofv_num, ofe, Aofe = MFI1D.patch_forces_ofe(np.asarray(all_forceterms), ofe_progression=True, use_weighted_st_dev=True)

        expected_Aofe = [0.46935202, 1.52425945, 1.25007619, 1.57995023, 1.38038675, 1.5779447,  1.45337313, 1.59059732, 1.49568188, 1.59485419]
        for i in range(len(Aofe)): self.assertTrue(abs(Aofe[i] - expected_Aofe[i]) < 1E-3)
        
        
    def test_bootstraping_1D(self):
        
        print("--> Running 1D bootstraping test")
        
        forceterms_forw = []; forceterms_back = []
        min_grid=0; max_grid=275; nbins=2201; X = np.linspace(min_grid, max_grid, nbins)

        #load force terms
        for i in range(1,6):
            [Ftot_den, Ftot_den2, Ftot, ofv_num] = MFI1D.load_pkl(path_data+"/forces_forw_"+str(i)+"_1D")
            forceterms_forw.append( [Ftot_den, Ftot] )
            [Ftot_den, Ftot_den2, Ftot, ofv_num] = MFI1D.load_pkl(path_data+"/forces_back_"+str(i)+"_1D")
            forceterms_back.append( [Ftot_den, Ftot] )
            
        #Boostrap
        [FES_avr, sd_fes, sd_fes_prog ] = MFI1D.bootstrap_forw_back(X, np.asarray(forceterms_forw), np.asarray(forceterms_back), n_bootstrap=50, set_fes_minima="first_value")
        
        for i in range(3, len(sd_fes_prog)): self.assertTrue(sd_fes_prog[i] < 2)

    
    def test_langevin_simulation_2D(self):
        
        print("--> Running 2D langevin simulation test")
        run_plumed.run_2D_Invernizzi(simulation_steps=100_000, sigma=0.1, height=1, biasfactor=10, initial_position_x=-1, initial_position_y=-1, file_extension="2D")

        # load data
        HILLS = MFI.load_HILLS_2D(hills_name=path_data+"/HILLSinve_2D")
        px, py = MFI.load_position_2D(position_name=path_data+"/positioninve_2D")

        self.assertTrue(HILLS.shape == (200, 7))
        self.assertTrue(px.shape == (2000,))
        self.assertTrue(py.shape == (2000,))
        self.assertTrue(HILLS[-5,0] > 100)
        self.assertTrue(abs(HILLS[-5,1]) < 10)
        self.assertTrue(abs(HILLS[-5,2]) < 10)
        self.assertTrue(HILLS[-5,5] < 2 and HILLS[-5,5] > 0)
        self.assertTrue(abs(px[-5]) < 10)        
        self.assertTrue(abs(py[-5]) < 10)        
  
        os.system("rm input plumed.dat positioninve_2D HILLSinve_2D stats.out")
    
    def test_MFI_2D(self):
        
        print("--> Running 2D MFI test")
        
        # define analytical surface
        gridx, gridy = np.linspace(-3, 3, 200), np.linspace(-2.5, 2.5, 400); 
        X, Y = np.meshgrid(gridx, gridy)
        inve_pot = 1.34549*X**4+1.90211*X**3*Y+3.92705*X**2*Y**2-6.44246*X**2-1.90211*X*Y**3+5.58721*X*Y+1.33481*X+1.34549*Y**4-5.55754*Y**2+0.904586*Y+18.5598
        inve_pot = inve_pot - np.min(inve_pot)
        
        # load data
        os.chdir(path_data)
        HILLS=MFI.load_HILLS_2D(hills_name="HILLSinve_long")
        [position_x, position_y] = MFI.load_position_2D(position_name="positioninve_long")
        
        # analyse data
        results = MFI.MFI_2D(HILLS = HILLS, position_x = position_x, position_y = position_y, bw = [0.12,0.12], nhills=int(len(HILLS)/2),
                    min_grid=[-3, -2.5], max_grid=[3, 2.5], nbins=[200,400], error_pace = int(len(HILLS)/5), FES_cutoff = 20, use_weighted_st_dev=False)
        [X, Y, Ftot_den, Ftot_x, Ftot_y, ofv, ofe, cutoff, volume_history, ofe_history, _ , time_history, Ftot_den2, ofv_num_x, ofv_num_y] = results
        
        [X, Y, FES] = MFI.FFT_intg_2D(Ftot_x, Ftot_y, min_grid=np.array((-3, -2.5)), max_grid=np.array((3, 2.5)))
        AD = abs(inve_pot - FES)*cutoff
        AAD = np.sum(AD) / np.count_nonzero(cutoff)
        
        # compare results
        self.assertTrue(AAD < 1.6 and AAD > 1E-15)
        
        
    def test_patch_simulations_2D(self):
        print("--> Running 2D patch test ")

        os.chdir(path_data)

        # define analytical surface
        gridx, gridy = np.linspace(-3, 3, 200), np.linspace(-2.5, 2.5, 400); 
        X, Y = np.meshgrid(gridx, gridy)
        inve_pot = 1.34549*X**4+1.90211*X**3*Y+3.92705*X**2*Y**2-6.44246*X**2-1.90211*X*Y**3+5.58721*X*Y+1.33481*X+1.34549*Y**4-5.55754*Y**2+0.904586*Y+18.5598
        inve_pot = inve_pot - np.min(inve_pot)
                
        n_simulations = 20
        master = []

        for simulation in np.arange(0,n_simulations):   
            [Ftot_den, Ftot_x, Ftot_y, volume_history, ofe_history, time_history, Ftot_den2, ofv_num_x, ofv_num_y] = MFI.load_pkl(path_data + "/forces_inve_" + str(simulation))
            master.append([Ftot_den, Ftot_den2, Ftot_x, Ftot_y, ofv_num_x, ofv_num_y])
            
        master_patch = MFI.patch_2D(master)
        [X, Y, FES] = MFI.FFT_intg_2D(master_patch[2], master_patch[3], min_grid=[-3, -2.5], max_grid=[3, 2.5])
        cutoff = MFI.get_cutoff(Ftot_den=master_patch[0], FX=master_patch[2], FY=master_patch[3], FES_cutoff=[20, [-3, -2.5], [3, 2.5], [0,0]]) 
        AD = abs(inve_pot[::2,:] - FES)*cutoff
        AAD = np.sum(AD) / np.count_nonzero(cutoff)
        
        self.assertTrue(AAD < 3.1 and AAD > 1E-15)
        
    def test_bootstraping_2D(self):
        print("--> Running 2D bootstraping test")
        
        os.chdir(path_data)

        # define analytical surface
        gridx, gridy = np.linspace(-3, 3, 200), np.linspace(-2.5, 2.5, 400); 
        X, Y = np.meshgrid(gridx, gridy)
        inve_pot = 1.34549*X**4+1.90211*X**3*Y+3.92705*X**2*Y**2-6.44246*X**2-1.90211*X*Y**3+5.58721*X*Y+1.33481*X+1.34549*Y**4-5.55754*Y**2+0.904586*Y+18.5598
        inve_pot = inve_pot - np.min(inve_pot)
                
        n_simulations = 20
        forces_all, master = [], []

        for simulation in np.arange(0,n_simulations):   
            [Ftot_den, Ftot_x, Ftot_y, volume_history, ofe_history, time_history, Ftot_den2, ofv_num_x, ofv_num_y] = MFI.load_pkl(path_data + "/forces_inve_" + str(simulation))
            forces_all.append([Ftot_den, Ftot_x, Ftot_y])
            master.append([Ftot_den, Ftot_den2, Ftot_x, Ftot_y, ofv_num_x, ofv_num_y])
            
        master_patch = MFI.patch_2D(master)
        [X, Y, FES] = MFI.FFT_intg_2D(master_patch[2], master_patch[3], min_grid=[-3, -2.5], max_grid=[3, 2.5])
        cutoff = MFI.get_cutoff(Ftot_den=master_patch[0], FX=master_patch[2], FY=master_patch[3], FES_cutoff=[20, [-3, -2.5], [3, 2.5], [0,0]]) 
        AD = abs(inve_pot[::2,:] - FES)*cutoff
        AAD = np.sum(AD) / np.count_nonzero(cutoff)

        [FES_avr, sd_fes, sd_fes_prog ] = MFI.bootstrap_2D_new(X, Y, np.array(forces_all), n_bootstrap=100, FES_cutoff=20) 
        AD_FES_avr = abs(inve_pot[::2,:] - FES_avr)*cutoff
        AAD_FES_avr = np.sum(AD_FES_avr) / np.count_nonzero(cutoff)

        self.assertTrue(AAD < 3.1 and AAD > 1E-15)
        self.assertTrue(AAD_FES_avr < 3.1 and AAD_FES_avr > 1E-15)
        
    def test_intg_1D(self):
        
        print("--> Running integration test with 1D analytical data")

        
        x = np.linspace(-2, 2, 500)
        y = 7*x**4 - 23*x**2
        y = y - y.min()
        dy = 28*x**3 - 46*x

        fes = MFI1D.intg_1D(dy, dx=x[1]-x[0])
        AD = abs(fes - y)
        AAD = AD.mean()
        
        self.assertTrue(AAD < 0.01 and AAD > 1E-15)
    
    def test_intg_2D_analytical(self):
        
        print("--> Running integration test with 2D analytical data")

        min_grid=np.array((-3, -2.6)); 
        max_grid=np.array((3.5, 2.5)); 
        nbins = np.array((200, 400))

        gridx = np.linspace(min_grid[0], max_grid[0], nbins[0]); 
        gridy = np.linspace(min_grid[1], max_grid[1], nbins[1])        
        X, Y = np.meshgrid(gridx, gridy)

        # Define analytical surface and its gradient
        F = 1.34549*X**4+1.90211*X**3*Y+3.92705*X**2*Y**2-6.44246*X**2-1.90211*X*Y**3+5.58721*X*Y+1.33481*X+1.34549*Y**4-5.55754*Y**2+0.904586*Y+18.5598
        F = F - np.min(F)
        fx = 4*1.34549*X**3+3*1.90211*X**2*Y+2*3.92705*X*Y**2-2*6.44246*X-1.90211*Y**3+5.58721*Y+1.33481
        fy = 1.90211*X**3+2*3.92705*X**2*Y-3*1.90211*X*Y**2+5.58721*X+4*1.34549*Y**3-2*5.55754*Y+0.904586
        
        [_, _, FES] = MFI.FFT_intg_2D(fx, fy, min_grid=min_grid, max_grid=max_grid, periodic=[0,0])

        AD = abs(F - FES)
        AAD = np.mean(AD)
        
        # compare results
        self.assertTrue(AAD < 0.1 and AAD > 1E-15)
        
        
    def test_intg_2D_simulated(self):
        
        print("--> Running integration test with 2D simulated data")
        
        [FES, Force_x, Force_y] = MFI.load_pkl(path_data + "/inve_forces_intg.pkl")
        
        min_grid=np.array((-3, -2.5)); 
        max_grid=np.array((3, 2.5)); 
        nbins = np.array((200, 400))

        gridx = np.linspace(min_grid[0], max_grid[0], nbins[0]); 
        gridy = np.linspace(min_grid[1], max_grid[1], nbins[1])
        X, Y = np.meshgrid(gridx, gridy)

        Z = 1.34549*X**4+1.90211*X**3*Y+3.92705*X**2*Y**2-6.44246*X**2-1.90211*X*Y**3+5.58721*X*Y+1.33481*X+1.34549*Y**4-5.55754*Y**2+0.904586*Y+18.5598
        Z = Z - np.min(Z)
        
        [_, _, FES] = MFI.FFT_intg_2D(Force_x, Force_y, min_grid=min_grid, max_grid=max_grid, periodic=[0,0])

        AD = abs(Z - FES)
        AD = np.where(Z < 20, AD, 0)
        AAD_cut = np.sum(AD) / np.count_nonzero(AD)

        # compare results
        self.assertTrue(AAD_cut < 1.0 and AAD_cut > 1E-15)



if __name__ == '__main__':
    unittest.main()

# To run all tests
# python -m unittest discover tests