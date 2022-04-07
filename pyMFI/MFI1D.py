import glob
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import numpy as np

def load_HILLS(hills_name = "HILLS"):
    for file in glob.glob(hills_name):
        hills = np.loadtxt(file)
        hills = hills[:-1]
        hills0 = hills[0]
        hills0[3] = 0
        hills = np.concatenate(([hills0],hills))
    return hills

#Load the trajectory (position) data
def load_position(position_name = "position"):
    for file1 in glob.glob(position_name):
        colvar = np.loadtxt(file1)
    return colvar[:-1, 1]

### Algorithm to run 1D MFI
#Run MFI algorithm with on the fly error calculation
def MFI_1D(HILLS = "HILLS", position = "position", bw = 1, kT = 1, min_grid=2, max_grid=2, nbins = 101, log_pace = 10, error_pace = 200, WellTempered=0):    
    
    grid = np.linspace(min_grid, max_grid, nbins)
    stride = int(len(position) / len(HILLS[:,1]))     
    const = (1 / (bw*np.sqrt(2*np.pi)*stride))
    total_number_of_hills=len(HILLS[:,1])
    bw2 = bw**2    

    # initialise force terms
    Fbias = np.zeros(len(grid))
    Ftot_num = np.zeros(len(grid))
    Ftot_den = np.zeros(len(grid))
    Ftot_den2 = np.zeros(len(grid))
    ofv = np.zeros(len(grid))
    ofe_history = []

    # Definition Gamma Factor, allows to switch between WT and regular MetaD
    if WellTempered < 1: 
        Gamma_Factor=1
    else:
        gamma = HILLS[0, 4]
        Gamma_Factor=(gamma - 1)/(gamma)


    for i in range(total_number_of_hills):
        # Build metadynamics potential
        s = HILLS[i, 1]  # center position of Gaussian
        sigma_meta2 = HILLS[i, 2] ** 2  # width of Gaussian
        gamma = HILLS[i, 4]  # scaling factor of Gaussian
        height_meta = HILLS[i, 3] * Gamma_Factor  # Height of Gaussian

        kernelmeta = np.exp(-0.5 * (((grid - s) ** 2) / (sigma_meta2)))
        Fbias = Fbias + height_meta * kernelmeta * ((grid - s) / (sigma_meta2))  # Bias force due to Metadynamics potentials

        # Estimate the biased proabability density
        pb_t = np.zeros(len(grid))
        Fpbt = np.zeros(len(grid))
        data = position[i * stride: (i + 1) * stride]  # positons of window of constant bias force.
        for j in range(stride):
            kernel = const * np.exp(- (grid - data[j]) ** 2 / (2 * bw2))  # probability density of 1 datapoint
            pb_t = pb_t + kernel  # probability density of window
            Fpbt = Fpbt + kT * kernel * (grid - data[j]) / bw2

        # Estimate of the Mean Force and error  for terms
        Ftot_den = Ftot_den + pb_t  # total probability density
        dfds = np.divide(Fpbt, pb_t, out=np.zeros_like(Fpbt), where=pb_t != 0) + Fbias
        Ftot_num = Ftot_num + pb_t * dfds
        Ftot = np.divide(Ftot_num, Ftot_den, out=np.zeros_like(Ftot_num), where=Ftot_den != 0)  # total force
        
        #additional terms for error calculation
        Ftot_den2 = Ftot_den2 + pb_t ** 2 #sum of (probability densities)^2
        ofv = ofv + pb_t * (dfds ** 2)   #sum of (weighted mean force of window)^2

        #Calculate error
        if (i + 1) % int(total_number_of_hills / error_pace) == 0:
            #ofe 
            Ftot_den_ratio = np.divide(Ftot_den2, (Ftot_den ** 2 - Ftot_den2), out=np.zeros_like(Ftot_den), where=Ftot_den > 1E-10)
            ofe = np.divide(ofv, Ftot_den, out=np.zeros_like(ofv), where=Ftot_den > 1E-10) - Ftot ** 2
            ofe = ofe * Ftot_den_ratio
            ofe = np.sqrt(ofe)
            ofe_history.append(sum(ofe)/nbins)
            if (i + 1) % int(total_number_of_hills / log_pace) == 0:
                print(str(round((i + 1) / total_number_of_hills * 100, 0)) + "%   OFE =", round(ofe_history[-1],4))
                 
    return [grid, Ftot_den, Ftot, ofe, ofe_history]

        
# Integrate Ftot, obtain FES 
def intg_1D(x,F):
    fes = []
    for j in range(len(x)): fes.append(integrate.simps(F[:j + 1], x[:j + 1]))
    fes = fes - min(fes)
    return fes


def plot_recap(X, FES, TOTAL_DENSITY, CONVMAP, CONV_history,lim=40): 
    fig, axs = plt.subplots(2,2,figsize=(12,8))
    
    axs[0,0].plot(X,FES);
    axs[0,0].set_ylim([0,lim])
    axs[0,0].set_ylabel('F(CV1)')
    axs[0,0].set_xlabel('CV1')
    
    axs[0,1].plot(X,CONVMAP);
    axs[0,1].set_ylabel('Mean Force Error')
    axs[0,1].set_xlabel('CV1')
    
    axs[1,0].plot(X,TOTAL_DENSITY);
    axs[1,0].set_ylabel('Count')
    axs[1,0].set_xlabel('CV1')
    
    axs[1,1].plot(range(len(CONV_history)), CONV_history);
    axs[1,1].set_ylabel('Average Mean Force Error')
    axs[1,1].set_xlabel('Number of Error Evaluations')