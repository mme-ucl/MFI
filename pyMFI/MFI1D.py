import glob
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import numpy as np
import pickle
import random


def load_HILLS(hills_name="HILLS"):
    for file in glob.glob(hills_name):
        hills = np.loadtxt(file)
        hills = hills[:-1]
        hills0 = hills[0]
        hills0[3] = 0
        hills = np.concatenate(([hills0], hills))
    return hills


# Load the trajectory (position) data
def load_position(position_name="position"):
    for file1 in glob.glob(position_name):
        colvar = np.loadtxt(file1)
    return colvar[:-1, 1]

#define indexing
def index(position, min_grid, grid_space):
    return int((position-min_grid)//grid_space) + 1


def find_periodic_point(x_coord, min_grid, max_grid, periodic):
    if periodic == 1:
        coord_list = []
        #There are potentially 2 points, 1 original and 1 periodic copy.
        coord_list.append(x_coord)
        #define grid extension
        grid_ext = 0.25 * (max_grid-min_grid)
        #check for copy
        if x_coord < min_grid+grid_ext: coord_list.append(x_coord + 2*np.pi)
        elif x_coord > max_grid-grid_ext: coord_list.append(x_coord - 2*np.pi)

        return coord_list
    else:
        return [x_coord]


def find_hp_force(hp_center, hp_kappa, grid, min_grid, max_grid, grid_space, periodic):
    F_harmonic = hp_kappa * (grid - hp_center)
    if periodic == 1:
        grid_length = max_grid - min_grid
        grid_centre = min_grid + grid_length/2
        if hp_center < grid_centre:
            index_period = index(hp_center + grid_length / 2, min_grid, grid_space)
            F_harmonic[index_period:] = hp_kappa * (grid[index_period:] - hp_center - grid_length)
        elif hp_center > grid_centre:
            index_period = index(hp_center - grid_length / 2, min_grid, grid_space)
            F_harmonic[:index_period] = hp_kappa * (grid[:index_period] - hp_center + grid_length)

    return F_harmonic

def find_lw_force(lw_center, lw_kappa, grid, min_grid, max_grid, grid_space, periodic):
    F_harmonic = np.where(grid < lw_center, 2 * lw_kappa * (grid - lw_center), 0)
    if periodic == 1:
        grid_length = max_grid - min_grid
        grid_centre = min_grid + grid_length/2
        if lw_center < grid_centre:
            index_period = index(lw_center + grid_length / 2, min_grid, grid_space)
            F_harmonic[index_period:] = 2 * lw_kappa * (grid[index_period:] - lw_center - grid_length)
        elif lw_center > grid_centre:
            index_period = index(lw_center - grid_length / 2, min_grid, grid_space)
            F_harmonic[:index_period] = 0

    return F_harmonic

def find_uw_force(uw_center, uw_kappa, grid, min_grid, max_grid, grid_space, periodic):
    F_harmonic = np.where(grid > uw_center, uw_kappa * (grid - uw_center), 0)
    if periodic == 1:
        grid_length = max_grid - min_grid
        grid_centre = min_grid + grid_length/2
        if uw_center < grid_centre:
            index_period = index(uw_center + grid_length / 2, min_grid, grid_space)
            F_harmonic[index_period:] = 0
        elif uw_center > grid_centre:
            index_period = index(uw_center - grid_length / 2, min_grid, grid_space)
            F_harmonic[:index_period] = 2 * uw_kappa * (grid[:index_period] - uw_center + grid_length)
    
    return F_harmonic


### Algorithm to run 1D MFI
# Run MFI algorithm with on the fly error calculation
def MFI_1D(HILLS="HILLS", position="position", bw=1, kT=1, min_grid=2, max_grid=2, nbins=101, log_pace=10,
           error_pace=200, WellTempered=0, periodic=0, hp_center=0.0, hp_kappa=0, lw_center=0.0, lw_kappa=0,
           uw_center=0.0, uw_kappa=0):
    grid = np.linspace(min_grid, max_grid, nbins)
    grid_space = (max_grid - min_grid) / (nbins-1)
    stride = int(len(position) / len(HILLS[:, 1]))
    const = (1 / (bw * np.sqrt(2 * np.pi) * stride))
    total_number_of_hills = len(HILLS[:, 1])
    bw2 = bw ** 2

    # initialise force terms
    Fbias = np.zeros(len(grid))
    Ftot_num = np.zeros(len(grid))
    Ftot_den = np.zeros(len(grid))
    Ftot_den2 = np.zeros(len(grid))
    ofv = np.zeros(len(grid))
    ofe_history = []

    #Calculate static force (form harmonic or wall potential)
    F_static = np.zeros(nbins)
    if hp_kappa > 0: F_static += find_hp_force(hp_center, hp_kappa, grid, min_grid, max_grid, grid_space, periodic)
    if lw_kappa > 0: F_static += find_lw_force(lw_center, lw_kappa, grid, periodic)
    if uw_kappa > 0: F_static += find_uw_force(uw_center, uw_kappa, grid, periodic)

    # Definition Gamma Factor, allows to switch between WT and regular MetaD
    if WellTempered < 1:
        Gamma_Factor = 1
    else:
        gamma = HILLS[0, 4]
        Gamma_Factor = (gamma - 1) / (gamma)

    for i in range(total_number_of_hills):
        # Build metadynamics potential
        s = HILLS[i, 1]  # center position of Gaussian
        sigma_meta2 = HILLS[i, 2] ** 2  # width of Gaussian
        height_meta = HILLS[i, 3] * Gamma_Factor  # Height of Gaussian

        periodic_images = find_periodic_point(s, min_grid, max_grid, periodic)
        for j in range(len(periodic_images)):
            kernelmeta = np.exp(-0.5 * (((grid - periodic_images[j]) ** 2) / (sigma_meta2)))
            Fbias = Fbias + height_meta * kernelmeta * ((grid - periodic_images[j]) / (sigma_meta2))  # Bias force due to Metadynamics potentials

        # Estimate the biased proabability density
        pb_t = np.zeros(len(grid))
        Fpbt = np.zeros(len(grid))
        data = position[i * stride: (i + 1) * stride]  # positons of window of constant bias force.
        for j in range(stride):
            periodic_images = find_periodic_point(data[j], min_grid, max_grid, periodic)
            for k in range(len(periodic_images)):
                kernel = const * np.exp(- (grid - periodic_images[k]) ** 2 / (2 * bw2))  # probability density of 1 datapoint
                pb_t = pb_t + kernel  # probability density of window
                Fpbt = Fpbt + kT * kernel * (grid - periodic_images[k]) / bw2

        # Estimate of the Mean Force and error  for terms
        Ftot_den = Ftot_den + pb_t  # total probability density
        dfds = np.divide(Fpbt, pb_t, out=np.zeros_like(Fpbt), where=pb_t != 0) + Fbias - F_static
        Ftot_num = Ftot_num + pb_t * dfds
        Ftot = np.divide(Ftot_num, Ftot_den, out=np.zeros_like(Ftot_num), where=Ftot_den != 0)  # total force

        # additional terms for error calculation
        Ftot_den2 = Ftot_den2 + pb_t ** 2  # sum of (probability densities)^2
        ofv = ofv + pb_t * (dfds ** 2)  # sum of (weighted mean force of window)^2

        # Calculate error
        if (i + 1) % int(total_number_of_hills / error_pace) == 0:
            # ofe
            Ftot_den_ratio = np.divide(Ftot_den2, (Ftot_den ** 2 - Ftot_den2), out=np.zeros_like(Ftot_den),
                                       where=Ftot_den > 1E-10)
            ofe = np.divide(ofv, Ftot_den, out=np.zeros_like(ofv), where=Ftot_den > 1E-10) - Ftot ** 2
            ofe = ofe * Ftot_den_ratio
            ofe = np.sqrt(ofe)
            ofe_history.append(sum(ofe) / nbins)
            if (i + 1) % int(total_number_of_hills / log_pace) == 0:
                print(str(round((i + 1) / total_number_of_hills * 100, 0)) + "%   OFE =", round(ofe_history[-1], 4))

    return [grid, Ftot_den, Ftot, ofe, ofe_history]


# Integrate Ftot, obtain FES
def intg_1D(x, F):
    fes = []
    for j in range(len(x)): fes.append(integrate.simps(F[:j + 1], x[:j + 1]))
    fes = fes - min(fes)
    return fes


def plot_recap(X, FES, TOTAL_DENSITY, CONVMAP, CONV_history, lim=40):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].plot(X, FES);
    axs[0, 0].set_ylim([0, lim])
    axs[0, 0].set_ylabel('F(CV1)')
    axs[0, 0].set_xlabel('CV1')

    axs[0, 1].plot(X, CONVMAP);
    axs[0, 1].set_ylabel('Mean Force Error')
    axs[0, 1].set_xlabel('CV1')

    axs[1, 0].plot(X, TOTAL_DENSITY);
    axs[1, 0].set_ylabel('Count')
    axs[1, 0].set_xlabel('CV1')

    axs[1, 1].plot(range(len(CONV_history)), CONV_history);
    axs[1, 1].set_ylabel('Average Mean Force Error')
    axs[1, 1].set_xlabel('Number of Error Evaluations')



# def patch_to_base_error(master0, master, return_terms=False):
#     [PD0, PD20, F0, OFV0] = master0
#     [PD, PD2, F, OFV] = master

#     # Patch master terms together
#     PD_patch = PD0 + PD
#     PD2_patch = PD20 + PD2
#     OFV_patch = OFV0 + OFV
#     F_patch = PD0 * F0 + PD * F
#     F_patch = np.divide(F_patch, PD_patch, out=np.zeros_like(F_patch), where=PD_patch > 0.001)

#     # calculate error
#     PD_ratio = np.divide(PD2_patch, (PD_patch ** 2 - PD2_patch), out=np.zeros_like(PD_patch), where=PD_patch > 1)
#     OFE = np.divide(OFV_patch, PD_patch, out=np.zeros_like(OFV_patch), where=PD_patch > 1) - F_patch ** 2
#     OFE = OFE * PD_ratio
#     OFE = np.sqrt(OFE)

#     # save terms
#     ofe_history.append(sum(OFE) / (np.count_nonzero(OFE)))

#     if return_terms == True:
#         return [PD_patch, PD2_patch, F_patch, OFV_patch, OFE]
#     else:
#         print(" ofe is: " + str(ofe_history[-1]))
#         return OFE

def patch_forces(force_vector):
    PD_patch = np.zeros(np.shape(force_vector[0][0]))
    F_patch = np.zeros(np.shape(force_vector[0][0]))
    for i in range(len(force_vector)):
        PD_patch += force_vector[i][0]
        F_patch += force_vector[i][0] * force_vector[i][1]
    F_patch = np.divide(F_patch, PD_patch, out=np.zeros_like(F_patch), where=PD_patch > 0.000001)

    return [PD_patch, F_patch]



def bootstrap_forw_back(grid, forward_force, backward_force, n_bootstrap):
   
    #Define terms that will be updated itteratively
    Ftot_inter = np.zeros(len(grid))
    Ftot_sum = np.zeros(len(grid))
    Ftot_den_sum = np.zeros(len(grid))
    Ftot_den2_sum = np.zeros(len(grid))
    FES_sum = np.zeros(len(grid))
    FES2_sum = np.zeros(len(grid))

    #store var and sd progression here
    variance_prog = []
    stdev_prog = []
    var_fes_prog  = []
    sd_fes_prog = []

    #Save patch force terms and FES
    force_patch_collection = []
    FES_collection = []

    #Patch forces
    [Ftot_den, Ftot] = patch_forces(np.concatenate((forward_force, backward_force)))

    #save non-random probability density
    Ftot_den_base = np.array(Ftot_den)


    for itteration in range(n_bootstrap):

        #Randomly choose 49 forward forces and backward forces
        force = []    
        for i in range(len(forward_force)):
            force.append(forward_force[random.randint(0,len(forward_force)-1)])
            force.append(backward_force[random.randint(0,len(forward_force)-1)])

                
                
        #patch forces to find average Ftot_den, Ftot and FES
        [Ftot_den, Ftot] = patch_forces(force)
        FES = intg_1D(grid,Ftot)
        FES = FES - FES[0]

        #Save terms
        force_patch_collection.append([Ftot_den, Ftot])
        FES_collection.append(FES)

        #calculate sums for variance
        Ftot_inter += Ftot_den * Ftot**2
        Ftot_sum += Ftot
        Ftot_den_sum += Ftot_den
        Ftot_den2_sum += Ftot_den**2
        
        FES_sum += FES
        FES2_sum += FES**2

        if itteration > 0:
            
            #calculate force variance
            Ftot_avr = Ftot_sum / (itteration+1)
            Ftot2_weighted = np.divide(Ftot_inter, Ftot_den_sum, out=np.zeros_like(Ftot_inter), where=Ftot_den_base>10)
            Ftot_den_ratio = np.divide(Ftot_den_sum ** 2, (Ftot_den_sum ** 2 - Ftot_den2_sum), out=np.zeros_like(Ftot_den_sum), where=Ftot_den_base > 10)
            variance = (Ftot2_weighted - Ftot_avr**2) * Ftot_den_ratio
            n_eff = np.divide(Ftot_den_sum ** 2, Ftot_den2_sum, out=np.zeros_like(Ftot_den), where=Ftot_den_base>10)
            stdev = np.where(Ftot_den_base > 10,  np.sqrt(variance / n_eff ), 0)
        
            #calculate FES variance
            FES_avr = FES_sum/ (itteration+1)
            var_fes = np.zeros(len(grid))
            for i in range(len(FES_collection)): 
                var_fes += (FES_collection[i] - FES_avr)**2
            var_fes = 1/(len(FES_collection)-1) * var_fes
            sd_fes = np.sqrt(var_fes)
            
            #save variance
            variance_prog.append(sum(variance)/len(grid))
            stdev_prog.append(sum(stdev)/len(grid))
            var_fes_prog.append(sum(var_fes)/len(grid))
            sd_fes_prog.append(sum(sd_fes)/len(grid))
        
        
        #print progress
        if (itteration+1) % 50 == 0:
            print(itteration+1, ": var:", round(variance_prog[-1],5), "     sd:", round(stdev_prog[-1],5), "      FES: var:", round(var_fes_prog[-1],3), "     sd:", round(sd_fes_prog[-1],3) )
            
    return [FES_avr, sd_fes, variance_prog, stdev_prog, var_fes_prog, sd_fes_prog ]



def save_npy(object, file_name):
    with open(file_name, "wb") as fw:
        np.save(fw, object)


def load_npy(name):
    with open(name, "rb") as fr:
        return np.load(fr)

def save_pkl(object, file_name):
    with open(file_name, "wb") as fw:
        pickle.dump(object, fw)


def load_pkl(name):
    with open(name, "rb") as fr:
        return pickle.load(fr)