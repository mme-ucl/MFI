import glob
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import numpy as np


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
        if hp_center < 0:
            index_period = index(hp_center + (max_grid - min_grid) / 2, min_grid, grid_space)
            F_harmonic[index_period:] = hp_kappa * (grid[index_period:] - (hp_center + (max_grid - min_grid)))
        elif hp_center > 0:
            index_period = index(hp_center - (max_grid - min_grid) / 2, min_grid, grid_space)
            F_harmonic[:index_period] = hp_kappa * (grid[:index_period] - (hp_center - (max_grid - min_grid)))

    return F_harmonic

def find_lw_force(lw_center, lw_kappa, grid, periodic):
    F_harmonic = np.where(grid < lw_center, lw_kappa * (grid - lw_center), 0)
    if periodic == 1:
        print("\n\n***ATTENTION, UPPER WALL FORCE DOESN'T CONTAIN PERIODIC FEATURES***\n\n")
    return F_harmonic

def find_uw_force(uw_center, uw_kappa, grid, periodic):
    F_harmonic = np.where(grid > uw_center, uw_kappa * (grid - uw_center), 0)
    if periodic == 1:
        print("\n\n***ATTENTION, UPPER WALL FORCE DOESN'T CONTAIN PERIODIC FEATURES***\n\n")
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
