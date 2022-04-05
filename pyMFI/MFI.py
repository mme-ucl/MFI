import glob
import matplotlib.pyplot as plt
import numpy as np

### Load files ####
def load_HILLS_2D(hills_name = "HILLS"):
    for file in glob.glob(hills_name):
        hills = np.loadtxt(file)
        hills = np.concatenate(([hills[0]], hills[:-1]))
        hills[0][5] = 0
    return hills

def load_position_2D(position_name = "position"):
    for file1 in glob.glob(position_name):
        colvar = np.loadtxt(file1)
        position_x = colvar[:-1, 1]
        position_y = colvar[:-1, 2]
    return [position_x, position_y]
#######

### Periodic CVs utils
def find_periodic_point(x_coord,y_coord,min_grid,max_grid,periodic):
    
    #Use periodic extension for defining PBC
    periodic_extension = periodic * 1 / 2
    grid_ext = (1/2) * periodic_extension * (max_grid-min_grid)
    
    coord_list = []
    #There are potentially 4 points, 1 original and 3 periodic copies
    coord_list.append([x_coord,y_coord])
    copy_record = [0,0,0,0]
    #check for x-copy
    if x_coord < min_grid[0]+grid_ext[0]:
        coord_list.append([x_coord + 2*np.pi,y_coord])
        copy_record[0] = 1
    elif x_coord > max_grid[0]-grid_ext[0]:
        coord_list.append([x_coord - 2*np.pi,y_coord])
        copy_record[1] = 1
    #check for y-copy
    if y_coord < min_grid[1]+grid_ext[1]:
        coord_list.append([x_coord, y_coord + 2 * np.pi])
        copy_record[2] = 1
    elif y_coord > max_grid[1]-grid_ext[1]:
        coord_list.append([x_coord, y_coord - 2 * np.pi])
        copy_record[3] = 1
    #check for xy-copy
    if sum(copy_record) == 2:
        if copy_record[0] == 1 and copy_record[2] == 1: coord_list.append([x_coord + 2 * np.pi, y_coord + 2 * np.pi])
        elif copy_record[1] == 1 and copy_record[2] == 1: coord_list.append([x_coord - 2 * np.pi, y_coord + 2 * np.pi])
        elif copy_record[0] == 1 and copy_record[3] == 1: coord_list.append([x_coord + 2 * np.pi, y_coord - 2 * np.pi])
        elif copy_record[1] == 1 and copy_record[3] == 1: coord_list.append([x_coord - 2 * np.pi, y_coord - 2 * np.pi])

    return coord_list

def find_cutoff_matrix(input_FES):
    len_x, len_y = np.shape(input_FES)
    cutoff_matrix = np.ones((len_x, len_y))
    for ii in range(len_x):
        for jj in range(len_y):
            if input_FES[ii][jj] >= Flim: cutoff_matrix[ii][jj] = 0
    return cutoff_matrix

def zero_to_nan(input_array):
    len_x, len_y = np.shape(input_array)
    for ii in range(len_x):
        for jj in range(len_y):
            if input_array[ii][jj] == 0: input_array[ii][jj] = float("Nan")
    return input_array

def find_FES_adj(X_old, Y_old, FES_old):
    # r = np.stack(["x_old_grid_mesh".ravel(), "y_old_grid_mesh".ravel()]).T
    r = np.stack([X_old.ravel(), Y_old.ravel()]).T
    # Sx = interpolate.CloughTocher2DInterpolator(r, "Z_values".ravel())
    Sx = interpolate.CloughTocher2DInterpolator(r, FES_old.ravel())
    # ri = np.stack(["x_new_grid_mesh".ravel(), "y_new_grid_mesh".ravel()]).T
    ri = np.stack([XREF.ravel(), YREF.ravel()]).T
    FES_new = Sx(ri).reshape(XREF.shape)

    return FES_new

###

### Main Mean Force Integration

def MFI_2D( HILLS = "HILLS", position_x = "position_x", position_y = "position_y", bw = 1, kT = 1, min_grid=np.array((-np.pi, -np.pi)), max_grid=np.array((np.pi, np.pi)), nbins = np.array((101,101)), log_pace = 10, error_pace = 200, WellTempered = 1, nhills = -1, periodic=0):    
    """Compute a time-independent estimate of the Mean Thermodynamic Force, i.e. the free energy gradient in 2D CV spaces. 

    Args:
        HILLS (str, optional): HILLS array. Defaults to "HILLS".
        position_x (str, optional): CV1 array. Defaults to "position_x".
        position_y (str, optional): CV2 array. Defaults to "position_y".
        bw (int, optional): Scalar, bandwidth for the construction of the KDE estimate of the biased probability density. Defaults to 1.
        kT (int, optional): Scalar, kT. Defaults to 1.
        min_grid (_type_, optional): Lower bound of the simulation domain. Defaults to np.array((-np.pi, -np.pi)).
        max_grid (_type_, optional): Upper bound of the simulation domain. Defaults to np.array((np.pi, np.pi)).
        nbins (int, optional): number of bins in CV1,CV2. Defaults to np.array((101,101)).
        log_pace (int, optional): Pace for outputting progress and convergence. Defaults to 10.
        error_pace (int, optional): Pace for the calculation of the on-the-fly measure of global convergence. Defaults to 200.
        WellTempered (int, optional): Is the simulation well tempered? . Defaults to 1.
        nhills (int, optional): Number of HILLS to analyse, -1 for the entire HILLS array. Defaults to -1, i.e. the entire dataset.
        periodic (int, optional): Is the CV space periodic? 1 for yes. Defaults to 0. 

    Returns:
        X: array of size (nbins[0], nbins[1]) - CV1 grid positions
        Y: array of size (nbins[0], nbins[1]) - CV2 grid positions
        Ftot_den: array of size (nbins[0], nbins[1]) - Cumulative biased probability density, equivalent to an unbiased histogram of samples in CV space. 
        Ftot_x:  array of size (nbins[0], nbins[1]) - CV1 component of the Mean Force. 
        Ftot_y:  array of size (nbins[0], nbins[1]) - CV2 component of the Mean Force. 
        ofe:  array of size (nbins[0], nbins[1]) - on the fly estimate of the local convergence
        ofe_history: array of size (1, total_number_of_hills) - running estimate of the global convergence of the mean force.
    """

    gridx = np.linspace(min_grid[0], max_grid[0], nbins[0])
    gridy = np.linspace(min_grid[1], max_grid[1], nbins[1])
    X, Y = np.meshgrid(gridx, gridy)
    stride = int(len(position_x) / len(HILLS[:,1]))     
    const = (1 / (bw*np.sqrt(2*np.pi)*stride))
    
    # Optional - analyse only nhills, if nhills is set
    if  nhills > 0: 
        total_number_of_hills=nhills
    else:
        total_number_of_hills=len(HILLS[:,1])
    bw2 = bw**2    

    # Initialize force terms
    Fbias_x = np.zeros(nbins)
    Fbias_y = np.zeros(nbins)
    Ftot_num_x = np.zeros(nbins)
    Ftot_num_y = np.zeros(nbins)
    Ftot_den = np.zeros(nbins)
    Ftot_den2 = np.zeros(nbins)
    ofv_x = np.zeros(nbins)
    ofv_y = np.zeros(nbins)
    ofe_history = []

    print("Total no. of Gaussians analysed: " + str(total_number_of_hills))

    # Definition Gamma Factor, allows to switch between WT and regular MetaD
    if WellTempered < 1: 
        Gamma_Factor=1
    else:
        gamma = HILLS[0, 6]
        Gamma_Factor=(gamma - 1)/(gamma)
      
        
    for i in range(total_number_of_hills):
        # Build metadynamics potential
        s_x = HILLS[i, 1]  # center x-position of Gaussian
        s_y = HILLS[i, 2]  # center y-position of Gaussian
        sigma_meta2_x = HILLS[i, 3] ** 2  # width of Gaussian
        sigma_meta2_y = HILLS[i, 4] ** 2  # width of Gaussian
        height_meta = HILLS[i, 5] * Gamma_Factor  # Height of Gaussian

        periodic_images = find_periodic_point(s_x,s_y,min_grid,max_grid,periodic)
        for j in range(len(periodic_images)):
            kernelmeta = np.exp(-0.5 * (((X - periodic_images[j][0]) ** 2) / sigma_meta2_x + ((Y - periodic_images[j][1]) ** 2) / sigma_meta2_y))  # potential erorr in calc. of s-s_t
            Fbias_x = Fbias_x + height_meta * kernelmeta * ((X - periodic_images[j][0]) / sigma_meta2_x);  
            Fbias_y = Fbias_y + height_meta * kernelmeta * ((Y - periodic_images[j][1]) / sigma_meta2_y);  

        # Biased probability density component of the force
        # Estimate the biased proabability density p_t ^ b(s)
        pb_t = np.zeros(nbins)
        Fpbt_x = np.zeros(nbins)
        Fpbt_y = np.zeros(nbins)

        data_x = position_x[i * stride: (i + 1) * stride]
        data_y = position_y[i * stride: (i + 1) * stride]

        for j in range(stride):
            periodic_images = find_periodic_point(data_x[j], data_y[j], min_grid, max_grid,periodic)
            for k in range(len(periodic_images)):
                kernel = const * np.exp(- (1 / (2 * bw2)) * ((X - periodic_images[k][0]) ** 2 + (Y - periodic_images[k][1]) ** 2)); 
                pb_t = pb_t + kernel;
                Fpbt_x = Fpbt_x + kernel * kT * (X - periodic_images[k][0]) / bw2
                Fpbt_y = Fpbt_y + kernel * kT * (Y - periodic_images[k][1]) / bw2
            
        # Calculate Mean Force
        Ftot_den = Ftot_den + pb_t;
        # Calculate x-component of Force
        dfds_x = np.divide(Fpbt_x * kT, pb_t, out=np.zeros_like(Fpbt_x), where=pb_t != 0) + Fbias_x
        Ftot_num_x = Ftot_num_x + pb_t * dfds_x
        Ftot_x = np.divide(Ftot_num_x, Ftot_den, out=np.zeros_like(Fpbt_x), where=Ftot_den != 0)
        # Calculate y-component of Force
        dfds_y = np.divide(Fpbt_y * kT, pb_t, out=np.zeros_like(Fpbt_y), where=pb_t != 0) + Fbias_y
        Ftot_num_y = Ftot_num_y + pb_t * dfds_y
        Ftot_y = np.divide(Ftot_num_y, Ftot_den, out=np.zeros_like(Fpbt_y), where=Ftot_den != 0)

        #calculate on the fly error components
        Ftot_den2 = Ftot_den2 + pb_t**2   
        # on the fly variance of the mean force
        ofv_x += pb_t * dfds_x**2
        ofv_y += pb_t * dfds_y**2

        # Compute Variance of the mean force every with 1/error_pace frequency
        if (i + 1) % int(total_number_of_hills / error_pace) == 0:       
            #calculate ofe (standard error)
            Ftot_den_ratio = np.divide(Ftot_den2, (Ftot_den**2 - Ftot_den2), out=np.zeros_like(Ftot_den), where=(Ftot_den**2 - Ftot_den2) != 0)
            ofe_x = np.divide(ofv_x, Ftot_den, out=np.zeros_like(ofv_x), where=Ftot_den != 0) - Ftot_x**2
            ofe_y = np.divide(ofv_y, Ftot_den, out=np.zeros_like(ofv_y), where=Ftot_den != 0) - Ftot_y**2       
            ofe_x = ofe_x * Ftot_den_ratio
            ofe_y = ofe_y * Ftot_den_ratio
            ofe = np.sqrt(abs(ofe_x) + abs(ofe_y))                
            ofe_history.append(sum(sum(ofe)) / (nbins[0]*nbins[1]))

        if (i+1) % (total_number_of_hills/log_pace) == 0: 
            print(str(i+1) + "/" + str(total_number_of_hills)+"| Average Mean Force Error: "+str(sum(sum(ofe)) / (nbins[0]*nbins[1])))
            
    return [X, Y, Ftot_den, Ftot_x, Ftot_y, ofe, ofe_history]


### Integration using Fast Fourier Transform (FFT integration) in 2D            
def FFT_intg_2D(FX, FY, min_grid=np.array((-np.pi, -np.pi)), max_grid=np.array((np.pi, np.pi)), nbins = np.array((101,101))):   
    
    gridx = np.linspace(min_grid[0], max_grid[0], nbins[0])
    gridy = np.linspace(min_grid[1], max_grid[1], nbins[1])
    grid_spacex = (max_grid[0] - min_grid[0]) / (nbins[0] - 1)
    grid_spacey = (max_grid[1] - min_grid[1]) / (nbins[1] - 1)
    X, Y = np.meshgrid(gridx, gridy)

    #Calculate frequency
    freq_1dx = np.fft.fftfreq(nbins[0], grid_spacex)
    freq_1dy = np.fft.fftfreq(nbins[1], grid_spacey)
    freq_x, freq_y = np.meshgrid(freq_1dx, freq_1dy)
    freq_hypot = np.hypot(freq_x, freq_y)
    freq_sq = np.where(freq_hypot != 0, freq_hypot ** 2, 1E-10)
    #FFTransform and integration
    fourier_x = (np.fft.fft2(FX) * freq_x) / (2 * np.pi * 1j * freq_sq)
    fourier_y = (np.fft.fft2(FY) * freq_y) / (2 * np.pi * 1j * freq_sq)
    #Reverse FFT
    fes_x = np.real(np.fft.ifft2(fourier_x))
    fes_y = np.real(np.fft.ifft2(fourier_y))
    #Construct whole FES
    fes = fes_x + fes_y
    fes = fes - np.min(fes)
    return [X, Y, fes]

#Equivalent to integration MS in Alanine dipeptide notebook.     
def intg_2D(FX, FY, min_grid=-np.pi, max_grid=np.pi, nbins = 101): 
    
    grid = np.linspace(min_grid, max_grid, nbins)
    X, Y = np.meshgrid(grid, grid)

    FdSx = np.cumsum(FX, axis=1)*np.diff(grid)[0]
    FdSy = np.cumsum(FY, axis=0)*np.diff(grid)[0]

    fes = np.zeros(FdSx.shape)
    for i in range(fes.shape[0]):
        for j in range(fes.shape[1]):
            fes[i,j] += np.sum([FdSy[i,0], -FdSy[0,0], FdSx[i,j], -FdSx[i,0]])

    fes = fes - np.min(fes)

    return [X, Y, fes]


def plot_recap_2D(X, Y, FES, TOTAL_DENSITY, CONVMAP, CONV_history): 
    fig, axs = plt.subplots(2,2,figsize=(10,8))
    cp=axs[0,0].contourf(X,Y,FES,levels=range(0,50,1),cmap='coolwarm',antialiased=False,alpha=0.8);
    cbar = plt.colorbar(cp, ax=axs[0,0])
    axs[0,0].set_ylabel('CV2')
    axs[0,0].set_xlabel('CV1')
    cp=axs[0,1].contourf(X,Y,CONVMAP,levels=range(0,20,1),cmap='coolwarm',antialiased=False,alpha=0.8);
    cbar = plt.colorbar(cp, ax=axs[0,1])
    axs[0,1].set_ylabel('CV2')
    axs[0,1].set_xlabel('CV1')
    cp=axs[1,0].contourf(X,Y,TOTAL_DENSITY,cmap='viridis',antialiased=False,alpha=0.8);
    cbar = plt.colorbar(cp, ax=axs[1,0])
    axs[1,0].set_ylabel('CV2')
    axs[1,0].set_xlabel('CV1')
    axs[1,1].plot(range(len(CONV_history)), CONV_history);
    axs[1,1].set_ylabel('Average Mean Force Error')
    axs[1,1].set_xlabel('Number of Error Evaluations')
