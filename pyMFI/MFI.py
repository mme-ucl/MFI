import glob
import numpy as np

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

def MFI_2D(HILLS = "HILLS", position_x = "position_x", position_y = "position_y", bw = 1, kT = 1, min_grid=-np.pi, max_grid=np.pi, nbins = 101, log_pace = 10, error_pace = 200):    
    grid = np.linspace(min_grid, max_grid, nbins)
    grid_space = (max_grid - min_grid) / (nbins - 1)
    X, Y = np.meshgrid(grid, grid)


    stride = int(len(position_x) / len(HILLS[:,1]))     
    const = (1 / (bw*np.sqrt(2*np.pi)*stride))
    total_number_of_hills=len(HILLS[:,1])
    bw2 = bw**2    

    count = 0
    # Initialize force terms
    Fbias_x = np.zeros((nbins, nbins))
    Fbias_y = np.zeros((nbins, nbins))
    Ftot_num_x = np.zeros((nbins, nbins))
    Ftot_num_y = np.zeros((nbins, nbins))
    Ftot_den = np.zeros((nbins, nbins))
    Ftot_den2 = np.zeros((nbins, nbins))
    ofv_x = np.zeros((nbins,nbins))
    ofv_y = np.zeros((nbins,nbins))
    ofe_history = []

    for i in range(total_number_of_hills):
        # Build metadynamics potential
        s_x = HILLS[i, 1]  # center x-position of Gaussian
        s_y = HILLS[i, 2]  # center y-position of Gaussian
        sigma_meta2_x = HILLS[i, 3] ** 2  # width of Gaussian
        sigma_meta2_y = HILLS[i, 4] ** 2  # width of Gaussian
        gamma = HILLS[i, 6]
        height_meta = HILLS[i, 5] * ((gamma - 1) / (gamma))  # Height of Gaussian

        kernelmeta = np.exp(-0.5 * (((X - s_x) ** 2) / sigma_meta2_x + ((Y - s_y) ** 2) / sigma_meta2_y)) 
        Fbias_x = Fbias_x + height_meta * kernelmeta * ((X - s_x) / sigma_meta2_x);  
        Fbias_y = Fbias_y + height_meta * kernelmeta * ((Y - s_y) / sigma_meta2_y);  

        # Biased probability density component of the force
        # Estimate the biased proabability density p_t ^ b(s)
        pb_t = np.zeros((nbins, nbins))
        Fpbt_x = np.zeros((nbins, nbins))
        Fpbt_y = np.zeros((nbins, nbins))

        data_x = position_x[i * stride: (i + 1) * stride]
        data_y = position_y[i * stride: (i + 1) * stride]
        for j in range(stride):
            kernel = const * np.exp(- ((X - data_x[j]) ** 2 + (Y - data_y[j]) ** 2) / (2 * bw2) )
            pb_t = pb_t + kernel;
            Fpbt_x = Fpbt_x + kernel * (X - data_x[j]) / bw2
            Fpbt_y = Fpbt_y + kernel * (Y - data_y[j]) / bw2

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
        ofv_x += pb_t * dfds_x**2
        ofv_y += pb_t * dfds_y**2

        if (i + 1) % int(total_number_of_hills / error_pace) == 0:       
            #calculate ofe (standard error)
            Ftot_den_ratio = np.divide(Ftot_den2, (Ftot_den**2 - Ftot_den2), out=np.zeros_like(Ftot_den), where=(Ftot_den**2 - Ftot_den2) != 0)
            ofe_x = np.divide(ofv_x, Ftot_den, out=np.zeros_like(ofv_x), where=Ftot_den != 0) - Ftot_x**2
            ofe_y = np.divide(ofv_y, Ftot_den, out=np.zeros_like(ofv_y), where=Ftot_den != 0) - Ftot_y**2       
            ofe_x = ofe_x * Ftot_den_ratio
            ofe_y = ofe_y * Ftot_den_ratio
            ofe = np.sqrt(abs(ofe_x) + abs(ofe_y))                
            ofe_history.append(sum(sum(ofe)) / (nbins**2))

        if (i+1) % (total_number_of_hills/log_pace) == 0: 
            print(str(i+1) + " / " + str(total_number_of_hills)+" Average Error: "+str(sum(sum(ofe)) / (nbins**2)))
            
    return [Ftot_den, Ftot_x, Ftot_y, ofe_history]

### Integrtion using Fast Fourier Transform (FFT integration) in 2D            
def FFT_intg_2D(FX, FY, min_grid=-np.pi, max_grid=np.pi, nbins = 101):
    
    grid = np.linspace(min_grid, max_grid, nbins)
    grid_space = (max_grid - min_grid) / (nbins - 1)
    X, Y = np.meshgrid(grid, grid)

    #Calculate frequency
    freq_1d = np.fft.fftfreq(nbins, grid_space)
    freq_x, freq_y = np.meshgrid(freq_1d, freq_1d)
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