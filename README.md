# Mean Force Integration

Inspired by thermodynamic integration, MFI is a method for the calculation of time-independent free energy profiles from history-dependent biased simulations via Mean Force Integration (MFI). MFI circumvents the need for computing the ensemble averages of the bias acting on the system c(t) and can be applied to different variants of metadynamics. Moreover, MFI naturally extends to aggregate information obtained from independent metadynamics simulations, allowing to converge free energy surfaces without the need to sample recrossing events in a single continuous trajectory. 

In this repository we collect MFI scripts and examples including a development version of __pyMFI__ a python implementation of the MFI algorithm, that includes on-the-fly convergence estimates and enlables to combine static and history-dependent biases. 

If you find this useful please cite: 

- Marinova, Veselina, and Matteo Salvalaglio. "Time-independent free energies from metadynamics via mean force integration." [The Journal of chemical physics 151.16 (2019)](164115.https://aip.scitation.org/doi/abs/10.1063/1.5123498),  [arXiv](https://arxiv.org/pdf/1907.08472.pdf)

- Bjola, Antoniu, and Matteo Salvalaglio. "Estimating Free Energy Surfaces and their Convergence from multiple, independent static and history-dependent biased molecular-dynamics simulations with Mean Force Integration." (2024).[ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/65affed69138d2316192b728)


### MFI in action: 
<video src="https://github.com/mme-ucl/MFI/raw/master/Movies_and_presentations/MFI_movie_error.mp4" align="center" width="1000px" controls></video>

### pyMFI

The stable pyMFI code is available [here](https://github.com/mme-ucl/pyMFI) and can be installed with _pip_ as: ```pip install -e git+https://github.com/mme-ucl/pyMFI.git#egg=pyMFI```

### Matlab scripts 
Matlab implementations of MFI are available in the folder _matlab_examples_. 
The matlab implementation of the MFI postprocessing scripts require the following script to perform integration of the mean force and compute free energy surfaces:  

**intgrad2.m**: numerical solution of the inverse gradient problem,  
John D'Errico (2021). Inverse (integrated) gradient 
(https://www.mathworks.com/matlabcentral/fileexchange/9734-inverse-integrated-gradient), 
MATLAB Central File Exchange. Retrieved July 4, 2021.
