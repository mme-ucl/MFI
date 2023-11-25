# Mean Force Integration

Inspired by thermodynamic integration, MFI is a method for the calculation of time-independent free energy profiles from history-dependent biased simulations via Mean Force Integration (MFI). MFI circumvents the need for computing the ensemble averages of the bias acting on the system c(t) and can be applied to different variants of metadynamics. Moreover, MFI naturally extends to aggregate information obtained from independent metadynamics simulations, allowing to converge free energy surfaces without the need to sample recrossing events in a single continuous trajectory. 

In this repository we have two example applications of MFI to a double well in 1D and to a Alanine dipeptide. 

If you find this useful please cite: 

- Marinova, Veselina, and Matteo Salvalaglio. "Time-independent free energies from metadynamics via mean force integration." [The Journal of chemical physics 151.16 (2019)](164115.https://aip.scitation.org/doi/abs/10.1063/1.5123498),  [arXiv](https://arxiv.org/pdf/1907.08472.pdf)

The posprocessing scripts are written in matlab and require the following dependencies (for integration): 

**intgrad2.m**: numerical solution of the inverse gradient problem,  
John D'Errico (2021). Inverse (integrated) gradient 
(https://www.mathworks.com/matlabcentral/fileexchange/9734-inverse-integrated-gradient), 
MATLAB Central File Exchange. Retrieved July 4, 2021.

### MFI in action: 
<video src="https://github.com/mme-ucl/mme-ucl.github.io/raw/main/images/MFI_movie.mp4" align="center" width="1000px" controls></video>
