clc
clear all
close all


%% Example of MFI applied to a 2D periodic CV space

%% Dependency: 
% intgrad2.m -> numerical solution of the inverse gradient problem 
% John D'Errico (2021). Inverse (integrated) gradient 
% (https://www.mathworks.com/matlabcentral/fileexchange/9734-inverse-integrated-gradient), 
% MATLAB Central File Exchange. Retrieved July 4, 2021.

kT=2.494339; % kJ/mol
periodic=1;  % 1 for yes, 0 for no!
stride=10;   % Number of points in the COLVAR file per point in the HILLS file
bw=0.1;      % Bandwidth for the KDE estimation of the biased probability density 
frequency=1; % Stride for the points in COLVAR to be used for every Gaussian (1=use all of them, 2=use 1/2 etc)

%% Define the grid
min_grid=[-pi -pi];
max_grid=[pi pi];
nbins=180;   

%% Define domain and grid for integration 
gridx=linspace(min_grid(1),max_grid(1),nbins); 
gridy=linspace(min_grid(2),max_grid(2),nbins); 
[GRIDX,GRIDY]=meshgrid(gridx,gridy);
%% Periodic CV Space? 
if periodic>0 
    pos=[-1 0 1];
else
   pos=0;
end
%%

%constants%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bw2=bw.^2;
const=(1./(bw.*sqrt(2.*pi).*(stride./frequency))); %2D with bw1=bw2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
%% INPUT FILES

%% initialise grids
PB_i=zeros(size(GRIDX));
Fi_X_NUM=zeros(size(GRIDX));
Fi_Y_NUM=zeros(size(GRIDY));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% import COLVAR file
myfilename = ['./position40'];
COLVAR=load(myfilename);
position_x=COLVAR(:,2);
position_y=COLVAR(:,3);
clear colvar
%% import HILLS file %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
myfilename = ['./hills40'];
HILLS=load(myfilename);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Number of Gaussian Kernel deposited to construct V_t(s)
total_number_of_hills=length(HILLS(:,1));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
F=total_number_of_hills*(stride/frequency);

%% Cycle over the updates of the bias potential V_t(s)%%%%%%%%%%%%%%%%%%%%%
%% Initialize force terms
  
%scalar terms (size GRIDX)
V_t=zeros(size(GRIDX)); 

% initialization of vectorial quantities for 2D surface: _x 1st CV, _y 2nd CV
% CV1
Fbias_x=zeros(size(GRIDX));
Ftot_num_x=zeros(size(GRIDX));
%CV2
Fbias_y=zeros(size(GRIDY));
Ftot_num_y=zeros(size(GRIDY));
 
%% weigths are scalar
Ftot_den=zeros(size(GRIDY));
c=0;  

%% How many HILLS? 
NHILLS=floor(total_number_of_hills/10);

for t=1:1:NHILLS

%% Build the metadynamics potential
%% center of the Gaussian deposited at time t
s_x=HILLS(t,2);
s_y=HILLS(t,3);
 
%% sigma 
sigma_meta2_x=HILLS(t,4).^2;
sigma_meta2_y=HILLS(t,5).^2;
 
%% height 
gamma=HILLS(t,7);
height_meta=HILLS(t,6).*(gamma-1)./(gamma); 
 
%% MetaD kernel (not adaptive)
kernelmeta=[ zeros(size(GRIDX))  ];

for k=1:length(pos)
    for l=1:length(pos)
        %% MetaD kernel (not adaptive)
        kernel_image=exp(-0.5.*((GRIDX-HILLS(t,2)+pos(k).*2.*pi).^2./(sigma_meta2_x)+...
                                (GRIDY-HILLS(t,3)+pos(l).*2.*pi).^2./(sigma_meta2_y)));
        kernelmeta=kernelmeta+kernel_image; 
        %% Metadynamics bias force
        Fbias_x=Fbias_x+height_meta.*kernel_image.*((GRIDX-HILLS(t,2)+pos(k).*2.*pi)./(sigma_meta2_x));
        Fbias_y=Fbias_y+height_meta.*kernel_image.*((GRIDY-HILLS(t,3)+pos(l).*2.*pi)./(sigma_meta2_y));
    end
end

%% Estimate the biased proabability density p_t^b(s)
%% datatpoints sampling the stationary p_t^b(s)
data_x=[position_x((t-1).*stride+1:(t).*stride)];
data_y=[position_y((t-1).*stride+1:(t).*stride)];
pb_t=zeros(size(GRIDX));
  
%% Initialise num and den of pbforce
Fpbt_num_x=zeros(size(GRIDX));
Fpbt_num_y=zeros(size(GRIDY));

for i=1:frequency:stride
for k=1:length(pos)
    for l=1:length(pos)
        %% Pb_t Kernel
        kernel=const.*exp(-0.5.*((GRIDX-data_x(i)+pos(k).*2.*pi).^2./bw2+(GRIDY-data_y(i)+pos(l).*2.*pi).^2./bw2));
        fx=kT.*kernel.*(GRIDX-data_x(i)+pos(k).*2.*pi)./bw2;
        fy=kT.*kernel.*(GRIDY-data_y(i)+pos(l).*2.*pi)./bw2;
        %% Pb_t mean force
        Fpbt_num_x=Fpbt_num_x+fx;
        Fpbt_num_y=Fpbt_num_y+fy;
        pb_t=pb_t+kernel;
    end
end
end
  
%% Avoid zeros
pb_t(pb_t==0)=eps;

%% Compute total components of the force.
Ftot_num_x=Ftot_num_x+Fbias_x.*pb_t+Fpbt_num_x;
Ftot_num_y=Ftot_num_y+Fbias_y.*pb_t+Fpbt_num_y;
Ftot_den=Ftot_den+pb_t;
Ftot_x=Ftot_num_x;
Ftot_y=Ftot_num_y; 

%% Display progress
clc
progress=['Progress: ',num2str(floor(t/NHILLS.*100)),' %'];
disp(progress)

end
fprintf(' -> done.\n');

%% Integrate to obtain the FES
FES=intgrad2(Ftot_x./Ftot_den,Ftot_y./Ftot_den,gridx(2)-gridx(1),gridy(2)-gridy(1));

%% 1D projections
Fx=-kT.*log(sum(exp(-FES./kT)));
Fy=-kT.*log(sum(exp(-FES'./kT)));

%% Define Energy Scale, ans set global minimum to zero. 
Flim=100; %
FES=FES-min(min(FES));
FES(FES>Flim)=NaN;

%% Save data
save dataMFI




