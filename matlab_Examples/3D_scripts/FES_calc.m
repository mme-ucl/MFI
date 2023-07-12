clc
clear all
close all

type=0
kT=2.49
periodic=1
min_grid=[1.5 -0.5 -pi];
max_grid=[4.3  5.6  pi];
nbins=50
stride=40
bw=0.03
pp=0
print_stride=50
frequency=5

%% Periodic CV Space? 
if periodic>0 
    pos=[-1 0 1];
else
   pos=0;
end
 
gridx=linspace(min_grid(1),max_grid(1),nbins); 
gridy=linspace(min_grid(2),max_grid(2),nbins); 
gridz=linspace(min_grid(3),max_grid(3),nbins); 

[GRIDX,GRIDY,GRIDZ]=meshgrid(gridx,gridy,gridz);
 
%constants%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bw2=bw.^2;
const=(1./(bw.*power(2.*pi,1.5).*(stride./frequency))); %2D with bw1=bw2 %% Generalise this!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
%% INPUT FILES

fesfiles = dir('run*'); %finding the files that match the pattern fes*.dat in directory fes
numfiles = length(fesfiles); % getting the number of the fes*.dat files 

PB_i=zeros(size(GRIDX));

Fi_X=zeros(size(GRIDX));
Fi_Y=zeros(size(GRIDY));
Fi_Z=zeros(size(GRIDZ));

Fi_X_NUM=zeros(size(GRIDX));
Fi_Y_NUM=zeros(size(GRIDY));
Fi_Z_NUM=zeros(size(GRIDZ));


for f=1:1:numfiles
tic
%% import HILLS file %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
myfilename = sprintf('run%d/HILLS.rate',f)
hills=importdata(myfilename);
HILLS=hills;%.data; 

if isempty(HILLS)==1 continue
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% import COLVAR file
myfilename = sprintf('run%d/COLVAR.rate',f)
colvar=importdata(myfilename);
position_x=colvar(:,2);
position_y=colvar(:,3);
position_z=colvar(:,4);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Number of Gaussian Kernel deposited to construct V_t(s)
total_number_of_hills=length(HILLS(:,1));
if total_number_of_hills*stride>length(colvar(:,1))
   total_number_of_hills=length(colvar(:,1))/stride;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Cycle over the updates of the bias potential V_t(s)%%%%%%%%%%%%%%%%%%%%%
%% Initialize force terms
  
%scalar terms (size GRIDX)
V_t=zeros(size(GRIDX));
  
% initialization of vectorial quantities for 2D hypersurface: _x 1st CV, _y 2nd CV
% CV1
Fbias_x=zeros(size(GRIDX));
Ftot_num_x=zeros(size(GRIDX));

%CV2
Fbias_y=zeros(size(GRIDY));
Ftot_num_y=zeros(size(GRIDY));
 
%CV3
Fbias_z=zeros(size(GRIDZ));
Ftot_num_z=zeros(size(GRIDZ));

%% weigths are scalar
Ftot_den=zeros(size(GRIDX));
  
c=0;  
for t=1:1:total_number_of_hills
 
%% Build the metadynamics potential
%% center of the Gaussian deposited at time t
s_x=HILLS(t,2);
s_y=HILLS(t,3);
s_z=HILLS(t,4);
 
%% sigma 
sigma_meta2_x=HILLS(t,5).^2;
sigma_meta2_y=HILLS(t,6).^2;
sigma_meta2_z=HILLS(t,7).^2;

 
%% height 
height_meta=HILLS(t,8); 
 
%% MetaD kernel (not adaptive)
kernelmeta=[ zeros(size(GRIDX))  ];

for k=1:length(pos)
            %% MetaD kernel (not adaptive)
            kernel_image=exp(-0.5.*((GRIDX-s_x).^2./(sigma_meta2_x)+...
                                    (GRIDY-s_y).^2./(sigma_meta2_y)+...
                                    (GRIDZ-s_z+pos(k).*2.*pi).^2./(sigma_meta2_z)));
            kernelmeta=kernelmeta+kernel_image; 
            %% Metadynamics bias force
            Fbias_x=Fbias_x+height_meta.*kernel_image.*((GRIDX-s_x)./(sigma_meta2_x));
            Fbias_y=Fbias_y+height_meta.*kernel_image.*((GRIDY-s_y)./(sigma_meta2_y));
            Fbias_z=Fbias_z+height_meta.*kernel_image.*((GRIDZ-s_z+pos(k).*2.*pi)./(sigma_meta2_z));
end


%% MetaD bias potential  [for comparison]
if type==1    
Gaussian_t=height_meta.*kernelmeta;    
V_t=V_t+Gaussian_t;
FES_metaD=-V_t-min(min(-V_t));
elseif type==2   
gamma=HILLS(1,9);  
Gaussian_t=height_meta.*kernelmeta;    % this works fine
V_t=V_t+((gamma-1)./gamma).*Gaussian_t; 
FES_metaD=-(gamma./(gamma-1)).*V_t; % correct
FES_metaD=FES_metaD-min(min(min(FES_metaD)));
end
  
  
%% Estimate the biased proabability density p_t^b(s)
%% datatpoints sampling the stationary p_t^b(s)
data_x=[position_x((t-1).*stride+1:(t).*stride)];
data_y=[position_y((t-1).*stride+1:(t).*stride)];
data_z=[position_z((t-1).*stride+1:(t).*stride)];

pb_t=zeros(size(GRIDX));
  
%% Initialise num and den of pbforce
Fpbt_num_x=zeros(size(GRIDX));
Fpbt_num_y=zeros(size(GRIDY));
Fpbt_num_z=zeros(size(GRIDZ));

% disp('Kernel Pb')
for i=1:frequency:stride
for k=1:length(pos)    
            %% Pb_t Kernel
            kernel=const.*exp(-0.5.*((GRIDX-data_x(i)).^2./bw2+...
                                        (GRIDY-data_y(i)).^2./bw2+...
                                        (GRIDZ-data_z(i)+pos(k).*2.*pi).^2./bw2));
            fx=kT.*kernel.*(GRIDX-data_x(i))./bw2;
            fy=kT.*kernel.*(GRIDY-data_y(i))./bw2;
            fz=kT.*kernel.*(GRIDZ-data_z(i)+pos(k).*2.*pi)./bw2;
            %% Pb_t mean force
            Fpbt_num_x=Fpbt_num_x+fx;
            Fpbt_num_y=Fpbt_num_y+fy;
            Fpbt_num_z=Fpbt_num_z+fz;
            pb_t=pb_t+kernel;
end

end

pb_t(pb_t==0)=eps;
 
Ftot_num_x=Ftot_num_x+Fpbt_num_x+Fbias_x.*pb_t;
Ftot_num_y=Ftot_num_y+Fpbt_num_y+Fbias_y.*pb_t;
Ftot_num_z=Ftot_num_z+Fpbt_num_z+Fbias_z.*pb_t;

Ftot_den=Ftot_den+pb_t;

Ftot_x=Ftot_num_x;
Ftot_y=Ftot_num_y;
Ftot_z=Ftot_num_z;

if mod(t,print_stride)==0 || t==total_number_of_hills
c=c+1; 
end
 
end
PB_i=PB_i+Ftot_den;

Fi_X_NUM=Fi_X_NUM+Ftot_x;
Fi_Y_NUM=Fi_Y_NUM+Ftot_y;
Fi_Z_NUM=Fi_Z_NUM+Ftot_z;

clear colvar
clear hills
clear position_x
clear position_y
clear position_z
clear s_x
clear s_y
clear s_z
toc
end

Fi_X=Fi_X_NUM./PB_i;
Fi_Y=Fi_Y_NUM./PB_i;
Fi_Z=Fi_Z_NUM./PB_i;

d_x=(max_grid(1)-min_grid(1))/nbins;
d_y=(max_grid(2)-min_grid(2))/nbins;
d_z=(max_grid(3)-min_grid(3))/nbins;

tic
BIG_FES=intgrad3(Fi_X,Fi_Y,Fi_Z,d_x,d_y,d_z,0);
toc
%BIG_FES=intgrad2(Fi_X,Fi_Y,d_x,d_y);
BIG_FES=BIG_FES-min(min(min(BIG_FES)));

%contourf(GRIDX,GRIDY,BIG_FES)
%tic
%contourslice(GRIDX,GRIDY,GRIDZ,BIG_FES,gridx,gridy,gridz)
%colorbar
%toc

save('BIG_FES.mat','BIG_FES')

exit
