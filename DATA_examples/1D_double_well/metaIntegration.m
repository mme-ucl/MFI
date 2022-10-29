function [grid,FES]=metaIntegration(type,HILLS,COLVAR,kT,min_grid,max_grid,nbins,stride,bw,pp,print_stride)

%%%% This function performs MFI on a 1D CV space 

%% dependencies:  cumint3.m function [numerical integration]

%% Arguments: 

% type:     Well tempered vs. non-well tempered: it is only needed to carry out
%           to estimate F(s)=-V(s)+C 
% HILLS:    HILLS file
% COLVAR:   COLVAR file
% kT:       value of kT
% min_grid: lower bound for the grid in CV space
% max_grid: upper bound for the grid in CV space
% stride:   number of points in the COLVAR file per point in the HILLS file
% bw:       bandwidth for the KDE of the biased probability density
% pp:       want to print error as a function of time? 1 for yes, 0 for no
% print_stride: how often shall you output your result? 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% define a grid on the CV domain
grid=linspace(min_grid,max_grid,nbins);

%% Analytical FES - for the paper
y=-5.*grid.^2+grid.^4;
dy=-10.*grid+4.*grid.^3;

%% Useful constants
bw2=bw.^2;
const=(1./(bw.*sqrt(2.*pi).*stride));

%% import HILLS file %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
hills=importdata(HILLS,' ',3);
HILLS=hills.data; clear hills.data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% import COLVAR file
colvar=importdata(COLVAR,' ',1);
position=colvar.data(:,2); clear colvar.data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Number of Gaussian Kernel deposited to construct V_t(s)
total_number_of_hills=length(hills.data(:,1));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Cycle over the updates of the bias potential V_t(s)%%%%%%%%%%%%%%%%%%%%%
%% Initialize force terms
V_t=zeros(size(grid));

Fbias=zeros(size(grid));

Ftot_num=zeros(size(grid));
Ftot_den=zeros(size(grid));

c=0;
for t=1:1:total_number_of_hills

%% Build metadynamics potential
s=HILLS(t,2);
sigma_meta2=HILLS(t,3).^2;
kernelmeta=exp(-0.5.*((grid-s).^2./(sigma_meta2)));

%% Metadynamics bias force 
if type==1  
height_meta=HILLS(t,4); 
Fbias=Fbias+height_meta.*kernelmeta.*((grid-s)./(sigma_meta2));

elseif type==2    
gamma=HILLS(1,5);   
height_meta=HILLS(t,4).*(gamma-1)./(gamma);
Fbias=Fbias+height_meta.*kernelmeta.*((grid-s)./(sigma_meta2));
end

%% Estimate the biased proabability density p_t^b(s)
%% datatpoints sampling the stationary p_t^b(s)
data=[position((t-1).*stride+1:(t).*stride)];
pb_t=zeros(size(grid));

%% Initialise num and den of pbforce
Fpbt_num=zeros(size(grid));
Fpbt_den=zeros(size(grid));

%% Biased probability density component of the force
for i=1:1:stride
kernel=const.*exp(-((grid-data(i)).^2)./2./bw2);
pb_t=pb_t+kernel;
Fpbt_num=Fpbt_num+kernel.*(grid-data(i))./bw2;
end

%% Estimate of the Mean Force
Ftot_num=Ftot_num+kT.*Fpbt_num+pb_t.*Fbias;
Ftot_den=Ftot_den+pb_t;

Ftot=Ftot_num./Ftot_den;

%% Numerical integration
FES=cumint3(grid',Ftot');

%% Rescale with respext to minimum (for comparison with analytical potential)
FES=FES-min(FES);

%% Wanna plot? 
if pp>0
hFig=figure(pp);
set(hFig,'Position',[100 100 800 300])

%% Compute Error 
if mod(t,print_stride)==0
  
c=c+1;
time(c)=c.*stride;
ERROR(c)=mean(abs(y-min(y)-FES'));


end
end


end

%% plot ERROR 
box on
hold on
area(time,sqrt(time).*ERROR,'FaceColor',[1.0 0 0],'LineStyle','none')
alpha(0.5)
set(gca,'FontSize',18,'LineWidth',2.0)
set(gca,'TickLabelInterpreter','latex')
xlabel('$$n_G$$','Interpreter','latex')
ylabel({'$$\overline{\epsilon({s})}\times\sqrt{n_G}$$ [k$$_B$$T$$\times\sqrt{n_G}$$]'},'Interpreter','latex')

xlim([time(1) time(end)])
drawnow


