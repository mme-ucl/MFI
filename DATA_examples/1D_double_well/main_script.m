clc    
clear all
close all


%% Run integration
[grid,fes]=metaIntegration(1,'HILLS','position',1,-2.3,2.3,100,500,0.04,1,50);


%% Plot FES
figure(2)
plot(grid,fes,'-','LineWidth',2.0,'Color',[1.0 0 0])
box on
set(gca,'FontSize',18,'LineWidth',2.0)
set(gca,'TickLabelInterpreter','latex')
xlabel('s','Interpreter','latex')
ylabel('F(s) [k$_B$T]','Interpreter','latex')
set(gca,'FontSize',18,'LineWidth',2.0)
set(gca,'TickLabelInterpreter','latex')
drawnow

xlim([-2.3 2.3])

