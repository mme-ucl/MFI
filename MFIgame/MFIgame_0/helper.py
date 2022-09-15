import matplotlib.pyplot as plt
from IPython import display

fig = plt.gcf()

plt.ion()

def plot(AAD, mean_AAD, Aofe, mean_Aofe, x, y, FES, ofe):
    
    # fig.set_size_inches(10,10)
    
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.suptitle("Training: Game AAD = " + str(len(AAD)) + " ||  Best AAD = " + str(round(mean_AAD[-1],3)) + "\n")
    
    #AAD progression
    plt.subplot(2,2,1)
    plt.plot(AAD, label="AAD of FES")
    plt.plot(mean_AAD, label="avr AAD of FES")
    plt.title('AAD progression')
    plt.xlabel('Number of Simulations')
    plt.ylabel('AAD in [kJ/mol]')      
    plt.ylim(ymin=0)    
    plt.text(len(AAD)-1, AAD[-1], str(round(AAD[-1],3)))
    plt.text(len(mean_AAD)-1, mean_AAD[-1], str(round(mean_AAD[-1],3)))
    plt.legend()
    
    #ofe progreassion
    plt.subplot(2,2,2)
    plt.plot(Aofe, label="OFE")
    plt.plot(mean_Aofe, label="avr OFE")
    plt.title('ofe progression')
    plt.xlabel('Number of Simulations')
    plt.ylabel('stdv. in [kJ/(mol*nm)]')      
    plt.ylim(ymin=0)    
    plt.text(len(Aofe)-1, Aofe[-1], str(round(Aofe[-1],3)))
    plt.text(len(mean_Aofe)-1, mean_Aofe[-1], str(round(mean_Aofe[-1],3)))
    plt.legend()   
    
    #FES
    plt.subplot(2,2,3)
    plt.plot(x, y, c="grey", label="analytical f")
    plt.plot(x, FES, c="red", label="FES")
    plt.title('Free Energy Surface')
    plt.xlabel('x in nm')
    plt.ylabel('Energy in [kJ/mol]')   
    
    #ofe
    plt.subplot(2,2,4)
    plt.plot(x, ofe, c="red")
    plt.title('On-the-fly error of mean force')
    plt.xlabel('x in nm')
    plt.ylabel('stdv. in [kJ/(mol*nm)]')  

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(.1)    
    
    
    
def plot_extended(AAD, mean_AAD, Aofe, mean_Aofe, AAD_game, mean_AAD_game, Aofe_game, mean_Aofe_game, x, y, Ftot_den, FES, ofe, parameter_list, name_list, param_extream, record, game_over, idx_new_simulation):
    

   # fig.set_size_inches(20,15) 

    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.suptitle("Training: Game number:" + str(len(AAD_game)) + " ||  Best AAD = " + str(round(record,3)) + "\n", fontsize=30)


    if game_over == 1:
        plt.rcParams.update({"axes.facecolor":"salmon"})
    else:
        plt.rcParams.update({"axes.facecolor":"white"})
            
 
    
    #AAD progression
    plt.subplot(3,4,1)
    plt.plot(AAD_game, label="AAD of FES")
    plt.plot(mean_AAD_game, label="avr AAD of FES")
    plt.title('AAD progression - by game')
    plt.xlabel('Number of Games')
    plt.ylabel('AAD in [kJ/mol]')      
    plt.ylim(ymin=0)
    # plt.text(len(AAD_game)-1, AAD_game[-1], str(round(AAD_game[-1],3)))
    # plt.text(len(mean_AAD_game)-1, mean_AAD_game[-1], str(round(mean_AAD_game[-1],3)))
    plt.legend()
    
    #ofe progreassion
    plt.subplot(3,4,2)
    plt.plot(Aofe_game, label="OFE")
    plt.plot(mean_Aofe_game, label="avr OFE")
    plt.title('ofe progression - by game')
    plt.xlabel('Number of Games')
    plt.ylabel('stdv. in [kJ/(mol*nm)]')      
    plt.ylim(ymin=0)    
    # plt.text(len(Aofe_game)-1, Aofe_game[-1], str(round(Aofe_game[-1],3)))
    # plt.text(len(mean_Aofe_game)-1, mean_Aofe_game[-1], str(round(mean_Aofe_game[-1],3)))
    plt.legend()   
    
    #FES
    plt.subplot(3,4,3)
    plt.plot(x, y, c="grey", label="analytical f")
    plt.plot(x, FES, c="red", label="FES")
    plt.title('Patched FES of current Game')
    plt.xlabel('x in nm')
    plt.ylabel('Energy in [kJ/mol]')     
    
    #ofe
    plt.subplot(3,4,4)
    plt.plot(x, ofe, c="red")
    plt.title('Patched OFE of current game')
    plt.xlabel('x in nm')
    plt.ylabel('stdv. in [kJ/(mol*nm)]')
    
           
    
    #AAD progression
    plt.subplot(3,4,5)
    plt.plot(AAD, label="AAD of FES")
    plt.plot(mean_AAD, label="avr AAD of FES")
    plt.title('AAD progression - by simulation')
    plt.xlabel('Number of Simulations')
    plt.ylabel('AAD in [kJ/mol]')      
    plt.ylim(ymin=0)    
    # plt.text(len(AAD)-1, AAD[-1], str(round(AAD[-1],3)))
    # plt.text(len(mean_AAD)-1, mean_AAD[-1], str(round(mean_AAD[-1],3)))
    for itter in range(len(idx_new_simulation)): plt.plot((idx_new_simulation[itter], idx_new_simulation[itter]), (0, 2.5), color="salmon", alpha=0.5)
    plt.legend()
    
    #ofe progreassion
    plt.subplot(3,4,6)
    plt.plot(Aofe, label="OFE")
    plt.plot(mean_Aofe, label="avr OFE")
    plt.title('ofe progression - by simulation')
    plt.xlabel('Number of Simulations')
    plt.ylabel('stdv. in [kJ/(mol*nm)]')      
    plt.ylim(ymin=0)    
    # plt.text(len(Aofe)-1, Aofe[-1], str(round(Aofe[-1],3)))
    # plt.text(len(mean_Aofe)-1, mean_Aofe[-1], str(round(mean_Aofe[-1],3)))
    for itter in range(len(idx_new_simulation)): plt.plot((idx_new_simulation[itter], idx_new_simulation[itter]), (0, 7.5), color="salmon", alpha=0.5)
    plt.legend()   
    
    #Ftot_den
    plt.subplot(3,4,7)
    plt.plot(x, Ftot_den, c="grey")
    plt.title('Probability density of last simulations')
    plt.xlabel('x in nm')
    plt.ylabel('')   
    plt.ylim()
     

    
    #parameter plots
    plt.subplot(3,4,8)
    idx = 0
    plt.plot(parameter_list[idx])
    plt.plot((0, len(parameter_list[idx])-1),(param_extream[idx][0], param_extream[idx][0]), color="grey", alpha=0.5)
    plt.plot((0, len(parameter_list[idx])-1),(param_extream[idx][1], param_extream[idx][1]), color="grey", alpha=0.5)
    plt.title(name_list[idx] + ' progression')
    plt.xlabel('Number of Simulations')
    plt.ylabel('')      
    for itter in range(len(idx_new_simulation)): plt.plot((idx_new_simulation[itter], idx_new_simulation[itter]), (param_extream[idx][0], param_extream[idx][1]), color="salmon", alpha=0.5)
    # plt.text(len(parameter_list[idx])-1, parameter_list[idx][-1], str(round(parameter_list[idx][-1],3)))
    
    plt.subplot(3,4,9)
    idx = 1
    plt.plot(parameter_list[idx])
    plt.plot((0, len(parameter_list[idx])-1),(param_extream[idx][0], param_extream[idx][0]), color="grey", alpha=0.5)
    plt.plot((0, len(parameter_list[idx])-1),(param_extream[idx][1], param_extream[idx][1]), color="grey", alpha=0.5)
    plt.title(name_list[idx] + ' progression')
    plt.xlabel('Number of Simulations')
    plt.ylabel('')      
    for itter in range(len(idx_new_simulation)): plt.plot((idx_new_simulation[itter], idx_new_simulation[itter]), (param_extream[idx][0], param_extream[idx][1]), color="salmon", alpha=0.5)
    # plt.text(len(parameter_list[idx])-1, parameter_list[idx][-1], str(round(parameter_list[idx][-1],3)))
        
    plt.subplot(3,4,10)
    idx = 2
    plt.plot(parameter_list[idx])
    plt.plot((0, len(parameter_list[idx])-1),(param_extream[idx][0], param_extream[idx][0]), color="grey", alpha=0.5)
    plt.plot((0, len(parameter_list[idx])-1),(param_extream[idx][1], param_extream[idx][1]), color="grey", alpha=0.5)
    plt.title(name_list[idx] + ' progression')
    plt.xlabel('Number of Simulations')
    plt.ylabel('')      
    for itter in range(len(idx_new_simulation)): plt.plot((idx_new_simulation[itter], idx_new_simulation[itter]), (param_extream[idx][0], param_extream[idx][1]), color="salmon", alpha=0.5)
    # plt.text(len(parameter_list[idx])-1, parameter_list[idx][-1], str(round(parameter_list[idx][-1],3))) 
    
    plt.subplot(3,4,11)
    idx = 3
    plt.plot(parameter_list[idx])
    plt.plot((0, len(parameter_list[idx])-1),(param_extream[idx][0], param_extream[idx][0]), color="grey", alpha=0.5)
    plt.plot((0, len(parameter_list[idx])-1),(param_extream[idx][1], param_extream[idx][1]), color="grey", alpha=0.5)
    plt.title(name_list[idx] + ' progression')
    plt.xlabel('Number of Simulations')
    plt.ylabel('')      
    for itter in range(len(idx_new_simulation)): plt.plot((idx_new_simulation[itter], idx_new_simulation[itter]), (param_extream[idx][0], param_extream[idx][1]), color="salmon", alpha=0.5)
    # plt.text(len(parameter_list[idx])-1, parameter_list[idx][-1], str(round(parameter_list[idx][-1],3)))
    
    plt.subplot(3,4,12)
    idx = 4
    plt.plot(parameter_list[idx])
    plt.plot((0, len(parameter_list[idx])-1),(param_extream[idx][0], param_extream[idx][0]), color="grey", alpha=0.5)
    plt.plot((0, len(parameter_list[idx])-1),(param_extream[idx][1], param_extream[idx][1]), color="grey", alpha=0.5)
    plt.title(name_list[idx] + ' progression')
    plt.xlabel('Number of Simulations')
    plt.ylabel('')      
    for itter in range(len(idx_new_simulation)): plt.plot((idx_new_simulation[itter], idx_new_simulation[itter]), (param_extream[idx][0], param_extream[idx][1]), color="salmon", alpha=0.5)
    # plt.text(len(parameter_list[idx])-1, parameter_list[idx][-1], str(round(parameter_list[idx][-1],3)))

    
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(.1)