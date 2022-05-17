
#import sys
import numpy as np
import MFI2 as MFI
import MEP as MEP
import pandas as pd
import json
import codecs
import matplotlib.pyplot as plt
#import matlab.engine

#####################################
#   PATCH INDEPENDENT SIMULATIONS   #
#####################################

master = []
all_hills= []
error_history = []

FP = np.zeros((200,200))



HILLS=MFI.load_HILLS_2D(hills_name="HILLS")
#plt.plot(HILLS[:,1],HILLS[:,2],'.')
#Read the Colvar File
[position_x, position_y] = MFI.load_position_2D(position_name="position")
#COMPUTE Mean force and weight of the simulation
[X, Y, Ftot_den, Ftot_x, Ftot_y, ofe_map, ofe_history, Ftot_den2, ofv_x, ofv_y] = MFI.MFI_2D(HILLS = HILLS, position_x = position_x, position_y = position_y, bw = 0.1, kT = 2.49, min_grid = np.array((0, -np.pi)), max_grid = np.array((16, np.pi)), nbins = np.array((200,200)), log_pace = 10, error_pace = 100, WellTempered = 0, nhills=-1, periodic=0)

master.append([Ftot_den, Ftot_den2, Ftot_x, Ftot_y, ofv_x, ofv_y])
[FX,FY,FD,error] = MFI.patch_2D_error(master)

error_history.append(sum(sum(error)) / (np.shape(error)[0]*np.shape(error)[1]))

dx = np.abs( X[0] - X[1] )
dy = np.abs( Y[0] - Y[1] )

#integration on a periodic domain
#[X, Y, FES] = MFI.intg_2D(FX, FY, min_grid=np.array((0, -np.pi)), max_grid=np.array((16, np.pi)), nbins = np.array((200,200)))
[X, Y, FES] = MFI.intgrad2(FX,FY,200,200,dx,dy,0,False,False,min_grid=np.array((0, -np.pi)), max_grid=np.array((16, np.pi)), nbins = np.array((200,200)))

# Postprocess the results
MFI.plot_recap_2D(X, Y, FES, FD,error,error_history,FES_lim=120, ofe_map_lim=50)

######################################
##            FIND MEP               #
######################################

Festosave = FES.tolist()
fesjson = json.dump(Festosave, codecs.open('fes.json', 'w', encoding = 'utf-8'))

#f = open('fes.json','r')
#lista = json.load(f)
#
#Z = lista
#
#Z_matrix = np.array(lista).reshape(200,200)
#X = np.linspace(0, 16, 200)
#Y = np.linspace(-np.pi, np.pi, 200)
#
## Indexes of local minima min_a and min_b 
##indice_1 = np.where(Z_matrix == 0.0)
##indice_2 = np.where(Z_matrix == np.min(Z_matrix[:,70:]))
#
#indice_1 = [61,35]  
##indice_1 = [61,29]
#indice_2 = [70,112]
#
#spacing = 14
#
## define constants
#mass = 1
#k = 100
#dt = 0.00000001
#tspan = 8000
#nreplica = spacing
#
##MEP.elastic_band(X,Y,Z_matrix,tspan,dt,mass,nreplica,k,indice_1,indice_2,spacing)
#MEP.steepest_descent(X,Y,Z_matrix,tspan,dt,mass,nreplica,k,indice_1,indice_2,spacing)

