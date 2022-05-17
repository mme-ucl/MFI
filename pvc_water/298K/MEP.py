#!/usr/bin/env python3

import numpy as np
import sys
import json
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from numba import jit

#@jit(nopython=True)
def interp_bilinear(X,Y,Z,x_star,y_star,nreplica):

    '''
    Bilinear interpolation: takes a matrix Z[1...m]x[1...n], two vectors X[1...m] and Y[1...n]  and the point outside the grid [x_star, y_star] as input and returns the interpolated point z_star.  (For now implemented only for square matrices mxm )
    
    '''

    z_star = np.zeros(nreplica)
    idx_new = np.zeros(nreplica) 
    idy_new = np.zeros(nreplica)

    # Find the indexes of the new position
    for i in range(1,nreplica-1):
                                                                  
        idx_list = np.array(np.where( X > x_star[i] ) )
        idy_list = np.array(np.where( Y > y_star[i] ) )
        idx_new[i] = idx_list[0,0]
        idy_new[i] = idy_list[0,0]


    idx_new = [ int(x) for x in idx_new ]
    idy_new = [ int(x) for x in idy_new ]
    
    for i in range(1,nreplica-1):

        z1 = Z[idy_new[i]-1][idx_new[i]-1]
        z2 = Z[idy_new[i]][idx_new[i]-1]      
        z3 = Z[idy_new[i]-1][idx_new[i]]
        z4 = Z[idy_new[i]-1][idx_new[i]]
        
        t = ( x_star[i] - X[idx_new[i]-1] )/(X[idx_new[i]] - X[idx_new[i]-1])
        u = ( y_star[i] - Y[idy_new[i]-1] )/(Y[idy_new[i]] - Y[idy_new[i]-1])

        z_star[i] = (1-t)*(1-u)*z1 + t*(1-u)*z2 + t*u*z3 + (1-t)*u*z4
        
        
    return z_star


#@jit
def derivative(X,Y,Z,nbins=200):

    '''
    Compute derivatives with central difference and backward difference for boundary points
         __
      __/__\__
     __/    \__
     
    '''            

    der_x = np.zeros((nbins,nbins))
    der_y = np.zeros((nbins,nbins))
    grad = np.zeros((nbins,nbins)) 

    # vector of derivatives wrt x   
    for j in range(1,len(X)-1):

        for i in range(1,len(X)-1):
    
            der_x[i,j] = (Z[i+1,j]-Z[i-1,j])/(abs(X[i+1]-X[i-1]))

	
    # vector of derivatives wrt y       
    for j in range(1,len(X)-1):                                      
 
        for i in range(1,len(Y)-1):
     
            der_y[j,i] = (Z[j,i+1]-Z[j,i-1])/(abs(Y[i+1]-Y[i-1])) 


    # compute the magnitude of the gradient
    for j in range(1,len(X)-1):

        for i in range(1,len(der_x)-1):

            grad[j,i] = ( der_x[j,i]**2 + der_y[j,i]**2 )**0.5


    return der_x, der_y, grad




def find_minima(X, Y, Z, grad):

    '''
    Simple search of stationary points and compute numerical Hessian to find minima (TODO)
    
    '''
    minima = []
    x_min = []
    y_min = []
    threshold = 0.3

    for i in range(1,200-1):

        for j in range(1,200-1):


            if grad[i,j] < threshold:
        
               minima.append( Z[i,j])
               x_min.append(X[i])
               y_min.append( Y[j]) 


    print("number of stationary points found for threshold "+str(threshold)+": "+str(len(minima))+"")


    return minima, x_min, y_min


#@jit(nopython=True)
def get_force(X,Y,gradient,nreplica,k_el):


    '''
    Compute forces based on the elastic band method (G. Henkelman, H. Jónsson, DOI:10.1063/1.1323224)
                                                                                                                                  
    '''
                                                                                                                              
    total_force = np.zeros((2,nreplica))
     
    for i in range(1,nreplica-1):
    
        #find unit vectors tangent to the path
        dr1 = np.array([X[i], Y[i]]) - np.array([X[i-1], Y[i-1]])
        dr2 = np.array([X[i+1], Y[i+1]]) - np.array([X[i], Y[i]])
        norm_dr1 = np.sqrt(np.sum(np.power(dr1, 2)))
        norm_dr2 = np.sqrt(np.sum(np.power(dr2, 2)))       
        tangent = dr1/norm_dr1 + dr2/norm_dr2
        norm_tangent =  np.sqrt(np.sum(np.power(tangent, 2)))
        versor_tangent = tangent/norm_tangent
    
        # compute elastic force parallel to the path (spring force)
        square_versor = np.dot(versor_tangent,versor_tangent)
        force_parallel = k_el*np.dot((dr2-dr1),square_versor)
           
        # compute force perpendicular to the path (true force)
        force_perpendicular =  gradient[:,i] -  np.dot(gradient[:,i],versor_tangent)
        
        # compute total force
        total_force[:,i] = force_parallel - force_perpendicular
        

    return  total_force


def steepest_descent(X,Y,Z,tspan,dt,mass,nreplica,k_el, indice_1, indice_2,spacing):

    [ x, y ] = np.meshgrid(X,Y)

    DX = np.linspace(indice_1[1],indice_2[1], spacing )
    DX = np.array(DX,dtype=int).reshape(spacing)
    DY = np.linspace(indice_1[0],indice_2[0], spacing )
    DY = np.array(DY,dtype=int).reshape(spacing)

    min_a = np.array([X[indice_1[1]],Y[indice_1[0]]])
    min_b = np.array([X[indice_2[1]],Y[indice_2[0]]])

    # find initial gradient (to be generalized)
    [der_x, der_y, grad] = derivative(X,Y,Z)
    gradient = np.array([der_x,der_y])

    idx_old = DX
    idy_old = DY
    pos_store = np.zeros((2,len(DX),tspan))
    pos_store[:,:,0] = np.array([X[idx_old],Y[idy_old]]) 
    idx_new = np.zeros(len(DX))
    idy_new = np.zeros(len(DY))
    idx_store = np.zeros((len(DX),tspan)) 
    idy_store = np.zeros((len(DX),tspan))
    alfa = 0.001
    threshold = 1e-3
    x_star = np.zeros(len(DX))
    y_star = np.zeros(len(DX))
    x_star[0] = X[idx_old[0]]
    y_star[0] = Y[idy_old[0]]
    x_star[-1] = X[idx_old[-1]]
    y_star[-1] = Y[idx_old[-1]]
    z_star_hist = np.zeros((len(DX),tspan))
    z_star_hist[:,0] = Z[idy_old,idx_old]
    
    # Caclulate force acting on each replica at time 0
    total_force = get_force(X[idx_old],Y[idy_old],gradient[:,idx_old,idy_old],nreplica,k_el)
    grad =  gradient[:,idx_old,idy_old]      
    xx = X[idx_old]
    yy = Y[idy_old]
    k = 0
    # time loop
    for j in range(1,tspan):
         
        
        pos = np.array([xx, yy]) + alfa*total_force
        pos_store[:,:,j] = pos
                
        x_star = pos[0][:]    
        y_star = pos[1][:] 
        
        z_star = interp_bilinear(X,Y,Z,x_star,y_star,len(DX))
        z_star_hist[:,j] = z_star   
        z_star_hist[0,j] = z_star_hist[0,0]
        z_star_hist[-1,j] = z_star_hist[-1,0]
    
        # Update Gradient (to be included into gradient function)
        dx = np.abs( X[0] - X[1] )
        dy = np.abs( Y[0] - Y[1] )
        x1 = np.zeros(len(DX))
        x2 = np.zeros(len(DX))
        y1 = np.zeros(len(DX))
        y2 = np.zeros(len(DX))   
    
        for i in  range(1,len(DX)-1):
         
            x1[i] = x_star[i] + dx
            x2[i] = x_star[i] - dx
            y1[i] = y_star[i] + dy
            y2[i] = y_star[i] - dy
            
        z_x1 = interp_bilinear(X,Y,Z,x1,y_star,len(DX))
        z_x2 = interp_bilinear(X,Y,Z,x2,y_star,len(DX))
    
        z_y1 = interp_bilinear(X,Y,Z,x_star,y1,len(DX))
        z_y2 = interp_bilinear(X,Y,Z,x_star,y2,len(DX))
    
        dzdx = ( z_x1 - z_x2 )/( 2*dx )
        dzdy = ( z_y1 - z_y2 )/( 2*dy )
    
        grad = np.array([dzdx, dzdy])
        
        total_force = get_force(xx,yy,grad,nreplica,k_el)

        norm_pos = np.sqrt(np.sum(np.power((pos_store[:,:,j]-pos_store[:,:,j-1]), 2)))

        if norm_pos < threshold:
                                                                                            
           print("Equilibrium positions found for threshold: "+str(threshold)+"")
                                                                                            
           break

        xx = x_star
        yy = y_star
        
        k = k + 1

    # Postprocessing 
    fig,ax = plt.subplots(1,2)

    for i in range(0,k):
    
        ax[0].cla()
        ax[1].cla()    
        idx = [ int(x) for x in idx_store[:,i] ]
        idy = [ int(x) for x in idy_store[:,i] ]
        ax[0].plot(pos_store[0,:,i],pos_store[1,:,i], color='black', marker='o', label='line with marker')
        cp=ax[0].contourf(x,y,Z,levels=range(0,100,1),cmap='YlGnBu_r',antialiased=False,alpha=0.8)    
        ax[0].set_ylim([-np.pi,np.pi])
        ax[0].set_xlim([0,10])
        ax[0].set_ylabel('CV2',fontsize=11)
        ax[0].set_xlabel('CV1',fontsize=11)
        ax[0].set_title(' X-Y Path ')
        ax[1].plot(pos_store[0,:,i],z_star_hist[:,i])
        ax[1].set_ylim([0,40])
        ax[1].set_xlim([0,10])
        ax[1].set_xlabel('d(C-C) [Bohr]')
        ax[1].set_title('Minimum Free Energy Path [kcal/mol]')
    
        plt.pause(0.0001)                                             
    
    
    #cbar = fig.colorbar(cp, ax=ax[0])   
    plt.show()

    return  pos_store, z_star_hist 


#@jit
def elastic_band(X,Y,Z,tspan,dt,mass,nreplica,k_el, indice_1, indice_2,spacing):

    '''
    Description

    Input Args:

    X [1...n] vector
    Y [1...n] vector
    Z [1...n]x[1...n] matrix 
    tspan total time of the simulation
    dt time step 
    mass ficticious mass of the replica points
    k_el elastic constant of the springs
    indice_1
    indice_2
    spacing number of replica points

    Output Args:
    
    '''

    [ x, y ] = np.meshgrid(X,Y)

    DX = np.linspace(indice_1[1],indice_2[1], spacing )
    DX = np.array(DX,dtype=int).reshape(spacing)
    DY = np.linspace(indice_1[0],indice_2[0], spacing )
    DY = np.array(DY,dtype=int).reshape(spacing)

    min_a = np.array([X[indice_1[1]],Y[indice_1[0]]])
    min_b = np.array([X[indice_2[1]],Y[indice_2[0]]])

    # find initial gradient (to be generalized)
    [der_x, der_y, grad] = derivative(X,Y,Z)
    gradient = np.array([der_x,der_y])
    
    # Initialize vector with random velocities
    #v0 = np.random.uniform(-1.,1.,[2,nreplica])*0.5
    v0 = np.random.rand(2,nreplica)*(-0.5)
    v0[0,0] = 0
    v0[0,-1] = 0
    v0[1,0] = 0
    v0[1,-1] = 0
    #v0 = np.zeros((2,nreplica))

    # Initialize vectors    
    v = v0
    idx_old = DX
    idy_old = DY
    pos_store = np.zeros((2,len(DX),tspan))
    pos_store[:,:,0] = np.array([X[idx_old],Y[idy_old]]) 
    idx_new = np.zeros(len(DX))
    idy_new = np.zeros(len(DY))
    idx_store = np.zeros((len(DX),tspan)) 
    idy_store = np.zeros((len(DX),tspan))
    threshold = 1e-3
    x_star = np.zeros(len(DX))
    y_star = np.zeros(len(DX))
    x_star[0] = X[idx_old[0]]
    y_star[0] = Y[idy_old[0]]
    x_star[-1] = X[idx_old[-1]]
    y_star[-1] = Y[idx_old[-1]]
    z_star_hist = np.zeros((len(DX),tspan))
    z_star_hist[:,0] = Z[idy_old,idx_old]
    
    # Caclulate force acting on each replica at time 0
    total_force = get_force(X[idx_old],Y[idy_old],gradient[:,idx_old,idy_old],nreplica,k_el)
           
    xx = X[idx_old]
    yy = Y[idy_old]


    
    # time loop
    for j in range(1,tspan):
    
        # Velocity Verlet Algorithm
        v_half = v+dt/2*total_force/mass
        pos = np.array([xx, yy]) + dt*v_half
        pos_store[:,:,j] = pos
        
        x_star = pos[0][:]    
        y_star = pos[1][:] 
        
        z_star = interp_bilinear(X,Y,Z,x_star,y_star,len(DX))
        z_star_hist[:,j] = z_star   
        z_star_hist[0,j] = z_star_hist[0,0]
        z_star_hist[-1,j] = z_star_hist[-1,0]
    
        # Update Gradient (to be included into gradient function)
        dx = np.abs( X[0] - X[1] )
        dy = np.abs( Y[0] - Y[1] )
        x1 = np.zeros(len(DX))
        x2 = np.zeros(len(DX))
        y1 = np.zeros(len(DX))
        y2 = np.zeros(len(DX))   
    
        for i in  range(1,len(DX)-1):
         
            x1[i] = x_star[i] + dx
            x2[i] = x_star[i] - dx
            y1[i] = y_star[i] + dy
            y2[i] = y_star[i] - dy
            
        z_x1 = interp_bilinear(X,Y,Z,x1,y_star,len(DX))
        z_x2 = interp_bilinear(X,Y,Z,x2,y_star,len(DX))
    
        z_y1 = interp_bilinear(X,Y,Z,x_star,y1,len(DX))
        z_y2 = interp_bilinear(X,Y,Z,x_star,y2,len(DX))
    
        dzdx = ( z_x1 - z_x2 )/( 2*dx )
        dzdy = ( z_y1 - z_y2 )/( 2*dy )
    
        grad = np.array([dzdx, dzdy])
        
        total_force = get_force(xx,yy,grad,nreplica,k_el)
    
        # Compute Max Force
        max_force = 0
        for i in range(1,nreplica-1):

            max_force = max_force + np.sqrt(np.sum(np.power(total_force[:,i], 2)))
        
        #print(max_force)
        if max_force < threshold:
  
           print("Equilibrium positions found for Max Force threshold: "+str(threshold)+"")

           break

        # Update Variables
        v_new = v_half + dt/2*total_force/mass
        v = v_new
        xx = x_star
        yy = y_star
    
    print(max_force)
    # Postprocessing 
    fig,ax = plt.subplots(1,2)
    
    for i in range(0,int(tspan/100)):
    
        ax[0].cla()
        ax[1].cla()    
        idx = [ int(x) for x in idx_store[:,i] ]
        idy = [ int(x) for x in idy_store[:,i] ]
        ax[0].plot(pos_store[0,:,100*i],pos_store[1,:,100*i], color='black', marker='o', label='line with marker')
        cp=ax[0].contourf(x,y,Z,levels=range(0,100,1),cmap='YlGnBu_r',antialiased=False,alpha=0.8)    
        ax[0].set_ylim([-np.pi,np.pi])
        ax[0].set_xlim([-np.pi,np.pi])
        ax[0].set_ylabel('CV2',fontsize=11)
        ax[0].set_xlabel('CV1',fontsize=11)
        ax[0].set_title(' X-Y Path ')
        ax[1].scatter(pos_store[0,:,100*i],z_star_hist[:,100*i])
        ax[1].set_ylim([0,60])
        ax[1].set_xlim([0,10])
        ax[1].set_xlabel('d(C-C) [Bohr]')
        ax[1].set_title('Minimum Free Energy Path [kcal/mol]')
    
        plt.pause(0.0001)                                             
    
    
    cbar = fig.colorbar(cp, ax=ax[0])   
    plt.show()

    return  pos_store, z_star_hist 

