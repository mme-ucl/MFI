import MFI1Dgame as MFI
import run_plumedgame as plumed
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time

# torch.set_grad_enabled(True) 

start = time.time()

x = np.linspace(-2,2,201)
y = 23*x**4 - 7*x**2
y = y - min(y)
z = np.zeros(201)

X = torch.tensor(x)
Y = torch.tensor(y)
Z = torch.tensor(z)

# WIDTH = 0.1
# HEIGHT = 0.1
# BIASFACTOR = 10 
# BW = 0.1

# X = torch.tensor([WIDTH, HEIGHT, BIASFACTOR, BW])
# w = [torch.tensor(1.0, dtype=torch.float32, requires_grad=True), torch.tensor(1.0, dtype=torch.float32, requires_grad=True), torch.tensor(1.0, dtype=torch.float32, requires_grad=True), torch.tensor(1.0, dtype=torch.float32, requires_grad=True)]

w = torch.tensor(0.1, dtype=torch.float32, requires_grad=True)

def MFI_RL(w):
    #Run simulation
    # plumed.run_langevin1D(100000, gaus_width=float(x[0] * w[0]), gaus_height=float(x[1] * w[1]), biasfactor=float(x[2] * w[2]))
    plumed.run_langevin1D(100000, gaus_width=float(0.1), gaus_height=float(w), biasfactor=float(10))

    #Read the Colvar File
    position = MFI.load_position(position_name="position")

    #Read the HILLS file
    HILLS=MFI.load_HILLS(hills_name="HILLS")

    #Compute the time-independent mean force
    # [grid, Ftot_den, Ftot, FES, AD, AAD, AD_f, AAD_f] = MFI.MFI_1D(HILLS = HILLS, position = position, bw = float(x[3] * w[3]))
    [grid, Ftot_den, Ftot, FES, AD, AAD, AD_f, AAD_f] = MFI.MFI_1D(HILLS = HILLS, position = position, bw = float(0.1))
    
    return [FES, AD, AAD]


print("t =", round(time.time() - start,4), "-> initialistaion done")
start = time.time()

# print("Initial AAD:", MFI_RL(X)[2])
# print("t =", round(time.time() - start,4), "-> t0")


learning_rate = 0.1
n_iters = 100
aad_history = []

loss = nn.MSELoss()
# optimizer = torch.optim.SGD(w, lr=learning_rate)
optimizer = torch.optim.SGD([w], lr=learning_rate, momentum=0.9)


for i in range(n_iters):
    
    [FES, AD, AAD] = MFI_RL(w)
    AD = torch.tensor(AD, requires_grad=True)
    
    aad_history.append(AAD)
    
    print(type(X))
    print(type(X[0]))
    print("")
    
    l = loss(Z, AD)
    
    l.backward()
    
    optimizer.step()
     
    optimizer.zero_grad()
    
    print("t =", round(time.time() - start,4), "-> n =", i, " ||  AAD =", AAD, " ||  w =", w, " ||  loss =", l)
    start = time.time()
    
    
    
plt.plot(range(len(aad_history)), aad_history)
plt.xlabel("Itterations")
plt.ylabel("AAD of FES to theoratical y")
plt.title("Parameter optimisation\nusing pytorch")
plt.show()



    
    