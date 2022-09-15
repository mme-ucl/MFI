import MFI1Dgame as MFI
import run_plumedgame as plumed
import numpy as np
import os
import matplotlib.pyplot as plt
import torch


a = [0,3,0]

aa = torch.tensor(a).clone()

print(np.argmax(a))
print(np.argmax(aa).item())

print(aa)

aaa = aa.clone()

print(aaa)

done = 0

print(max(a))
    
    


exit()

position = MFI.load_position(position_name="position")

HILLS=MFI.load_HILLS(hills_name="HILLS")

HILLS = HILLS[:1]
position = position[:10]
print("len HILLS:", len(HILLS))

[X, Ftot_den, Ftot_den2, Ftot, ofv_num] = MFI.MFI_1D(HILLS = HILLS, position = position, bw = 0.05)


plt.figure(1)
plt.plot(X, Ftot_den)


plt.figure(2)
plt.plot(position[:10])
plt.show()

exit()


print("\n\n")


user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
print("\n\n")
print(user_paths)
    
exit()

x = np.linspace(-2,2,100)

xx = x * x
y = x*2

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(x,y)
plt.title("title")
plt.subplot(1,2,2)
plt.plot(x,y)

plt.show()







exit()






#Run simulation
plumed.run_langevin1D(50000, gaus_width=0.1, gaus_height=1, biasfactor=5)


os.system("rm bck.*")

#Read the Colvar File
position = MFI.load_position(position_name="position")

#Read the HILLS file
HILLS=MFI.load_HILLS(hills_name="HILLS")

# #Compute the time-independent mean force
# [grid, Ftot_den, Ftot, FES, AD, AAD, AD_f, AAD_f] = MFI.MFI_1D(HILLS = HILLS, position = position, bw = 0.05)

# print("AAD: ", AAD)

# MFI.plot_FES(grid, FES)

# MFI.plot_FES(grid, AD)


plt.plot(range(len(position)), position)
plt.show()

print(np.shape(position))
