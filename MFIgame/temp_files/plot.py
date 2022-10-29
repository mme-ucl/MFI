import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource




gridx = np.linspace(-2,2,100)
gridy = np.linspace(-2,2,100)
x,y = np.meshgrid(gridx, gridy)


z = 10 * x**2

# z = np.where(z > 20, np.nan, z)

# fig = plt.figure(figsize=(4,4))
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_surface(X,Y,Halfpipe, color="yellow")


# ax.set_zlim(0,100)
# ax.set_xlim(-1,1)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
# # fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )
# plt.show()



## Mayavi
# surf = mlab.surf(x, y, z, colormap='RdYlBu', warp_scale='auto')
# # Change the visualization parameters.
# surf.actor.property.interpolation = 'phong'
# surf.actor.property.specular = 0.1
# surf.actor.property.specular_power = 5



## Matplotlib
fig = plt.figure()
ax = fig.gca(projection='3d')

# Create light source object.
ls = LightSource(azdeg=180, altdeg=60)
# Shade data, creating an rgb array.
rgb = ls.shade(z, plt.cm.RdYlBu)
surf = ax.plot_surface(x, y, z, color="yellow", rstride=1, cstride=1, linewidth=0,
                       antialiased=False, alpha=0.5)#, facecolors=rgb)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.view_init(85, 90)
plt.axis("off")
plt.savefig('/home/antoniu/Desktop/Windows/halfpipe.png', dpi=600, transparent="true", bbox_inches='tight')
plt.show()

# mlab.show()