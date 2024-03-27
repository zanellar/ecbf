# Existing code
import os
import numpy as np
import matplotlib.pyplot as plt
from ecbf.utils.paths import PLOTS_PATH

C1 = 20
C2 = 10

x = np.linspace(-20, 20, 400)
y = np.linspace(-20, 20, 400)
x, y = np.meshgrid(x, y)

z = -(x**2 + y**2 - C2**2) * (x**2 + y**2 - C1**2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis', alpha=0.8)
 
# Plot the plane at z=0
z_plane = np.zeros((400, 400))
ax.plot_surface(x, y, z_plane, color='black', alpha=0.3, rstride=50, cstride=50)
  
# Highlight the intersection
ax.contour(x, y, z, [0], colors='red')
mask = z < 0 
ax.contour(x, y, z * mask, zdir='z', offset=0, cmap='Reds', levels=100, alpha=0.5)

plt.savefig(os.path.join(PLOTS_PATH, 'barrier_function.png'))  # Save the figure before showing it 
plt.show() 

 