from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib.animation import FuncAnimation

velocity = Dataset("../../ModelOutput/fPlaneTC_particles/velocity.nc")
particles = Dataset("../../ModelOutput/fPlaneTC_particles/particles.nc")
vorticity = Dataset("../../ModelOutput/fPlaneTC_particles/vorticity.nc")

dim = velocity.dimensions
var = velocity.variables

fig = plt.figure()
ax = fig.gca(projection='3d')

x, y, z = np.meshgrid(
    var['xC'],
    var['yC'],
    var['zC'],
)

u = np.swapaxes(var['u'][0, :], 0, 2)
v = np.swapaxes(var['v'][0, :], 0, 2)

print("Arrays Loaded")

ax.quiver(x, y, z, u, v, 0, length=0.1)

plt.show()
