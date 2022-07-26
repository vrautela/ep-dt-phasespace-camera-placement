import matplotlib.pyplot as plt
import numpy as np
from optimizer import gdop
# Test of GDOP vs triangulability metrics
# I need to create a fixed camera, a fixed point, and a camera that moves around the scene


c1 = [3, 0, 0, 90, 180] 
p = (0, 0, 0)

x = []
y = []

# make c2 move in a circle of radius 3 around the origin
for alpha in range(360):
    # define c2 to be a camera which is alpha degrees along the circumference of a circle of radius 3
    # phi can be from [0, 360]
    phi = (alpha + 180) % 360
    c2 = [3 * np.cos(np.deg2rad(alpha)), 3 * np.sin(np.deg2rad(alpha)), 0, 90, phi]

    cams = []
    cams.extend(c1)
    cams.extend(c2)

    x.append(alpha)
    y.append(gdop(cams, p))


fig, ax = plt.subplots()
ax.scatter(x, y)
ax.set_xlabel('Angle between cams (degrees)')
ax.set_ylabel('GDOP')
# ax.set_xticks(np.arange(0, 360, 10))
plt.show()