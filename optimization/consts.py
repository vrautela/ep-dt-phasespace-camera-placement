import numpy as np

# TODO: change these to reflect the most accurate specs
# FOV lower bound (objects must be at least this far to be seen)
fov_lower_bound = 1
# FOV upper bound a.k.a height of the cone (objects further than this are out of range)
fov_upper_bound = 5
# FOV degree (a.k.a angle between cone axis and slant)
fov_degree = 60
# FOV base radius (a.k.a radius of the base of the cone)
fov_base_radius = fov_upper_bound * np.tan(np.deg2rad(fov_degree))

# dimensions of V (in m)
V_x = 3.98
V_y = 12.03
V_z = 2.9
V = (V_x, V_y, V_z)

# number of cameras
N = 8

# numbers of position variables (and also number of angle variables)
NUM_VAR = 5

# the scale of the grid
epsilon = 0.5