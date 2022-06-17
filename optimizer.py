import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize

# FOV range (a.k.a height of the cone)
fov_range = 5
# FOV degree (a.k.a angle between cone axis and slant)
fov_degree = 50
# FOV base radius (a.k.a radius of the base of the cone)
fov_base_radius = fov_range * np.tan(np.deg2rad(fov_degree))

# dimensions of V (in m)
V_x = 5
V_y = 5
V_z = 5
V = (V_x, V_y, V_z)

# number of cameras
N = 2

# numbers of position variables (and also number of angle variables)
N_p = 3 * N

# the scale of the grid
epsilon = 0.1


# return the number of points that lie in the FOV of at least two cameras
def objective_function(x):
    total = 0

    # loop over all points in the grid defined by cutting V every epsilon meters
    n_x = V_x / epsilon
    n_y = V_y / epsilon
    n_z = V_z / epsilon
    for a in range(n_x):
        p_x = epsilon * a 
        for b in range(n_y):
            p_y = epsilon * b
            for c in range(n_z):
                p_z = epsilon * c
                grid_point = np.array([p_x, p_y, p_z])

                fov_count = 0
                # loop over each camera to see if the point at (p_x, p_y, p_z) lies in its FOV
                for i in range(N_p):
                    cam_x = x[i]
                    cam_y = x[i+1]
                    cam_z = x[i+2]
                    cam_theta_x = x[N_p+i]
                    cam_theta_y = x[N_p+i+1]
                    cam_theta_z = x[N_p+i+2]

                    pos_vec = np.array([cam_x, cam_y, cam_z])
                    angles = np.array([cam_theta_x, cam_theta_y, cam_theta_z])
                    orientation_vec = np.cos(np.deg2rad(angles))

                    if in_fov(pos_vec, orientation_vec, grid_point):
                        fov_count += 1

                    if fov_count >= 2:
                        total += 1
                        break
    return total


def in_fov(pos_vec, orientation_vec, grid_point):
    # get vector from tip of cone to point being tested
    tip_to_point = grid_point - pos_vec 
    # project t2p onto the orientation vector to find out how far down the cone (from the tip) the point lies
    cone_dist = np.dot(tip_to_point, orientation_vec)

    if cone_dist < 0 or cone_dist > fov_range:
        return False

    radius_at_cone_dist = (cone_dist / fov_range) * fov_base_radius
    proj_of_point_onto_axis = cone_dist * orientation_vec
    dist_from_axis = np.linalg.norm(tip_to_point - proj_of_point_onto_axis)

    return dist_from_axis < radius_at_cone_dist



if __name__ == '__main__':
    '''
    We want to set up and solve a constrained optimization problem (using a pre-built optimizer)
    The steps required to do so are:
    1. Define an objective function
    2. Define the constraints/bounds 
    3. Provide initial guess/values (??)
    4. Run the optimization routine

    Step 1 is the most work and should be its own module(s). 
    It is quite difficult to pin down how to define the objective function.
    '''

    # Each of the cameras' x, y, and z must be greater than 0 and less than the corresponding term in V
    # There are 3N x, y, and z positions to track and 3N angles to track (so 6N variables in total)

    # lower_bounds = [0] * (6 * N)
    pos_upper_bounds = [V_x, V_y, V_z] * N
    angle_upper_bounds = [180] * (N_p)
    upper_bounds = pos_upper_bounds + angle_upper_bounds
    bounds = Bounds(0, upper_bounds)

    # provide an initial guess of the cameras' positions
    # TODO: change this so that each variable is chosen randomly from between the appropriate bounds
    x0 = [0] * (6 * N)
    res = minimize(objective_function, x0, bounds=bounds)