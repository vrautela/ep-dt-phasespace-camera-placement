from math import sqrt
import numpy as np
from scipy.optimize import Bounds, minimize, NonlinearConstraint

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
NUM_VAR = 5

# the scale of the grid
epsilon = 1


# return the number of points that lie in the FOV of at least two cameras
def objective_function(x):
    total = 0

    # loop over all points in the grid defined by cutting V every epsilon meters
    n_x = int(V_x / epsilon)
    n_y = int(V_y / epsilon)
    n_z = int(V_z / epsilon)
    for a in range(n_x):
        p_x = epsilon * a 
        for b in range(n_y):
            p_y = epsilon * b
            for c in range(n_z):
                p_z = epsilon * c
                grid_point = np.array([p_x, p_y, p_z])
                # print(p_x, p_y, p_z)

                fov_count = 0
                # loop over each camera to see if the point at (p_x, p_y, p_z) lies in its FOV
                for i in range(N):
                    cam_x = x[i]
                    cam_y = x[i+1]
                    cam_z = x[i+2]
                    cam_theta = x[i+3]
                    cam_phi = x[i+4]

                    pos_vec = np.array([cam_x, cam_y, cam_z])
                    # compute the unit vector defining the orientation based on the angles in spherical coords
                    orientation_x = np.cos(cam_phi) * np.sin(cam_theta)
                    orientation_y = np.sin(cam_phi) * np.sin(cam_theta)
                    orientation_z = np.cos(cam_theta)
                    orientation_vec = np.array([orientation_x, orientation_y, orientation_z])

                    if in_fov(pos_vec, orientation_vec, grid_point):
                        # print("in fov")
                        fov_count += 1

                    if fov_count >= 2:
                        # print("more than 2")
                        total += 1
                        break
    # TODO: be careful with the sign of total (depending on whether max'ing or min'ing the obj. func.) 
    return -total


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


def random_position_and_orientation():
    x = np.random.random() * V_x
    y = np.random.random() * V_y
    z = np.random.random() * V_z
    
    # generate spherical coordinate angles (in radians) to define the orientation of the FOV
    theta = np.deg2rad(np.random.random() * 180)
    phi = np.deg2rad(np.random.random() * 360)

    return [x, y, z, theta, phi]


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

    # the positions can't be less than 0, and the unit vec components can't be less than -1
    # lower_bounds = [0, 0, 0, 0, 0] * N
    # the positions can't be greater than the dimensions of V, and the unit vec components can't be greater than 1
    upper_bounds = [V_x, V_y, V_z, 180, 360] * N
    bounds = Bounds(0, upper_bounds)

    # provide an initial guess of the cameras' positions
    # TODO: change this so that each variable is chosen randomly from between the appropriate bounds
    x0 = []
    for i in range(N):
        po = random_position_and_orientation()
        print(po)
        x0.extend(po)
    
    print("minimizing")
    # res = minimize(objective_function, x0, bounds=bounds)
    res = minimize(objective_function, x0, bounds=bounds, options={"eps":1, "disp":True})
    print(res)

    solution = res.x
    print(solution)