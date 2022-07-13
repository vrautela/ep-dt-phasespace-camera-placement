from guesses import gen_guess_box
from math import sqrt
import numpy as np
from scipy.optimize import Bounds, minimize, NonlinearConstraint

# FOV lower bound (objects must be at least this far to be seen)
fov_lower_bound = 1
# FOV upper bound a.k.a height of the cone (objects further than this are out of range)
fov_upper_bound = 5
# FOV degree (a.k.a angle between cone axis and slant)
fov_degree = 50
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


def gdop_objective_function(x):
    total = 0

    cam_positions = []
    for i in range(N):
        cam_start = i * NUM_VAR
        cam_positions.append(x[cam_start])
        cam_positions.append(x[cam_start+1])
        cam_positions.append(x[cam_start+2])


    # loop over all points in the grid defined by cutting V every epsilon meters
    # add 1 to each of the dimensions
    n_x = int(V_x / epsilon) + 1
    n_y = int(V_y / epsilon) + 1
    n_z = int(V_z / epsilon) + 1
    num_seen_points = 0
    total_points = n_x * n_y * n_z
    for a in range(n_x):
        p_x = epsilon * a 
        for b in range(n_y):
            p_y = epsilon * b
            for c in range(n_z):
                p_z = epsilon * c
                grid_point = np.array([p_x, p_y, p_z])

                reachable_cams = []
                # loop over each camera to see if the point at (p_x, p_y, p_z) lies in its FOV
                fov_count = 0
                for i in range(N):
                    cam_start = i * NUM_VAR
                    cam_x = x[cam_start]
                    cam_y = x[cam_start+1]
                    cam_z = x[cam_start+2]
                    cam_theta = x[cam_start+3]
                    cam_phi = x[cam_start+4]

                    pos_vec = np.array([cam_x, cam_y, cam_z])
                    # compute the unit vector defining the orientation based on the angles in spherical coords
                    orientation_x = np.cos(cam_phi) * np.sin(cam_theta)
                    orientation_y = np.sin(cam_phi) * np.sin(cam_theta)
                    orientation_z = np.cos(cam_theta)
                    orientation_vec = np.array([orientation_x, orientation_y, orientation_z])

                    if in_fov(pos_vec, orientation_vec, grid_point):
                        fov_count += 1
                        reachable_cams.extend([cam_x, cam_y, cam_z])

                if fov_count >= 2:
                    total += gdop(reachable_cams, grid_point) 
                    num_seen_points += 1
    
    # how to maximize coverage??
    coverage = num_seen_points / total_points
    # print('coverage: ', coverage)
    penalty_factor = 1 / coverage
    # print('penalty factor: ', penalty_factor)
    obj = total * (penalty_factor ** 6)
    # print('obj: ', obj)
    return obj


# TODO: write tests for gdop
# PRECONDITION: n >= 2
def gdop(sats, receiver):
    x = receiver[0]
    y = receiver[1]
    z = receiver[2]

    pre_A = []
    NUM_SAT_VAR = 3

    n = len(sats) // NUM_SAT_VAR
    for i in range(n):
        sat_start = i * NUM_SAT_VAR 
        x_i = sats[sat_start]
        y_i = sats[sat_start+1]
        z_i = sats[sat_start+2]
        R_i = sqrt((x_i - x)**2 + (y_i - y)**2 + (z_i - z)**2)
        # TODO: how to properly handle the case of a satellite being directly on a receiver?
        if R_i == 0:
            continue
        else:
            ith_vec = [(x_i - x)/R_i, (y_i - y)/R_i, (z_i - z)/R_i, -1]
        pre_A.append(ith_vec)
        
    A = np.array(pre_A)
    Q = np.linalg.pinv(np.matmul(A.T, A))
    return sqrt(np.trace(Q))

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
                    cam_start = i * NUM_VAR
                    cam_x = x[cam_start]
                    cam_y = x[cam_start+1]
                    cam_z = x[cam_start+2]
                    cam_theta = x[cam_start+3]
                    cam_phi = x[cam_start+4]

                    pos_vec = np.array([cam_x, cam_y, cam_z])
                    # compute the unit vector defining the orientation based on the angles in spherical coords
                    orientation_x = np.cos(cam_phi) * np.sin(cam_theta)
                    orientation_y = np.sin(cam_phi) * np.sin(cam_theta)
                    orientation_z = np.cos(cam_theta)
                    orientation_vec = np.array([orientation_x, orientation_y, orientation_z])

                    if in_fov(pos_vec, orientation_vec, grid_point):
                        fov_count += 1

                    if fov_count >= 2:
                        total += 1
                        break
    return -total


def in_fov(pos_vec, orientation_vec, grid_point):
    # get vector from tip of cone to point being tested
    tip_to_point = grid_point - pos_vec 
    # project t2p onto the orientation vector to find out how far down the cone (from the tip) the point lies
    cone_dist = np.dot(tip_to_point, orientation_vec)

    if cone_dist < fov_lower_bound or cone_dist > fov_upper_bound:
        return False

    radius_at_cone_dist = (cone_dist / fov_upper_bound) * fov_base_radius
    proj_of_point_onto_axis = cone_dist * orientation_vec
    dist_from_axis = np.linalg.norm(tip_to_point - proj_of_point_onto_axis)

    return dist_from_axis < radius_at_cone_dist


def main():
    '''
    We want to set up and solve a constrained optimization problem (using a pre-built optimizer)
    The steps required to do so are:
    1. Define an objective function
    2. Define the constraints/bounds 
    3. Provide initial guess/values
    4. Run the optimization routine
    '''

    # the positions can't be less than 0, and the unit vec components can't be less than -1
    # lower_bounds = [0, 0, 0, 0, 0] * N
    # the positions can't be greater than the dimensions of V, and the unit vec components can't be greater than 1
    # upper_bounds = [V_x, V_y, V_z, np.deg2rad(180), np.deg2rad(360)] * N
    lower_bounds = [
        0, 0, 0, np.deg2rad(0), np.deg2rad(0),
        0, 0, 0, np.deg2rad(90), np.deg2rad(0),
        0, 0, 0, np.deg2rad(0), np.deg2rad(270),
        0, 0, 0, np.deg2rad(90), np.deg2rad(270),
        0, 0, 0, np.deg2rad(0), np.deg2rad(90),
        0, 0, 0, np.deg2rad(90), np.deg2rad(90),
        0, 0, 0, np.deg2rad(0), np.deg2rad(180),
        0, 0, 0, np.deg2rad(90), np.deg2rad(180),
    ]
    upper_bounds = [
        V_x, V_y, V_z, np.deg2rad(90), np.deg2rad(90),
        V_x, V_y, V_z, np.deg2rad(180), np.deg2rad(90),
        V_x, V_y, V_z, np.deg2rad(90), np.deg2rad(360),
        V_x, V_y, V_z, np.deg2rad(180), np.deg2rad(360),
        V_x, V_y, V_z, np.deg2rad(90), np.deg2rad(180),
        V_x, V_y, V_z, np.deg2rad(180), np.deg2rad(180),
        V_x, V_y, V_z, np.deg2rad(90), np.deg2rad(270),
        V_x, V_y, V_z, np.deg2rad(180), np.deg2rad(270),
    ]
    bounds = Bounds(lower_bounds, upper_bounds)

    # NOTE: x0 is a list of floats because that is how one specifies upper and lower bounds using scipy's minimize
    x0 = gen_guess_box(V_x, V_y, V_z)

    print("minimizing")
    # res = minimize(objective_function, x0, bounds=bounds)
    res = minimize(gdop_objective_function, x0, bounds=bounds, options={"eps":0.1, "disp":True})
    print(res)

    f_value = res.fun
    num_points = int(V_x / epsilon) * int(V_y / epsilon) * int(V_z / epsilon)
    # print(str(-f_value) + " out of " + str(num_points))
    print(x0)


if __name__ == '__main__':
    main()