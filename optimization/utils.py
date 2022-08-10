from math import sqrt
import numpy as np
from optimization.consts import epsilon, fov_base_radius, fov_lower_bound, fov_upper_bound


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
    # print(f'A:\n{A}')
    A_times_transpose = np.matmul(A.T, A)
    # print(f'A times transpose:\n{A_times_transpose}')
    Q = np.linalg.inv(A_times_transpose)
    # print(f'Q:\n{Q}')
    return sqrt(np.trace(Q))


def grid_dimensions(V_x, V_y, V_z):
    n_x = int(V_x / epsilon) + 1
    n_y = int(V_y / epsilon) + 1
    n_z = int(V_z / epsilon) + 1

    return n_x, n_y, n_z


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

# Return the rotation matrix that rotates vector a onto vector b
def rotation_matrix_align_two_vecs(a: np.ndarray, b: np.ndarray):
    """
    R = I + [v]_x + (1/1+c)([v]_x)^2 
    """
    v = np.cross(a, b)
    c = np.dot(a, b)

    v1, v2, v3 = v[0], v[1], v[2]
    v_ss_cp = np.array([[0,-v3,v2], [v3,0,-v1], [-v2,v1,0]])
    
    I = np.eye(3)
    R = I + v_ss_cp + (1/(1+c))*np.linalg.matrix_power(v_ss_cp, 2)

    return R
