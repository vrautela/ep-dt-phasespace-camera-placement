import math
from math import sqrt
import numpy as np
from scipy.optimize import Bounds, minimize, NonlinearConstraint
from typing import List

from optimization.guesses import gen_guess_box
from optimization.obstacles import CylinderObstacle, Obstacle

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


def grid_dimensions(V_x, V_y, V_z):
    n_x = int(V_x / epsilon) + 1
    n_y = int(V_y / epsilon) + 1
    n_z = int(V_z / epsilon) + 1

    return n_x, n_y, n_z


def average_reciprocal_gdop(x, obstacles: List[Obstacle]):
    total = 0

    # loop over all points in the grid defined by cutting V every epsilon meters
    n_x, n_y, n_z = grid_dimensions(V_x, V_y, V_z)
    for a in range(n_x):
        p_x = epsilon * a 
        for b in range(n_y):
            p_y = epsilon * b
            for c in range(n_z):
                p_z = epsilon * c
                grid_point = np.array([p_x, p_y, p_z])

                # loop over each camera to see if the point at (p_x, p_y, p_z) lies in the FOVs of 
                # two triangulable cameras (angle between them is between 40 and 140 degrees)
                triangulable_cams = []
                reachable_cams = []
                triangulable = False
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

                    # check if this camera can see the point
                    if in_fov(pos_vec, orientation_vec, grid_point):
                        obscured = False
                        # check if any of the obstacles obscures the camera's vision
                        for ob in obstacles:
                            if ob.does_line_segment_intersect(pos_vec, grid_point):
                                obscured = True
                                break
                        
                        if not obscured:
                            # TODO: change this so the position of the sensors inside the camera is accurate
                            # Each camera has two sensors inside
                            sensor_sep = 0.01
                            reachable_cams.extend([cam_x + sensor_sep, cam_y, cam_z])
                            reachable_cams.extend([cam_x - sensor_sep, cam_y, cam_z])

                            NUM_POS_VARS = 3
                            MIN_CAMS_VISIBLE = 6
                            if not triangulable and len(reachable_cams) >= MIN_CAMS_VISIBLE * NUM_POS_VARS:
                                # check triangulability of this camera with all the other reachable ones
                                for cam in triangulable_cams:
                                    comp_pos_vec, comp_orientation_vec = cam[0], cam[1]
                                    # get vector from point to pos and point to comp_pos
                                    point_to_pos = np.subtract(pos_vec, grid_point)
                                    point_to_comp_pos = np.subtract(comp_pos_vec, grid_point)
                                    # compute angle between the two pos vectors (in degrees) and see if triangulable
                                    alpha = angle_between(point_to_pos, point_to_comp_pos)

                                    # check if the angle between the two cameras falls between the triangulability bounds  
                                    if 40 < alpha and alpha < 140:
                                        triangulable = True
                                triangulable_cams.append((pos_vec, orientation_vec))

                if triangulable:
                    g = gdop(reachable_cams, grid_point)
                    f = 1/g
                    total += (f * (epsilon ** 3))

    volume = V_x * V_y * V_z 
    # we make objective negative so the function can be minimized
    obj = -total / volume
    return obj



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
    n_x, n_y, n_z = grid_dimensions(V_x, V_y, V_z)
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
    # print(f'A:\n{A}')
    A_times_transpose = np.matmul(A.T, A)
    # print(f'A times transpose:\n{A_times_transpose}')
    Q = np.linalg.inv(A_times_transpose)
    # print(f'Q:\n{Q}')
    return sqrt(np.trace(Q))


# return the value of the objective function calculated based on the number of points
# that lie in the FOV of at least two cameras that are sufficiently triangulable
def count_objective_function(x):
    total = 0

    # loop over all points in the grid defined by cutting V every epsilon meters
    n_x, n_y, n_z = grid_dimensions(V_x, V_y, V_z)
    total_points = n_x * n_y * n_z
    num_seen_points = 0
    for a in range(n_x):
        p_x = epsilon * a 
        for b in range(n_y):
            p_y = epsilon * b
            for c in range(n_z):
                p_z = epsilon * c
                grid_point = np.array([p_x, p_y, p_z])

                # loop over each camera to see if the point at (p_x, p_y, p_z) lies in the FOVs of 
                # two triangulable cameras (angle between them is between 40 and 140 degrees)
                triangulable_cams = []
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
                        # check triangulability of this camera with all the other reachable ones
                        # if triangulatable, then increment the total and break out of this loop
                        # else, just add this camera to the reachable ones
                        triangulable = False
                        for cam in triangulable_cams:
                            comp_pos_vec, comp_orientation_vec = cam[0], cam[1]
                            # get vector from point to pos and point to comp_pos
                            point_to_pos = np.subtract(pos_vec, grid_point)
                            point_to_comp_pos = np.subtract(comp_pos_vec, grid_point)
                            # compute angle between the two pos vectors (in degrees) and see if triangulable
                            alpha = angle_between(point_to_pos, point_to_comp_pos)

                            # TODO: try varying the angle bounds
                            if 40 < alpha and alpha < 140:
                                triangulable = True
                                break

                        if triangulable:
                            total += 1
                            break
                        else:
                            triangulable_cams.append((pos_vec, orientation_vec))

                if fov_count >= 2:
                    num_seen_points += 1

    coverage = num_seen_points / total_points
    obj = -sigmoid(total) * coverage
    # obj can range from -1 to 0
    return obj


def sigmoid(x):
    return 1/(1 + math.exp(-x))


def unit_vector(vector):
    return vector / np.linalg.norm(vector)

# return angle between vectors v1 and v2 IN DEGREES
def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


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

    # Spherical coordinates with azimuthal angle theta and polar angle theta
    lower_bounds = [0, 0, 0, 0, 0] * N
    upper_bounds = [V_x, V_y, V_z, np.deg2rad(180), np.deg2rad(360)] * N
    # lower_bounds = [
    #     0, 0, 0, np.deg2rad(0), np.deg2rad(0),
    #     0, 0, 0, np.deg2rad(90), np.deg2rad(0),
    #     0, 0, 0, np.deg2rad(0), np.deg2rad(270),
    #     0, 0, 0, np.deg2rad(90), np.deg2rad(270),
    #     0, 0, 0, np.deg2rad(0), np.deg2rad(90),
    #     0, 0, 0, np.deg2rad(90), np.deg2rad(90),
    #     0, 0, 0, np.deg2rad(0), np.deg2rad(180),
    #     0, 0, 0, np.deg2rad(90), np.deg2rad(180),
    # ]
    # upper_bounds = [
    #     V_x, V_y, V_z, np.deg2rad(90), np.deg2rad(90),
    #     V_x, V_y, V_z, np.deg2rad(180), np.deg2rad(90),
    #     V_x, V_y, V_z, np.deg2rad(90), np.deg2rad(360),
    #     V_x, V_y, V_z, np.deg2rad(180), np.deg2rad(360),
    #     V_x, V_y, V_z, np.deg2rad(90), np.deg2rad(180),
    #     V_x, V_y, V_z, np.deg2rad(180), np.deg2rad(180),
    #     V_x, V_y, V_z, np.deg2rad(90), np.deg2rad(270),
    #     V_x, V_y, V_z, np.deg2rad(180), np.deg2rad(270),
    # ]
    bounds = Bounds(lower_bounds, upper_bounds)

    x0 = gen_guess_box(V_x, V_y, V_z)

    # specify the obstacles present in the space
    obstacles = [CylinderObstacle(np.array([1.5,6,1.5]), np.array([2.5,6,1.5]), 1)]

    res = minimize(average_reciprocal_gdop, x0, args=(obstacles,), bounds=bounds, options={"eps":0.1, "disp":True})
    print(res)

if __name__ == '__main__':
    main()
