import math
import numpy as np
from optimizer import epsilon, gdop, grid_dimensions, in_fov, N, NUM_VAR, V_x, V_y, V_z

def penalty_weighted_gdop_sum(x):
    total = 0

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
    
    coverage = num_seen_points / total_points
    penalty_factor = 1 / coverage
    metric = sigmoid(total) * penalty_factor
    return metric


def sigmoid(x):
    return 1/(1 + math.exp(-x))

def num_points_below_gdop_threshold(x, threshold):
    total = 0

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
                    g = gdop(reachable_cams, grid_point) 
                    if g < threshold:
                        total += 1
                    num_seen_points += 1
    
    coverage = num_seen_points / total_points
    penalty_factor = 1 / coverage
    metric = total * penalty_factor
    return metric