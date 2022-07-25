import numpy as np
from optimizer import angle_between, epsilon, gdop, grid_dimensions, in_fov, N, NUM_VAR
from optimizer import sigmoid, V_x, V_y, V_z


# TODO: define another metric that counts the number of points that are 
# sufficiently triangulable (i.e. in the range [40, 140])
def overall_triangulability(x):
    total = 0

    kappa = 90
    worst_triangulability_angle = 90

    n_x, n_y, n_z = grid_dimensions(V_x, V_y, V_z)
    for a in range(n_x):
        p_x = epsilon * a 
        for b in range(n_y):
            p_y = epsilon * b
            for c in range(n_z):
                p_z = epsilon * c
                grid_point = np.array([p_x, p_y, p_z])

                triangulable_cams = []
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
                        triangulable_cams.append((pos_vec, orientation_vec))

                if fov_count == 0:
                    total += (worst_triangulability_angle + kappa)
                elif fov_count == 1:
                    total += worst_triangulability_angle
                else:
                    min_triangulability_angle = 90
                    # loop through each pair of cameras and find the angle closest to 90 DEGREES
                    for j in range(len(triangulable_cams)):
                        cam_j = triangulable_cams[j]
                        pos_j, orientation_j = cam_j
                        for k in range(j+1, len(triangulable_cams)):
                            cam_k = triangulable_cams[k] 
                            pos_k, orientation_k = cam_k

                            # get vector from point to pos and point to comp_pos
                            point_to_j = np.subtract(pos_j, grid_point)
                            point_to_k = np.subtract(pos_k, grid_point)
                            # compute angle between the two pos vectors (in degrees) and see if triangulable
                            alpha = angle_between(point_to_j, point_to_k)

                            min_triangulability_angle = min(min_triangulability_angle, alpha)

                    total += min_triangulability_angle
    return total


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
    # TODO: try including linear renormalizing constant (instead of sigmoid)
    return metric


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


def coverage(x):
    # loop over all points in the grid defined by cutting V every epsilon meters
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

                if fov_count >= 2:
                    num_seen_points += 1
    
    coverage = num_seen_points / total_points
    return coverage