import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
from optimal_camera_placement.optimization.optimizer import angle_between, sigmoid
from scipy.optimize import Bounds, minimize
# Test of GDOP vs triangulability metrics


# create a fixed camera, a fixed point, and a camera that moves around the point in a circle
def two_cam_test():
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
        cams.extend([c1[0], c1[1], c1[2]]) 
        cams.extend([c2[0], c2[1], c2[2]]) 

        x.append(alpha)
        y.append(gdop(cams, p))


    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xlabel('Angle between cams (degrees)')
    ax.set_ylabel('GDOP')
    plt.show()


# set up two fixed cams in the xy plane (with angular separation 135 degrees) and have a third cam
# rotate in the yz plane
def three_cam_test():
    c1 = [3, 0, 0, 90, 180] 
    c3 = [3 * np.cos(np.deg2rad(135)), 3 * np.sin(np.deg2rad(135)), 0, 90, 315]
    p = (0, 0, 0)

    x = []
    y = []

    # make c2 move in a circle of radius 3 around the origin in the yz plane
    for alpha in range(360):
        # define c2 to be a camera which is alpha degrees along the circumference of a circle of radius 3
        # phi can be from [0, 360]
        phi = (alpha + 180) % 360
        c2 = [0, 3 * np.cos(np.deg2rad(alpha)), 3 * np.sin(np.deg2rad(alpha)), 90, phi]

        cams = []
        cams.extend([c1[0], c1[1], c1[2]]) 
        cams.extend([c2[0], c2[1], c2[2]]) 
        cams.extend([c3[0], c3[1], c3[2]]) 

        x.append(alpha)
        y.append(gdop(cams, p))


    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xlabel('Angle between cam and xy plane (degrees)')
    ax.set_ylabel('GDOP')
    plt.show()


V_x = 5
V_y = 5
N = 3
NUM_VAR = 2
epsilon = 0.5
# TODO: need to implement a version of GDOP that uses just x,y coordinates
def optimal_2d_three_cam_test():
    lower_bounds = [-2.5] * N * NUM_VAR
    upper_bounds = [2.5] * N * NUM_VAR
    bounds = Bounds(lower_bounds, upper_bounds)

    # initial guess contains just positions of c1, c2, and c3
    # at theta1 = 0, theta2 = 90, theta3 = 180
    x0 = [2, 0, 0, 2, -2, 0]
    # x0 = [1, 0, 1, 2, -0.5, -2]

    print("minimizing")
    res = minimize(average_reciprocal_gdop, x0, bounds=bounds, options={"eps":0.1, "disp":True})
    print(res)


def count_objective_function(x):
    total = 0

    # loop over all points in the grid defined by cutting V every epsilon meters
    n_x, n_y = grid_dimensions(V_x, V_y)
    total_points = n_x * n_y
    num_seen_points = 0
    for a in range(n_x):
        p_x = epsilon * a 
        for b in range(n_y):
            p_y = epsilon * b
            grid_point = np.array([p_x, p_y])

            # loop over each camera to see if the point at (p_x, p_y, p_z) lies in the FOVs of 
            # two triangulable cameras (angle between them is between 40 and 140 degrees)
            triangulable_cams = []
            fov_count = 0
            for i in range(N):
                cam_start = i * NUM_VAR
                cam_x = x[cam_start]
                cam_y = x[cam_start+1]

                pos_vec = np.array([cam_x, cam_y])
                # compute the unit vector defining the orientation based on the angles in spherical coords

                fov_count += 1
                # check triangulability of this camera with all the other reachable ones
                # if triangulatable, then increment the total and break out of this loop
                # else, just add this camera to the reachable ones
                triangulable = False
                for cam in triangulable_cams:
                    comp_pos_vec = cam
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
                    triangulable_cams.append(pos_vec)

            if fov_count >= 2:
                num_seen_points += 1

    coverage = num_seen_points / total_points
    obj = -sigmoid(total) * coverage
    # obj can range from -1 to 0
    return obj


def average_reciprocal_gdop(x):
    total = 0

    # loop over all points in the grid defined by cutting V every epsilon meters
    grid_point = np.array([0, 0])

    # loop over each camera to see if the point at (p_x, p_y, p_z) lies in the FOVs of 
    # two triangulable cameras (angle between them is between 40 and 140 degrees)
    triangulable_cams = []
    reachable_cams = []
    triangulable = False
    for i in range(N):
        cam_start = i * NUM_VAR
        cam_x = x[cam_start]
        cam_y = x[cam_start+1]

        pos_vec = np.array([cam_x, cam_y])

        # TODO: change this so the position of the sensors inside the camera is accurate
        # Each camera has two sensors inside
        reachable_cams.extend([cam_x + 0.01, cam_y])
        reachable_cams.extend([cam_x - 0.01, cam_y])

        # check triangulability of this camera with all the other reachable ones
        if not triangulable:
            for cam in triangulable_cams:
                comp_pos_vec = cam
                # get vector from point to pos and point to comp_pos
                point_to_pos = np.subtract(pos_vec, grid_point)
                point_to_comp_pos = np.subtract(comp_pos_vec, grid_point)
                # compute angle between the two pos vectors (in degrees) and see if triangulable
                alpha = angle_between(point_to_pos, point_to_comp_pos)

                # TODO: try varying the angle bounds
                if 40 < alpha and alpha < 140:
                    triangulable = True
            triangulable_cams.append(pos_vec)
        
    if triangulable:
        g = gdop(reachable_cams, grid_point)
        f = 1/g
        total += (f * (epsilon ** 2))

    volume = V_x * V_y
    # we make objective negative so the function can be minimized
    obj = -total / volume
    return obj


def grid_dimensions(V_x, V_y):
    n_x = int(V_x / epsilon) + 1
    n_y = int(V_y / epsilon) + 1

    return n_x, n_y


# TODO: write tests for gdop
# PRECONDITION: n >= 2
def gdop(sats, receiver):
    x = receiver[0]
    y = receiver[1]

    pre_A = []
    NUM_SAT_VAR = 2

    n = len(sats) // NUM_SAT_VAR
    for i in range(n):
        sat_start = i * NUM_SAT_VAR 
        x_i = sats[sat_start]
        y_i = sats[sat_start+1]
        R_i = sqrt((x_i - x)**2 + (y_i - y)**2)
        # TODO: how to properly handle the case of a satellite being directly on a receiver?
        if R_i == 0:
            continue
        else:
            ith_vec = [(x_i - x)/R_i, (y_i - y)/R_i, -1]
        pre_A.append(ith_vec)
        
    A = np.array(pre_A)
    A_times_transpose = np.matmul(A.T, A)
    Q = np.linalg.inv(A_times_transpose)
    return sqrt(np.trace(Q))


def main():
    optimal_2d_three_cam_test()

if __name__ == '__main__':
    main()