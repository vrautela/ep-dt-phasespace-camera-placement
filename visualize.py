import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from optimization.consts import fov_base_radius, fov_upper_bound, V_x, V_y, V_z
from optimization.obstacles import CylinderObstacle
from optimization.utils import rotation_matrix_align_two_vecs

def sph2cart(r, theta, phi):
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)

    return x, y, z

def convert_array_to_cartesian(a):
    a2 = []
    for v in a:
        ox, oy, oz = sph2cart(1, v[3], v[4])
        a2.append([v[0], v[1], v[2], ox, oy, oz])
    
    return np.array(a2)


def connect_two_points(ax, p1, p2):
    (x1, y1, z1) = p1
    (x2, y2, z2) = p2
    ax.plot([x1,x2],[y1,y2],[z1,z2], color='black')


def plot_fovs(ax, soa):
    X, Y, Z, U, V, W = zip(*soa)
    ax.quiver(X, Y, Z, U, V, W, color='green')

    NUM_CAMS = len(X)
    for i in range(NUM_CAMS):
        vertex = np.array([X[i], Y[i], Z[i]])
        axis = np.array([U[i], V[i], W[i]]) 
        plot_cone(ax, vertex, axis)


def plot_cone(ax, vertex, axis):
    k = 90
    theta = np.linspace(0,2*np.pi,k)

    x = fov_base_radius * np.cos(theta)
    y = fov_base_radius * np.sin(theta)
    z = [fov_upper_bound] * k

    # 2. rotate cone endpoints so that the z-axis is aligned with the cone axis
    axis = axis / np.linalg.norm(axis)
    z_hat = np.array([0,0,1])
    R = rotation_matrix_align_two_vecs(z_hat, axis)

    for i in range(k):
        endpoint = np.array([x[i], y[i], z[i]])
        endpoint_rot = np.matmul(R, endpoint)
        # 3. shift cone points by vertex
        for j in range(3):
            endpoint_rot[j] += vertex[j]
        
        # 4. plot lines from vertex to endpoint
        connect_two_points(ax, vertex, endpoint_rot)


# TODO: change main so that it reads from result.txt
def main():

    obstacles = [CylinderObstacle(np.array([1.5,6,1.5]), np.array([2.5,6,1.5]), 0.5)]

    solution = [ 0.        ,  3.56804253,  0.        ,  0.77552265,  0.4802122 ,
        0.11486522,  0.8690316 ,  2.9       ,  2.35207239,  0.69989155,
        0.        , 11.34447468,  0.        ,  0.829442  ,  5.57522773,
        0.        ,  9.34701093,  2.9       ,  2.33266765,  5.7501486 ,
        3.90661141,  1.95856593,  0.        ,  0.5680979 ,  2.02945637,
        3.97061831,  3.85644206,  2.9       ,  2.33583625,  2.55654001,
        3.98      ,  8.99297681,  0.        ,  0.81386387,  3.546532  ,
        3.98      , 10.96330585,  2.9       ,  2.20996887,  3.5915717 ]

    pre_soa = [[solution[5*i], solution[5*i+1], solution[5*i+2], solution[5*i+3], solution[5*i+4]] for i in range(8)]
    soa = convert_array_to_cartesian(pre_soa)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plot_fovs(ax, soa)

    for ob in obstacles:
        ob.add_to_plot(fig, ax)

    # ax.set_xlim([0, V_x])
    # ax.set_ylim([0, V_y])
    # ax.set_zlim([0, V_z])
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')

    plt.show()

if __name__ == "__main__":
    main()