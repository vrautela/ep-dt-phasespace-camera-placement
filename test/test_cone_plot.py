from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

from optimization.consts import fov_base_radius, fov_degree, fov_lower_bound, fov_upper_bound
from optimization.utils import rotation_matrix_align_two_vecs


def connect_two_points(ax, p1, p2):
    (x1, y1, z1) = p1
    (x2, y2, z2) = p2
    ax.plot([x1,x2],[y1,y2],[z1,z2], color='blue')


def main():
    # fig = plt.figure()
    # ax = fig.add_subplot(111,projection='3d')

    # # Set up the grid in polar
    # theta = np.linspace(0,2*np.pi,90)
    # r = np.linspace(0,3,50)
    # T, R = np.meshgrid(theta, r)

    # # Then calculate X, Y, and Z
    # X = R * np.cos(T)
    # Y = R * np.sin(T)
    # Z = np.sqrt(X**2 + Y**2)/fov_degree

    # print(f'X: {X}')
    # print(f'len X: {X.shape}')
    # print(f'Y: {Y}')
    # print(f'len Y: {Y.shape}')
    # print(f'Z: {Z}')
    # print(f'len Z: {Z.shape}')

    # # for i in range(50):
    # #     for j in range(90):
    # #         print(f'({X[i][j]}, {Y[i][j]}, {Z[i][j]})')

    # cone_axis = np.array([2,1,1])
    # z_hat = np.array([0,0,1])
    # R = rotation_matrix_align_two_vecs(z_hat, cone_axis)

    # X_new = []
    # Y_new = []
    # Z_new = []
    # for i in range(50):
    #     X_row = []
    #     Y_row = []
    #     Z_row = []
    #     for j in range(90):
    #         p = np.array([X[i][j], Y[i][j], Z[i][j]]) 
    #         p_rot = np.matmul(R, p)
    #         X_row.append(p_rot[0])
    #         Y_row.append(p_rot[1])
    #         Z_row.append(p_rot[2])
    #     X_new.append(X_row)
    #     Y_new.append(Y_row)
    #     Z_new.append(Z_row)

    # X_new = np.array(X_new)
    # Y_new = np.array(Y_new)
    # Z_new = np.array(Z_new)

    # # Set the Z values outside your range to NaNs so they aren't plotted
    # # Z[Z < fov_lower_bound] = np.nan
    # # Z[Z > fov_upper_bound] = np.nan
    # # ax.plot_wireframe(X, Y, Z)

    # # Z_new[Z_new < fov_lower_bound] = np.nan
    # # Z_new[Z_new > fov_upper_bound] = np.nan
    # ax.plot_wireframe(X_new, Y_new, Z_new)

    # # ax.set_zlim(0,2)

    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')

    # 1. points in a cone w/ vertex at origin and axis pointing in z dir
    x0 = 0
    y0 = 0
    z0 = 0
    vertex = np.array([x0, y0, z0])

    k = 90
    theta = np.linspace(0,2*np.pi,k)

    x = fov_base_radius * np.cos(theta)
    y = fov_base_radius * np.sin(theta)
    z = [fov_upper_bound] * k

    # k2 = 360
    # theta2 = np.linspace(0,2*np.pi, k2)
    # x_base = fov_base_radius * np.cos(theta2)
    # y_base = fov_base_radius * np.sin(theta2) 
    # z_base = [fov_upper_bound] * k2 


    # 2. rotate cone endpoints so that the z-axis is aligned with the cone axis
    axis = np.array([1,1,0])
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

    # for i in range(k2):
    #     p = np.array([x_base[i], y_base[i], z_base[i]])
    #     # 2. rotate cone endpoints so that the z-axis is aligned with the cone axis
    #     p_rot = np.matmul(R, p)
    #     # 3. shift cone points by vertex
    #     for j in range(3):
    #         p_rot += vertex[j]
    #     x_base[i] = p_rot[0]
    #     y_base[i] = p_rot[1]
    #     z_base[i] = p_rot[2]
    # # 4. plot base of cone
    # ax.plot(x_base, y_base, z_base) 

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()

if __name__ == '__main__':
    main()