import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from optimizer import in_fov

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

def create_visibility_grid(x):
    # loop over all points in the grid defined by cutting V every epsilon meters
    n_x = int(V_x / epsilon)
    n_y = int(V_y / epsilon)
    n_z = int(V_z / epsilon)

    visibility_grid = []
    for a in range(n_x):
        p_x = epsilon * a 
        visibility_plane = []
        for b in range(n_y):
            p_y = epsilon * b
            visibility_row = []
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
                visibility_row.append(fov_count)
            visibility_plane.append(visibility_row)
        visibility_grid.append(visibility_plane)
    return np.array(visibility_grid)



def create_plots(solution):
    '''
    We need to loop over all of the points in the grid and create a multidimensional array
    which associates each point in 3d space with the number of cameras that can see it

    Then, following the image slicer example (https://matplotlib.org/stable/gallery/event_handling/image_slices_viewer.html)
    and the colormap example (https://matplotlib.org/stable/tutorials/colors/colormap-manipulation.html)
    I can create a tool that allows you to scroll through the slices and see
    what the camera values are for each cross section
    '''

    # visibility grid is a 3d numpy array
    visibility_grid = create_visibility_grid(solution)

    print('grid shape: ', visibility_grid.shape)
    # for now just create a plot at midplane (y = 6)
    xlen, ylen, zlen = visibility_grid.shape
    # midplane data should be a 2d array (at y = 6)
    midplane_data = visibility_grid[:, ylen // 2, :]

    print('midplane data: ', midplane_data)

    z_reversed_plot_data = np.transpose(midplane_data)
    print('z reversed plot data: ', z_reversed_plot_data)
    plot_data = z_reversed_plot_data[::-1]
    print('plot data: ', plot_data)

    fig, ax = plt.subplots()
    im = ax.imshow(plot_data, cmap="magma_r")
    ax.set_xticks(np.arange(xlen), [i * epsilon for i in range(xlen)])
    ax.set_yticks(np.arange(zlen), [i * epsilon for i in range(zlen - 1, -1, -1)])

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Number of overlapping FOVs", rotation=-90, va="bottom")

    ax.set_title('Camera visibility at Y = 6 m')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')

    plt.show()



def main():
    solution = [ 0.34519832,  1.0259704 ,  0.25396691,  1.29661232,  1.2120396 ,
        0.34500489,  1.03120701,  2.76729332,  1.84616327,  1.20955543,
        0.30413374, 11.41595725,  0.24936854,  1.2997024 ,  5.07884353,
        0.30277722, 11.39593676,  2.72525332,  1.84084388,  5.07583847,
        3.79446325,  1.01985884,  0.25066651,  1.29552767,  1.94118441,
        3.79462174,  1.02350275,  2.76593641,  1.84739336,  1.94143476,
        3.73833906, 11.41623886,  0.21123267,  1.29926624,  4.35031007,
        3.75234813, 11.4249762 ,  2.71981602,  1.84278996,  4.40572014]
    create_plots(solution)

if __name__ == '__main__':
    main()