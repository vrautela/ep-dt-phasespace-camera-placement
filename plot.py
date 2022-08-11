from matplotlib.colors import ListedColormap
import matplotlib.pylab as plt
import numpy as np

from optimization.consts import epsilon, N, NUM_VAR, V, V_x, V_y, V_z
from optimization.obstacles import CylinderObstacle 
from optimization.utils import gdop, grid_dimensions, in_fov

n_x, n_y, n_z = grid_dimensions(V_x, V_y, V_z)


def create_gdop_grid(x, obstacles):
    gdop_grid = []
    for a in range(n_x):
        p_x = epsilon * a 
        gdop_plane = []
        for b in range(n_y):
            p_y = epsilon * b
            gdop_row = []
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
                        obscured = False
                        # check if any of the obstacles obscures the camera's vision
                        for ob in obstacles:
                            if ob.does_line_segment_intersect(pos_vec, grid_point):
                                obscured = True
                                break
                        
                        if not obscured:
                            fov_count += 1
                            reachable_cams.extend([cam_x + 0.01, cam_y, cam_z])
                            reachable_cams.extend([cam_x - 0.01, cam_y, cam_z])
                            # reachable_cams.extend([cam_x, cam_y, cam_z])
                
                if fov_count >= 3:
                    gdop_row.append(gdop(reachable_cams, grid_point))
                else:
                    gdop_row.append(np.NaN)

            gdop_plane.append(gdop_row)
        gdop_grid.append(gdop_plane)

    return np.array(gdop_grid)


def create_visibility_grid(x, obstacles):
    # loop over all points in the grid defined by cutting V every epsilon meters
    n_x, n_y, n_z = grid_dimensions(V_x, V_y, V_z)

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
                        obscured = False
                        for ob in obstacles:
                            if ob.does_line_segment_intersect(pos_vec, grid_point):
                                obscured = True
                                break
                        
                        if not obscured:
                            fov_count += 1
                visibility_row.append(fov_count)
            visibility_plane.append(visibility_row)
        visibility_grid.append(visibility_plane)
    return np.array(visibility_grid)


# A heatmap that allows you to scroll through cross-sections along the y-axis
class ScrollableHeatmap:
    def __init__(self, ax, visibility_grid):
        self.ax = ax
        self.visibility_grid = visibility_grid

        xlen, self.ylen, zlen = visibility_grid.shape
        self.ind = self.ylen // 2
        # orient the plot data so that the origin is at x = 0, z = 0
        self.plot_data = np.transpose(self.visibility_grid[:, self.ind, :])[::-1]

        row_labels = [i * epsilon for i in range(zlen - 1, -1, -1)] 
        col_labels = [i * epsilon for i in range(xlen)] 

        self.vmin = np.amin(self.visibility_grid)
        self.vmax = np.amax(self.visibility_grid)

        self.im, self.cbar = heatmap(
                                self.plot_data, row_labels, col_labels, cmap="magma_r", 
                                cbarlabel="Number of overlapping FOVs", vmin=self.vmin, vmax=self.vmax
                            )
        self.ax.set_title(f'Camera visibility at Y = {self.ind * epsilon} m')

    def on_scroll(self, event):
        # print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = min(self.ind + 1, self.ylen - 1)
        else:
            self.ind = max(self.ind - 1, 0)
        self.update()

    def update_data(self):
        self.plot_data = np.transpose(self.visibility_grid[:, self.ind, :])[::-1]
        # print(self.plot_data)

    def update(self):
        self.update_data()
        self.im.set_data(self.plot_data)
        self.ax.set_title(f'Camera visibility at Y = {self.ind * epsilon} m')
        self.im.axes.figure.canvas.draw()


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    return im, cbar


def create_heatmap_plot(solution):
    visibility_grid = create_visibility_grid(solution)

    fig, ax = plt.subplots()
    heatmap = ScrollableHeatmap(ax, visibility_grid)
    fig.canvas.mpl_connect('scroll_event', heatmap.on_scroll)

    plt.show()


# Create a scatter plot where each point is colorcoded based on whether it is visible or not
def create_visibility_3d_scatter_plot(solution, obstacles):
    visibility_grid = create_visibility_grid(solution, obstacles)

    xs = []
    ys = []
    zs = []
    for a in range(n_x):
        p_x = epsilon * a 
        for b in range(n_y):
            p_y = epsilon * b
            for c in range(n_z):
                p_z = epsilon * c
                xs.append(p_x)
                ys.append(p_y)
                zs.append(p_z)
    
    v = visibility_grid.flatten()
    c = []
    for el in v:
        if el >= 2:
            c.append('red') 
        else:
            c.append('black')

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    num_els = len(xs)
    s = 8

    x_visible = [xs[i] for i in range(num_els) if c[i] == 'red']
    y_visible = [ys[i] for i in range(num_els) if c[i] == 'red']
    z_visible = [zs[i] for i in range(num_els) if c[i] == 'red']
    p_visible = ax.scatter(x_visible, y_visible, z_visible, c='red', alpha=1, s=s, label='visible')
    
    x_invisible = [xs[i] for i in range(num_els) if c[i] == 'black']
    y_invisible = [ys[i] for i in range(num_els) if c[i] == 'black']
    z_invisible = [zs[i] for i in range(num_els) if c[i] == 'black']
    p_invisible = ax.scatter(x_invisible, y_invisible, z_invisible, c='black', alpha=1, s=s, label='invisible')

    for ob in obstacles:
        ob.add_to_plot(fig, ax)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title('Visibility values')
    ax.legend(loc='upper left', bbox_to_anchor=(1,1))

    plt.show()


def create_3d_scatter_plot(solution):
    # gdop_grid[x][y][z] == gdop value @ (x,y,z) || NaN
    gdop_grid = create_gdop_grid(solution)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    xs = []
    ys = []
    zs = []
    for a in range(n_x):
        p_x = epsilon * a 
        for b in range(n_y):
            p_y = epsilon * b
            for c in range(n_z):
                p_z = epsilon * c
                xs.append(p_x)
                ys.append(p_y)
                zs.append(p_z)
                
    print(np.nanmax(gdop_grid))
    print(np.nanmin(gdop_grid))
    print(np.nanmean(gdop_grid))
    print(np.nanstd(gdop_grid))

    clipped_grid = replace_outliers_with_nan(gdop_grid.flatten())

    mi = np.nanmin(clipped_grid)
    ma = np.nanmax(clipped_grid)
    c = (clipped_grid - mi) / (ma - mi)

    x_nan = []
    y_nan = []
    z_nan = []
    x_num = []
    y_num = []
    z_num = []
    c_num = []
    for i in range(len(c)):
        if not np.isfinite(c[i]):
            x_nan.append(xs[i])
            y_nan.append(ys[i])
            z_nan.append(zs[i])
        else:
            x_num.append(xs[i])
            y_num.append(ys[i])
            z_num.append(zs[i])
            c_num.append(c[i])

    p1 = ax.scatter(x_nan, y_nan, z_nan, color='pink', alpha=1)
    p2 = ax.scatter(x_num, y_num, z_num, c=c_num, alpha=1, cmap='magma')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    # title includes the original min/max of the gdop grid (so colorbar is fraction of that max)
    ax.set_title(f'GDOP values w/ min={round(mi, 2)} and max={round(ma, 2)}') 

    cbar = fig.colorbar(p2, ax=ax)
    cbar.set_label('% of max GDOP value')


    plt.show()


def replace_outliers_with_nan(data, num_std_devs = 2):
    print(f'num replaced: {len(data[abs(data - np.nanmean(data)) > num_std_devs * np.nanstd(data)])}')
    data[abs(data - np.nanmean(data)) > num_std_devs * np.nanstd(data)] = np.nan
    return data


def create_discrete_gdop_3d_scatter_plot(solution, obstacles):
    gdop_grid = create_gdop_grid(solution, obstacles)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    xs = []
    ys = []
    zs = []
    for a in range(n_x):
        p_x = epsilon * a 
        for b in range(n_y):
            p_y = epsilon * b
            for c in range(n_z):
                p_z = epsilon * c
                xs.append(p_x)
                ys.append(p_y)
                zs.append(p_z)
                
    print(np.nanmax(gdop_grid))
    print(np.nanmin(gdop_grid))
    print(np.nanmean(gdop_grid))
    print(np.nanstd(gdop_grid))

    clipped_grid = gdop_grid.flatten()

    print()
    print(np.nanmax(clipped_grid))
    print(np.nanmin(clipped_grid))
    print(np.nanmean(clipped_grid))
    print(np.nanstd(clipped_grid))

    c = clipped_grid

    x_nan = []
    y_nan = []
    z_nan = []
    x_num = []
    y_num = []
    z_num = []
    c_num = []
    color_2 = 'black'
    color_5 = 'blue'
    color_10 = 'yellow'
    color_20 = 'orange'
    color_else = 'red'
    for i in range(len(c)):
        if not np.isfinite(c[i]):
            x_nan.append(xs[i])
            y_nan.append(ys[i])
            z_nan.append(zs[i])
        else:
            x_num.append(xs[i])
            y_num.append(ys[i])
            z_num.append(zs[i])
            if c[i] < 2:
                c_num.append(color_2)
            elif c[i] < 5:
                c_num.append(color_5)
            elif c[i] < 10:
                c_num.append(color_10)
            elif c[i] < 20:
                c_num.append(color_20)
            else:
                c_num.append(color_else)


    s = 8
    p_nan = ax.scatter(x_nan, y_nan, z_nan, color='pink', alpha=1, s=s, label='g undefined')

    num_els = len(x_num)
    x_2 = [x_num[i] for i in range(num_els) if c_num[i] == color_2]
    y_2 = [y_num[i] for i in range(num_els) if c_num[i] == color_2]
    z_2 = [z_num[i] for i in range(num_els) if c_num[i] == color_2]
    p_2 = ax.scatter(x_2, y_2, z_2, color=color_2, alpha=1, s=s, label='g < 2')

    x_5 = [x_num[i] for i in range(num_els) if c_num[i] == color_5]
    y_5 = [y_num[i] for i in range(num_els) if c_num[i] == color_5]
    z_5 = [z_num[i] for i in range(num_els) if c_num[i] == color_5]
    p_5 = ax.scatter(x_5, y_5, z_5, color=color_5, alpha=1, s=s, label='2 < g < 5')

    x_10 = [x_num[i] for i in range(num_els) if c_num[i] == color_10]
    y_10 = [y_num[i] for i in range(num_els) if c_num[i] == color_10]
    z_10 = [z_num[i] for i in range(num_els) if c_num[i] == color_10]
    p_10 = ax.scatter(x_10, y_10, z_10, color=color_10, alpha=1, s=s, label='5 < g < 10')

    x_20 = [x_num[i] for i in range(num_els) if c_num[i] == color_20]
    y_20 = [y_num[i] for i in range(num_els) if c_num[i] == color_20]
    z_20 = [z_num[i] for i in range(num_els) if c_num[i] == color_20]
    p_20 = ax.scatter(x_20, y_20, z_20, color=color_20, alpha=1, s=s, label='10 < g < 20')

    x_else = [x_num[i] for i in range(num_els) if c_num[i] == color_else]
    y_else = [y_num[i] for i in range(num_els) if c_num[i] == color_else]
    z_else = [z_num[i] for i in range(num_els) if c_num[i] == color_else]
    p_else = ax.scatter(x_else, y_else, z_else, color=color_else, alpha=1, s=s, label='20 < g')

    for ob in obstacles:
        ob.add_to_plot(fig, ax)


    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'GDOP values')

    ax.legend(loc='upper left', bbox_to_anchor=(1,1))

    plt.show()


def main():
    solution = [ 0.        ,  3.56804253,  0.        ,  0.77552265,  0.4802122 ,
        0.11486522,  0.8690316 ,  2.9       ,  2.35207239,  0.69989155,
        0.        , 11.34447468,  0.        ,  0.829442  ,  5.57522773,
        0.        ,  9.34701093,  2.9       ,  2.33266765,  5.7501486 ,
        3.90661141,  1.95856593,  0.        ,  0.5680979 ,  2.02945637,
        3.97061831,  3.85644206,  2.9       ,  2.33583625,  2.55654001,
        3.98      ,  8.99297681,  0.        ,  0.81386387,  3.546532  ,
        3.98      , 10.96330585,  2.9       ,  2.20996887,  3.5915717 ]
    obstacles = [CylinderObstacle(np.array([1.5,6,1.5]), np.array([2.5,6,1.5]), 0.5)]
    create_visibility_3d_scatter_plot(solution, obstacles)
if __name__ == '__main__':
    main()