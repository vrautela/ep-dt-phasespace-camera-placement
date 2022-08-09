from matplotlib.colors import ListedColormap
import matplotlib.pylab as plt
import numpy as np
from ..optimization.optimizer import gdop, grid_dimensions, in_fov

# dimensions of V (in m)
V_x = 3.98
V_y = 12.03
V_z = 2.9
V = (V_x, V_y, V_z)

n_x, n_y, n_z = grid_dimensions(V_x, V_y, V_z)

# number of cameras
N = 8

# numbers of position variables (and also number of angle variables)
NUM_VAR = 5

# the scale of the grid
epsilon = 0.5


def create_gdop_grid(x):
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


def create_visibility_grid(x):
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


def create_visibility_3d_scatter_plot(solution):
    visibility_grid = create_visibility_grid(solution)

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

    p = ax.scatter(xs, ys, zs, c=c, alpha=1)

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

    print()
    print(np.nanmax(clipped_grid))
    print(np.nanmin(clipped_grid))
    print(np.nanmean(clipped_grid))
    print(np.nanstd(clipped_grid))

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


def create_discrete_3d_scatter_plot(solution):
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
                c_num.append('black')
            elif c[i] < 5:
                c_num.append('blue')
            elif c[i] < 10:
                c_num.append('yellow')
            elif c[i] < 20:
                c_num.append('orange')
            else:
                c_num.append('red')


    p1 = ax.scatter(x_nan, y_nan, z_nan, color='pink', alpha=1)
    p2 = ax.scatter(x_num, y_num, z_num, c=c_num, alpha=1)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    # title includes the original min/max of the gdop grid (so colorbar is fraction of that max)
    ax.set_title(f'GDOP values')

    # TODO: add key/legend to the plot


    plt.show()


def main():
    solution = [ 0.1617669 ,  2.65104963,  0.        ,  0.91824958,  0.97310717,
        0.        ,  3.76484067,  2.9       ,  2.10560244,  0.69934691,
        0.        , 11.06708181,  0.        ,  1.16079753,  5.4891033 ,
        0.02591085,  9.02950831,  2.9       ,  2.28789139,  5.98938951,
        3.81101486,  6.13502485,  0.        ,  0.76304537,  2.00578746,
        3.63758389,  3.06531044,  2.88805054,  2.1806488 ,  2.30498138,
        3.67245603,  8.9512966 ,  0.        ,  0.95314404,  3.68103238,
        3.77598245, 11.00919657,  2.9       ,  2.20416452,  3.91420568]
    create_discrete_3d_scatter_plot(solution)

if __name__ == '__main__':
    main()