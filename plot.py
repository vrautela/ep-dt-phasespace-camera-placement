import matplotlib.pylab as plt
import numpy as np
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

        self.im, self.cbar = heatmap(self.plot_data, row_labels, col_labels, cmap="magma_r", cbarlabel="Number of overlapping FOVs")
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


def create_plots(solution):
    visibility_grid = create_visibility_grid(solution)

    fig, ax = plt.subplots()
    heatmap = ScrollableHeatmap(ax, visibility_grid)
    fig.canvas.mpl_connect('scroll_event', heatmap.on_scroll)

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