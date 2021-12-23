import plotly.graph_objs as go
import matplotlib.pyplot as plt
import numpy as np
# from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors, KDTree, DistanceMetric


def KNN_from_vectors(V, K):
    nbrs = NearestNeighbors(n_neighbors=K + 1, algorithm='kd_tree').fit(V)
    distances, indices = nbrs.kneighbors(V)
    distances, indices = distances[:, 1:], indices[:, 1:]
    return distances, indices


def add_sphere(ax, scale):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = scale * np.outer(np.cos(u), np.sin(v))
    y = scale * np.outer(np.sin(u), np.sin(v))
    z = scale * np.outer(np.ones(np.size(u)), np.cos(v))
    # Setting an edgecolor appears to solve the issue of the surface hiding the points behind it.
    # However, you may want to do this if plotting the triangulation network on the surface
    ax.plot_surface(x, y, z, alpha=0.04, edgecolor='k')

# def add_delaunay(ax, emb):
#     tri = Delaunay(emb)
#     edges = np.concatenate((tri.simplices[:, :2], tri.simplices[:, 1:3]), axis=0)
#     for j in range(len(edges)):
#         line = np.array([emb[edges[j, 0]], emb[edges[j, 1]]])
#         ax.plot3D(line[:, 0], line[:, 1], line[:, 2],'-', markerfacecolor='black', linewidth=1, color='black')

def add_KNN_network(ax, emb, plot_cap):
    distances, indices = KNN_from_vectors(emb, 4)
    for j in range(len(indices)):
        for neighbor in range(indices.shape[1]):
            line = np.array([emb[j], emb[indices[j, neighbor]]])
            if plot_cap and emb[j, 2] > 0:
                ax.plot3D(line[:, 0], line[:, 1], line[:, 2], '-', linewidth=0.5, color='black')
            elif plot_cap==False:
                ax.plot3D(line[:, 0], line[:, 1], line[:, 2], '-', linewidth=0.5, color='black')



class NDScatter:
    """
    Class for plotting result of UOS algorithm

        Parameters
    ------------
    emb: ndarray
            Each row should represent a vector in emb.shape[1]-dimensional space

    UOS: UniformOrientationSampling object
            UniformOrientationSampling object. See Uniform_Orientation_Sampling.py

    make_raster:  bool
            whether to plot matrix containing embedding when dimensions>3

    save_path: str
            where to save plots
    """

    def __init__(self, emb, UOS, make_raster=False, save_path=None):
        self.emb = emb
        self.dimensions = UOS.dimensions
        self.pop_size = UOS.pop_size
        self.iterations = UOS.iterations
        self.approach = UOS.approach
        self.save_path = save_path
        self.make_raster = make_raster

    def scatter2D(self):
        a = 7
        fig = plt.figure(figsize=(1.7778 * a,
                                  a))  # e.g. figsize=(4, 3) --> img saved has resolution (400, 300) width by height when using dpi='figure' in savefig

        plt.scatter(self.emb[:, 0], self.emb[:, 1], s=20, label='emb',
                    edgecolors='k', marker='o', facecolors='none')

        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.0])

        plt.title('Embedding')
        plt.gca().set_aspect('equal')
        # plt.legend(loc="lower right")
        plt.show()
        if self.save_path is not None:
            fig.savefig(self.save_path + 'emb_' + self.approach + '_' + str(self.dimensions) + 'D_N' + str(
                self.pop_size) + '.png', transparent=False, dpi='figure', bbox_inches=None)

    def scatter3D_view(self):
        if self.emb.shape[1] == 3:
            fig = go.Figure(data=[go.Scatter3d(
                name='training images',
                x=self.emb[:, 0],
                y=self.emb[:, 1],
                z=self.emb[:, 2],
                mode='markers',
                marker=dict(
                    size=1.5, color='black', symbol='circle')
            )])

            fig.show()

            if self.save_path is not None:
                fig.write_image(
                    self.save_path + 'emb_' + self.approach + '_' + str(
                        self.dimensions) + 'D_N' + str(self.pop_size) + '.png')

    def plot_raster(self):
        plt.figure(figsize=(20, 7))
        ax = plt.axes()
        plt.pcolormesh(np.arange(0, self.dimensions, 1), np.arange(0, self.pop_size), self.emb, shading='nearest',
                       cmap='inferno')
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        ax.set_yscale('linear')

        ax.set_xlabel('component', fontsize=15)
        ax.set_ylabel('orientation vector', fontsize=15)

        cbar = plt.colorbar(ax=ax)
        cbar.set_label('value', fontsize=15)
        plt.show()

    def scatter3D_mpl(self, plot_cap=True):

        a = 8  # hundred pixels
        phi = 55
        theta = 45

        fig2 = plt.figure(figsize=(1.7778 * a, a))
        ax = fig2.add_subplot(projection='3d')
        add_sphere(ax, 0.995)
        cap = self.emb[:, 2] > 0
        marker_size = lambda popsize: 100/popsize**0.6

        if plot_cap and self.pop_size>8:
            ax.plot3D(self.emb[cap, 0], self.emb[cap, 1], self.emb[cap, 2], 'o', markerfacecolor='black',
                      markersize=marker_size(self.pop_size), color='black')
        else:
            ax.plot3D(self.emb[:, 0], self.emb[:, 1], self.emb[:, 2], 'o', markerfacecolor='black',
                      markersize=marker_size(self.pop_size), color='black')

        # add_delaunay(ax, self.emb)  #failed since some tri
        # add_KNN_network(ax, self.emb, plot_cap)

        ax.dist = 8
        ax.view_init(phi, theta)  # view_init(elev=None, azim=None)
        ax.grid(False)

        if self.save_path is not None:
            fig2.savefig(self.save_path + 'emb_' + self.approach + '_' + str(self.dimensions) + 'D_N' + str(
                self.pop_size) + '.png', transparent=False, dpi='figure', bbox_inches=None)

    def ndscatter(self):
        if self.emb.shape[1] == 2:
            self.scatter2D()
        elif self.emb.shape[1] == 3:
            self.scatter3D_view()
        elif self.dimensions > 3 and self.make_raster:
            self.plot_raster()
