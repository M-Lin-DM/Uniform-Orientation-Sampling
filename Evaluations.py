import numpy as np
from utils import normalize_rows
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from Plotting_tools import NDScatter


def angle_between(p, d):
    return np.arccos(np.dot(p, d)/(np.linalg.norm(p) * np.linalg.norm(d)))

def angle_distance_matrix(V):
    """
    Parameters
    ------------
    V: ndarray
            Each row should represent a vector in V.shape[1]-dimensional space

    Returns
    ---------
    M:  ndarray
            square array of shape (V.shape[0], V.shape[0]) . Distance matrix recording the angle between two vectors.   "
    """

    V = normalize_rows(V)
    M = np.arccos(V @ V.transpose())
    M[np.isnan(M)] = 0  # there were some nans appearing on the diagonal of the matrix.
    return M  # distance separating all points, in terms of angle between them

def KNNCustomDistanceMetric(distance_matrix, K):
    """
    We would like to determine whether the algorithm has successfully sampled uniformly on a hypersphere when the
    embedding dimensionality is greater than 3.

    Parameters
    ------------
    distance_matrix: ndarray
            Precomputed Distance matrix

    K: int
            Number of nearest neighbors' distances to pool in taking the distribution of such distances.

    Returns
    ---------
    distances:  ndarray
            distances from point at row i to its nearest neighbors. See sklearn.neighbors docs.

    indices: ndarray
            See sklearn.neighbors docs.
    """
    # precomputed distance matrix. I found this is the easiest way to implement a custom distance metric in KNN
    # search from NearestNeighbors documentation: If metric is “precomputed”, [X aka "theta"] is assumed to be a
    # distance matrix and must be square during fit":
    nbrs_obj = NearestNeighbors(n_neighbors=K + 1, algorithm='brute', metric='precomputed').fit(distance_matrix)
    # if metric == ‘precomputed’, default input for X ie data=None:
    distances, indices = nbrs_obj.kneighbors()
    distances, indices = distances[:, 1:], indices[:, 1:]
    #untested alternatives if using data table instead of dist matrix
    # kdt = KDTree(X, leaf_size=30, metric=dist)
    # distances, indices = kdt.query(X, k=2, return_distance=True)
    return distances, indices



class KNNDistanceDistribution:
    """
    Inspect the distance distribution of the K-nearest neighbors, aggregated over all points. Compare it to the
    same distribution when taking a naive approach to orientation sampling.

        Parameters
    ------------
    emb: ndarray
            Each row should represent a vector in emb.shape[1]-dimensional space

    UOS: UniformOrientationSampling object
            UniformOrientationSampling object. See Uniform_Orientation_Sampling.py

    save_path_naive:  str
            where to save plot of naive, projection-based embedding

    save_path_histogram: str
            where to save plot of nearest-neighbors distance distribution

    show_hist: bool
            whether to plot histograms
    """
    def __init__(self, emb, UOS, save_path_naive=None, save_path_histogram=None, show_hist=True):
        self.emb = emb
        self.dimensions = UOS.dimensions
        self.pop_size = UOS.pop_size
        self.approach = UOS.approach
        self.K_include = int(UOS.pop_size*0.1)  # number of nearest neighbors to include in the collection of distances that will be histogrammed
        self.save_path_naive = save_path_naive
        self.save_path_histogram = save_path_histogram
        self.UOS = UOS
        self.show_hist = show_hist

    def generate_naive_sample(self):
        return normalize_rows(np.random.rand(self.pop_size, self.dimensions) - 0.5)

    def get_KNNDD(self):
        """
        Computes and plots the distributions of distances from a point to its nearest neighbors. This is used to
        inspect the embedding and compare it to a naive approach emb. The distribution acts as a signature since
        here should be sharp peaks in probability at certain distances away from the center point.
        """
        max_angle = np.pi / 2
        points_per_bin = 25

        if self.pop_size <= 20:
            print('pop_size is too small to perform this evaluation. Statistics taken may be too noisy. Increase pop_size above 20 or decrease \'K_include\' in KNNDistanceDistribution class')
            pass
        else:
            # get KNN distances for our algorithms embedding:
            theta = angle_distance_matrix(self.emb)
            distances, indices = KNNCustomDistanceMetric(theta, self.K_include)
            edges = np.linspace(0, max_angle, int(len(distances.flatten())/points_per_bin), endpoint=True)  # edges in distance from center point

            # get KNN distances for naive embedding:
            naive_emb = self.generate_naive_sample()
            theta_naive = angle_distance_matrix(naive_emb)
            distances_naive, indices_naive = KNNCustomDistanceMetric(theta_naive, self.K_include)

            fig, axs = plt.subplots(2, 1, sharey=True, sharex=True, tight_layout=False)
            axs[0].hist(distances_naive.flatten(), bins=edges, density=True, color='k')
            axs[0].set_title('Naive Embedding')
            # axs[0].set_xlabel('distance from point to neighbor (rad)', fontsize=10)
            axs[0].set_ylabel('probability density', fontsize=10)
            # plt.tick_params('x', labelbottom=False)  # make these tick labels invisible

            axs[1].hist(distances.flatten(), bins=edges, density=True)
            axs[1].set_title('KNN repulsion Embedding')
            axs[1].set_xlabel('distance from point to neighbor (rad)', fontsize=10)
            axs[1].set_ylabel('probability density', fontsize=10)
            plt.tick_params('x', labelsize=9)
            plt.xlim(0, 0.8)

            if self.show_hist:
                plt.show()  # show distributions

            if self.save_path_histogram is not None:
                fig.savefig(self.save_path_histogram + 'hist_' + self.approach + '_' + str(self.dimensions) + 'D_N' + str(self.pop_size) + '_K_include' + str(self.K_include) + '.png', transparent=False, dpi='figure', bbox_inches=None)


    def save_naive_emb(self):
        naive_emb = self.generate_naive_sample()
        if self.save_path_naive is not None:
            if self.emb.shape[1] == 3:
                plotter = NDScatter(naive_emb, self.UOS, make_raster=False, save_path=self.save_path_naive)
                plotter.scatter3D_mpl()  # saves plot to save_path