import numpy as np
import copy
from sklearn.neighbors import NearestNeighbors

def normalize_rows(dat):
    """
    Parameters
    ------------
    dat: ndarray
            Each row should represent a vector in dat.shape[1]-dimensional space

    Returns
    ---------
    out:  ndarray
            array with same size as dat where each row has euclidean norm equal to 1
    """
    out = dat / np.sqrt(np.sum(dat ** 2, 1))[:, None]
    out[np.where(np.isnan(out))] = 0
    return out


class Pair_separation:

    def __init__(self, eta):
        self.eta = eta

    def update_points(self, V):
        grahm = np.tril(V @ V.transpose(), -1)
        # print(np.argmax(grahm))
        p, d = np.unravel_index(np.argmax(grahm), grahm.shape)
        p2d = V[d] - V[p]
        V[p] -= p2d * self.eta
        V[d] += p2d * self.eta

        return normalize_rows(V)


class Bubble_tea_potential:

    def __init__(self, beta, gamma=0.8, alpha1=2, alpha2=3):
        self.beta = beta
        self.gamma = gamma
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def update_points(self, V):
        theta = np.arccos(V @ V.transpose())  # distance separating all points, in terms of angle between them
        W = copy.deepcopy(V)

        # FoT = lambda x: np.log((x+self.gamma)**self.alpha1) / (x+self.gamma)**self.alpha2 # magnitude of forcing weight as a function of theta
        # FoT = lambda x : -1 / (x+0.5) + 0.5
        # FoT = lambda x: x*0 - self.beta
        # FoT = lambda x: 0.1*(x - np.pi/2)
        FoT = lambda x: -0.3/(x**2+0.5)

        inter_particle_force = FoT(theta)

        for p in range(V.shape[0]):
            neib = list(range(V.shape[0]))
            neib.remove(p)
            #
            neg_phat = -normalize_rows(V[p][None, :])  #unit vector pointing in direction opposite p. we will project p2all_hat onto this
            p2all_hat = normalize_rows(V[neib] - V[p][None, :])  # unit vectors pointing from p to all other points (pointing THROUGH the sphere)
            proj_on_p_hat = np.repeat(neg_phat, V.shape[0] - 1, axis=0) * p2all_hat @ neg_phat.transpose()
            p2all_hat_tangent = p2all_hat - proj_on_p_hat  # by subtracting the component of p2all_hat that lies along the negative p_hat, we should get a set of vectors tangent to the hypersphere, originating from point p

            # W[p] += np.mean(p2all_hat * inter_particle_force[p][neib, None], 0)
            # W[p] += np.mean(p2all_hat, 0)
            W[p] += np.mean(p2all_hat_tangent * inter_particle_force[p][neib, None], 0)


        if np.any(np.isnan(W)):  # must use the isnan function. W==np.nan will return false
            # print(W)
            raise (Exception('Nan found in emb'))

        return normalize_rows(W)


class KNN_repulsion:

    def __init__(self, zeta, K=1):
        self.zeta = zeta  # learning rate for each point
        self.K = K

    def update_points(self, V):
        # The algorithm moves each point away from its K-nearest neighbors. A single loop over all points in V. K=1 is optimal.

        nbrs = NearestNeighbors(n_neighbors=self.K+1, algorithm='kd_tree').fit(V)
        distances, indices = nbrs.kneighbors(V)
        indices = indices[:, 1:]
        W = copy.deepcopy(V)

        for p in range(V.shape[0]):
            neibs = V[indices[p]]
            p2d = neibs - V[p][None, :]  # set of vectors pointing from p to its KNN
            W[p] -= np.mean(self.zeta * p2d, 0)  # move away from ones kNN
        return normalize_rows(W)

