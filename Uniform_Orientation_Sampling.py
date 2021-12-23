import numpy as np
from tqdm import tqdm
from utils import normalize_rows, Pair_separation, Bubble_tea_potential, KNN_repulsion
from Plotting_tools import NDScatter
from Evaluations import KNNDistanceDistribution


class UniformOrientationSampling:
    """Main class in charge of  specifying the parameters and running the optimization program.

    Parameters
    ------------
    dimensions: int
            dimensionality of the embedding/ambient space. Dimensions above 20 haven't been tested.

    pop_size: int
            Number of points (ie orientations) to learn.

    iterations: int
            Number of passes over the set of pop_size points. Each iteration updates all positions once.

    approach: str
            algorithm to use for updating positions. 'KNN_repulsion' is the only one that has been shown to work reliably

    """

    def __init__(self, dimensions=3, pop_size=20, iterations=1000, approach='KNN_repulsion'):
        self.dimensions = dimensions
        self.pop_size = pop_size
        self.iterations = iterations
        self.approach = approach

    def initialize_emb(self):
        dat = np.random.rand(self.pop_size, self.dimensions) - 0.5
        return normalize_rows(dat)

    def update_emb(self, V):
        # calls a given updater, which typically cycles over all points, updating each once
        if self.approach == "pair_separation":
            updater = Pair_separation(0.05)
            return updater.update_points(V)

        elif self.approach == "bubble_tea_potential":
            updater = Bubble_tea_potential(0.02)
            return updater.update_points(V)

        elif self.approach == "KNN_repulsion":
            updater = KNN_repulsion(0.01, K=1)
            return updater.update_points(V)

    def run_optimizer(self):
        embedding = self.initialize_emb()
        # emb_list = []

        for t in tqdm(range(self.iterations)):
            embedding = self.update_emb(embedding)
            # emb_list.append(embedding)

        return embedding


UOS = UniformOrientationSampling(dimensions=10, pop_size=500, iterations=1500, approach='KNN_repulsion')

embedding = UOS.run_optimizer()

plotter = NDScatter(embedding, UOS, make_raster=True)
plotter.ndscatter()

KNNDD = KNNDistanceDistribution(embedding, UOS, show_naive=True)
KNNDD.get_KNNDD()
