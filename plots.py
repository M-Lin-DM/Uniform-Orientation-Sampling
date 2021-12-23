import numpy as np
from Uniform_Orientation_Sampling import UniformOrientationSampling
from Plotting_tools import NDScatter
from Evaluations import KNNDistanceDistribution

Pops = [int(j) for j in np.logspace(1, 5, num=5, endpoint=True, base=5)]
save_path_emb = 'C:/Users/mirbe/Documents/Experiments/ABM_OOP-based/figures/emb3D/'
save_path_naive = 'C:/Users/mirbe/Documents/Experiments/ABM_OOP-based/figures/emb3Dnaive/'
save_path_hist = 'C:/Users/mirbe/Documents/Experiments/ABM_OOP-based/figures/histograms/'

for s in Pops:

    UOS = UniformOrientationSampling(dimensions=3, pop_size=s, iterations=1500, approach='KNN_repulsion')
    embedding = UOS.run_optimizer()

    # for plotting optimized embedding
    plotter = NDScatter(embedding, UOS, make_raster=True, save_path=save_path_emb)
    plotter.scatter3D_mpl(plot_cap=True)

    KNNDD = KNNDistanceDistribution(embedding, UOS, save_path_naive=save_path_naive, save_path_histogram=save_path_hist)
    KNNDD.get_KNNDD()

    KNNDD.save_naive_emb()

