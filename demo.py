from Uniform_Orientation_Sampling import UniformOrientationSampling
from Plotting_tools import NDScatter
from Evaluations import KNNDistanceDistribution

# Instantiate object with problem parameters
UOS = UniformOrientationSampling(dimensions=3, pop_size=30, iterations=1500, approach='KNN_repulsion')

# Optimize set of pop_size points
embedding = UOS.run_optimizer()

# Instantiate plotting tool
plotter = NDScatter(embedding, UOS, make_raster=True)

# run ndscatter() to automatically plot a result independent of dimensionality. For dimensions>3 it should simply
# plot the data matrix containing the end result. You could also use dimensionality reduction, but most embeddings will
# look just like a filled ball of points.
plotter.ndscatter()

# Compute and plot distribution of distances from a center point to nearest neighbors.
KNNDD = KNNDistanceDistribution(embedding, UOS, show_hist=True)
KNNDD.get_KNNDD()