#%%
import numpy as np
from scipy.spatial import Delaunay
from scipy.special import factorial
from scipy.spatial.distance import pdist, squareform
import pickle


fn = "/Users/bpedigo/JHU_code/DLMB/weighted_graph_models/X_hat.pkl"
xhat = pickle.load(open(fn, "rb"))

delaun = Delaunay(xhat)
simplices = delaun.simplices
simplex_verts = xhat[simplices]
# n_simplex x n_points/simplex x n_dims
simp = simplex_verts[0]
#%%
def calculate_simplex_volume(points):
    v0 = points[0]
    dist = points[1:] - v0[np.newaxis, :]
    out = np.linalg.det(dist)
    n = points.shape[1]
    out *= 1 / factorial(n)
    return np.abs(out)


def calculate_simplex_area(points):
    j = points.shape[0] - 1
    dist = squareform(pdist(points))
    dist = dist ** 2
    new_dist = np.ones((dist.shape[0] + 1, dist.shape[1] + 1))
    new_dist[1:, 1:] = dist
    new_dist[0, 0] = 0
    det = np.linalg.det(new_dist)
    det /= 2 ** j
    det /= factorial(j) ** 2
    return np.sqrt(np.abs(det))


# calculate_simplex_volume(simp)

p = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1]])
print(calculate_simplex_area(p))

simp
