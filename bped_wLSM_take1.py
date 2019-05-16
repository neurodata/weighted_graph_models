#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os

from mpl_toolkits.mplot3d import Axes3D

from graspy.plot import *
from graspy.simulations import sample_edges
from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed
from graspy.utils import *

try:
    os.chdir(os.path.join(os.getcwd(), "weighted_graph_models"))
    print(os.getcwd())
except:
    pass

from wLSM_utils import *


get_ipython().run_line_magic("matplotlib", "inline")

#%%
############################
pois_range = np.array(range(1, 40, 5))
hw_scale_range = 0.5 / (pois_range)

pois_range = [5]
hw_scale_range = [1]  # [0.5]
p = pois_range[0]
h = hw_scale_range[0]

n_hw_nodes = 1000
n_modified_verts = 100
n_blob1_verts = 100
n_blob2_verts = 100
n_blob3_verts = 100
pois_scale0 = p
pois_scale1 = p * 10
hw_scale = h
n_components = 7
acorn = np.random.seed(8888)
############################

X = sample_hw_latent(
    n_hw_nodes, n_modified_verts, pois_scale0, pois_scale1, acorn=acorn
)

# could turn this on to add some sbm masses
# mu1 = np.array([0.2, 0.05, 0.05])
# mu2 = np.array([0.05, 0.2, 0.05])
# mu3 = np.array([0.05, 0.05, 0.2])
# X = np.concatenate((X, np.tile(mu1, (n_blob1_verts, 1))))
# X = np.concatenate((X, np.tile(mu2, (n_blob2_verts, 1))))
# X = np.concatenate((X, np.tile(mu3, (n_blob3_verts, 1))))

n_verts = X.shape[0]

P = hw_scale * X @ X.T

graph_uw = sample_edges(P, directed=False, loops=False)
print(np.mean(graph_uw))
verts = np.array(range(n_verts))
verts_mod = np.random.choice(range(n_hw_nodes), n_modified_verts, replace=False)

lambda_mat = X @ X.T * pois_scale0
lambda_mat[np.ix_(verts_mod, verts_mod)] = P[np.ix_(verts_mod, verts_mod)] * pois_scale1

graph_w = np.random.poisson(lambda_mat)
graph_w = symmetrize(graph_w)
graph_w = np.multiply(graph_w, graph_uw)
heatmap(graph_w, transform="log")

ase = AdjacencySpectralEmbed(n_components=n_components)

# different options for regularizing
graph_embed = graph_w
# graph_embed = graph_embed / np.sum(graph_embed, axis=0)[:,np.newaxis]
# graph_embed = graph_w
# graph_embed = augment_diagonal(graph_embed)
# graph_embed += 1 / graph_embed.size
# graph_embed = symmetrize(graph_embed)
lam_mat = concentration_regularize(graph_w, sum_edges=True, weight=10)
# graph_embed = graph_w * lam_mat
heatmap(graph_embed)
# graph_embed = pass_to_ranks(graph_embed)
Xhat = ase.fit_transform(graph_embed)

labels = np.isin(verts, verts_mod)
# pairplot(X, labels=labels, legend_name="Modified")
pairplot(Xhat, labels=labels, legend_name="Modified")
screeplot(graph_embed, cumulative=False, show_first=20)

# expectation = np.sum(hw_scale ** 2 * (X @ X.T) ** 2 * pois_scale0)
expectation = P * lambda_mat
expectation = expectation.sum()
print(expectation)
print(graph_w.sum())

# lse_embed = LaplacianSpectralEmbed(form="R-DAD", regularizer=10).fit_transform(graph_w)
# pairplot(lse_embed, labels=labels)

# pairplot(X)
#%%
latent = AdjacencySpectralEmbed().fit_transform(lambda_mat)
pairplot(latent, labels=labels, legend_name="Modified")

l2 = X.copy()
l2 *= np.sqrt(p)
l2[verts_mod] = 10 * np.sqrt(p) * X[verts_mod]
pairplot(l2, labels=labels, legend_name="Modified")

#%%
from scipy.linalg import orthogonal_procrustes

# Do not perturb anything
n_verts = 1000
pois_scale0 = 5
X = sample_hw_latent(n_verts)
Y = np.sqrt(pois_scale0) * X
P = X @ X.T
Y[verts_mod] = (
    np.sqrt(pois_scale1) * X[verts_mod]
)  # if we do this we get perfect recovery
lambda_mat = pois_scale0 * P
lambda_mat[np.ix_(verts_mod, verts_mod)] = pois_scale1 * P[np.ix_(verts_mod, verts_mod)]
# lambda_mat = Y @ Y.T
latent = AdjacencySpectralEmbed(n_components=3).fit_transform(lambda_mat)
# pairplot(latent)
R, scale = orthogonal_procrustes(latent, Y)
latent = latent @ R
pl = np.concatenate((latent, Y), axis=0)
labels = np.array(n_verts * ["estimated"] + n_verts * ["true"])
pairplot(pl, labels)

#%%

