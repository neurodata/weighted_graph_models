from graspy.embed import AdjacencySpectralEmbed as ASE
from graspy.cluster import GaussianCluster as GCLUST
from graspy.simulations import sbm

import numpy as np
import matplotlib.pyplot as plt

from graspy.plot import pairplot

def hardy_weinberg(theta):
    """
    Maps a value from [0, 1] to the hardy weinberg curve.
    """
    return [theta**2, 2*theta*(1-theta), (1 - theta)**2]

def sample(shape, function, params):
    """
    A function to sample from a general numpy random module.
    """
    if np.array(params).ndim is 1:
        return function(*params, shape)
    elif np.array(params).ndim is 2:
        if params.shape == shape:
            samples = np.ones(shape)
            for i in range(shape[0]):
                for j in range(i+1, shape[0]):
                    samples[i,j] = function(params[i,j], 1)
                    samples[j,i] = samples[i,j]
            return samples

def get_latent_positions(sample):
    """
    Mapping from [0, 1]^{len(samples)} to { R^{3} }^{len(samples)}
    """
    return np.array(list(map(hardy_weinberg, sample)))

def wHardy_Weinberg(n, m, c0, c1, density=np.random.uniform, params=[0,1], acorn=None):
    """
    zero-inflated Z_+ weighted LSM network (model & methods)

    Let T_1, ... T_n ~iid density on [0,1].

    Let h: [0,1] to R^d and X_i = h(T_i) so that the latent positions lie on a one-dimensional curve in R^d.
    (start with h(t) = [t^2 , 2t(1-t) , (1-t)^2]^{T} so the X_i's are on Hardy-Weinberg in Delta^2 subset [0,1]^3 subset R^3.)
    (for sanity check: do ase(G) into R^3 for such an LSM, and you should get Xhat's around HW (up to orthogonal transformation).)
    Let p_ij = X_i^{T} X_j.
    Let B_ij ~ Bernoulli(p_ij) and Z_ij ~ G(p_ij) be independent -- independent of each other, and independent across ij.
    Let W_ij = Z_ij * I{B_ij}, so W_ij is 0-inflated -- 0 with probability 1 - p_ij and weighted otherwise.
    (start with G(p_ij) = Poisson(c0 * p_ij) so we have a 0-inflated Poisson LSM.) 
    that's H0.
    for HA:
    generate null G.
    choose m vertices uniformly at random -- S subset V = [n].
    let H be the induced subgraph Omega(S;G).
    let the edges W_ij for this induced subgraph to be of the form 
      W_ij is 0 with probability 1 - p_ij and is Poisson(c1 * p_ij) with probability p_ij.

    so ... for the "start with" case
    we have just four parameters: n, m, c, c'.
    """

    if acorn is None:
        acorn = np.random.randint(10**6)
    
    np.random.seed(acorn)
    V = range(n)
    V1 = np.random.choice(V, m, replace=False)

    t = sample(n, density, params)

    X = get_latent_positions(t)
    P = X @ X.T

    A0 = sbm(np.ones(n).astype(int), P)

    pois = np.random.poisson
    L0 = c0*np.ones((n,n))*P
    
    Z0 = sample((n,n), pois, L0)
    W0 = A0 * Z0

    L1 = c1*np.ones((n,n))*P
    A1 = sbm(np.ones(n).astype(int), P)

    Z1 = sample((n,n), pois, L0)
    transplant = sample((n,n), pois, L1)

    Z1[np.ix_(V1, V1)] = transplant[np.ix_(V1, V1)]

    W1 = A1 * Z1

    ase0 = ASE(n_components=min(100, n - 1))
    ase0.fit(W0)

    ase1 = ASE(n_components=min(100, n - 1))
    ase1.fit(W1)
    
    return ase0, ase1