#!/usr/bin/env python
# coding: utf-8

import numpy as np
from wLSM_utils import *
from graspy.simulations import sbm
from graspy.embed import AdjacencySpectralEmbed as ASE
from graspy.cluster import GaussianCluster as GCLUST
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import adjusted_rand_score as ari
from scipy.spatial import Delaunay

from itertools import combinations
from scipy.stats import norm
import scipy.optimize as optimize

from joblib import Parallel, delayed

def generate_cyclops(X, n, pi, density=None, density_params=[0,1], acorn=None):
    if acorn is None:
        acorn = np.random.randint(10**6)
    np.random.seed(acorn)
    
    counts = np.random.multinomial(n, [pi, 1 - pi]).astype(int)
    
    if density is None:
        density = np.random.uniform
        U = sample(counts[0], density, density_params)
        X_L = get_latent_positions(U)
    else:
#         U = sample(counts[0], density, density_params)
        density_params = np.array(density_params)
        d = len(density_params)
        if density_params.ndim == 1:
            pass
        else:
            X_temp = np.stack([sample(counts[0], density, density_params[i]) for i in range(d)], axis=1)
            quad = np.sum(np.array([3, 3])*X_temp[:,:2]**2, axis=1)[:, np.newaxis]
            print(quad, X_temp[0, 0], X_temp[0, 1], X_temp[0, 0]**2 + X_temp[0, 1]**2)
            X_L = np.concatenate((X_temp[:,:2], quad), axis=1)
        
    X = X[:, np.newaxis].T
    
    All_X = np.concatenate((X_L, X), axis = 0)
    
    P = All_X @ All_X.T
    
    A = sbm(np.concatenate((np.ones(counts[0]).astype(int), [counts[1]])), P)
    
    return A, counts

def quadratic(data, params):
    if data.ndim == 1:
        sum_ = np.sum(data[:-1]**2 * params[:-1]) + params[-1]
        return sum_
    elif data.ndim == 2:
        sums = np.sum(data[:, :-1]**2 * params[:-1], axis = 1) + params[-1]
        return sums
    else:
        raise ValueError("unsuppored data")
        
def quadratic_log_likelihood(data, params, curve_density=False):
    n, d = data.shape
    fitted_Z = quadratic(data, params)
    residuals = fitted_Z - data[:, -1] # assuming data.ndim == 2
    std = np.std(residuals, ddof=1)
    
    log_likelihood = 0
    for i in range(n):
        log_likelihood += np.log(norm.pdf(residuals[i], fitted_Z[i], std))
        
    return log_likelihood

def func(data, a, b, c):
    if data.ndim == 1:
        return a*(data[0])**2 + b*data[1]**2 + c
    return a*(data[:, 0])**2 + b*data[:, 1]**2 + c

def monte_carlo_integration(data, func, params, M, acorn=None):
    if acorn is None:
        acorn = np.random.randint(10**6)
    np.random.seed(acorn)
    
    n, d = data.shape
    
    maxes = np.array([max(data[:, i]) for i in range(d-1)])
    mins = np.array([min(data[:, i]) for i in range(d-1)])
    area = np.prod(maxes - mins)
    
    sample = np.zeros((M, d-1))
    
    for coord in range(d-1):
        sample[:, coord] = np.random.uniform(mins[coord], maxes[coord])
        
    sum_f = 0
    for it in range(M):
        sum_f += func(sample[it, :], params[0], params[1], params[2])
        
    estimated_integral = (area)*(1/M)*sum_f
    
    return estimated_integral

def for_loop_function(combo, X_hat, est_labels, true_labels, gclust_model, M):
    print(combo)
    n, d = X_hat.shape
    unique_labels, counts = np.unique(est_labels, return_counts=True)
    K = len(unique_labels)

    class_idx = np.array([np.where(c_hat == u)[0] for u in unique_labels])
    temp_quad_labels = np.concatenate(class_idx[combo])

    surface_count = np.sum(counts[temp_quad_labels])
    surface_prop = surface_count/n

    temp_n = len(temp_quad_labels)
    temp_K = K - len(combo)
    temp_mean_params = temp_K * d
    temp_cov_params = temp_K * d * (d + 1) / 2
    temp_quad_params = (d - 1)*2 + d - 1 + (d - 1) * (d - 2) / 2 + 1
    temp_n_params = temp_mean_params + temp_cov_params
    temp_n_params = temp_quad_params + temp_K - 1
    
    temp_label = min(combo)
    new_counts = np.zeros(temp_K)
    for i in range(new_counts):
        if unique_labels[i] == temp_label:
            new_counts = surface_count
        else:
            new_counts = counts[i]

    new_props = (surface_count / n)**counts
    prop_log_likelihoods = np.sum(np.log(new_props))

    temp_c_hat = c_hat.copy()
    temp_c_hat[temp_quad_labels] = temp_label
    
    params, pcov = optimize.curve_fit(func, X_hat[temp_quad_labels, :2], X_hat[temp_quad_labels, 2])
    
    # integral = abs(monte_carlo_integration(X_hat[temp_quad_labels], func, params, M))
    delly = Delaunay(X_hat[temp_quad_labels ,:-1])
    content = np.sum([calculate_simplex_content(X_hat[temp_quad_labels][del_]) for del_ in delly.simplices])
    
    quad_log_likelihood = quadratic_log_likelihood(X_hat[temp_quad_labels], params, curve_density=False)
    quad_log_likelihood -= temp_n * np.log(content)
    gmm_log_likelihood = np.sum(gclust.model_.score_samples(X_hat[-temp_quad_labels]))

    log_likeli = quad_log_likelihood + gmm_log_likelihood + prop_log_likelihoods
    
    bic_ = 2*log_likeli - temp_n_params * np.log(n)

    ari_ = ari(c, temp_c_hat)
    
    print(likeli, ari_, bic_)
    return [combo, likeli, ari_, bic_]

X = np.array([0.2, 0.2, 0.2])

n = 1000
pi = 0.9

A, counts = generate_cyclops(X, n, pi, None)
c = [0]*counts[0]
c += [1]*counts[1]

true_labels = c

ase = ASE(n_components=3)
X_hat = ase.fit_transform(A)

gclust_model = GCLUST(max_components=8)
est_labels = gclust.fit_predict(X_hat)

loglikelihoods = [np.sum(gclust.model_.score_samples(X_hat))]
combos = [None]
aris = [ari(c, est_labels)]
bic = [gclust.model_.bic(X_hat)]

unique_labels = np.unique(est_labels)

class_idx = np.array([np.where(est_labels == u)[0] for u in unique_labels])

for k in range(len(unique_labels)):
    for combo in list(combinations(np.unique(est_labels), k+1)):
        combo = np.array(list(combo)).astype(int)
        combos.append(combo)

M = 10**8

condensed_func = lambda combo : for_loop_function(combo, X_hat, est_labels, true_labels, gclust_model, M)
results = Parallel(n_jobs=15)(delayed(condensed_func)(combo) for combo in combos[1:])

new_combos = [None]
for i in range(len(combos[1:])):
    new_combos.append(results[i][0])
    loglikelihoods.append(results[i][1])
    aris.append(results[i][2])
    bic.append(results[i][3])

import _pickle as pickle
pickle.dump(true_labels, open('true_labels_cyclops_par.pkl', 'wb'))
pickle.dump(est_labels, open('est_labels_cyclops_par.pkl', 'wb'))
pickle.dump(class_idx, open('class_idx_cyclops_par.pkl', 'wb'))
pickle.dump(bic, open('bic_cyclops_par.pkl', 'wb'))
pickle.dump(aris, open('aris_cyclops_par.pkl', 'wb'))
pickle.dump(loglikelihoods, open('loglikelihoods_cyclops_par.pkl', 'wb'))
pickle.dump(new_combos, open('combos_cyclops_par.pkl', 'wb'))



# K = len(unique_labels)
# unique_labels = np.unique(c_hat)

# M = 10**8

# class_idx = np.array([np.where(c_hat == u)[0] for u in unique_labels])

# loglikelihoods = [np.sum(gclust.model_.score_samples(X_hat))]
# combos = [None]
# aris = [ari(c, c_hat)]
# bic = [gclust.model_.bic(X_hat)]

# n, d = X_hat.shape

# for k in range(len(unique_labels)):
#     for combo in list(combinations(np.unique(c_hat), k+1)):
#         combo = np.array(list(combo)).astype(int)
#         combos.append(combo)
        
#         print(combo)
        
#         temp_quad_labels = np.concatenate(class_idx[combo])
#         temp_n = len(temp_quad_labels)
#         temp_K = K - len(combo)
#         temp_mean_params = temp_K * d
#         temp_cov_params = temp_K * d * (d + 1) / 2
#         temp_quad_params = (d - 1)*2 + d - 1 + (d - 1) * (d - 2) / 2 + 1
#         temp_n_params = temp_mean_params + temp_cov_params
#         temp_n_params = temp_quad_params + temp_K - 1
        
#         temp_label = min(combo)
#         temp_c_hat = c_hat.copy()
#         temp_c_hat[temp_quad_labels] = temp_label
        
#         params, pcov = optimize.curve_fit(func, X_hat[temp_quad_labels, :2], X_hat[temp_quad_labels, 2])
        
#         integral = monte_carlo_integration(X_hat[temp_quad_labels], func, params, M)
#         temp_density = 1/integral
        
#         quad_log_likelihood = quadratic_log_likelihood(X_hat[temp_quad_labels], params, curve_density=False)
#         quad_log_likelihood += temp_n * np.log(temp_density)
#         gmm_log_likelihood = np.sum(gclust.model_.score_samples(X_hat[-temp_quad_labels]))
        
#         loglikelihoods.append(quad_log_likelihood + gmm_log_likelihood)
#         aris.append(ari(c, temp_c_hat))
#         bic.append(2*gmm_log_likelihood*(n - temp_n) + 2*quad_log_likelihood*temp_n -
#                         temp_n_params * np.log(n))

# import _pickle as pickle
# pickle.dump(bic, open('bic_cyclops.pkl', 'wb'))
# pickle.dump(aris, open('aris_cyclops.pkl', 'wb'))
# pickle.dump(loglikelihoods, open('loglikelihoods_cyclops.pkl', 'wb'))
# pickle.dump(combos, open('combos_cyclops.pkl', 'wb'))


# def monte_carlo_integration(data, func, params, M, acorn=None):
#     if acorn is None:
#         acorn = np.random.randint(10**6)
#     np.random.seed(acorn)
    
#     n, d = data.shape
    
#     maxes = np.array([max(data[:, i]) for i in range(d-1)])
#     mins = np.array([min(data[:, i]) for i in range(d-1)])
#     area = np.prod(maxes - mins)
    
#     sample = np.zeros((M, d-1))
    
#     for coord in range(d-1):
#         sample[:, coord] = np.random.uniform(mins[coord], maxes[coord])
        
#     sum_f = 0
#     for it in range(M):
#         sum_f += func(sample[it, :], params[0], params[1], params[2])
        
#     estimated_integral = (area)*(1/M)*sum_f
    
#     return estimated_integral
