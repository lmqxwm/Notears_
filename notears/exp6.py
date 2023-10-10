import numpy as np
import utils
import linear
import pandas as pd
import scipy
import math
import scipy
import multiprocessing as mp
import time
from itertools import permutations
import random
from functools import partial
import linear2
from golem import golem
import tensorflow as tf 

# d (int): num of nodes
# s0 (int): expected num of edges
# graph_type (str): ER, SF, BP
# n (int): num of samples, n=inf mimics population risk
# sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson


def restore_from_per(per_ind, W):
    W_re = np.zeros(W.shape)
    for i in range(len(per_ind)):
        for j in range(len(per_ind)):
            W_re[i, j] = W[per_ind[i], per_ind[j]]
    return W_re

def estimate_once_notears(d, graph_type, sem_type, seed=1, lambda1=0, loss_type='l2', BB=100, noise=None):
    print("=============processing seed=", seed, "================")
    #utils.set_random_seed(seed)
    n = d-1
    s0 = n
    print("n, d, s0, graph_type, sem_type, lambda1, loss_type", "-", n, d, s0, graph_type, sem_type, lambda1, loss_type)
    
    B_true = utils.simulate_dag(d, s0, graph_type)
    W_true = utils.simulate_parameter(B_true)
    B = np.min([math.factorial(d), BB])
    results = np.zeros([6, B+2])

    if noise == "normal":
        noise_scale = np.abs(np.random.normal(1, 1, d))
    elif noise == "uni":
        noise_scale = np.random.uniform(0.5, 2, d)
    else:
        noise_scale = None
    
    X = utils.simulate_linear_sem(W_true, n, sem_type, noise_scale=noise_scale)

    M = X @ W_true
    if loss_type == 'l2':
        R = X - M
        results[0, 0] = 0.5 / X.shape[0] * (R ** 2).sum()
    elif loss_type == 'logistic':
        results[0, 0] = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
    elif loss_type == 'poisson':
        S = np.exp(M)
        results[0, 0] = 1.0 / X.shape[0] * (S - X * M).sum()
    else:
        raise ValueError('unknown loss type')
    
    #all_perm = list(permutations(range(X.shape[1])))
    #utils.set_random_seed(seed+d)
    inds = [i for i in range(d)]
    w_est, results[:5, 1] = linear.notears_linear(X, W_true=W_true, lambda1=0, loss_type=loss_type)
    results[5, 1] = np.linalg.norm(w_est-W_true, ord='fro')
    for i in range(B):
        random.shuffle(inds)
        w_est, results[:5, i+2] = linear.notears_linear(X[:, inds], W_true=W_true, lambda1=0, loss_type="l2")
        w_est_re = restore_from_per(inds, w_est)
        results[5, 1] = np.linalg.norm(w_est_re-W_true, ord='fro')
        
    results = pd.DataFrame(results.T, columns=[
    "loss_est", "loss_l1", "obj_aug", "obj_dual", "h", "fro_dist"])
    
    results.to_csv("./results_loss2/result_"+str(seed)+"_"+sem_type+"_"+str(d)+".csv", index=False)


def estimate_template_dagma(sem_type, seed=1, lambda1=0, loss_type='l2', BB=100):
    print("=============processing seed=", seed, "================")
    d = 12
    #utils.set_random_seed(seed)
    n = 2 * d
    s0 = d-1
    print("n, d, s0, em_type, lambda1, loss_type", "-", n, d, s0, sem_type, lambda1, loss_type)
    
    B_true = np.zeros([12, 12])
    for i in range(12):
        if (i+1) % 3 == 0:
            B_true[i-1, i] = 1
            B_true[i-2, i] = 1
    W_true = utils.simulate_parameter(B_true)
    X = utils.simulate_linear_sem(W_true, n, sem_type, noise_scale=np.abs(np.random.normal(1, 1, d)))
    model = linear2.DagmaLinear(loss_type=loss_type)

    # B = np.min([math.factorial(d), BB])
    B = 1296
    results = np.zeros([5, B+2])
    results[0, 0] = 0.5 / X.shape[0] * ((X-X@W_true) ** 2).sum()
    inds = [i for i in range(d)]
    #results[:, 1] = linear.notears_linear(X, W_true=W_true, lambda1=0, loss_type=loss_type)
    results[:, 1] = model.fit(X, lambda1=0)
    count = 0
    for a1 in permutations([0,1,2]):
        for a2 in permutations([3,4,5]):
            for a3 in permutations([6,7,8]):
                for a4 in permutations([9,10,11]):
                    count += 1
                    new_ind = list(a1) + list(a2) + list(a3) + list(a4)
                    results[:, count+1] = model.fit(X[:, new_ind], lambda1=0)

    results = pd.DataFrame(results.T, columns=[
    "loss_est", "loss_l1", "obj_aug", "obj_dual", "h"])
    
    results.to_csv("./results_loss2/result_"+str(seed)+"_"+sem_type+"_"+str(d)+".csv", index=False)

def estimate_once_dagma(d, graph_type, sem_type, seed=1, lambda1=0, loss_type='l2', BB=100):
    print("=============processing seed=", seed, "================")
    #utils.set_random_seed(seed)
    n = d-1
    s0 = n
    print("n, d, s0, graph_type, sem_type, lambda1, loss_type", "-", n, d, s0, graph_type, sem_type, lambda1, loss_type)
    
    B_true = utils.simulate_dag(d, s0, graph_type)
    W_true = utils.simulate_parameter(B_true)
    X = utils.simulate_linear_sem(W_true, n, sem_type, noise_scale=np.random.uniform(0.5, 3, d))
    
    model = linear2.DagmaLinear(loss_type=loss_type)
    

    B = np.min([math.factorial(d), BB])
    results = np.zeros(B+2)

    M = X @ W_true
    if loss_type == 'l2':
        R = X - M
        results[0] = 0.5 / X.shape[0] * (R ** 2).sum()
    elif loss_type == 'logistic':
        results[0] = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
    elif loss_type == 'poisson':
        S = np.exp(M)
        results[0] = 1.0 / X.shape[0] * (S - X * M).sum()
    else:
        raise ValueError('unknown loss type')

    #all_perm = list(permutations(range(X.shape[1])))
    #utils.set_random_seed(seed+d)
    inds = [i for i in range(d)]
    results[1] = model.fit(X, lambda1=0)
    for i in range(B):
         random.shuffle(inds)
         results[i+2] = model.fit(X[:, inds], lambda1=0)

    results = pd.DataFrame(results.T, columns=[
    "loss_est"])
    
    results.to_csv("./results_loss2/result_"+str(seed)+"_"+sem_type+"_"+str(d)+".csv", index=False)

def estimate_once_golem(d, graph_type, sem_type, seed=1, lambda1=0, loss_type='l2', BB=100, noise=None):
    print("=============processing seed=", seed, "================")
    #utils.set_random_seed(seed)
    n = d-1
    s0 = n
    print("n, d, s0, graph_type, sem_type, lambda1, loss_type", "-", n, d, s0, graph_type, sem_type, lambda1, loss_type)
    
    B_true = utils.simulate_dag(d, s0, graph_type)
    W_true = utils.simulate_parameter(B_true)
    B = np.min([math.factorial(d), BB])
    results = np.zeros(B+2)

    if noise == "normal":
        noise_scale = np.abs(np.random.normal(1, 1, d))
    elif noise == "uni":
        noise_scale = np.random.uniform(1, 2, d)
    else:
        noise_scale = None
    
    X = utils.simulate_linear_sem(W_true, n, sem_type, noise_scale=noise_scale)

    M = X @ W_true
    if loss_type == 'l2':
        R = X - M
        results[0] = 0.5 * np.sum(np.log(np.sum(R**2, axis=0))) - np.log(np.abs(np.linalg.det(np.eye(d)-W_true)))
    elif loss_type == 'poisson':
        S = np.exp(M)
        results[0] = 1.0 / X.shape[0] * (S - X * M).sum()
    else:
        raise ValueError('unknown loss type')
    
    #all_perm = list(permutations(range(X.shape[1])))
    #utils.set_random_seed(seed+d)
    inds = [i for i in range(d)]
    results[1] = golem(X, 0, 5, False, seed=seed)
    for i in range(B):
         random.shuffle(inds)
         results[i+2] = golem(X[:, inds], 0, 5, False, seed=seed)

    results = pd.DataFrame(results.T, columns=[
    "loss_est"])
    
    results.to_csv("./results_loss2/result_"+str(seed)+"_"+sem_type+"_"+str(d)+".csv", index=False)


if __name__ == '__main__':
    
    semtypes = ["gauss", "gumbel", "uniform"]
    ds = [5, 10, 15, 20, 25, 30]
    pool = mp.Pool(processes=6)
    with pool:
        # pool.map(partial(estimate_template, 
        #       seed=5037, lambda1=0), 
        #       semtypes)

        for sm in range(len(semtypes)):
            pool.map(partial(estimate_once_golem, 
              graph_type="ER", seed=5040, sem_type=semtypes[sm], lambda1=0, noise="normal"), 
              ds)
            pool.map(partial(estimate_once_golem, 
              graph_type="ER", seed=5041, sem_type=semtypes[sm], lambda1=0, noise="uni"), 
              ds)
        #     pool.map(partial(estimate_once, 
        #       graph_type="BP", seed=5026, sem_type=semtypes[sm], lambda1=0, noise="normal"),
        #       ds)
        #     pool.map(partial(estimate_once, 
        #       graph_type="ER", seed=5029, sem_type=semtypes[sm], lambda1=0, noise="uni"), 
        #       ds)
            # pool.map(partial(estimate_once_dagma, 
            #   graph_type="ER", seed=5023, sem_type=semtypes[sm], lambda1=0), 
            #   ds)
        # ds = [5, 10, 15, 20]
        # pool.map(partial(estimate_once, 
        #       graph_type="ER", sem_type="logistic", seed=5028, lambda1=0, loss_type='logistic'),
        #       ds)
        # pool.map(partial(estimate_once, 
        #       graph_type="ER", sem_type="poisson", seed=5028, lambda1=0, loss_type='poisson'),
        #       ds)
        # pool.map(partial(estimate_once_dagma, 
        #       graph_type="ER", sem_type="logistic", seed=5030, lambda1=0, loss_type='logistic'),
        #       ds)
        # pool.map(partial(estimate_once_dagma, 
        #       graph_type="ER", sem_type="poisson", seed=5030, lambda1=0, loss_type='poisson'),
        #       ds)
    pool.close()