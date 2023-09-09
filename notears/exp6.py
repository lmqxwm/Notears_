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

# d (int): num of nodes
# s0 (int): expected num of edges
# graph_type (str): ER, SF, BP
# n (int): num of samples, n=inf mimics population risk
# sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson


#def estimate_once(n, d, s0, graph_type, sem_type, seed=1, lambda1=0, loss_type='l2', B=100):
def estimate_once(d, graph_type, sem_type, seed=1, lambda1=0, loss_type='l2', BB=100):
    print("=============processing seed=", seed, "================")
    #utils.set_random_seed(seed)
    n = d-1
    s0 = n
    print("n, d, s0, graph_type, sem_type, lambda1, loss_type", "-", n, d, s0, graph_type, sem_type, lambda1, loss_type)
    
    B_true = utils.simulate_dag(d, s0, graph_type)
    W_true = utils.simulate_parameter(B_true)
    X = utils.simulate_linear_sem(W_true, n, sem_type, noise_scale=np.random.uniform(1, 5, d))

    B = np.min([math.factorial(d), BB])
    results = np.zeros([5, B+2])
    results[0, 0] = 0.5 / X.shape[0] * ((X-X@W_true) ** 2).sum()
    #all_perm = list(permutations(range(X.shape[1])))
    #utils.set_random_seed(seed+d)
    inds = [i for i in range(d)]
    results[:, 1] = linear.notears_linear(X, W_true=W_true, lambda1=0, loss_type=loss_type)
    for i in range(B):
         random.shuffle(inds)
         results[:, i+2] = linear.notears_linear(X[:, inds], W_true=W_true, lambda1=0, loss_type="l2")

    results = pd.DataFrame(results.T, columns=[
    "loss_est", "loss_l1", "obj_aug", "obj_dual", "h"])
    
    results.to_csv("./results_loss2/result_"+str(seed)+"_"+sem_type+"_"+str(d)+".csv", index=False)

def estimate_template(d, sem_type, seed=1, lambda1=0, loss_type='l2', BB=100):
    print("=============processing seed=", seed, "================")
    #utils.set_random_seed(seed)
    n = d-1
    s0 = n
    print("n, d, s0, em_type, lambda1, loss_type", "-", n, d, s0, sem_type, lambda1, loss_type)
    
    B_true = np.zeros([12, 12])
    for i in range(12):
        if (i+1) % 3 == 0:
            B_true[i-1, i] = 1
            B_true[i-2, i] = 1
    W_true = utils.simulate_parameter(B_true)
    X = utils.simulate_linear_sem(W_true, n, sem_type, noise_scale=np.random.uniform(1, 5, d))

    B = np.min([math.factorial(d), BB])
    results = np.zeros([5, B+2])
    results[0, 0] = 0.5 / X.shape[0] * ((X-X@W_true) ** 2).sum()
    #all_perm = list(permutations(range(X.shape[1])))
    #utils.set_random_seed(seed+d)
    inds = [i for i in range(d)]
    results[:, 1] = linear.notears_linear(X, W_true=W_true, lambda1=0, loss_type=loss_type)
    for i in range(B):
         random.shuffle(inds)
         results[:, i+2] = linear.notears_linear(X[:, inds], W_true=W_true, lambda1=0, loss_type="l2")

    results = pd.DataFrame(results.T, columns=[
    "loss_est", "loss_l1", "obj_aug", "obj_dual", "h"])
    
    results.to_csv("./results_loss2/result_"+str(seed)+"_"+sem_type+"_"+str(d)+".csv", index=False)

if __name__ == '__main__':
    
    semtypes = ["gauss", "gumbel", "uniform"]
    ds = [5, 10, 15, 20, 25, 30]
    pool = mp.Pool(processes=6)
    with pool:
        for sm in range(len(semtypes)):
            pool.map(partial(estimate_once, 
              graph_type="ER", seed=5010, sem_type=semtypes[sm], lambda1=0), 
              ds)
        pool.map(partial(estimate_template, 
              d=12, seed=5011, lambda1=0), 
              semtypes)
    pool.close()