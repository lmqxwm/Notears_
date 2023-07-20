import numpy as np
import utils
import linear
import pandas as pd
import scipy
import math
import scipy
import multiprocessing as mp
import time

# d (int): num of nodes
# s0 (int): expected num of edges
# graph_type (str): ER, SF, BP
# n (int): num of samples, n=inf mimics population risk
# sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson


def estimate_once(n, d, s0, graph_type, sem_type, lambda1=0.1, loss_type='l2'):
    B_true = utils.simulate_dag(d, s0, graph_type)
    W_true = utils.simulate_parameter(B_true)
    X = utils.simulate_linear_sem(W_true, n, sem_type)
    return linear.notears_linear(X, lambda1=lambda1, loss_type=loss_type)
    #print("n, d, s0, graph_type, sem_type", "\n", n, d, s0, graph_type, sem_type, "\n",  utils.count_accuracy(B_true, W_est_notears!=0))

def estimate_one_seed(seed=1):
    print("=============processing seed=", seed, "================")
    utils.set_random_seed(seed)
    ds = [10, 20, 40, 60, 80, 10, 10, 10]
    ns = [10, 10, 10, 10, 10, 100, 500, 1000]
    #graphtypes = ["ER", "SF", "BP"]
    semtypes = ["gauss", "exp", "gumbel", "uniform"]
    lambdas = [0]
    losstypes = ["l2"]
    result = np.zeros([len(ds)*len(ns)*len(lambdas)*len(semtypes)*len(losstypes), 12])
    result = pd.DataFrame(result, columns=["n", "d", "s0", "graph_type", "sem_type", "lambda1", "loss_type", "loss_est", "loss_l1", "obj_aug", "obj_dual", "h"])
    #t1 = time.time()
    count = 0
    
    for lam in range(len(lambdas)):
        for dd in range(len(ds)):
            s0s = [ds[dd]]
            #s0s = [1, ds[dd], 2*ds[dd], int(math.comb(ds[dd], 2)/2)]
            # if ds[dd] > 7 and graphtypes[g] != "BP":
            #     s0s = [1, int(ds[dd]/2), ds[dd]-1, 2*ds[dd], 3*ds[dd]]
            # if graphtypes[g] == "BP":
            #     s0s = [1, int(ds[dd]/2), ds[dd]-1]
            # if ds[dd] <= 7 and graphtypes[g] == "SF":
            #     s0s = [1, int(ds[dd]/2), ds[dd]-1, 2*ds[dd], 3*ds[dd]]
            # if ds[dd] <= 7 and graphtypes[g] == "ER":
            #     s0s = [1, int(ds[dd]/2), ds[dd]-1, 2*ds[dd]]
            #s0s = [int(math.comb(ds[dd], 2)/2)]
            for ss in range(len(s0s)):
                for t in range(len(semtypes)):
                    graph_type = "ER"
                    n = ns[dd]
                    d = ds[dd]
                    s0 = s0s[ss]
                    sem_type = semtypes[t]
                    lambda1 = lambdas[lam]
                    loss_type = losstypes[0]
                    result.iloc[count, 0:7] = n, d, s0, graph_type, sem_type, lambda1, loss_type
                    result.iloc[count, 7:12] = estimate_once(n, d, s0, graph_type, sem_type, lambda1, loss_type)
                    count += 1
                    print( "n, d, s0, graph_type, sem_type, lambda1, loss_type", "-", n, d, s0, graph_type, sem_type, lambda1, loss_type)
                    result.to_csv("./results_loss/result_"+str(seed)+".csv", index=False)
                    #t2 = time.time()
                    #print("processed", t2-t1, "s")

if __name__ == '__main__':

    pool = mp.Pool(processes=6)
    with pool:
        pool.map(estimate_one_seed, [i+9000 for i in range(20)])
    pool.close()


## 补充s0s d/2, d, 
## 尝试n=20/15，