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
    return linear.notears_linear(X, W_true=W_true, lambda1=lambda1, loss_type=loss_type)
    #print("n, d, s0, graph_type, sem_type", "\n", n, d, s0, graph_type, sem_type, "\n",  utils.count_accuracy(B_true, W_est_notears!=0))

def estimate_one_seed(seed=200):
    print("=============processing seed=", seed, "================")
    utils.set_random_seed(seed)
    ds = np.arange(2, 10+1, dtype=int)
    #graphtypes = ["ER", "SF", "BP"]
    semtypes = ["gauss", "gumbel", "uniform"]
    result = np.zeros([len(ds), 22])
    result = pd.DataFrame(result, columns=["n", "d", "s0", "graph_type", "sem_type", "lambda1", "loss_type", \
    "loss_est", "loss_l1", "obj_aug", "obj_dual", "h",
    "loss_est_2", "loss_l1_2", "obj_aug_", "obj_dual_2", "h_2",
    "loss_est_t", "loss_l1_t", "obj_aug_t", "obj_dual_t", "h_t",])
    
    for ss in range(len(semtypes)):
        count = 0
        sem_type = semtypes[ss]
        for dd in range(len(ds)):
            graph_type = "ER"
            d = ds[dd]
            n = d-1
            s0 = d-1
            lambda1 = 0
            loss_type = "l2"

            result.iloc[count, 0:7] = n, d, s0, graph_type, sem_type, lambda1, loss_type
            result.iloc[count, 7:22] = estimate_once(n, d, s0, graph_type, sem_type, lambda1, loss_type)
            count += 1
            print( "n, d, s0, graph_type, sem_type, lambda1, loss_type", "-", n, d, s0, graph_type, sem_type, lambda1, loss_type)
            result.to_csv("./results_loss/result_"+str(seed)+"_"+sem_type+".csv", index=False)


if __name__ == '__main__':
    
    #semtypes = ["gauss", "gumbel", "uniform"]
    pool = mp.Pool(processes=6)
    with pool:
        pool.map(estimate_one_seed, [i+201 for i in range(9)])
    pool.close()


