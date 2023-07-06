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


def estimate_once(n, d, s0, graph_type, sem_type):
    B_true = utils.simulate_dag(d, s0, graph_type)
    W_true = utils.simulate_parameter(B_true)
    X = utils.simulate_linear_sem(W_true, n, sem_type)
    W_est_notears = linear.notears_linear(X, lambda1=0.1, loss_type='l2')
    r = utils.count_accuracy(B_true, W_est_notears!=0)
    r6 = scipy.spatial.distance.hamming((W_est_notears!=0).reshape([-1,1]), B_true.reshape([-1,1]))
    return r['fdr'], r['tpr'], r['fpr'], r['shd'], r['nnz'], r6
    #print("n, d, s0, graph_type, sem_type", "\n", n, d, s0, graph_type, sem_type, "\n",  utils.count_accuracy(B_true, W_est_notears!=0))

def estimate_one_seed(seed=1):
    print("====================================")
    print("processing seed=", seed, "================")
    utils.set_random_seed(seed)
    ds = [5, 10, 20, 40]
    graphtypes = ["ER", "SF", "BP"]
    semtypes = ["gauss", "exp", "gumbel", "uniform", "logistic", "poisson"]
    result = np.zeros([len(ds)*5*len(semtypes), 11])
    result = pd.DataFrame(result, columns=["n", "d", "s0", "graph_type", "sem_type", "fdr", 
                    "tpr", "fpr", "shd", "nnz", "hamming"])
    #t1 = time.time()
    for g in range(len(graphtypes)):
        count = 0
        for dd in range(len(ds)):
            #s0s = [1, ds[dd], 2*ds[dd], int(math.comb(ds[dd], 2)/2)]
            if ds[dd] > 7 and graphtypes[g] != "BP":
                s0s = [1, int(ds[dd]/2), ds[dd]-1, 2*ds[dd], 3*ds[dd]]
            if graphtypes[g] == "BP":
                s0s = [1, int(ds[dd]/2), ds[dd]-1]
            if ds[dd] <= 7 and graphtypes[g] == "SF":
                s0s = [1, int(ds[dd]/2), ds[dd]-1, 2*ds[dd], 3*ds[dd]]
            if ds[dd] <= 7 and graphtypes[g] == "ER":
                s0s = [1, int(ds[dd]/2), ds[dd]-1, 2*ds[dd]]
            #s0s = [int(math.comb(ds[dd], 2)/2)]
            for s in range(len(s0s)):
                for t in range(len(semtypes)):
                    graph_type = graphtypes[g]
                    n = 100
                    d = ds[dd]
                    s0 = s0s[s]
                    sem_type = semtypes[t]
                    result.iloc[count, 0:5] = n, d, s0, graph_type, sem_type
                    result.iloc[count, 5:11] = estimate_once(n, d, s0, graph_type, sem_type)
                    count += 1
                    print( "n, d, s0, graph_type, sem_type", "-", n, d, s0, graph_type, sem_type)
                    result.to_csv("./results/result_"+graph_type+"_"+str(seed)+".csv", index=False)
                    #t2 = time.time()
                    #print("processed", t2-t1, "s")

if __name__ == '__main__':

    pool = mp.Pool(processes=6)
    with pool:
        pool.map(estimate_one_seed, [i+600 for i in range(20)])
    pool.close()
            