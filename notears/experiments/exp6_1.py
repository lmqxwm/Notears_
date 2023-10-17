import exp6
import numpy as np
import pandas as pd
import scipy
import math
import multiprocessing as mp
from functools import partial

if __name__ == '__main__':
    print("start!")
    exp6.estimate_once_notears(d=10, sem_type="gauss", graph_type="BP", seed=5055, lambda1=0, BB=500, noise="normal")
    semtypes = ["gauss", "gumbel", "uniform"]
    ds = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
    pool = mp.Pool(processes=10)
    with pool:
        # pool.map(partial(estimate_template, 
        #       seed=5037, lambda1=0), 
        #       semtypes)

        for sm in range(len(semtypes)):
            # pool.map(partial(estimate_once_golem, 
            #   graph_type="ER", seed=5040, sem_type=semtypes[sm], lambda1=0, noise="normal"), 
            #   ds)
            # pool.map(partial(estimate_once_golem, 
            #   graph_type="ER", seed=5041, sem_type=semtypes[sm], lambda1=0, noise="uni"), 
            #   ds)
            print("work!")
            pool.map(partial(exp6.estimate_once_notears, 
              graph_type="BP", seed=5055, sem_type=semtypes[sm], lambda1=0, BB=500, noise="normal"),
              ds)
            pool.map(partial(exp6.estimate_once_notears, 
              graph_type="BP", seed=5056, sem_type=semtypes[sm], lambda1=0, BB=500, noise="uni"), 
              ds)
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