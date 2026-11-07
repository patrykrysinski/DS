import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product,permutations
import random
np.seterr(divide='ignore') ## ignore divide by 0

T0 = 1000
temp_for_plot = T0
M=300
N=15
alpha =0.85
k=0.1

objective(3,2)
def objective(x,y):
    return (x**2 + y -11)**2 + (x+y**2-7)**2

def optimization(T0=1000, M=300, N=15, alpha=0.85, k=0.1):
    x = 2 #np.random.normal()
    y = 1 #np.random.normal()
    z = objective(x,y)
    z_list = []
    T_list= []
    T = T0
    m_help = 0
    for m,n in product(range(M),range(N)):
        if m != m_help:
            z_list.append(z)
            T_list.append(T)
            T = T*alpha
        x_temp = x + np.random.normal()*k
        y_temp = y + np.random.normal()*k
        z_temp = objective(x_temp,y_temp)
        if z_temp <= z:
           x = x_temp
           y = y_temp
           z = z_temp
        else:
            rand_number = np.random.random()
            exp_fun = 1 / (np.exp((z_temp - z) /T))
            if rand_number <= exp_fun:
                x = x_temp
                y = y_temp
                z = z_temp

        m_help = m
    return z_list, T_list,x,y
z, T,x,y = optimization(M=3000) 

plt.plot(T,z)

## simulated annealing combinatorial problem

X0 = ["B","D","A","E","C","F","G","H"]
random.shuffle(X0)
X0
list(permutations(X0))[0]
def optimization_combinatorial(T0=1000, M=300, N=15, alpha=0.85, k=0.1, solutions = ["A","B","C","D","E","F","G","H"]):
    
    Dist = pd.DataFrame([[0,1,2,3,1,2,3,4],[1,0,1,2,2,1,2,3],[2,1,0,1,3,2,1,2],
                      [3,2,1,0,4,3,2,1],[1,2,3,4,0,1,2,3],[2,1,2,3,1,0,1,2],
                      [3,2,1,2,2,1,0,1],[4,3,2,1,3,2,1,0]],
                    columns=solutions,
                    index=solutions)

    Flow = pd.DataFrame([[0,5,2,4,1,0,0,6],[5,0,3,0,2,2,2,0],[2,3,0,0,0,0,0,5],
                      [4,0,0,0,5,2,2,10],[1,2,0,5,0,10,0,0],[0,2,0,2,10,0,5,1],
                      [0,2,0,2,0,5,0,10],[6,0,5,10,0,1,10,0]],
                    columns=solutions,
                    index=solutions
    )
    random.shuffle(solutions)
    X0 = solutions.copy()
    new_dist_df =Dist.reindex(columns = X0,index = X0)
    New_dist = np.array(Dist.reindex(columns = X0,index = X0))

    obj_fun_start = pd.DataFrame(New_dist*Flow)
    onj_fun_start_sum = sum(obj_fun_start)