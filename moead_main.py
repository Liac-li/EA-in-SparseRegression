from utils import DataLoader
from modules import Sparse_MOEAD

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import random

random.seed(114514)

def f1(mask, x, y):
    lr = LinearRegression()
    pos = mask != 0
    tmp = x[:, pos]
    lr.fit(tmp, y)
    mse = np.mean((lr.predict(tmp) - y)**2)
    return mse

def f2(mask, x, y):
    return np.sum(mask)

data_name = "sonar"
date_set = DataLoader("./data")

if __name__ == '__main__':
    x, y = date_set.load_data(data_name)
    module = Sparse_MOEAD(fns=[f1, f2], popSize=64, t=5) # neighors' num is 5

    iterTime = 1 
    tmp = []
    for i in range(iterTime):
        tmp.append(module.run(x, y, maxIterNum=200))
        
    _, fitness = random.choice(tmp)
    Xs = [item[0] for item in fitness]
    Ys = [item[1] for item in fitness]
    
    plt.scatter(Xs, Ys, facecolor='none', edgecolor='b')
    
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.title(f"{data_name}_moead")
    plt.show()

