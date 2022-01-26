from utils import DataLoader
from modules import Sparse_NSGA_II

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import random

random.seed(114514)


def f1(X, y, mask):
    lr = LinearRegression()
    pos = mask != 0
    tmp = X[:, pos]
    # print(tmp.shape, y.shape)
    lr.fit(tmp, y)
    mse = np.mean((lr.predict(tmp) - y)**2)
    return mse


def f2(X, Y, mask):
    return np.sum(mask)


data_name = 'triazines'
data_set = DataLoader('./data')

if __name__ == '__main__':
    x, y = data_set.load_data(data_name)

    module = Sparse_NSGA_II(judge_fn=[f1, f2])

    iterTimes = 10
    tmp = []
    for i in range(iterTimes):
        tmp.append(module.run(x, y, popSize=32, T=500))

    _, fitness, fits = random.choice(tmp)
    Xs = [i[0] for i in fitness]
    Ys = [i[1] for i in fitness]

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.scatter(Xs, Ys, facecolor='none', edgecolor='b')
    ax1.set_xlabel('f1')
    ax1.set_ylabel('f2')
    # ax1.set_title(data_name)

    ax2.plot(fits, '.--')
    ax2.set_xlabel('Evolution Loop num')
    ax2.set_ylabel('mean of all functions')

    plt.title(f'{data_name}_nsga-ii')
    plt.show()
