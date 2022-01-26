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


data_name = "triazines"
date_set = DataLoader("./data")

if __name__ == '__main__':
    x, y = date_set.load_data(data_name)

    iterTime = 1
    tmp = []
    for i in range(iterTime):
        module = Sparse_MOEAD(fns=[f1, f2], popSize=64,
                              t=5)  # neighors' num is 2
        tmp.append(module.run(x, y, maxIterNum=50))

    # fit = [x[-1] for x in tmp]
    # fit = np.array(fit)
    # print(fit.shape)

    # fits = np.mean(fit)

    _, fitness, fits = random.choice(tmp)
    Xs = [item[0] for item in fitness]
    Ys = [item[1] for item in fitness]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(Xs, Ys, facecolor='none', edgecolor='b')
    ax1.set_xlabel('f1')
    ax1.set_ylabel('f2')

    ax2.plot(fits, '.--')
    ax2.set_xlabel('Evo times')
    ax2.set_ylabel('Mean of all FV in EP')

    plt.title(f"{data_name}_moead")
    plt.show()
