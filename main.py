from utils import DataLoader
from modules import POSS

import random
import numpy as np
import matplotlib.pyplot as plt

data_set = DataLoader('./data')
Poss_SMC = POSS(keepMse=True)('smc')
data_name = 'ionosphere'
random.seed(114514)

if __name__ == '__main__':

    x, y = data_set.load_data(data_name)
    print(np.nan in x, np.nan in y)

    # x = (x - x.mean(axis=0)) / x.std(axis=0)
    # y = (y - y.mean(axis=0)) / y.std(axis=0)

    print(x.mean(), x.var())
    # print(x[1:2], y[1:2])
    k = 8

    iterTimes = 1
    tmp = []
    for i in range(iterTimes):
        tmp.append(Poss_SMC(x, y, k))

    _, tmp, pareto = random.choice(tmp)

    tmp = np.array(tmp)
    Xs = tmp[:, 0].reshape(tmp.shape[0])
    Ys = tmp[:, 1].reshape(tmp.shape[0])

    f1 = []
    f2 = []
    for item in pareto:
        f1.append(item[0])
        f2.append(item[1])

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(Xs, Ys, ',--')
    ax1.set_xlabel('iter Times')
    ax1.set_ylabel('mse')

    ax2.scatter(f1, f2, facecolor='none', edgecolor='b')
    ax2.set_xlabel('SMC')
    ax2.set_ylabel('|X|')

    plt.title(f'{data_name}_POSS')
    plt.show()
