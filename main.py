from utils import DataLoader
from modules import POSS

import random
import numpy as np
import matplotlib.pyplot as plt

data_set = DataLoader('./data')
Poss_SMC = POSS(keepMse=True)('smc')
data_name = 'sonar'
random.seed(114514)

if __name__ == '__main__':

    x, y = data_set.load_data(data_name)

    x = (x - x.mean(axis=0)) / x.std(axis=0)
    y = (y - y.mean(axis=0)) / y.std(axis=0)

    print(x.mean(), x.var())
    # print(x[1:2], y[1:2])
    k = 8

    iterTimes = 10
    tmp = []
    for i in range(iterTimes):
        selectedVar_SMC, mse_seq = Poss_SMC(x, y, k)
        tmp.append(mse_seq)

    tmp = random.choice(tmp)
    tmp = np.array(tmp)
    Xs = tmp[:, 0].reshape(tmp.shape[0])
    Ys = tmp[:, 1].reshape(tmp.shape[0])

    plt.plot(Xs, Ys, ',--')
    plt.xlabel('iter Times')
    plt.ylabel('mse')
    plt.title(data_name)
    plt.show()
