import math
import numpy as np
import random

from sklearn.linear_model import LinearRegression


class POSS(object):
    def __init__(self, keepMse=False):
        self.validMethods = {'mse': self.POSS_MSE, 'smc': self.POSS_SMC}
        self.k = 8
        self.keepMSE = keepMse

        self.cache = []
        self.last_data = np.inf

    def __call__(self, method_name):
        method_name = method_name.lower()
        if not method_name in self.validMethods.keys():
            raise ValueError(f'method {method_name} is not supported')

        return self.validMethods[method_name]  # return poss_fn

    def POSS_SMC(self, x, y, k=None):
        if k is None:
            k = self.k

        m, n = x.shape
        population = np.zeros((1, n))
        popSize = 1
        fitness = np.zeros((1, 2))
        fitness[0, 0] = np.NINF
        fitness[0, 1] = np.sum(population)

        T = math.ceil(n * k * k * 2 * np.exp(1))  # total iterations' number
        print(f'[Log] Iter Times T: {T}')

        for i in range(T):

            offspring = np.abs(
                population[random.randint(0, popSize - 1), :] -
                np.random.choice([1, 0], size=(1, n), p=[1 / n, 1 - 1 / n]))
            offspringFit = np.zeros(2)  # [fitness, size]
            offspringFit[1] = np.sum(offspring)
            # print(offspringFit[1])

            if offspringFit[1] == 0 or offspringFit[1] >= 2 * k:  # infeasible sol
                offspringFit[0] = np.NINF
            else:
                """
                    math: SMC = (var{z} - mse{z}) / var{z} => 1 - mse{z} 
                    if E[x] = 0, var[x] = 1
                """
                lr = LinearRegression()
                pos = offspring[0] == 1
                tmp = x[:, pos]
                lr.fit(tmp, y)
                mse = np.mean((lr.predict(tmp) - y)**2)
                # TODO: get linear regression coefficients
                offspringFit[0] = 1 - mse
                # offspringFit[0] = mse

                if self.keepMSE:
                    if self.last_data > mse:
                        self.last_data = mse
                    self.cache.append([i, self.last_data])

            #  1: size -> less is better, 0: fitness -> great is better
            if np.sum((fitness[:popSize, 0] > offspringFit[0]) *
                      (fitness[:popSize, 1] <= offspringFit[1])) + np.sum(
                          (fitness[:popSize, 0] >= offspringFit[0]) *
                          (fitness[:popSize, 1] <
                           offspringFit[1])) > 0:  # ! dominated
                pass
            else:
                # remove worse solution in population
                deleteIdx = (fitness[:popSize, 0] <= offspringFit[0]) * (
                    fitness[:popSize, 1] >= offspringFit[1])

                ndelete = (deleteIdx == 0)
                population = np.concatenate(
                    (population[ndelete, :], offspring))
                fitness = np.concatenate(
                    (fitness[ndelete, :], offspringFit[..., np.newaxis].T))
                popSize = population.shape[0]
                # print(popSize)

        validIdx = fitness[:, 1] <= k
        maxVal = np.max(fitness[validIdx, 0])
        seq = fitness[:, 0] >= maxVal
        res = population[seq, :][0]
        print(res)

        return res, self.cache

    def POSS_MSE(self, x, y, k=None):
        raise NotImplementedError
