import math
from matplotlib.pyplot import new_figure_manager
import numpy as np
import random
from copy import deepcopy
from tqdm import trange

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

            if offspringFit[
                    1] == 0 or offspringFit[1] >= 2 * k:  # infeasible sol
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


class Sparse_NSGA_II():

    def __init__(self, judge_fn=None, isLess=True, popSize=16):
        if judge_fn is None:
            raise ValueError('judging function at least one')

        self.judge_fns = judge_fn
        self.isLess = isLess
        self.popSize = 16
        self.k = 8

        self.populations = None  # (m * features)
        self.PopFitness = None  # (m * fitness)
        self.perot_front = []  # [[rank1], [rank2], ...]
        self.Dist = {}  # (m,)

        self.X = None
        self.Y = None

    def _init_param(self, X, Y, popSize=None):
        if popSize is not None:
            self.popSize = popSize
        self.X = X
        self.Y = Y

        m, size = X.shape
        self.populations = [
            np.random.choice([1, 0], size=size, p=[1 / size, 1 - 1 / size])
            for i in range(self.popSize)
        ]
        for idx in range(len(self.populations)):
            while np.sum(self.populations[idx]) == 0:
                self.populations[idx] = np.random.choice(
                    [1, 0], size=size, p=[1 / size, 1 - 1 / size])

        self.PopFitness = [
            self.getFitness(self.X, self.Y, pop, self.judge_fns)
            for pop in self.populations
        ]

    @staticmethod
    def isDomiated(x_fitness, y_fitness, less=True):
        """
            Parameters:
                less: true => less is better
        """
        if not less:
            # x dominates y
            if False not in (x_fitness >=
                             y_fitness) and np.sum(x_fitness > y_fitness) > 0:
                return True
        else:
            if False not in (x_fitness <=
                             y_fitness) and np.sum(x_fitness < y_fitness) > 0:
                return True

        return False

    def fast_non_dominated_sort(self):
        popSize = len(self.populations)

        S = [[] for i in range(popSize)]
        n = np.zeros(popSize)
        rank = np.zeros(popSize)
        newfront = [[]]

        # print(len(self.populations), len(self.PopFitness))

        for x in range(popSize):
            S[x] = []
            n[x] = 0
            for y in range(popSize):
                if self.isDomiated(self.PopFitness[x], self.PopFitness[y],
                                   self.isLess):
                    if y not in S[x]:
                        S[x].append(y)
                elif self.isDomiated(self.PopFitness[y], self.PopFitness[x],
                                     self.isLess):
                    n[x] += 1

            if n[x] == 0:
                rank[x] = 0
                if x not in newfront[0]:
                    newfront[0].append(x)

        i = 0
        while newfront[i] != []:
            Q = []
            for x in newfront[i]:
                for y in S[x]:
                    n[y] -= 1
                    if n[y] == 0:
                        rank[y] = i + 1
                        if y not in Q:
                            Q.append(y)
            i += 1
            newfront.append(Q)

        del newfront[-1]

        # print(newfront)
        return newfront

    def get_crowdingDist(self, frontSols):
        """
            Parameters:
                frontSols: [idx1, idx2, ...] with same rank

            Return :
                dist: {idx: dist, ...}
        """
        dist = {}
        for idx in frontSols:
            dist[idx] = 0

        front = [(idx, self.PopFitness[idx]) for idx in frontSols]

        if len(front) <= 2:
            dist[front[0][0]] = np.inf
            dist[front[-1][0]] = np.inf
            return dist

        for f in range(len(self.judge_fns)):
            front = sorted(front, key=lambda x: x[-1][f])
            dist[front[0][0]] = np.inf
            dist[front[-1][0]] = np.inf

            fitness = [item[-1][f] for item in front]

            for j in range(1, len(front) - 1):
                dist[front[j][0]] += (front[j + 1][-1][f] - front[j - 1][-1][f]
                                      ) / max(fitness) - min(fitness)

        return dist

    @staticmethod
    def getFitness(X, Y, mask, fns):
        popFitness = [fn(X, Y, mask) for fn in fns]
        popFitness = np.array(popFitness)
        return popFitness

    # TODO: add evolution algorithm
    def genNewPopulation(self):
        """
            parent selection: binary tourant
            crossover: one point
            surviver: N + N selection
        """
        offspring = []
        offspringFit = []
        _, featureNum = self.X.shape

        for i in range(self.popSize):
            idx = random.randint(0, featureNum)
            tmp = random.choices(self.populations, k=2)
            # print(type(tmp[0]), tmp)
            newOffspring = np.concatenate((tmp[0][:idx], tmp[1][idx:]),
                                          axis=0)  # crossover
            # print(newOffspring)
            newOffspring += np.random.choice(
                [1, 0],
                size=featureNum,
                p=[1 / featureNum, 1 - 1 / featureNum])  # mutation

            while np.sum(newOffspring) == 0:
                newOffspring = np.random.choice(
                    [0, 1],
                    size=featureNum,
                    p=[1 / featureNum, 1 - 1 / featureNum])
            # newOffspring = np.array(newOffspring)

            offspring.append(newOffspring)
            # print(np.sum(newOffspring), newOffspring)

            offspringFit.append(
                self.getFitness(self.X, self.Y, newOffspring, self.judge_fns))

        # offspringFit = np.array(offspringFit)
        lassPop = deepcopy(self.populations)
        self.populations += offspring
        self.PopFitness += offspringFit
        # self.PopFitness = np.concatenate((self.PopFitness, offspringFit),
        #  axis=0)

        fronts = self.fast_non_dominated_sort()
        nextPop = []
        cnt = 0
        for front in fronts:
            dist = self.get_crowdingDist(front)
            dist = sorted(dist.items(), key=lambda x: x[1], reverse=True)
            for i in range(self.popSize - cnt):
                if i < len(front):
                    nextPop.append(self.populations[dist[i][0]])
                else:
                    break
            cnt = len(nextPop)

        assert len(nextPop) == self.popSize
        self.populations = nextPop
        self.PopFitness = [
            self.getFitness(self.X, self.Y, pop, self.judge_fns)
            for pop in self.populations
        ]
        # self.PopFitness = np.array(self.PopFitness)

    def run(self, X, Y, popSize=None, T=1000):
        self._init_param(X, Y, popSize)

        for i in trange(T):
            self.genNewPopulation()
            # if i % 100 == 0:
            #     print(f"{i}/{T}")

        return self.populations, self.PopFitness


class Sparse_MOEAD():

    def __init__(self, fns=None, popSize=10, t=None):
        if fns is None:
            raise ValueError("judge functions can't be none")

        self.fns = fns
        self.popSize = popSize

        self.maxIterNum = 50
        self.tSize = None
        if t is None:
            self.tSize = 5
        elif t < 1:
            raise ValueError("Neighbors's num must greater than 1")
        else:
            self.tSize = t  # neightbors's num
        self.EP_ID = []
        self.EP_FV = []

        # following need be initilazed
        self.featureNum = None
        self.X = None
        self.Y = None

        self.population = []
        self.popFV = []
        self.W = None  # (m, )
        self.W_BI_T = []
        self.Z = None

    def _init_params(self, X, Y):
        self.X = X
        self.Y = Y
        self.featureNum = X.shape[1]

        self.W = np.random.rand(len(self.fns))
        self.W /= np.sum(self.W)

        for bi_idx in range(self.W.shape[0]):
            Bi = self.W[bi_idx]
            dist = np.sum((self.W - Bi)**2, axis=1)
            Bt = np.argsort(dist)
            Bt = Bt[1:self.tSize + 1]  # dist(x,x) must be 0
            self.W_BI_T.append(Bt)
        self.W_BI_T = np.array(self.W_BI_T)

        self.population = [
            np.random.choice([0, 1],
                             size=(self.popSize, X.shape[1]),
                             p=[1 / X.shape[1], 1 - 1 / X.shape[1]])
            for i in range(X.shape[1])
        ]

        for pop in self.population:
            popFV = self.getFitness(pop, self.fns)
            self.popFV.append(popFV)

        self.Z = np.zeros_like(self.W)

    def getFitness(self, pop, fns):
        pass

    def genNextY(self, wi, pop, popk, popl):
        pops = [pop, popk, popl]
        cbxfs = [self.cbxf(wi, pop) for pop in pops]
        cbxfs = np.array(cbxfs)
        best = np.argmin(cbxfs)

        Y0 = pops[best]

        # crossover & mutation
        offsprings = []
        for i in range(len(pops)):
            crossIdx = np.random.randint(len(pops[i]))
            next = (i + 1) % len(pops)
            offspring = np.concatenate(
                (pops[i][:crossIdx], pops[next][crossIdx:]), axis=0)
            while np.sum(offspring) == 0:
                offspring = np.random.choice(
                    [0, 1],
                    size=self.featureNum,
                    p=[1 / self.featureNum, 1 - 1 / self.featureNum])
            offsprings.append(offspring)

        tmp = [pop, popk, popl] + offsprings
        cbxfs = [self.cbxf(wi, pop) for pop in tmp]
        cbxfs = np.array(cbxfs)
        best = np.argmin(cbxfs)
        Y1 = tmp[best]

        fn = np.random.randint(len(self.fns))
        if np.random.rand() < 0.5:
            FY0 = self.getFitness(Y0, self.fns[fn])
            FY1 = self.getFitness(Y1, self.fns[fn])
            if FY0 < FY1:
                return Y0
            else:
                return Y1
        return Y1

    @staticmethod
    def cbxf_dist(w, f, z):
        return w * np.abs(f - z)

    def cbxf(self, popId, pop):
        max = self.Z[0]
        ri = self.W[popId]
        FV_x = self.getFitness(pop, self.fns)
        for i in range(len(self.fns)):
            fi = self.cbxf_dist(ri[i], FV_x[i], self.Z[i])
            if fi > max:
                max = fi
        return max

    def run(self, X, Y, maxIterNum=None):
        if maxIterNum is not None:
            self.maxIterNum = maxIterNum

        self._init_params(X, Y)

        for loopN in range(self.maxIterNum):
            for popId, pop in enumerate(self.population):
                """
                    1: preproduction y
                    2: Improvement on y
                    3: Update Z
                    4: Update Neighbors' solutions
                """
                Bi = self.W_BI_T[popId]
                k = np.random.randint(0, self.tSize)
                l = np.random.randint(0, self.tSize)
                idk = Bi[k]
                idl = Bi[l]
                popk = self.population[idk]
                popl = self.population[idl]

                Y = self.genNextY(popId, pop, popk, popl)

                cbxf_x = self.cbxf(popId, pop)
                cbxf_y = self.cbxf(popId, Y)

                if cbxf_y < cbxf_x:
                    # TODO: update pop, z, EP[ID/FV]
                    pass

        return self.population[self.EP_ID, :], self.popFV[self.EP_ID, :]
