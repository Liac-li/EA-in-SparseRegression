import os

import pandas
import numpy as np


class DataLoader(object):
    def __init__(self, data_folder):
        self.path = data_folder
        self.data_pool = {
            "svmguide3": "svmguide3_1.csv",
            "ionosphere": "ionosphere/archive/ionosphere.csv",
            "sonar": "sonar/archive/sonar.csv",
            "triazines": "triazines.csv",
        }

    @staticmethod
    def coder(s):
        if isinstance(s, int) or isinstance(s, float):
            return s

        if s.isdigit():
            return s
        else:
            res = 0
            for c in s:
                res += ord(c)
            return str(res)

    def load_data(self, dataset_name):
        if dataset_name.lower() not in self.data_pool.keys():
            raise ValueError("Dataset name not found; {}".format(dataset_name))

        data_path = os.path.join(self.path,
                                 self.data_pool[dataset_name.lower()])

        pd = pandas.read_csv(data_path, delimiter=',')
        pd[pd.columns[-1]] = pd[pd.columns[-1]].apply(self.coder)
        pd.fillna(0)

        x = pd[pd.columns[:-1]].to_numpy(dtype=np.float128)
        y = pd[pd.columns[-1]].to_numpy(dtype=np.float128)
        y = y[..., np.newaxis]
        # print(y)
        x[x != x] = 0
        y[y != y] = 0

        print(f"[Log] load  x as {x.shape}, y as {y.shape}")

        return x, y
