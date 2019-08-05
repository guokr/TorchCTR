#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class Titanic(Dataset):
    def __init__(self):
        None

    def load_data(self):
        data = pd.read_csv("~/.torchctr/titanic/titanic_train.txt", sep="\t", header=None, engine="python")
        data = data.to_numpy()

        self.x = data[:, 1:].astype(np.float32)
        self.y = data[:, 1].astype(np.float32)
        # self.f = np.max(self.x, axis=0)
        return (self.x, self.y)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def preprocess(self, target):
        target[target <= 3] = 0
        target[target > 3] = 1
        return target
