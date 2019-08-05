#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import pandas as pd
from .base import BaseDataset

class Titanic(BaseDataset):
    def __init__(self):
        None

    def build_data(self):
        self.data = pd.read_csv("~/.torchctr/titanic/titanic_train.txt", sep="\t", header=None, engine="python")

        non_categorical = ["I{}".format(_) for _ in range(1, 3)]
        categorical = ["C{}".format(_) for _ in range(1, 13)]

        self.y_column = "click"
        self.x_columns = categorical + non_categorical
        self.data.columns = [self.y_column] + self.x_columns

        self.preprocess_x(non_categorical=non_categorical)

        self.preprocess_y()

        self.x = self.data[self.x_columns].to_numpy().astype(np.int)
        self.y = self.data[self.y_column].to_numpy().astype(np.float32)

        self.feature_dims = np.max(self.x, axis=0)

