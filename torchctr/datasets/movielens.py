#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import pandas as pd
from .base import BaseDataset


class MovieLens(BaseDataset):
    def __init__(self):
        super().__init__()

    def build_data(self):
        data = pd.read_csv(
            "~/.torchctr/ml-1m/ratings_dummy.dat",
            sep="::",
            header=None,
            engine="python",
        )
        # dont need timestamp/ the last column
        self.data = data[data.columns[:-1]]
        self.data.columns = ["userID", "itemID", "rating"]

        self.y_column = "rating"
        self.x_columns = [c for c in self.data.columns if c != self.y_column]

        self.data = self.data.astype({c: str for c in self.x_columns})
        # process x
        self.preprocess_x()
        # build feature
        self.preprocess_y(y_func=lambda x: 1 if x>3 else 0)

        self.x = self.data[self.x_columns].to_numpy().astype(np.int)
        self.y = self.data[self.y_column].to_numpy().astype(np.float32)
        self.feature_dims = np.max(self.x, axis=0)
