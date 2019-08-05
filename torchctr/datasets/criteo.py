#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import pandas as pd
from .base import BaseDataset

class Criteo(BaseDataset):
    def __init__(self):
        super().__init__()

    def load_data(self):
        data = pd.read_csv("~/.torchctr/criteo/train_mini.txt", sep="\t", header=None)

        non_categorical = ["I{}".format(_) for _ in range(1, 14)]
        categorical = ["C{}".format(_) for _ in range(1, 27)]
        data.columns  = ["click"] + non_categorical + categorical

        self.y_column = "click"
        self.x_columns = non_categorical + categorical



