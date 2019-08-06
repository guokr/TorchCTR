#!/usr/bin/env python
# encoding: utf-8

import math
from torch.utils.data import Dataset
from collections import defaultdict


class BaseDataset(Dataset):
    def __init__(self):
        self.feature_mapper = {}
        self.field_mapper = {}

    def load_data(self):
        raise NotImplementedError

    def preprocess_x(self, non_categorical=[], non_categorical_func=None):
        if non_categorical_func == None:
            print("| Warning | Didn't specify the func for dense field, so we will use default log ")
            non_categorical_func = lambda x: int(math.log(x+10) ** 2)

        self.field_mapper = {
            field: idx for idx, field in enumerate(self.x_columns)
        }

        feature_counter = defaultdict(lambda: defaultdict(int))

        for c in self.x_columns:
            if c in non_categorical:
                self.data[c] = self.data[c].apply(lambda x: non_categorical_func(x))
            di = self.data[c].value_counts()
            di.index = di.index.astype(str)
            feature_counter[self.field_mapper[c]] = di.to_dict()

        feature_mapper = {
            i: {feat for feat, c in cnt.items() if c > 0}
            for i, cnt in feature_counter.items()
        }
        self.feature_mapper = {
            i: {feat: idx for idx, feat in enumerate(cnt)}
            for i, cnt in feature_mapper.items()
        }

        self.data = self.data.astype({c: str for c in self.x_columns})

        for c in self.x_columns:
            self.data[c] = self.data[c].apply(
                lambda x: self.feature_mapper[self.field_mapper[c]][x]
            )

    def preprocess_y(self, y_func=None):
        if y_func == None:
            print("| Warning | Didn't specify the func for target column, so we will use raw data")

        # self.data[self.y_column] = self.data[self.y_column]

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
