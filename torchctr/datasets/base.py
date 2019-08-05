#!/usr/bin/env python
# encoding: utf-8

from torch.utils.data import Dataset
from collections import defaultdict


class BaseDataset(Dataset):
    def __init__(self):
        self.feature_mapper = {}
        self.field_mapper = {}

    def load_data(self):
        raise NotImplementedError

    def preprocess_x(self):
        self.field_mapper = {
            field: idx for idx, field in enumerate(self.x_columns)
        }

        feature_counter = defaultdict(lambda: defaultdict(int))
        for c in self.x_columns:
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

        for c in self.x_columns:
            self.data[c] = self.data[c].apply(
                lambda x: self.feature_mapper[self.field_mapper[c]][x]
            )

    def preprocess_y(self):
        self.data[self.y_column] = self.data[self.y_column].apply(
            lambda x: 1 if x > 3 else 0
        )

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
