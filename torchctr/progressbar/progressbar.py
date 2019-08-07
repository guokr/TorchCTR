#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score


class ProgressBar:
    def __init__(self, dataloader, metrics, desc):
        self.tqdm = tqdm(dataloader, desc=desc)
        self.metrics = metrics
        self.targets, self.predictions = [], []
        self.summary = {}

    def __iter__(self):
        return iter(self.tqdm)

    def eval(self, target, prediction, append_dict):
        self.predictions.extend(prediction)
        self.targets.extend(target)

        res = {}
        if "auc" in self.metrics:
            try:
                auc_score = roc_auc_score(target, prediction)
            except ValueError:
                # target all same class
                auc_score = np.nan
            res["auc"] = "{:.3f}".format(auc_score)

        if "acc" in self.metrics:
            acc_score = accuracy_score(target, [1 if x>0.5 else 0 for x in prediction])
            res["acc"] = "{:.3f}".format(acc_score)

        for k, v in append_dict.items():
            res[k] = "{:.3}".format(v)

        self.tqdm.set_postfix(res)

    def summarize(self):
        if "auc" in self.metrics:
            self.summary["auc"] = roc_auc_score(self.targets, self.predictions)
        if "acc" in self.metrics:
            self.summary["acc"] = accuracy_score(self.targets, [1 if x>0.5 else 0 for x in self.predictions])
