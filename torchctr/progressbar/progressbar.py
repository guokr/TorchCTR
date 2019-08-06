#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


class ProgressBar:
    def __init__(self, dataloader, metrics, desc):
        self.tqdm = tqdm(dataloader, desc=desc)
        self.metrics = metrics

    def __iter__(self):
        return iter(self.tqdm)

    def eval(self, target, prediction, append_dict):
        res = {}
        if "auc" in self.metrics:
            try:
                auc_score = roc_auc_score(target, prediction)
            except ValueError:
                auc_score = np.nan


            res["auc"] = "{:.2f}".format(auc_score)

        for k,v in append_dict.items():
            res[k] = "{:.2f}".format(v)

        self.tqdm.set_postfix(res)
