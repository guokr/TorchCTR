#!/usr/bin/env python
# encoding: utf-8

import torch
from torchctr.layers import LinearLayer


class LogisticRegression(torch.nn.Module):
    """
    Simple LR with sigmoid or not

    """

    def __init__(self, feature_dims):
        super().__init__()
        self.linear = LinearLayer(feature_dims)

    def forward(self, x, sigmoid=True):
        if sigmoid:
            return torch.sigmoid(self.linear(x).squeeze(1))
