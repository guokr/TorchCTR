#!/usr/bin/env python
# encoding: utf-8

import torch
from torchctr.layers import EmbeddingLayer, LinearLayer


class FactorizationMachineLayer(torch.nn.Module):
    ## second order
    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        res = square_of_sum - sum_of_square

        if self.reduce_sum:
            return 0.5 * torch.sum(res, dim=1, keepdim=True)
        else:
            return res


class FactorizationMachine(torch.nn.Module):
    """
    FactorizationMachine Model
    """
    def __init__(self, feature_dims, embed_dim):
        super().__init__()
        self.embedding = EmbeddingLayer(feature_dims, embed_dim)
        self.linear = LinearLayer(feature_dims)
        self.fm = FactorizationMachineLayer(reduce_sum=True)

    def forward(self, x, sigmoid=True):
        # add the first order and second order
        fm_full = self.linear(x) + self.fm(self.embedding(x))
        if sigmoid:
            return torch.sigmoid(fm_full.squeeze(1))
