#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import torch


class LinearLayer(torch.nn.Module):
    def __init__(self, num_features, output_dim=1):
        super().__init__()
        self.weights_embed = torch.nn.Embedding(
            sum(num_features) + 2, output_dim
        )
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.feature_loc_offsets = torch.tensor(
            np.array((0, *np.cumsum(num_features)[:-1])), dtype=torch.long
        )
        # print(self.feature_loc_offsets)

    def forward(self, x):
        adjusted_x = x + self.feature_loc_offsets
        return torch.sum(self.weights_embed(adjusted_x), dim=1) + self.bias
