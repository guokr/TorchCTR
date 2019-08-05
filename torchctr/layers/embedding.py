#!/usr/bin/env python
# encoding: utf-8

import torch
import numpy as np


class EmbeddingLayer(torch.nn.Module):
    ### same as linear layer but with embedding dims
    def __init__(self, num_features, embed_dim):
        super().__init__()
        self.weights_embed = torch.nn.Embedding(
            sum(num_features) + 1, embed_dim
        )
        self.feature_loc_offsets = torch.tensor(
            np.array((0, *np.cumsum(num_features)[:-1])), dtype=torch.long
        )
        torch.nn.init.xavier_uniform_(self.weights_embed.weight.data)

    def forward(self, x):
        x = x + self.feature_loc_offsets
        return self.weights_embed(x)
