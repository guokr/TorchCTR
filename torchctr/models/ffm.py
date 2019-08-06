#!/usr/bin/env python
# encoding: utf-8

import torch
import numpy as np
from torchctr.layers import LinearLayer


class FieldAwareFactorizationMachineLayer(torch.nn.Module):
    def __init__(self, feature_dims, embed_dim):
        super().__init__()
        self.num_fields = len(feature_dims)
        self.weights_embed = torch.nn.ModuleList(
            [
                torch.nn.Embedding(sum(feature_dims) + 1, embed_dim)
                for _ in range(self.num_fields)
            ]
        )

        self.feature_loc_offsets = torch.tensor(
            np.array((0, *np.cumsum(feature_dims)[:-1])), dtype=torch.long
        )

        for weight_emb in self.weights_embed:
            torch.nn.init.xavier_uniform_(weight_emb.weight.data)

    def forward(self, x):
        adjusted_x = x + self.feature_loc_offsets
        embedded_x = [
            self.weights_embed[i](adjusted_x) for i in range(self.num_fields)
        ]

        cross_results = []
        # second order product
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                cross_results.append(
                    embedded_x[j][:, i, :] * embedded_x[i][:, j, :]
                )

        return torch.stack(cross_results, dim=1)


class FieldAwareFactorizationMachine(torch.nn.Module):
    def __init__(self, feature_dims, embed_dim):
        super().__init__()
        self.linear = LinearLayer(feature_dims)
        self.ffm = FieldAwareFactorizationMachineLayer(feature_dims, embed_dim)

    def forward(self, x, sigmoid=True):
        ffm_part = torch.sum(torch.sum(self.ffm(x), dim=1), dim=1, keepdim=True)
        ffm_full = self.linear(x) + ffm_part
        if sigmoid:
            return torch.sigmoid(ffm_full.squeeze(1))
