#!/usr/bin/env python
# encoding: utf-8

import torch


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        dim_pairs = list(zip(dims, dims[1:]))

        layers = []
        for p in dim_pairs[:-1]:
            layers.append(torch.nn.Linear(p[0], p[1]))
            layers.append(torch.nn.BatchNorm1d(p[1]))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))

        layers.append(torch.nn.Linear(dim_pairs[-1][0], dim_pairs[-1][1]))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
