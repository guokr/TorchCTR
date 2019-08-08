#!/usr/bin/env python
# encoding: utf-8

import torch
from torchctr.layers import EmbeddingLayer, LinearLayer, MultiLayerPerceptron
from torchctr.models.factorization_machine import FactorizationMachineLayer


class NeuralFactorizationMachine(torch.nn.Module):
    def __init__(self, feature_dims, embed_dim, hidden_dims):
        super().__init__()
        self.embedding = EmbeddingLayer(feature_dims, embed_dim)
        self.linear = LinearLayer(feature_dims)
        self.fm_second_order = FactorizationMachineLayer(reduce_sum=False)
        self.mlp_input_dim = embed_dim
        self.mlp = MultiLayerPerceptron(input_dim =self.mlp_input_dim,
                                        hidden_dims=hidden_dims,
                                        output_dim=1)

    def forward(self, x, sigmoid=True):
        linear_part = self.linear(x)
        deep_part = self.mlp(self.fm_second_order(self.embedding(x)))
        nfm = linear_part + deep_part
        if sigmoid:
            return torch.sigmoid(nfm.squeeze(1))
