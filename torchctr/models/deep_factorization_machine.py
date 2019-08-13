#!/usr/bin/env python
# encoding: utf-8

import torch
from torchctr.layers import LinearLayer, EmbeddingLayer, MultiLayerPerceptron
from torchctr.models.factorization_machine import FactorizationMachineLayer
from torchctr.models.checker import Checker


class DeepFactorizationMachine(torch.nn.Module):
    @Checker.model_param_check
    def __init__(self, feature_dims, embed_dim, hidden_dims):
        super().__init__()
        self.fm_second_order = FactorizationMachineLayer()
        self.fm_linear = LinearLayer(feature_dims)
        self.embedding = EmbeddingLayer(feature_dims, embed_dim)
        self.mlp_input_dim = embed_dim * len(feature_dims)

        self.mlp = MultiLayerPerceptron(
            input_dim=self.mlp_input_dim, hidden_dims=hidden_dims, output_dim=1
        )

    def forward(self, x, sigmoid=True):
        fm_part = self.fm_second_order(self.embedding(x)) + self.fm_linear(x)
        deep_part = self.mlp(self.embedding(x).view(-1, self.mlp_input_dim))
        deep_fm = fm_part + deep_part
        if sigmoid:
            return torch.sigmoid(deep_fm.squeeze(1))
