#!/usr/bin/env python
# encoding: utf-8

import torch
from torchctr.layers import MultiLayerPerceptron, LinearLayer
from torchctr.models.fieldaware_factorization_machine import FieldAwareFactorizationMachineLayer

class FieldAwareNeuralFactorizationMachine(torch.nn.Module):
    def __init__(self, feature_dims, embed_dim, hidden_dims):
        super().__init__()
        self.linear = LinearLayer(feature_dims)
        self.ffm_second_order = FieldAwareFactorizationMachineLayer(feature_dims, embed_dim)
        self.mlp_input_dim = int(len(feature_dims) * (len(feature_dims)-1) / 2 * embed_dim) # 9*8 or 8*7 must be even number
        self.mlp = MultiLayerPerceptron(input_dim = self.mlp_input_dim,
                                        hidden_dims = hidden_dims,
                                        output_dim = 1)

    def forward(self, x, sigmoid=True):
        ffm_part = self.ffm_second_order(x)
        deep_part = self.mlp(ffm_part.view(-1, self.mlp_input_dim))
        linear_part = self.linear(x)

        fnfm = linear_part + deep_part
        if sigmoid:
            return torch.sigmoid(fnfm.squeeze(1))



