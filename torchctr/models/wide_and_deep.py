#!/usr/bin/env python
# encoding: utf-8

import torch

from torchctr.layers import LinearLayer, EmbeddingLayer, MultiLayerPerceptron

class WideAndDeepModel(torch.nn.Module):
    def __init__(self, feature_dims, embed_dim, hidden_dims):
        super().__init__()
        self.linear = LinearLayer(feature_dims)
        self.embedding  = EmbeddingLayer(feature_dims, embed_dim)
        self.mlp_input_dim = embed_dim * len(feature_dims)
        self.mlp = MultiLayerPerceptron(input_dim = self.mlp_input_dim,
                                        hidden_dims = hidden_dims,
                                        output_dim = 1)

    def forward(self, x, sigmoid=True):
        linear_part = self.linear(x)
        mlp_part = self.mlp(self.embedding(x).view(-1, self.mlp_input_dim))
        wide_and_deep = linear_part + mlp_part
        if sigmoid:
            return torch.sigmoid(wide_and_deep.squeeze(1))
