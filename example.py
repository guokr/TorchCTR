#!/usr/bin/env python
# encoding: utf-8

from torchctr.datasets import MovieLens
from torchctr.models import LogisticRegression, FactorizationMachine
from torchctr.trainer import Trainer

dataset = MovieLens()
dataset.build_data()

# dataset = Titanic()
# dataset = Avazu()

dims = dataset.feature_dims
print("dataset dims", dims)

# model = LogisticRegression(dims)
model = FactorizationMachine(dims, embed_dim=4)

trainer = Trainer(model, dataset)
trainer.train()
