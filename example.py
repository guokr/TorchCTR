#!/usr/bin/env python
# encoding: utf-8

from torchctr.datasets import MovieLens, Titanic
from torchctr.models import LogisticRegression, FactorizationMachine
from torchctr.trainer import Trainer

# dataset = MovieLens()
# dataset = Avazu()
dataset = Titanic()
dataset.build_data()


dims = dataset.feature_dims
print("dataset dims", dims)

# model = LogisticRegression(dims)
model = FactorizationMachine(dims, embed_dim=4)

trainer = Trainer(model, dataset)
trainer.train()
