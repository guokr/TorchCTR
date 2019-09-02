#!/usr/bin/env python
# encoding: utf-8

from torchctr.datasets import MovieLens, Titanic, Avazu
from torchctr.models import LogisticRegression, FactorizationMachine, FieldAwareFactorizationMachine
from torchctr.models import  WideAndDeepModel
from torchctr.models import FieldAwareNeuralFactorizationMachine, DeepFactorizationMachine, NeuralFactorizationMachine

from torchctr.trainer import Trainer

# dataset = MovieLens()
dataset = Avazu()
# dataset = Titanic()
dataset.build_data()

dims = dataset.feature_dims
print("dataset dims", dims)

# model = LogisticRegression(dims)
model = FactorizationMachine(feature_dims=dims)
# model = FieldAwareFactorizationMachine(dims, embed_dim=4)
# model = WideAndDeepModel(dims, embed_dim=4, hidden_dims=[10,10,10])
# model = DeepFactorizationMachine(dims, embed_dim=4, hidden_dims=[10, 10, 10])
# model = NeuralFactorizationMachine(dims, embed_dim=4, hidden_dims=[10, 10, 10])
# model = FieldAwareNeuralFactorizationMachine(dims, embed_dim=4, hidden_dims=[10, 10, 10])

hyper_parameters = {
    "batch_size": 32,
    "device": "cpu",
    "learning_rate": 0.01,
    "weight_decay": 1e-6,
    "epochs": 30,
    "metrics": ["auc", "acc"],
}

trainer = Trainer(model, dataset, hyper_parameters)
trainer.train(dashboard_address="localhost:8081")
# trainer.save_model("checkpoints/test.pt")
