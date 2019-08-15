#!/usr/bin/env python
# encoding: utf-8

from torchctr.datasets import Titanic
from torchctr.models import FactorizationMachine
from torchctr.trainer import Trainer

# Load the well known Kaggle Titanic dataset
dataset = Titanic()
dataset.build_data()

# Now we build the famous Factorization Machine model
model = FactorizationMachine(feature_dims=dataset.feature_dims)

# Also we need to build a trainer for our model and dasaset
trainer = Trainer(model, dataset)

# At last we train it
trainer.train()

# save the model to local disk
trainer.save_model("checkpoints/test.pt")
