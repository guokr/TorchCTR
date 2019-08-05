#!/usr/bin/env python
# encoding: utf-8

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(self, model, dataset, param={}):
        self.model = model
        self.dataset = dataset
        self.param = self.build_param(param)
        self.trainer_setup = self.build_trainer()

    def build_param(self, param):
        print("| building parameters ...")
        default_param = {
            "batch_size": 32,
            "num_workers": 4,
            "device": "cpu",
            "learning_rate": 0.01,
            "weight_decay": 1e-6,
            "epochs": 10,
            "print_interval": 1000,
        }

        for k, v in param:
            if k in default_param:
                default_param[k] = v

        return default_param

    def build_trainer(self):
        print("| building trainer ...")
        train_length = int(len(self.dataset) * 0.8)
        valid_length = len(self.dataset) - train_length

        train_dataset, valid_dataset = torch.utils.data.random_split(
            self.dataset, (train_length, valid_length)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.param.get("batch_size"),
            num_workers=self.param.get("num_workers"),
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.param.get("batch_size"),
            num_workers=self.param.get("num_workers"),
        )

        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.param.get("learning_rate"),
            weight_decay=self.param.get("weight_decay"),
        )

        trainer_setup = {
            "train_loader": train_loader,
            "valid_loader": valid_loader,
            "criterion": criterion,
            "optimizer": optimizer,
        }

        return trainer_setup

    def train(self):
        print("| start training ...")
        self.model.train()
        for e in range(self.param.get("epochs")):
            self.train_step()

    def train_step(self):
        total_loss = 0
        for i, (fields, target) in enumerate(
            tqdm(
                self.trainer_setup.get("train_loader"),
                smoothing=0,
                mininterval=1.0,
            )
        ):
            fields, target = (
                fields.to(self.param.get("device")),
                target.to(self.param.get("device")),
            )
            y = self.model(fields)
            loss = self.trainer_setup.get("criterion")(y, target.float())
            self.model.zero_grad()
            loss.backward()
            self.trainer_setup.get("optimizer").step()
            total_loss += loss.item()

            if (i + 1) % self.param.get("print_interval") == 0:
                print("loss:", total_loss / self.param.get("print_interval"))
                total_loss = 0
