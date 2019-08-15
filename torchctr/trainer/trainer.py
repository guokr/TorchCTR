#!/usr/bin/env python
# encoding: utf-8

import os
import torch
import requests
from torch.utils.data import DataLoader
from torchctr.progressbar import ProgressBar
from torchctr.dashboard import MetricLogger


class Trainer:
    def __init__(self, model, dataset, param={}):
        self.model = model
        self.dataset = dataset
        self.param = self.build_param(param)
        self.trainer_setup = self.build_trainer()

    def build_param(self, param):
        print("| building parameters ...")
        default_param = {
            "batch_size": 128,
            "num_workers": 4,
            "device": "cpu",
            "learning_rate": 0.01,
            "weight_decay": 1e-6,
            "epochs": 10,
            "metrics": ["auc"],
        }

        for k, v in param.items():
            if k in default_param:
                default_param[k] = v

        return default_param

    def build_trainer(self):
        print("| building trainer ...")
        train_length = int(len(self.dataset) * 0.9)
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

        logger = MetricLogger()

        trainer_setup = {
            "train_loader": train_loader,
            "valid_loader": valid_loader,
            "criterion": criterion,
            "optimizer": optimizer,
            "logger": logger
        }

        return trainer_setup

    def train(self, dashboard_address=None):
        dashboard_status = False

        if dashboard_address is None:
            print("| Didn't find dashboard")
        else:
            try:
                requests.get(url="http://{}/ping".format(dashboard_address))
                dashboard_status = True
            except requests.exceptions.RequestException as e:
                print("| ERROR in Dashboard connection: {}".format(e))
                print("| Back to general training")

        print("| Start training ...")
        for e in range(self.param.get("epochs")):
            self.train_step(e + 1)
            self.valid_step()
            if dashboard_status is True:
                self.trainer_setup.get("logger").send(dashboard_address)

    def train_step(self, epoch):
        self.model.train()
        progress_bar = ProgressBar(
            self.trainer_setup.get("train_loader"),
            self.param.get("metrics"),
            desc="| Training {}/{}".format(epoch, self.param.get("epochs")),
        )

        for (fields, target) in progress_bar:
            fields, target = (
                fields.to(self.param.get("device")),
                target.to(self.param.get("device")),
            )
            y = self.model(fields)

            loss = self.trainer_setup.get("criterion")(y, target.float())
            self.model.zero_grad()
            loss.backward()
            self.trainer_setup.get("optimizer").step()

            progress_bar.eval(
                target.tolist(), y.tolist(), {"loss": loss.item()}
            )

        progress_bar.summarize()
        self.trainer_setup.get("logger").log(trace="train",
                                             stats=progress_bar.summary)


    def valid_step(self):
        self.model.eval()
        progress_bar = ProgressBar(
            self.trainer_setup.get("valid_loader"),
            self.param.get("metrics"),
            desc="| Validating",
        )

        for (fields, target) in progress_bar:
            fields, target = (
                fields.to(self.param.get("device")),
                target.to(self.param.get("device")),
            )
            y = self.model(fields)
            loss = self.trainer_setup.get("criterion")(y, target.float())

            progress_bar.eval(
                target.tolist(), y.tolist(), {"loss": loss.item()}
            )

        progress_bar.summarize()
        self.trainer_setup.get("logger").log(trace="validation",
                                             stats=progress_bar.summary)

    def save_model(self, file_fullpath):
        path, filename = os.path.split(file_fullpath)
        if os.path.exists(path) or path == "":
            torch.save(self.model, file_fullpath)
        else:
            print("| Didn't find dir, so we will create it")
            os.mkdir(path)
            torch.save(self.model, file_fullpath)
