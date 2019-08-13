#!/usr/bin/env python
# encoding: utf-8

import requests
from datetime import datetime
from collections import defaultdict


class MetricLogger:
    def __init__(self, name=str(datetime.now())):
        self.logs = defaultdict(lambda: defaultdict(list))
        self.name = name

    def log(self, trace, stats):
        for metric, value in stats.items():
            self.logs[metric][trace].append(value)

    def send(self, dashboard_address):
        requests.post(url="http://{}/log".format(dashboard_address), json={"logs": self.logs, "logger": self.name})
