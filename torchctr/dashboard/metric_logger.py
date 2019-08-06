#!/usr/bin/env python
# encoding: utf-8

import requests
from collections import defaultdict


class MetricLogger:
    def __init__(self):
        self.logs = defaultdict(lambda: defaultdict(list))

    def log(self, trace, stats):
        for metric, value in stats.items():
            self.logs[trace][metric].append(value)

    def send(self, dashboard_address):
        requests.post(url="http://{}/log".format(dashboard_address), json=self.logs)
