#!/usr/bin/env python
# encoding: utf-8

from collections import defaultdict


class MetricLogger:
    def __init__(self):
        self.logs = defaultdict(list)

    def log(self, trace, stats):
        self.logs[trace].append(stats)
