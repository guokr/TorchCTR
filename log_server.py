#!/usr/bin/env python
# encoding: utf-8

from torchctr.dashboard import Dashboard

v = Dashboard(page_title="Hello TorchCTR", page_desc="I'm monitoring metrics ")
v.run()
