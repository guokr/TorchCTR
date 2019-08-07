#!/usr/bin/env python
# encoding: utf-8

from torchctr.dashboard import Dashboard

v = Dashboard(page_title="Hello TorchCTR", page_desc="I'm monitoring metrics",
              host="0.0.0.0", port=8081)
v.run()
