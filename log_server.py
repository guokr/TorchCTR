#!/usr/bin/env python
# encoding: utf-8

from torchctr.dashboard import Dashboard

v = Dashboard(page_title="Test Title", page_desc="----------")
v.load()
v.run()
