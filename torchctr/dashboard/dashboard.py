#!/usr/bin/env python
# encoding: utf-8


import dash
import dash_html_components as html
from collections import defaultdict

from flask import Flask, request
from .div_builder import div_build

external_stylesheets = ["http://codepen.io/chriddyp/pen/bWLwgP.css"]


class Dashboard:
    def __init__(
        self, page_title="Dashboard", page_desc="", host="0.0.0.0", port=8080, debug=False
    ):
        self.page_title = page_title
        self.page_desc = page_desc

        self.host = host
        self.port = port
        self.debug = debug
        self.flask_app = Flask(__name__)

        self.dash_app = dash.Dash(
            __name__,
            server=self.flask_app,
            external_stylesheets=external_stylesheets,
        )
        self.layout_dict = defaultdict(list)
        self.layout_dict["header"] = self.build_header()

        self.build_layout()
        self.build_routes()

    def build_layout(self):
        layout_contents = []
        for div_name, div_content in self.layout_dict.items():
            layout_contents.append(html.Div(div_content))

        self.dash_app.layout = html.Div(layout_contents)

    def build_header(self):
        header_div = [
            html.H1(children=self.page_title, style={"textAlign": "center"}),
            html.Div(children=self.page_desc, style={"textAlign": "center"}),
        ]

        return html.Div(header_div)

    def build_routes(self):
        self.flask_app.add_url_rule("/ping", "ping", self.ping, methods=["GET"])
        self.flask_app.add_url_rule("/log", "log", self.log, methods=["POST"])

    def ping(self):
        return "OK"

    def log(self):
        json_data = request.json
        div_name, div_content = div_build(json_data)
        self.layout_dict[div_name] = div_content
        self.build_layout()

        return ("", 204)

    def run(self):
        self.dash_app.run_server(
            debug=self.debug, host=self.host, port=self.port
        )
