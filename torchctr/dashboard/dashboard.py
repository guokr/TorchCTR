#!/usr/bin/env python
# encoding: utf-8


import dash
import dash_core_components as dcc
import dash_html_components as html

from flask import Flask, request

external_stylesheets = ["http://codepen.io/chriddyp/pen/bWLwgP.css"]


class Dashboard:
    def __init__(
        self, page_title="", page_desc="", host="0.0.0.0", port=8080, debug=True
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

        self.build_routes()
        self.build_layout()

    def build_routes(self):
        self.flask_app.add_url_rule("/ping", "ping", self.ping, methods=["GET"])
        self.flask_app.add_url_rule("/log", "log", self.log, methods=["POST"])

    def ping(self):
        return "OK"

    def log(self):
        d = request.json
        dash_data = []
        for trace, value in d.items():
            dash_data.append(
                {
                    "x": [i for i in range(1, len(value["auc"]) + 1)],
                    "y": [i for i in value["auc"]],
                    "type": "line",
                    "name": trace,
                }
            )
        self.build_layout(data=dash_data)
        return ("", 204)

    def build_layout(self, data=[]):
        self.dash_app.layout = html.Div(
            children=[
                html.H1(children=self.page_title),
                html.Div(children=self.page_desc),
                dcc.Graph(
                    id="example-graph",
                    figure={"data": data, "layout": {"title": "AUC"}},
                ),
            ]
        )

    def run(self):
        self.dash_app.run_server(debug=self.debug, host=self.host, port=self.port)
