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
        divs = []
        for metric, trace_log in d.items():
            dash_data = []
            for trace, log_values in trace_log.items():
                dash_data.append(
                    {
                        "x": [i for i in range(1, len(log_values) + 1)],
                        "y": log_values,
                        "type": "line",
                        "name": "{}-{}".format(metric, trace),
                    }
                )

            divs.append(self.build_div(dash_data, title=metric))

        self.build_layout(divs)
        return ("", 204)

    def build_div(self, data, title):
        single_div = html.Div(
            dcc.Graph(
                id="{}".format(title),
                style={"width": "45vh", "display": "inline-block"},
                figure={"data": data, "layout": {"title": title}},
            )
        )
        return single_div

    def div_resize(self, divs):
        number_graphs = len(divs)
        percent_width = 100 // number_graphs
        for div in divs:
            div.children.style["width"] = "{}vh".format(percent_width)

        return divs

    def build_layout(self, divs=[]):
        if divs != []:
            divs = self.div_resize(divs)

        self.dash_app.layout = html.Div(
            [
                html.Div(
                    [
                        html.H1(
                            children=self.page_title,
                            style={"textAlign": "center"},
                        ),
                        html.Div(
                            children=self.page_desc,
                            style={"textAlign": "center"},
                        ),
                    ]
                ),
                html.Div(divs, style={"columnCount": len(divs)}),
            ]
        )

    def run(self):
        self.dash_app.run_server(
            debug=self.debug, host=self.host, port=self.port
        )
