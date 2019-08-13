#!/usr/bin/env python
# encoding: utf-8

import dash_html_components as html
import dash_core_components as dcc


def div_build(json_data):
    div_data = json_data["logs"]
    div_name = json_data["logger"]

    div_temp = []
    for metric, trace_log in div_data.items():
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

        div_temp.append(build_graph(dash_data, title=metric, div_name=div_name))
    div = build_div(div_temp, div_name)
    return div


def build_graph(data, title, div_name):
    single_div = html.Div(
        dcc.Graph(
            id="{}-{}".format(div_name, title),
            style={"width": "45vh", "display": "inline-block"},
            figure={"data": data, "layout": {"title": title}},
        )
    )
    return single_div


def div_resize(graphs):
    number_graphs = len(graphs)
    percent_width = 90 // number_graphs
    for graph in graphs:
        graph.children.style["width"] = "{}vh".format(percent_width)

    return graphs


def build_div(div, div_name):
    div = div_resize(div)

    div = [
        html.H2(children=div_name, style={"textAlign": "center"}),
        html.Div(div, style={"columnCount": len(div)}),
    ]
    return div_name, html.Div(div)
