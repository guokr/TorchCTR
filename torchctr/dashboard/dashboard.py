#!/usr/bin/env python
# encoding: utf-8

import dash
import dash_core_components as dcc
import dash_html_components as html

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

class Dashboard():
    def __init__(self, page_title="", page_desc=""):
        self.page_title = page_title
        self.page_desc = page_desc
        self.dash_app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    def load(self, path=""):
        self.dash_app.layout = html.Div(children=[
            html.H1(children='Hello Dash'),

            html.Div(children='''
                Dash: A web application framework for Python.
            '''),

            dcc.Graph(
                id='example-graph',
                figure={
                    'data': [
                        {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                        {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
                    ],
                    'layout': {
                        'title': 'Dash Data Visualization'
                    }
                }
            )
        ])

    def run(self, debug=False, host="0.0.0.0", port=8080):
        self.dash_app.run_server(debug=debug,
                                 host=host,
                                 port=port)

