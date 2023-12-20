#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import pickle


yearly_dataframes = []
for y in range(2016, 2022):
    yearly_dataframes.append(pickle.load(open( str(y)+".p", "rb" )))

df = pd.concat(yearly_dataframes)
# fig = px.scatter_geo(df, locations="location",
#                 size="size", # lifeExp is a column of gapminder
#                 hover_name="description", # column to add to hover information
#                 projection="orthographic",animation_frame="year")

fig = px.choropleth(df, locations="location",
                     color="size",
                     color_continuous_scale = "Magenta",
                     hover_name="description",
                     projection="orthographic",
                     animation_frame="year")


app = Dash(__name__)
app.layout = html.Div([
    html.H4('World News'),
    html.H4('Text'),
    dcc.Graph(figure=fig,style={"margin":0,"height":"90vh"})
])




app.run_server(debug=True)