from vrae.vrae import VRAE
# from vrae.utils import *
import numpy as np
import torch
import plotly
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import yaml
from attrdict import AttrDict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
import dash
import pandas as pd
import plotly.graph_objs as go
import dash_core_components as dcc
import dash_html_components as html
labels = np.load("data_for_dash/EGC_labels.npy")
y_pred_total = np.load("data_for_dash/EGC_y_pred_total.npy")
z_run_sep = np.load("data_for_dash/EGC_z_run_sep.npy")
test_data_sample = np.load("data_for_dash/EGC_test_data_sample.npy")
X_decoded_sample = np.load("data_for_dash/EGC_X_decoded_sample.npy")
[y_pred_sc_model, y_pred_kmean, y_pred_meanshift_model, y_pred_GMM_model, y_pred_DBSCAN_model] = y_pred_total

labels_unique = np.unique(labels).astype(np.int32)
labels_unique_list = labels_unique.tolist()
y_pred_unique = np.unique(y_pred_kmean).astype(np.int32)
y_pred_unique_unique_list = y_pred_unique.tolist()
y_pred_sc_model_unique = np.unique(y_pred_sc_model).astype(np.int32)
y_pred_sc_model_unique_unique_list = y_pred_sc_model_unique.tolist()

app = dash.Dash()

# ## plotly
test_data_sample_time_steps = np.arange(0, test_data_sample.shape[0])
fig = go.Figure()
fig.add_trace(go.Scatter(x=test_data_sample_time_steps, y=np.squeeze(test_data_sample),
                         mode='lines+markers',
                         name='test_data_sample'))
fig.add_trace(go.Scatter(x=test_data_sample_time_steps, y=X_decoded_sample,
                         mode='lines+markers',
                         name='X_decoded_sample'))



fig_names = ["y_pred_GMM_model", "y_pred_sc_model", "y_pred_kmean"]

app.layout = html.Div(children=[
    # Dropdown
    # html.Div([
    #     html.H1(children='Dropdown test'),
    #     EGC200_fig_dropdown
    # ]),
    # dcc.Graph(id='fig_plot'),
    # All elements from the top of the page

    # Dropdown
    html.Div([
        html.H2(children='EGC200_DROPDOWM'),
        dcc.Dropdown(
            id='EGC200_fig_dropdown',
            options=[{'label': x, 'value': x} for x in fig_names],
            value=None
        ),
        dcc.Graph(id='fig_plot'),
    ]),

    #
    html.Div([
        html.H2(children='LSTM_VAE CLUSTERING y_pred_GMM_model'),

        html.Div(children='''
    Clustering Result with y_pred_GMM_model
'''),

        dcc.Graph(
            id='y_pred_GMM_model',
            figure={
                'data': [
                    go.Scatter(
                        x=list(z_run_sep[np.where(y_pred_GMM_model == i)[0], 0]),
                        y=list(z_run_sep[np.where(y_pred_GMM_model == i)[0], 1]),
                        text="class" + str(i),
                        mode='markers',
                        opacity=0.7,
                        marker={
                            'size': 15,
                            'line': {'width': 0.5, 'color': 'white'}
                        },
                        name=str(i)
                    ) for i in y_pred_sc_model_unique_unique_list
                ],
            }
        ),
    ]),
    # y_pred_sc_model
    html.Div([
        html.H2(children='LSTM_VAE CLUSTERING y_pred_sc_model'),

        html.Div(children='''
     Clustering Result with y_pred_sc_model
 '''),

        dcc.Graph(
            id='y_pred_sc_model',
            figure={
                'data': [
                    go.Scatter(
                        x=list(z_run_sep[np.where(y_pred_sc_model == i)[0], 0]),
                        y=list(z_run_sep[np.where(y_pred_sc_model == i)[0], 1]),
                        text="class" + str(i),
                        mode='markers',
                        opacity=0.7,
                        marker={
                            'size': 15,
                            'line': {'width': 0.5, 'color': 'white'}
                        },
                        name=str(i)
                    ) for i in y_pred_sc_model_unique_unique_list
                ],
            }
        ),
    ]),
    # KMEAN
    html.Div([
        html.H1(children='LSTM_VAE CLUSTERING y_pred_kmean'),

        html.Div(children='''
            Clustering Result with y_pred_kmean
        '''),

        dcc.Graph(
            id='y_pred_kmean',
            figure={
                'data': [
                    go.Scatter(
                        x=list(z_run_sep[np.where(y_pred_kmean == i)[0], 0]),
                        y=list(z_run_sep[np.where(y_pred_kmean == i)[0], 1]),
                        text="class" + str(i),
                        mode='markers',
                        opacity=0.7,
                        marker={
                            'size': 15,
                            'line': {'width': 0.5, 'color': 'white'}
                        },
                        name=str(i)
                    ) for i in y_pred_unique
                ],
            }
        ),
    ]),
    # New Div for all elements in the new 'row' of the page
    html.Div([
        html.H1(children='LSTM_VAE GROUNDTRUTH'),

        html.Div(children='''
            ground truth of data
        '''),

        dcc.Graph(
            id='ground truth',
            figure={
                'data': [
                    go.Scatter(
                        x=list(z_run_sep[np.where(labels == i)[0], 0]),
                        y=list(z_run_sep[np.where(labels == i)[0], 1]),
                        text="class" + str(i),
                        mode='markers',
                        opacity=0.7,
                        marker={
                            'size': 15,
                            'line': {'width': 0.5, 'color': 'white'}
                        },
                        name=str(i)
                    ) for i in labels_unique_list
                ],

            }
        ),
    ]),
    html.Div([
        html.H1(children='DECODED'),

        html.Div(children='''
            ground truth and decoded trajectory
        '''),
        dcc.Graph(figure=fig, id='DECODED')
    ]),
])


@app.callback(
    dash.dependencies.Output('fig_plot', 'figure'),
    [dash.dependencies.Input('EGC200_fig_dropdown', 'value')])
def update_output(EGC200_fig_dropdown):
    return name_to_figure(EGC200_fig_dropdown)


def name_to_figure(EGC200_fig_dropdown):
    figure = go.Figure()
    if EGC200_fig_dropdown == 'y_pred_GMM_model':
        return {'data': [
            go.Scatter(
                x=list(z_run_sep[np.where(y_pred_GMM_model == i)[0], 0]),
                y=list(z_run_sep[np.where(y_pred_GMM_model == i)[0], 1]),
                text="class" + str(i),
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 15,
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name=str(i)
            ) for i in y_pred_sc_model_unique_unique_list
        ]}

    elif EGC200_fig_dropdown == 'y_pred_sc_model':
        return {
            'data': [
                go.Scatter(
                    x=list(z_run_sep[np.where(y_pred_sc_model == i)[0], 0]),
                    y=list(z_run_sep[np.where(y_pred_sc_model == i)[0], 1]),
                    text="class" + str(i),
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name=str(i)
                ) for i in y_pred_sc_model_unique_unique_list
            ],
        }
    elif EGC200_fig_dropdown == 'y_pred_kmean':
        return {'data': [
            go.Scatter(
                x=list(z_run_sep[np.where(y_pred_kmean == i)[0], 0]),
                y=list(z_run_sep[np.where(y_pred_kmean == i)[0], 1]),
                text="class" + str(i),
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 15,
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name=str(i)
            ) for i in y_pred_unique
        ]}




if __name__ == '__main__':
    app.run_server(debug=True)
