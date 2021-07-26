from vrae.vrae import VRAE
from vrae.utils import *
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

############# ECG200200
ECG200_labels = np.load("data_for_dash/ECG200_labels.npy")
ECG200_y_pred_total = np.load("data_for_dash/ECG200_y_pred_total.npy")
ECG200_z_run_sep = np.load("data_for_dash/ECG200_z_run_sep.npy")
ECG200_test_data_sample = np.load("data_for_dash/ECG200_test_data_sample.npy")
ECG200_X_decoded_sample = np.load("data_for_dash/ECG200_X_decoded_sample.npy")
[ECG200_y_pred_sc_model, ECG200_y_pred_kmean, ECG200_y_pred_meanshift_model, ECG200_y_pred_GMM_model,
 ECG200_y_pred_DBSCAN_model] = ECG200_y_pred_total

ECG200_labels_unique = np.unique(ECG200_labels).astype(np.int32)
ECG200_labels_unique_list = ECG200_labels_unique.tolist()
ECG200_y_kmean_pred_unique = np.unique(ECG200_y_pred_kmean).astype(np.int32)
ECG200_y_kmean_pred_unique_list = ECG200_y_kmean_pred_unique.tolist()
ECG200_y_pred_sc_model_unique = np.unique(ECG200_y_pred_sc_model).astype(np.int32)
ECG200_y_pred_sc_model_unique_list = ECG200_y_pred_sc_model_unique.tolist()
# ## plotly
ECG200_test_data_sample_time_steps = np.arange(0, ECG200_test_data_sample.shape[0])
# ECG200_test_data_sample_time_steps = np.arange(0, ECG200_test_data_sample.shape[0])

############# ECG5000
ECG5000_labels = np.load("data_for_dash/ECG5000_labels.npy")
ECG5000_y_pred_total = np.load("data_for_dash/ECG5000_y_pred_total.npy")
ECG5000_z_run_sep = np.load("data_for_dash/ECG5000_z_run_sep.npy")
ECG5000_test_data_sample = np.load("data_for_dash/ECG5000_test_data_sample.npy")
ECG5000_X_decoded_sample = np.load("data_for_dash/ECG5000_X_decoded_sample.npy")
[ECG5000_y_pred_sc_model, ECG5000_y_pred_kmean, ECG5000_y_pred_meanshift_model, ECG5000_y_pred_GMM_model,
 ECG5000_y_pred_DBSCAN_model] = ECG5000_y_pred_total

ECG5000_labels_unique = np.unique(ECG5000_labels).astype(np.int32)
ECG5000_labels_unique_list = ECG5000_labels_unique.tolist()
ECG5000_y_kmean_pred_unique = np.unique(ECG5000_y_pred_kmean).astype(np.int32)
ECG5000_y_kmean_pred_unique_list = ECG5000_y_kmean_pred_unique.tolist()
ECG5000_y_pred_sc_model_unique = np.unique(ECG5000_y_pred_sc_model).astype(np.int32)
ECG5000_y_pred_sc_model_unique_list = ECG5000_y_pred_sc_model_unique.tolist()
# ## plotly
ECG5000_test_data_sample_time_steps = np.arange(0, ECG5000_test_data_sample.shape[0])
# ECG5000_test_data_sample_time_steps = np.arange(0, ECG5000_test_data_sample.shape[0])

############ FASHIONMNIST
FASHIONMNIST_labels = np.load("data_for_dash/FASHIONMNIST_labels.npy")
FASHIONMNIST_y_pred_total = np.load("data_for_dash/FASHIONMNIST_y_pred_total.npy")
FASHIONMNIST_z_run_sep = np.load("data_for_dash/FASHIONMNIST_z_run_sep.npy")
[FASHIONMNIST_y_pred_sc_model, FASHIONMNIST_y_pred_kmean, FASHIONMNIST_y_pred_meanshift_model,
 FASHIONMNIST_y_pred_GMM_model, FASHIONMNIST_y_pred_DBSCAN_model] = FASHIONMNIST_y_pred_total
FASHIONMNIST_labels_unique = np.unique(FASHIONMNIST_labels).astype(np.int32)
FASHIONMNIST_labels_unique_list = FASHIONMNIST_labels_unique.tolist()
FASHIONMNIST_y_pred_unique = np.unique(FASHIONMNIST_y_pred_kmean).astype(np.int32)
FASHIONMNIST_y_pred_unique_unique_list = FASHIONMNIST_y_pred_unique.tolist()
FASHIONMNIST_y_pred_sc_model_unique = np.unique(FASHIONMNIST_y_pred_sc_model).astype(np.int32)
FASHIONMNIST_y_pred_sc_model_unique_list = FASHIONMNIST_y_pred_sc_model_unique.tolist()

############ MNIST
MNIST_labels = np.load("data_for_dash/MNIST_labels.npy")
MNIST_y_pred_total = np.load("data_for_dash/MNIST_y_pred_total.npy")
MNIST_z_run_sep = np.load("data_for_dash/MNIST_z_run_sep.npy")
[MNIST_y_pred_sc_model, MNIST_y_pred_kmean, MNIST_y_pred_meanshift_model,
 MNIST_y_pred_GMM_model, MNIST_y_pred_DBSCAN_model] = MNIST_y_pred_total
MNIST_labels_unique = np.unique(MNIST_labels).astype(np.int32)
MNIST_labels_unique_list = MNIST_labels_unique.tolist()
MNIST_y_pred_unique = np.unique(MNIST_y_pred_kmean).astype(np.int32)
MNIST_y_pred_unique_unique_list = MNIST_y_pred_unique.tolist()
MNIST_y_pred_sc_model_unique = np.unique(MNIST_y_pred_sc_model).astype(np.int32)
MNIST_y_pred_sc_model_unique_list = MNIST_y_pred_sc_model_unique.tolist()

# ############ CIFAR10
CIFAR10_labels = np.load("data_for_dash/CIFAR10_labels.npy")
CIFAR10_y_pred_total = np.load("data_for_dash/CIFAR10_y_pred_total.npy")
CIFAR10_z_run_sep = np.load("data_for_dash/CIFAR10_z_run_sep.npy")
[CIFAR10_y_pred_sc_model, CIFAR10_y_pred_kmean, CIFAR10_y_pred_meanshift_model,
 CIFAR10_y_pred_GMM_model, CIFAR10_y_pred_DBSCAN_model] = CIFAR10_y_pred_total
CIFAR10_labels_unique = np.unique(CIFAR10_labels).astype(np.int32)
CIFAR10_labels_unique_list = CIFAR10_labels_unique.tolist()
CIFAR10_y_pred_unique = np.unique(CIFAR10_y_pred_kmean).astype(np.int32)
CIFAR10_y_pred_unique_unique_list = CIFAR10_y_pred_unique.tolist()
CIFAR10_y_pred_sc_model_unique = np.unique(CIFAR10_y_pred_sc_model).astype(np.int32)
CIFAR10_y_pred_sc_model_unique_list = CIFAR10_y_pred_sc_model_unique.tolist()

############ ElectricDevices
ElectricDevices_labels = np.load("data_for_dash/ElectricDevices_labels.npy")
ElectricDevices_y_pred_total = np.load("data_for_dash/ElectricDevices_y_pred_total.npy")
ElectricDevices_z_run_sep = np.load("data_for_dash/ElectricDevices_z_run_sep.npy")
ElectricDevices_test_data_sample = np.load("data_for_dash/ElectricDevices_test_data_sample.npy")
ElectricDevices_X_decoded_sample = np.load("data_for_dash/ElectricDevices_X_decoded_sample.npy")
[ElectricDevices_y_pred_sc_model, ElectricDevices_y_pred_kmean, ElectricDevices_y_pred_meanshift_model,
 ElectricDevices_y_pred_GMM_model, ElectricDevices_y_pred_DBSCAN_model] = ElectricDevices_y_pred_total
ElectricDevices_labels_unique = np.unique(ElectricDevices_labels).astype(np.int32)
ElectricDevices_labels_unique_list = ElectricDevices_labels_unique.tolist()
ElectricDevices_y_pred_unique = np.unique(ElectricDevices_y_pred_kmean).astype(np.int32)
ElectricDevices_y_pred_unique_unique_list = ElectricDevices_y_pred_unique.tolist()
ElectricDevices_y_pred_sc_model_unique = np.unique(ElectricDevices_y_pred_sc_model).astype(np.int32)
ElectricDevices_y_pred_sc_model_unique_list = ElectricDevices_y_pred_sc_model_unique.tolist()
# ## plotly
ElectricDevices_test_data_sample_time_steps = np.arange(0, ElectricDevices_test_data_sample.shape[0])
# ElectricDevices_test_data_sample_time_steps = np.arange(0, ElectricDevices_test_data_sample.shape[0])

app = dash.Dash()
app.title = 'My Title'
# # ## plotly
# ECG_test_data_sample_time_steps = np.arange(0, ECG_test_data_sample.shape[0])
# ECG_test_data_sample_time_steps = np.arange(0, ECG_test_data_sample.shape[0])
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=ECG_test_data_sample_time_steps, y=np.squeeze(ECG_test_data_sample),
#                          mode='lines+markers',
#                          name='test_data_sample'))
# fig.add_trace(go.Scatter(x=ECG_test_data_sample_time_steps, y=ECG_X_decoded_sample,
#                          mode='lines+markers',
#                          name='X_decoded_sample'))


fig_names = ["ECG200_y_pred_GMM_model", "ECG200_y_pred_sc_model", "ECG200_y_pred_kmean",
             "ECG5000_y_pred_GMM_model", "ECG5000_y_pred_sc_model", "ECG5000_y_pred_kmean",
             "FASHIONMNIST_y_pred_GMM_model", "FASHIONMNIST_y_pred_sc_model", "FASHIONMNIST_y_pred_kmean",
             "MNIST_y_pred_GMM_model", "MNIST_y_pred_sc_model", "MNIST_y_pred_kmean",
             "ElectricDevices_y_pred_GMM_model", "ElectricDevices_y_pred_sc_model", "ElectricDevices_y_pred_kmean",
             "CIFAR10_y_pred_GMM_model", "CIFAR10_y_pred_sc_model", "CIFAR10_y_pred_kmean",]

GroundTruth_fig_names = ["ECG200", "ECG5000", "FASHIONMNIST", "MNIST", "ElectricDevices", "CIFAR10"]
app.layout = html.Div(children=[
    html.H4('Variational Autoencoder with LSTM for Time Series Analysis'),
    html.Center(
        'Variational Autoencoders (VAE) is one powerful tool to extract interpretable features from time series. By condensing the data to a smaller latent space it avoids the curse of dimensionality and sensitivity to noisy input data. The scatter plot below are ground truth of the TS dataset in latent space in comparison with the clustering results with different clustering methods for chosen dataset in latent space. The reconstructed result of chosen dataset is shown at the bottom.'),

    # Dropdown
    # html.Div([
    #     html.H1(children='Dropdown test'),
    #     fig_dropdown
    # ]),
    # dcc.Graph(id='fig_plot'),
    # All elements from the top of the page

    # Dropdown
    html.Div([
        html.H2(children='Clustering'),
        dcc.Dropdown(
            id='fig_dropdown',
            options=[{'label': x, 'value': x} for x in fig_names],
            value=None
        ),
        dcc.Graph(id='fig_plot'),
    ], style={'display': 'inline-block', 'width': '49%'}),
    html.Div([
        html.H6(children='GroundTruth'),
        dcc.Dropdown(
            id='GroundTruth_fig_dropdown',
            options=[{'label': x, 'value': x} for x in GroundTruth_fig_names],
            value=None
        ),
        dcc.Graph(id='GroundTruth_fig_plot',
                  ),
    ], style={'display': 'inline-block', 'width': '49%'}),

    # # New Div for all elements in the new 'row' of the page
    # html.Div([
    #     html.H1(children='LSTM_VAE GROUNDTRUTH'),
    #
    #     html.Div(children='''
    #         ground truth of data
    #     '''),
    #
    #     dcc.Graph(
    #         id='ground truth',
    #         figure={
    #             'data': [
    #                 go.Scatter(
    #                     x=list(ECG_z_run_sep[np.where(ECG_labels == i)[0], 0]),
    #                     y=list(ECG_z_run_sep[np.where(ECG_labels == i)[0], 1]),
    #                     text="class" + str(i),
    #                     mode='markers',
    #                     opacity=0.7,
    #                     marker={
    #                         'size': 15,
    #                         'line': {'width': 0.5, 'color': 'white'}
    #                     },
    #                     name=str(i)
    #                 ) for i in ECG_labels_unique_list
    #             ],
    #
    #         }
    #     ),
    # ]),
    # html.Div([
    #     html.H1(children='DECODED'),
    #
    #     html.Div(children='''
    #         ground truth and decoded trajectory
    #     '''),
    #     dcc.Graph(figure=fig, id='DECODED')
    # ]),
    html.Div([
        html.H1(children='DECODED'),
        dcc.Dropdown(
            id='decoded_fig_dropdown',
            options=[{'label': x, 'value': x} for x in GroundTruth_fig_names],
            value=None
        ),
        dcc.Graph(id='decoded_fig_plot',
                  ),
    ], style={'display': 'inline-block', 'width': '49%'}),
])


##############  Predict
@app.callback(
    dash.dependencies.Output('fig_plot', 'figure'),
    [dash.dependencies.Input('fig_dropdown', 'value'),
     dash.dependencies.Input('GroundTruth_fig_dropdown', 'value')])
def update_output(fig_dropdown, GroundTruth_fig_dropdown):
    return name_to_figure(fig_dropdown, GroundTruth_fig_dropdown)


def name_to_figure(fig_dropdown, GroundTruth_fig_dropdown):
    # figure = go.Figure()
    if GroundTruth_fig_dropdown == 'ECG200' and ("ECG200" not in fig_dropdown):
        return {'data': [
            go.Scatter(
                x=list(ECG200_z_run_sep[np.where(ECG200_y_pred_kmean == i)[0], 0]),
                y=list(ECG200_z_run_sep[np.where(ECG200_y_pred_kmean == i)[0], 1]),
                text="class" + str(i),
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 15,
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name=str(i)
            ) for i in ECG200_y_kmean_pred_unique
        ]}
    elif GroundTruth_fig_dropdown == 'ECG5000' and ("ECG5000" not in fig_dropdown):
        return {'data': [
            go.Scatter(
                x=list(ECG200_z_run_sep[np.where(FASHIONMNIST_y_pred_kmean == i)[0], 0]),
                y=list(ECG200_z_run_sep[np.where(FASHIONMNIST_y_pred_kmean == i)[0], 1]),
                text="class" + str(i),
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 15,
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name=str(i)
            ) for i in ECG5000_y_kmean_pred_unique
        ]}
    elif GroundTruth_fig_dropdown == 'FASHIONMNIST' and ("FASHIONMNIST" not in fig_dropdown):
        return {'data': [
            go.Scatter(
                x=list(ECG200_z_run_sep[np.where(FASHIONMNIST_y_pred_kmean == i)[0], 0]),
                y=list(ECG200_z_run_sep[np.where(FASHIONMNIST_y_pred_kmean == i)[0], 1]),
                text="class" + str(i),
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 15,
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name=str(i)
            ) for i in FASHIONMNIST_y_pred_unique
        ]}
    elif GroundTruth_fig_dropdown == 'MNIST' and ("MNIST" not in fig_dropdown):
        return {'data': [
            go.Scatter(
                x=list(ECG200_z_run_sep[np.where(MNIST_y_pred_kmean == i)[0], 0]),
                y=list(ECG200_z_run_sep[np.where(MNIST_y_pred_kmean == i)[0], 1]),
                text="class" + str(i),
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 15,
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name=str(i)
            ) for i in MNIST_y_pred_unique
        ]}
    elif GroundTruth_fig_dropdown == 'ElectricDevices' and ("ElectricDevices" not in fig_dropdown):
        return {'data': [
            go.Scatter(
                x=list(ElectricDevices_z_run_sep[np.where(ElectricDevices_y_pred_kmean == i)[0], 0]),
                y=list(ElectricDevices_z_run_sep[np.where(ElectricDevices_y_pred_kmean == i)[0], 1]),
                text="class" + str(i),
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 15,
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name=str(i)
            ) for i in ElectricDevices_y_pred_unique
        ]}

    elif GroundTruth_fig_dropdown == 'CIFAR10' and ("CIFAR10" not in fig_dropdown):
        return {'data': [
            go.Scatter(
                x=list(CIFAR10_z_run_sep[np.where(CIFAR10_y_pred_kmean == i)[0], 0]),
                y=list(CIFAR10_z_run_sep[np.where(CIFAR10_y_pred_kmean == i)[0], 1]),
                text="class" + str(i),
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 15,
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name=str(i)
            ) for i in CIFAR10_y_pred_unique
        ]}

    elif fig_dropdown == 'ECG200_y_pred_GMM_model':
        return {'data': [
            go.Scatter(
                x=list(ECG200_z_run_sep[np.where(ECG200_y_pred_GMM_model == i)[0], 0]),
                y=list(ECG200_z_run_sep[np.where(ECG200_y_pred_GMM_model == i)[0], 1]),
                text="class" + str(i),
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 15,
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name=str(i)
            ) for i in ECG200_y_pred_sc_model_unique_list
        ]}

    elif fig_dropdown == 'ECG200_y_pred_sc_model':
        return {
            'data': [
                go.Scatter(
                    x=list(ECG200_z_run_sep[np.where(ECG200_y_pred_sc_model == i)[0], 0]),
                    y=list(ECG200_z_run_sep[np.where(ECG200_y_pred_sc_model == i)[0], 1]),
                    text="class" + str(i),
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name=str(i)
                ) for i in ECG200_y_pred_sc_model_unique_list
            ],
        }
    elif fig_dropdown == 'ECG200_y_pred_kmean':
        return {'data': [
            go.Scatter(
                x=list(ECG200_z_run_sep[np.where(ECG200_y_pred_kmean == i)[0], 0]),
                y=list(ECG200_z_run_sep[np.where(ECG200_y_pred_kmean == i)[0], 1]),
                text="class" + str(i),
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 15,
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name=str(i)
            ) for i in ECG200_y_kmean_pred_unique
        ]}




    elif fig_dropdown == 'ECG5000_y_pred_GMM_model':
        return {'data': [
            go.Scatter(
                x=list(ECG5000_z_run_sep[np.where(ECG5000_y_pred_GMM_model == i)[0], 0]),
                y=list(ECG5000_z_run_sep[np.where(ECG5000_y_pred_GMM_model == i)[0], 1]),
                text="class" + str(i),
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 15,
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name=str(i)
            ) for i in ECG5000_y_pred_sc_model_unique_list
        ]}

    elif fig_dropdown == 'ECG5000_y_pred_sc_model':
        return {
            'data': [
                go.Scatter(
                    x=list(ECG5000_z_run_sep[np.where(ECG5000_y_pred_sc_model == i)[0], 0]),
                    y=list(ECG5000_z_run_sep[np.where(ECG5000_y_pred_sc_model == i)[0], 1]),
                    text="class" + str(i),
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name=str(i)
                ) for i in ECG5000_y_pred_sc_model_unique_list
            ],
        }
    elif fig_dropdown == 'ECG5000_y_pred_kmean':
        return {'data': [
            go.Scatter(
                x=list(ECG5000_z_run_sep[np.where(ECG5000_y_pred_kmean == i)[0], 0]),
                y=list(ECG5000_z_run_sep[np.where(ECG5000_y_pred_kmean == i)[0], 1]),
                text="class" + str(i),
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 15,
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name=str(i)
            ) for i in ECG5000_y_kmean_pred_unique
        ]}



    elif fig_dropdown == 'FASHIONMNIST_y_pred_GMM_model':
        return {'data': [
            go.Scatter(
                x=list(FASHIONMNIST_z_run_sep[np.where(FASHIONMNIST_y_pred_GMM_model == i)[0], 0]),
                y=list(FASHIONMNIST_z_run_sep[np.where(FASHIONMNIST_y_pred_GMM_model == i)[0], 1]),
                text="class" + str(i),
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 15,
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name=str(i)
            ) for i in FASHIONMNIST_y_pred_sc_model_unique_list
        ]}

    elif fig_dropdown == 'FASHIONMNIST_y_pred_sc_model':
        return {
            'data': [
                go.Scatter(
                    x=list(FASHIONMNIST_z_run_sep[np.where(FASHIONMNIST_y_pred_sc_model == i)[0], 0]),
                    y=list(FASHIONMNIST_z_run_sep[np.where(FASHIONMNIST_y_pred_sc_model == i)[0], 1]),
                    text="class" + str(i),
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name=str(i)
                ) for i in FASHIONMNIST_y_pred_sc_model_unique_list
            ],
        }
    elif fig_dropdown == 'FASHIONMNIST_y_pred_kmean':
        return {'data': [
            go.Scatter(
                x=list(FASHIONMNIST_z_run_sep[np.where(FASHIONMNIST_y_pred_kmean == i)[0], 0]),
                y=list(FASHIONMNIST_z_run_sep[np.where(FASHIONMNIST_y_pred_kmean == i)[0], 1]),
                text="class" + str(i),
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 15,
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name=str(i)
            ) for i in FASHIONMNIST_y_pred_unique
        ]}



    elif fig_dropdown == 'MNIST_y_pred_GMM_model':
        return {'data': [
            go.Scatter(
                x=list(MNIST_z_run_sep[np.where(MNIST_y_pred_GMM_model == i)[0], 0]),
                y=list(MNIST_z_run_sep[np.where(MNIST_y_pred_GMM_model == i)[0], 1]),
                text="class" + str(i),
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 15,
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name=str(i)
            ) for i in MNIST_y_pred_sc_model_unique_list
        ]}

    elif fig_dropdown == 'MNIST_y_pred_sc_model':
        return {
            'data': [
                go.Scatter(
                    x=list(MNIST_z_run_sep[np.where(MNIST_y_pred_sc_model == i)[0], 0]),
                    y=list(MNIST_z_run_sep[np.where(MNIST_y_pred_sc_model == i)[0], 1]),
                    text="class" + str(i),
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name=str(i)
                ) for i in MNIST_y_pred_sc_model_unique_list
            ],
        }
    elif fig_dropdown == 'MNIST_y_pred_kmean':
        return {'data': [
            go.Scatter(
                x=list(MNIST_z_run_sep[np.where(MNIST_y_pred_kmean == i)[0], 0]),
                y=list(MNIST_z_run_sep[np.where(MNIST_y_pred_kmean == i)[0], 1]),
                text="class" + str(i),
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 15,
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name=str(i)
            ) for i in MNIST_y_pred_unique
        ]}



    elif fig_dropdown == ' CIFAR10_y_pred_GMM_model':
        return {'data': [
            go.Scatter(
                x=list( CIFAR10_z_run_sep[np.where( CIFAR10_y_pred_GMM_model == i)[0], 0]),
                y=list( CIFAR10_z_run_sep[np.where( CIFAR10_y_pred_GMM_model == i)[0], 1]),
                text="class" + str(i),
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 15,
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name=str(i)
            ) for i in  CIFAR10_y_pred_sc_model_unique_list
        ]}

    elif fig_dropdown == ' CIFAR10_y_pred_sc_model':
        return {
            'data': [
                go.Scatter(
                    x=list( CIFAR10_z_run_sep[np.where( CIFAR10_y_pred_sc_model == i)[0], 0]),
                    y=list( CIFAR10_z_run_sep[np.where( CIFAR10_y_pred_sc_model == i)[0], 1]),
                    text="class" + str(i),
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name=str(i)
                ) for i in  CIFAR10_y_pred_sc_model_unique_list
            ],
        }
    elif fig_dropdown == ' CIFAR10_y_pred_kmean':
        return {'data': [
            go.Scatter(
                x=list( CIFAR10_z_run_sep[np.where( CIFAR10_y_pred_kmean == i)[0], 0]),
                y=list( CIFAR10_z_run_sep[np.where( CIFAR10_y_pred_kmean == i)[0], 1]),
                text="class" + str(i),
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 15,
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name=str(i)
            ) for i in  CIFAR10_y_pred_unique
        ]}




    elif fig_dropdown == 'ElectricDevices_y_pred_GMM_model':
        return {'data': [
            go.Scatter(
                x=list(ElectricDevices_z_run_sep[np.where(ElectricDevices_y_pred_GMM_model == i)[0], 0]),
                y=list(ElectricDevices_z_run_sep[np.where(ElectricDevices_y_pred_GMM_model == i)[0], 1]),
                text="class" + str(i),
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 15,
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name=str(i)
            ) for i in ElectricDevices_y_pred_sc_model_unique_list
        ]}

    elif fig_dropdown == 'ElectricDevices_y_pred_sc_model':
        return {
            'data': [
                go.Scatter(
                    x=list(ElectricDevices_z_run_sep[np.where(ElectricDevices_y_pred_sc_model == i)[0], 0]),
                    y=list(ElectricDevices_z_run_sep[np.where(ElectricDevices_y_pred_sc_model == i)[0], 1]),
                    text="class" + str(i),
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name=str(i)
                ) for i in ElectricDevices_y_pred_sc_model_unique_list
            ],
        }
    elif fig_dropdown == 'ElectricDevices_y_pred_kmean':
        return {'data': [
            go.Scatter(
                x=list(ElectricDevices_z_run_sep[np.where(ElectricDevices_y_pred_kmean == i)[0], 0]),
                y=list(ElectricDevices_z_run_sep[np.where(ElectricDevices_y_pred_kmean == i)[0], 1]),
                text="class" + str(i),
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 15,
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name=str(i)
            ) for i in ElectricDevices_y_pred_unique
        ]}



    else:
        return {}


###################### GroundTruth
@app.callback(
    dash.dependencies.Output('GroundTruth_fig_plot', 'figure'),
    [dash.dependencies.Input('GroundTruth_fig_dropdown', 'value'),
     dash.dependencies.Input('fig_dropdown', 'value')])
def GroundTruth_update_output(GroundTruth_fig_dropdown, fig_dropdown):
    return GroundTruth_name_to_figure(GroundTruth_fig_dropdown, fig_dropdown)


def GroundTruth_name_to_figure(GroundTruth_fig_dropdown, fig_dropdown):
    # figure = go.Figure()
    if GroundTruth_fig_dropdown == 'FASHIONMNIST':
        return {'data': [
            go.Scatter(
                x=list(FASHIONMNIST_z_run_sep[FASHIONMNIST_labels == i, 0]),
                y=list(FASHIONMNIST_z_run_sep[FASHIONMNIST_labels == i, 1]),
                text="class" + str(i),
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 15,
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name=str(i)
            ) for i in FASHIONMNIST_labels_unique_list
        ],
        }

    elif GroundTruth_fig_dropdown == 'MNIST':
        return {'data': [
            go.Scatter(
                x=list(MNIST_z_run_sep[MNIST_labels == i, 0]),
                y=list(MNIST_z_run_sep[MNIST_labels == i, 1]),
                text="class" + str(i),
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 15,
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name=str(i)
            ) for i in MNIST_labels_unique_list
        ],
        }


    elif GroundTruth_fig_dropdown == 'ECG200':
        return {
            'data': [
                go.Scatter(
                    x=list(ECG200_z_run_sep[np.where(ECG200_y_pred_sc_model == i)[0], 0]),
                    y=list(ECG200_z_run_sep[np.where(ECG200_y_pred_sc_model == i)[0], 1]),
                    text="class" + str(i),
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name=str(i)
                ) for i in ECG200_y_pred_sc_model_unique_list
            ],
        }

    elif GroundTruth_fig_dropdown == 'ECG5000':
        return {
            'data': [
                go.Scatter(
                    x=list(ECG5000_z_run_sep[np.where(ECG5000_y_pred_sc_model == i)[0], 0]),
                    y=list(ECG5000_z_run_sep[np.where(ECG5000_y_pred_sc_model == i)[0], 1]),
                    text="class" + str(i),
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name=str(i)
                ) for i in ECG5000_y_pred_sc_model_unique_list
            ],
        }

    elif GroundTruth_fig_dropdown == 'CIFAR10':
        return {
            'data': [
                go.Scatter(
                    x=list(CIFAR10_z_run_sep[np.where(CIFAR10_y_pred_sc_model == i)[0], 0]),
                    y=list(CIFAR10_z_run_sep[np.where(CIFAR10_y_pred_sc_model == i)[0], 1]),
                    text="class" + str(i),
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name=str(i)
                ) for i in CIFAR10_y_pred_sc_model_unique_list
            ],
        }
    elif GroundTruth_fig_dropdown == 'ElectricDevices':
        return {
            'data': [
                go.Scatter(
                    x=list(ElectricDevices_z_run_sep[np.where(ElectricDevices_y_pred_sc_model == i)[0], 0]),
                    y=list(ElectricDevices_z_run_sep[np.where(ElectricDevices_y_pred_sc_model == i)[0], 1]),
                    text="class" + str(i),
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name=str(i)
                ) for i in ElectricDevices_y_pred_sc_model_unique_list
            ],
        }
    else:
        return {}


###################### Decoded
@app.callback(
    dash.dependencies.Output('decoded_fig_plot', 'figure'),
    [dash.dependencies.Input('decoded_fig_dropdown', 'value'),
     dash.dependencies.Input('GroundTruth_fig_dropdown', 'value')])
def update_output(decoded_fig_dropdown, GroundTruth_fig_dropdown):
    return decoded_name_to_figure(decoded_fig_dropdown, GroundTruth_fig_dropdown)


def decoded_name_to_figure(decoded_fig_dropdown, GroundTruth_fig_dropdown):
    if GroundTruth_fig_dropdown == 'ECG200' or ("ECG200" in decoded_fig_dropdown):
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=ECG200_test_data_sample_time_steps, y=np.squeeze(ECG200_test_data_sample),
                                 mode='lines+markers',
                                 name='test_data_sample'))
        fig.add_trace(go.Scatter(x=ECG200_test_data_sample_time_steps, y=ECG200_X_decoded_sample,
                                 mode='lines+markers',
                                 name='X_decoded_sample'))
        return fig
    elif GroundTruth_fig_dropdown == 'ECG5000' or ("ECG5000" in decoded_fig_dropdown):
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=ECG5000_test_data_sample_time_steps, y=np.squeeze(ECG5000_test_data_sample),
                                 mode='lines+markers',
                                 name='test_data_sample'))
        fig.add_trace(go.Scatter(x=ECG5000_test_data_sample_time_steps, y=ECG5000_X_decoded_sample,
                                 mode='lines+markers',
                                 name='X_decoded_sample'))
        return fig
    elif GroundTruth_fig_dropdown == 'ElectricDevices' or ("ElectricDevices" in decoded_fig_dropdown):
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(x=ElectricDevices_test_data_sample_time_steps, y=np.squeeze(ElectricDevices_test_data_sample),
                       mode='lines+markers',
                       name='test_data_sample'))
        fig.add_trace(go.Scatter(x=ElectricDevices_test_data_sample_time_steps, y=ElectricDevices_X_decoded_sample,
                                 mode='lines+markers',
                                 name='X_decoded_sample'))
        return fig
    else:
        return {}


if __name__ == '__main__':
    app.run_server(debug=True)
