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

from plotly.offline import init_notebook_mode, iplot


# https://plotly.com/python/line-charts/
# https://plotly.com/python/getting-started/#installation

def main(config, X_train):
    hidden_size = config.hidden_size
    hidden_layer_depth = config.hidden_layer_depth
    latent_length = config.latent_length
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    n_epochs = config.n_epochs
    dropout_rate = config.dropout_rate
    optimizer = config.optimizer  # options: ADAM, SGD
    cuda = config.cuda  # options: True, False
    print_every = config.print_every
    clip = config.clip  # options: True, False
    max_grad_norm = config.max_grad_norm
    loss = config.loss  # options: SmoothL1Loss, MSELoss
    block = config.block  # options: LSTM, GRU
    dload = config.dload

    sequence_length = X_train.shape[1]

    number_of_features = X_train.shape[2]

    vrae = VRAE(sequence_length=sequence_length,
                number_of_features=number_of_features,
                hidden_size=hidden_size,
                hidden_layer_depth=hidden_layer_depth,
                latent_length=latent_length,
                batch_size=batch_size,
                learning_rate=learning_rate,
                n_epochs=n_epochs,
                dropout_rate=dropout_rate,
                optimizer=optimizer,
                cuda=cuda,
                print_every=print_every,
                clip=clip,
                max_grad_norm=max_grad_norm,
                loss=loss,
                block=block,
                dload=dload)

    return vrae


# if __name__ == '__main__':
    # plotly.offline.init_notebook_mode()
    # Input parameters
index = 23
dload = './ECG200_model_dir/' + str(index)  # download directory
image_save = "./ECG200_images/" + str(index)
dload_path = Path(dload)
plots = Path(dload + "/plots")
if not Path.exists(dload_path):
    Path.mkdir(dload_path)
if not Path.exists(plots):
    Path.mkdir(plots)

torch.manual_seed(0)

# Hyper parameters
cwd = Path.cwd()
with open(str(cwd / "config.yml")) as handle:
    config = yaml.load(handle, Loader=yaml.FullLoader)
    config_dict = config.copy()
    config = AttrDict(config)

with open(str(dload + "/" + "config.yml"), "w") as handle:
    yaml.dump(config_dict, handle)

# Load data and preprocess
X_train, X_val, y_train, y_val = open_newdata('ECG200', ratio_train=0.9,dataset="ECG200")
num_classes = len(np.unique(y_train))
base = np.min(y_train)  # Check if data is 0-based
if base != 0:
    y_train -= base
y_val -= base
train_dataset = TensorDataset(torch.from_numpy(X_train))
test_dataset = TensorDataset(torch.from_numpy(X_val))
# Save the model to be fetched later
# vrae.save('vrae.pth')
# print("Save Model successfully")
# To load a presaved model, execute:
Model_Num = index
# Fit the model onto dataset
config.dload = dload
vrae = main(config, X_train)
# vrae.fit(train_dataset, save=True)
# vrae.save('vrae.pth')
vrae.load('./ECG200_model_dir/{}/vrae.pth'.format(str(Model_Num)))
# # loss = vrae.compute_loss(test_dataset)
# # # Transform the input timeseries to encoded latent vectors
z_run = vrae.transform(test_dataset)
print("z_run", z_run.shape)
# # Visualize using PCA and tSNE
z_run_sep, labels, y_pred_total = plot_clustering(z_run, y_val, engine='matplotlib', download=True,
                                                          folder_name=image_save)
##Transform the input dataset to encoded latent vectors
z_run_train = vrae.transform(train_dataset, save=False)
ground_truth_label = y_train[:z_run_train.shape[0]]
clf = svm.SVC()
clf.fit(z_run_train, ground_truth_label)
y_pred = clf.predict(z_run)
labels_for_z_run = y_train[:z_run.shape[0]]
f1_final_score = f1_score(y_true=labels_for_z_run, y_pred=y_pred, average='weighted')
recall_final_score = recall_score(y_true=labels_for_z_run, y_pred=y_pred, average='weighted')
accuracy_final_score = accuracy_score(y_true=labels_for_z_run, y_pred=y_pred)
precision_final_score = precision_score(y_true=labels_for_z_run, y_pred=y_pred, average='weighted')
print("accuracy of the svm", accuracy_score(labels_for_z_run, y_pred))
print("f1_final_score ", f1_final_score)
print("recall_final_score", recall_final_score)
print("accuracy_final_score", accuracy_final_score)
print("precision_final_score", precision_final_score)
# # Visulalizing the input time series vs the Output time series
X_decoded = vrae.reconstruct(test_dataset)
for i in range(1):
    # Plotting test_dataset
    test_data_sample = test_dataset.tensors[0].numpy()[i, :]
    test_data_sample_time_steps = np.arange(0, test_data_sample.shape[0])
    plt.plot(test_data_sample_time_steps, np.squeeze(test_data_sample))
    X_decoded_sample = np.squeeze(X_decoded)[:, i]
    X_decoded_sample_time_steps = np.arange(0, X_decoded_sample.shape[0])
    print("X_decoded_sample_time_steps.shape", X_decoded_sample_time_steps.shape)
    print("X_decoded_sample.shape", X_decoded_sample.shape)
    print("test_data_sample.shape", np.squeeze(test_data_sample).shape)
    plt.plot(X_decoded_sample_time_steps, X_decoded_sample)
    plt.savefig("./ECG200_model_dir/{}/plots/{}_compare.png".format(str(Model_Num), str(i)))
    plt.show()
    mean_error = np.mean(np.abs(X_decoded_sample - np.squeeze(test_data_sample)))
    print(mean_error)
    # ## plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_data_sample_time_steps, y=np.squeeze(test_data_sample),
                             mode='lines+markers',
                             name='test_data_sample'))
    fig.add_trace(go.Scatter(x=test_data_sample_time_steps, y=X_decoded_sample,
                             mode='lines+markers',
                             name='X_decoded_sample'))
    np.save("data_for_dash/EGC_test_data_sample.npy", test_data_sample)
    np.save("data_for_dash/EGC_X_decoded_sample.npy", X_decoded_sample)
    # fig.show()
print("After")

#######################################
##### GroundTruth and Clustering Result
#######################################
np.save("data_for_dash/EGC_labels.npy",labels)
np.save("data_for_dash/EGC_y_pred_total.npy",y_pred_total)
np.save("data_for_dash/EGC_z_run_sep.npy",z_run_sep)
[y_pred_sc_model, y_pred_kmean, y_pred_meanshift_model, y_pred_GMM_model, y_pred_DBSCAN_model] = y_pred_total

labels_unique = np.unique(labels).astype(np.int32)
labels_unique_list = labels_unique.tolist()
y_pred_unique = np.unique(y_pred_kmean).astype(np.int32)
y_pred_unique_unique_list = y_pred_unique.tolist()
y_pred_sc_model_unique = np.unique(y_pred_sc_model).astype(np.int32)
y_pred_sc_model_unique_unique_list = y_pred_sc_model_unique.tolist()

app = dash.Dash()

# EGC_fig_dropdown = html.Div([
#     dcc.Dropdown(
#         id='EGC_fig_dropdown',
#         options=[{'label': x, 'value': x} for x in fig_names],
#         value=None
#     )])
# fig_plot = html.Div(id='fig_plot')
# app.layout = html.Div([EGC_fig_dropdown, fig_plot])


fig_names = ["y_pred_GMM_model", "y_pred_sc_model", "y_pred_kmean"]

app.layout = html.Div(children=[
    # Dropdown
    # html.Div([
    #     html.H1(children='Dropdown test'),
    #     EGC_fig_dropdown
    # ]),
    # dcc.Graph(id='fig_plot'),
    # All elements from the top of the page

    # Dropdown
    html.Div([
        html.H2(children='DROPDOWM'),
        dcc.Dropdown(
            id='EGC_fig_dropdown',
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
                        x=list(z_run_sep[np.where(y_pred == i)[0], 0]),
                        y=list(z_run_sep[np.where(y_pred == i)[0], 1]),
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
    [dash.dependencies.Input('EGC_fig_dropdown', 'value')])
def update_output(EGC_fig_dropdown):
    return name_to_figure(EGC_fig_dropdown)


def name_to_figure(EGC_fig_dropdown):
    figure = go.Figure()
    if EGC_fig_dropdown == 'y_pred_GMM_model':
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

    elif EGC_fig_dropdown == 'y_pred_sc_model':
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
    elif EGC_fig_dropdown == 'y_pred_kmean':
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
