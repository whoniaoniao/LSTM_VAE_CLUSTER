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
import dash

from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
import plotly.graph_objs as go
import dash
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html

def main(config):
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
# general = "ElectricDevices"  # Dataset Name:
general ="ElectricDevices" # "ACSF1","ElectricDevices" 7, "Computers", "ECG200" 23, "ECG5000",2.
# plotly.offline.init_notebook_mode()
# Input parameters
index = 7
dload = './{}/{}_model_dir/'.format(str(general),str(general)) + str(index)  # download directory
image_save = "./{}/{}_images/".format(str(general),str(general)) + str(index)
dload_path = Path(dload)

plots = Path(dload + "/plots")

Path(dload_path).mkdir(parents=True, exist_ok=True)
Path(image_save).mkdir(parents=True, exist_ok=True)
Path(plots).mkdir(parents=True, exist_ok=True)
# if not Path.exists(dload_path):
#     Path.mkdir(dload_path)
# if not Path.exists(plots):
#     Path.mkdir(plots)

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
# X_train, X_val, y_train, y_val = open_newdata_ED('data', ratio_train=0.9)
X_train, X_val, y_train, y_val = open_newdata_ED('{}'.format(str(general)), ratio_train=0.9, dataset=general)

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
vrae = main(config)
if not Path.exists(Path(dload + "/vrae.pth")):
    vrae.fit(train_dataset, save=True)
    vrae.save('vrae.pth')
else:
    vrae.load(dload + "/vrae.pth")
# loss = vrae.compute_loss(test_dataset)
# # # Transform the input timeseries to encoded latent vectors
z_run = vrae.transform(test_dataset)
# # # # Visualize using PCA and tSNE
z_run_sep, labels, y_pred_total = plot_clustering(z_run, y_val, engine='matplotlib', download=True,
                                                  folder_name=image_save)
# ##Transform the input dataset to encoded latent vectors
z_run_train = vrae.transform(train_dataset, save=False)
# labels = y_train[:z_run_train.shape[0]]
# clf = svm.SVC()
# clf.fit(z_run_train, labels)
# y_pred = clf.predict(z_run)
# labels_for_z_run = y_train[:z_run.shape[0]]
# f1_final_score = f1_score(y_true=labels_for_z_run, y_pred=y_pred, average='weighted')
# recall_final_score = recall_score(y_true=labels_for_z_run, y_pred=y_pred, average='weighted')
# accuracy_final_score = accuracy_score(y_true=labels_for_z_run, y_pred=y_pred)
# precision_final_score = precision_score(y_true=labels_for_z_run, y_pred=y_pred, average='weighted')
# print("accuracy of the svm", accuracy_score(labels_for_z_run, y_pred))
# print("f1_final_score ", f1_final_score)
# print("recall_final_score", recall_final_score)
# print("accuracy_final_score", accuracy_final_score)
# print("precision_final_score", precision_final_score)
# Visulalizing the input time series vs the Output time series
X_decoded = vrae.reconstruct(test_dataset)
for i in range(10):
    # Plotting test_dataset
    test_data_sample = test_dataset.tensors[0].numpy()[i, :]
    test_data_sample_time_steps = np.arange(0, test_data_sample.shape[0])
    plt.plot(test_data_sample_time_steps, np.squeeze(test_data_sample))
    X_decoded_sample = np.squeeze(X_decoded)[:, i]
    X_decoded_sample_time_steps = np.arange(0, X_decoded_sample.shape[0])
    plt.plot(X_decoded_sample_time_steps, X_decoded_sample)
    plt.savefig(plots / Path("{}_compare.png".format(str(i))))
    plt.show()
    mean_error = np.mean(np.abs(X_decoded_sample - np.squeeze(test_data_sample)))
    # print(mean_error)
    np.save("data_for_dash/{}_test_data_sample.npy".format(str(general)), test_data_sample)
    np.save("data_for_dash/{}_X_decoded_sample.npy".format(str(general)), X_decoded_sample)
print("After")

np.save("data_for_dash/{}_labels.npy".format(str(general)), labels)
np.save("data_for_dash/{}_y_pred_total.npy".format(str(general)), y_pred_total)
np.save("data_for_dash/{}_z_run_sep.npy".format(str(general)), z_run_sep)


###################### Plot
general_labels = np.load("data_for_dash/{}_labels.npy".format(general) )
general_y_pred_total = np.load("data_for_dash/{}_y_pred_total.npy".format(general))
general_z_run_sep = np.load("data_for_dash/{}_z_run_sep.npy".format(general))
general_test_data_sample = np.load("data_for_dash/{}_test_data_sample.npy".format(general))
general_X_decoded_sample = np.load("data_for_dash/{}_X_decoded_sample.npy".format(general))
[general_y_pred_sc_model, general_y_pred_kmean, general_y_pred_meanshift_model, general_y_pred_GMM_model, general_y_pred_DBSCAN_model] = general_y_pred_total
general_labels_unique = np.unique(general_labels).astype(np.int32)
general_labels_unique_list = general_labels_unique.tolist()
general_y_kmean_pred_unique = np.unique(general_y_pred_kmean).astype(np.int32)
general_y_kmean_pred_unique_list = general_y_kmean_pred_unique.tolist()
general_y_pred_sc_model_unique = np.unique(general_y_pred_sc_model).astype(np.int32)
general_y_pred_sc_model_unique_list = general_y_pred_sc_model_unique.tolist()
# ## plotly
general_test_data_sample_time_steps = np.arange(0, general_test_data_sample.shape[0])
# ElectricDevices_test_data_sample_time_steps = np.arange(0, ElectricDevices_test_data_sample.shape[0])

app = dash.Dash()
app.title = 'My Title'

fig_names = ["{}_y_pred_GMM_model".format("general"), "{}_y_pred_sc_model".format("general"), "{}_y_pred_kmean".format("general")]


GroundTruth_fig_names = [general]



app.layout = html.Div(children=[
html.H4('Heating System Temperature Data Visulation'),
    html.Center(
        'The purpose of the scatter plot below is to prove if a temperature reset strategy is implemented on the hydronic heating system. At various outside air temperature conditions, the hot water temperature should fluctuate to save energy.'),

    # Dropdown
    # html.Div([
    #     html.H1(children='Dropdown test'),
    #     fig_dropdown
    # ]),
    # dcc.Graph(id='fig_plot'),
    # All elements from the top of the page

    # Dropdown
    html.Div([
        html.H2(children='{}_Clustering'.format(general)),
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
    ],style={'display': 'inline-block', 'width': '49%'}),

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
    #                     x=list(EGC_z_run_sep[np.where(EGC_labels == i)[0], 0]),
    #                     y=list(EGC_z_run_sep[np.where(EGC_labels == i)[0], 1]),
    #                     text="class" + str(i),
    #                     mode='markers',
    #                     opacity=0.7,
    #                     marker={
    #                         'size': 15,
    #                         'line': {'width': 0.5, 'color': 'white'}
    #                     },
    #                     name=str(i)
    #                 ) for i in EGC_labels_unique_list
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
def update_output(fig_dropdown,GroundTruth_fig_dropdown):
    return name_to_figure(fig_dropdown,GroundTruth_fig_dropdown)

def name_to_figure(fig_dropdown,GroundTruth_fig_dropdown):
    # figure = go.Figure()
    # if GroundTruth_fig_dropdown == 'EGC200' and ("EGC" not in fig_dropdown):
    #     return {'data': [
    #         go.Scatter(
    #             x=list(EGC_z_run_sep[np.where(EGC_y_pred_kmean == i)[0], 0]),
    #             y=list(EGC_z_run_sep[np.where(EGC_y_pred_kmean == i)[0], 1]),
    #             text="class" + str(i),
    #             mode='markers',
    #             opacity=0.7,
    #             marker={
    #                 'size': 15,
    #                 'line': {'width': 0.5, 'color': 'white'}
    #             },
    #             name=str(i)
    #         ) for i in EGC_y_kmean_pred_unique
    #     ]}
    # elif GroundTruth_fig_dropdown == 'FASHIONMNIST' and ("FASHIONMNIST" not in fig_dropdown):
    #     return {'data': [
    #         go.Scatter(
    #             x=list(EGC_z_run_sep[np.where(FASHIONMNIST_y_pred_kmean == i)[0], 0]),
    #             y=list(EGC_z_run_sep[np.where(FASHIONMNIST_y_pred_kmean == i)[0], 1]),
    #             text="class" + str(i),
    #             mode='markers',
    #             opacity=0.7,
    #             marker={
    #                 'size': 15,
    #                 'line': {'width': 0.5, 'color': 'white'}
    #             },
    #             name=str(i)
    #         ) for i in FASHIONMNIST_y_pred_unique
    #     ]}
    # elif GroundTruth_fig_dropdown == 'ElectricDevices' and ("ElectricDevices" not in fig_dropdown):
    #     return {'data': [
    #         go.Scatter(
    #             x=list(ElectricDevices_z_run_sep[np.where(ElectricDevices_y_pred_kmean == i)[0], 0]),
    #             y=list(ElectricDevices_z_run_sep[np.where(ElectricDevices_y_pred_kmean == i)[0], 1]),
    #             text="class" + str(i),
    #             mode='markers',
    #             opacity=0.7,
    #             marker={
    #                 'size': 15,
    #                 'line': {'width': 0.5, 'color': 'white'}
    #             },
    #             name=str(i)
    #         ) for i in ElectricDevices_y_pred_unique
    #     ]}
    if fig_dropdown == 'general_y_pred_GMM_model':
        return {'data': [
            go.Scatter(
                x=list(general_z_run_sep[np.where(general_y_pred_GMM_model == i)[0], 0]),
                y=list(general_z_run_sep[np.where(general_y_pred_GMM_model == i)[0], 1]),
                text="class" + str(i),
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 15,
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name=str(i)
            ) for i in general_y_pred_sc_model_unique_list
        ]}

    elif fig_dropdown == 'general_y_pred_sc_model':
        return {
            'data': [
                go.Scatter(
                    x=list(general_z_run_sep[np.where(general_y_pred_sc_model == i)[0], 0]),
                    y=list(general_z_run_sep[np.where(general_y_pred_sc_model == i)[0], 1]),
                    text="class" + str(i),
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name=str(i)
                ) for i in general_y_pred_sc_model_unique_list
            ],
        }
    elif fig_dropdown == 'general_y_pred_kmean':
        return {'data': [
            go.Scatter(
                x=list(general_z_run_sep[np.where(general_y_pred_kmean == i)[0], 0]),
                y=list(general_z_run_sep[np.where(general_y_pred_kmean == i)[0], 1]),
                text="class" + str(i),
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 15,
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name=str(i)
            ) for i in general_y_kmean_pred_unique
        ]}
    # elif fig_dropdown == 'FASHIONMNIST_y_pred_GMM_model':
    #     return {'data': [
    #         go.Scatter(
    #             x=list(FASHIONMNIST_z_run_sep[np.where(FASHIONMNIST_y_pred_GMM_model == i)[0], 0]),
    #             y=list(FASHIONMNIST_z_run_sep[np.where(FASHIONMNIST_y_pred_GMM_model == i)[0], 1]),
    #             text="class" + str(i),
    #             mode='markers',
    #             opacity=0.7,
    #             marker={
    #                 'size': 15,
    #                 'line': {'width': 0.5, 'color': 'white'}
    #             },
    #             name=str(i)
    #         ) for i in FASHIONMNIST_y_pred_sc_model_unique_list
    #     ]}
    #
    # elif fig_dropdown == 'FASHIONMNIST_y_pred_sc_model':
    #     return {
    #         'data': [
    #             go.Scatter(
    #                 x=list(FASHIONMNIST_z_run_sep[np.where(FASHIONMNIST_y_pred_sc_model == i)[0], 0]),
    #                 y=list(FASHIONMNIST_z_run_sep[np.where(FASHIONMNIST_y_pred_sc_model == i)[0], 1]),
    #                 text="class" + str(i),
    #                 mode='markers',
    #                 opacity=0.7,
    #                 marker={
    #                     'size': 15,
    #                     'line': {'width': 0.5, 'color': 'white'}
    #                 },
    #                 name=str(i)
    #             ) for i in FASHIONMNIST_y_pred_sc_model_unique_list
    #         ],
    #     }
    # elif fig_dropdown == 'FASHIONMNIST_y_pred_kmean':
    #     return {'data': [
    #         go.Scatter(
    #             x=list(FASHIONMNIST_z_run_sep[np.where(FASHIONMNIST_y_pred_kmean == i)[0], 0]),
    #             y=list(FASHIONMNIST_z_run_sep[np.where(FASHIONMNIST_y_pred_kmean == i)[0], 1]),
    #             text="class" + str(i),
    #             mode='markers',
    #             opacity=0.7,
    #             marker={
    #                 'size': 15,
    #                 'line': {'width': 0.5, 'color': 'white'}
    #             },
    #             name=str(i)
    #         ) for i in FASHIONMNIST_y_pred_unique
    #     ]}
    # elif fig_dropdown == 'ElectricDevices_y_pred_GMM_model':
    #     return {'data': [
    #         go.Scatter(
    #             x=list(ElectricDevices_z_run_sep[np.where(ElectricDevices_y_pred_GMM_model == i)[0], 0]),
    #             y=list(ElectricDevices_z_run_sep[np.where(ElectricDevices_y_pred_GMM_model == i)[0], 1]),
    #             text="class" + str(i),
    #             mode='markers',
    #             opacity=0.7,
    #             marker={
    #                 'size': 15,
    #                 'line': {'width': 0.5, 'color': 'white'}
    #             },
    #             name=str(i)
    #         ) for i in ElectricDevices_y_pred_sc_model_unique_list
    #     ]}
    #
    # elif fig_dropdown == 'ElectricDevices_y_pred_sc_model':
    #     return {
    #         'data': [
    #             go.Scatter(
    #                 x=list(ElectricDevices_z_run_sep[np.where(ElectricDevices_y_pred_sc_model == i)[0], 0]),
    #                 y=list(ElectricDevices_z_run_sep[np.where(ElectricDevices_y_pred_sc_model == i)[0], 1]),
    #                 text="class" + str(i),
    #                 mode='markers',
    #                 opacity=0.7,
    #                 marker={
    #                     'size': 15,
    #                     'line': {'width': 0.5, 'color': 'white'}
    #                 },
    #                 name=str(i)
    #             ) for i in ElectricDevices_y_pred_sc_model_unique_list
    #         ],
    #     }
    # elif fig_dropdown == 'ElectricDevices_y_pred_kmean':
    #     return {'data': [
    #         go.Scatter(
    #             x=list(ElectricDevices_z_run_sep[np.where(ElectricDevices_y_pred_kmean == i)[0], 0]),
    #             y=list(ElectricDevices_z_run_sep[np.where(ElectricDevices_y_pred_kmean == i)[0], 1]),
    #             text="class" + str(i),
    #             mode='markers',
    #             opacity=0.7,
    #             marker={
    #                 'size': 15,
    #                 'line': {'width': 0.5, 'color': 'white'}
    #             },
    #             name=str(i)
    #         ) for i in ElectricDevices_y_pred_unique
    #     ]}
    else:
        return {}

###################### GroundTruth
@app.callback(
    dash.dependencies.Output('GroundTruth_fig_plot', 'figure'),
    [dash.dependencies.Input('GroundTruth_fig_dropdown', 'value'),
     dash.dependencies.Input('fig_dropdown', 'value')])

def GroundTruth_update_output(GroundTruth_fig_dropdown,fig_dropdown):
    return GroundTruth_name_to_figure(GroundTruth_fig_dropdown,fig_dropdown)

def GroundTruth_name_to_figure(GroundTruth_fig_dropdown,fig_dropdown) :
    # figure = go.Figure()
    # if GroundTruth_fig_dropdown == 'FASHIONMNIST':
    return {'data': [
                go.Scatter(
                    x=list(general_z_run_sep[general_labels == i, 0]),
                    y=list(general_z_run_sep[general_labels == i, 1]),
                    text="class" + str(i),
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name=str(i)
                ) for i in general_labels_unique_list
            ],
}

    # elif GroundTruth_fig_dropdown == 'EGC200':
    #     return {
    #         'data': [
    #             go.Scatter(
    #                 x=list(EGC_z_run_sep[np.where(EGC_y_pred_sc_model == i)[0], 0]),
    #                 y=list(EGC_z_run_sep[np.where(EGC_y_pred_sc_model == i)[0], 1]),
    #                 text="class" + str(i),
    #                 mode='markers',
    #                 opacity=0.7,
    #                 marker={
    #                     'size': 15,
    #                     'line': {'width': 0.5, 'color': 'white'}
    #                 },
    #                 name=str(i)
    #             ) for i in EGC_y_pred_sc_model_unique_list
    #         ],
    #     }
    # elif GroundTruth_fig_dropdown == 'ElectricDevices':
    #     return {
    #         'data': [
    #             go.Scatter(
    #                 x=list(ElectricDevices_z_run_sep[np.where(ElectricDevices_y_pred_sc_model == i)[0], 0]),
    #                 y=list(ElectricDevices_z_run_sep[np.where(ElectricDevices_y_pred_sc_model == i)[0], 1]),
    #                 text="class" + str(i),
    #                 mode='markers',
    #                 opacity=0.7,
    #                 marker={
    #                     'size': 15,
    #                     'line': {'width': 0.5, 'color': 'white'}
    #                 },
    #                 name=str(i)
    #             ) for i in ElectricDevices_y_pred_sc_model_unique_list
    #         ],
    #     }
    # else:
    #     return {}

###################### Decoded
@app.callback(
    dash.dependencies.Output('decoded_fig_plot', 'figure'),
    [dash.dependencies.Input('decoded_fig_dropdown', 'value'),
     dash.dependencies.Input('GroundTruth_fig_dropdown', 'value')])
def update_output(decoded_fig_dropdown,GroundTruth_fig_dropdown):
    return decoded_name_to_figure(decoded_fig_dropdown,GroundTruth_fig_dropdown)

def decoded_name_to_figure(decoded_fig_dropdown,GroundTruth_fig_dropdown):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=general_test_data_sample_time_steps, y=np.squeeze(general_test_data_sample),
                             mode='lines+markers',
                             name='test_data_sample'))
    fig.add_trace(go.Scatter(x=general_test_data_sample_time_steps, y=general_X_decoded_sample,
                             mode='lines+markers',
                             name='X_decoded_sample'))
    return fig
    # if GroundTruth_fig_dropdown == 'EGC200' or ("EGC" in decoded_fig_dropdown):
    #
    #     fig.add_trace(go.Scatter(x=EGC_test_data_sample_time_steps, y=np.squeeze(EGC_test_data_sample),
    #                              mode='lines+markers',
    #                              name='test_data_sample'))
    #     fig.add_trace(go.Scatter(x=EGC_test_data_sample_time_steps, y=EGC_X_decoded_sample,
    #                              mode='lines+markers',
    #                              name='X_decoded_sample'))
    #     return fig
    # elif GroundTruth_fig_dropdown == 'ElectricDevices' or ("ElectricDevices" in decoded_fig_dropdown):
    #     fig.add_trace(go.Scatter(x=ElectricDevices_test_data_sample_time_steps, y=np.squeeze(ElectricDevices_test_data_sample),
    #                              mode='lines+markers',
    #                              name='test_data_sample'))
    #     fig.add_trace(go.Scatter(x=ElectricDevices_test_data_sample_time_steps, y=ElectricDevices_X_decoded_sample,
    #                              mode='lines+markers',
    #                              name='X_decoded_sample'))
    #     return fig
    # else:
    #     return {}
# if __name__ == '__main__':
# app.run_server(debug=True)
