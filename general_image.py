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
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as tfs
import pickle
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

def to_img(x):
    x = (x + 1.) * 0.5
    x = x.clamp(0, 1)
    print("x_size", x.size())
    x = x.view(28, 28)
    return x

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

    # TODO MNIST and FASHION_MNIST10
    sequence_length = 28

    number_of_features = 28

    # TODO CIFAR10
    # sequence_length = 32 * 32
    #
    # number_of_features = 3

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

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

    return vrae

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    general ="MNIST" # "FASHION_MINST",1. "MNIST",2.
    index = 3
    dload = './{}_model_dir/'.format(str(general),str(general)) + str(index)  # download directory
    image_save = "./{}_images/".format(str(general),str(general)) + str(index)
    dload_path = Path(dload)

    plots = Path(dload + "/plots")

    Path(dload_path).mkdir(parents=True, exist_ok=True)
    Path(image_save).mkdir(parents=True, exist_ok=True)
    Path(plots).mkdir(parents=True, exist_ok=True)

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
    # X_train, X_val, y_train, y_val = open_newdata_ED('{}'.format(str(general)), ratio_train=0.9, dataset=general)
    # num_classes = len(np.unique(y_train))
    # Load data and preprocess
    transform = tfs.Compose([tfs.ToTensor(), tfs.Normalize([0.5], [0.5])])  # classes = ('plane', 'car', 'bird', 'cat',

    # TODO for MNIST
    trainset = torchvision.datasets.MNIST(root='./data_1', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size,
                                              shuffle=True, num_workers=4)
    testset = torchvision.datasets.MNIST(root='./data_1', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size,
                                             shuffle=False, num_workers=4)

    # TODO for FASHIONMNIST
    # trainset = torchvision.datasets.FashionMNIST(root='./data_2', train=True,
    #                                              download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size,
    #                                           shuffle=True, num_workers=4)
    # testset = torchvision.datasets.FashionMNIST(root='./data_2', train=False,
    #                                             download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size,
    #                                          shuffle=False, num_workers=4)

    # Load data and preprocess TODO for CIFAR10
    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
    #                                         download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size,
    #                                           shuffle=True, num_workers=4)
    # testset = torchvision.datasets.CIFAR10(root='./data', train=False,
    #                                        download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size,
    #                                          shuffle=False, num_workers=4)

    # TODO Save the model to be fetched later
    # vrae.save('vrae.pth')
    # print("Save Model successfully")
    # To load a presaved model, execute:
    Model_Num = index
    # Fit the model onto dataset
    config.dload = dload
    vrae_cnn = main(config)
    if not Path.exists(Path(dload + "/vrae_Cnn.pth")):
        # vrae_cnn.fit(trainset, save=True, mode="")
        # TODO CIFAR10: , rgb_mode=True
        vrae_cnn.fit_cnn(trainloader, save=True, cnn_mode=True) # , rgb_mode=True
        vrae_cnn.save('vrae_Cnn.pth')
    else:
        vrae_cnn.load(dload + "/vrae_Cnn.pth")
    # TODO CIFAR10: , rgb_mode=True
    z_run, y_val = vrae_cnn.transform(testset, save=True, cnn_mode=True) # , rgb_mode=True
    z_run_sep, labels, y_pred_total = plot_clustering(z_run, y_val, engine='matplotlib', download=True,
                                                      folder_name=image_save)
    # TODO CIFAR10: , rgb_mode=True
    z_run_train, y_train = vrae_cnn.transform(trainset, save=False, cnn_mode=True) # , rgb_mode=True
    # TODO CIFAR10: , rgb_mode=True
    X_decoded = vrae_cnn.reconstruct(testset, cnn_mode=True) # , rgb_mode=True
    # for i in range(10):
    #     # Plotting test_dataset
    #     test_data_sample = testset.tensors[0].numpy()[i, :]
    #     test_data_sample_time_steps = np.arange(0, test_data_sample.shape[0])
    #     plt.plot(test_data_sample_time_steps, np.squeeze(test_data_sample))
    #     X_decoded_sample = np.squeeze(X_decoded)[:, i]
    #     X_decoded_sample_time_steps = np.arange(0, X_decoded_sample.shape[0])
    #     plt.plot(X_decoded_sample_time_steps, X_decoded_sample)
    #     plt.savefig(plots / Path("{}_compare.png".format(str(i))))
    #     plt.show()
    #     mean_error = np.mean(np.abs(X_decoded_sample - np.squeeze(test_data_sample)))
    #     # print(mean_error)
    #     np.save("data_for_dash/{}_test_data_sample.npy".format(str(general)), test_data_sample)
    #     np.save("data_for_dash/{}_X_decoded_sample.npy".format(str(general)), X_decoded_sample)
    # print("After")

    np.save("data_for_dash/{}_labels.npy".format(str(general)), labels)
    np.save("data_for_dash/{}_y_pred_total.npy".format(str(general)), y_pred_total)
    np.save("data_for_dash/{}_z_run_sep.npy".format(str(general)), z_run_sep)

 ###################### Plot
    general_labels = np.load("data_for_dash/{}_labels.npy".format(general) )
    general_y_pred_total = np.load("data_for_dash/{}_y_pred_total.npy".format(general))
    general_z_run_sep = np.load("data_for_dash/{}_z_run_sep.npy".format(general))
    # general_test_data_sample = np.load("data_for_dash/{}_test_data_sample.npy".format(general))
    # general_X_decoded_sample = np.load("data_for_dash/{}_X_decoded_sample.npy".format(general))
    [general_y_pred_sc_model, general_y_pred_kmean, general_y_pred_meanshift_model, general_y_pred_GMM_model, general_y_pred_DBSCAN_model] = general_y_pred_total
    general_labels_unique = np.unique(general_labels).astype(np.int32)
    general_labels_unique_list = general_labels_unique.tolist()
    general_y_kmean_pred_unique = np.unique(general_y_pred_kmean).astype(np.int32)
    general_y_kmean_pred_unique_list = general_y_kmean_pred_unique.tolist()
    general_y_pred_sc_model_unique = np.unique(general_y_pred_sc_model).astype(np.int32)
    general_y_pred_sc_model_unique_list = general_y_pred_sc_model_unique.tolist()
    # ## plotly
    # general_test_data_sample_time_steps = np.arange(0, general_test_data_sample.shape[0])
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
    # @app.callback(
    #     dash.dependencies.Output('decoded_fig_plot', 'figure'),
    #     [dash.dependencies.Input('decoded_fig_dropdown', 'value'),
    #      dash.dependencies.Input('GroundTruth_fig_dropdown', 'value')])
    # def update_output(decoded_fig_dropdown,GroundTruth_fig_dropdown):
    #     return decoded_name_to_figure(decoded_fig_dropdown,GroundTruth_fig_dropdown)

    # def decoded_name_to_figure(decoded_fig_dropdown,GroundTruth_fig_dropdown):
    #     fig = go.Figure()
    #     fig.add_trace(go.Scatter(x=general_test_data_sample_time_steps, y=np.squeeze(general_test_data_sample),
    #                              mode='lines+markers',
    #                              name='test_data_sample'))
    #     fig.add_trace(go.Scatter(x=general_test_data_sample_time_steps, y=general_X_decoded_sample,
    #                              mode='lines+markers',
    #                              name='X_decoded_sample'))
    #     return fig
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
    app.run_server(debug=True)
