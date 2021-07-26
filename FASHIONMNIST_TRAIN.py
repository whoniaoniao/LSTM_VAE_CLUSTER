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
import dash
import pandas as pd
import plotly.graph_objs as go
import dash_core_components as dcc
import dash_html_components as html


# import plotly.express as px


def to_img(x):
    x = (x + 1.) * 0.5
    x = x.clamp(0, 1)
    # print("x_size", x.size())
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

    sequence_length = 28

    number_of_features = 28

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
    transform = tfs.Compose([tfs.ToTensor(), tfs.Normalize([0.5], [0.5])])  # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # plotly.offline.init_notebook_mode()
    # Input parameters
    index = 1
    dload = './FASHION_MINST_model_dir/' + str(index)  # download directory
    image_save = "./FASHION_MINST_images/" + str(index)
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
    trainset = torchvision.datasets.FashionMNIST(root='./data_2', train=True,
                                                 download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size,
                                              shuffle=True, num_workers=4)
    testset = torchvision.datasets.FashionMNIST(root='./data_2', train=False,
                                                download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size,
                                             shuffle=False, num_workers=4)

    # get some random training images
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # Save the model to be fetched later
    # vrae.save('vrae.pth')
    # print("Save Model successfully")
    # To load a presaved model, execute:
    Model_Num = index
    # Fit the model onto dataset
    config.dload = dload
    vrae_cnn = main(config)
    # vrae_cnn.fit_cnn(trainloader, save=True, cnn_mode=True)
    # vrae_cnn.save('vrae_Cnn.pth')
    vrae_cnn.load('./FASHION_MINST_model_dir/{}/vrae_Cnn.pth'.format(str(Model_Num)))
    # loss = vrae.compute_loss(test_dataset)
    # # Transform the input timeseries to encoded latent vectors
    z_run, y_val = vrae_cnn.transform(testset, save=True, cnn_mode=True)
    # # Visualize using PCA and tSNE
    z_run_sep, y_pred_colors, colors = plot_clustering(z_run, y_val, engine='matplotlib', download=True,
                                                       folder_name=image_save)
    ##Transform the input dataset to encoded latent vectors
    # z_run_train, y_train = vrae_cnn.transform(trainset, save=False, cnn_mode=True)
    # labels = y_train
    # clf = svm.SVC()
    # clf.fit(z_run_train, labels)
    # s = pickle.dumps(clf)
    # clf = pickle.loads(s)

    # y_pred = clf.predict(z_run)
    # labels_for_z_run = y_train
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
    X_decoded = vrae_cnn.reconstruct(testset, cnn_mode=True)
    print("rae_cnn.reconstruct(testset,cnn_mode=True)", X_decoded.size)
    X_decoded_tmp = torch.tensor(X_decoded)
    print(" torch.tensor(X_decoded)", X_decoded_tmp.size())
    X_decoded = torch.tensor(X_decoded).permute(1, 0, 2)
    # print(X_decoded)
    # print(type(X_decoded))
    print("torch.tensor(X_decoded).permute(1, 2, 0)", X_decoded.size())
    # X_decoded = torch.tensor(X_decoded)
    # decode_img = to_img(X_decoded).squeeze()
    # decode_img = decode_img.data.numpy() * 255
    # print(decode_img.size)
    # for i in range(16):
    #     plt.imshow(decode_img[i,:,:].astype('uint8'), cmap='gray')
    # plt.show()
    print("After")

    # X_decoded = X_decoded.reshape(-1, 28, 28)

    # # # Plotting
    # for i in range(config.batch_size):
    #     # Plotting
    #
    #     X_decoded_sample = X_decoded[i, :, :]
    #     decode_img = to_img(X_decoded_sample)
    #     decode_img = decode_img.data.numpy() * 255
    #     print(X_decoded_sample.shape)
    #     imgplot = plt.imshow(decode_img.astype('uint8'), cmap='gray')
    #
    #     # imgplot = plt.imshow(X_decoded_sample)
    #     plt.savefig("./FASHION_MINST_model_dir/{}/plots/{}_compare.png".format(str(Model_Num), str(i)))
    #     plt.show()
########################
app = dash.Dash()



#
#
 # STYLE 1
fig_names = ['LSTM_VAE_CLUSTERING', 'LSTM_VAE_GROUNDTRUTH']
fig_dropdown = html.Div([
    dcc.Dropdown(
        id='fig_dropdown',
        options=[{'label': x, 'value': x} for x in fig_names],
        value=None
    )])
fig_plot = html.Div(id='fig_plot')
app.layout = html.Div([fig_dropdown, fig_plot])

@app.callback(
dash.dependencies.Output('fig_plot', 'children'),
[dash.dependencies.Input('fig_dropdown', 'value')])
def update_output(fig_name):
    return name_to_figure(fig_name)





def name_to_figure(fig_name):
    figure = go.Figure()
    if fig_name == 'LSTM_VAE_CLUSTERING':
        for i in range(10):
            figure.add_trace(go.Scatter(x=list(z_run_sep[y_pred_colors == i, 0]),
                        y=list(z_run_sep[y_pred_colors == i, 1]),
                        text="class" + str(i),
                        mode='markers',
                        opacity=0.7,
                        marker={
                            'size': 15,
                            'line': {'width': 0.5, 'color': 'white'}
                        },
                        name=str(i)))
    elif fig_name == 'LSTM_VAE_GROUNDTRUTH':
        for i in range(10):
            figure.add_trace(go.Scatter(x=list(z_run_sep[colors == i, 0]),
                                        y=list(z_run_sep[colors == i, 1]),
                                        text="class" + str(i),
                                        mode='markers',
                                        opacity=0.7,
                                        marker={
                                            'size': 15,
                                            'line': {'width': 0.5, 'color': 'white'}
                                        },
                                        name=str(i)))
    return dcc.Graph(figure=figure)

app.run_server(debug=True, use_reloader=False)

# STYLE 2
app.layout = html.Div(children=[


    # All elements from the top of the page
    html.Div([
        html.H1(children='LSTM_VAE CLUSTERING'),

        html.Div(children='''
            Clustering Result with KMEAN
        '''),

        dcc.Graph(
            id='prediction',
            figure={
                'data': [
                    go.Scatter(
                        x=list(z_run_sep[y_pred_colors == i, 0]),
                        y=list(z_run_sep[y_pred_colors == i, 1]),
                        text="class" + str(i),
                        mode='markers',
                        opacity=0.7,
                        marker={
                            'size': 15,
                            'line': {'width': 0.5, 'color': 'white'}
                        },
                        name=str(i)
                    ) for i in range(10)
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
                        x=list(z_run_sep[colors == i, 0]),
                        y=list(z_run_sep[colors == i, 1]),
                        text="class" + str(i),
                        mode='markers',
                        opacity=0.7,
                        marker={
                            'size': 15,
                            'line': {'width': 0.5, 'color': 'white'}
                        },
                        name=str(i)
                    ) for i in range(10)
                ],

            }
        ),
    ]),
])

if __name__ == '__main__':
    app.run_server()
