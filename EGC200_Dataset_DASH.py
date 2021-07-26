import dash
import pandas as pd
import plotly.graph_objs as go
import dash_core_components as dcc
import dash_html_components as html

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
from EGC200train import main
import warnings

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

# To load a presaved model
Model_Num = index
config.dload = dload

# Fit the model onto dataset
vrae = main(config, X_train)
#If the model has to be saved, with the learnt parameters use:
# vrae.fit(train_dataset, save=True)

## To load a presaved model, execute
vrae.load('./ECG200_model_dir/{}/vrae.pth'.format(str(Model_Num)))

# Transform the input timeseries to encoded latent vectors
#If the latent vectors have to be saved, pass the parameter `save`
z_run = vrae.transform(test_dataset)
print("z_run", z_run.shape)

# Visualize using PCA and tSNE for clustering
z_run_sep, y_pred_colors, colors = plot_clustering(z_run, y_val, engine='matplotlib', download=False, folder_name=image_save)

########################
app = dash.Dash()

df = pd.read_csv(
    'https://gist.githubusercontent.com/chriddyp/' +
    '5d1ea79569ed194d432e56108a04d188/raw/' +
    'a9f9e8076b837d541398e999dcbac2b2826a81f8/' +
    'gdp-life-exp-2007.csv')


print(df.continent.unique())
app.layout = html.Div([
    dcc.Graph(
        id='life-exp-vs-gdp',
        figure={
            'data': [
                go.Scatter(
                    x=z_run_sep[:, 0],
                    y=z_run_sep[:, 1],
                    text=i,
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name=i
                ) for i in ["predict"]
            ],
            'layout': go.Layout(
                xaxis={'type': 'log', 'title': 'Clustering'},
                yaxis={'title': '平均寿命'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
            )
        }
    )
])

if __name__ == '__main__':
    app.run_server()