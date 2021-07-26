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

# transform = tfs.Compose([tfs.ToTensor(), tfs.Normalize([0.5], [0.5])])  # classes = ('plane', 'car', 'bird', 'cat',
#
# # TODO for MNIST
# MNIST_trainset = torchvision.datasets.MNIST(root='./data_1', train=True,
#                                       download=True, transform=transform)
# MNIST_trainloader = torch.utils.data.DataLoader(MNIST_trainset, batch_size=16,
#                                           shuffle=True, num_workers=4)
# MNIST_testset = torchvision.datasets.MNIST(root='./data_1', train=False,
#                                      download=True, transform=transform)
# MNIST_testloader = torch.utils.data.DataLoader(MNIST_testset, batch_size=16,
#                                          shuffle=False, num_workers=4)
#
# MNIST_Train_Density = MNIST_trainset.train_data.shape[0] / (MNIST_trainset.train_data.shape[0] + MNIST_testset.test_data.shape[0])
#
# MNIST_Density = [MNIST_Train_Density, 1- MNIST_Train_Density]
#
# # TODO for FASHIONMNIST
# FASHIONMNIST_trainset = torchvision.datasets.FashionMNIST(root='./data_2', train=True,
#                                              download=True, transform=transform)
# FASHIONMNIST_trainloader = torch.utils.data.DataLoader(FASHIONMNIST_trainset, batch_size=16,
#                                           shuffle=True, num_workers=4)
# FASHIONMNIST_testset = torchvision.datasets.FashionMNIST(root='./data_2', train=False,
#                                             download=True, transform=transform)
# FASHIONMNIST_testloader = torch.utils.data.DataLoader(FASHIONMNIST_testset, batch_size=16,
#                                          shuffle=False, num_workers=4)
# FASHIONMNIST_Train_Density = FASHIONMNIST_trainset.train_data.shape[0] / (FASHIONMNIST_trainset.train_data.shape[0] + FASHIONMNIST_testset.test_data.shape[0])
#
# FASHIONMNIST_Density = [FASHIONMNIST_Train_Density, 1- FASHIONMNIST_Train_Density]
#
# # Load data and preprocess TODO for CIFAR10
# CIFAR10_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
# CIFAR10_trainloader = torch.utils.data.DataLoader(CIFAR10_trainset, batch_size=16,
#                                           shuffle=True, num_workers=4)
# CIFAR10_testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
# CIFAR10_testloader = torch.utils.data.DataLoader(CIFAR10_testset, batch_size=16,
#                                          shuffle=False, num_workers=4)

# CIFAR10_Train_Density = CIFAR10_trainset.train_data.shape[0] / (CIFAR10_trainset.train_data.shape[0] + CIFAR10_testset.test_data.shape[0])
# CIFAR10_Density = [CIFAR10_Train_Density, 1- CIFAR10_Train_Density]

# # TODO ECG200, val is test
# ECG200_X_train, ECG200_X_val, ECG200_y_train, ECG200_y_val = open_newdata('ECG200', ratio_train=0.9,dataset="ECG200")
# ECG200_Train_Density = ECG200_X_train.shape[0]/(ECG200_X_train.shape[0] + ECG200_X_val.shape[0])
# ECG200_Density = [ECG200_Train_Density, 1- ECG200_Train_Density]


# # TODO ECG5000, val is test
# ECG5000_X_train, ECG5000_X_val, ECG5000_y_train, ECG5000_y_val = open_newdata('ECG5000', ratio_train=0.9,dataset="ECG5000")
# ECG5000_Train_Density = ECG5000_X_train.shape[0]/(ECG5000_X_train.shape[0] + ECG5000_X_val.shape[0])
# ECG5000_Density = [ECG5000_Train_Density, 1- ECG5000_Train_Density]
#
#
# # TODO ElectricDevices, val is test
# ElectricDevices_X_train, ElectricDevices_X_val, ElectricDevices_y_train, ElectricDevices_y_val = open_newdata_ED('ElectricDevices', ratio_train=0.9)

# TODO
general_name = "ElectricDevices"
X_train, X_val, y_train, y_val = open_newdata_ED(general_name, ratio_train=0.9,dataset=general_name)

#######
##Plot
######

# Data set
y_total = np.concatenate((y_train, y_val), axis=None)

total_value = y_total.shape[0]
total_value_train = y_train.shape[0]
total_value_test = y_val.shape[0]

labels_unique = np.unique(y_total)
print("labels_unique", labels_unique)

label_density = list()
train_dataset_label_density = list()
test_dataset_label_density = list()

for label in labels_unique:
    value = y_total[y_total == label].shape[0]
    density = value/total_value
    label_density.append(density)

    value_train = y_train[y_train == label].shape[0]
    train_density = value_train/total_value_train
    train_dataset_label_density.append(train_density)

    value_test = y_val[y_val == label].shape[0]
    test_density = value_test/total_value_test
    test_dataset_label_density.append(test_density)

Total_Density = np.array([train_dataset_label_density, test_dataset_label_density])
# bars = ('MNIST', 'FASHIONMNIST', 'CIFAR10', 'ECG200', 'ECG5000')

import numpy as np

# 显示中文字体为SimHei
plt.rcParams['font.sans-serif'] = ['SimHei']

Train = Total_Density[0, :]
Test = Total_Density[1,:]

labels = labels_unique

fig, ax = plt.subplots(figsize=(8, 5), dpi=80)
width_1 = 0.4

ax.bar(np.arange(len(Train)), Train, width=width_1, tick_label=labels, label="Train")

ax.bar(np.arange(len(Test)) + width_1, Test, width=width_1, tick_label=labels, label="Test")

ax.legend()
plt.title(general_name)
plt.show()
