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


def to_img(x):
    x = x / 2 + 0.5
    # x = x.clamp(0, 1)
    print("x_size", x.size())
    print("x type", type(x))
    x = x.numpy()
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

    # TODO best practice

    # sequence_length = 32
    #
    # number_of_features = 32 * 3

    sequence_length = 32 * 32

    number_of_features = 3

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
    index = 4
    dload = './CIFAR10_model_dir/' + str(index)  # download directory
    image_save = "./CIFAR10_images/" + str(index)
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
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size,
                                              shuffle=True, num_workers=4)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
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
    # vrae_cnn.fit_cnn(trainloader, save=True, cnn_mode=True, rgb_mode=True)
    # vrae_cnn.save('vrae_Cnn.pth')
    vrae_cnn.load('./CIFAR10_model_dir/{}/vrae_Cnn.pth'.format(str(Model_Num)))
    # # loss = vrae.compute_loss(test_dataset)
    # # # Transform the input timeseries to encoded latent vectors
    z_run, y_val = vrae_cnn.transform(testset, save=True, cnn_mode=True, rgb_mode=True)
    # # # Visualize using PCA and tSNE
    plot_clustering(z_run, y_val, engine='matplotlib', download=True, folder_name=image_save)
    # ##Transform the input dataset to encoded latent vectors
    # z_run_train = vrae.transform(train_dataset, save=False)
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
    # # Visulalizing the input time series vs the Output time series
    X_decoded = vrae_cnn.reconstruct(testset, cnn_mode=True, rgb_mode=True)
    X_decoded = torch.tensor(X_decoded)
    print(" torch.tensor(X_decoded)", X_decoded.size())  # torch.Size([32, 16, 96])
    X_decoded = X_decoded.permute(1, 0, 2)  # from torch.Size([32, 16, 96]) to torch.Size([16, 32, 96]))
    X_decoded = X_decoded.reshape(config.batch_size, 32, 32, 3)
    X_decoded = X_decoded.permute(0, 3, 2, 1)
    print("  X_decoded.permute(0, 3, 2, 1)", X_decoded.size())
    # X_decoded = torch.tensor(X_decoded).permute(1, 2, 0)
    # print(X_decoded)
    # print(type(X_decoded))
    # print(X_decoded.size)
    # X_decoded = torch.tensor(X_decoded)
    # decode_img = to_img(X_decoded).squeeze()
    # decode_img = decode_img.data.numpy() * 255
    # print(decode_img.size)
    # for i in range(16):
    #     plt.imshow(decode_img[i,:,:].astype('uint8'), cmap='gray')
    # plt.show()
    # print("X_decoded", X_decoded.size)
    # print("After")
    # X_decoded = X_decoded.reshape(16, 3, 28)
    # X_decoded = X_decoded.reshape(-1, 28, 28)

    # # plotting decoded
    for i in range(config.batch_size):
        # Plotting

        X_decoded_sample = X_decoded[i, :, :, :]  # torch.Size([3, 32, 32])
        # print("X_decoded_sample_raw", X_decoded_sample.size)
        # X_decoded_sample = X_decoded_sample.reshape(32, 32, 3)
        print("X_decoded_sample", X_decoded_sample.size())  # torch.Size([3, 32, 32])
        X_decoded_sample = X_decoded_sample / 2 + 0.5
        # x = x.clamp(0, 1)
        decode_img = X_decoded_sample.numpy() * 255
        # decode_img = decode_img.data.numpy() * 255
        # print("decode_img", decode_img.shape)
        # print("decode_img", type(decode_img))  # ([3, 32, 32])
        # decode_img = decode_img.reshape(32,)*255
        # decode_img = decode_img.reshape(32, 32, 3)
        # print("decode_img", decode_img.shape)
        # print("decode_img", type(decode_img))
        plt.imshow(decode_img.transpose((1, 2, 0)).astype('uint8'))

        # imgplot = plt.imshow(X_decoded_sample)
        plt.savefig("./CIFAR10_model_dir/{}/plots/{}_compare.png".format(str(Model_Num), str(i)))
        plt.show()
    # print("After")
