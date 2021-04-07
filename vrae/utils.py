from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import numpy as np
from random import randint
import os
import matplotlib.pyplot as plt

from plotly.graph_objs import *
import plotly
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, MeanShift, DBSCAN
from sklearn import metrics
from sklearn import svm
from sklearn import mixture
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
from scipy.io import arff


# def svm_latent_space(data, label):
#     clf = svm.SVC()
#     clf.fit(data, label)
#     return clf

def plot_clustering(z_run, labels, engine='plotly', download=False, folder_name='clustering'):
    """
    Given latent variables for all timeseries, and output of k-means, run PCA and tSNE on latent vectors and color the points using cluster_labels.
    :param z_run: Latent vectors for all input tensors
    :param labels: Cluster labels for all input tensors
    :param engine: plotly/matplotlib
    :param download: If true, it will download plots in `folder_name`
    :param folder_name: Download folder to dump plots
    :return:
    """

    def plot_clustering_plotly(z_run, labels):

        labels = labels[:z_run.shape[0]]  # because of weird batch_size

        hex_colors = []
        for _ in np.unique(labels):
            hex_colors.append('#%06X' % randint(0, 0xFFFFFF))

        colors = [hex_colors[int(i)] for i in labels]

        z_run_pca = TruncatedSVD(n_components=12).fit_transform(z_run)
        z_run_tsne = TSNE(perplexity=80, min_grad_norm=1E-12, n_iter=3000).fit_transform(z_run)

        trace = Scatter(
            x=z_run_pca[:, 0],
            y=z_run_pca[:, 1],
            mode='markers',
            marker=dict(color=colors)
        )

        data = Data([trace])
        layout = Layout(
            title='PCA on z_run',
            showlegend=False
        )
        fig = Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)

        trace = Scatter(
            x=z_run_tsne[:, 0],
            y=z_run_tsne[:, 1],
            mode='markers',
            marker=dict(color=colors)
        )
        data = Data([trace])
        layout = Layout(
            title='tSNE on z_run',
            showlegend=False
        )
        fig = Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)

    def plot_clustering_matplotlib(z_run, labels, download, folder_name, image_mode=False):
        # print("z_run type", type(z_run))
        # print("z_run shape", z_run.shape)
        # print("labels type", type(labels))
        # print("labels shape", len(labels))
        labels = labels[:z_run.shape[0]]  # because of weird batch_size
        labels = np.squeeze(labels)
        print("z_run.shape", z_run.shape)
        print("labels", labels)

        # labels = [t.numpy() for t in labels]
        labels_unique = np.unique(labels)
        print("labels_unique", labels_unique)
        n_clusters_ = len(labels_unique)
        print("n_clusters_", n_clusters_)
        hex_colors = dict()
        # labels = [t[0].astype(int) for t in labels]
        for label in labels_unique:
            hex_colors[int(label)] = ('#%06X' % randint(0, 0xFFFFFF))
        colors = [hex_colors[int(i)] for i in labels]
        print("colors", colors)
        z_run_pca = TruncatedSVD(n_components=10).fit_transform(z_run)
        z_run_tsne = TSNE(perplexity=80, min_grad_norm=1E-12, n_iter=3000).fit_transform(z_run)
        # SpectralClustering
        clustering_name = "Kmeans"
        # sc_model = SpectralClustering(n_clusters=n_clusters_)
        # y_pred = sc_model.fit_predict(z_run)

        # Kmean
        kmeans_model = KMeans(n_clusters=n_clusters_, random_state=0)
        y_pred = kmeans_model.fit_predict(z_run)  # Labels of each point

        # Mean Shift
        # meanshift_model = MeanShift(bandwidth=n_clusters_)
        # y_pred = meanshift_model.fit_predict(z_run)

        # # GMM
        # GMM_model = mixture.GaussianMixture(n_components=n_clusters_, covariance_type="full")
        # y_pred = GMM_model.fit_predict(z_run)

        # DBSCAN
        # DBSCAN_model = DBSCAN(eps = 0.01)
        # y_pred = DBSCAN_model.fit_predict(z_run)

        print("y_pred", y_pred)
        prefict_unique = np.unique(y_pred)
        print("prefict_unique", prefict_unique)
        hex_colors_pre = dict()
        for pre in prefict_unique:
            hex_colors_pre[int(pre)] = ('#%06X' % randint(0, 0xFFFFFF))
        y_pred_colors = [hex_colors_pre[int(i)] for i in y_pred]
        print("y_pred_colors", y_pred_colors)
        # # OPTICS
        # OPTICS_model = OPTICS(eps=0.8, min_samples=10)
        # y_pred = OPTICS_model.fit_predict(z_run)

        # cls = svm_latent_space(z_run, labels)
        # cls.predict()
        # clustering latent variable
        latent_dataset = {"z_run_pca": z_run_pca, "z_run_tsne": z_run_tsne}
        for name, z_run_sep in latent_dataset.items():

            # Kmean
            # kmeans_model = KMeans(n_clusters=n_clusters_, random_state=1)
            # y_pred = kmeans_model.fit_predict(z_run_sep) #Labels of each point

            # name
            # # Mean Shift
            # meanshift_model = MeanShift(bandwidth=n_clusters_)
            # y_pred = meanshift_model.fit_predict(z_run_sep)
            # SVM

            # metrics.silhouette_score(z_run_sep, labels, metric='euclidean')
            labels_for_metrics = np.squeeze(labels)
            print("labels_for_metrics", labels_for_metrics.shape)
            print("y_pred", y_pred.shape)
            accuracy = metrics.adjusted_mutual_info_score(labels_for_metrics, y_pred)  # 完全一样则为1，也可能为0
            f1_final_score = f1_score(y_true=labels, y_pred=y_pred, average='weighted')
            recall_final_score = recall_score(y_true=labels, y_pred=y_pred, average='weighted')
            accuracy_final_score = accuracy_score(y_true=labels, y_pred=y_pred)
            precision_final_score = precision_score(y_true=labels, y_pred=y_pred, average='weighted')

            accuracy = round(accuracy, 2)
            f1_final_score = round(f1_final_score, 2)
            accuracy_final_score = round(accuracy_final_score, 2)
            precision_final_score = round(precision_final_score, 2)
            recall_final_score = round(recall_final_score, 2)
            print(
                "***************the accuracy of the clustering {} and dim_red {} is: {}************************".format(
                    "Kmeans", name, str(accuracy)))

            plt.scatter(z_run_sep[:, 0], z_run_sep[:, 1], c=y_pred_colors, marker='+')
            title = "predict_clustering_{} on {} ".format(clustering_name, name) + " Acc: " + str(
                accuracy) + " f1_score: {}, recall_score: {}, accuracy_score: {}, precision_score: {}.png".format(
                f1_final_score, recall_final_score, accuracy_final_score, precision_final_score)
            plt.title(title)
            if download:
                if os.path.exists(folder_name):
                    pass
                else:
                    os.mkdir(folder_name)
                plt.savefig(folder_name + "/" + title)
            else:
                plt.show()
            plt.clf()

            plt.scatter(z_run_sep[:, 0], z_run_sep[:, 1], c=colors, marker='*', linewidths=0)
            title = "Groundtruth on " + name + ".png"
            plt.title(title)
            if download:
                if os.path.exists(folder_name):
                    pass
                else:
                    os.mkdir(folder_name)
                plt.savefig(folder_name + "/" + title)
            else:
                plt.show()
            plt.clf()

            # both together

            title = "Groundtruth and predict clustering " + name + ".png"
            plt.subplot(1, 2, 1)
            plt.scatter(z_run_sep[:, 0], z_run_sep[:, 1], c=y_pred_colors, marker='+')
            plt.title("predict")
            plt.subplot(1, 2, 2)
            plt.scatter(z_run_sep[:, 0], z_run_sep[:, 1], c=colors, marker='*', linewidths=0)
            plt.title("groundtruth")
            if download:
                if os.path.exists(folder_name):
                    pass
                else:
                    os.mkdir(folder_name)
                plt.savefig(folder_name + "/" + title)
            else:
                plt.show()
            plt.clf()

        # if download:
        #     if os.path.exists(folder_name):
        #         pass
        #     else:
        #         os.mkdir(folder_name)
        #     plt.savefig(folder_name + "/pca.png")
        # else:
        #     plt.show()

        # plt.scatter(z_run_tsne[:, 0], z_run_tsne[:, 1], c=colors, marker='*', linewidths=0)
        # plt.title('tSNE on z_run')
        # if download:
        #     if os.path.exists(folder_name):
        #         pass
        #     else:
        #         os.mkdir(folder_name)
        #     plt.savefig(folder_name + "/tsne.png")
        # else:
        #     plt.show()

    if (download == False) & (engine == 'plotly'):
        plot_clustering_plotly(z_run, labels)
    if (download) & (engine == 'plotly'):
        print("Can't download plotly plots")
    if engine == 'matplotlib':
        plot_clustering_matplotlib(z_run, labels, download, folder_name)


def open_data(direc, ratio_train=0.8, dataset="ECG5000"):
    """Input:
    direc: location of the UCR archive
    ratio_train: ratio to split training and testset
    dataset: name of the dataset in the UCR archive"""
    datadir = direc + '/' + dataset + '/' + dataset
    data_train = np.loadtxt(datadir + '_TRAIN', delimiter=',')
    data_test_val = np.loadtxt(datadir + '_TEST', delimiter=',')[:-1]
    data = np.concatenate((data_train, data_test_val), axis=0)
    data = np.expand_dims(data, -1)

    N, D, _ = data.shape

    ind_cut = int(ratio_train * N)
    ind = np.random.permutation(N)
    return data[ind[:ind_cut], 1:, :], data[ind[ind_cut:], 1:, :], data[ind[:ind_cut], 0, :], data[ind[ind_cut:], 0, :]


def open_newdata(direc, ratio_train=0.8, dataset="ECG200"):
    """Input:
    direc: location of the UCR archive
    ratio_train: ratio to split training and testset
    dataset: name of the dataset in the UCR archive"""
    datadir = direc + '/' + dataset
    data_train = np.loadtxt(datadir + '_TRAIN.csv', delimiter=',')
    data_test_val = np.loadtxt(datadir + '_TEST.csv', delimiter=',')[:-1]
    data = np.concatenate((data_train, data_test_val), axis=0)
    data = np.expand_dims(data, -1)

    N, D, _ = data.shape

    ind_cut = int(ratio_train * N)
    ind = np.random.permutation(N)
    return data[ind[:ind_cut], :-1, :], data[ind[ind_cut:], :-1, :], data[ind[:ind_cut], -1, :], data[ind[ind_cut:], -1,
                                                                                                 :]


def open_Cricketdata(direc, ratio_train=0.8, dataset="Cricket"):
    """Input:
    direc: location of the UCR archive
    ratio_train: ratio to split training and testset
    dataset: name of the dataset in the UCR archive"""
    datadir = direc + '/' + dataset
    data_train = np.genfromtxt(datadir + '_TRAIN.csv')
    data_test_val = np.genfromtxt(datadir + '_TEST.csv')[:-1]
    data = np.concatenate((data_train, data_test_val), axis=0)
    data = np.expand_dims(data, -1)

    N, D, _ = data.shape

    ind_cut = int(ratio_train * N)
    ind = np.random.permutation(N)
    return data[ind[:ind_cut], 1:, :], data[ind[ind_cut:], 1:, :], data[ind[:ind_cut], 0, :], data[ind[ind_cut:], 0, :]


def open_Cricketdata_6d(direc, ratio_train=0.8, dataset="CricketDimension"):
    """Input:
    direc: location of the UCR archive
    ratio_train: ratio to split training and testset
    dataset: name of the dataset in the UCR archive"""

    from scipy.io import arff


    # data = arff.loadarff('dataset.arff')

    # df = pd.DataFrame(data[0])

    datadir = direc + '/' + dataset
    data_train = np.array(arff.loadarff(datadir + '{}_TRAIN.arff'.format(1))[0].tolist(), dtype=np.float)
    data_test_val = np.array(arff.loadarff(datadir + '{}_TEST.arff'.format(1))[0].tolist(), dtype=np.float)[:-1]

    length = 1197
    num_of_classes = 12
    dim = 6
    train_set_volume = 108
    test_set_volume = 71

    label_data = np.concatenate((data_train, data_test_val), axis=0)
    label_data = np.expand_dims(label_data, -1)

    N, D, _ = label_data.shape

    ind_cut = int(ratio_train * N)
    ind = np.random.permutation(N)

    y_train, y_val = label_data[ind[:ind_cut], -1, :], label_data[ind[ind_cut:], -1, :]

    # data_train_whole = np.zeros((train_set_volume * dim, length))
    # data_test_val_whole = np.zeros((test_set_volume * dim, length))
    data_train_whole = np.empty((0, length), dtype=float)
    data_test_val_whole = np.empty((0, length), dtype=float)
    data_train_tmp = np.empty((0, length), dtype=float)
    data_test_val_tmp = np.empty((0, length), dtype=float)
    for i in range(1, dim + 1):
        data_train = np.array(arff.loadarff(datadir + '{}_TRAIN.arff'.format(i))[0].tolist(), dtype=np.float)
        data_test_val = np.array(arff.loadarff(datadir + '{}_TEST.arff'.format(i))[0].tolist(), dtype=np.float)[:-1]
        data_train_tmp = np.append(data_train_tmp, data_train[:, :-1], axis=0)
        data_test_val_tmp = np.append(data_test_val_tmp, data_test_val[:, :-1], axis=0)

    for set_i in range(train_set_volume):
        for dim_i in range(1, dim + 1):
            tmp = np.expand_dims(data_train_tmp[set_i * dim_i, :], axis=0)
            data_train_whole = np.append(data_train_whole, tmp, axis=0)

    for set_i in range(test_set_volume):
        for dim_i in range(1, dim + 1):
            tmp = np.expand_dims(data_test_val_tmp[set_i * dim_i, :], axis=0)
            data_test_val_whole = np.append(data_test_val_whole, tmp, axis=0)

    whole_data = np.concatenate((data_train_whole, data_test_val_whole), axis=0)
    whole_data = np.expand_dims(whole_data, -1)
    whole_data_final_X_train = np.empty((0, length), dtype=float)
    whole_data_final_X_train = np.expand_dims(whole_data_final_X_train, -1)
    whole_data_final_X_val = np.empty((0, length), dtype=float)
    whole_data_final_X_val = np.expand_dims(whole_data_final_X_val, -1)
    idx_train = ind[:ind_cut] * dim
    idx_test = ind[ind_cut:] * dim
    for tmp_idx_train in idx_train:
        whole_data_final_X_train = np.append(whole_data_final_X_train,
                                             whole_data[tmp_idx_train: tmp_idx_train + 6, :, :], axis=0)

    for tmp_idx_test in idx_test:
        whole_data_final_X_val = np.append(whole_data_final_X_val, whole_data[tmp_idx_test: tmp_idx_test + 6, :, :],
                                           axis=0)

    return whole_data_final_X_train, whole_data_final_X_val, y_train, y_val


def open_newdata_ED(direc, ratio_train=0.8, dataset="ElectricDevices"):
    """Input:
    direc: location of the UCR archive
    ratio_train: ratio to split training and testset
    dataset: name of the dataset in the UCR archive"""
    datadir = direc + '/' + dataset
    data_train = np.loadtxt(datadir + '_TRAIN.csv', delimiter=',')
    data_test_val = np.loadtxt(datadir + '_TEST.csv', delimiter=',')[:-1]
    data = np.concatenate((data_train, data_test_val), axis=0)
    data = np.expand_dims(data, -1)

    N, D, _ = data.shape

    ind_cut = int(ratio_train * N)
    ind = np.random.permutation(N)
    return data[ind[:ind_cut], :-1, :], data[ind[ind_cut:], :-1, :], data[ind[:ind_cut], -1, :], data[ind[ind_cut:], -1,
                                                                                                 :]


def cvs_to_numpy(direc, ratio_train=0.8, dataset="ECG5000"):
    datadir = direc + '/' + dataset + '/' + dataset


if __name__ == "__main__":
    cvs_to_numpy(direc='data', ratio_train=0.9, dataset="normalized")
