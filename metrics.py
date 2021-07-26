import numpy as np
general = "ECG5000"
# TODO ECG5000
general_labels = np.load("data_for_dash/{}_labels.npy".format(general))
general_y_pred_total = np.load("data_for_dash/{}_y_pred_total.npy".format(general))
general_z_run_sep = np.load("data_for_dash/{}_z_run_sep.npy".format(general))