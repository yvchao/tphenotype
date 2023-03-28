# coding: utf-8
# pylint: disable=unspecified-encoding

import os
import random
import sys

import numpy as np
import tensorflow as tf
from class_Seq2Seq import DCN_Seq2Seq

# performance metrics
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.model_selection import train_test_split

sys.path.append("../..")

data_mode = "ADNI"  # 'CF_comorbidity_select'

# IMPORT DATASET
if data_mode == "CF":
    npz = np.load("../../data/CF/data.npz")

    data_x = npz["data_x"]
    data_y = npz["data_y"]
    data_y_onehot = npz["data_y_onehot"]
    feat_list = npz["feat_list"]

    data_x = data_x[:, :, 1:]

elif data_mode == "CF_comorbidity":
    npz = np.load("../../data/CF_comorbidity/data_como.npz")

    data_x = npz["data_x"]
    data_y = npz["data_y"]
    feat_list = npz["feat_list"]
    label_list = npz["label_list"]
    selected_list = npz["selected_list"]

    data_y_selected = data_y[:, :, np.where([f in selected_list for f in label_list])[0]]

elif data_mode == "CF_comorbidity_select":
    npz = np.load("../../data/CF_comorbidity/data_como.npz")

    data_x = npz["data_x"]
    data_y = npz["data_y"]
    feat_list = npz["feat_list"]
    label_list = npz["label_list"]
    selected_list = npz["selected_list"]

    data_y = data_y[:, :, np.where([f in selected_list for f in label_list])[0]]
    label_list = selected_list

    tmp_onehot = np.zeros([np.shape(data_y)[0], np.shape(data_y)[1], 8])

    tmp_onehot[np.sum(data_y == [0, 0, 0], axis=2) == 3] = [1, 0, 0, 0, 0, 0, 0, 0]
    tmp_onehot[np.sum(data_y == [0, 0, 1], axis=2) == 3] = [0, 1, 0, 0, 0, 0, 0, 0]
    tmp_onehot[np.sum(data_y == [0, 1, 0], axis=2) == 3] = [0, 0, 1, 0, 0, 0, 0, 0]
    tmp_onehot[np.sum(data_y == [0, 1, 1], axis=2) == 3] = [0, 0, 0, 1, 0, 0, 0, 0]
    tmp_onehot[np.sum(data_y == [1, 0, 0], axis=2) == 3] = [0, 0, 0, 0, 1, 0, 0, 0]
    tmp_onehot[np.sum(data_y == [1, 0, 1], axis=2) == 3] = [0, 0, 0, 0, 0, 1, 0, 0]
    tmp_onehot[np.sum(data_y == [1, 1, 0], axis=2) == 3] = [0, 0, 0, 0, 0, 0, 1, 0]
    tmp_onehot[np.sum(data_y == [1, 1, 1], axis=2) == 3] = [0, 0, 0, 0, 0, 0, 0, 1]

    tmp_onehot[np.sum(np.abs(data_x), axis=2) == 0] = [0, 0, 0, 0, 0, 0, 0, 0]  # put all 0's for not selected ones..

    data_y = tmp_onehot

elif data_mode == "ADNI":
    npz = np.load("../../data/ADNI/data.npz")

    data_x = npz["data_x"]
    data_y = npz["data_y"]
    feat_list = npz["feat_list"]
    label_list = npz["label_list"]


# -- PARAMETER LOGGING
def save_logging(dictionary, log_name):
    with open(log_name, "w") as f:
        for key, value in dictionary.items():
            if "activate_fn" in key:
                value = str(value).split(" ")[1]

            f.write("%s:%s\n" % (key, value))


def load_logging(filename):
    data = dict()
    with open(filename) as f:

        def is_float(input_):
            try:
                _ = float(input_)
            except ValueError:
                return False
            return True

        for line in f.readlines():
            if ":" in line:
                key, value = line.strip().split(":", 1)

                if "activate_fn" in key:
                    if value == "relu":
                        value = tf.nn.relu
                    elif value == "elu":
                        value = tf.nn.elu
                    elif value == "tanh":
                        value = tf.nn.tanh
                    else:
                        raise ValueError("ERROR: wrong choice of activation function!")
                    data[key] = value
                else:
                    if value.isdigit():
                        data[key] = int(value)
                    elif is_float(value):
                        data[key] = float(value)
                    elif value == "None":
                        data[key] = None
                    else:
                        data[key] = value
            else:
                pass  # deal with bad lines of text here
    return data


# --- USER-DEFINED FUNCTIONS


def log(x):
    return tf.log(x + 1e-8)


def div(x, y):
    return tf.div(x, (y + 1e-8))


def get_seq_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    tmp_length = tf.reduce_sum(used, 1)
    tmp_length = tf.cast(tmp_length, tf.int32)
    return tmp_length


def f_get_minibatch(mb_size_, x, y):
    idx = range(np.shape(x)[0])
    idx = random.sample(idx, mb_size_)

    x_mb_ = x[idx, :, :].astype(float)
    y_mb_ = y[idx, :, :].astype(float)

    return x_mb_, y_mb_


def f_get_prediction_scores(y_true_, y_pred_):
    if np.sum(y_true_) == 0:  # no label for running roc_auc_curves
        auroc_ = -1.0
        auprc_ = -1.0
    else:
        auroc_ = roc_auc_score(y_true_, y_pred_)
        auprc_ = average_precision_score(y_true_, y_pred_)
    return (auroc_, auprc_)


def purity_score(y_true_, y_pred_):
    # compute contingency matrix (also called confusion matrix)
    c_matrix = contingency_matrix(y_true_, y_pred_)
    # return purity
    return np.sum(np.amax(c_matrix, axis=0)) / np.sum(c_matrix)  # pyright: ignore


# --- INITIALIZATION

h_dim_FC = 50  # for fully_connected layers
h_dim_RNN = 50

x_dim = np.shape(data_x)[2]  # pyright: ignore
y_dim = np.shape(data_y)[2]  # pyright: ignore

num_layer_encoder = 1
num_layer_predictor = 2

z_dim = h_dim_RNN * num_layer_encoder

max_length = np.shape(data_x)[1]  # pyright: ignore

rnn_type = "LSTM"  # GRU, LSTM

input_dims = {"x_dim": x_dim, "y_dim": y_dim, "max_length": max_length}

network_settings = {
    "h_dim_encoder": h_dim_RNN,
    "num_layers_encoder": num_layer_encoder,
    "rnn_type": rnn_type,
    "rnn_activate_fn": tf.nn.tanh,
    "h_dim_predictor": h_dim_FC,
    "num_layers_predictor": num_layer_predictor,
    "fc_activate_fn": tf.nn.relu,
}

OUT_ITERATION = 5

seed = 1234

ITERATION = 30000
check_step = 100

# for out_itr in [0,1,2,3,4]:
for out_itr in [4]:
    tr_data_x, te_data_x, tr_data_y, te_data_y = train_test_split(
        data_x, data_y, test_size=0.2, random_state=seed + out_itr  # pyright: ignore
    )

    tr_data_x, va_data_x, tr_data_y, va_data_y = train_test_split(
        tr_data_x, tr_data_y, test_size=0.2, random_state=seed + out_itr
    )

    tf.reset_default_graph()

    # Turn on xla optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    model = DCN_Seq2Seq(sess, "dcn_S2S", input_dims, network_settings)

    save_path = "../../{}/dcn_S2S/init/itr{}/".format(data_mode, out_itr)

    if not os.path.exists(save_path + "/models/"):
        os.makedirs(save_path + "/models/")

    if not os.path.exists(save_path + "/results/"):
        os.makedirs(save_path + "/results/")

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    lr_rate = 1e-3
    keep_prob = 0.7
    mb_size = 128

    avg_loss_ae = 0
    for itr in range(ITERATION):
        x_mb, y_mb = f_get_minibatch(mb_size, tr_data_x, tr_data_y)

        _, tmp_loss_ae = model.train_ae(x_mb, y_mb, 0.001, lr_rate, keep_prob)
        avg_loss_ae += tmp_loss_ae / check_step

        if (itr + 1) % check_step == 0:
            tmp_d, tmp_x, tmp_y, tmp_m = model.predict_outputs(va_data_x)

            y_pred = tmp_y.reshape([-1, y_dim])[tmp_m.reshape([-1]) == 1]
            y_true = va_data_y.reshape([-1, y_dim])[tmp_m.reshape([-1]) == 1]  # pyright: ignore

            AUROC = np.zeros([y_dim])
            AUPRC = np.zeros([y_dim])
            for y_idx in range(y_dim):
                auroc, auprc = f_get_prediction_scores(y_true[:, y_idx], y_pred[:, y_idx])
                AUROC[y_idx] = auroc
                AUPRC[y_idx] = auprc

            print(
                "ITR {}: loss_mle={:.4f} | va_auroc:{:.4f}, va_auprc:{:.4f}".format(
                    itr + 1, avg_loss_ae, np.mean(AUROC), np.mean(AUPRC)
                )
            )

            avg_loss_ae = 0

    saver.save(sess, save_path + "models/dcn_S2S_init_v3")
    save_logging(network_settings, save_path + "models/network_settings_v3.txt")

    tmp_d, tmp_x, tmp_y, tmp_m = model.predict_outputs(te_data_x)

    y_pred = tmp_y.reshape([-1, y_dim])[tmp_m.reshape([-1]) == 1]
    y_true = te_data_y.reshape([-1, y_dim])[tmp_m.reshape([-1]) == 1]  # pyright: ignore

    AUROC = np.zeros([y_dim])
    AUPRC = np.zeros([y_dim])
    for y_idx in range(y_dim):
        auroc, auprc = f_get_prediction_scores(y_true[:, y_idx], y_pred[:, y_idx])
        AUROC[y_idx] = auroc
        AUPRC[y_idx] = auprc

    print(AUROC)
    print(AUPRC)
