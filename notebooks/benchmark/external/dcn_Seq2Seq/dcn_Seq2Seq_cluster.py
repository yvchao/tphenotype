# coding: utf-8
# pylint: disable=unspecified-encoding

import os
import random
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from class_Seq2Seq import DCN_Seq2Seq

# performance metrics
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    normalized_mutual_info_score,
    roc_auc_score,
)
from sklearn.metrics.cluster import contingency_matrix
from sklearn.model_selection import train_test_split

sys.path.append("../..")

data_mode = "CF_comorbidity_select"  # 'CF_comorbidity_select'

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


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    c_matrix = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(c_matrix, axis=0)) / np.sum(c_matrix)  # pyright: ignore


def get_all_x(x_):
    tmp_length = np.sum(np.sum(np.abs(x_), axis=2) != 0, axis=1)

    tmp_x_ = np.zeros([np.shape(x_)[0] * max_length, max_length, x_dim])
    for i in range(np.shape(x_)[0]):
        for t in range(tmp_length[i]):
            tmp_x_[(i * max_length) + t, : (t + 1), :] = x_[i, : (t + 1), :]

    tmp_x_ = tmp_x_[np.sum(np.sum(np.abs(tmp_x_), axis=2), axis=1) != 0]
    return tmp_x_


x_dim = np.shape(data_x)[2]  # pyright: ignore
y_dim = np.shape(data_y)[2]  # pyright: ignore
max_length = np.shape(data_x)[1]  # pyright: ignore
rnn_type = "LSTM"  # GRU, LSTM

OUT_ITERATION = 5

seed = 1234

num_Cluster = 14
K = num_Cluster

lr_rate = 1e-4
keep_prob = 0.7
mb_size = 128

alpha = 0.1  # L_CLUSTER

ITERATION = 10000
check_step = 100

RESULT_NMI = np.zeros([OUT_ITERATION, 1])
RESULT_RI = np.zeros([OUT_ITERATION, 1])
RESULT_PURITY = np.zeros([OUT_ITERATION, 1])

for out_itr in [0, 1, 2, 3, 4]:
    tr_data_x, te_data_x, tr_data_y, te_data_y = train_test_split(
        data_x, data_y, test_size=0.2, random_state=seed + out_itr  # pyright: ignore
    )

    tr_data_x, va_data_x, tr_data_y, va_data_y = train_test_split(
        tr_data_x, tr_data_y, test_size=0.2, random_state=seed + out_itr
    )

    load_path = "../../{}/dcn_S2S/init/itr{}/".format(data_mode, out_itr)

    input_dims = {"x_dim": x_dim, "y_dim": y_dim, "max_length": max_length}

    tf.reset_default_graph()

    # Turn on xla optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    network_settings = load_logging(load_path + "models/network_settings_v3.txt")

    z_dim = network_settings["h_dim_encoder"] * network_settings["num_layers_encoder"]
    model = DCN_Seq2Seq(sess, "dcn_S2S", input_dims, network_settings)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    saver.restore(sess, load_path + "models/dcn_S2S_init_v3")

    # -- CLUSTER INITIALIZATION
    km = MiniBatchKMeans(n_clusters=num_Cluster, batch_size=mb_size)
    tr_z = model.predict_Z(tr_data_x)

    _ = km.fit(tr_z)
    mu = km.cluster_centers_

    avg_loss_total = 0
    avg_loss_ae = 0
    avg_loss_clu = 0
    for itr in range(ITERATION):
        x_mb, y_mb = f_get_minibatch(mb_size, tr_data_x, tr_data_y)

        s_mb = km.predict(model.predict_Z(x_mb))
        _, tmp_loss_total, tmp_loss_ae, tmp_loss_clu = model.train_total(
            x_mb, y_mb, s_mb, mu, num_Cluster, 0.001, alpha, lr_rate, keep_prob
        )

        avg_loss_total += tmp_loss_total / check_step
        avg_loss_ae += tmp_loss_ae / check_step
        avg_loss_clu += tmp_loss_clu / check_step

        if (itr + 1) % check_step == 0:
            km = MiniBatchKMeans(n_clusters=num_Cluster, batch_size=mb_size, init=mu)
            tr_z = model.predict_Z(tr_data_x)

            _ = km.fit(tr_z)
            mu = km.cluster_centers_

            print(
                "ITR {}: loss_total={:.4f}\t loss_mle={:.4f} \t loss_clue={:.4f}".format(
                    itr + 1, avg_loss_total, avg_loss_ae, avg_loss_clu
                )
            )
            avg_loss_total = 0
            avg_loss_ae = 0
            avg_loss_clu = 0

    save_path = "../../{}/dcn_S2S/K{}/itr{}/".format(data_mode, K, out_itr)

    if not os.path.exists(save_path + "/models/"):
        os.makedirs(save_path + "/models/")

    if not os.path.exists(save_path + "/results/"):
        os.makedirs(save_path + "/results/")

    saver.save(sess, save_path + "models/dcn_S2S_clustered_v3")

    save_logging(network_settings, save_path + "models/network_settings.txt")
    np.savez(save_path + "models/embeddings.npz", km=km, mu=mu)  # pyright: ignore

    # -- CLUSTERING PERFORMANCE CHECK
    tmp_x = get_all_x(te_data_x)
    tmp_z = model.predict_Z(tmp_x)

    _, _, _, tmp_m = model.predict_outputs(te_data_x)

    z_dim = np.shape(tmp_z)[-1]

    # tmp_z  = tmp_z.reshape([-1, z_dim])[tmp_m.reshape([-1]) == 1]
    pred_y = km.predict(tmp_z)

    true_y = (te_data_y * np.tile(np.expand_dims(tmp_m, axis=2), [1, 1, y_dim])).reshape([-1, y_dim])  # pyright: ignore
    true_y = true_y[(tmp_m.reshape([-1]) == 1)]
    true_y = np.argmax(true_y, axis=1)

    tmp_nmi = normalized_mutual_info_score(true_y, pred_y)
    tmp_ri = adjusted_rand_score(true_y, pred_y)
    tmp_purity = purity_score(true_y, pred_y)

    pd.DataFrame([[tmp_nmi, tmp_ri, tmp_purity]], columns=["NMI", "RI", "PURITY"], index=["itr" + str(out_itr)]).to_csv(
        save_path + "results/nmi_ir_purity.csv"
    )

    print("ITR{} - K{} |  NMI:{:.4f}, RI:{:.4f}, PURITY:{:.4f}".format(out_itr, K, tmp_nmi, tmp_ri, tmp_purity))

    RESULT_NMI[out_itr, 0] = tmp_nmi
    RESULT_RI[out_itr, 0] = tmp_ri
    RESULT_PURITY[out_itr, 0] = tmp_purity

pd.DataFrame(RESULT_NMI, columns=["NMI"], index=["itr" + str(out_itr) for out_itr in range(OUT_ITERATION)]).to_csv(
    "../../{}/dcn_S2S/K{}/".format(data_mode, K) + "results_nmi.csv"
)

pd.DataFrame(RESULT_RI, columns=["RI"], index=["itr" + str(out_itr) for out_itr in range(OUT_ITERATION)]).to_csv(
    "../../{}/dcn_S2S/K{}/".format(data_mode, K) + "results_ri.csv"
)

pd.DataFrame(
    RESULT_PURITY, columns=["PURITY"], index=["itr" + str(out_itr) for out_itr in range(OUT_ITERATION)]
).to_csv("../../{}/dcn_S2S/K{}/".format(data_mode, K) + "results_purity.csv")
