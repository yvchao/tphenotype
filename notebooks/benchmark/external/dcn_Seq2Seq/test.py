import os
import random
import sys

fd = os.open("/dev/null", os.O_WRONLY)
os.dup2(fd, 2)

_EPSILON = 1e-08

# In[2]:
import argparse
import copy
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle5 as pickle
import tensorflow as tf
import utils_network as utils
from class_Seq2Seq import DCN_Seq2Seq
from pandas.api.types import is_numeric_dtype, is_string_dtype

# performance metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    homogeneity_score,
    normalized_mutual_info_score,
    roc_auc_score,
)
from sklearn.metrics.cluster import contingency_matrix
from sklearn.model_selection import train_test_split
from tensorflow.contrib.layers import fully_connected as FC_Net
from tensorflow.python.ops.rnn import _transpose_batch_time


### PARAMETER LOGGING
def save_logging(dictionary, log_name):
    with open(log_name, "w") as f:
        for key, value in dictionary.items():
            if "activate_fn" in key:
                value = str(value).split(" ")[1]

            f.write("%s:%s\n" % (key, value))


def load_logging(filename):
    data = dict()
    with open(filename) as f:

        def is_float(input):
            try:
                num = float(input)
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


# In[5]:


def log(x):
    return tf.log(x + 1e-8)


def div(x, y):
    return tf.div(x, (y + 1e-8))


def get_seq_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    tmp_length = tf.reduce_sum(used, 1)
    tmp_length = tf.cast(tmp_length, tf.int32)
    return tmp_length


def f_get_minibatch(mb_size, x, y):
    idx = range(np.shape(x)[0])
    idx = random.sample(idx, mb_size)

    x_mb = x[idx].astype(float)
    y_mb = y[idx].astype(float)

    return x_mb, y_mb


def get_all_x(x_):
    tmp_length = np.sum(np.sum(np.abs(x_), axis=2) != 0, axis=1)

    tmp_x = np.zeros([np.shape(x_)[0] * max_length, max_length, x_dim])
    for i in range(np.shape(x_)[0]):
        for t in range(tmp_length[i]):
            # only take the last observation
            if t != tmp_length[i] - 1:
                continue
            tmp_x[(i * max_length) + t, : (t + 1), :] = x_[i, : (t + 1), :]

    tmp_x = tmp_x[np.sum(np.sum(np.abs(tmp_x), axis=2), axis=1) != 0]
    return tmp_x


### PERFORMANCE METRICS:
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
    return np.sum(np.amax(c_matrix, axis=0)) / np.sum(c_matrix)


def data_to_stats(x):
    if is_numeric_dtype(x):
        return f"{np.mean(x):.2f}±{np.std(x):.2f}"
    else:
        return x[0]


def set_random_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--cluster-number", default=4, type=int)
    parser.add_argument("-d", "--data", default="ADNI", type=str)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    args = parser.parse_args()

    K = args.cluster_number
    data = args.data
    epochs = args.epochs

    data_mode = data

    with open(f"../../data/{data}_data.pkl", "rb") as file:
        splits = pickle.load(file)

    OUT_ITERATION = len(splits)

    RESULT_NMI = np.zeros([OUT_ITERATION, 1])
    RESULT_RI = np.zeros([OUT_ITERATION, 1])
    RESULT_PURITY = np.zeros([OUT_ITERATION, 1])

    seed = args.seed
    set_random_seed(seed)

    # Initialization
    data_mode = data

    h_dim_FC = 50  # for fully_connected layers
    h_dim_RNN = 50

    num_layer_encoder = 1
    num_layer_predictor = 2

    x_dim = np.shape(splits[0][0]["x"])[2] + 1  # plus 1 because of delta t
    y_dim = np.shape(splits[0][0]["y"])[2]

    max_length = np.shape(splits[0][0]["x"])[1]
    y_type = "categorical"

    z_dim = h_dim_RNN * num_layer_encoder

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

    set_random_seed(seed)

    lr_rate = 1e-4
    keep_prob = 1.0
    mb_size = 128

    alpha = 0.1  # L_CLUSTER

    RESULT_NMI = np.zeros([OUT_ITERATION, 1])
    RESULT_RI = np.zeros([OUT_ITERATION, 1])
    RESULT_PURITY = np.zeros([OUT_ITERATION, 1])

    RESULT_AUROC = np.zeros([OUT_ITERATION, y_dim])
    RESULT_AUPRC = np.zeros([OUT_ITERATION, y_dim])

    dfs = []

    num_Cluster = K
    model_preds = []

    for out_itr in range(OUT_ITERATION):
        print("======= K: {}   OUT_ITERATION: {} ======".format(K, out_itr))
        print("load data")
        dataset = splits[out_itr]
        dataset = copy.deepcopy(dataset)
        for subset in dataset:
            x = subset["x"]
            t = subset["t"]
            m = subset["range_mask"]
            delta_t = np.zeros((*t.shape, 1))
            delta_t[:, 1:, 0] = t[:, 1:] - t[:, :-1]
            delta_t[m == 0] = 0
            x = np.concatenate([delta_t, x], axis=-1)
            subset["x"] = x

        train_set, valid_set, test_set = dataset

        tr_data_x, tr_data_y = train_set["x"], train_set["y"]
        va_data_x, va_data_y = valid_set["x"], valid_set["y"]
        te_data_x, te_data_y = test_set["x"], test_set["y"]

        load_path = "./{}/dcn_S2S/K{}/itr{}/".format(data_mode, K, out_itr)

        input_dims = {"x_dim": x_dim, "y_dim": y_dim, "max_length": max_length}

        tf.reset_default_graph()

        # Turn on xla optimization
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_random_seed(seed)
        sess = tf.Session(config=config)

        network_settings = load_logging(load_path + "models/network_settings.txt")

        z_dim = network_settings["h_dim_encoder"] * network_settings["num_layers_encoder"]
        model = DCN_Seq2Seq(sess, "dcn_S2S", input_dims, network_settings)

        saver = tf.train.Saver()

        saver.restore(sess, load_path + "models/dcn_S2S_clustered_v3")

        npz = np.load(load_path + "models/embeddings.npz", allow_pickle=True)
        km = npz["km"].item()
        probs = npz["probs"]

        model_output = {}

        model_output["method"] = "SEQ2SEQ"

        ### CLUSTERING PERFORMANCE CHECK
        tmp_x = get_all_x(te_data_x)
        tmp_z = model.predict_Z(tmp_x)

        _, _, _, tmp_m = model.predict_outputs(te_data_x)

        z_dim = np.shape(tmp_z)[-1]

        # tmp_z  = tmp_z.reshape([-1, z_dim])[tmp_m.reshape([-1]) == 1]
        pred_y = km.predict(tmp_z)

        pred_y_full = np.zeros(te_data_y.shape[:2])
        pred_y_full[tmp_m == 1] = pred_y

        # cluster  prediction
        model_output["c_pred"] = pred_y_full

        pred_y = (pred_y_full * tmp_m).reshape([-1, 1])
        pred_y = pred_y[(tmp_m.reshape([-1, 1]) == 1)[:, 0], 0]

        true_y = (te_data_y * np.tile(np.expand_dims(tmp_m, axis=2), [1, 1, y_dim])).reshape([-1, y_dim])
        true_y = true_y[(tmp_m.reshape([-1]) == 1)]
        true_y = np.argmax(true_y, axis=1)

        tmp_nmi = normalized_mutual_info_score(true_y, pred_y)
        tmp_ri = adjusted_rand_score(true_y, pred_y)
        tmp_purity = purity_score(true_y, pred_y)

        RESULT_NMI[out_itr, 0] = tmp_nmi
        RESULT_RI[out_itr, 0] = tmp_ri
        RESULT_PURITY[out_itr, 0] = tmp_purity

        df = pd.DataFrame()
        df.loc[0, "Model"] = "AC-TPC"
        df.loc[0, "Purity"] = RESULT_PURITY[out_itr, 0]
        df.loc[0, "NMI"] = RESULT_NMI[out_itr, 0]
        df.loc[0, "ARI"] = RESULT_RI[out_itr, 0]

        # cluster-based label prediction
        tmp_y = np.zeros_like(te_data_y)
        for k in range(K):
            tmp_y[pred_y_full == k] = probs[k]
        model_output["y_pred"] = tmp_y

        y_pred = tmp_y.reshape([-1, y_dim])[tmp_m.reshape([-1]) == 1]
        y_true = te_data_y.reshape([-1, y_dim])[tmp_m.reshape([-1]) == 1]

        AUROC = np.zeros([y_dim])
        AUPRC = np.zeros([y_dim])
        for y_idx in range(y_dim):
            auroc, auprc = f_get_prediction_scores(y_true[:, y_idx], y_pred[:, y_idx])
            AUROC[y_idx] = auroc
            AUPRC[y_idx] = auprc

        RESULT_AUROC[out_itr, :] = AUROC
        RESULT_AUPRC[out_itr, :] = AUPRC
        df.loc[0, "AUROC"] = np.mean(RESULT_AUROC[out_itr, :])
        df.loc[0, "AUPRC"] = np.mean(RESULT_AUPRC[out_itr, :])
        dfs.append(df)

        model_preds.append(model_output)

    results = pd.concat(dfs).reset_index(drop=True)
    summary = results.apply(data_to_stats).to_frame().T

    # In[22]:
    print(summary)

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"./{output_dir}/{data}_preds.pkl", "wb") as file:
        pickle.dump(model_preds, file, pickle.HIGHEST_PROTOCOL)

    print(f"model prediction saved to output/{data}_preds.pkl")
