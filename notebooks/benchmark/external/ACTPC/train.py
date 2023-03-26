import copy
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fd = os.open("/dev/null", os.O_WRONLY)
os.dup2(fd, 2)

# In[2]:
import argparse

import pickle5 as pickle
import tensorflow as tf

# user defined
import utils_network as utils
from class_model_v7 import DeepTPC_ICLR

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
    print("Initialize models")
    h_dim_FC = 50  # for fully_connected layers
    h_dim_RNN = 50

    num_layer_encoder = 1
    num_layer_selector = 2
    num_layer_predictor = 2

    x_dim = np.shape(splits[0][0]["x"])[2] + 1  # plus 1 because of delta t
    y_dim = np.shape(splits[0][0]["y"])[2]

    max_length = np.shape(splits[0][0]["x"])[1]
    y_type = "categorical"

    z_dim = h_dim_RNN * num_layer_encoder

    rnn_type = "LSTM"  # GRU, LSTM

    # In[10]:

    input_dims = {"x_dim": x_dim, "y_dim": y_dim, "y_type": y_type, "max_cluster": K, "max_length": max_length}

    network_settings = {
        "h_dim_encoder": h_dim_RNN,
        "num_layers_encoder": num_layer_encoder,
        "rnn_type": rnn_type,
        "rnn_activate_fn": tf.nn.tanh,
        "h_dim_selector": h_dim_FC,
        "num_layers_selector": num_layer_selector,
        "h_dim_predictor": h_dim_FC,
        "num_layers_predictor": num_layer_predictor,
        "fc_activate_fn": tf.nn.relu,
    }

    # In[11]:

    RESULT_NMI = np.zeros([OUT_ITERATION, 1])
    RESULT_RI = np.zeros([OUT_ITERATION, 1])
    RESULT_PURITY = np.zeros([OUT_ITERATION, 1])

    # In[12]:
    tf.reset_default_graph()

    # Turn on xla optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    seed = args.seed
    set_random_seed(seed)
    sess = tf.Session(config=config)

    model = DeepTPC_ICLR(sess, "Deep_TPC", input_dims, network_settings)

    lr_rate1 = 0.001
    keep_prob = 1.0
    mb_size = 128

    ITERATION = epochs
    check_step = int(epochs / 5)

    for out_itr in range(OUT_ITERATION):
        print("========== OUT ITERATION: {} ========== ".format(out_itr))
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

        save_path = "./{}/proposed/init/itr{}/".format(data_mode, out_itr)

        if not os.path.exists(save_path + "/models/"):
            os.makedirs(save_path + "/models/")

        if not os.path.exists(save_path + "/results/"):
            os.makedirs(save_path + "/results/")

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer(), feed_dict={model.E: np.zeros([K, z_dim]).astype(float)})

        avg_loss2 = 0
        for itr in range(ITERATION):
            x_mb, y_mb = f_get_minibatch(mb_size, tr_data_x, tr_data_y)

            _, tmp_loss2 = model.train_mle(x_mb, y_mb, lr_rate1, keep_prob)
            avg_loss2 += tmp_loss2 / check_step

            if (itr + 1) % check_step == 0:
                tmp_y, tmp_m = model.predict_y_hats(va_data_x)

                y_pred = tmp_y.reshape([-1, y_dim])[tmp_m.reshape([-1]) == 1]
                y_true = va_data_y.reshape([-1, y_dim])[tmp_m.reshape([-1]) == 1]

                AUROC = np.zeros([y_dim])
                AUPRC = np.zeros([y_dim])
                for y_idx in range(y_dim):
                    auroc, auprc = f_get_prediction_scores(y_true[:, y_idx], y_pred[:, y_idx])
                    AUROC[y_idx] = auroc
                    AUPRC[y_idx] = auprc

                print(
                    "ITR {:05d}: loss_2={:.3f} | va_auroc:{:.3f}, va_auprc:{:.3f}".format(
                        itr + 1, avg_loss2, np.mean(AUROC), np.mean(AUPRC)
                    )
                )
                avg_loss2 = 0

        saver.save(sess, save_path + "models/model_v7_K{}".format(K))
        save_logging(network_settings, save_path + "models/network_settings_v7_K{}.txt".format(K))

        tmp_y, tmp_m = model.predict_y_hats(te_data_x)

        y_pred = tmp_y.reshape([-1, y_dim])[tmp_m.reshape([-1]) == 1]
        y_true = te_data_y.reshape([-1, y_dim])[tmp_m.reshape([-1]) == 1]

        AUROC = np.zeros([y_dim])
        AUPRC = np.zeros([y_dim])
        for y_idx in range(y_dim):
            auroc, auprc = f_get_prediction_scores(y_true[:, y_idx], y_pred[:, y_idx])
            AUROC[y_idx] = auroc
            AUPRC[y_idx] = auprc

        print(AUROC)
        print(AUPRC)

    print("Start trainig AC-TPC")

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

        load_path = "./{}/proposed/init/itr{}/".format(data_mode, out_itr)

        input_dims = {"x_dim": x_dim, "y_dim": y_dim, "y_type": y_type, "max_cluster": K, "max_length": max_length}

        tf.reset_default_graph()

        # Turn on xla optimization
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_random_seed(seed)
        sess = tf.Session(config=config)

        network_settings = load_logging(load_path + "models/network_settings_v7_K{}.txt".format(K))
        z_dim = network_settings["num_layers_encoder"] * network_settings["h_dim_encoder"]

        model = DeepTPC_ICLR(sess, "Deep_TPC", input_dims, network_settings)

        saver = tf.train.Saver()

        saver.restore(sess, load_path + "models/model_v7_K{}".format(K))

        # ### INITIALIZE EMBEDDING AND SELECTOR

        # In[21]:

        def initialize_embedding(model_, x_, K_):
            #         tmp_z, _, tmp_m = model_.predict_zs_and_pis_m2(x_)
            #         # tmp_z, _, tmp_m = model_.predict_zs_and_pis_m1(x_)

            #         tmp_z  = (tmp_z * np.tile(np.expand_dims(tmp_m, axis=2), [1,1,z_dim])).reshape([-1, z_dim])
            #         tmp_z  = tmp_z[np.sum(np.abs(tmp_z), axis=1) != 0]

            #         km     = KMeans(n_clusters = K_, init='k-means++')
            #         _      = km.fit(tmp_z)
            #         tmp_e      = km.cluster_centers_

            #         tmp_s  = km.predict(tmp_z)

            #         tmp_e  = np.zeros([K_, z_dim])
            #         for k in range(K_):
            #             tmp_e[k, :] = np.mean(tmp_z[tmp_s == k])

            tmp_z, _, _ = model_.predict_zs_and_pis_m2(x_)
            tmp_y, tmp_m = model.predict_y_hats(x_)

            tmp_z = (tmp_z * np.tile(np.expand_dims(tmp_m, axis=2), [1, 1, z_dim])).reshape([-1, z_dim])
            tmp_z = tmp_z[np.sum(np.abs(tmp_z), axis=1) != 0]

            tmp_y = (tmp_y * np.tile(np.expand_dims(tmp_m, axis=2), [1, 1, y_dim])).reshape([-1, y_dim])
            tmp_y = tmp_y[np.sum(np.abs(tmp_y), axis=1) != 0]

            km = KMeans(n_clusters=K_, init="k-means++", random_state=seed)
            _ = km.fit(tmp_y)
            tmp_ey = km.cluster_centers_
            tmp_s = km.predict(tmp_y)

            tmp_e = np.zeros([K_, z_dim])
            for k in range(K_):
                tmp_e[k, :] = tmp_z[np.argmin(np.sum(np.abs(tmp_y - tmp_ey[k, :]), axis=1)), :]

            return tmp_e, tmp_s, tmp_z

        # In[22]:

        mb_size = 128
        M = int(tr_data_x.shape[0] / mb_size)  # for main algorithm
        keep_prob = 1.0
        lr_rate1 = 1e-3
        lr_rate2 = 1e-3

        c_idx = 0  # selected index to keep track of embeddings...

        # In[23]:

        print("=============================================")
        print("===== INITIALIZING EMBEDDING & SELECTOR =====")
        # K-means over the latent encodings
        e, s_init, tmp_z = initialize_embedding(model, tr_data_x, K)
        sess.run(model.EE.initializer, feed_dict={model.E: e})

        # update selector wrt initial classes
        ITERATION = epochs
        check_step = int(epochs / 5)

        avg_loss_s = 0
        for itr in range(ITERATION):
            z_mb, s_mb = f_get_minibatch(mb_size, tmp_z, s_init)
            _, tmp_loss_s = model.train_selector(z_mb, s_mb, lr_rate1, k_prob=keep_prob)

            avg_loss_s += tmp_loss_s / check_step
            if (itr + 1) % check_step == 0:
                print("ITR:{:04d} | Loss_s:{:.4f}".format(itr + 1, avg_loss_s))
                avg_loss_s = 0

        tmp_ybars = model.predict_yy(e)
        # print( np.round(tmp_ybars[:,c_idx], 2) )
        print("=============================================")

        # In[100]:

        # # check selector outputs and intialized classes
        # pred_y, tmp_m = model.predict_s_sample(tr_data_x)

        # pred_y = pred_y.reshape([-1, 1])[tmp_m.reshape([-1]) == 1]
        # print(np.unique(pred_y), )

        # plt.hist(s_init, bins=15, color='C0', alpha=0.5)
        # plt.hist(pred_y[:, 0], bins=15, color='C1', alpha=0.5)
        # plt.show()
        # plt.close()

        # In[24]:
        """
            L1: predictive clustering loss
            L2: prediction loss
            L3: sample-wise entropy
            L4: average-wise entropy (not used in the paper)

            L_A = L1 + alpha * L3 + gamma * L4
            L_C = L1 + beta * L2

            ### to map with the paper (set gamma=0, beta=0)
            L_A = L1 + alpha * L3
            L_C = L1 + delta * L2
        """

        alpha = 3.0  # L3 (L3 is the sample-wise entropy loss)
        beta = 0.1  # L2 (L2 in the model is not used in the paper)
        gamma = 0.0  # L4
        delta = 1.0  # Ly is the embedding loss (this is beta in the paper)

        new_e = np.copy(e)

        print("=============================================")
        print("========== TRAINING MAIN ALGORITHM ==========")

        ITERATION = epochs
        check_step = int(epochs / 10)

        avg_loss_c_L1 = 0
        avg_loss_c_L2 = 0
        avg_loss_a_L1 = 0
        avg_loss_a_L3 = 0
        avg_loss_E = 0
        avg_loss_Ey = 0

        for itr in range(ITERATION):
            e = np.copy(new_e)

            for _ in range(M):
                x_mb, y_mb = f_get_minibatch(mb_size, tr_data_x, tr_data_y)

                _, tmp_loss_c_L1, tmp_loss_c_L2 = model.train_critic(x_mb, y_mb, beta, lr_rate1, keep_prob)
                avg_loss_c_L1 += tmp_loss_c_L1 / (M * check_step)
                avg_loss_c_L2 += tmp_loss_c_L2 / (M * check_step)

                x_mb, y_mb = f_get_minibatch(mb_size, tr_data_x, tr_data_y)

                _, tmp_loss_a_L1, tmp_loss_a_L3 = model.train_actor(x_mb, y_mb, alpha, gamma, lr_rate2, keep_prob)
                avg_loss_a_L1 += tmp_loss_a_L1 / (M * check_step)
                avg_loss_a_L3 += tmp_loss_a_L3 / (M * check_step)

            for _ in range(M):
                x_mb, y_mb = f_get_minibatch(mb_size, tr_data_x, tr_data_y)

                _, tmp_loss_E, tmp_loss_Ey = model.train_embedding(x_mb, y_mb, delta, lr_rate1, keep_prob)
                avg_loss_E += tmp_loss_E / (M * check_step)
                avg_loss_Ey += tmp_loss_Ey / (M * check_step)

            new_e = sess.run(model.EE)

            if (itr + 1) % check_step == 0:
                tmp_ybars = model.predict_yy(new_e)
                print(np.round(tmp_ybars[:, c_idx], 2))

                print(
                    "ITR {:04d}: loss_L2={:.3f}  loss_L1_c={:.3f}  loss_L1_a={:.3f}  loss_L1_E={:.3f}  loss_Ey={:.3f}  loss_L3_a={:.3f}".format(
                        itr + 1,
                        avg_loss_c_L2,
                        avg_loss_c_L1,
                        avg_loss_a_L1,
                        avg_loss_E,
                        avg_loss_Ey,
                        avg_loss_a_L3,
                    )
                )
                avg_loss_c_L1 = 0
                avg_loss_c_L2 = 0
                avg_loss_a_L1 = 0
                avg_loss_a_L3 = 0
                avg_loss_E = 0
                avg_loss_Ey = 0

        print("=============================================")

        save_path = "./{}/proposed/K{}/itr{}/".format(data_mode, K, out_itr)

        if not os.path.exists(save_path + "/models/"):
            os.makedirs(save_path + "/models/")

        if not os.path.exists(save_path + "/results/"):
            os.makedirs(save_path + "/results/")

        saver.save(sess, save_path + "models/model_v7_K{}_clustered".format(K))

        save_logging(network_settings, save_path + "models/network_settings_v7_K{}.txt".format(K))
        np.savez(save_path + "models/embeddings.npz", e=e)

        # In[29]:

        _, tmp_pi, tmp_m = model.predict_zbars_and_pis_m2(te_data_x)

        tmp_pi = tmp_pi.reshape([-1, K])[tmp_m.reshape([-1]) == 1]

        ncol = nrow = int(np.ceil(np.sqrt(K)))
        plt.figure(figsize=[4 * ncol, 2 * nrow])
        for k in range(K):
            plt.subplot(ncol, nrow, k + 1)
            plt.hist(tmp_pi[:, k])
        plt.suptitle("Clustering assignment probabilities")
        # plt.show()
        plt.savefig(save_path + "results/figure_clustering_assignments.png")
        plt.close()

        # In[147]:

        # check selector outputs and intialized classes
        # pred_y, tmp_m = model.predict_s_sample(tr_data_x)

        # pred_y = pred_y.reshape([-1, 1])[tmp_m.reshape([-1]) == 1]
        # print(np.unique(pred_y), )

        # plt.hist(pred_y[:, 0], bins=15, color='C1', alpha=1.0)
        # plt.show()
        # plt.savefig(save_path + 'results/figure_clustering_hist.png')
        # plt.close()

        # In[30]:

        tmp_y, tmp_m = model.predict_y_bars(te_data_x)

        y_pred = tmp_y.reshape([-1, y_dim])[tmp_m.reshape([-1]) == 1]
        y_true = te_data_y.reshape([-1, y_dim])[tmp_m.reshape([-1]) == 1]

        AUROC = np.zeros([y_dim])
        AUPRC = np.zeros([y_dim])
        for y_idx in range(y_dim):
            auroc, auprc = f_get_prediction_scores(y_true[:, y_idx], y_pred[:, y_idx])
            AUROC[y_idx] = auroc
            AUPRC[y_idx] = auprc

        print("ITR{} - K{} | Cluster - ROC:{:.4f}, PRC:{:.4f}".format(out_itr, K, np.mean(AUROC), np.mean(AUPRC)))

        # In[137]:

        tmp_y, tmp_m = model.predict_y_hats(te_data_x)

        y_pred = tmp_y.reshape([-1, y_dim])[tmp_m.reshape([-1]) == 1]
        y_true = te_data_y.reshape([-1, y_dim])[tmp_m.reshape([-1]) == 1]

        AUROC = np.zeros([y_dim])
        AUPRC = np.zeros([y_dim])
        for y_idx in range(y_dim):
            auroc, auprc = f_get_prediction_scores(y_true[:, y_idx], y_pred[:, y_idx])
            AUROC[y_idx] = auroc
            AUPRC[y_idx] = auprc

        print("ITR{} - K{} | Predictor - ROC:{:.4f}, PRC:{:.4f}".format(out_itr, K, np.mean(AUROC), np.mean(AUPRC)))

        # In[31]:

        pred_y, tmp_m = model.predict_s_sample(te_data_x)

        pred_y = (pred_y * tmp_m).reshape([-1, 1])
        pred_y = pred_y[(tmp_m.reshape([-1, 1]) == 1)[:, 0], 0]

        true_y = (te_data_y * np.tile(np.expand_dims(tmp_m, axis=2), [1, 1, y_dim])).reshape([-1, y_dim])
        true_y = true_y[(tmp_m.reshape([-1]) == 1)]
        true_y = np.argmax(true_y, axis=1)

        tmp_nmi = normalized_mutual_info_score(true_y, pred_y)
        tmp_ri = adjusted_rand_score(true_y, pred_y)
        tmp_purity = purity_score(true_y, pred_y)

        pd.DataFrame(
            [[tmp_nmi, tmp_ri, tmp_purity]], columns=["NMI", "RI", "PURITY"], index=["itr" + str(out_itr)]
        ).to_csv(save_path + "results/nmi_ir_purity.csv")

        print("ITR{} - K{} |  NMI:{:.4f}, RI:{:.4f}, PURITY:{:.4f}".format(out_itr, K, tmp_nmi, tmp_ri, tmp_purity))

        RESULT_NMI[out_itr, 0] = tmp_nmi
        RESULT_RI[out_itr, 0] = tmp_ri
        RESULT_PURITY[out_itr, 0] = tmp_purity
        # homogeneity_score(true_y, pred_y)

    # In[139]:

    pd.DataFrame(RESULT_NMI, columns=["NMI"], index=["itr" + str(out_itr) for out_itr in range(OUT_ITERATION)]).to_csv(
        "./{}/proposed/K{}/".format(data_mode, K) + "results_nmi.csv"
    )

    pd.DataFrame(RESULT_RI, columns=["RI"], index=["itr" + str(out_itr) for out_itr in range(OUT_ITERATION)]).to_csv(
        "./{}/proposed/K{}/".format(data_mode, K) + "results_ri.csv"
    )

    pd.DataFrame(
        RESULT_PURITY, columns=["PURITY"], index=["itr" + str(out_itr) for out_itr in range(OUT_ITERATION)]
    ).to_csv("./{}/proposed/K{}/".format(data_mode, K) + "results_purity.csv")
