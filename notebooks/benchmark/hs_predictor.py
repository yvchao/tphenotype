#!/usr/bin/env python
# coding: utf-8

import itertools
import os
import pickle

import numpy as np
import pandas as pd
import torch
from benchmark import Cls_config, Encoder_config, Predictor_config, loss_weights
from tqdm import auto

from tphenotype import Predictor


def evaluate_predictor(method, config, loss_weights_, splits, seed=0, epochs=50):
    results = []
    for i, dataset in auto.tqdm(enumerate(splits), total=len(splits), desc=f"{method.__name__}"):
        train_set, valid_set, test_set = dataset

        torch.random.manual_seed(seed + i)
        torch.use_deterministic_algorithms(True)
        model = method(**config)
        roc = model.evaluate_predictor_params(
            train_set, test_set, loss_weights_, valid_set=valid_set, epochs=epochs, verbose=False
        )
        results.append(roc)
    results = np.array(results)
    return results, model  # pyright: ignore


os.makedirs("hyperparam_selection", exist_ok=True)


def load_data(dataname, verbose=False):
    with open(f"data/{dataname}_data_hs.pkl", "rb") as file:
        splits = pickle.load(file)

    if dataname == "Synth":
        feat_list = ["x1", "x2"]
        temporal_dims = [0, 1]
    elif dataname == "ADNI":
        feat_list = ["APOE4", "CDRSB", "Hippocampus"]
        temporal_dims = [1, 2]
    elif dataname == "ICU":
        feat_list = ["Age", "Gender", "GCS", "PaCO2"]
        temporal_dims = [2, 3]
    else:
        raise ValueError(f"unknown dataset {dataname}")

    if verbose:
        tr_set, va_set, te_set = splits[0]
        _, T, x_dim = tr_set["x"].shape
        _, _, y_dim = tr_set["y"].shape
        print(dataname)
        print(f"total samples: {len(tr_set['x'])+len(va_set['x'])+len(te_set['x'])}")

        print(f"max length: {T}")
        print(f"x_dim: {x_dim}")
        print(f"y_dim: {y_dim}")

        print(f"features: {feat_list}")
        print(f"temporal dims: {temporal_dims}")
    return splits, feat_list, temporal_dims


def hyperparam_selection_predictor(dataname, search_space, encoder_config, loss_weights_, seed=0, epochs=50):
    splits, feat_list, temporal_dims = load_data(dataname, verbose=True)  # pylint: disable=unused-variable
    tr_set, va_set, te_set = splits[0]  # pylint: disable=unused-variable
    _, T, x_dim = tr_set["x"].shape  # pylint: disable=unused-variable
    _, _, y_dim = tr_set["y"].shape

    # Configuration
    K = 1

    cls_config = Cls_config.copy()
    cls_config["K"] = K
    cls_config["steps"] = [-1]
    predictor_config = Predictor_config.copy()
    predictor_config["x_dim"] = x_dim
    predictor_config["y_dim"] = y_dim
    predictor_config["time_series_dims"] = temporal_dims
    predictor_config["cls_config"] = cls_config
    predictor_config["encoder_config"] = encoder_config

    print("T_phenotype config:")
    print(predictor_config)
    result_file = f"hyperparam_selection/{dataname}_predictor.csv"
    if os.path.exists(result_file):
        search_space = {}

    scores = pd.DataFrame(columns=["roc_mean", "roc_std", "config"])
    for i, comb in enumerate(itertools.product(*search_space.values())):
        if len(comb) == 0:
            continue

        test_config = predictor_config.copy()
        test_loss_weights = loss_weights_.copy()
        msg = []
        for j, k_ in enumerate(search_space.keys()):
            if k_ in predictor_config:
                test_config[k_] = comb[j]
            elif k_ in loss_weights_:
                test_loss_weights[k_] = comb[j]
            msg.append(f"{k_}:{comb[j]}")

        msg = ",".join(msg)
        print(f"test config {msg} ...")
        print("loss config:")
        print(test_loss_weights)
        results, model = evaluate_predictor(  # pylint: disable=unused-variable
            Predictor,
            test_config,
            test_loss_weights,
            splits,
            seed=seed,
            epochs=epochs,
        )
        scores.loc[i, "roc_mean"] = np.mean(results)  # pyright: ignore
        scores.loc[i, "roc_std"] = np.std(results)  # pyright: ignore
        scores.loc[i, "config"] = msg
        scores.to_csv(result_file)

    scores = pd.read_csv(result_file, index_col=0)
    scores = scores.astype({"roc_mean": "float"})
    best = scores["roc_mean"].idxmax()
    print("Optimal hyperparameters:")
    print(scores.loc[best, "config"])


def read_config(config_str):
    config = {}
    for item in config_str.split(","):
        key, val = item.split(":")
        config[key] = val
    return config


search_space_ = {
    "hidden_size": [5, 10],
    "num_layer": [2, 3, 4],
}

for dataname_ in ["Synth", "ICU", "ADNI"]:
    result_file_ = f"hyperparam_selection/{dataname_}_encoder.csv"
    scores_ = pd.read_csv(result_file_, index_col=0)
    scores_ = scores_.astype({"mse_mean": "float"})
    best_ = scores_["mse_mean"].idxmin()

    test_loss_weights_ = loss_weights.copy()
    encoder_config_ = Encoder_config.copy()

    config_ = read_config(scores_.loc[best_, "config"])
    for k, v in config_.items():
        if k in test_loss_weights_:
            default_v = test_loss_weights_[k]
            test_loss_weights_[k] = type(default_v)(v)
        if k in encoder_config_:
            default_v = encoder_config_[k]
            encoder_config_[k] = type(default_v)(v)

    hyperparam_selection_predictor(dataname_, search_space_, encoder_config_, test_loss_weights_)
