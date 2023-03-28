#!/usr/bin/env python
# coding: utf-8

import itertools
import os
import pickle

import numpy as np
import pandas as pd
import torch
from benchmark import (
    Cls_config,
    Encoder_config,
    Predictor_config,
    evaluate,
    loss_weights,
)
from tqdm import auto

from tphenotype import Predictor


def evaluate_cluster(method, config, loss_weights_, splits, seed=0, epochs=50, steps=(-1,), metric="Hprc", **kwargs):
    cache_name = kwargs.get("cache_name", None)
    results = []
    for i, dataset in auto.tqdm(enumerate(splits), total=len(splits), desc=f"{method.__name__}"):
        train_set, valid_set, test_set = dataset

        torch.random.manual_seed(seed + i)
        model = method(**config)
        save_dir = "model_cache"
        model_name = f"{cache_name}-{i}.pt"
        model_path = f"{save_dir}/{model_name}"
        if cache_name is not None and os.path.exists(model_path):
            model = model.load(model_path)
            model.fit_clusters(train_set, verbose=False)
        else:
            os.makedirs(save_dir, exist_ok=True)
            model = model.fit(train_set, loss_weights_, valid_set=valid_set, epochs=epochs, verbose=False)
            model.save(save_dir, model_name)
        scores = evaluate(model, test_set, steps)
        results.append(scores[metric])
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


def hyperparam_selection_cluster(dataname, search_space, predictor_config, loss_weights_, seed=0, epochs=50):
    splits, feat_list, temporal_dims = load_data(dataname, verbose=True)  # pylint: disable=unused-variable
    tr_set, va_set, te_set = splits[0]  # pylint: disable=unused-variable
    _, T, x_dim = tr_set["x"].shape  # pylint: disable=unused-variable
    _, _, y_dim = tr_set["y"].shape

    # Configuration
    cls_config = Cls_config.copy()
    cls_config["K"] = 1
    cls_config["steps"] = [-1]
    predictor_config["x_dim"] = x_dim
    predictor_config["y_dim"] = y_dim
    predictor_config["time_series_dims"] = temporal_dims
    predictor_config["cls_config"] = cls_config

    print("T_phenotype config:")
    print(predictor_config)
    result_file = f"hyperparam_selection/{dataname}_K.csv"
    if os.path.exists(result_file):
        search_space = {}

    scores = pd.DataFrame(columns=["H_mean", "H_std", "config"])
    for i, comb in enumerate(itertools.product(*search_space.values())):
        if len(comb) == 0:
            continue

        test_config = cls_config.copy()
        msg = []
        for j, k_ in enumerate(search_space.keys()):
            if k_ in cls_config:
                test_config[k_] = comb[j]
            msg.append(f"{k_}:{comb[j]}")

        msg = ",".join(msg)
        print(f"test config {msg} ...")
        print("loss config:")
        print(loss_weights_)
        predictor_config["cls_config"] = test_config

        results, model = evaluate_cluster(  # pylint: disable=unused-variable
            Predictor, predictor_config, loss_weights_, splits, seed=seed, epochs=epochs, cache_name=f"hs-{dataname}"
        )
        scores.loc[i, "H_mean"] = np.mean(results)  # pyright: ignore
        scores.loc[i, "H_std"] = np.std(results)  # pyright: ignore
        scores.loc[i, "config"] = msg
        scores.to_csv(result_file)

    scores = pd.read_csv(result_file, index_col=0)
    scores = scores.astype({"H_mean": "float"})
    best = scores["H_mean"].idxmax()
    print("Optimal hyperparameters:")
    print(scores.loc[best, "config"])


def read_config(config_str):
    config = {}
    for item in config_str.split(","):
        key, val = item.split(":")
        config[key] = val
    return config


search_space_ = {
    "K": [2, 3, 4, 5],
}

for dataname_ in ["Synth", "ICU", "ADNI"]:
    result_file_ = f"hyperparam_selection/{dataname_}_encoder.csv"
    scores_ = pd.read_csv(result_file_, index_col=0)
    scores_ = scores_.astype({"mse_mean": "float"})
    best_ = scores_["mse_mean"].idxmin()

    test_loss_weights = loss_weights.copy()
    encoder_config = Encoder_config.copy()

    config_ = read_config(scores_.loc[best_, "config"])
    for k, v in config_.items():
        if k in test_loss_weights:
            default_v = test_loss_weights[k]
            test_loss_weights[k] = type(default_v)(v)
        if k in encoder_config:
            default_v = encoder_config[k]
            encoder_config[k] = type(default_v)(v)

    result_file_ = f"hyperparam_selection/{dataname_}_predictor.csv"
    scores_ = pd.read_csv(result_file_, index_col=0)
    scores_ = scores_.astype({"roc_mean": "float"})
    best_ = scores_["roc_mean"].idxmax()

    predictor_config_ = Predictor_config.copy()

    config_ = read_config(scores_.loc[best_, "config"])
    for k, v in config_.items():
        if k in predictor_config_:
            default_v = predictor_config_[k]
            predictor_config_[k] = type(default_v)(v)

    predictor_config_["encoder_config"] = encoder_config

    hyperparam_selection_cluster(dataname_, search_space_, predictor_config_, test_loss_weights)
