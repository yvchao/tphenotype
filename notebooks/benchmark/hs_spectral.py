#!/usr/bin/env python
# coding: utf-8

import itertools
import os
import pickle

import numpy as np
import pandas as pd
import torch
from benchmark import evaluate, loss_weights
from tqdm import auto

from tphenotype.baselines import SpectralDTW


def evaluate_predictor(method, config, loss_weights_, splits, seed=0, epochs=50, steps=(-1,), metric="Hprc"):
    results = []
    for i, dataset in auto.tqdm(enumerate(splits), total=len(splits), desc=f"{method.__name__}"):
        train_set, valid_set, test_set = dataset

        torch.random.manual_seed(seed + i)
        torch.use_deterministic_algorithms(True)
        model = method(**config)
        model = model.fit(train_set, loss_weights_, valid_set=valid_set, epochs=epochs, verbose=False)
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


def hyperparam_selection_predictor(dataname, search_space, K, seed=0, epochs=50):
    splits, feat_list, temporal_dims = load_data(dataname, verbose=True)  # pylint: disable=unused-variable
    tr_set, va_set, te_set = splits[0]  # pylint: disable=unused-variable
    _, T, x_dim = tr_set["x"].shape  # pylint: disable=unused-variable
    _, _, y_dim = tr_set["y"].shape  # pylint: disable=unused-variable

    # Configuration
    spectral_config = {"K": K, "sigma": 1.0}

    result_file = f"hyperparam_selection/{dataname}_Spectral.csv"
    if os.path.exists(result_file):
        search_space = {}

    scores = pd.DataFrame(columns=["H_mean", "H_std", "config"])
    for i, comb in enumerate(itertools.product(*search_space.values())):
        if len(comb) == 0:
            continue

        test_config = spectral_config.copy()
        msg = []
        for j, k in enumerate(search_space.keys()):
            if k in spectral_config:
                test_config[k] = comb[j]
            msg.append(f"{k}:{comb[j]}")

        msg = ",".join(msg)
        print(f"test config {msg} ...")

        metric = "Hprc" if dataname != "Synth" else "PURITY"
        results, model = evaluate_predictor(  # pylint: disable=unused-variable
            SpectralDTW, test_config, loss_weights, splits, seed=seed, epochs=epochs, metric=metric
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
    "sigma": [0.1, 1.0, 5, 10],
}

for dataname_ in ["Synth"]:
    # calculation is only feasible on synthetic data
    result_file_ = f"hyperparam_selection/{dataname_}_K_orig.csv"
    scores_ = pd.read_csv(result_file_, index_col=0)
    scores_ = scores_.astype({"H_mean": "float"})
    best_ = scores_["H_mean"].idxmax()

    config_ = read_config(scores_.loc[best_, "config"])

    K_ = int(config_["K"])
    K_ = 3

    hyperparam_selection_predictor(dataname_, search_space_, K_)
