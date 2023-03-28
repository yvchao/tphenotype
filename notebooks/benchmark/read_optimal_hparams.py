#!/usr/bin/env python
# coding: utf-8

import pandas as pd


def read_config(config_str):
    config = {}
    for item in config_str.split(","):
        key, val = item.split(":")
        config[key] = val
    return config


for dataname in ["Synth", "ICU", "ADNI"]:
    result_file = f"hyperparam_selection/{dataname}_encoder.csv"
    scores = pd.read_csv(result_file, index_col=0)
    scores = scores.astype({"mse_mean": "float"})
    best = scores["mse_mean"].idxmin()
    config_encoders = read_config(scores.loc[best, "config"])

    result_file = f"hyperparam_selection/{dataname}_predictor.csv"
    scores = pd.read_csv(result_file, index_col=0)
    scores = scores.astype({"roc_mean": "float"})
    best = scores["roc_mean"].idxmax()

    config_predictor = read_config(scores.loc[best, "config"])

    result_file = f"hyperparam_selection/{dataname}_K.csv"
    scores = pd.read_csv(result_file, index_col=0)
    scores = scores.astype({"H_mean": "float"})
    best = scores["H_mean"].idxmax()

    config_K = read_config(scores.loc[best, "config"])

    config_ = {**config_encoders, **config_predictor, **config_K}
    print(f"Dataset {dataname}")
    print("optimal hyperparameters:")
    print(config_)
    print("\n")
