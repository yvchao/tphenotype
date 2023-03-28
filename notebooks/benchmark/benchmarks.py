#!/usr/bin/env python
# coding: utf-8

import os
import pickle

import numpy as np
import pandas as pd
import torch
from benchmark import (
    Cls_config,
    Encoder_config,
    KME2P_config,
    Predictor_config,
    evaluate,
    loss_weights,
)
from tqdm import auto

from tphenotype import JointPredictor, Predictor
from tphenotype.baselines import KMDTW, KME2P, KMLaplace, SpectralDTW

output_dir = "benchmark_results"
os.makedirs(output_dir, exist_ok=True)


def benchmark(method, config, splits, loss_weights_, steps=(-1,), epochs=50, seed=0, **kwargs):
    dtype = "float32"  # noqa F841 # pylint: disable=unused-variable
    dataname = kwargs.get("dataname", None)

    results = []
    for i, dataset in auto.tqdm(enumerate(splits), total=len(splits), desc=f"{method.__name__}"):
        train_set, valid_set, test_set = dataset

        torch.random.manual_seed(seed + i)
        torch.use_deterministic_algorithms(True)

        model = method(**config)
        save_dir = "model_cache"
        model_name = f"bm-{dataname}-{model.name}-{i}.pt"
        model_path = f"{save_dir}/{model_name}"
        if dataname is not None and os.path.exists(model_path):
            model = model.load(model_path)
        else:
            os.makedirs(save_dir, exist_ok=True)
            model = model.fit(train_set, loss_weights_, valid_set=valid_set, epochs=epochs, verbose=False)
            model.save(save_dir, model_name)

        scores = evaluate(model, test_set, steps)
        results.append(scores)
    results = pd.DataFrame(results)
    summary = results.apply(get_ci)
    return summary


def get_ci(series, decimals=3):
    if series.dtype == "object":
        return series.iloc[0]

    stats = series.agg(["mean", "sem"])
    mean = np.format_float_positional(stats["mean"], decimals)
    ci = np.format_float_positional(1.96 * stats["sem"], decimals)
    out = f"{mean}+-{ci}"
    return out


def load_data(dataname, verbose=False):
    with open(f"data/{dataname}_data.pkl", "rb") as file:
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


def run_benchmark(dataname, splits, setup_list, seed=0, epochs=50):
    result_file = f"{output_dir}/{dataname}_benchmark.csv"
    if os.path.exists(result_file):
        existing_df = pd.read_csv(result_file, index_col=0)
    else:
        existing_df = pd.DataFrame(columns=["method"])
    results = []
    epochs = 50
    for model, config, loss_weights_ in auto.tqdm(setup_list, desc="setups"):
        model_name = model(**config).name
        if (existing_df["method"] == model_name).any():
            print(f"benchmark is already done for {model_name}, skipping")
            continue
        result = benchmark(model, config, splits, loss_weights_, seed=seed, epochs=epochs, dataname=dataname)
        results.append(result)
        results_df = pd.DataFrame(results)
        results_df = pd.concat([existing_df, results_df], axis=0)
        results_df["n"] = len(splits)
        results_df["epochs"] = epochs

        results_df.to_csv(result_file)

    return results


def read_config(config_str):
    config = {}
    for item in config_str.split(","):
        key, val = item.split(":")
        config[key] = val
    return config


def prepare_benchmark(dataname):
    splits, feat_list, temporal_dims = load_data(dataname, verbose=True)  # pylint: disable=unused-variable
    tr_set, va_set, te_set = splits[0]  # pylint: disable=unused-variable
    _, T, x_dim = tr_set["x"].shape  # pylint: disable=unused-variable
    _, _, y_dim = tr_set["y"].shape

    result_file = f"hyperparam_selection/{dataname}_encoder.csv"
    scores = pd.read_csv(result_file, index_col=0)
    scores = scores.astype({"mse_mean": "float"})
    best = scores["mse_mean"].idxmin()
    config_encoders = read_config(scores.loc[best, "config"])

    # result_file = f'hyperparam_selection/{dataname}_predictor.csv'
    # scores = pd.read_csv(result_file, index_col=0)
    # scores = scores.astype({'roc_mean': 'float'})
    # best = scores['roc_mean'].idxmax()
    # config_predictor = read_config(scores.loc[best, 'config'])

    result_file = f"hyperparam_selection/{dataname}_K_orig.csv"
    scores = pd.read_csv(result_file, index_col=0)
    scores = scores.astype({"H_mean": "float"})
    best = scores["H_mean"].idxmax()
    config_K = read_config(scores.loc[best, "config"])

    result_file = f"hyperparam_selection/{dataname}_E2Py.csv"
    scores = pd.read_csv(result_file, index_col=0)
    scores = scores.astype({"H_mean": "float"})
    best = scores["H_mean"].idxmax()
    config_e2py = read_config(scores.loc[best, "config"])

    result_file = f"hyperparam_selection/{dataname}_E2Pz.csv"
    scores = pd.read_csv(result_file, index_col=0)
    scores = scores.astype({"H_mean": "float"})
    best = scores["H_mean"].idxmax()
    config_e2pz = read_config(scores.loc[best, "config"])

    K = int(config_K["K"])
    if dataname == "Synth":
        K = 3
    print(f"dataset {dataname}, cluster num K= {K}")

    setup_list = []

    e2py_config = KME2P_config.copy()
    e2py_config["K"] = K
    e2py_config["x_dim"] = x_dim
    e2py_config["y_dim"] = y_dim
    e2py_config["latent_space"] = "y"
    e2py_config["hidden_size"] = int(config_e2py["hidden_size"])
    e2py_config["num_layers"] = int(config_e2py["num_layers"])

    e2py_loss_weights = loss_weights.copy()
    setup_list.append((KME2P, e2py_config, e2py_loss_weights))

    e2pz_config = e2py_config.copy()
    e2pz_config["latent_space"] = "z"
    e2pz_config["hidden_size"] = int(config_e2pz["hidden_size"])
    e2pz_config["num_layers"] = int(config_e2pz["num_layers"])
    e2pz_config["latent_size"] = int(config_e2pz["latent_size"])
    e2pz_loss_weights = loss_weights.copy()
    setup_list.append((KME2P, e2pz_config, e2pz_loss_weights))

    kmdtw_config = {"K": K}
    kmdtw_loss_weights = loss_weights.copy()
    setup_list.append((KMDTW, kmdtw_config, kmdtw_loss_weights))

    encoder_config = Encoder_config.copy()
    encoder_config["pole_separation"] = float(config_encoders["pole_separation"])
    encoder_config["max_degree"] = int(config_encoders["max_degree"])

    predictor_loss_weights = loss_weights.copy()
    predictor_loss_weights["pole"] = float(config_encoders["pole"])
    predictor_loss_weights["real"] = float(config_encoders["real"])

    cls_config = Cls_config.copy()
    cls_config["K"] = K
    cls_config["steps"] = [-1]
    predictor_config = Predictor_config.copy()
    predictor_config["x_dim"] = x_dim
    predictor_config["y_dim"] = y_dim
    predictor_config["time_series_dims"] = temporal_dims
    predictor_config["cls_config"] = cls_config
    predictor_config["encoder_config"] = encoder_config
    # predictor_config['hidden_size'] = int(config_predictor['hidden_size'])
    # predictor_config['num_layer'] = int(config_predictor['num_layer'])
    setup_list.append((Predictor, predictor_config, predictor_loss_weights))

    KMLaplace_config = predictor_config.copy()
    KMLaplace_config["K"] = K
    setup_list.append((KMLaplace, KMLaplace_config, predictor_loss_weights))
    setup_list.append((JointPredictor, predictor_config, predictor_loss_weights))

    result_file = f"hyperparam_selection/{dataname}_Spectral.csv"
    if os.path.exists(result_file):
        scores = pd.read_csv(result_file, index_col=0)
        scores = scores.astype({"H_mean": "float"})
        best = scores["H_mean"].idxmax()
        config_spectral = read_config(scores.loc[best, "config"])

        spectral_config = {"K": K, "sigma": float(config_spectral["sigma"])}
        setup_list.append((SpectralDTW, spectral_config, loss_weights))
    return splits, setup_list


if __name__ == "__main__":
    for dataname_ in ["Synth", "ICU", "ADNI"]:
        splits_, setup_list_ = prepare_benchmark(dataname_)
        run_benchmark(dataname_, splits_, setup_list_)
