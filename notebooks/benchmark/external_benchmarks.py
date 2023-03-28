#!/usr/bin/env python
# coding: utf-8

import pickle

import numpy as np
import pandas as pd
from benchmark import (
    data_interpolation,
    get_auc_scores,
    get_ci,
    get_cls_scores,
    select_by_steps,
)


def calculate_performance(preds, dataset, steps=(-1,)):
    x = dataset["x"]
    t = dataset["t"]
    c = dataset.get("c", None)
    y = dataset["y"]
    mask = dataset["mask"]

    method = preds["method"]
    c_pred = preds["c_pred"]
    y_pred = preds["y_pred"]
    c_pred = select_by_steps(c_pred, mask, steps)
    y_pred = select_by_steps(y_pred, mask, steps)

    if c is not None:
        c_true = select_by_steps(c, mask, steps)
    else:
        c_true = None

    y_true = select_by_steps(y, mask, steps)

    AUROC, AUPRC = get_auc_scores(y_true, y_pred)
    auc_scores = {"ROC": np.mean(AUROC), "PRC": np.mean(AUPRC)}

    x_sel = select_by_steps(x, mask, steps, sub_sequence=True, keepdims=True)
    t_sel = select_by_steps(t, mask, steps, sub_sequence=True, keepdims=False)
    x_interp = data_interpolation(t_sel, x_sel)
    cls_scores = get_cls_scores(c_true=c_true, c_pred=c_pred, x=x_interp, y_true=y_true)

    mixed1 = 2 / (1 / auc_scores.get("ROC", 1e-10) + 1 / cls_scores.get("Silhouette_auc", 1e-10))
    mixed2 = 2 / (1 / auc_scores.get("PRC", 1e-10) + 1 / cls_scores.get("Silhouette_auc", 1e-10))
    scores = {"method": method, **auc_scores, **cls_scores, "Hroc": mixed1, "Hprc": mixed2}
    return scores


def evaluation(preds, data, steps=(-1,)):
    with open(preds, "rb") as file:
        model_preds = pickle.load(file)

    with open(data, "rb") as file:
        splits = pickle.load(file)

    results = []
    for i, model_pred in enumerate(model_preds):
        dataset = splits[i]
        _, _, test_set = dataset
        scores = calculate_performance(model_pred, test_set, steps)
        results.append(scores)
    results = pd.DataFrame(results)
    summary = results.apply(get_ci)
    return summary


def update_results(results, preds, data):
    result = evaluation(preds, data)
    result["n"] = 5
    result["epochs"] = 200
    idx = results.index[results["method"] == result["method"]]
    if len(idx) == 0:
        new_row = results.index.max()
        results.loc[new_row + 1] = result
    elif len(idx) == 1:
        pass
    else:
        print("error: duplicate results")
    return results


outdir = "benchmark_results"

for dataname in ["Synth", "ICU", "ADNI"]:
    data_ = f"data/{dataname}_data.pkl"
    benchmark = f"{outdir}/{dataname}_benchmark.csv"
    results_ = pd.read_csv(benchmark, index_col=0)

    for model in ["ACTPC", "dcn_Seq2Seq"]:
        preds_ = f"external/{model}/output/{dataname}_preds.pkl"
        results_ = update_results(results_, preds_, data_)

    results_.to_csv(f"{outdir}/{dataname}_benchmark_complete.csv")
