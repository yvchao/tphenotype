#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from tphenotype.utils import data_split, get_one_hot, select_by_steps
from tphenotype.utils.dataset import cut_windowed_data

# In[2]:

source_data = Path("../../data")

# In[3]:

os.makedirs("data", exist_ok=True)


def split_datasets(dataset, dataname, n_trial=5, test_size=0.2, seed=0, dtype="float32"):
    splits = []
    for trail in range(n_trial):
        train_set, test_set = data_split(dataset, test_size=test_size, random_state=seed + trail, dtype=dtype)
        train_set, valid_set = data_split(train_set, test_size=0.2, random_state=seed + trail, dtype=dtype)
        splits.append([train_set, valid_set, test_set])

    filename = f"{dataname}_data.pkl"

    with open(f"data/{filename}", "wb") as out:
        pickle.dump(splits, out, pickle.HIGHEST_PROTOCOL)


def split_datasets_hs(dataset, dataname, k_fold=5, test_size=0.2, seed=0, dtype="float32"):
    splits = []

    kf = KFold(n_splits=k_fold, shuffle=True, random_state=seed)
    for train_index, test_index in kf.split(dataset["x"]):
        train_set = {k: v[train_index].astype(dtype) for k, v in dataset.items()}
        test_set = {k: v[test_index].astype(dtype) for k, v in dataset.items()}
        train_set, valid_set = data_split(train_set, test_size=0.2, random_state=seed, dtype=dtype)
        splits.append([train_set, valid_set, test_set])

    filename = f"{dataname}_data_hs.pkl"

    with open(f"data/{filename}", "wb") as out:
        pickle.dump(splits, out, pickle.HIGHEST_PROTOCOL)


# In[4]:

# ADNI -- 3 labels: NL (normal) -> mild cognitive impairment (MCI) -> Dementia
# X features
# Delta = 0.5 (6 months)
# AGE seems to be useless (always constant)
# Y features
# DX_Dementia, DX_MCI, DX_NL
# Right sensoring is indicated with zeros

npz = np.load(source_data / "real-world/ADNI/data_with_orig.npz")

data_x = npz["data_x"]
data_x_orig = npz["data_x_orig"]
data_y = npz["data_y"]
feat_list = npz["feat_list"]
label_list = npz["label_list"]

data_delta = data_x[:, :, 0]
data_t = np.cumsum(data_delta, axis=1)
feat_list = feat_list[1:]
data_x = data_x[:, :, 1:]
data_x_orig = data_x_orig[:, :, 1:]
data_mask = data_x.any(axis=2)

considered = [
    "APOE4",
    "Hippocampus",
    "CDRSB",
]
(considered_dims,) = np.where(np.isin(feat_list, considered))
considered = feat_list[considered_dims]
data_x = data_x[:, :, considered_dims]
data_x_orig = data_x_orig[:, :, considered_dims]

window_size = 6
t, x, y, m = cut_windowed_data(data_t, data_x, data_y, data_mask, window_size=window_size)
_, x_orig, _, range_m = cut_windowed_data(
    data_t, data_x_orig, data_y, data_mask, window_size=window_size, range_mask=True
)

t_orig = t.copy()
t_orig[range_m == 0] = 1e10
t = t_orig - t_orig.min(axis=-1, keepdims=True)
t[range_m == 0] = 0
t_orig[range_m == 0] = 0
t_scale = t.max() - t.min()
t = (t - t.min()) / t_scale

N, T, x_dim = x.shape
_, _, y_dim = y.shape
temporal_feats = ["Hippocampus", "CDRSB"]
(temporal_dims,) = np.where(np.isin(considered, temporal_feats))

dataset_ADNI = {
    "t": t,
    "t_orig": t_orig,
    "x": x,
    "x_orig": x_orig,
    "y": y,
    "mask": m,
    "range_mask": range_m,
}

split_datasets(dataset_ADNI, "ADNI")
split_datasets_hs(dataset_ADNI, "ADNI", k_fold=3)
print(considered)
print(temporal_dims)

# In[5]:

# ICU

npz = np.load(source_data / "real-world/physionet/selected_data.npz")

data_x = npz["data_x"]
data_x_orig = npz["data_x_orig"]
data_y = npz["data_y"]
data_t = npz["data_t"]
data_mask = npz["data_mask"]
feat_list = npz["feat_list"]
data_y = get_one_hot(data_y, 2)

([idx],) = np.where(feat_list == "ICUType")
ICUType = data_x_orig[:, 0, idx]

sel = (ICUType == 4) | (ICUType == 3)

data_x = data_x[sel]
data_x_orig = data_x_orig[sel]
data_y = data_y[sel]
data_t = data_t[sel]
data_mask = data_mask[sel]

(idx,) = np.where(np.isin(feat_list, ["Age", "Gender", "GCS", "PaCO2"]))
data_x = data_x[:, :, idx]
feat_list = feat_list[idx]

window_size = 24
t, x, y, m = cut_windowed_data(data_t, data_x, data_y, data_mask, window_size=window_size)
_, x_orig, _, range_m = cut_windowed_data(
    data_t, data_x_orig, data_y, data_mask, window_size=window_size, range_mask=True
)

t_orig = t.copy()
t_orig[range_m == 0] = 1e10
t = t_orig - t_orig.min(axis=-1, keepdims=True)
t[range_m == 0] = 0
t_orig[range_m == 0] = 0
t_scale = t.max() - t.min()
t = (t - t.min()) / t_scale

N, T, x_dim = x.shape
_, _, y_dim = y.shape
temporal_feats = ["GCS", "PaCO2"]
(temporal_dims,) = np.where(np.isin(feat_list, temporal_feats))

dataset_ICU = {
    "t": t,
    "t_orig": t_orig,
    "x": x,
    "x_orig": x_orig,
    "y": y,
    "mask": m,
    "range_mask": range_m,
}

split_datasets(dataset_ICU, "ICU", test_size=0.4)
split_datasets_hs(dataset_ICU, "ICU", test_size=0.4, k_fold=3)

print(feat_list)
print(temporal_dims)

# In[6]:

# Synthetic
data = np.load(source_data / "synthetic/data-mixed.npz")
data_t, data_x, data_y, data_c, data_mask = data["t"], data["x"], data["y"], data["c"], data["mask"]

N, T = data_y.shape
y_onehot = get_one_hot(data_y.reshape((N * T,)) - 1, 2).reshape((N, T, 2))

window_size = 2
t, x, y, m = cut_windowed_data(data_t, data_x, y_onehot, data_mask, window_size=window_size)
_, _, c, range_m = cut_windowed_data(data_t, data_x, data_c, data_mask, window_size=window_size, range_mask=True)
c = c[:, :, 0]

t_orig = t.copy()
t_orig[range_m == 0] = 1e10
t = t_orig - t_orig.min(axis=-1, keepdims=True)
t[range_m == 0] = 0
t_orig[range_m == 0] = 0
t_scale = t.max() - t.min()
t = (t - t.min()) / t_scale

feat_list = ["x1", "x2"]
temporal_dims = [0, 1]

dataset_Synth = {
    "t": t,
    "t_orig": t_orig,
    "x": x,
    "y": y,
    "mask": m,
    "range_mask": range_m,
    "c": c,
}

split_datasets(dataset_Synth, "Synth")
split_datasets_hs(dataset_Synth, "Synth", k_fold=3)

print(feat_list)
print(temporal_dims)

# In[7]:

with open("data/ADNI_data.pkl", "rb") as file:
    datasets = pickle.load(file)
    train_set, valid_set, test_set = datasets[1]

# In[8]:

train_set["x"].shape, valid_set["x"].shape, test_set["x"].shape

# In[9]:

with open("data/ICU_data.pkl", "rb") as file:
    datasets = pickle.load(file)
    train_set, valid_set, test_set = datasets[1]

# In[10]:

train_set["x"].shape, valid_set["x"].shape, test_set["x"].shape

# In[11]:

with open("data/Synth_data.pkl", "rb") as file:
    datasets = pickle.load(file)
    train_set, valid_set, test_set = datasets[1]

# In[12]:

train_set["x"].shape, valid_set["x"].shape, test_set["x"].shape

# In[ ]:
