# pylint: disable=attribute-defined-outside-init,unused-argument

import numpy as np
from dtaidistance import dtw_ndim
from sklearn.cluster import SpectralClustering

from ..base_model import BaseModel


def slice_sub_sequences(x, mask=None):
    sample_size, series_size, x_dim = x.shape
    x = x.reshape((sample_size, 1, series_size, x_dim))
    x = np.repeat(x, series_size, axis=1)
    for t in range(series_size):
        x[:, t, -(t + 1) :, :] = x[:, t, : t + 1, :]
        x[:, t, :t, :] = 0
    if mask is not None:
        x = x[mask[:, :] == 1.0]
    else:
        x = x.reshape(-1, series_size, x_dim)
    return x


class SpectralDTW(BaseModel):
    def __init__(self, K, sigma, **kwargs):
        super().__init__()
        self.name = "Spectral-DTW-D"
        self.K = K
        self.sigma = sigma

    def fit(self, train_set, *args, **kwargs):
        x = train_set["x"]
        y = train_set["y"]
        mask = train_set["mask"]
        self.corpus_x = slice_sub_sequences(x, mask)
        self.corpus_y = y[mask[:, :] == 1.0]
        self.corpus_size = len(self.corpus_x)
        return self

    def predict_cluster(self, x, t, mask, *args):
        _, _, x_dim = x.shape
        x = slice_sub_sequences(x, mask)
        x_concat = np.concatenate([self.corpus_x, x], axis=0)
        distance = dtw_ndim.distance_matrix_fast(x_concat.astype(np.double), ndim=x_dim)
        W = np.exp(-(distance**2) / self.sigma)

        self.cls = SpectralClustering(n_clusters=self.K, random_state=0, affinity="precomputed")
        c_pred = self.cls.fit_predict(W)
        cluster_labels = c_pred[: self.corpus_size]
        cluster_pred = c_pred[self.corpus_size :]
        clusters = np.unique(cluster_labels)

        _, y_dim = self.corpus_y.shape
        self.cluster_y = np.zeros((len(clusters), y_dim))
        cluster_idx = np.zeros_like(cluster_pred)
        for i, c in enumerate(clusters):
            self.cluster_y[i] = np.mean(self.corpus_y[cluster_labels == c], axis=0)
            cluster_idx[cluster_pred == c] = i
        return cluster_idx

    def predict_proba(self, x, t, mask, *args):
        cluster = self.predict_cluster(x, t, mask)
        labels = self.cluster_y[cluster]
        return labels

    def save(self, path=".", name=None):
        pass

    def load(self, filename):
        pass
