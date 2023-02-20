import numpy as np
import pickle
from dtaidistance.dtw import dtw_cc
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import type_metric, distance_metric
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

from ..base_model import BaseModel


def dtw_distance(s1, s2, shape=None):
    s1 = s1.astype(np.double).reshape(shape)
    s2 = s2.astype(np.double).reshape(shape)
    return dtw_cc.distance_ndim(s1, s2)


def slice_sub_sequences(x, mask=None):
    sample_size, series_size, x_dim = x.shape
    x = x.reshape((sample_size, 1, series_size, x_dim))
    x = np.repeat(x, series_size, axis=1)
    for t in range(series_size):
        x[:, t, -(t + 1):, :] = x[:, t, :t + 1, :]
        x[:, t, :t, :] = 0
    if mask is not None:
        x = x[mask[:, :] == 1.0]
    else:
        x = x.reshape(-1, series_size, x_dim)
    return x


class KMDTW(BaseModel):

    def __init__(self, K, **kwargs):
        super().__init__()
        self.name = 'KM-DTW-D'
        self.K = K

    def fit(self, train_set, *args, **kwargs):
        x = train_set['x']
        y = train_set['y']
        mask = train_set['mask']
        x = slice_sub_sequences(x, mask)
        sample_size, series_size, x_dim = x.shape
        x = x.reshape((sample_size, -1))
        y = y[mask[:, :] == 1.0]

        initial_centers = kmeans_plusplus_initializer(x, self.K, random_state=0).initialize()
        initial_centers = np.array(initial_centers)

        def distance_func(s1, s2, shape=(series_size, x_dim)):
            return dtw_distance(s1, s2, shape)

        metric = distance_metric(type_metric.USER_DEFINED, func=distance_func)

        # create K-Means algorithm with specific distance metric
        self.cls = kmeans(x, initial_centers, metric=metric, ccore=False)

        # run cluster analysis and obtain results
        self.cls.process()

        self.centers = np.array(self.cls.get_centers()).reshape((self.K, series_size, x_dim))
        _, y_dim = y.shape
        clusters = self.cls.get_clusters()
        self.cluster_y = np.zeros((len(clusters), y_dim))
        for i, c in enumerate(clusters):
            self.cluster_y[i] = np.mean(y[c], axis=0)
        return self

    def save(self, path='.', name=None):
        pass

    def load(self, filename):
        pass

    def predict_cluster(self, x, t):
        sample_size, time_steps, feature_size = x.shape
        x = slice_sub_sequences(x)
        x = x.reshape((-1, time_steps * feature_size))
        cluster_idx = self.cls.predict(x)
        cluster_idx = cluster_idx.reshape((sample_size, time_steps))
        return cluster_idx

    def predict_proba(self, x, t):
        cluster = self.predict_cluster(x, t)
        labels = self.cluster_y[cluster]
        return labels
