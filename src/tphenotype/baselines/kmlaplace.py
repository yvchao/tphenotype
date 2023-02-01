from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import type_metric, distance_metric
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import numpy as np
import torch
from ..model.predictor import Predictor


class KMLaplace(Predictor):

    def __init__(self, K, **kwargs):
        super().__init__(**kwargs)

        self.K = K
        self.name = 'KM-Laplace'
        self.embed_size = len(self.static_dims) + self.extra_dim

    def fit(self,
            train_set,
            loss_weights,
            valid_set=None,
            learning_rate=0.1,
            batch_size=50,
            epochs=100,
            max_grad_norm=1,
            tolerance=None,
            device=None,
            parameters=None,
            verbose=True,
            **kwargs):
        args = locals().copy()    # shallow copy
        # remove the self variable
        args.pop('self')

        # stage 1 - train the encoder
        self._fit_encoders(train_set, loss_weights, valid_set, args, learning_rate, verbose)

        # stage 2 - train the clustering scheme

        # perform Kmeans on the learned representation space
        t, x, y, mask = train_set['t'], train_set['x'], train_set['y'], train_set['mask']
        embeds = self.encode(x, t)

        if verbose:
            print('perform Kmeans on Laplace embedding')
        # remove sensored samples
        embeds = embeds[mask == True]
        y = y[mask == 1]

        initial_centers = kmeans_plusplus_initializer(embeds, self.K, random_state=0).initialize()
        initial_centers = np.array(initial_centers)
        metric = distance_metric(type_metric.EUCLIDEAN_SQUARE)

        # create K-Means algorithm with specific distance metric
        self.kmeans_instance = kmeans(embeds, initial_centers, metric=metric, ccore=True)

        # run cluster analysis and obtain results
        self.kmeans_instance.process()

        self.centers = np.array(self.kmeans_instance.get_centers()).astype('float32')
        _, y_dim = y.shape
        clusters = self.kmeans_instance.get_clusters()
        self.cluster_y = np.zeros((len(clusters), y_dim))
        for i, c in enumerate(clusters):
            self.cluster_y[i] = np.mean(y[c], axis=0)

        return self

    def predict_cluster(self, x, t):
        sample_size, series_size, _ = x.shape
        embeds = self.encode(x, t)

        embeds = embeds.reshape((-1, self.embed_size))
        cluster_idx = self.kmeans_instance.predict(embeds)
        cluster_idx = cluster_idx.reshape((sample_size, series_size))
        return cluster_idx

    def predict_proba(self, x, t):
        cluster = self.predict_cluster(x, t)
        labels = self.cluster_y[cluster]
        return labels
