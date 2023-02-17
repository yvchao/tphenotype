import torch
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import type_metric, distance_metric
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import numpy as np

from .e2p import E2P
from ..utils.decorators import numpy_io


class KME2P(E2P):

    def __init__(self, K: int, **kwargs):
        
        if kwargs.get('latent_space','z')!='z':
            kwargs['latent_size'] = kwargs['hidden_size']
            
        super(KME2P, self).__init__(**kwargs)

        self.K = K
        self.name = f'KM-E2P({self.latent_space})'
        

    def fit(self, train_set, loss_weights, **kwargs):
        # train the neural network first
        super(KME2P, self).fit(train_set, loss_weights, **kwargs)

        # perform Kmeans on the learned representation space
        t, x, y, mask = train_set['t'], train_set['x'], train_set['y'], train_set['mask']
        embeds = self.encode(x, t)
        # remove sensored samples
        embeds = embeds[mask == True]

        initial_centers = kmeans_plusplus_initializer(embeds, self.K, random_state=0).initialize()
        initial_centers = np.array(initial_centers)
        metric = distance_metric(type_metric.EUCLIDEAN_SQUARE)

        # create K-Means algorithm with specific distance metric
        self.kmeans_instance = kmeans(embeds, initial_centers, metric=metric, ccore=False)

        # run cluster analysis and obtain results
        self.kmeans_instance.process()

        self.centers = np.array(self.kmeans_instance.get_centers()).astype('float32')
        return self


    def predict_proba_g(self, x, t):
        proba = super().predict_proba(x,t)
        return proba

    def predict_proba(self, x, t):
        sample_size, series_size, _ = x.shape
        cluster_idx = self.predict_cluster(x, t)

        if self.latent_space == 'y':
            probs = self.centers[cluster_idx.reshape((-1,))]
        elif self.latent_space == 'z':
            z = self.centers[cluster_idx.reshape((-1,))]
            with torch.no_grad():
                z = torch.from_numpy(z).to(self.device)
                logits = self.predictor(z)
                probs = torch.softmax(logits,dim=-1).cpu().numpy()
        elif self.latent_space == 'y-1':
            z = self.centers[cluster_idx.reshape((-1,))]
            with torch.no_grad():
                z = torch.from_numpy(z).to(self.device)
                probs = torch.softmax(z,dim=-1).cpu().numpy()
        else:
            raise NotImplementedError()

        probs = probs.reshape((sample_size, series_size, self.y_dim))
        return probs

    def predict_cluster(self, x, t):
        sample_size, series_size, _ = x.shape
        embeds = self.encode(x, t)

        embeds = embeds.reshape((-1, self.embed_size))
        cluster_idx = self.kmeans_instance.predict(embeds)
        cluster_idx = cluster_idx.reshape((sample_size, series_size))
        return cluster_idx
