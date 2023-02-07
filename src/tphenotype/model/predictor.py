import numpy as np
import torch
from ..base_model import NNBaseModel
from ..nn import MLP
from ..utils.utils import check_shape, select_by_steps, EPS
from ..utils.decorators import device_init, numpy_io, run_in_batch
from ..utils.metrics import get_auc_scores
from ..utils.losses import cross_entropy

from .encoder import LaplaceEncoder
from .sync_clustering import SyncClustering
from .cluster_explainer import fit_cluster, explain

# default encoder configuration
default_encoder_config = {
    'num_poles': 4,    # number of poles
    'max_degree': 1,    # maximum degree of poles
    'hidden_size': 20,    # number of hidden units in neural networks
    'num_layers': 2,    # number of layers in MLP of the encoder (1 layer RNN + n layer MLP)
    'pole_separation': 0.6,    # minimum distance between distinct poles 
    'freq_scaler': 30,    # scale up the imaginary part to help learning
    'window_size': None,    # whether or not to include time delay terms
    'equivariant_embed': True,    # whether or not to sort the poles (useless during training)
}

default_encoder_loss_weights = {
    'rmse': 1.0,    # regression loss
    'cont': 0.1,    # minimize used components
    'pole': 1.0,    # seperation of poles
    'real': 0.01,    # only generate real function (penalty for imaginary part)
}

default_cls_config = {
    'K': 4,
    'steps': [-1],
    'tol': 1e-6,
    'test_num': 50,
}


class Predictor(NNBaseModel):

    @device_init
    def __init__(self,
                 x_dim,
                 y_dim,
                 time_series_dims=[],
                 feature_names=None,
                 hidden_size=20,
                 num_layer=3,
                 global_bias=False,
                 random_state=0,
                 categorical=True,
                 **kwargs):

        super().__init__()

        self.name = 'T-Phenotype'
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.global_bias = global_bias
        self.random_state = random_state
        self.categorical = categorical

        if self.categorical:
            self.output_fn = lambda z: torch.softmax(z, dim=-1)
        else:
            self.output_fn = torch.sigmoid

        self.rng = np.random.default_rng(self.random_state)

        self.time_series_dims = time_series_dims    # dimensions of time-series features
        self.static_dims = np.delete(np.arange(x_dim), self.time_series_dims)
        if feature_names is not None:
            assert len(feature_names) == self.x_dim
            self.feature_names = feature_names
        else:
            self.feature_names = [f'x_{i+1}' for i in range(self.x_dim)]

        encoder_config = kwargs.get('encoder_config', None)
        if encoder_config is not None:
            encoders = [LaplaceEncoder(**encoder_config) for _ in self.time_series_dims]
            self.encoders = torch.nn.ModuleList(encoders)
        else:
            encoders = [LaplaceEncoder(**default_encoder_config) for _ in self.time_series_dims]
            self.encoders = torch.nn.ModuleList(encoders)

        self.extra_dim = None
        if len(self.time_series_dims) > 0:
            self.extra_dim = len(self.time_series_dims) * self.encoders[0].get_embed_size()
        else:
            self.extra_dim = 0

        if self.global_bias:
            self.bias = torch.nn.Parameter(torch.zeros((y_dim,)))
        else:
            self.bias = 0.0
        self.g = MLP(
            len(self.static_dims) + self.extra_dim,
            self.y_dim,
            self.hidden_size,
            self.num_layer,
            bias=not self.global_bias)

        cls_config = kwargs.get('cls_config', None)
        if cls_config is not None:
            self.cls = SyncClustering(predictor=self, **cls_config)
        else:
            self.cls = SyncClustering(predictor=self, **default_cls_config)

    def _encode(self, x, t):
        x_rep = []
        if len(self.static_dims) > 0:
            x_rep.append(x[:, :, self.static_dims])

        for i, d in enumerate(self.time_series_dims):
            x_d = x[:, :, d]
            # normalize the embedding to aovid saturation
            x_d_embed = self.encoders[i]._get_embed(x_d, t, normalize=True)
            x_rep.append(x_d_embed)
        x_rep = torch.cat(x_rep, dim=-1)

        return x_rep

    @numpy_io
    @run_in_batch
    def encode(self, x, t):
        t = check_shape(t).to(self.device)
        x = x.to(self.device)
        with torch.no_grad():
            x_rep = self._encode(x, t)
        return x_rep

    @numpy_io
    @run_in_batch
    def rep_to_curve(self, x_rep):
        # embed: ...  x embed_size
        shape = x_rep.shape
        if len(shape) == 2:
            x_rep = x_rep[:, np.newaxis, :]

        with torch.no_grad():
            new_x = [x_rep[:, :, :self.x_dim]]
            for i in range(len(self.time_series_dims)):
                encoder = self.encoders[i]
                embed_size = encoder.get_embed_size()
                times = torch.linspace(0, 1, steps=2 * encoder.window_size, device=self.device)[np.newaxis, :]
                embed_i = x_rep[:, :, self.x_dim + i * embed_size:self.x_dim + (i + 1) * embed_size]
                poles, coeffs = encoder._embed_split(embed_i)
                curve_i = encoder._decode(poles, coeffs, times, windowed=False)
                new_x.append(torch.real(curve_i))
            new_x = torch.cat(new_x, dim=-1)
        return new_x

    def _get_probs(self, z):
        probs = self.output_fn(z + self.bias)
        return probs

    def forward(self, input):
        # t: batch_size x series_size
        # x: batch_size x series_size x x_dim
        # y: batch_size x series_size x y_dim
        # mask: batch_size x series_size
        # t = input['t']
        x_rep = input['x_rep']
        #y = input['y']
        #mask = input['mask']

        z = self.g(x_rep)
        prob = self._get_probs(z)

        out = {}
        out["prob"] = prob
        return out

    @numpy_io
    def predict_proba_g(self, x, t):
        t = check_shape(t)
        with torch.no_grad():
            x_rep = self._encode(x, t)
            out = self.forward({'x_rep': x_rep})
        return out["prob"]

    @numpy_io
    @run_in_batch
    def predict_proba_from_x_rep(self, x_rep):
        # x_rep: batch_size x x_rep_size
        # x_rep = check_shape(x_rep)
        with torch.no_grad():
            z = self.g(x_rep)
            prob = self._get_probs(z)
        return prob

    @numpy_io
    def embed(self, x, t):
        t = check_shape(t)
        with torch.no_grad():
            x_rep = self._encode(x, t)
            z = self.g(x_rep)
        return z

    # def _smooth_decision_boundary(self, batch):
    #     x_rep = batch['x_rep']
    #     y = batch['y']
    #     mask = batch['mask']
    #
    #     batch_size, series_size, _ = x_rep.shape
    #     half_batch = batch_size // 2
    #     x_rep1 = x_rep[:half_batch]
    #     x_rep2 = x_rep[half_batch:2 * half_batch]
    #     y1 = y[:half_batch]
    #     y2 = y[half_batch:2 * half_batch]
    #     mask1 = mask[:half_batch]
    #     mask2 = mask[half_batch:2 * half_batch]
    #     alpha = 0.5 * np.ones(2)    # 0.5 -- encourage binary weights
    #     w = self.rng.dirichlet(alpha, size=(half_batch, series_size))
    #     w = torch.from_numpy(w).to(x_rep.dtype)
    #     x_rep_w = x_rep1 * w[:, :, [0]] + x_rep2 * w[:, :, [1]]
    #     y_w = y1 * w[:, :, [0]] + y2 * w[:, :, [1]]
    #     mask_w = mask1 * mask2    # only consider valid samples
    #     z_w = self.g(x_rep_w)
    #     prob_w = self._get_probs(z_w)
    #
    #     loss = cross_entropy(prob_w, y_w, mask=mask_w[:, :])
    #     return loss

    def _calculate_train_losses(self, batch):
        # t: batch_size x series_size
        # x: batch_size x series_size x x_dim
        # y: batch_size x series_size x y_dim
        # mask: batch_size x series_size
        # t = batch['t']
        # x = batch['x']
        y = batch['y']
        mask = batch['mask']

        out = self.forward(batch)
        y_pred = out['prob']

        losses = {}
        losses['ce'] = cross_entropy(y_pred, y, mask=mask[:, :])
        # losses['aux'] = self._smooth_decision_boundary(batch)

        # no gradient for the encoder
        # for d in self.time_series_dims:
        #     xd = x[:,:,d]
        #     encoder_losses = self.encoder.expose_loss(xd,t,mask)
        #     for k,v in encoder_losses.items():
        #         losses[k] = losses.get(k,0.0) + v / len(self.time_series_dims)
        return losses

    def _calculate_valid_losses(self, batch):
        # t = valid_set['t']
        # x = valid_set['x']
        mask = batch['mask'].cpu()
        y = batch['y'].cpu()

        out = self.forward(batch)
        y_pred = out['prob'].detach().cpu()

        losses = {}
        losses['ce'] = cross_entropy(y_pred, y, mask=mask[:, :])

        # no gradient for the encoder
        #     for d in self.time_series_dims:
        #         xd = x[:,:,d]
        #         encoder_losses = self.encoder.expose_loss(xd,t,mask)
        #         losses['rmse'] = losses.get('rmse',0.0) + encoder_losses['rmse'] / len(self.time_series_dims)
        #

        AUROC, AUPRC = get_auc_scores(y, y_pred, mask=mask)
        losses['AUROC'] = torch.tensor(np.mean(AUROC))
        losses['AUPRC'] = torch.tensor(np.mean(AUPRC))

        return losses

    def _encode_dataset(self, dataset):
        x = dataset['x']
        t = dataset['t']

        dataset['x_rep'] = self.encode(x, t)

        return dataset

    def predict_cluster(self, x, t, mask=None, steps=None):
        cluster_label = self.cls.predict(x, t, mask, steps)
        return cluster_label

    def predict_proba_cluster(self, x, t, mask=None, steps=None):
        probs = self.cls.predict_proba(x, t, mask, steps)
        return probs

    def predict_proba(self, x, t, mask=None, steps=None):
        t = check_shape(t)
        clusters = self.predict_cluster(x, t, mask, steps)

        proba = np.zeros((*clusters.shape[:2], self.y_dim))
        # outliers without cluster assignment
        proba[clusters == -1] = 0.5
        for i in range(self.cls.n_clusters):
            mask = clusters == i
            p = self.cls.kmeans.get_cluster(i)['p']
            proba[mask] = p
        return proba

    def _get_x_rep_feature_names(self):
        if len(self.time_series_dims) > 0:
            num_poles = self.encoders[0].num_poles
            max_degree = self.encoders[0].max_degree
        else:
            num_poles = 0
            max_degree = 0

        extra_feat = []
        extra_feat += [[f'pole_{i+1}_re', f'pole_{i+1}_im'] for i in range(num_poles)]
        extra_feat += [
            [f'coef_{i+1}_{d+1}_re', f'coef_{i+1}_{d+1}_im'] for i in range(num_poles) for d in range(max_degree)
        ]
        extra_feat = [item for sublist in extra_feat for item in sublist]

        extra_feat_list = []
        for dim in self.time_series_dims:
            feat = self.feature_names[dim]
            extra_feat_list += [f'{feat}_{item}' for item in extra_feat]
        feat_list = []
        for dim in self.static_dims:
            feat_list.append(self.feature_names[dim])
        x_rep_feat_names = feat_list + extra_feat_list

        return x_rep_feat_names

    def explain_cluster(self):
        x = self.cls.x_corpus
        c = self.cls.label_corpus
        self.exp = fit_cluster(x, c, max_depth=3)
        feature_names = self._get_x_rep_feature_names()
        description = explain(self.exp, feature_names=feature_names)
        return description

    def _extract_time_series_data(self, dataset, x_dim):
        x = dataset['x']
        t = dataset['t']
        mask = dataset['range_mask']

        fs = x[:, :, x_dim]
        ts = t
        ms = mask
        return {'f': fs, 't': ts, 'mask': ms}

    def _fit_encoders(self, train_set, loss_weights, valid_set, args, learning_rate, verbose):
        # 1.1 concatenate all time-series and create a new dataset for encoder
        for i, x_dim in enumerate(self.time_series_dims):
            encoder = self.encoders[i]
            encoder_args = args.copy()
            encoder_args['train_set'] = self._extract_time_series_data(train_set, x_dim)
            if valid_set is not None:
                encoder_args['valid_set'] = self._extract_time_series_data(valid_set, x_dim)
            else:
                encoder_args['valid_set'] = None

            e_loss_weights = {}

            # 1.2 fill loss weights dict with default values if not specified
            for k, v in default_encoder_loss_weights.items():
                if k not in loss_weights:
                    e_loss_weights[k] = v
                else:
                    e_loss_weights[k] = loss_weights[k]

            encoder_args['loss_weights'] = e_loss_weights

            encoder_args['parameters'] = [{'params': encoder.parameters(), 'lr': learning_rate}]
            # 1.3 train the encoder
            if verbose:
                print(f'stage 1 - fit the {encoder.name} {i+1}/{len(self.time_series_dims)}')
            encoder.fit(**encoder_args)
            # enable sorting of the poles to obtain equivariant representation
            encoder.equivariant_embed = True

    def evaluate_encoder_params(self, train_set, test_set, loss_weights, valid_set=None, learning_rate=0.1, batch_size=50, epochs=100, max_grad_norm=1,tolerance=None, device=None, verbose=True, **kwargs):
        args = locals().copy()    # shallow copy
        # remove the self variable
        args.pop('self')
        self._fit_encoders(train_set, loss_weights, valid_set, args, learning_rate, verbose)

        loss = 0.0
        for i, x_dim in enumerate(self.time_series_dims):
            encoder = self.encoders[i]
            test_set_i = self._extract_time_series_data(test_set, x_dim)
            f = test_set_i['f']
            t = test_set_i['t']
            m = test_set_i['mask']

            poles, coeffs = encoder.encode(f, t)
            # batch_size x series_size x series_size
            batch_size, series_size = f.shape
            last_step = np.sum(m,axis=-1).astype('int') - 1
            f_rec_r,f_rec_i = encoder.decode(poles, coeffs, t)
            f_rec = f_rec_r + 1j*f_rec_i
            f_rec = f_rec[np.arange(batch_size),last_step]
            diff = np.square(np.abs(f - f_rec))
            diff = np.sum(diff*m, axis=-1) / (np.sum(m,axis=-1)+EPS)
            loss += np.mean(diff)
        loss /= len(self.time_series_dims)
        return loss 
            

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
        if args.get('skip_encoder',False):
            if verbose:
                print('skip training the encoder')
        else:
            self._fit_encoders(train_set, loss_weights, valid_set, args, learning_rate, verbose)

        # stage 2 - train the predictor
        # no gradient for the encoder
        predictor_args = args.copy()
        predictor_args['train_set'] = self._encode_dataset(train_set)
        if valid_set is not None:
            predictor_args['valid_set'] = self._encode_dataset(valid_set)
        else:
            predictor_args['valid_set'] = None
        predictor_args['parameters'] = [{'params': self.g.parameters(), 'lr': learning_rate}]
        if verbose:
            print(f'stage 2 - fit the predictor')
        super().fit(**predictor_args)

        # stage 3 - clustering based on similarity graph
        if verbose:
            print(f'stage 3 - clustering on similarity graph')
        x = train_set['x']
        t = train_set['t']
        mask = train_set['mask']

        self.cls.verbose = verbose
        self.cls.fit(x, t, mask)
        if verbose:
            print(f'done')

        return self
