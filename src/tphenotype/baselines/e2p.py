import torch
import numpy as np

from ..base_model import NNBaseModel
from ..nn import TimeSeriesEncoder, MLP
from ..utils.decorators import device_init, numpy_io
from ..utils.metrics import get_auc_scores
from ..utils.losses import cross_entropy


class E2P(NNBaseModel):

    @device_init
    def __init__(self,
                 x_dim: int,
                 y_dim: int,
                 latent_size: int,
                 hidden_size: int,
                 num_layers: int,
                 latent_space: str = 'z',
                 **kwargs):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.latent_space = latent_space
        if self.latent_space == 'y':
            self.embed_size = self.y_dim
        elif self.latent_space == 'z':
            self.embed_size = self.latent_size
        elif self.latent_space == 'y-1':
            self.embed_size = self.y_dim
        else:
            raise NotImplementedError(f'unknown latent space {self.latent_space}')
        self.name = f'E2P({self.latent_space})'

        self.encoder = TimeSeriesEncoder(self.x_dim + 1, self.latent_size, self.hidden_size, self.num_layers)
        self.predictor = MLP(self.latent_size, self.y_dim, self.hidden_size, self.num_layers)

    def forward(self, input):
        # t: batch_size x series_size
        # x: batch_size x series_size x x_dim
        # y: batch_size x series_size x y_dim
        # mask: batch_size x series_size
        t, x = input['t'], input['x']

        z = self._encode(x, t)
        logits = self.predictor(z)

        # probs: batch_size x input_steps x y_dim
        probs = torch.softmax(logits, dim=-1)
        out = {}
        out['prob'] = probs

        return out

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

        AUROC, AUPRC = get_auc_scores(y, y_pred, mask=mask)
        losses['AUROC'] = torch.tensor(np.mean(AUROC))
        losses['AUPRC'] = torch.tensor(np.mean(AUPRC))
        return losses

    def _encode(self, x, t):
        dt = torch.zeros_like(t)
        dt[:, 1:] = t[:, 1:] - t[:, :-1]
        xt = torch.cat([x, dt[:, :, np.newaxis]], dim=-1)
        # z: batch_size x series_size x latent_size
        z = self.encoder(xt)
        return z

    @numpy_io
    def encode(self, x, t):
        x = x.to(self.device)
        t = t.to(self.device)
        with torch.no_grad():
            z = self._encode(x, t)
            if self.latent_space == 'y':
                logits = self.predictor(z)
                embeds = torch.softmax(logits, dim=-1)
            elif self.latent_space == 'y-1':
                embeds = self.predictor(z)
            elif self.latent_space == 'z':
                embeds = z
            else:
                embeds = z
        return embeds

    @numpy_io
    def predict_proba(self, x, t):
        x = x.to(self.device)
        t = t.to(self.device)
        with torch.no_grad():
            out = self.forward({'x': x, 't': t})
            # probs: batch_size x series_size x y_dim
            probs = out['prob']
        return probs

    def predict_cluster(self, x, t):
        raise NotImplementedError()
