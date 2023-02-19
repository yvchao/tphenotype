import torch
import numpy as np

from ..base_model import NNBaseModel
from ..nn import MLP
from ..utils.decorators import device_init, numpy_io
from ..utils.utils import EPS


class S2S(NNBaseModel):

    @device_init
    def __init__(self, latent_size: int, hidden_size: int, num_layers: int, **kwargs):
        super().__init__()

        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.name = f'S2S'

        self.encoder = torch.nn.GRU(2, self.latent_size, self.num_layers, batch_first=True)

        self.decoder_rec = torch.nn.GRU(2, self.latent_size, self.num_layers, batch_first=True)
        self.h0_r = torch.nn.Parameter(torch.zeros((self.num_layers, 1, self.latent_size)))

        self.predictor = MLP(self.latent_size, 1, hidden_size=self.hidden_size, num_layers=self.num_layers)

    def forward(self, input):
        # t: batch_size x series_size
        # x: batch_size x series_size x x_dim
        # y: batch_size x series_size x y_dim
        # mask: batch_size x series_size
        t, x = input['t'], input['f']

        z = self._encode(x, t)
        x_rec = self._decode(z, t)

        return x_rec

    def _calculate_train_losses(self, batch):
        # t: batch_size x series_size
        # x: batch_size x series_size x x_dim
        # y: batch_size x series_size x y_dim
        # mask: batch_size x series_size
        # t = batch['t']
        x = batch['f']
        mask = batch['mask']

        x_rec = self.forward(batch)
        diff = torch.square(x - x_rec)
        mse = torch.sum(diff * mask, dim=-1) / (torch.sum(mask, dim=-1) + EPS)
        mse = torch.mean(mse)
        losses = {}
        losses['mse'] = mse
        return losses

    def _calculate_valid_losses(self, batch):
        # t = valid_set['t']
        # x = valid_set['x']
        x = batch['f']
        mask = batch['mask']

        x_rec = self.forward(batch)
        diff = torch.square(x - x_rec)
        mse = torch.sum(diff * mask, dim=-1) / (torch.sum(mask, dim=-1) + EPS)
        mse = torch.mean(mse)

        losses = {}
        losses['mse'] = mse
        return losses

    def _encode(self, x, t):
        dt = torch.zeros_like(t)
        dt[:, 1:] = t[:, 1:] - t[:, :-1]
        xt = torch.cat([x[:, :, np.newaxis], dt[:, :, np.newaxis]], dim=-1)
        # z: batch_size x series_size x latent_size
        z, _ = self.encoder(xt)
        return z[:, -1]

    def _decode(self, z, t):
        batch_size, series_size = t.shape
        dt = torch.zeros_like(t)
        dt[:, 1:] = t[:, 1:] - t[:, :-1]

        # batch_size x 1
        x_hat = self.predictor(z)[:, np.newaxis, :]
        h_tau = self.h0_r.expand(self.num_layers, batch_size, self.latent_size).contiguous()
        xs = [x_hat]
        for tau in range(1, series_size):
            xt = torch.cat([x_hat, dt[:, np.newaxis, [-tau]]], dim=-1)
            z_tau, h_tau = self.decoder_rec(xt, h_tau)
            x_hat = self.predictor(z_tau)
            xs.append(x_hat)
        xs = torch.cat(xs, dim=-2)
        xs = torch.flip(xs, dims=[1])
        return xs[:, :, 0]

    @numpy_io
    def decode(self, z, t):
        z = z.to(self.device)
        t = t.to(self.device)
        with torch.no_grad():
            x_rec = self._decode(z, t)
        return x_rec

    @numpy_io
    def encode(self, x, t):
        x = x.to(self.device)
        t = t.to(self.device)
        with torch.no_grad():
            z = self._encode(x, t)
        return z
