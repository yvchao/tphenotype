import torch
import numpy as np
from ..base_model import NNBaseModel
from ..nn import TimeSeriesEncoder
from ..utils import sort_complex
from ..utils.utils import check_shape, EPS
from ..utils.decorators import device_init, numpy_io
from .symbolic_eval import get_transfer_function, get_function, plot


def pairwise_distances(x):
    #x should be at least two dimensional
    diff = x[..., :, np.newaxis, :] - x[..., np.newaxis, :, :]
    dist = torch.norm(diff, p='fro', dim=-1)
    return dist


def find_similar_series(curves, threshold):
    dist = pairwise_distances(curves)
    # mask = dist <= threshold
    # dist[mask] = 1
    # dist[~mask] = 0
    dist = torch.square(dist)
    dist = torch.exp(-dist)
    return dist


class LaplaceEncoder(NNBaseModel):

    @device_init
    def __init__(self,
                 num_poles=4,
                 max_degree=1,
                 pole_separation=0.1,
                 freq_scaler=10.0,
                 coeff_scaler=5.0,
                 window_size=None,
                 hidden_size=20,
                 num_layers=2,
                 equivariant_embed=True,
                 **kwargs):
        super().__init__()
        self.name = 'Laplace Encoder'
        self.num_poles = num_poles    # number of poles in the lapace transform
        self.max_degree = max_degree    # maximum degree of poles in the lapace transform
        assert self.num_poles >= 2
        assert self.max_degree >= 1
        self.window_size = window_size    # window size
        self.embed_size = 2 * (self.num_poles + self.max_degree * self.num_poles)

        self.freq_scaler = freq_scaler
        assert self.freq_scaler >= 1
        self.pole_range = {'re': [-10, 10], 'im': [-freq_scaler, freq_scaler]}
        self.coeff_range = {'max': coeff_scaler}

        self.hidden_size = hidden_size    # hidden unit number of rnn and mlp
        self.num_layers = num_layers    # layer number of mlp
        self.delta_pole = pole_separation    # threshold for identical poles: |p_i-p_j|< \delta => identical
        assert self.delta_pole >= 0
        self.equivariant_embed = equivariant_embed    # whether or not to reorder the poles to get an equivariant representation

        # encoder = rnn (1 layer) + mlp (num_layer layers)
        self.encoder = TimeSeriesEncoder(
            input_size=2, output_size=self.embed_size, hidden_size=self.hidden_size, num_layers=self.num_layers)

    def get_embed_size(self):
        return self.embed_size

    def F(self, poles, coeffs, idx, step):
        p = poles[idx, step]
        c = coeffs[idx, step]
        F, s = get_transfer_function(p, c)
        return F

    def f(self, poles, coeffs, idx, step):
        p = poles[idx, step]
        c = coeffs[idx, step]
        f, t = get_function(p, c, return_complex=False)
        return f

    def plot(self, poles, coeffs, idx, step):
        p = poles[idx, step]
        c = coeffs[idx, step]
        f, t = get_function(p, c, num_digits=10)
        fig = plot(f, t, range=(-0.05, 1.05))
        return fig

    @numpy_io
    def encode(self, f, t):
        # encode a time-series via poles and coefficients
        f = check_shape(f).to(self.device)
        t = check_shape(t).to(self.device)

        with torch.no_grad():
            poles, coeffs = self._encode(f, t)

        return poles, coeffs

    @numpy_io
    def decode(self, poles, coeffs, times):
        # reconstruct a time-series at given time points from poles and coefficients

        poles = poles.to(self.device)
        coeffs = coeffs.to(self.device)
        times = check_shape(times).to(self.device)

        with torch.no_grad():
            fs = self._decode(poles, coeffs, times)
            fs_r = torch.real(fs)
            fs_i = torch.imag(fs)
        return fs_r, fs_i

    @numpy_io
    def get_embed(self, f, t):
        f = check_shape(f).to(self.device)
        t = check_shape(t).to(self.device)
        with torch.no_grad():
            embed = self._get_embed(f, t)
        return embed

    def _get_embed(self, f, t, normalize=False):
        poles, coeffs = self._encode(f, t)
        embed = self._laplace_to_embed(poles, coeffs, normalize)
        return embed

    def _laplace_to_embed(self, poles, coeffs, normalize=False):
        batch_size, series_size, _ = poles.shape
        poles = torch.view_as_real(poles)
        coeffs = torch.view_as_real(coeffs)
        if normalize:
            re_min, re_max = self.pole_range['re']
            poles_re = poles[:, :, :, 0] / max(-re_min, re_max)
            poles_im = poles[:, :, :, 1] / self.freq_scaler
            poles = torch.stack([poles_re, poles_im], dim=-1)

            coeffs = coeffs / self.coeff_range['max']

        poles = poles.reshape((batch_size, series_size, -1))
        coeffs = coeffs.reshape((batch_size, series_size, -1))
        embed = torch.cat([poles, coeffs], dim=-1)
        return embed

    @numpy_io
    def embed_split(self, embed):
        embed = embed.to(self.device)
        poles, coeffs = self._embed_split(embed)
        return poles, coeffs

    def _embed_split(self, embed):
        # convert the embedding into poles and coefficiets
        batch_size, series_size, _ = embed.shape
        poles_embed_size = 2 * self.num_poles
        coeffs_embed_size = 2 * (self.max_degree * self.num_poles)
        poles = embed[:, :, :poles_embed_size].reshape((batch_size, series_size, self.num_poles, 2))
        poles_re = poles[:, :, :, 0]

        # manually scale up the imaginary part such that the output of NN can be small
        poles_im = self.freq_scaler * torch.tanh(poles[:, :, :, 1])

        # most time-series would not
        # - decay too fast ~ e^(-10t)
        # - increase too fast ~ e^(t)
        re_min, re_max = self.pole_range['re']
        poles_re = torch.clamp(poles_re, min=re_min, max=re_max)
        # poles_re = poles_re + (poles_re_clamp - poles_re).detach()

        # high frequency components are usually noises
        # poles_im_clamp = torch.clamp(poles_im,max=10*self.freq_scaler,min=-10*self.freq_scaler)
        # poles_im = poles_im + (poles_im_clamp - poles_im).detach()

        constrained_poles = torch.stack([poles_re, poles_im], dim=-1)
        poles = torch.view_as_complex(constrained_poles)

        coeffs = embed[:, :, poles_embed_size:poles_embed_size + coeffs_embed_size].reshape(
            (batch_size, series_size, self.num_poles, self.max_degree, 2))

        multiplier = self.coeff_range['max']
        coeffs = multiplier * torch.tanh(coeffs)
        coeffs = torch.view_as_complex(coeffs)

        # ensure the equivariant representation: the order of poles doesn't matter
        # we use a lexical order to sort the poles (and coefficients) as the output
        if self.equivariant_embed:
            poles, coeffs = self._sort_poles(poles, coeffs)
        return poles.contiguous(), coeffs.contiguous()

    def _sort_poles(self, poles, coeffs):
        idx = np.apply_along_axis(
            lambda x: sort_complex(x, threshold=self.delta_pole), axis=-1, arr=poles.detach().cpu())
        idx = torch.from_numpy(idx).to(self.device)
        poles = torch.take_along_dim(poles, idx, dim=-1)
        coeffs = torch.take_along_dim(coeffs, idx[..., np.newaxis], dim=-2)
        return poles, coeffs

    def _unfold(self, series, fill_value=0.0):
        batch_size, _ = series.shape
        padding = fill_value * series.new_ones((batch_size, self.window_size - 1))
        series = torch.cat([padding, series], dim=1)
        # batch_size x series_size x window_size
        unfold = series.unfold(1, self.window_size, 1)
        return unfold

    def _encode(self, f, t):
        # f: batch_size x series_size
        # t: batch_size x series_size
        batch_size, series_size = f.shape
        dt = torch.zeros_like(t)
        dt[:, 1:] = t[:, 1:] - t[:, :-1]
        if self.window_size is not None:
            # dt: batch_size x series_size x window_size
            dt = self._unfold(dt)
            dt[:, :, 0] = 0    # set initial delta t to zero
            # f: batch_size x series_size x window_size
            f = self._unfold(f)
            ft = torch.stack([f, dt], dim=-1)
            ft = ft.reshape((batch_size * series_size, self.window_size, 2))
            # batch_size*series_size x window_size x embed_size
            embed = self.encoder(ft)
            embed = embed.reshape((batch_size, series_size, self.window_size, self.embed_size))
            # batch_size x series_size x embed_size
            embed = embed[:, :, -1, :]
        else:
            ft = torch.stack([f, dt], dim=-1)
            # embed: batch_size x series_size x embed_size
            embed = self.encoder(ft)

        poles, coeffs = self._embed_split(embed)
        return poles, coeffs

    def _ilt(self, poles, coeffs, times):
        # poles: ... x series_size x num_poles
        # coeffs: ... x series_size x num_poles x max_degree
        # times: ... x series_size x window_size
        components = 0
        for d in range(self.max_degree):
            c_id = coeffs[..., :, np.newaxis, :, d]
            p_i = poles[..., :, np.newaxis, :]
            t = times[..., :, :, np.newaxis]
            # t = torch.clamp(t, min=0.0, max=1.0)
            components += 1.0 * (t > 0) * c_id * torch.pow(t, d) * torch.exp(p_i * t) / np.math.factorial(d)
        fs = torch.sum(components, axis=-1)
        # fs: ... x series_size x window_size
        return fs

    def _decode(self, poles, coeffs, times, windowed=True):
        # poles: batch_size x series_size x num_poles
        # coeffs: batch_size x series_size x num_poles x max_degree
        # times: batch_size x series_size
        if windowed and self.window_size is not None:
            # times: batch_size x series_size x window_size
            times = self._unfold(times, fill_value=100.0)
            time_begin, _ = torch.min(times, dim=-1, keepdim=True)
            times = times - time_begin + EPS
            times[times > 10.0] = -EPS
        else:
            # times: batch_size x 1 x window_size
            times = times[:, np.newaxis, :]
        # fs: ... x series_size x window_size
        fs = self._ilt(poles, coeffs, times)
        return fs

    def forward(self, input):
        f = input['f']
        t = input['t']

        poles, coeffs = self._encode(f, t)
        f_rec = self._decode(poles, coeffs, t)
        out = {}
        out['f_rec'] = f_rec
        out['poles'] = poles
        out['coeffs'] = coeffs
        return out

    def _rmse_f(self, f, f_rec):
        # RMSE between observed f and reconstructed f_rec
        # f: batch_size x series_size
        # f_rec: batch_size x series_size x series_size

        if self.window_size is not None:
            # f: batch_size x series_size x window_size
            f = self._unfold(f)
            diff = f - f_rec
        else:
            f = f[:, np.newaxis, :]
            # diff: batch_size x series_size x series_size
            diff = f - f_rec
            # only consider the relevant time window for each step
            # step 1: 1 ~ 1
            # step 2: 1 ~ 2
            # ...
            # step T: 1 ~ T
            # thus, we use the torch.tril function to set the upper triangular part to be zero
            diff = torch.tril(diff)

        # absolute error of two complex functions
        diff = torch.abs(diff)
        mse = torch.mean(torch.square(diff), dim=-1)
        # rmse: batch_size x series_size
        # rmse = torch.sqrt(mse)
        return mse

    def _zero_imag(self, poles, coeffs):
        # evaluate the imaginary part of ILT[F(s)] at larger scale
        batch_size, series_size, _ = poles.shape
        # randomly scale time steps
        t_scale = torch.linspace(0, 1, steps= series_size, device=self.device) \
                + 0.5 / series_size * torch.randn((batch_size, series_size), device=self.device)
        t_scale = torch.clamp(t_scale, min=0, max=1)
        # f_rec: batch_size x series_size x series_size
        f_rec = self._decode(poles, coeffs, t_scale, windowed=False)
        f_rec_im = torch.imag(f_rec)
        # f_rec_im = torch.nan_to_num(f_rec_im,posinf=1e10,neginf=-1e10)
        mse = torch.mean(torch.square(f_rec_im), dim=-1)
        # rmse: batch_size x series_size
        # rmse = torch.sqrt(mse)
        return mse

    # def _l1_coeffs(self, coeffs):
    #     # l1 regularizer to encourage minimum number of components to be used
    #     batch_size, series_size, num_poles, max_degree = coeffs.shape
    #     coeffs = coeffs.reshape((batch_size, series_size, num_poles * max_degree))
    #     rs = torch.abs(coeffs)
    #     r_max, _ = torch.max(rs, dim=-1, keepdim=True)
    #     norms = rs / r_max.detach()
    #     l1 = torch.sum(norms, dim=-1)
    #     # l1: batch_size x series_size
    #     return l1

    def _unique(self, poles, coeffs, mask):
        batch_size, series_size, _ = poles.shape
        if self.window_size is not None:
            length = self.window_size
        else:
            length = series_size
        times = torch.linspace(0, 1, steps=length, device=self.device)[np.newaxis, :]
        poles = poles[mask == 1.0].reshape((-1, 1, self.num_poles))
        coeffs = coeffs[mask == 1.0].reshape((-1, 1, self.num_poles, self.max_degree))
        # batch_size x 1 x window_size
        curves = self._decode(poles, coeffs, times, windowed=False)
        curves = torch.real(curves)[:, 0, :]
        embed = self._laplace_to_embed(poles, coeffs).reshape((-1, self.embed_size))
        similarity = find_similar_series(curves, threshold=0.1).detach()
        dist_w = pairwise_distances(embed)
        dist_w = torch.square(dist_w)
        loss = similarity * dist_w
        loss = torch.mean(loss)
        return loss

    def _separation_poles(self, poles):
        # penalty term for very close poles

        # poles: batch_size x series_size x num_poles
        diff = poles[:, :, :, np.newaxis] - poles[:, :, np.newaxis, :]
        # condition: batch_size x series_size x num_poles x num_poles
        # condition_{i,j} = max(0, delta - |pole_i - pole_j|)
        condition = torch.clamp(self.delta_pole - torch.abs(diff), min=0)
        # separation: batch_size x series_size
        separation = 0.5 * (torch.sum(condition, dim=(-2, -1)) - self.num_poles * self.delta_pole)
        return separation

    def expose_loss(self, f, t, mask):
        data = {'f': f, 't': t, 'mask': mask}
        losses = self._calculate_train_losses(data)
        return losses

    def _calculate_train_losses(self, batch):
        out = self.forward(batch)

        f = batch['f']
        mask = batch['mask']
        f_rec = out['f_rec']
        poles = out['poles']
        coeffs = out['coeffs']

        # rmse: batch_size x series_size
        rmse = self._rmse_f(f, f_rec)
        rmse = torch.sum(rmse * mask) / (torch.sum(mask) + EPS)

        # contrastive loss
        contrast = self._unique(poles[:10], coeffs[:10], mask[:10])

        # separation: batch_size x series_size
        separation = self._separation_poles(poles)
        separation = torch.sum(separation * mask) / (torch.sum(mask) + EPS)

        # real: zero imagine part
        real = self._zero_imag(poles, coeffs)
        real = torch.sum(real * mask) / (torch.sum(mask) + EPS)

        losses = {}
        losses['rmse'] = rmse
        losses['cont'] = contrast
        losses['pole'] = separation
        losses['real'] = real

        return losses

    def _calculate_valid_losses(self, batch):
        f = batch['f']
        mask = batch['mask']
        out = self.forward(batch)
        f_rec = out['f_rec']
        # poles = out['poles']
        # coeffs = out['coeffs']

        # rmse: batch_size x series_size
        rmse = self._rmse_f(f, f_rec)
        rmse = torch.sum(rmse * mask) / (torch.sum(mask) + EPS)

        losses = {}
        losses['rmse'] = rmse
        return losses
