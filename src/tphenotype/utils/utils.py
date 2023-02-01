import torch
import numpy as np

EPS = 1e-10


def get_summary(metrics):
    info = [f'{metric}:{value:.2f}' for metric, value in metrics.items()]
    return ','.join(info)


def calculate_loss(losses, weights):
    loss = torch.tensor(0.0)
    for k, w in weights.items():
        loss = loss + losses.get(k, 0.0) * w
    return loss


def get_valid_indices(masks):
    valid_indicies = np.full_like(masks, -1, dtype=int)
    for i, mask in enumerate(masks):
        idx, = np.where(mask == 1)
        size = len(idx)
        if size == 0:
            continue
        valid_indicies[i, -size:] = idx
    return valid_indicies


def select_by_steps(series, mask, steps, sub_sequence=False, keepdims=False):
    seqs = []
    shape = series.shape
    if len(shape) <= 1 or len(shape) > 3:
        raise NotImplementedError()
    elif len(shape) == 2:
        series = series[:, :, np.newaxis]
    else:
        pass

    _, _, feature_dim = series.shape
    valid_steps = get_valid_indices(mask)
    max_length = int(np.max(valid_steps)) + 1
    indicies = np.arange(len(series))
    for step in steps:
        t = valid_steps[:, step]
        valid = (t >= 0)
        idx = indicies[valid]
        t = t[valid].astype('int')

        if sub_sequence:
            seq = np.zeros((len(idx), max_length, feature_dim))
            for i in range(max_length):
                t_i = t - i
                mask = t_i >= 0
                seq_i = series[idx[mask], t_i[mask]]
                seq[mask, max_length - 1 - i] = seq_i
        else:
            seq = series[idx, t]

        seqs.append(seq)
    series = np.concatenate(seqs, axis=0)
    if not keepdims:
        series = np.squeeze(series)
    return series


def batch_KL(P, Q):
    # P, Q: batch_size, y_dim
    d_1 = P[..., :, np.newaxis, :] * (np.log(P[..., :, np.newaxis, :] + EPS) - np.log(Q[..., np.newaxis, :, :] + EPS))
    d_0 = (1 - P[..., :, np.newaxis, :]) * (
        np.log(1 - P[..., :, np.newaxis, :] + EPS) - np.log(1 - Q[..., np.newaxis, :, :] + EPS))
    d = d_1 + d_0
    # d: batch_size x batch_size
    d = np.mean(d, axis=-1)
    return d


def plogp_q(p, q):
    return p * (np.log(p + EPS) - np.log(q + EPS))


def d_kl(p, q):
    # p,q: ... x y_dim for categorical distribution
    # d: ...
    d = np.sum(plogp_q(p, q), axis=-1)
    return d


def batch_d(p, q):
    # ... x batch_size_p x batch_size_q x y_dim
    m = 0.5 * (p[..., :, np.newaxis, :] + q[..., np.newaxis, :, :])
    # ... x batch_size_p x batch_size_q
    d_p_m = d_kl(p[..., :, np.newaxis, :], m)
    d_q_m = d_kl(q[..., np.newaxis, :, :], m)
    d = 0.5 * (d_p_m + d_q_m)
    return d


def check_shape(series):
    # ensure the time series is a 2-D array
    shape = series.shape
    if len(shape) == 1:
        series = series[np.newaxis, :]
    elif len(shape) == 2:
        pass
    else:
        raise NotImplementedError()
    return series
