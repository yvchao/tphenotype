import torch

from .utils import EPS


def safe_div(num, den, eps=EPS):
    out = (num + eps) / (den + eps)
    return out


def cross_entropy(input_, target, mask=None, weight=None, eps=EPS):
    # loss = target * torch.log(input_ + eps) + (1 - target) * torch.log(1 - input_ + eps)
    # categorical cross entropy
    loss = target * torch.log(input_ + eps)
    loss = torch.sum(loss, dim=-1)
    if weight is not None:
        loss = loss * weight
    if mask is not None:
        loss = torch.sum(loss * mask)
        loss = -safe_div(loss, torch.sum(mask))
    else:
        loss = -torch.mean(loss)
    return loss
