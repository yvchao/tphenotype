from functools import wraps

import numpy as np
import torch

EPS = 1e-10


def device_init(init):
    @wraps(init)
    def decorate(self, *arg, **kwargs):
        init(self, *arg, **kwargs)
        device = kwargs.get("device", None)
        if device is not None:
            assert device in ["cpu", "cuda"]
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    return decorate


def tuple_to(input, to):
    out = tuple(to(i) for i in input)
    return out


def dict_to(dict, to):
    for k, v in dict.items():
        dict[k] = to(v)
    return dict


def numpy_to_tensor(array):
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array)
    else:
        return array


def tensor_to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    else:
        return tensor


def numpy_io(io):
    @wraps(io)
    def decorate(self, *arg, **kwargs):
        arg = tuple_to(arg, numpy_to_tensor)
        kwargs = dict_to(kwargs, numpy_to_tensor)
        out = io(self, *arg, **kwargs)
        if isinstance(out, tuple):
            out = tuple_to(out, tensor_to_numpy)
        elif isinstance(out, dict):
            out = dict_to(out, tensor_to_numpy)
        else:
            out = tensor_to_numpy(out)
        return out

    return decorate


def run_in_batch(fn, batch_size=1000):
    @wraps(fn)
    def decorate(self, *args):
        indices = torch.arange(len(args[0]), dtype=torch.long)
        xs = []
        for k in range(0, len(indices), batch_size):
            idx = indices[k : k + batch_size]
            arg_k = tuple(a[idx] for a in args)
            x = fn(self, *arg_k)
            xs.append(x)
        xs = torch.cat(xs, dim=0)
        return xs

    return decorate
