import numpy as np
import torch
import torch.utils.data
from sklearn.model_selection import train_test_split

from .utils import select_by_steps


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets, dtype=int).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


def wrap_data(data_seq, keys, dtype=None):
    dataset = {keys[i]: np.array(data_seq[i], dtype=dtype) for i in range(len(keys))}
    return dataset


def data_split(dataset, test_size=0.2, random_state=0, dtype=None):
    keys = list(dataset.keys())
    values = [dataset[k] for k in keys]
    split = train_test_split(*values, test_size=test_size, random_state=random_state)
    train_set = [split[i] for i in range(len(split)) if i % 2 == 0]
    test_set = [split[i] for i in range(len(split)) if i % 2 == 1]
    train_set = wrap_data(train_set, keys, dtype=dtype)
    test_set = wrap_data(test_set, keys, dtype=dtype)
    return train_set, test_set


def reverse_seq(seq, mask):
    seq_new = np.zeros_like(seq)
    for i, m in enumerate(mask):
        valid = m == 1
        size = np.sum(valid)
        if size == 0:
            continue
        seq_new[i, :size] = seq[i, valid]
    return seq_new


def cut_windowed_data(t, x, y, m, window_size=10, steps=(-1,), range_mask=False):
    mask_sel = select_by_steps(m, m, steps, sub_sequence=True)
    x_sel = select_by_steps(x, m, steps, sub_sequence=True, keepdims=True)
    t_sel = select_by_steps(t, m, steps, sub_sequence=True)
    y_sel = select_by_steps(y, m, steps, sub_sequence=True, keepdims=True)
    t_start = np.max(t_sel, axis=-1, keepdims=True) - window_size
    mask_sel[t_sel <= t_start] = 0
    max_length = np.max(mask_sel.sum(axis=-1)).astype("int")
    x_sel = reverse_seq(x_sel, mask_sel)[:, :max_length]
    t_sel = reverse_seq(t_sel, mask_sel)[:, :max_length]
    y_sel = reverse_seq(y_sel, mask_sel)[:, :max_length]
    m_sel = reverse_seq(mask_sel, mask_sel)[:, :max_length]
    end_mask = np.zeros_like(m_sel)
    end_mask[np.arange(len(m_sel)), m_sel.sum(axis=-1).astype("int") - 1] = 1
    if not range_mask:
        m_sel = end_mask

    length = np.argmax(end_mask, axis=-1) + 1
    keep = length > 1
    return t_sel[keep], x_sel[keep], y_sel[keep], m_sel[keep]


class Dataset(torch.utils.data.Dataset):
    def update_property(self, property_, value):
        setattr(self, property_, value)

    def __init__(self, dataset):
        super().__init__()
        self.keys = dataset.keys()
        for k in self.keys:
            self.update_property(k, dataset[k])
            self.sample_size = len(getattr(self, k))

    def __len__(self):
        return self.sample_size

    def __getitem__(self, index):
        ret = {k: getattr(self, k)[index] for k in self.keys}
        return ret

    def get_all_data(self):
        ret = {k: getattr(self, k) for k in self.keys}
        for k, v in ret.items():
            if isinstance(v, torch.Tensor):
                pass
            elif isinstance(v, np.ndarray):
                ret[k] = torch.from_numpy(v)
            else:
                raise ValueError(f"Unknown data type {type(v)} for {k}.")
        return ret
