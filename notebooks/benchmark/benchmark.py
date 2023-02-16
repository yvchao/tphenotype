import torch
import pandas as pd
import numpy as np
from numpy.core.multiarray import interp
from tqdm import auto

from tphenotype.utils import get_auc_scores, get_cls_scores, select_by_steps, data_split


def batch_interp(t, tp, fp):
    return np.stack([interp(t, tp[i], fp[i]) for i in range(fp.shape[0])], axis=0)


def data_interpolation(t, x, samples=20):
    times = np.linspace(0, 1, num=samples)
    _, _, x_dim = x.shape
    xs = []
    for i in range(x_dim):
        x_interp = batch_interp(times, t, x[:, :, i])
        xs.append(x_interp)
    xs = np.stack(xs, axis=-1)
    return xs


def evaluate(model, dataset, steps=[-1]):
    x = dataset['x']
    t = dataset['t']
    c = dataset.get('c', None)
    y = dataset['y']
    mask = dataset['mask']

    try:
        c_pred = model.predict_cluster(x, t, mask, steps)
        y_pred = model.predict_proba(x, t, mask, steps)
    except:
        try:
            c_pred = model.predict_cluster(x, t)
            c_pred = select_by_steps(c_pred, mask, steps)
        except:
            c_pred = None

        try:
            y_pred = model.predict_proba(x, t)
            y_pred = select_by_steps(y_pred, mask, steps)

        except:
            y_pred = None

    if c is not None:
        c_true = select_by_steps(c, mask, steps)
    else:
        c_true = None

    y_true = select_by_steps(y, mask, steps)

    if y_pred is not None:
        AUROC, AUPRC = get_auc_scores(y_true, y_pred)
        auc_scores = {'ROC': np.mean(AUROC), 'PRC': np.mean(AUPRC)}
    else:
        auc_scores = {}

    if c_pred is not None:
        x_sel = select_by_steps(x, mask, steps, sub_sequence=True, keepdims=True)
        t_sel = select_by_steps(t, mask, steps, sub_sequence=True, keepdims=False)
        x_interp = data_interpolation(t_sel, x_sel)
        cls_scores = get_cls_scores(c_true=c_true, c_pred=c_pred, x=x_interp, y_true=y_true)
    else:
        cls_scores = {}

    mixed1 = 2/ (1/auc_scores.get('ROC',1e-10) + 1/cls_scores.get('Silhouette_auc',1e-10))
    mixed2 = 2/ (1/auc_scores.get('PRC',1e-10) + 1/cls_scores.get('Silhouette_auc',1e-10))
    scores = {'method': model.name, **auc_scores, **cls_scores, 'Hroc':mixed1,'Hprc':mixed2}
    return scores


def get_ci(series, decimals=3):
    if series.dtype == 'object':
        return series.iloc[0]

    stats = series.agg(['mean', 'sem'])
    mean = np.format_float_positional(stats['mean'], decimals)
    ci = np.format_float_positional(1.96 * stats['sem'], decimals)
    out = f'{mean}+-{ci}'
    return out


def benchmark_old(method, config, dataset, loss_weights, steps=[-1], epochs=50, n=3, seed=0):
    dtype='float32'
    
    #train_set, valid_set, test_set = dataset
    #N, T, _ = dataset['x'].shape

    results = []
    for i in auto.tqdm(range(n), desc=f'{method.__name__}'):
        train_set, test_set = data_split(dataset, test_size=0.2, random_state=seed+i, dtype=dtype)
        train_set, valid_set = data_split(train_set, test_size=0.2, random_state=seed+i, dtype=dtype)
    
        torch.random.manual_seed(seed+i)
        model = method(**config)
        model = model.fit(train_set, loss_weights, valid_set=valid_set, epochs=epochs, verbose=False)
        scores = evaluate(model, test_set, steps)
        results.append(scores)
    results = pd.DataFrame(results)
    summary = results.apply(get_ci)
    return summary

def benchmark(method, config, splits, loss_weights, steps=[-1], epochs=50, seed=0):
    dtype='float32'
    

    results = []
    for i,dataset in auto.tqdm(enumerate(splits), total=len(splits), desc=f'{method.__name__}'):
        train_set, valid_set, test_set = dataset
    
        torch.random.manual_seed(seed+i)
        model = method(**config)
        model = model.fit(train_set, loss_weights, valid_set=valid_set, epochs=epochs, verbose=False)
        scores = evaluate(model, test_set, steps)
        results.append(scores)
    results = pd.DataFrame(results)
    summary = results.apply(get_ci)
    return summary


KME2P_config = {
    'K': None,
    'x_dim': None,
    'y_dim': None,
    'latent_size': 10,
    'hidden_size': 10,
    'num_layers': 2,
    'device': 'cpu',
}

Encoder_config = {
    'num_poles': 4,    # number of poles
    'max_degree': 1,    # maximum degree of poles
    'hidden_size': 10,    # number of hidden units in neural networks
    'num_layers': 1,    # number of layers in MLP of the encoder (1 layer RNN + n layer MLP)
    'pole_separation': 1.0,    # minimum distance between distinct poles 
    'freq_scaler': 20,    # scale up the imaginary part to help learning
    'window_size': None,    # whether or not to include time delay terms
    'equivariant_embed': True,    # whether or not to sort the poles (useless during training)
    'device': 'cpu',
}

Cls_config = {
    'K': None,
    'steps': [-1],
    'tol': 1e-6,
    'test_num': 50,
}

Predictor_config = {
    'x_dim': None,
    'y_dim': None,
    'time_series_dims': None,
    'hidden_size': 10,
    'num_layer': 3,
    'global_bias': False,
    'encoder_config': None,
    'cls_config': None,
    'categorical': True,
    'device': 'cpu',
}

loss_weights = {
    'ce': 1.0,
    'rmse': 1.0,
    'cont':0.01,
    'pole': 1.0,
    'real': 0.1
}
