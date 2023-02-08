#!/usr/bin/env python
# coding: utf-8

import pickle
from tqdm import auto
import numpy as np
import pandas as pd
import itertools
import torch
import os
import json
from tphenotype import LaplaceEncoder, Predictor
from tphenotype.baselines import E2P, KME2P, KMDTW
from tphenotype.utils import get_auc_scores, get_cls_scores, select_by_steps
from benchmark import benchmark, KME2P_config, Predictor_config, Encoder_config, Cls_config, loss_weights



def evaluate_encoder(method, config, loss_weights, splits,seed=0, epochs=50):
    results = []
    for i,dataset in auto.tqdm(enumerate(splits), total=len(splits), desc=f'{method.__name__}'):
        train_set, valid_set, test_set = dataset

        torch.random.manual_seed(seed+i)
        model = method(**config)
        mse = model.evaluate_encoder_params(train_set, test_set, loss_weights, valid_set=valid_set, epochs=epochs, verbose=False)
        results.append(mse)
    results = np.array(results)
    return results, model

os.makedirs('hyperparam_selection', exist_ok=True)



# In[5]:


def load_data(dataname, verbose=False):
    with open(f'data/{dataname}_data_hs.pkl', 'rb') as file:
        splits = pickle.load(file)
        
    if dataname == 'Synth':
        feat_list=['x1','x2']
        temporal_dims = [0,1]
    elif dataname == 'ADNI':
        feat_list = ['APOE4','CDRSB','Hippocampus']
        temporal_dims = [1,2]
    elif dataname == 'ICU':
        feat_list = ['Age', 'Gender', 'GCS', 'PaCO2']
        temporal_dims = [2, 3]
    else:
        raise ValueError(f'unknown dataset {dataname}')
    
    if verbose:
        tr_set,va_set,te_set = splits[0]
        _, T, x_dim = tr_set['x'].shape
        _, _, y_dim = tr_set['y'].shape
        print(dataname)
        print(f"total samples: {len(tr_set['x'])+len(va_set['x'])+len(te_set['x'])}")
        
        print(f'max length: {T}')
        print(f'x_dim: {x_dim}')
        print(f'y_dim: {y_dim}')

        print(f'features: {feat_list}')
        print(f'temporal dims: {temporal_dims}')
    return splits, feat_list, temporal_dims


# In[7]:


def hyperparam_selection_encoder(dataname,seed=0, epochs = 50):
    splits, feat_list, temporal_dims = load_data(dataname, verbose=True)
    tr_set,va_set,te_set = splits[0]
    _, T, x_dim = tr_set['x'].shape
    _, _, y_dim = tr_set['y'].shape

    # Configuration
    K=1

    encoder_config = Encoder_config.copy()
    encoder_config['window_size']=None
    encoder_config['pole_separation']=2.0
    encoder_config['freq_scaler']=20

    cls_config = Cls_config.copy()
    cls_config['K'] = K
    cls_config['steps'] = [-1]
    predictor_config = Predictor_config.copy()
    predictor_config['x_dim'] = x_dim
    predictor_config['y_dim'] = y_dim
    predictor_config['time_series_dims'] = temporal_dims
    predictor_config['cls_config'] = cls_config
    predictor_config['encoder_config'] = encoder_config

    loss_weights['cont']=0.01

    print('T_phenotype config:')
    print(predictor_config)
    result_file = f'hyperparam_selection/{dataname}_encoder.csv'

    scores = pd.DataFrame(columns=['mse_mean','mse_std', 'config'])
    for i,comb in enumerate(itertools.product(*search_space.values())):
        test_config = encoder_config.copy()
        test_loss_weights = loss_weights.copy()
        msg = []
        for j,k in enumerate(search_space.keys()):
            if k in encoder_config:
                test_config[k] = comb[j]
            elif k in loss_weights:
                test_loss_weights[k]=comb[j]
            msg.append(f'{k}:{comb[j]}')
        predictor_config['encoder_config'] = test_config
        msg = ','.join(msg)
        print(f'test config {msg} ...')
        results, model = evaluate_encoder(Predictor,predictor_config,test_loss_weights,splits,seed=seed, epochs=epochs)
        scores.loc[i,'mse_mean'] = np.mean(results)
        scores.loc[i,'mse_std'] = np.std(results)
        scores.loc[i,'config'] = msg
        scores.to_csv(result_file)

    scores = pd.read_csv(result_file, index_col=0)
    scores = scores.astype({'mse_mean':'float'})
    best = scores['mse_mean'].idxmin()
    print('Optimal hyperparameters:')
    print(scores.loc[best,'config'])


# In[8]:
# In[4]:


loss_weights = {
    'rmse': 1.0,
    'cont':0.01,
    'pole': 1.0,
    'real': 0.1
}

search_space = {
    'pole':[1.0,10.0],
    'real':[0.1,1.0],
    'pole_separation':[1.0, 2.0],
    'max_degree': [1,2],
}



for dataname in ['Synth', 'ICU', 'ADNI']:
    hyperparam_selection_encoder(dataname)


# In[ ]:




