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
from tphenotype.baselines import E2P, KME2P, KMDTW, SpectralDTW
from tphenotype.utils import get_auc_scores, get_cls_scores, select_by_steps
from benchmark import benchmark,evaluate, KME2P_config, Predictor_config, Encoder_config, Cls_config, loss_weights


def evaluate_predictor(method, config, loss_weights, splits,seed=0, epochs=50, steps=[-1], metric='Hprc'):
    results = []
    for i,dataset in auto.tqdm(enumerate(splits), total=len(splits), desc=f'{method.__name__}'):
        train_set, valid_set, test_set = dataset

        torch.random.manual_seed(seed+i)
        model = method(**config)
        model = model.fit(train_set, loss_weights, valid_set=valid_set, epochs=epochs, verbose=False)
        scores = evaluate(model, test_set, steps)
        results.append(scores[metric])
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


def hyperparam_selection_predictor(dataname,search_space, K, seed=0, epochs = 50):
    splits, feat_list, temporal_dims = load_data(dataname, verbose=True)
    tr_set,va_set,te_set = splits[0]
    _, T, x_dim = tr_set['x'].shape
    _, _, y_dim = tr_set['y'].shape

    # Configuration
    spectral_config = {'K':K, 'sigma':1.0}

    result_file = f'hyperparam_selection/{dataname}_Spectral.csv'
    if os.path.exists(result_file):
        search_space = {}

    scores = pd.DataFrame(columns=['H_mean','H_std', 'config'])
    for i,comb in enumerate(itertools.product(*search_space.values())):
        if len(comb)==0:
            continue
            
        test_config = spectral_config.copy()
        msg = []
        for j,k in enumerate(search_space.keys()):
            if k in spectral_config:
                test_config[k] = comb[j]
            msg.append(f'{k}:{comb[j]}')
        
        msg = ','.join(msg)
        print(f'test config {msg} ...')
        
        results, model = evaluate_predictor(SpectralDTW ,test_config,loss_weights,splits,seed=seed, epochs=epochs)
        scores.loc[i,'H_mean'] = np.mean(results)
        scores.loc[i,'H_std'] = np.std(results)
        scores.loc[i,'config'] = msg
        scores.to_csv(result_file)

    scores = pd.read_csv(result_file, index_col=0)
    scores = scores.astype({'H_mean':'float'})
    best = scores['H_mean'].idxmax()
    print('Optimal hyperparameters:')
    print(scores.loc[best,'config'])


# In[8]:
# In[4]:

def read_config(config_str):
    config = {}
    for item in config_str.split(','):
        key, val = item.split(':')
        config[key] = val
    return config


search_space = {
    'sigma':[0.1, 1.0, 5, 10],
}


for dataname in ['Synth','ICU', 'ADNI']:
    result_file = f'hyperparam_selection/{dataname}_K.csv'
    scores = pd.read_csv(result_file, index_col=0)
    scores = scores.astype({'H_mean':'float'})
    best = scores['H_mean'].idxmax()
    
    config = read_config(scores.loc[best,'config'])
    
    K = config['K']
    
    hyperparam_selection_predictor(dataname, search_space, K)


# In[ ]:




