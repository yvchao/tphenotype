K = 16


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

import random
import os, sys

from tensorflow.python.ops.rnn import _transpose_batch_time
from sklearn.model_selection import train_test_split

#performance metrics
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import normalized_mutual_info_score, homogeneity_score, adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix

#user defined
import utils_network as utils
from class_model_v7 import DeepTPC_ICLR

# In[2]:


data_mode = 'CF_comorbidity_select'  #{'CF', 'CF_comorbidity', 'CF_comorbidity_select', 'ADNI'}


# In[3]:


# IMPORT DATASET

if data_mode == 'CF':
    npz = np.load('./data/CF/data.npz')

    data_x        = npz['data_x']
    data_y        = npz['data_y']
    data_y_onehot = npz['data_y_onehot']
    feat_list     = npz['feat_list']
    
elif data_mode == 'CF_comorbidity':
    npz = np.load('./data/CF_comorbidity/data_como.npz')
    
    data_x        = npz['data_x']
    data_y        = npz['data_y']
    feat_list     = npz['feat_list']
    label_list    = npz['label_list']
    selected_list = npz['selected_list']
    
    data_y_selected = data_y[:, :, np.where([f in selected_list for f in label_list])[0]]
    
elif data_mode == 'CF_comorbidity_select':
    npz = np.load('./data/CF_comorbidity/data_como.npz')
    
    data_x        = npz['data_x']
    data_y        = npz['data_y']
    feat_list     = npz['feat_list']
    label_list    = npz['label_list']
    selected_list = npz['selected_list']
    
    data_y        = data_y[:, :, np.where([f in selected_list for f in label_list])[0]]
    label_list    = selected_list
    
    tmp_onehot = np.zeros([np.shape(data_y)[0], np.shape(data_y)[1], 8])

    tmp_onehot[np.sum(data_y == [0,0,0], axis=2) == 3] = [1, 0, 0, 0, 0, 0, 0, 0]
    tmp_onehot[np.sum(data_y == [0,0,1], axis=2) == 3] = [0, 1, 0, 0, 0, 0, 0, 0]
    tmp_onehot[np.sum(data_y == [0,1,0], axis=2) == 3] = [0, 0, 1, 0, 0, 0, 0, 0]
    tmp_onehot[np.sum(data_y == [0,1,1], axis=2) == 3] = [0, 0, 0, 1, 0, 0, 0, 0]
    tmp_onehot[np.sum(data_y == [1,0,0], axis=2) == 3] = [0, 0, 0, 0, 1, 0, 0, 0]
    tmp_onehot[np.sum(data_y == [1,0,1], axis=2) == 3] = [0, 0, 0, 0, 0, 1, 0, 0]
    tmp_onehot[np.sum(data_y == [1,1,0], axis=2) == 3] = [0, 0, 0, 0, 0, 0, 1, 0]
    tmp_onehot[np.sum(data_y == [1,1,1], axis=2) == 3] = [0, 0, 0, 0, 0, 0, 0, 1]

    tmp_onehot[np.sum(np.abs(data_x), axis=2) == 0] = [0, 0, 0, 0, 0, 0, 0, 0] #put all 0's for not selected ones..

    data_y = tmp_onehot

elif data_mode == 'ADNI':
    npz = np.load('./data/ADNI/data.npz')

    data_x        = npz['data_x']
    data_y        = npz['data_y']
    feat_list     = npz['feat_list']
    label_list    = npz['label_list']
# In[4]:


### PARAMETER LOGGING
def save_logging(dictionary, log_name):
    with open(log_name, 'w') as f:
        for key, value in dictionary.items():
            if 'activate_fn' in key:
                value = str(value).split(' ')[1]
                
            f.write('%s:%s\n' % (key, value))


def load_logging(filename):
    data = dict()
    with open(filename) as f:
        def is_float(input):
            try:
                num = float(input)
            except ValueError:
                return False
            return True

        for line in f.readlines():
            if ':' in line:
                key,value = line.strip().split(':', 1)
                
                if 'activate_fn' in key:
                    if value == 'relu':
                        value = tf.nn.relu
                    elif value == 'elu':
                        value = tf.nn.elu
                    elif value == 'tanh':
                        value = tf.nn.tanh
                    else:
                        raise ValueError('ERROR: wrong choice of activation function!')
                    data[key] = value
                else:
                    if value.isdigit():
                        data[key] = int(value)
                    elif is_float(value):
                        data[key] = float(value)
                    elif value == 'None':
                        data[key] = None
                    else:
                        data[key] = value
            else:
                pass # deal with bad lines of text here    
    return data


# In[5]:


def log(x): 
    return tf.log(x + 1e-8)

def div(x, y):
    return tf.div(x, (y + 1e-8))

def get_seq_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    tmp_length = tf.reduce_sum(used, 1)
    tmp_length = tf.cast(tmp_length, tf.int32)
    return tmp_length


def f_get_minibatch(mb_size, x, y):
    idx = range(np.shape(x)[0])
    idx = random.sample(idx, mb_size)

    x_mb   = x[idx].astype(float)    
    y_mb   = y[idx].astype(float)    

    return x_mb, y_mb


# In[6]:


### PERFORMANCE METRICS:
def f_get_prediction_scores(y_true_, y_pred_):
    if np.sum(y_true_) == 0: #no label for running roc_auc_curves
        auroc_ = -1.
        auprc_ = -1.
    else:
        auroc_ = roc_auc_score(y_true_, y_pred_)
        auprc_ = average_precision_score(y_true_, y_pred_)
    return (auroc_, auprc_)


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    c_matrix = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(c_matrix, axis=0)) / np.sum(c_matrix)






# In[9]:


h_dim_FC   = 50 #for fully_connected layers
h_dim_RNN  = 50

x_dim = np.shape(data_x)[2]
y_dim = np.shape(data_y)[2]

if data_mode == 'CF':
    y_type = 'categorical'
elif data_mode == 'CF_comorbidity':
    y_type = 'binary'
elif data_mode == 'CF_comorbidity_select':
    y_type = 'categorical'
elif data_mode == 'ADNI':
    y_type = 'categorical'
    
    
num_layer_encoder    = 1
num_layer_selector   = 2
num_layer_predictor  = 2

z_dim = h_dim_RNN * num_layer_encoder

max_length = np.shape(data_x)[1]

rnn_type          = 'LSTM' #GRU, LSTM


# In[10]:


input_dims ={
    'x_dim': x_dim,
    'y_dim': y_dim,
    'y_type': y_type,
    'max_cluster': K,
    'max_length': max_length    
}

network_settings ={
    'h_dim_encoder': h_dim_RNN,
    'num_layers_encoder': num_layer_encoder,
    'rnn_type': rnn_type,
    'rnn_activate_fn': tf.nn.tanh,

    'h_dim_selector': h_dim_FC,
    'num_layers_selector': num_layer_selector,
    
    'h_dim_predictor': h_dim_FC,
    'num_layers_predictor': num_layer_predictor,
    
    'fc_activate_fn': tf.nn.relu
}


# In[11]:


OUT_ITERATION = 5

RESULT_NMI    = np.zeros([OUT_ITERATION, 1])
RESULT_RI     = np.zeros([OUT_ITERATION, 1])
RESULT_PURITY = np.zeros([OUT_ITERATION, 1])


# In[12]:


tf.reset_default_graph()

# Turn on xla optimization
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

model = DeepTPC_ICLR(sess, "Deep_TPC", input_dims, network_settings)


seed = 1234
tf.set_random_seed(0)
random.seed(0)
np.random.seed(0)

lr_rate1   = 0.001
keep_prob  = 0.7
mb_size    = 128

ITERATION  = 30000
check_step = 1000

for out_itr in [0,1,2,3,4]:
# for out_itr in [0]:
    print("========== OUt ITERATION: {} ========== ".format(out_itr))

    tr_data_x,te_data_x, tr_data_y,te_data_y = train_test_split(
        data_x, data_y, test_size=0.2, random_state=seed+out_itr
    )

    tr_data_x,va_data_x, tr_data_y,va_data_y = train_test_split(
        tr_data_x, tr_data_y, test_size=0.2, random_state=seed+out_itr
    )


    save_path = './{}/proposed/init/itr{}/'.format(data_mode, out_itr)

    if not os.path.exists(save_path + '/models/'):
        os.makedirs(save_path + '/models/')

    if not os.path.exists(save_path + '/results/'):
        os.makedirs(save_path + '/results/')



    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer(), feed_dict={model.E:np.zeros([K, z_dim]).astype(float)})

    avg_loss2  = 0
    for itr in range(ITERATION):
        x_mb, y_mb = f_get_minibatch(mb_size, tr_data_x, tr_data_y)
        
        _, tmp_loss2 = model.train_mle(x_mb, y_mb, lr_rate1, keep_prob)
        avg_loss2 += tmp_loss2/check_step
        
        if (itr+1)%check_step == 0:                
            tmp_y, tmp_m = model.predict_y_hats(va_data_x)

            y_pred = tmp_y.reshape([-1, y_dim])[tmp_m.reshape([-1]) == 1]
            y_true = va_data_y.reshape([-1, y_dim])[tmp_m.reshape([-1]) == 1]
            
            AUROC = np.zeros([y_dim])
            AUPRC = np.zeros([y_dim])
            for y_idx in range(y_dim):
                auroc, auprc = f_get_prediction_scores(y_true[:, y_idx], y_pred[:, y_idx])
                AUROC[y_idx] = auroc
                AUPRC[y_idx] = auprc
                
            print ("ITR {:05d}: loss_2={:.3f} | va_auroc:{:.3f}, va_auprc:{:.3f}".format(
                    itr+1, avg_loss2, np.mean(AUROC), np.mean(AUPRC))
                  )        
            avg_loss2 = 0
            
    saver.save(sess, save_path + 'models/model_v7_K{}'.format(K))
    save_logging(network_settings, save_path + 'models/network_settings_v7_K{}.txt'.format(K))

    tmp_y, tmp_m = model.predict_y_hats(te_data_x)

    y_pred = tmp_y.reshape([-1, y_dim])[tmp_m.reshape([-1]) == 1]
    y_true = te_data_y.reshape([-1, y_dim])[tmp_m.reshape([-1]) == 1]

    AUROC = np.zeros([y_dim])
    AUPRC = np.zeros([y_dim])
    for y_idx in range(y_dim):
        auroc, auprc = f_get_prediction_scores(y_true[:, y_idx], y_pred[:, y_idx])
        AUROC[y_idx] = auroc
        AUPRC[y_idx] = auprc

    print(AUROC)
    print(AUPRC)


# In[18]:


# f_idx = 4

# print(label_list[f_idx])

# plt.hist(y_pred[:, f_idx])
# plt.show()
# plt.close
