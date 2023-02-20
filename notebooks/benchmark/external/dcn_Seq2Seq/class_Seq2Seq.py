_EPSILON = 1e-08

import numpy as np
import pandas as pd

import os, sys
import random
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected as FC_Net
from tensorflow.python.ops.rnn import _transpose_batch_time

from sklearn.model_selection import train_test_split

import utils_network as utils

##### USER-DEFINED FUNCTIONS


def log(x):
    return tf.log(x + 1e-8)


def div(x, y):
    return tf.div(x, (y + 1e-8))


def get_seq_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    tmp_length = tf.reduce_sum(used, 1)
    tmp_length = tf.cast(tmp_length, tf.int32)
    return tmp_length


class DCN_Seq2Seq:

    def __init__(self, sess, name, input_dims, network_settings):
        self.sess = sess
        self.name = name

        # INPUT/OUTPUT DIMENSIONS
        self.x_dim = input_dims['x_dim']    #features + delta
        self.y_dim = input_dims['y_dim']
        self.max_length = input_dims['max_length']

        # Encoder
        self.h_dim_f = network_settings['h_dim_encoder']    #encoder/decoder nodes
        self.num_layers_f = network_settings['num_layers_encoder']    #encoder/decoder layers
        self.rnn_type = network_settings['rnn_type']
        self.rnn_activate_fn = network_settings['rnn_activate_fn']

        # Predictor (predicting the next input/label values)
        self.h_dim_g = network_settings['h_dim_predictor']    #predictor nodes
        self.num_layers_g = network_settings['num_layers_predictor']    #predictor layers

        self.fc_activate_fn = network_settings['fc_activate_fn']    #selector & predictor

        # Latent Space
        self.z_dim = self.h_dim_f * self.num_layers_f

        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            #### PLACEHOLDER DECLARATION
            self.mb_size = tf.placeholder(tf.int32, [], name='batch_size')

            self.lr_rate = tf.placeholder(tf.float32)
            self.keep_prob = tf.placeholder(tf.float32)    #keeping rate

            self.mb_size = tf.placeholder(tf.int32, [], name='batch_size')
            self.lr_rate = tf.placeholder(tf.float32, name='learning_rate')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_probability')

            self.x = tf.placeholder(tf.float32, [None, self.max_length, self.x_dim], name='inputs')
            self.y_onehot = tf.placeholder(tf.float32, [None, self.max_length, self.y_dim], name='labels_onehot')

            #FOR CLUSTERING
            self.E = tf.placeholder(tf.float32, [None, self.z_dim], name='embeddings')    #mu
            self.K = tf.placeholder(tf.int32, [], name='num_Cluster')
            self.s = tf.placeholder(tf.int32, [None], name='s')
            s_one_hot = tf.one_hot(self.s, self.K, name='s_one_hot')

            # LOSS PARAMETERS
            self.a = tf.placeholder(tf.float32, name='alpha')
            self.a_delta = tf.placeholder(tf.float32, name='alpha_delta')

            inputs = self.x    #delta + feature
            #             inputs = tf.concat([self.x,self.y_onehot], axis=2, name='inputs') #delta + feature + label
            '''
                ##### CREATE MASK
                    - rnn_mask_age, rnn_mask_static, rnn_mask_delta, rnn_mask_timevarying
            '''
            # CREATE RNN MASK:
            seq_length = get_seq_length(self.x)
            tmp_range = tf.expand_dims(tf.range(0, self.max_length, 1), axis=0)
            self.rnn_mask1 = tf.cast(tf.equal(tmp_range, tf.expand_dims(seq_length - 1, axis=1)),
                                     tf.float32)    #last observation
            self.rnn_mask2 = tf.cast(tf.less_equal(tmp_range, tf.expand_dims(seq_length - 1, axis=1)),
                                     tf.float32)    #all available observation

            # Only consider last available observation
            self.rnn_mask2 = self.rnn_mask1

            enc_cell = utils.create_rnn_cell(
                self.h_dim_f, self.num_layers_f, self.keep_prob, self.rnn_type, activation_fn=self.rnn_activate_fn)
            dec_cell = utils.create_rnn_cell(
                self.h_dim_f, self.num_layers_f, self.keep_prob, self.rnn_type, activation_fn=self.rnn_activate_fn)

            with tf.variable_scope('Encoder'):
                initial_state = enc_cell.zero_state(self.mb_size, tf.float32)

                encoder_outputs, last_enc_state = tf.nn.dynamic_rnn(
                    enc_cell, inputs=inputs, initial_state=initial_state, dtype=tf.float32)

            with tf.variable_scope('Decoder'):
                dummy_zero_input = tf.zeros(
                    shape=[self.mb_size, self.x_dim + self.y_dim], dtype=tf.float32, name='dummy_zero_input')

                output_ta = tf.TensorArray(size=self.max_length, dtype=tf.float32, clear_after_read=False)

                #                 output_ta = tf.TensorArray(size=self.mb_size, dtype=tf.float32)

                def get_logits(tmp_h, reuse=tf.AUTO_REUSE):
                    with tf.variable_scope('Logit_Function', reuse=reuse):
                        for e in range(self.num_layers_g):
                            if e == 0:
                                h_delta = tmp_h
                            h_delta = tf.contrib.layers.fully_connected(
                                inputs=h_delta, num_outputs=self.h_dim_g, activation_fn=self.fc_activate_fn)
                        h_delta = tf.contrib.layers.fully_connected(
                            inputs=h_delta, num_outputs=1, activation_fn=tf.nn.softplus)

                        for e in range(self.num_layers_g):
                            if e == 0:
                                h_x = tmp_h
                            h_x = tf.contrib.layers.fully_connected(
                                inputs=h_x, num_outputs=self.h_dim_g, activation_fn=self.fc_activate_fn)
                        h_x = tf.contrib.layers.fully_connected(
                            inputs=h_x, num_outputs=self.x_dim - 1, activation_fn=None)

                        for e in range(self.num_layers_g):
                            if e == 0:
                                h_y = tmp_h
                            h_y = tf.contrib.layers.fully_connected(
                                inputs=h_y, num_outputs=self.h_dim_g, activation_fn=self.fc_activate_fn)
                        h_y = tf.contrib.layers.fully_connected(
                            inputs=h_y, num_outputs=self.y_dim, activation_fn=tf.nn.softmax)

                        #                     return tf.concat([h_delta, h_x, h_y], axis=1)
                        return h_delta, h_x, h_y


#                 def get_sample(tmp_output):
#                     for e in range(self.num_layers_g):
#                         if e == 0:
#                             h = tmp_output
#                         h = tf.contrib.layers.fully_connected(inputs=h, num_outputs=self.h_dim_g, activation_fn=self.fc_activate_fn)

#                     h_delta    = tf.contrib.layers.fully_connected(inputs=h, num_outputs=1, activation_fn=tf.nn.softplus)
#                     h_x        = tf.contrib.layers.fully_connected(inputs=h, num_outputs=self.x_dim-1, activation_fn=None)
#                     h_y        = tf.contrib.layers.fully_connected(inputs=h, num_outputs=self.y_dim, activation_fn=tf.nn.softmax)

#                     dist_y     = tf.contrib.distributions.Categorical(probs=h_y, dtype=tf.int32)
#                     sample_y   = dist_y.sample()

#                     y_sampled  = tf.one_hot(sample_y, self.y_dim)

#                     return tf.concat([h_delta, h_x, y_sampled], axis=1)

                def loop_fn(time, cell_output, cell_state, loop_state):
                    emit_output = cell_output    # == None for time == 0

                    if cell_output is None:
                        next_cell_state = last_enc_state
                        next_sampled_input = dummy_zero_input
                        next_loop_state = output_ta
                    else:    # pass the last state to the next
                        next_cell_state = cell_state

                        tmp_z = utils.create_concat_state_h(next_cell_state, self.num_layers_f, self.rnn_type)

                        tmp_d, tmp_x, tmp_y_logits = get_logits(tmp_z)
                        dist_y = tf.contrib.distributions.Categorical(probs=tmp_y_logits, dtype=tf.int32)
                        sample_y = dist_y.sample()
                        tmp_y = tf.one_hot(sample_y, self.y_dim)

                        next_sampled_input = tf.concat([tmp_d, tmp_x, tmp_y], axis=1)

                        #                         next_sampled_input = get_sample(cell_output)  # sampling from multinomial
                        next_loop_state = loop_state.write(time - 1, tf.concat([tmp_d, tmp_x, tmp_y_logits], axis=1))

                    elements_finished = (time >= self.max_length)
                    #this gives the break-point (no more recurrence after the max_length)
                    finished = tf.reduce_all(elements_finished)
                    next_input = tf.cond(finished,
                                         lambda: tf.zeros([self.mb_size, self.x_dim + self.y_dim], dtype=tf.float32),
                                         lambda: next_sampled_input)

                    #                     next_input = next_sampled_input
                    #                     next_input = FC_Net(inputs=next_sampled_input, num_outputs=self.h_dim2, activation_fn=tf.nn.relu)

                    return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)

                decoder_emit_ta, _, loop_state_ta = tf.nn.raw_rnn(dec_cell, loop_fn)

            self.Z = utils.create_concat_state_h(last_enc_state, self.num_layers_f, self.rnn_type)

            outputs = _transpose_batch_time(loop_state_ta.stack())

            self.outputs = outputs

            self.outputs_delta = tf.reshape(outputs[:, :, 0], [-1, self.max_length, 1])
            self.inputs_delta = tf.reshape(inputs[:, :, 0], [-1, self.max_length, 1])

            self.outputs_x = tf.reshape(outputs[:, :, 1:self.x_dim], [-1, self.max_length, (self.x_dim - 1)])
            self.inputs_x = tf.reshape(inputs[:, :, 1:self.x_dim], [-1, self.max_length, (self.x_dim - 1)])

            self.outputs_y = tf.reshape(outputs[:, :, self.x_dim:], [-1, self.max_length, self.y_dim])
            #             self.inputs_y       = tf.reshape(inputs[:, :, self.x_dim:], [-1, self.max_length, self.y_dim])

            ### RECONSTRUCTION LOSS

            loss_delta = tf.reduce_mean(
                tf.reduce_sum(
                    self.rnn_mask2 * tf.reduce_sum((self.inputs_delta - self.outputs_delta)**2, axis=2), axis=1))
            loss_x = tf.reduce_mean(
                tf.reduce_sum(self.rnn_mask2 * tf.reduce_sum((self.inputs_x - self.outputs_x)**2, axis=2), axis=1))
            loss_y = tf.reduce_mean(
                tf.reduce_sum(self.rnn_mask2 * tf.reduce_sum(-self.y_onehot * log(self.outputs_y), axis=2), axis=1))

            self.LOSS_AE = self.a_delta * loss_delta + loss_x + loss_y

            ### CLUSTER LOSS
            Z_expanded = tf.tile(tf.expand_dims(self.Z, axis=1), [1, self.K, 1])    #[None, num_Cluster, 2]
            MU_expanded = tf.tile(tf.expand_dims(self.E, axis=0), [self.mb_size, 1, 1])    #[None, num_Cluster, 2]
            dist_z_expanded = tf.reduce_sum((Z_expanded - MU_expanded)**2, axis=2)    #[None, num_Cluster]

            dist_z_homo = tf.reduce_sum(dist_z_expanded * s_one_hot, axis=1)    #[None]
            dist_z_hetero = tf.reduce_sum(dist_z_expanded * (1. - s_one_hot), axis=1)    #[None]

            dist_z_homo = tf.reduce_mean(dist_z_homo)
            dist_z_hetero = tf.reduce_mean(dist_z_hetero)

            self.LOSS_CLU = dist_z_homo

            self.LOSS_TOTAL = self.LOSS_AE + self.a * self.LOSS_CLU

            self.solver_AE = tf.train.AdamOptimizer(self.lr_rate).minimize(self.LOSS_AE)
            self.solver_TOTAL = tf.train.AdamOptimizer(self.lr_rate).minimize(self.LOSS_TOTAL)

    def train_ae(self, x_, y_onehot_, a_delta, lr_train, k_prob):
        return self.sess.run(
            [self.solver_AE, self.LOSS_AE],
            feed_dict={
                self.x: x_,
                self.y_onehot: y_onehot_,
                self.a_delta: a_delta,
                self.mb_size: np.shape(x_)[0],
                self.lr_rate: lr_train,
                self.keep_prob: k_prob
            })

    def train_total(self, x_, y_onehot_, s_, E_, K_, a_delta, a, lr_train, k_prob):
        return self.sess.run(
            [self.solver_TOTAL, self.LOSS_TOTAL, self.LOSS_AE, self.LOSS_CLU],
            feed_dict={
                self.x: x_,
                self.y_onehot: y_onehot_,
                self.s: s_,
                self.E: E_,
                self.K: K_,
                self.mb_size: np.shape(x_)[0],
                self.a_delta: a_delta,
                self.a: a,
                self.lr_rate: lr_train,
                self.keep_prob: k_prob
            })

    def predict_Z(self, x_):
        return self.sess.run(self.Z, feed_dict={self.x: x_, self.mb_size: np.shape(x_)[0], self.keep_prob: 1.0})

    def predict_outputs(self, x_):
        return self.sess.run([self.outputs_delta, self.outputs_x, self.outputs_y, self.rnn_mask2],
                             feed_dict={
                                 self.x: x_,
                                 self.mb_size: np.shape(x_)[0],
                                 self.keep_prob: 1.0
                             })
