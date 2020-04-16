# -*- coding: utf-8 -*-
"""
@author: jeff sadler, Feb 2020
based off code by Xiaowei Jia

"""

from __future__ import print_function, division
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class RGCN(layers.Layer):
    def __init__(self, hidden_size, pred_out_size, A):
        """

        :param hidden_size: [int] the number of hidden units
        :param pred_out_size: [int] the number of outputs to produce in
        fine-tuning
        :param n_phys_vars: [int] the number of outputs to produce in
        pre-training
        :param A: [numpy array] adjacency matrix
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.A = A.astype('float32')

        # set up the layer
        self.lstm = tf.keras.layers.LSTMCell(hidden_size)

        ### set up the weights ###
        w_initializer = tf.random_normal_initializer(stddev=0.02)

        # was Wg1
        self.W_graph_h = self.add_weight(shape=[hidden_size, hidden_size],
                                         initializer=w_initializer,
                                         name='W_graph_h')
        # was bg1
        self.b_graph_h = self.add_weight(shape=[hidden_size],
                                         initializer='zeros', name='b_graph_h')
        # was Wg2
        self.W_graph_c = self.add_weight(shape=[hidden_size, hidden_size],
                                         initializer=w_initializer,
                                         name='W_graph_c')
        # was bg2
        self.b_graph_c = self.add_weight(shape=[hidden_size],
                                         initializer='zeros', name='b_graph_c')

        # was Wa1
        self.W_h_cur = self.add_weight(shape=[hidden_size, hidden_size],
                                       initializer=w_initializer,
                                       name='W_h_cur')
        # was Wa2
        self.W_h_prev = self.add_weight(shape=[hidden_size, hidden_size],
                                        initializer=w_initializer,
                                        name='W_h_prev')
        # was ba
        self.b_h = self.add_weight(shape=[hidden_size], initializer='zeros',
                                   name='b_h')

        # was Wc1
        self.W_c_cur = self.add_weight(shape=[hidden_size, hidden_size],
                                       initializer=w_initializer,
                                       name='W_c_cur')
        # was Wc2
        self.W_c_prev = self.add_weight(shape=[hidden_size, hidden_size],
                                        initializer=w_initializer,
                                        name='W_c_prev')
        # was bc
        self.b_c = self.add_weight(shape=[hidden_size], initializer='zeros',
                                   name='b_c')

        # was W2
        self.W_out = self.add_weight(shape=[hidden_size, pred_out_size],
                                     initializer=w_initializer,
                                     name='W_out')
        # was b2
        self.b_out = self.add_weight(shape=[pred_out_size], initializer='zeros',
                                     name='b_out')

    @tf.function
    def call(self, inputs, **kwargs):
        graph_size = self.A.shape[0]
        hidden_state_prev, cell_state_prev = (tf.zeros([graph_size,
                                                        self.hidden_size]),
                                              tf.zeros([graph_size,
                                                        self.hidden_size]))
        out = []
        n_steps = inputs.shape[1]
        for t in range(n_steps):
            h_graph = tf.nn.tanh(tf.matmul(self.A, tf.matmul(hidden_state_prev,
                                                             self.W_graph_h)
                                           + self.b_graph_h))
            c_graph = tf.nn.tanh(tf.matmul(self.A, tf.matmul(cell_state_prev,
                                                             self.W_graph_c)
                                           + self.b_graph_c))

            seq, state = self.lstm(inputs[:, t, :], states=[hidden_state_prev,
                                                            cell_state_prev])
            hidden_state_cur, cell_state_cur = state

            h_update = tf.nn.sigmoid(tf.matmul(hidden_state_cur, self.W_h_cur)
                                     + tf.matmul(h_graph, self.W_h_prev)
                                     + self.b_h)
            c_update = tf.nn.sigmoid(tf.matmul(cell_state_cur, self.W_c_cur)
                                     + tf.matmul(c_graph, self.W_c_prev)
                                     + self.b_c)

            # out_pred = tf.matmul(h_update, self.W_out) + self.b_out
            # dimensions = 1
            out_pred_q = tf.matmul(h_update, self.W_out) + self.b_out
            out_pred_t = tf.matmul(concat(h_update, out_pred_q), self.W_out2) + self.b_out2
            out_pred = concat(out_pred_q, out_pred_t)
            out.append(out_pred)

            hidden_state_prev = h_update
            cell_state_prev = c_update
        out = tf.stack(out)
        out = tf.transpose(out, [1, 0, 2])
        return out


class RGCNModel(tf.keras.Model):
    def __init__(self, hidden_size, pred_out_size, A):
        """
        :param hidden_size: [int] the number of hidden units
        :param pred_out_size: [int] the number of outputs to produce in
        fine-tuning
        :param n_phys_vars: [int] the number of outputs to produce in
        pre-training
        :param A: [numpy array] adjacency matrix
        """
        super().__init__()
        self.rgcn_layer = RGCN(hidden_size, pred_out_size, A)

    def call(self, inputs, **kwargs):
        output = self.rgcn_layer(inputs)
        return output


def rmse_masked(data, y_pred):
    """
    Compute cost as RMSE with masking (the tf.where call replaces pred_s-y_s
    with 0 when y_s is nan; num_y_s is a count of just those non-nan
    observations) so we're only looking at predictions with corresponding
    observations available
    (credit: @aappling-usgs)
    :param data: [tensor] true (observed) y values. these may have nans and 
    sample weights
    :param y_pred: [tensor] predicted y values
    :return: rmse (one value for each training sample)
    """
    weights = data[:, :, -2:]
    y_true = data[:, :, :-2]

    # ensure y_pred, weights, and y_true are all tensors the same data type
    y_true = tf.convert_to_tensor(y_true)
    weights = tf.convert_to_tensor(weights)
    y_true = tf.cast(y_true, y_pred.dtype)
    weights = tf.cast(weights, y_pred.dtype)

    # make all zero-weighted observations 'nan' so they don't get counted at all
    # in the loss calculation
    y_true = tf.where(weights == 0, np.nan, y_true)

    # count the number of non-nans
    num_y_true = tf.cast(tf.math.count_nonzero(~tf.math.is_nan(y_true)),
                         tf.float32)
    zero_or_error = tf.where(tf.math.is_nan(y_true),
                             tf.zeros_like(y_true),
                             y_pred - y_true)
    wgt_zero_or_err = (zero_or_error * weights)
    sum_squared_errors = tf.reduce_sum(tf.square(wgt_zero_or_err))
    rmse_loss = tf.sqrt(sum_squared_errors / num_y_true)
    return rmse_loss


