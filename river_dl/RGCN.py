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
    def __init__(self, hidden_size, A, flow_in_temp=False, rand_seed=None):
        """

        :param hidden_size: [int] the number of hidden units
        :param n_phys_vars: [int] the number of outputs to produce in
        pre-training
        :param A: [numpy array] adjacency matrix
        :param flow_in_temp: [bool] whether the flow predictions should feed
        into the temp predictions
        :param rand_seed: [int] the random seed for initialization
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.A = A.astype('float32')
        self.flow_in_temp = flow_in_temp

        # set up the layer
        self.lstm = tf.keras.layers.LSTMCell(hidden_size)

        ### set up the weights ###
        w_initializer = tf.random_normal_initializer(stddev=0.02,
                                                     seed=rand_seed)

        # was Wg2
        self.W_graph_c = self.add_weight(shape=[hidden_size, hidden_size],
                                         initializer=w_initializer,
                                         name='W_graph_c')
        # was bg2
        self.b_graph_c = self.add_weight(shape=[hidden_size],
                                         initializer='zeros', name='b_graph_c')


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

        if self.flow_in_temp:
            # was W2
            self.W_out_flow = self.add_weight(shape=[hidden_size, 1],
                                              initializer=w_initializer,
                                              name='W_out')
            # was b2
            self.b_out_flow = self.add_weight(shape=[1], initializer='zeros',
                                              name='b_out')

            self.W_out_temp = self.add_weight(shape=[hidden_size+1, 1],
                                              initializer=w_initializer,
                                              name='W_out')

            self.b_out_temp = self.add_weight(shape=[1], initializer='zeros',
                                              name='b_out')
        else:
            # was W2
            self.W_out = self.add_weight(shape=[hidden_size, 2],
                                         initializer=w_initializer,
                                         name='W_out')
            # was b2
            self.b_out = self.add_weight(shape=[2],
                                         initializer='zeros', name='b_out')

    @tf.function
    def call(self, inputs, **kwargs):
        graph_size = self.A.shape[0]
        hidden_state, cell_state_prev = (tf.zeros([graph_size,
                                                        self.hidden_size]),
                                              tf.zeros([graph_size,
                                                        self.hidden_size]))
        out = []
        n_steps = inputs.shape[1]
        for t in range(n_steps):
            c_graph = tf.nn.tanh(tf.matmul(self.A, tf.matmul(cell_state_prev,
                                                             self.W_graph_c)
                                           + self.b_graph_c))

            seq, state = self.lstm(inputs[:, t, :], states=[hidden_state,
                                                            cell_state_prev])
            hidden_state, cell_state_cur = state

            c_update = tf.nn.sigmoid(tf.matmul(cell_state_cur, self.W_c_cur)
                                     + tf.matmul(c_graph, self.W_c_prev)
                                     + self.b_c)

            if self.flow_in_temp:
                out_pred_q = tf.matmul(hidden_state, self.W_out_flow) +\
                             self.b_out_flow
                out_pred_t = tf.matmul(tf.concat([hidden_state, out_pred_q],
                                                 axis=1),
                                       self.W_out_temp) + self.b_out_temp
                out_pred = tf.concat([out_pred_t, out_pred_q], axis=1)
            else:
                out_pred = tf.matmul(hidden_state, self.W_out) + self.b_out

            out.append(out_pred)

            cell_state_prev = c_update
        out = tf.stack(out)
        out = tf.transpose(out, [1, 0, 2])
        return out


class RGCNModel(tf.keras.Model):
    def __init__(self, hidden_size, A, flow_in_temp=False, rand_seed=None):
        """
        :param hidden_size: [int] the number of hidden units
        :param n_phys_vars: [int] the number of outputs to produce in
        pre-training
        :param A: [numpy array] adjacency matrix
        :param flow_in_temp: [bool] whether the flow predictions should feed
        into the temp predictions
        :param rand_seed: [int] the random seed for initialization
        """
        super().__init__()
        self.rgcn_layer = RGCN(hidden_size, A, flow_in_temp, rand_seed)

    def call(self, inputs, **kwargs):
        output = self.rgcn_layer(inputs)
        return output


def rmse_masked_one_var(y_true, y_pred, weights):
    num_y_true = tf.cast(tf.math.count_nonzero(~tf.math.is_nan(y_true)),
                         tf.float32)
    zero_or_error = tf.where(tf.math.is_nan(y_true),
                             tf.zeros_like(y_true),
                             y_pred - y_true)
    wgt_zero_or_err = (zero_or_error * weights)
    sum_squared_errors = tf.reduce_sum(tf.square(wgt_zero_or_err))
    rmse_loss = tf.sqrt(sum_squared_errors / num_y_true)
    return rmse_loss


def weighted_masked_rmse(temperature_weight=0.5):
    """
    calculate a weighted, masked rmse. 
    :param temperature_weight: [float] weight between 0 and 1. The difference
    between one and the temperature_weight becomes the flow_weight. If you want
    to weight the rmse of temperature and the rmse of flow equally,
    the temperature_weight would be 0.5
    """

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

        # make all zero-weighted observations 'nan' so they don't get counted
        # at all in the loss calculation
        y_true = tf.where(weights == 0, np.nan, y_true)

        weights_temp = weights[:, :, 0]
        weights_flow = weights[:, :, 1]
        true_temp = y_true[:, :, 0]
        true_flow = y_true[:, :, 1]
        pred_temp = y_pred[:, :, 0]
        pred_flow = y_pred[:, :, 1]
        rmse_temp = rmse_masked_one_var(true_temp, pred_temp, weights_temp)
        rmse_flow = rmse_masked_one_var(true_flow, pred_flow, weights_flow)

        rmse_loss = rmse_temp * temperature_weight +\
                    rmse_temp * (1 - temperature_weight)

        return rmse_loss

    return rmse_masked


