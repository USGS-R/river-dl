# -*- coding: utf-8 -*-
"""
@author: jeff sadler, Feb 2020
based off code by Xiaowei Jia

"""

from __future__ import print_function, division
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import datetime
import random
from load_data import read_process_data, process_adj_matrix
from tensorflow.keras.losses import MSE


###### build model ######

class rgcn(layers.Layer):
    def __init__(self, hidden_size, pred_out_size, n_phys_vars, A,
                 pretrain=False):
        """

        :param hidden_size: [int] the number of hidden units
        :param pred_out_size: [int] the number of outputs to produce in
        fine-tuning
        :param n_phys_vars: [int] the number of outputs to produce in
        pre-training
        :param A: [numpy array] adjacency matrix
        :param pretrain: [bool] whether you are pretraining or not
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.n_phys_vars = n_phys_vars
        self.A = A
        self.pretrain = pretrain

        ### set up the weights ###

        # todo: does this make it so all of the weights are initialized to the same values?
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
        self.lstm = tf.keras.layers.LSTMCell(hidden_size)

        if self.pretrain:
            # was Wp
            self.W_phys = self.add_weight(shape=[hidden_size, n_phys_vars],
                                          initializer=w_initializer,
                                          name='W_phys')
            # was bp
            self.b_phys = self.add_weight(shape=[n_phys_vars],
                                          initializer='zeros',
                                          name='b_phys')
        else:
            # was W2
            self.W_out = self.add_weight(shape=[hidden_size, pred_out_size],
                                         initializer=w_initializer,
                                         name='W_out')
            # was b2
            self.b_out = self.add_weight(shape=[n_classes], initializer='zeros',
                                         name='b_out')

    @tf.function
    def call(self, inputs, **kwargs):
        # todo: read in the dimensions from A for first dim
        hidden_state_prev, cell_state_prev = (tf.zeros([42, self.hidden_size]),
                                              tf.zeros([42, self.hidden_size]))
        out = []
        n_steps = inputs.shape[1]
        A = self.A.astype('float32')
        for t in range(n_steps):
            h_graph = tf.nn.tanh(tf.matmul(A, tf.matmul(hidden_state_prev,
                                                        self.W_graph_h)
                                           + self.b_graph_h))
            c_graph = tf.nn.tanh(tf.matmul(A, tf.matmul(cell_state_prev,
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

            if self.pretrain:
                out_phys = tf.matmul(h_update, self.W_phys) + self.b_phys
                out.append(out_phys)
            else:
                out_pred = tf.matmul(h_update, self.W_out) + self.b_out
                out.append(out_pred)

            hidden_state_prev = h_update
            cell_state_prev = c_update
        out = tf.stack(out)
        out = tf.transpose(out, [1, 0, 2])
        return out


class rgcn_model(tf.keras.Model):
    def __init__(self, hidden_size, pred_out_size, n_phys_vars, A,
                 pretrain=False):
        """
        :param hidden_size: [int] the number of hidden units
        :param pred_out_size: [int] the number of outputs to produce in
        fine-tuning
        :param n_phys_vars: [int] the number of outputs to produce in
        pre-training
        :param A: [numpy array] adjacency matrix
        :param pretrain: [bool] whether you are pretraining or not
        """
        super().__init__()
        self.rgcn_layer = rgcn(hidden_size, pred_out_size, n_phys_vars, A,
                               pretrain)

    def call(self, inputs, **kwargs):
        output = self.rgcn_layer(inputs)
        return output

###### Declare constants ######
learning_rate = 0.01
learning_rate_pre = 0.005
epochs = 100
epochs_pre = 200  # 70
batch_size = 365  # days
batch_offset = 0.5 # for the batches, offset half the year
hidden_size = 20
input_size = 20 - 2

# number of physical variables
n_phys_vars = 2
# the number of river segments
n_seg = 42
n_total_sample = 13149
n_years = int(n_total_sample)/365
# the number of sections to divide the total samples in
cv_divisions = 3
# the cross-validation index
cv_idx = 2
# number of samples in each division
n_samp_per_div = n_total_sample / cv_divisions
n_trn_yrs = int(2 * n_years/cv_divisions) # twice as much training as testing
# number of time steps per batch or sequence that the model will be trained on
# one year of daily data
n_step = 365
n_batch = (n_years * 2) - 1

n_classes = 1

kb = 1.0

A = process_adj_matrix()

data = read_process_data(trn_ratio=0.67, batch_offset=1)
# iterate over epochs
model = rgcn_model(hidden_size, 1, 2, A=A, pretrain=True)

epochs = 3
optimizer = tf.optimizers.Adam(learning_rate=learning_rate_pre)

tf.random.set_seed(23)
x_trn = data['x_trn']
model(x_trn[0, :, :, :])
n_batch, n_seg, n_day, n_feat = x_trn.shape
x_trn = np.reshape(x_trn, [n_batch * n_seg, n_day, n_feat])
y_trn = data['y_trn']
y_trn = np.reshape(y_trn, [n_batch * n_seg, n_day, 2])
model.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
model.fit(x=x_trn, y=y_trn, epochs=epochs, batch_size=42)
