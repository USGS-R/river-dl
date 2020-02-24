# -*- coding: utf-8 -*-
"""
@author: jeff sadler, Feb 2020
based off code by Xiaowei Jia

"""

from __future__ import print_function, division
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random


###### build model ######

class rgcn(layers.Layer):

    def __init__(self, hidden_size, pred_out_size, n_phys_vars):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_phys_vars = n_phys_vars

        ### set up the weights ###

        w_initializer = tf.random_normal_initializer(stddev=0.02)
        # was W2
        self.W_out = self.add_weight(shape=[hidden_size, n_classes],
                                  initializer=w_initializer)
        # was b2
        self.b_out = self.add_weight(shape=[n_classes], initializer='zeros')

        # was Wg1
        self.W_graph_h = self.add_weight(shape=[hidden_size, hidden_size],
                                   initializer=w_initializer)
        # was bg1
        self.b_graph_h = self.add_weight(shape=[hidden_size], initializer='zeros')
        # was Wg2
        self.W_graph_c = self.add_weight(shape=[hidden_size, hidden_size],
                                   initializer=w_initializer)
        # was bg2
        self.b_graph_c = self.add_weight(shape=[hidden_size], initializer='zeros')

        # was Wa1
        self.W_h_cur = self.add_weight(shape=[hidden_size, hidden_size],
                                   initializer=w_initializer)
        # was Wa2
        self.W_h_prev = self.add_weight(shape=[hidden_size, hidden_size],
                                   initializer=w_initializer)
        # was ba
        self.b_h = self.add_weight(shape=[hidden_size], initializer='zeros')

        # was Wc1
        self.W_c_cur = self.add_weight(shape=[hidden_size, hidden_size],
                                   initializer=w_initializer)
        # was Wc2
        self.W_c_prev = self.add_weight(shape=[hidden_size, hidden_size],
                                   initializer=w_initializer)
        # was bc
        self.b_c = self.add_weight(shape=[hidden_size], initializer='zeros')

        # was Wp
        self.W_phys = self.add_weight(shape=[hidden_size, phy_size],
                                  initializer=w_initializer)
        # was bp
        self.b_phys = self.add_weight(shape=[phy_size], initializer='zeros')


    def call(self, inputs, A, pretrain=False):
        lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=False,
                                    return_state=True)
        hidden_state_prev, cell_state_prev = lstm(inputs)
        out = []
        for t in range(1, n_steps):
            h_graph = tf.nn.tanh(tf.matmul(A, tf.matmul(hidden_state_prev,
                                                        self.W_graph_h)
                                           + self.b_graph_h))
            c_graph = tf.nn.tanh(tf.matmul(A, tf.matmul(cell_state_prev,
                                                        self.W_graph_c)
                                           + self.b_graph_c))

            hidden_state_cur, cell_state_cur = lstm(inputs[:, t, :],
                                                    initial_state=[hidden_state_prev, cell_state_prev])

            h_update = tf.nn.sigmoid(tf.matmul(hidden_state_cur, self.W_h_cur)
                                     + tf.matmul(h_graph, self.W_h_prev)
                                     + self.b_h)
            c_update = tf.nn.sigmoid(tf.matmul(cell_state_cur, self.W_c_cur)
                                     + tf.matmul(c_graph, self.W_c_prev)
                                     + self.b_c)

            out.append(h_update)
            hidden_state_prev = h_update
            cell_state_prev = c_update

        out = tf.reshape(out, [-1, self.hidden_size])
        if pretrain:
            out_phys = tf.matmul(out, self.W_phys) + self.b_phys
            # out_phys = tf.stack(out_phys, axis=1)
            # out_phys = out_phys[:, :, :self.n_phys_vars]
            # out_phys = tf.reshape(out_phys, [-1])
            return out_phys
        else:
            out_pred = tf.matmul(out, self.W_out) + self.b_out
            # out_pred.append(h_update)
            # out_pred = tf.stack(out_pred, axis=1)
            # out_pred = tf.reshape(out_pred, [-1, self.hidden_size])
            return out_pred

###### Declare constants ######
learning_rate = 0.01
learning_rate_pre = 0.005
epochs = 100
epochs_pre = 200#70
batch_size = 2000
hidden_size = 20 
input_size = 20-2
phy_size = 2
Time_steps = 13149 # total number of time steps
cv_idx = 2 # 0,1,2 for cross validation, 4383 lenth for each one
length = 4383 # number of time steps in testing
npic = 12 # number of years in length
n_steps = int(length/npic) # cut it to 16 pieces #43 #12 #46 # number of time steps to train on
n_classes = 1 
N_sec = (npic-1)*2+1 # 23 - the number of sec (not sure what sec means)
N_seg = 42
kb=1.0

A = 'graph'
