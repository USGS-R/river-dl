# -*- coding: utf-8 -*-
"""
@author: jeff sadler, Feb 2020
based off code by Xiaowei Jia

"""

from __future__ import print_function, division
import tensorflow as tf
from tensorflow.keras import layers


class RGCN(layers.Layer):
    def __init__(self, hidden_size, A, flow_in_temp=False, rand_seed=None):
        """

        :param hidden_size: [int] the number of hidden units
        :param A: [numpy array] adjacency matrix
        :param flow_in_temp: [bool] whether the flow predictions should feed
        into the temp predictions
        :param rand_seed: [int] the random seed for initialization
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.A = A.astype("float32")
        self.flow_in_temp = flow_in_temp

        # set up the layer
        self.lstm = tf.keras.layers.LSTMCell(hidden_size)

        ### set up the weights ###
        w_initializer = tf.random_normal_initializer(
            stddev=0.02, seed=rand_seed
        )

        # was Wg1
        self.W_graph_h = self.add_weight(
            shape=[hidden_size, hidden_size],
            initializer=w_initializer,
            name="W_graph_h",
        )
        # was bg1
        self.b_graph_h = self.add_weight(
            shape=[hidden_size], initializer="zeros", name="b_graph_h"
        )
        # was Wg2
        self.W_graph_c = self.add_weight(
            shape=[hidden_size, hidden_size],
            initializer=w_initializer,
            name="W_graph_c",
        )
        # was bg2
        self.b_graph_c = self.add_weight(
            shape=[hidden_size], initializer="zeros", name="b_graph_c"
        )

        # was Wa1
        self.W_h_cur = self.add_weight(
            shape=[hidden_size, hidden_size],
            initializer=w_initializer,
            name="W_h_cur",
        )
        # was Wa2
        self.W_h_prev = self.add_weight(
            shape=[hidden_size, hidden_size],
            initializer=w_initializer,
            name="W_h_prev",
        )
        # was ba
        self.b_h = self.add_weight(
            shape=[hidden_size], initializer="zeros", name="b_h"
        )

        # was Wc1
        self.W_c_cur = self.add_weight(
            shape=[hidden_size, hidden_size],
            initializer=w_initializer,
            name="W_c_cur",
        )
        # was Wc2
        self.W_c_prev = self.add_weight(
            shape=[hidden_size, hidden_size],
            initializer=w_initializer,
            name="W_c_prev",
        )
        # was bc
        self.b_c = self.add_weight(
            shape=[hidden_size], initializer="zeros", name="b_c"
        )

        if self.flow_in_temp:
            # was W2
            self.W_out_flow = self.add_weight(
                shape=[hidden_size, 1], initializer=w_initializer, name="W_out"
            )
            # was b2
            self.b_out_flow = self.add_weight(
                shape=[1], initializer="zeros", name="b_out"
            )

            self.W_out_temp = self.add_weight(
                shape=[hidden_size + 1, 1],
                initializer=w_initializer,
                name="W_out",
            )

            self.b_out_temp = self.add_weight(
                shape=[1], initializer="zeros", name="b_out"
            )
        else:
            # was W2
            self.W_out = self.add_weight(
                shape=[hidden_size, 2], initializer=w_initializer, name="W_out"
            )
            # was b2
            self.b_out = self.add_weight(
                shape=[2], initializer="zeros", name="b_out"
            )

    @tf.function
    def call(self, inputs, **kwargs):
        graph_size = self.A.shape[0]
        hidden_state_prev, cell_state_prev = (
            tf.zeros([graph_size, self.hidden_size]),
            tf.zeros([graph_size, self.hidden_size]),
        )
        out = []
        n_steps = inputs.shape[1]
        for t in range(n_steps):
            h_graph = tf.nn.tanh(
                tf.matmul(
                    self.A,
                    tf.matmul(hidden_state_prev, self.W_graph_h)
                    + self.b_graph_h,
                )
            )
            c_graph = tf.nn.tanh(
                tf.matmul(
                    self.A,
                    tf.matmul(cell_state_prev, self.W_graph_c) + self.b_graph_c,
                )
            )

            seq, state = self.lstm(
                inputs[:, t, :], states=[hidden_state_prev, cell_state_prev]
            )
            hidden_state_cur, cell_state_cur = state

            h_update = tf.nn.sigmoid(
                tf.matmul(hidden_state_cur, self.W_h_cur)
                + tf.matmul(h_graph, self.W_h_prev)
                + self.b_h
            )
            c_update = tf.nn.sigmoid(
                tf.matmul(cell_state_cur, self.W_c_cur)
                + tf.matmul(c_graph, self.W_c_prev)
                + self.b_c
            )

            if self.flow_in_temp:
                out_pred_q = (
                    tf.matmul(h_update, self.W_out_flow) + self.b_out_flow
                )
                out_pred_t = (
                    tf.matmul(
                        tf.concat([h_update, out_pred_q], axis=1),
                        self.W_out_temp,
                    )
                    + self.b_out_temp
                )
                out_pred = tf.concat([out_pred_t, out_pred_q], axis=1)
            else:
                out_pred = tf.matmul(h_update, self.W_out) + self.b_out

            out.append(out_pred)

            hidden_state_prev = h_update
            cell_state_prev = c_update
        out = tf.stack(out)
        out = tf.transpose(out, [1, 0, 2])
        return out


class RGCNModel(tf.keras.Model):
    def __init__(self, hidden_size, A, flow_in_temp=False, rand_seed=None):
        """
        :param hidden_size: [int] the number of hidden units
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
