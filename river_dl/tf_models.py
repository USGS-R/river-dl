# -*- coding: utf-8 -*-
"""
@author: jeff sadler, Feb 2020
based off code by Xiaowei Jia

"""

from __future__ import print_function, division
import tensorflow as tf
from tensorflow.keras import layers


class RGCN(layers.Layer):
    def __init__(
        self, hidden_size, A, recurrent_dropout=0, dropout=0, rand_seed=None,
    ):
        """

        :param hidden_size: [int] the number of hidden units
        :param A: [numpy array] adjacency matrix
        :param recurrent_dropout: [float] value between 0 and 1 for the
        probability of a recurrent element to be zero
        :param dropout: [float] value between 0 and 1 for the probability of an
        input element to be zero
        :param rand_seed: [int] the random seed for initialization
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.A = A.astype("float32")

        # set up the layer
        self.lstm = tf.keras.layers.LSTMCell(
            hidden_size, recurrent_dropout=recurrent_dropout, dropout=dropout
        )

        ### set up the weights ###
        w_initializer = tf.initializers.glorot_normal(seed=rand_seed)

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

    @tf.function
    def call(self, inputs, **kwargs):
        h_list = []
        c_list = []
        n_steps = inputs.shape[1]
        # set the initial h & c states to the supplied h and c states if using
        # DA, or 0's otherwise
        hidden_state_prev = tf.cast(kwargs["h_init"], tf.float32)
        cell_state_prev = tf.cast(kwargs["c_init"], tf.float32)
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

            ## Update dropout masks
            self.lstm.reset_dropout_mask()
            self.lstm.reset_recurrent_dropout_mask()

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

            hidden_state_prev = h_update
            cell_state_prev = c_update

            h_list.append(h_update)
            c_list.append(c_update)

        h_list = tf.stack(h_list)
        c_list = tf.stack(c_list)
        h_list = tf.transpose(h_list, [1, 0, 2])
        c_list = tf.transpose(c_list, [1, 0, 2])
        return h_list, c_list


class RGCNModel(tf.keras.Model):
    def __init__(
        self,
        hidden_size,
        A,
        num_tasks=1,
        recurrent_dropout=0,
        dropout=0,
        rand_seed=None,
    ):
        """
        :param hidden_size: [int] the number of hidden units
        :param A: [numpy array] adjacency matrix
        :param num_tasks: [int] number of prediction tasks to perform -
        currently supports either 1 or 2 prediction tasks
        :param recurrent_dropout: [float] value between 0 and 1 for the
        probability of a recurrent element to be zero
        :param dropout: [float] value between 0 and 1 for the probability of an
        input element to be zero
        into the temp predictions
        :param rand_seed: [int] the random seed for initialization
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_tasks = num_tasks
        self.recurrent_dropout = recurrent_dropout
        self.dropout = dropout

        self.rgcn_layer = RGCN(
            hidden_size, A, recurrent_dropout, dropout, rand_seed
        )

        self.states = None

        self.dense_main = layers.Dense(1, name="dense_main")
        if self.num_tasks == 2:
            self.dense_aux = layers.Dense(1, name="dense_aux")

    def call(self, inputs, **kwargs):
        batch_size = tf.shape(inputs)[0]
        h_init = kwargs.get("h_init", tf.zeros([batch_size, self.hidden_size]))
        c_init = kwargs.get("c_init", tf.zeros([batch_size, self.hidden_size]))
        h_gr, c_gr = self.rgcn_layer(inputs, h_init=h_init, c_init=c_init)
        self.states = h_gr[:, -1, :], c_gr[:, -1, :]

        if self.num_tasks == 1:
            main_prediction = self.dense_main(h_gr)
            return main_prediction
        elif self.num_tasks == 2:
            main_prediction = self.dense_main(h_gr)
            aux_prediction = self.dense_aux(h_gr)
            return tf.concat([main_prediction, aux_prediction], axis=2)
        else:
            raise ValueError(
                f"This model only supports 1 or 2 tasks (not {self.num_tasks})"
            )



## LSTM
class LSTMModel(tf.keras.Model):
    def __init__(
        self, hidden_size, num_tasks=1, recurrent_dropout=0, dropout=0,
    ):
        """
        :param hidden_size: [int] the number of hidden units
        :param num_tasks: [int] number of tasks (variables_to_log to be predicted)
        :param recurrent_dropout: [float] value between 0 and 1 for the
        probability of a recurrent element to be zero
        :param dropout: [float] value between 0 and 1 for the probability of an
        input element to be zero
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_tasks = num_tasks
        self.rnn_layer = layers.LSTM(
            hidden_size,
            return_sequences=True,
            stateful=True,
            return_state=True,
            recurrent_dropout=recurrent_dropout,
            dropout=dropout,
        )
        self.dense_main = layers.Dense(1, name="dense_main")
        if self.num_tasks == 2:
            self.dense_aux = layers.Dense(1, name="dense_aux")
        self.states = None

    @tf.function
    def call(self, inputs, **kwargs):
        batch_size = tf.shape(inputs)[0]
        h_init = kwargs.get("h_init", tf.zeros([batch_size, self.hidden_size]))
        c_init = kwargs.get("c_init", tf.zeros([batch_size, self.hidden_size]))
        self.rnn_layer.reset_states(states=[h_init, c_init])
        x, h, c = self.rnn_layer(inputs)
        self.states = h, c
        if self.num_tasks == 1:
            main_prediction = self.dense_main(x)
            return main_prediction
        elif self.num_tasks == 2:
            main_prediction = self.dense_main(x)
            aux_prediction = self.dense_aux(x)
            return tf.concat([main_prediction, aux_prediction], axis=2)
        else:
            raise ValueError(
                f"This model only supports 1 or 2 tasks (not {self.num_tasks})"
            )


### GRU
class GRUModel(LSTMModel):
    def __init__(
        self, hidden_size, num_tasks=1, dropout=0, recurrent_dropout=0,
    ):
        """
        :param hidden_size: [int] the number of hidden units
        :param num_tasks: [int] number of tasks (variables_to_log to be predicted)
        :param recurrent_dropout: [float] value between 0 and 1 for the
        probability of a recurrent element to be zero
        :param dropout: [float] value between 0 and 1 for the probability of an
        """
        super().__init__(hidden_size, num_tasks=num_tasks)
        self.rnn_layer = layers.GRU(
            hidden_size,
            recurrent_dropout=recurrent_dropout,
            dropout=dropout,
            return_sequences=True,
            name="rnn_shared",
        )
