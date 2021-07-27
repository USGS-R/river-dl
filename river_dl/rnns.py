# -*- coding: utf-8 -*-
from __future__ import print_function, division
import tensorflow as tf
from tensorflow.keras import layers


class LSTMModel(tf.keras.Model):
    def __init__(
        self, hidden_size, num_tasks=1, recurrent_dropout=0, dropout=0,
    ):
        """
        :param hidden_size: [int] the number of hidden units
        :param num_tasks: [int] number of tasks (variables to be predicted)
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


class GRUModel(LSTMModel):
    def __init__(
        self, hidden_size, num_tasks=1, dropout=0, recurrent_dropout=0,
    ):
        """
        :param hidden_size: [int] the number of hidden units
        :param num_tasks: [int] number of tasks (variables to be predicted)
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
