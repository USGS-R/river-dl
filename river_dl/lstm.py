# -*- coding: utf-8 -*-
"""
@author: jeff sadler, Feb 2020
based off code by Xiaowei Jia

"""

from __future__ import print_function, division
import tensorflow as tf
from tensorflow.keras import layers


class LSTMModel(tf.keras.Model):
    def __init__(self, hidden_size):
        """
        :param hidden_size: [int] the number of hidden units
        """
        super().__init__()
        self.lstm_layer = layers.LSTM(hidden_size, return_sequences=True)
        self.dense_temp = layers.Dense(1)
        self.dense_flow = layers.Dense(1)

    def call(self, inputs, **kwargs):
        x = self.lstm_layer(inputs)
        temp_prediction = self.dense_flow(x)
        flow_prediction = self.dense_flow(x)
        return tf.concat([temp_prediction, flow_prediction], axis=2)


