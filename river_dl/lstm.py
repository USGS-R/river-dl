# -*- coding: utf-8 -*-
"""
@author: jeff sadler, Feb 2020
based off code by Xiaowei Jia

"""

from __future__ import print_function, division
import tensorflow as tf
from tensorflow.keras import layers
from RGCN import rmse_masked_one_var


class LSTMModel(tf.keras.Model):
    def __init__(self, hidden_size):
        """
        :param hidden_size: [int] the number of hidden units
        """
        super().__init__()
        self.lstm_layer = layers.LSTM(hidden_size, return_sequences=True,
                                      name='lstm_shared')
        self.dense_main = layers.Dense(1, name='dense_main')
        self.dense_aux = layers.Dense(1, name='dense_aux')

    @tf.function
    def call(self, inputs, **kwargs):
        x = self.lstm_layer(inputs)
        main_prediction = self.dense_main(x)
        aux_prediction = self.dense_aux(x)
        return tf.concat([main_prediction, aux_prediction], axis=2)


def adjust_gradient(main_grad, aux_grad):
    projection = tf.minimum(tf.matmul(main_grad, aux_grad), 0) * main_grad / tf.matmul(main_grad, main_grad)
    return aux_grad - projection


def get_variables(trainable_variables, name):
    return [v for v in trainable_variables if name in v.name]


class LSTMGradSimilarity(LSTMModel):
    @tf.function
    def train_step(self, data):
        x, y = data

        with tf.GradientTape(persistent=True) as tape:
            y_pred = self(x, training=True) # forward pass

            loss_main = rmse_masked_one_var(y, y_pred, 0)
            loss_aux = rmse_masked_one_var(y, y_pred, 1)

        trainable_vars = self.trainable_variables

        main_out_vars = get_variables(trainable_vars, 'dense_main')
        aux_out_vars = get_variables(trainable_vars, 'dense_aux')
        shared_vars = get_variables(trainable_vars, 'lstm_shared')

        # get gradients
        gradient_main_out = tape.gradient(loss_main, main_out_vars)
        gradient_aux_out = tape.gradient(loss_aux, aux_out_vars)
        gradient_shared_main = tape.gradient(loss_main, shared_vars)
        gradient_shared_aux = tape.gradient(loss_aux, shared_vars)

        # adjust auxiliary gradient
        adjusted_aux_grad = adjust_gradient(gradient_shared_main,
                                            gradient_shared_aux)
        # combined adjusted auxiliary gradient and main gradient
        combined_grad = gradient_shared_main + adjusted_aux_grad

        # apply gradients
        self.optimizer.apply_gradients(zip(gradient_main_out, main_out_vars))
        self.optimizer.apply_gradients(zip(gradient_aux_out, aux_out_vars))
        self.optimizer.apply_gradients(zip(combined_grad, shared_vars))
