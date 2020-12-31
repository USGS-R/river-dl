# -*- coding: utf-8 -*-
from __future__ import print_function, division
import tensorflow as tf
from tensorflow.keras import layers
from river_dl.loss_functions import nnse_masked_one_var, nnse_one_var_samplewise


class LSTMModel(tf.keras.Model):
    def __init__(
        self,
        hidden_size,
        gradient_correction=False,
        lamb=1,
        dropout=0,
        grad_log_file=None
    ):
        """
        :param hidden_size: [int] the number of hidden units
        """
        super().__init__()
        self.gradient_correction = gradient_correction
        self.grad_log_file = grad_log_file
        self.lamb = lamb
        self.rnn_layer = layers.LSTM(
            hidden_size,
            return_sequences=True,
            name="rnn_shared",
            recurrent_dropout=dropout
        )
        self.dense_main = layers.Dense(1, name="dense_main")
        self.dense_aux = layers.Dense(1, name="dense_aux")

    @tf.function
    def call(self, inputs, **kwargs):
        x = self.rnn_layer(inputs)
        main_prediction = self.dense_main(x)
        aux_prediction = self.dense_aux(x)
        return tf.concat([main_prediction, aux_prediction], axis=2)

    @tf.function
    def train_step(self, data):
        x, y = data

        # If I don't do one forward pass before starting the gradient tape,
        # the thing hangs
        _ = self(x)
        with tf.GradientTape(persistent=True) as tape:
            y_pred = self(x, training=True)  # forward pass

            loss_main = nnse_one_var_samplewise(y, y_pred, 0)
            loss_aux = nnse_one_var_samplewise(y, y_pred, 1)

        trainable_vars = self.trainable_variables

        main_out_vars = get_variables(trainable_vars, "dense_main")
        aux_out_vars = get_variables(trainable_vars, "dense_aux")
        shared_vars = get_variables(trainable_vars, "rnn_shared")

        # get gradients
        gradient_main_out = tape.gradient(loss_main, main_out_vars)
        gradient_aux_out = tape.gradient(loss_aux, aux_out_vars)
        gradient_shared_main = tape.gradient(loss_main, shared_vars)
        gradient_shared_aux = tape.gradient(loss_aux, shared_vars)

        if self.gradient_correction:
            # adjust auxiliary gradient
            gradient_shared_aux = adjust_gradient_list(
                gradient_shared_main, gradient_shared_aux, self.grad_log_file
            )
        combined_gradient = combine_gradients_list(
            gradient_shared_main, gradient_shared_aux, lamb=self.lamb
        )

        # apply gradients
        self.optimizer.apply_gradients(zip(gradient_main_out, main_out_vars))
        self.optimizer.apply_gradients(zip(gradient_aux_out, aux_out_vars))
        self.optimizer.apply_gradients(zip(combined_gradient, shared_vars))
        return {"loss_main": loss_main, "loss_aux": loss_aux}


class GRUModel(LSTMModel):
    def __init__(self, hidden_size, lamb=1):
        """
        :param hidden_size: [int] the number of hidden units
        """
        super().__init__(hidden_size, lamb=lamb)
        self.rnn_layer = layers.GRU(
            hidden_size, return_sequences=True, name="rnn_shared"
        )


def adjust_gradient(main_grad, aux_grad, logfile=None):
    # flatten tensors
    main_grad_flat = tf.reshape(main_grad, [-1])
    aux_grad_flat = tf.reshape(aux_grad, [-1])

    # project and adjust
    projection = (
        tf.minimum(tf.reduce_sum(main_grad_flat * aux_grad_flat), 0)
        * main_grad_flat
        / tf.reduce_sum(main_grad_flat * main_grad_flat)
    )
    if logfile:
        logfile = "file://" + logfile
        tf.print(tf.reduce_sum(projection), output_stream=logfile, sep=",")
    projection = tf.cond(
        tf.math.is_nan(tf.reduce_sum(projection)),
        lambda: tf.zeros(aux_grad_flat.shape),
        lambda: projection,
    )
    adjusted = aux_grad_flat - projection
    return tf.reshape(adjusted, aux_grad.shape)


def get_variables(trainable_variables, name):
    return [v for v in trainable_variables if name in v.name]


def combine_gradients_list(main_grads, aux_grads, lamb=1):
    return [main_grads[i] + lamb * aux_grads[i] for i in range(len(main_grads))]


def adjust_gradient_list(main_grads, aux_grads, logfile=None):
    return [
        adjust_gradient(main_grads[i], aux_grads[i], logfile)
        for i in range(len(main_grads))
    ]
