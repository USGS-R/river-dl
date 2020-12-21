import numpy as np
import tensorflow as tf


@tf.function
def rmse(y_true, y_pred, weights):
    num_y_true = tf.cast(
        tf.math.count_nonzero(~tf.math.is_nan(y_true)), tf.float32
    )
    if num_y_true > 0:
        zero_or_error = tf.where(
            tf.math.is_nan(y_true), tf.zeros_like(y_true), y_pred - y_true
        )
        wgt_zero_or_err = zero_or_error * weights
        sum_squared_errors = tf.reduce_sum(tf.square(wgt_zero_or_err))
        rmse_loss = tf.sqrt(sum_squared_errors / num_y_true)
    else:
        rmse_loss = 0.0
    return rmse_loss


@tf.function
def nnse(y_true, y_pred):
    num_y_true = tf.cast(
        tf.math.count_nonzero(~tf.math.is_nan(y_true)), tf.float32
    )
    if num_y_true > 0:
        # get mean accounting for nans
        zero_or_val = tf.where(
            tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true
        )
        obs_mean = tf.reduce_sum(zero_or_val) / num_y_true

        zero_or_error = tf.where(
            tf.math.is_nan(y_true), tf.zeros_like(y_true), y_pred - y_true
        )

        denom = tf.reduce_sum(tf.math.abs(zero_or_val - obs_mean))
        numerator = tf.reduce_sum(zero_or_error)
        nse = 1 - numerator / denom
        nnse = 1 / (2 - nse)
        nnse_loss = 1 - nnse
    else:
        nnse_loss = 0.0
    return nnse_loss


@tf.function
def nnse_masked_one_var(data, y_pred, var_idx):
    y_true, y_pred, weights = y_data_components(data, y_pred, var_idx)
    return nnse(y_true, y_pred, weights)


@tf.function
def y_data_components(data, y_pred, var_idx):
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

    weights = weights[:, :, var_idx]
    y_true = y_true[:, :, var_idx]
    y_pred = y_pred[:, :, var_idx]
    return y_true, y_pred, weights


@tf.function
def rmse_masked_one_var(data, y_pred, var_idx):
    y_true, y_pred, weights = y_data_components(data, y_pred, var_idx)
    return rmse(y_true, y_pred, weights)


@tf.function
def weighted_masked_rmse(lamb=0.5):
    """
    calculate a weighted, masked rmse.
    :param lamb: [float] (short for lambda). The factor that the auxiliary loss
    will be multiplied by before added to the main loss.
    """

    def rmse_masked_combined(data, y_pred):
        rmse_main = rmse_masked_one_var(data, y_pred, 0)
        rmse_aux = rmse_masked_one_var(data, y_pred, 1)
        rmse_loss = (1 - lamb) * rmse_main + lamb * rmse_aux
        return rmse_loss

    return rmse_masked_combined


