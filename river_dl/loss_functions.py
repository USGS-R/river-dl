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


def nse(y_true, y_pred):
    zero_or_error = tf.where(
        tf.math.is_nan(y_true), tf.zeros_like(y_true), y_pred - y_true
    )

    numerator = tf.reduce_sum(tf.square(zero_or_error))

    deviation = dev_masked(y_true)
    denominator = tf.reduce_sum(tf.square(deviation))
    return 1 - numerator / denominator  


def nnse(y_true, y_pred):
    return 1 / (2 - nse(y_true, y_pred))


def nnse_loss(y_true, y_pred):
    return 1 - nnse(y_true, y_pred)


@tf.function
def nnse_masked_one_var(data, y_pred, var_idx):
    y_true, y_pred, weights = y_data_components(data, y_pred, var_idx)
    return nnse_loss(y_true, y_pred)


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


def mean_masked(y):
    num_vals = tf.cast(
        tf.math.count_nonzero(~tf.math.is_nan(y)), tf.float32
    )
    # get mean accounting for nans
    zero_or_val = tf.where(
        tf.math.is_nan(y), tf.zeros_like(y), y
    )
    mean = tf.reduce_sum(zero_or_val) / num_vals
    return mean


def dev_masked(y):
    mean = mean_masked(y)
    zero_or_dev = tf.where(
        tf.math.is_nan(y), tf.zeros_like(y), y - mean
    )
    return zero_or_dev


def std_masked(y):
    dev = dev_masked(y)
    num_vals = tf.cast(
        tf.math.count_nonzero(~tf.math.is_nan(y)), tf.float32
    )
    numerator = tf.reduce_sum(tf.square(dev))
    denominator = num_vals - 1
    return tf.sqrt(numerator/denominator)


def pearsons_r(y_true, y_pred):
    y_true_dev = dev_masked(y_true)
    y_pred_dev = dev_masked(y_pred)
    numerator = tf.reduce_sum(y_true_dev * y_pred_dev)
    ss_dev_true = tf.reduce_sum(tf.square(y_true_dev))
    ss_pred_true = tf.reduce_sum(tf.square(y_pred_dev))
    denominator = tf.sqrt(ss_dev_true * ss_pred_true)
    return numerator/denominator


def kge(y_true, y_pred):
    r = pearsons_r(y_true, y_pred)
    mean_true = mean_masked(y_true)
    mean_pred = mean_masked(y_pred)
    std_true = std_masked(y_true)
    std_pred = std_masked(y_pred)

    r_component = tf.square(r - 1)
    std_component = tf.square((std_pred/std_true) - 1)
    bias_component = tf.square((mean_pred/mean_true) - 1)
    return 1 - tf.sqrt(r_component + std_component + bias_component)


def norm_kge(y_true, y_pred):
    """
    normalized kge so it's scaled from 0 to 1
    """
    return 1 / (2 - kge(y_true, y_pred))


def kge_loss(y_true, y_pred):
    """
    making it a loss, so low is good, high is bad
    """
    return 1 - norm_kge(y_true, y_pred)


def kge_loss_one_var(data, y_pred, var_idx):
    y_true, y_pred, weights = y_data_components(data, y_pred, var_idx)
    return kge_loss(y_true, y_pred)
