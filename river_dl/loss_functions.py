import numpy as np
import math as m
import tensorflow as tf

#from river_dl.train import train_model


@tf.function
def rmse(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    num_y_true = tf.cast(
        tf.math.count_nonzero(~tf.math.is_nan(y_true)), tf.float32
    )
    if num_y_true > 0:
        zero_or_error = tf.where(
            tf.math.is_nan(y_true), tf.zeros_like(y_true), y_pred - y_true
        )
        sum_squared_errors = tf.reduce_sum(tf.square(zero_or_error))
        rmse_loss = tf.sqrt(sum_squared_errors / num_y_true)
    else:
        rmse_loss = 0.0
    return rmse_loss


def sample_avg_nse(y_true, y_pred):
    """
    calculate the sample averaged nse, i.e., it will calculate the nse across
    each of the samples (the 1st dimension of the arrays) and then average those
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    zero_or_error = tf.where(
        tf.math.is_nan(y_true), tf.zeros_like(y_true), y_pred - y_true
    )

    # add a small value to the deviation to prevent instability
    deviation = dev_masked(y_true) + 0.1

    numerator_samplewise = tf.reduce_sum(tf.square(zero_or_error), axis=1)
    denomin_samplewise = tf.reduce_sum(tf.square(deviation), axis=1)
    nse_samplewise = 1 - numerator_samplewise / denomin_samplewise
    nse_samplewise_avg = tf.reduce_sum(nse_samplewise) / tf.cast(
        tf.shape(y_true)[0], tf.float32
    )
    return nse_samplewise_avg


def nse(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    zero_or_error = tf.where(
        tf.math.is_nan(y_true), tf.zeros_like(y_true), y_pred - y_true
    )

    deviation = dev_masked(y_true)
    numerator = tf.reduce_sum(tf.square(zero_or_error))
    denominator = tf.reduce_sum(tf.square(deviation))
    return 1 - numerator / denominator


def nnse(y_true, y_pred):
    return 1 / (2 - nse(y_true, y_pred))


def nnse_loss(y_true, y_pred):
    return 1 - nnse(y_true, y_pred)


def samplewise_nnse_loss(y_true, y_pred):
    nnse_val = 1 / (2 - sample_avg_nse(y_true, y_pred))
    return 1 - nnse_val


def multitask_nse(lambdas):
    return multitask_loss(lambdas, nnse_loss)


def multitask_samplewise_nse(lambdas):
    return multitask_loss(lambdas, samplewise_nnse_loss)

    
def multitask_rmse(lambdas):
    return multitask_loss(lambdas, rmse)


def multitask_kge(lambdas):
    return multitask_loss(lambdas, kge_loss)


def multitask_loss(lambdas, loss_func):
    """
    calculate a weighted multi-task loss for a given number of variables_to_log with a
    given loss function
    :param lambdas: [array-like float] The factor that losses will be
    multiplied by before being added together.
    :param loss_func: [function] Loss function that will be used to calculate
    the loss of each variable. Must take as input parameters [y_true, y_pred]
    """

    def combine_loss(y_true, y_pred):
        losses = []
        n_vars = y_pred.shape[-1]
        for var_id in range(n_vars):
            ind_var_loss = loss_func(y_true[:, :, var_id], y_pred[:, :, var_id])
            weighted_ind_var_loss = lambdas[var_id] * ind_var_loss
            losses.append(weighted_ind_var_loss)
        total_loss = sum(losses)
        return total_loss

    return combine_loss


def mean_masked(y):
    num_vals = tf.cast(tf.math.count_nonzero(~tf.math.is_nan(y)), tf.float32)
    # get mean accounting for nans
    zero_or_val = tf.where(tf.math.is_nan(y), tf.zeros_like(y), y)
    mean = tf.reduce_sum(zero_or_val) / num_vals
    return mean


def dev_masked(y):
    mean = mean_masked(y)
    zero_or_dev = tf.where(tf.math.is_nan(y), tf.zeros_like(y), y - mean)
    return zero_or_dev


def std_masked(y):
    dev = dev_masked(y)
    num_vals = tf.cast(tf.math.count_nonzero(~tf.math.is_nan(y)), tf.float32)
    numerator = tf.reduce_sum(tf.square(dev))
    denominator = num_vals - 1
    return tf.sqrt(numerator / denominator)


def pearsons_r(y_true, y_pred):
    y_true_dev = dev_masked(y_true)
    y_pred_dev = dev_masked(y_pred)
    numerator = tf.reduce_sum(y_true_dev * y_pred_dev)
    ss_dev_true = tf.reduce_sum(tf.square(y_true_dev))
    ss_pred_true = tf.reduce_sum(tf.square(y_pred_dev))
    denominator = tf.sqrt(ss_dev_true * ss_pred_true)
    return numerator / denominator


def kge(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    r = pearsons_r(y_true, y_pred)
    mean_true = mean_masked(y_true)
    mean_pred = mean_masked(y_pred)
    std_true = std_masked(y_true)
    std_pred = std_masked(y_pred)

    r_component = tf.square(r - 1)
    std_component = tf.square((std_pred / std_true) - 1)
    bias_component = tf.square((mean_pred / mean_true) - 1)
    return 1 - tf.sqrt(r_component + std_component + bias_component)


def norm_kge(y_true, y_pred):
    """
    normalized kge so it's scaled from 0 to 1
    """
    return 1 / (2 - kge(y_true, y_pred))


def kge_norm_loss(y_true, y_pred):
    """
    making it a loss, so low is good, high is bad
    """
    return 1 - norm_kge(y_true, y_pred)


def kge_loss(y_true, y_pred):
    return -1 * kge(y_true, y_pred)

def weighted_masked_rmse_gw(loss_function_main, temp_index,temp_mean, temp_sd,gw_mean, gw_std, lambda_Ar=0, lambda_delPhi=0, num_task=2):
    """
    calculate a weighted, masked rmse that includes the groundwater terms
    :param lamb, lamb2, lamb3: [float] (short for lambda). The factor that the auxiliary loss, Ar loss, and deltaPhi loss
    will be multiplied by before added to the main loss.
    :param temp_index: [int]. Index of temperature column (0 if temp is the primary prediction, 1 if it is the auxiliary).
    :param temp_mean: [float]. Mean temperature for unscaling the predicted temperatures.
    :param temp_sd: [float]. Standard deviation of temperature for unscalind the predicted temperatures.
    """

    def rmse_masked_combined_gw(data, y_pred):
     
        Ar_obs, Ar_pred, delPhi_obs, delPhi_pred = GW_loss_prep(temp_index,data, y_pred, temp_mean, temp_sd,gw_mean, gw_std, num_task)
        rmse_Ar = rmse(Ar_obs,Ar_pred)
        rmse_delPhi = rmse(delPhi_obs,delPhi_pred)

        
        rmse_loss = loss_function_main(data[:,:,:num_task],y_pred) + lambda_Ar*rmse_Ar +lambda_delPhi*rmse_delPhi

        tf.debugging.assert_all_finite(
            rmse_loss, 'Nans is a bad loss to have'
        )
        return rmse_loss
    return rmse_masked_combined_gw


def GW_loss_prep(temp_index, data, y_pred, temp_mean, temp_sd, gw_mean, gw_std, num_task):
    # assumes that axis 0 of data and y_pred are the reaches and axis 1 are daily values
    # assumes the first two columns of data are the observed flow and temperature, and the remaining
    # ones (extracted here) are the data for gw analysis
    y_true = data[:, :, num_task:]

    y_pred_temp = y_pred[:, :, int(temp_index):(int(temp_index) + 1)]  # extract just the predicted temperature
    # unscale the predicted temps prior to calculating the amplitude and phase
    y_pred_temp = y_pred_temp * temp_sd + temp_mean
    y_pred_temp = tf.squeeze(y_pred_temp)
    y_pred_mean = tf.reduce_mean(y_pred_temp, 1, keepdims=True)
    temp_demean = y_pred_temp - y_pred_mean
    fft_tf = tf.signal.rfft(temp_demean)
    Phiw = tf.math.angle(fft_tf)
    phiIndex = tf.argmax(tf.abs(fft_tf), 1)
    idx = tf.stack(
        [tf.reshape(tf.range(tf.shape(Phiw)[0]), (-1, 1)),
         tf.reshape(tf.cast(phiIndex, tf.int32), (tf.shape(phiIndex)[0], 1))],
        axis=-1)
    Phiw_out = tf.squeeze(tf.gather_nd(Phiw, idx))

    Aw = tf.reduce_max(tf.abs(fft_tf), 1) / fft_tf.shape[1]  # tf.shape(fft_tf, out_type=tf.dtypes.float32)[1]

    y_true_air = y_true[:, :, -1]
    y_true_air_mean = tf.reduce_mean(y_true_air, 1, keepdims=True)
    air_demean = y_true_air - y_true_air_mean
    fft_tf_air = tf.signal.rfft(air_demean)
    Phia = tf.math.angle(fft_tf_air)

    phiIndex_air = tf.argmax(tf.abs(fft_tf_air), 1)
    ida = tf.stack(
        [tf.reshape(tf.range(tf.shape(Phia)[0]), (-1, 1)),
         tf.reshape(tf.cast(phiIndex_air, tf.int32), (tf.shape(phiIndex_air)[0], 1))],
        axis=-1)
    Phia_out = tf.squeeze(tf.gather_nd(Phia, ida))

    Aa = tf.reduce_max(tf.abs(fft_tf_air), 1) / fft_tf.shape[1]  # tf.shape(fft_tf_air, out_type=tf.dtypes.float32)[1]

    # calculate and scale predicted values
    # delPhi_pred = the difference in phase between the water temp and air temp sinusoids, in days
    delPhi_pred = ((Phiw_out - Phia_out) * 365 / (2 * m.pi) - gw_mean[1]) / gw_std[1]
    # Ar_pred = the ratio of the water temp and air temp amplitudes
    Ar_pred = (Aw / Aa - gw_mean[0]) / gw_std[0]

    return y_true[:, 0, 0], Ar_pred, y_true[:, 0, 1], delPhi_pred


