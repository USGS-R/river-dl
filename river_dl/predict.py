import pandas as pd
import numpy as np
import xarray as xr
import datetime
import torch
import tensorflow as tf
from numpy.lib.npyio import NpzFile
from river_dl.torch_utils import predict_torch
from river_dl.postproc_utils import prepped_array_to_df
from river_dl.preproc_utils import (
    scale,
    convert_batch_reshape,
    coord_as_reshaped_array,
)

def get_data_if_file(d):
    """
    rudimentary check if data .npz file is already loaded. if not, load it
    :param d:
    :return:
    """
    if isinstance(d, NpzFile) or isinstance(d, dict):
        return d
    else:
        return np.load(d, allow_pickle=True)


def unscale_output(y_scl, y_std, y_mean, y_vars, log_vars=None):
    """
    unscale output data given a standard deviation and a mean value for the
    outputs
    :param y_scl: [pd dataframe] scaled output data (predicted or observed)
    :param y_std:[numpy array] array of standard deviation of variables_to_log [n_out]
    :param y_mean:[numpy array] array of variable means [n_out]
    :param y_vars: [list-like] y_dataset variable names
    :param log_vars: [list-like] which variables_to_log (if any) were logged in data
    prep
    :return: unscaled data
    """
    y_unscaled = y_scl.copy()
    # I'm replacing just the variable columns. I have to specify because, at
    # least in some cases, there are other columns (e.g., "seg_id_nat" and
    # date")
    y_unscaled[y_vars] = (y_scl[y_vars] * y_std) + y_mean
    if log_vars:
        y_unscaled[log_vars] = np.exp(y_unscaled[log_vars])
    return y_unscaled




def predict_from_io_data(
    model,
    io_data,
    partition,
    outfile,
    log_vars=False,
    trn_offset = 1.0,
    tst_val_offset = 1.0,
    spatial_idx_name="seg_id_nat",
    time_idx_name="date",
):
    """
    make predictions from trained model
    :param io_data: [str] directory to prepped data file
    :param partition: [str] must be 'trn' or 'tst'; whether you want to predict
    for the train or the dev period
    :param outfile: [str] the file where the output data should be stored
    :param log_vars: [list-like] which variables_to_log (if any) were logged in data
    prep
    :param trn_offset: [str] value for the training offset
    :param tst_val_offset: [str] value for the testing and validation offset
    :return: [pd dataframe] predictions
    """
    io_data = get_data_if_file(io_data)
    if partition == "trn":
        keep_portion = trn_offset
    else:
        keep_portion = tst_val_offset

    preds = predict(
        model,
        io_data[f"x_{partition}"],
        io_data[f"ids_{partition}"],
        io_data[f"times_{partition}"],
        io_data["y_std"],
        io_data["y_mean"],
        io_data["y_obs_vars"],
        keep_last_portion=keep_portion,
        outfile=outfile,
        log_vars=log_vars,
        spatial_idx_name=spatial_idx_name,
        time_idx_name=time_idx_name,
    )
    return preds


def predict(
    model,
    x_data,
    pred_ids,
    pred_dates,
    y_stds,
    y_means,
    y_vars,
    keep_last_portion=1.0,
    outfile=None,
    log_vars=False,
    spatial_idx_name="seg_id_nat",
    time_idx_name="date",
):
    """
    use trained model to make predictions
    :param model: [tf model] trained TF model to use for predictions
    :param x_data: [np array] numpy array of scaled and centered x_data
    :param pred_ids: [np array] the ids of the segments (same shape as x_data)
    :param pred_dates: [np array] the dates of the segments (same shape as
    x_data)
    :param keep_last_portion: [float] fraction of the predictions to keep starting
    from the *end* of the predictions (0-1). (1 means you keep all of the
    predictions, .75 means you keep the final three quarters of the predictions). Alternatively, if
    keep_last_portion is > 1 it's taken as an absolute number of predictions to retain from the end of the
    prediction sequence.
    :param y_stds:[np array] the standard deviation of the y_dataset data
    :param y_means:[np array] the means of the y_dataset data
    :param y_vars:[np array] the variable names of the y_dataset data
    :param outfile: [str] the file where the output data should be stored
    :param log_vars: [list-like] which variables_to_log (if any) were logged in data
    prep
    :return: out predictions
    """
    num_segs = len(np.unique(pred_ids))

    if issubclass(type(model), torch.nn.Module):
        if len(x_data.shape) > 3: #Catch for dealing with different GraphWaveNet vs RGCN output, consider changing to bool argument
            y_pred = predict_torch(x_data, model, batch_size=5)
            y_pred=y_pred.transpose(1,3)
            pred_ids = np.transpose(pred_ids,(0,3,2,1))
            pred_dates=np.transpose(pred_dates,(0,3,2,1))
        else:
            y_pred = predict_torch(x_data, model, batch_size=num_segs)
    elif issubclass(type(model), tf.keras.Model):
        y_pred = model.predict(x_data, batch_size=num_segs)
    else:
        raise TypeError("Model must be a torch.nn.Module or tf.Keras.Model")
    # keep only specified part of predictions
    if keep_last_portion>1:
        frac_seq_len = int(keep_last_portion)
    else:
        frac_seq_len = round(pred_ids.shape[1] * (keep_last_portion))

    y_pred = y_pred[:, -frac_seq_len:,...]
    pred_ids = pred_ids[:, -frac_seq_len:,...]
    pred_dates = pred_dates[:, -frac_seq_len:,...]

    y_pred_pp = prepped_array_to_df(y_pred, pred_dates, pred_ids, y_vars, spatial_idx_name, time_idx_name)

    y_pred_pp = unscale_output(y_pred_pp, y_stds, y_means, y_vars, log_vars,)

    if outfile:
        y_pred_pp.to_feather(outfile)
    return y_pred_pp


def mean_or_std_dataset_from_np(data, data_label, var_names_label):
    """
    turn a numpy data array of means or standard deviations into a xarry dataset
    :param data: the numpy NpzFile
    :param data_label: [str] the label for the values you want to turn into the
    xarray dataset (i.e., "x_mean" or "x_std")
    :param var_names_label: [str] the label of the data you want to become the
    variable names of the xarray dataset (i.e., "x_cols")
    :return:xarray dataset of the means or standard deviations
    """
    df = pd.DataFrame([data[data_label]], columns=data[var_names_label])
    # take the "min" to drop index level. it's only one value per variable
    # so the minis meaningless
    ds = df.to_xarray().min()
    return ds


def swap_first_seq_halves(x_data, batch_size):
    """
    make an additional batch from the first batch. the additional batch will
    have the first and second halves of the original first batch switched

    :param x_data: [np array] x data with shape [nseg * nbatch, seq_len, nfeat]
    :param batch_size: [int] the size of the batch (number of segments)
    :return: [np array] original data with an additional batch
    """
    first_batch = x_data[:batch_size, :, :]
    seq_len = x_data.shape[1]
    half_size = round(seq_len / 2)
    first_half_first_batch = first_batch[:, :half_size, :]
    second_half_first_batch = first_batch[:, half_size:, :]
    swapped = np.concatenate(
        [second_half_first_batch, first_half_first_batch], axis=1
    )
    new_x_data = np.concatenate([swapped, x_data], axis=0)
    return new_x_data


def predict_one_date_range(
    model,
    ds_x_scaled,
    train_io_data,
    seq_len,
    start_date,
    end_date,
    spatial_idx_name="seg_id_nat",
    time_idx_name="date",
    log_vars=None,
    keep_last_frac=1.0,
    offset=0.5,
    swap_halves_of_first_seq=False,
):
    """
    make predictions for one date range. This was broken out to be able to do
    the "beginning - middle - end" predictions more easily

    :param model: loaded tensorflow model
    :param ds_x_scaled: [xr array] scaled x data
    :param train_io_data: [np NpzFile] data containing the y_std, y_mean, y_vars
    :param seq_len: [int] length of the prediction sequences (usu. 365)
    :param start_date: [str or date] the start date of the predictions
    :param end_date: [str or date] the end date of the predictions
    :param spatial_idx_name: [str] name of column that is used for spatial
        index (e.g., 'seg_id_nat')
    :param time_idx_name: [str] name of column that is used for temporal index
        (usually 'time')
    :param log_vars: [list-like] which variables_to_log (if any) were logged in data
    prep
    :param keep_last_frac: [float] fraction of the predictions to keep starting
    from the *end* of the predictions (0-1). (1 means you keep all of the
    predictions, .75 means you keep the final three quarters of the predictions).
    Values greater than 1 are taken as a constant number of predictions to keep from the
    end of the sequence.
    :param offset: [float] 0-1, how to offset the batches (e.g., 0.5 means that
    the first batch will be 0-365 and the second will be 182-547). Values greater than
    1 are taken as a constant number of observations to offset by.
    :param swap_halves_of_first_seq: [bool] whether or not to make an
    *additional* sequence from the first sequence. The additional sequence will
    be the first sequence with the first and last halves swapped. The last half
    of the the first sequence serves as a stand-in spin-up period for ths first
    half predictions. This option makes most sense only when keep_last_portion=0.5.
    :return: [pd dataframe] the predictions
    """
    ds_x_scaled = ds_x_scaled[train_io_data["x_vars"]]
    x_data = ds_x_scaled.sel(date=slice(start_date, end_date))
    x_batches = convert_batch_reshape(
        x_data,
        seq_len=seq_len,
        offset=offset,
        spatial_idx_name=spatial_idx_name,
        time_idx_name=time_idx_name,
    )
    x_batch_ids = coord_as_reshaped_array(
        x_data,
        spatial_idx_name,
        seq_len=seq_len,
        offset=offset,
        spatial_idx_name=spatial_idx_name,
        time_idx_name=time_idx_name,
    )
    x_batch_dates = coord_as_reshaped_array(
        x_data,
        time_idx_name,
        seq_len=seq_len,
        offset=offset,
        spatial_idx_name=spatial_idx_name,
        time_idx_name=time_idx_name,
    )
    num_segs = len(np.unique(x_batch_ids))

    if swap_halves_of_first_seq:
        x_batches = swap_first_seq_halves(x_batches, num_segs)
        x_batch_ids = swap_first_seq_halves(x_batch_ids, num_segs)
        x_batch_dates = swap_first_seq_halves(x_batch_dates, num_segs)

    predictions = predict(
        model,
        x_batches,
        x_batch_ids,
        x_batch_dates,
        train_io_data["y_std"],
        train_io_data["y_mean"],
        train_io_data["y_obs_vars"],
        keep_last_portion=keep_last_frac,
        log_vars=log_vars,
        spatial_idx_name=spatial_idx_name,
        time_idx_name=time_idx_name
    )
    return predictions


def predict_from_arbitrary_data(
    raw_data_file,
    pred_start_date,
    pred_end_date,
    train_io_data,
    model,
    spatial_idx_name="seg_id_nat",
    time_idx_name="date",
    seq_len=365,
    log_vars=None,
):
    """
    make predictions given raw data that is potentially independent from the
    data used to train the model

    :param raw_data_file: [str] path to zarr dataset with x data that you want
    to use to make predictions
    :param pred_start_date: [str] start date of predictions (fmt: YYYY-MM-DD)
    :param pred_end_date: [str] end date of predictions (fmt: YYYY-MM-DD)
    :param train_io_data: [str or np NpzFile] the path to or the loaded data
    that was used to train the model. This file must contain the variables_to_log
    names, the standard deviations, and the means of the X and Y variables_to_log. Only
    in with this information can the model be used properly
    :param model: [tf model] model to use for predictions
    :param spatial_idx_name: [str] name of column that is used for spatial
        index (e.g., 'seg_id_nat')
    :param time_idx_name: [str] name of column that is used for temporal index
        (usually 'time')
    :param seq_len: [int] length of input sequences given to model
    :param flow_in_temp: [bool] whether the flow should be an input into temp
    for the rgcn model
    :param log_vars: [list-like] which variables_to_log (if any) were logged in data
    prep
    :return: [pd dataframe] the predictions
    """
    train_io_data = get_data_if_file(train_io_data)

    ds = xr.open_zarr(raw_data_file)

    ds_x = ds[train_io_data["x_vars"]]

    x_stds = mean_or_std_dataset_from_np(train_io_data, "x_std", "x_vars")
    x_means = mean_or_std_dataset_from_np(train_io_data, "x_mean", "x_vars")

    ds_x_scaled, _, _ = scale(ds_x, std=x_stds, mean=x_means)

    pred_start_date = datetime.datetime.strptime(pred_start_date, "%Y-%m-%d")
    # look back half of the sequence length before the prediction start date.
    # if present, this serves as a half-sequence warm-up period
    inputs_start_date = pred_start_date - datetime.timedelta(round(seq_len / 2))

    # get the "middle" predictions
    middle_predictions = predict_one_date_range(
        model,
        ds_x_scaled,
        train_io_data,
        seq_len,
        inputs_start_date,
        pred_end_date,
        log_vars=log_vars,
        spatial_idx_name=spatial_idx_name,
        time_idx_name=time_idx_name,
        keep_last_frac=0.5,
        offset=0.5,
    )

    # get the "beginning" predictions
    start_dates_end = pred_start_date + datetime.timedelta(seq_len)

    beginning_predictions = predict_one_date_range(
        model,
        ds_x_scaled,
        train_io_data,
        seq_len,
        pred_start_date,
        start_dates_end,
        log_vars=log_vars,
        spatial_idx_name=spatial_idx_name,
        time_idx_name=time_idx_name,
        keep_last_frac=1,
        offset=0.5,
        swap_halves_of_first_seq=True,
    )

    # get the "end" predictions
    end_date_end = datetime.datetime.strptime(
        pred_end_date, "%Y-%m-%d"
    ) + datetime.timedelta(1)
    end_dates_start = end_date_end - datetime.timedelta(seq_len)

    end_predictions = predict_one_date_range(
        model,
        ds_x_scaled,
        train_io_data,
        seq_len,
        end_dates_start,
        end_date_end,
        log_vars=log_vars,
        spatial_idx_name=spatial_idx_name,
        time_idx_name=time_idx_name,
        keep_last_frac=1,
        offset=1,
    )

    # trim beginning and end predictions
    predictions_beginning_trim = beginning_predictions[
        beginning_predictions[time_idx_name] < middle_predictions[time_idx_name].min()
    ]
    predictions_end_trim = end_predictions[
        end_predictions[time_idx_name] > middle_predictions[time_idx_name].max()
    ]

    predictions_combined = pd.concat(
        [predictions_beginning_trim, middle_predictions, predictions_end_trim]
    )
    return predictions_combined
