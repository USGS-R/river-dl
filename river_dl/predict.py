import pandas as pd
import numpy as np
import xarray as xr
import datetime

from river_dl.RGCN import RGCNModel
from river_dl.postproc_utils import prepped_array_to_df
from river_dl.preproc_utils import (
    scale,
    convert_batch_reshape,
    coord_as_reshaped_array,
)
from river_dl.rnns import LSTMModel, GRUModel
from river_dl.train import get_data_if_file


def unscale_output(y_scl, y_std, y_mean, data_cols, logged_q=False):
    """
    unscale output data given a standard deviation and a mean value for the
    outputs
    :param y_scl: [pd dataframe] scaled output data (predicted or observed)
    :param y_std:[numpy array] array of standard deviation of variables [n_out]
    :param y_mean:[numpy array] array of variable means [n_out]
    :param data_cols:
    :param logged_q: [bool] whether the model predicted log of discharge. if
    true, the exponent of the discharge will be executed
    :return:
    """
    yscl_data = y_scl[data_cols]
    y_unscaled_data = (yscl_data * y_std) + y_mean
    y_scl[data_cols] = y_unscaled_data
    if logged_q:
        y_scl["seg_outflow"] = np.exp(y_scl["seg_outflow"])
    return y_scl


def load_model_from_weights(
    model_type,
    model_weights_dir,
    hidden_size,
    dist_matrix=None,
    flow_in_temp=False,
):
    """

    :param flow_in_temp:
    :param model_type: [str] model to use either 'rgcn', 'lstm', or 'gru'
    :param model_weights_dir: [str] directory to saved model weights
    :param hidden_size: [int] the number of hidden units in model
    :param dist_matrix: [np array] the distance matrix if using 'rgcn'
    :param flow_in_temp: [bool] whether the flow should be an input into temp
    :return:
    """
    if model_type == "rgcn":
        model = RGCNModel(hidden_size, A=dist_matrix, flow_in_temp=flow_in_temp)
    elif model_type.startswith("lstm"):
        model = LSTMModel(hidden_size)
    elif model_type == "gru":
        model = GRUModel(hidden_size)

    model.load_weights(model_weights_dir)
    return model


def predict_from_io_data(
    model_type,
    model_weights_dir,
    hidden_size,
    io_data,
    partition,
    outfile,
    flow_in_temp=False,
    logged_q=False,
):
    """
    make predictions from trained model
    :param model_type: [str] model to use either 'rgcn', 'lstm', or 'gru'
    :param model_weights_dir: [str] directory to saved model weights
    :param io_data: [str] directory to prepped data file
    :param hidden_size: [int] the number of hidden units in model
    :param partition: [str] must be 'trn' or 'tst'; whether you want to predict
    for the train or the dev period
    :param outfile: [str] the file where the output data should be stored
    :param flow_in_temp: [bool] whether the flow should be an input into temp
    :param logged_q: [bool] whether the discharge was logged in training. if True
    the exponent of the discharge will be taken in the model unscaling
    :return: [pd dataframe] predictions
    """
    io_data = get_data_if_file(io_data)

    model = load_model_from_weights(
        model_type,
        model_weights_dir,
        hidden_size,
        io_data.get("dist_matrix"),
        flow_in_temp,
    )

    if partition != "trn":
        keep_only_second_half = True
    else:
        keep_only_second_half = False

    preds = predict(
        model,
        io_data[f"x_{partition}"],
        io_data[f"ids_{partition}"],
        io_data[f"dates_{partition}"],
        io_data[f"y_std"],
        io_data[f"y_mean"],
        io_data[f"y_vars"],
        keep_only_second_half=keep_only_second_half,
        outfile=outfile,
        logged_q=logged_q,
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
    keep_only_second_half=True,
    outfile=None,
    logged_q=False,
):
    """
    use trained model to make predictions
    :param model: the trained TF model
    :param x_data: [np array] numpy array of scaled and centered x_data
    :param pred_ids: [np array] the ids of the segments (same shape as x_data)
    :param pred_dates: [np array] the dates of the segments (same shape as
    x_data)
    :param keep_only_second_half: [bool] whether or not to remove the first half
    of the sequence predictions. This allows states to "warm up"
    :param y_stds:[np array] the standard deviation of the y data
    :param y_means:[np array] the means of the y data
    :param y_vars:[np array] the variable names of the y data
    :param outfile: [str] the file where the output data should be stored
    :param logged_q: [str] whether the discharge was logged in training. if True
    the exponent of the discharge will be taken in the model unscaling
    :return: out predictions
    """
    num_segs = len(np.unique(pred_ids))
    y_pred = model.predict(x_data, batch_size=num_segs)
    if keep_only_second_half:
        half_seq_len = round(y_pred.shape[1] / 2)
        y_pred = y_pred[:, half_seq_len:, :]
        pred_ids = pred_ids[:, half_seq_len:, :]
        pred_dates = pred_dates[:, half_seq_len:, :]

    y_pred_pp = prepped_array_to_df(y_pred, pred_dates, pred_ids, y_vars,)

    y_pred_pp = unscale_output(y_pred_pp, y_stds, y_means, y_vars, logged_q,)

    if outfile:
        y_pred_pp.to_feather(outfile)
    return y_pred_pp


def mean_or_std_dataset_from_np(data, data_label, var_names_label):
    """
    turn a numpy data array into a xarry dataset
    :param data: the numpy NpzFile
    :param data_label: [str] the label for the values you want to turn into the
    xarray dataset (i.e., "x_mean" or "x_std")
    :param var_names_label: [str] the label of the data you want to become the
    variable names of the xarray dataset (i.e., "x_cols")
    :return:xarray dataset
    """
    df = pd.DataFrame([data[data_label]], columns=data[var_names_label])
    # take the "mean" to drop index level. it's only one value per variable
    # so the mean is 'mean'ingless :)
    ds = df.to_xarray().mean()
    return ds


def predict_one_date_range(
    model,
    ds_x_scaled,
    train_io_data,
    seq_len,
    start_date,
    end_date,
    logged_q=False,
    keep_second_half=True,
    offset=0.5,
):
    ds_x_scaled = ds_x_scaled[train_io_data["x_cols"]]
    x_data = ds_x_scaled.sel(date=slice(start_date, end_date))
    x_batches = convert_batch_reshape(x_data, seq_len=seq_len, offset=offset)
    x_batch_ids = coord_as_reshaped_array(
        x_data, "seg_id_nat", seq_len=seq_len, offset=offset
    )
    x_batch_dates = coord_as_reshaped_array(
        x_data, "date", seq_len=seq_len, offset=offset
    )
    predictions = predict(
        model,
        x_batches,
        x_batch_ids,
        x_batch_dates,
        train_io_data["y_std"],
        train_io_data["y_mean"],
        train_io_data["y_vars"],
        keep_only_second_half=keep_second_half,
        logged_q=logged_q,
    )
    return predictions


def predict_from_arbitrary_data(
    raw_data_file,
    pred_start_date,
    pred_end_date,
    train_io_data,
    model_weights_dir,
    model_type,
    hidden_size,
    dist_matrix=None,
    flow_in_temp=False,
    logged_q=False,
):
    """

    :param raw_data_file:
    :param pred_start_date:
    :param pred_end_date:
    :param train_io_data:
    :param model_weights_dir:
    :param model_type:
    :param hidden_size:
    :param dist_matrix:
    :param flow_in_temp:
    :param logged_q:
    :return:
    """
    train_io_data = get_data_if_file(train_io_data)

    if model_type == "rgcn":
        if not dist_matrix:
            dist_matrix = train_io_data.get("dist_matrix")
        if not isinstance(dist_matrix, np.ndarray):
            raise ValueError(
                "model type is 'rgcn', but there is no" "distance matrix"
            )

    model = load_model_from_weights(
        model_type, model_weights_dir, hidden_size, dist_matrix, flow_in_temp,
    )

    ds = xr.open_zarr(raw_data_file)

    ds_x = ds[train_io_data["x_cols"]]

    x_stds = mean_or_std_dataset_from_np(train_io_data, "x_std", "x_cols")
    x_means = mean_or_std_dataset_from_np(train_io_data, "x_mean", "x_cols")

    ds_x_scaled, _, _ = scale(ds_x, std=x_stds, mean=x_means)
    seq_len = 365

    pred_start_date = datetime.datetime.strptime(pred_start_date, "%Y-%m-%d")
    inputs_start_date = pred_start_date - datetime.timedelta(183)

    # get the "middle" predictions
    middle_predictions = predict_one_date_range(
        model,
        ds_x_scaled,
        train_io_data,
        seq_len,
        inputs_start_date,
        pred_end_date,
        logged_q,
        keep_second_half=True,
        offset=0.5,
    )

    # get the "beginning" predictions
    start_dates_end = pred_start_date + datetime.timedelta(365)

    beginning_predictions = predict_one_date_range(
        model,
        ds_x_scaled,
        train_io_data,
        seq_len,
        pred_start_date,
        start_dates_end,
        logged_q,
        keep_second_half=False,
        offset=1,
    )

    # get the "end" predictions
    end_date_end = datetime.datetime.strptime(
        pred_end_date, "%Y-%m-%d"
    ) + datetime.timedelta(1)
    end_dates_start = end_date_end - datetime.timedelta(365)

    end_predictions = predict_one_date_range(
        model,
        ds_x_scaled,
        train_io_data,
        seq_len,
        end_dates_start,
        end_date_end,
        logged_q,
        keep_second_half=False,
        offset=1,
    )

    # trim beginning and end predictions
    predictions_beginning_trim = beginning_predictions[
        beginning_predictions["date"] < middle_predictions["date"].min()
    ]
    predictions_end_trim = end_predictions[
        end_predictions["date"] > middle_predictions["date"].max()
    ]

    predictions_combined = pd.concat(
        [predictions_beginning_trim, middle_predictions, predictions_end_trim]
    )
    return predictions_combined
