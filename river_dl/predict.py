import numpy as np

from river_dl.RGCN import RGCNModel
from river_dl.postproc_utils import prepped_array_to_df
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
    
    if partition != 'trn':
        keep_only_first_half = True
    else:
        keep_only_first_half = False
        
    preds = predict(
        model,
        io_data[f"x_{partition}"],
        io_data[f"ids_{partition}"],
        io_data[f"dates_{partition}"],
        io_data[f"y_std"],
        io_data[f"y_mean"],
        io_data[f"y_vars"],
        keep_only_first_half=keep_only_first_half,
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
    keep_only_first_half=True,
    outfile=None,
    logged_q=False,
):
    """
    use trained model to make predictions and then evaluate those predictions.
    nothing is returned but three files are saved an rmse_flow, rmse_temp, and
    predictions feather file.
    :param model: the trained TF model
    :param x_data: [np array] numpy array of scaled and centered x_data
    :param pred_ids: [np array] the ids of the segments (same shape as x_data)
    :param pred_dates: [np array] the dates of the segments (same shape as
    x_data)
    :param keep_only_first_half: [bool] whether or not to remove the first half
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
    if keep_only_first_half:
        half_seq_len = round(y_pred.shape[1]/2)
        y_pred = y_pred[:, half_seq_len:, :]
        pred_ids = pred_ids[:, half_seq_len:, :]
        pred_dates = pred_dates[:, half_seq_len:, :]
        
    y_pred_pp = prepped_array_to_df(y_pred, pred_dates, pred_ids, y_vars,)

    y_pred_pp = unscale_output(y_pred_pp, y_stds, y_means, y_vars, logged_q,)

    if outfile:
        y_pred_pp.to_feather(outfile)
    return y_pred_pp
