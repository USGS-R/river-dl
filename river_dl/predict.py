import numpy as np

from RGCN import RGCNModel
from postproc_utils import prepped_array_to_df
from rnns import LSTMModel, GRUModel
from train import get_data_if_file


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


def predict_from_weights(
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
    :param model_weights_dir: [str] directory to saved model weights
    :param io_file: [str] directory to prepped data file
    :param hidden_size: [int] the number of hidden units in model
    :param partition: [str] must be 'trn' or 'tst'; whether you want to predict
    for the train or the dev period
    :param outfile: [str] the file where the output data should be stored
    :param flow_in_temp: [bool] whether the flow should be an input into temp
    :param logged_q: [bool] whether the discharge was logged in training. if True
    the exponent of the discharge will be taken in the model unscaling
    :param half_tst: [bool] whether or not to halve the testing data so some
    can be held out
    :param model: [str] model to use either 'rgcn', 'lstm', or 'gru'
    :return:
    """
    io_data = get_data_if_file(io_data)
    if model_type == "rgcn":
        model = RGCNModel(
            hidden_size, A=io_data["dist_matrix"], flow_in_temp=flow_in_temp
        )
    elif model_type.startswith("lstm"):
        model = LSTMModel(hidden_size)
    elif model_type == "gru":
        model = GRUModel(hidden_size)

    model.load_weights(model_weights_dir)
    preds = predict(model, io_data, partition, outfile, logged_q=logged_q)
    return preds


def predict(model, io_data, partition, outfile, logged_q=False):
    """
    use trained model to make predictions and then evaluate those predictions.
    nothing is returned but three files are saved an rmse_flow, rmse_temp, and
    predictions feather file.
    :param model_file: the trained TF model
    :param io_data: [dict] dictionary or .npz file with all x_data, y_data,
    and dist matrix
    :param half_tst: [bool] whether or not to halve the testing data so some
    can be held out
    :param partition: [str] must be 'trn' or 'tst'; whether you want to predict
    for the train or the dev period
    :param outfile: [str] the file where the output data should be stored
    :param logged_q: [str] whether the discharge was logged in training. if True
    the exponent of the discharge will be taken in the model unscaling
    :return:[none]
    """
    io_data = get_data_if_file(io_data)

    if partition in ["trn", "val", "tst"]:
        pass
    else:
        raise ValueError('partition arg needs to be "trn" or "val" or "tst"')

    num_segs = len(np.unique(io_data["ids_trn"]))
    y_pred = model.predict(io_data[f"x_{partition}"], batch_size=num_segs)
    y_pred_pp = prepped_array_to_df(
        y_pred,
        io_data[f"dates_{partition}"],
        io_data[f"ids_{partition}"],
        io_data["y_vars"],
    )

    y_pred_pp = unscale_output(
        y_pred_pp,
        io_data["y_std"],
        io_data["y_mean"],
        io_data["y_vars"],
        logged_q,
    )

    y_pred_pp.to_feather(outfile)
    return y_pred_pp