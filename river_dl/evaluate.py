import pandas as pd
import numpy as np

from river_dl.postproc_utils import fmt_preds_obs
from river_dl.loss_functions import rmse, nse, kge


def filter_negative_preds(y_true, y_pred):
    """
    filters out negative predictions and prints a warning if there are >5% of predictions as negative
    :param y_true: [array-like] observed y values
    :param y_pred: [array-like] predicted y values
    :return: [array-like] filtered data
    """
    # print a warning if there are a lot of negatives
    n_negative = len(y_pred[y_pred < 0])
    perc_negative = n_negative / len(y_pred)
    if perc_negative > 0.05:
        print(
            f"Warning than 5% of predictions were negative {n_negative} of\
                {len(y_pred)}"
        )
    # filter out negative predictions
    y_true = np.where(y_pred < 0, np.nan, y_true)
    y_pred = np.where(y_pred < 0, np.nan, y_pred)
    return y_true, y_pred


def rmse_logged(y_true, y_pred):
    """
    compute the rmse of the logged data
    :param y_true: [array-like] observed y values
    :param y_pred: [array-like] predicted y values
    :return: [float] the rmse of the logged data
    """
    y_true, y_pred = filter_negative_preds(y_true, y_pred)
    return rmse(np.log(y_true), np.log(y_pred))


def nse_logged(y_true, y_pred):
    """
    compute the rmse of the logged data
    :param y_true: [array-like] observed y values
    :param y_pred: [array-like] predicted y values
    :return: [float] the nse of the logged data
    """
    y_true, y_pred = filter_negative_preds(y_true, y_pred)
    return nse(np.log(y_true), np.log(y_pred))


def filter_by_percentile(y_true, y_pred, percentile, less_than=True):
    """
    filter an array by a percentile of the observations. The data less than
    or greater than if `less_than=False`) will be changed to NaN
    :param y_true: [array-like] observed y values
    :param y_pred: [array-like] predicted y values
    :param percentile: [number] percentile number 0-100
    :param less_than: [bool] whether you want the data *less than* the
    percentile. If False, the data greater than the percentile will remain.
    :return: [array-like] filtered data
    """
    percentile_val = np.nanpercentile(y_true, percentile)
    if less_than:
        y_true_filt = np.where(y_true < percentile_val, y_true, np.nan)
        y_pred_filt = np.where(y_true < percentile_val, y_pred, np.nan)
    else:
        y_true_filt = np.where(y_true > percentile_val, y_true, np.nan)
        y_pred_filt = np.where(y_true > percentile_val, y_pred, np.nan)
    return y_true_filt, y_pred_filt


def percentile_metric(y_true, y_pred, metric, percentile, less_than=True):
    """
    compute an evaluation metric for a specified percentile of the observations
    :param y_true: [array-like] observed y values
    :param y_pred: [array-like] predicted y values
    :param metric: [function] metric function
    :param percentile: [number] percentile number 0-100
    :param less_than: [bool] whether you want the data *less than* the
    percentile. If False, the data greater than the percentile will remain.
    """
    y_true_filt, y_pred_filt = filter_by_percentile(
        y_true, y_pred, percentile, less_than
    )
    return metric(y_true_filt, y_pred_filt)


def calc_metrics(df):
    """
    calculate metrics (e.g., rmse and nse)
    :param df:[pd dataframe] dataframe of observations and predictions for
    one reach. dataframe must have columns "obs" and "pred"
    :return: [pd Series] various evaluation metrics (e.g., rmse and nse)
    """
    obs = df["obs"].values
    pred = df["pred"].values
    if len(obs) > 10:
        metrics = {
            "rmse": rmse(obs, pred).numpy(),
            "nse": nse(obs, pred).numpy(),
            "rmse_top10": percentile_metric(
                obs, pred, rmse, 90, less_than=False
            ).numpy(),
            "rmse_bot10": percentile_metric(
                obs, pred, rmse, 10, less_than=True
            ).numpy(),
            "rmse_logged": rmse_logged(obs, pred).numpy(),
            "nse_top10": percentile_metric(
                obs, pred, nse, 90, less_than=False
            ).numpy(),
            "nse_bot10": percentile_metric(
                obs, pred, nse, 10, less_than=True
            ).numpy(),
            "nse_logged": nse_logged(obs, pred).numpy(),
            "kge": kge(obs, pred).numpy(),
            "rmse_logged": rmse_logged(obs, pred),
            "nse_top10": percentile_metric(obs, pred, nse, 90, less_than=False),
            "nse_bot10": percentile_metric(obs, pred, nse, 10, less_than=True),
            "nse_logged": nse_logged(obs, pred),
        }

    else:
        metrics = {
            "rmse": np.nan,
            "nse": np.nan,
            "rmse_top10": np.nan,
            "rmse_bot10": np.nan,
            "rmse_logged": np.nan,
            "nse_top10": np.nan,
            "nse_bot10": np.nan,
            "nse_logged": np.nan,
            "kge": np.nan,
        }
    return pd.Series(metrics)


def overall_metrics(
    pred_file, obs_file, variable, partition, group=None, outfile=None
):
    """
    calculate metrics for a certain group (or no group at all) for a given
    partition and variable
    :param pred_file: [str] path to predictions feather file
    :param obs_file: [str] path to observations zarr file
    :param variable: [str] variable for which the metrics are being calculated
    :param partition: [str] data partition for which metrics are calculated
    :param group: [str or list] which group the metrics should be computed for.
    Currently only supports 'seg_id_nat' (segment-wise metrics), 'month'
    (month-wise metrics), ['seg_id_nat', 'month'] (metrics broken out by segment
    and month), and None (everything is left together)
    :param outfile: [str] file where the metrics should be written
    :return: [pd dataframe] the condensed metrics
    """
    data = fmt_preds_obs(pred_file, obs_file, variable)
    data.reset_index(inplace=True)
    if not group:
        metrics = calc_metrics(data)
        # need to convert to dataframe and transpose so it looks like the others
        metrics = pd.DataFrame(metrics).T
    elif group == "seg_id_nat":
        metrics = data.groupby("seg_id_nat").apply(calc_metrics).reset_index()
    elif group == "month":
        metrics = (
            data.groupby(data["date"].dt.month)
            .apply(calc_metrics)
            .reset_index()
        )
    elif group == ["seg_id_nat", "month"]:
        metrics = (
            data.groupby([data["date"].dt.month, "seg_id_nat"])
            .apply(calc_metrics)
            .reset_index()
        )
    else:
        raise ValueError("group value not valid")
    metrics["variable"] = variable
    metrics["partition"] = partition
    if outfile:
        metrics.to_csv(outfile, header=True, index=False)
    return metrics


def combined_metrics(
    obs_temp,
    obs_flow,
    pred_trn=None,
    pred_val=None,
    pred_tst=None,
    group=None,
    outfile=None,
):
    """
    calculate the metrics for flow and temp and training and test sets for a
    given grouping
    :param pred_trn: [str] path to training prediction feather file
    :param pred_val: [str] path to validation prediction feather file
    :param pred_tst: [str] path to testing prediction feather file
    :param obs_temp: [str] path to observations temperature zarr file
    :param obs_flow: [str] path to observations flow zarr file
    :param group: [str or list] which group the metrics should be computed for.
    Currently only supports 'seg_id_nat' (segment-wise metrics), 'month'
    (month-wise metrics), ['seg_id_nat', 'month'] (metrics broken out by segment
    and month), and None (everything is left together)
    :param outfile: [str] csv file where the metrics should be written
    :return: combined metrics
    """
    df_all = []
    if pred_trn:
        trn_temp = overall_metrics(pred_trn, obs_temp, "temp", "trn", group)
        trn_flow = overall_metrics(pred_trn, obs_flow, "flow", "trn", group)
        df_all.extend([trn_temp, trn_flow])
    if pred_val:
        val_temp = overall_metrics(pred_val, obs_temp, "temp", "val", group)
        val_flow = overall_metrics(pred_val, obs_flow, "flow", "val", group)
        df_all.extend([val_temp, val_flow])
    if pred_tst:
        tst_temp = overall_metrics(pred_tst, obs_temp, "temp", "tst", group)
        tst_flow = overall_metrics(pred_tst, obs_flow, "flow", "tst", group)
        df_all.extend([tst_temp, tst_flow])
    df_all = pd.concat(df_all, axis=0)
    if outfile:
        df_all.to_csv(outfile, index=False)
    return df_all
