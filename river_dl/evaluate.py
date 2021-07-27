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
    pred_file,obs_file, partition, spatial_idx_name, time_idx_name, group=None, outfile=None
):
    """
    calculate metrics for a certain group (or no group at all) for a given
    partition and variable
    :param pred_file: [str] path to predictions feather file
    :param obs_file: [str] path to observations zarr file
    :param partition: [str] data partition for which metrics are calculated
    :param spatial_idx_name: [str] name of column that is used for spatial
        index (e.g., 'seg_id_nat')
    :param time_idx_name: [str] name of column that is used for temporal index
        (usually 'time')
    :param group: [str or list] which group the metrics should be computed for.
    Currently only supports 'seg_id_nat' (segment-wise metrics), 'month'
    (month-wise metrics), ['seg_id_nat', 'month'] (metrics broken out by segment
    and month), and None (everything is left together)
    :param outfile: [str] file where the metrics should be written
    :return: [pd dataframe] the condensed metrics
    """
    var_data = fmt_preds_obs(pred_file, obs_file, spatial_idx_name,
                             time_idx_name)
    var_metrics_list = []

    for data_var, data in var_data.items():
        data.reset_index(inplace=True)
        if not group:
            metrics = calc_metrics(data)
            # need to convert to dataframe and transpose so it looks like the
            # others
            metrics = pd.DataFrame(metrics).T
        elif group == "seg_id_nat":
            metrics = data.groupby(spatial_idx_name).apply(calc_metrics).reset_index()
        elif group == "month":
            metrics = (
            data.groupby(
            data[time_idx_name].dt.month)
            .apply(calc_metrics)
            .reset_index()
            )
        elif group == ["seg_id_nat", "month"]:
            metrics = (
            data.groupby(
            [data[time_idx_name].dt.month,
            spatial_idx_name])
            .apply(calc_metrics)
            .reset_index()
            )
        else:
            raise ValueError("group value not valid")

        metrics["variable"] = data_var
        metrics["partition"] = partition
        var_metrics_list.append(metrics)
        var_metrics = pd.concat(var_metrics_list)
    if outfile:
        var_metrics.to_csv(outfile, header=True, index=False)
    return var_metrics


def combined_metrics(
    obs_file,
    pred_trn=None,
    pred_val=None,
    pred_tst=None,
    spatial_idx_name="seg_id_nat",
    time_idx_name="date",
    group=None,
    outfile=None,
):
    """
    calculate the metrics for flow and temp and training and test sets for a
    given grouping
    :param obs_file: [str] path to observations zarr file
    :param pred_trn: [str] path to training prediction feather file
    :param pred_val: [str] path to validation prediction feather file
    :param pred_tst: [str] path to testing prediction feather file
    :param spatial_idx_name: [str] name of column that is used for spatial
        index (e.g., 'seg_id_nat')
    :param time_idx_name: [str] name of column that is used for temporal index
        (usually 'time')
    :param group: [str or list] which group the metrics should be computed for.
    Currently only supports 'seg_id_nat' (segment-wise metrics), 'month'
    (month-wise metrics), ['seg_id_nat', 'month'] (metrics broken out by segment
    and month), and None (everything is left together)
    :param outfile: [str] csv file where the metrics should be written
    :return: combined metrics
    """
    df_all = []
    if pred_trn:
        trn_metrics = partition_metrics(pred_file=pred_trn,
                                        obs_file=obs_file,
                                        partition="trn",
                                        spatial_idx_name=spatial_idx_name,
                                        time_idx_name=time_idx_name,
                                        group=group)
        df_all.extend([trn_metrics])
    if pred_val:
        val_metrics = partition_metrics(pred_file=pred_val,
                                        obs_file=obs_file,
                                        partition="val",
                                        spatial_idx_name=spatial_idx_name,
                                        time_idx_name=time_idx_name,
                                        group=group)
    df_all.extend([val_metrics])
    if pred_tst:
        tst_metrics = partition_metrics(pred_file=pred_tst,
                                        obs_file=obs_file,
                                        partition="tst",
                                        spatial_idx_name=spatial_idx_name,
                                        time_idx_name=time_idx_name,
                                        group=group)
        df_all.extend([tst_metrics])
    df_all = pd.concat(df_all, axis=0)
    if outfile:
        df_all.to_csv(outfile, index=False)
    return df_all
