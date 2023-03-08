import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from river_dl.postproc_utils import fmt_preds_obs


def filter_negative_preds(y_true, y_pred):
    """
    filters out negative predictions and prints a warning if there are >5% of predictions as negative
    :param y_true: [array-like] observed y_dataset values
    :param y_pred: [array-like] predicted y_dataset values
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

def filter_nan_preds(y_true,y_pred):
    y_pred = y_pred[~np.isnan(y_true)]
    y_true = y_true[~np.isnan(y_true)]
    return(y_true, y_pred)

def nse_eval(y_true, y_pred):
    y_true, y_pred = filter_nan_preds(y_true,y_pred)
    mean = np.mean(y_true)
    deviation = y_true - mean
    error = y_pred-y_true
    numerator = np.sum(np.square(error))
    denominator = np.sum(np.square(deviation))
    return 1 - numerator / denominator


def rmse_eval(y_true, y_pred):
    y_true, y_pred = filter_nan_preds(y_true, y_pred)
    n = len(y_true)
    sum_squared_error = np.sum(np.square(y_pred-y_true))
    rmse = np.sqrt(sum_squared_error/n)
    return rmse

def bias_eval(y_true,y_pred):
    y_true, y_pred = filter_nan_preds(y_true, y_pred)
    bias = np.mean(y_pred-y_true)
    return bias

def rmse_logged(y_true, y_pred):
    """
    compute the rmse of the logged data
    :param y_true: [array-like] observed y_dataset values
    :param y_pred: [array-like] predicted y_dataset values
    :return: [float] the rmse of the logged data
    """
    y_true, y_pred = filter_nan_preds(y_true, y_pred)
    y_true, y_pred = filter_negative_preds(y_true, y_pred)
    return rmse_eval(np.log(y_true), np.log(y_pred))


def nse_logged(y_true, y_pred):
    """
    compute the nse of the logged data
    :param y_true: [array-like] observed y_dataset values
    :param y_pred: [array-like] predicted y_dataset values
    :return: [float] the nse of the logged data
    """
    y_true, y_pred = filter_nan_preds(y_true, y_pred)
    y_true, y_pred = filter_negative_preds(y_true, y_pred)
    return nse_eval(np.log(y_true), np.log(y_pred))


def kge_eval(y_true, y_pred):
    y_true, y_pred = filter_nan_preds(y_true, y_pred)
    #Need to have > 1 observation to compute correlation.
    #This could be < 2 due to percentile filtering
    if len(y_true) > 1:
        r, _ = pearsonr(y_pred, y_true)
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        std_true = np.std(y_true)
        std_pred = np.std(y_pred)
        r_component = np.square(r - 1)
        std_component = np.square((std_pred / std_true) - 1)
        bias_component = np.square((mean_pred / mean_true) - 1)
        result = 1 - np.sqrt(r_component + std_component + bias_component)
    else:
        result = np.nan
    return result

def kge_logged(y_true, y_pred):
    """
    compute the kge of the logged data
    :param y_true: [array-like] observed y_dataset values
    :param y_pred: [array-like] predicted y_dataset values
    :return: [float] the nse of the logged data
    """
    y_true, y_pred = filter_nan_preds(y_true, y_pred)
    y_true, y_pred = filter_negative_preds(y_true, y_pred)
    return kge_eval(np.log(y_true), np.log(y_pred))

def filter_by_percentile(y_true, y_pred, percentile, less_than=True):
    """
    filter an array by a percentile of the observations. The data less than
    or greater than if `less_than=False`) will be changed to NaN
    :param y_true: [array-like] observed y_dataset values
    :param y_pred: [array-like] predicted y_dataset values
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
    :param y_true: [array-like] observed y_dataset values
    :param y_pred: [array-like] predicted y_dataset values
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
    obs, pred = filter_nan_preds(obs, pred)

    if len(obs) > 20:
        metrics = {
            "rmse": rmse_eval(obs, pred),
            "nse": nse_eval(obs, pred),
            "rmse_top10": percentile_metric(
                obs, pred, rmse_eval, 90, less_than=False
            ),
            "rmse_bot10": percentile_metric(
                obs, pred, rmse_eval, 10, less_than=True
            ),
            "rmse_logged": rmse_logged(obs, pred),
            "mean_bias": bias_eval(obs,pred),
            "mean_bias_top10":percentile_metric(
                obs, pred, bias_eval, 90, less_than=False
            ),
            "mean_bias_bot10": percentile_metric(
                obs, pred, bias_eval, 10, less_than=True
            ),
            "nse_top10": percentile_metric(
                obs, pred, nse_eval, 90, less_than=False
            ),
            "nse_bot10": percentile_metric(
                obs, pred, nse_eval, 10, less_than=True
            ),
            "nse_logged": nse_logged(obs, pred),
            "kge": kge_eval(obs, pred),
            "kge_logged": kge_logged(obs, pred),
            "kge_top10": percentile_metric(obs, pred, kge_eval, 90, less_than=False),
            "kge_bot10": percentile_metric(obs, pred, kge_eval, 10, less_than=True)
        }

    else:
        metrics = {
            "rmse": np.nan,
            "nse": np.nan,
            "rmse_top10": np.nan,
            "rmse_bot10": np.nan,
            "rmse_logged": np.nan,
            "mean_bias": np.nan,
            "mean_bias_top10": np.nan,
            "mean_bias_bot10": np.nan,
            "nse_top10": np.nan,
            "nse_bot10": np.nan,
            "nse_logged": np.nan,
            "kge": np.nan,
            "kge_logged": np.nan,
            "kge_top10": np.nan,
            "kge_bot10": np.nan
        }
    return pd.Series(metrics)


def partition_metrics(
        preds,
        obs_file,
        partition,
        spatial_idx_name="seg_id_nat",
        time_idx_name="date",
        group=None,
        id_dict=None,
        outfile=None,
        val_sites=None,
        test_sites=None,
        train_sites=None,
):
    """
    calculate metrics for a certain group (or no group at all) for a given
    partition and variable
    :param preds: [str or DataFrame] path to predictions feather file or Pandas
    DataFrame of predictions
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
    :param id_dict: [dict] dictionary of id_dict where dict keys are the id
    names and dict values are the id values. These are added as columns to the
    metrics information
    :param outfile: [str] file where the metrics should be written
    :param val_sites: [list] sites to exclude from training and test metrics
    :param test_sites: [list] sites to exclude from validation and training metrics
    :param train_sites: [list] sites to exclude from validation and test metrics
    :return: [pd dataframe] the condensed metrics
    """
    var_data = fmt_preds_obs(preds, obs_file, spatial_idx_name,
                             time_idx_name)
    var_metrics_list = []

    for data_var, data in var_data.items():
        data.reset_index(inplace=True)
        # mask out validation and test sites from trn partition
        if train_sites and partition == 'trn':
            # simply use the train sites when specified.
            data = data[data[spatial_idx_name].isin(train_sites)]
        else:
            #check if validation or testing sites are specified
            if val_sites and partition == 'trn':
                data = data[~data[spatial_idx_name].isin(val_sites)]
            if test_sites and partition == 'trn':
                data = data[~data[spatial_idx_name].isin(test_sites)]
        # mask out training and test sites from val partition
        if val_sites and partition == 'val':
            data = data[data[spatial_idx_name].isin(val_sites)]
        else:
            if test_sites and partition=='val':
                data = data[~data[spatial_idx_name].isin(test_sites)]
            if train_sites and partition=='val':
                data = data[~data[spatial_idx_name].isin(train_sites)]
        # mask out training and validation sites from val partition
        if test_sites and partition == 'tst':
            data = data[data[spatial_idx_name].isin(tst_sites)]
        else:
            if train_sites and partition=='tst':
                data = data[~data[spatial_idx_name].isin(train_sites)]
            if val_sites and partition=='tst':
                data = data[~data[spatial_idx_name].isin(val_sites)]

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
        if id_dict:
            for id_name, id_val in id_dict.items():
                metrics[id_name] = id_val
        var_metrics_list.append(metrics)
        var_metrics = pd.concat(var_metrics_list).round(6)
    if outfile:
        var_metrics.to_csv(outfile, header=True, index=False)
    return var_metrics


def combined_metrics(
    obs_file,
    pred_data=None,
    pred_trn=None,
    pred_val=None,
    pred_tst=None,
    val_sites=None,
    test_sites=None,
    train_sites=None,
    spatial_idx_name="seg_id_nat",
    time_idx_name="date",
    group=None,
    id_dict=None,
    outfile=None,
):
    """
    calculate the metrics for flow and temp and training and test sets for a
    given grouping
    :param obs_file: [str] path to observations zarr file
    :param pred_data: [dict] dict where keys are partition labels and values 
    are the corresponding prediction data file or predictions as a pandas
    dataframe. If pred_data is provided, this will be used and none of
    pred_trn, pred_val, or pred_tst will be used.
    :param pred_trn: [str or DataFrame] path to training prediction feather file
    or training predictions as pandas dataframe
    :param pred_val: [str or DataFrame] path to validation prediction feather
    file or validation predictions as pandas dataframe
    :param pred_tst: [str or DataFrame] path to testing prediction feather file
    or test predictions as pandas dataframe
    :param val_sites: [list] sites to exclude from training metrics
    :param test_sites: [list] sites to exclude from validation and training metrics
    :param spatial_idx_name: [str] name of column that is used for spatial
        index (e.g., 'seg_id_nat')
    :param time_idx_name: [str] name of column that is used for temporal index
        (usually 'time')
    :param group: [str or list] which group the metrics should be computed for.
    Currently only supports 'seg_id_nat' (segment-wise metrics), 'month'
    (month-wise metrics), ['seg_id_nat', 'month'] (metrics broken out by segment
    and month), and None (everything is left together)
    :param id_dict: [dict] dictionary of id_dict where dict keys are the id
    names and dict values are the id values. These are added as columns to the
    metrics information
    :param outfile: [str] csv file where the metrics should be written
    :return: combined metrics
    """

    if pred_data and not all(v is None for v in [pred_trn, pred_val, pred_tst]):
        print("Warning: pred_data and pred_trn/_val/ or _tst were provided.\n"
                "Only pred_data will be used")

    if not pred_data:
        pred_data = {}
        if pred_trn:
            pred_data['trn'] = pred_trn
        if pred_val:
            pred_data['val'] = pred_val
        if pred_tst:
            pred_data['tst'] = pred_tst

    if not pred_data:
        raise KeyError("No prediction data was provided")

    df_all = []
    for partition, preds in pred_data.items():
        metrics = partition_metrics(preds=preds,
                                    obs_file=obs_file,
                                    partition=partition,
                                    spatial_idx_name=spatial_idx_name,
                                    time_idx_name=time_idx_name,
                                    id_dict=id_dict,
                                    group=group,
                                    val_sites = val_sites,
                                    test_sites = test_sites,
                                    train_sites=train_sites)
        df_all.extend([metrics])

    df_all = pd.concat(df_all, axis=0)
    if outfile:
        df_all.to_csv(outfile, index=False)
    return df_all
