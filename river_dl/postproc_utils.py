import os
import json
import pandas as pd
import xarray as xr
import numpy as np
from river_dl.RGCN import RGCNModel
from river_dl.train import get_data_if_file


def prepped_array_to_df(data_array, dates, ids, col_names):
    """
    convert prepped x or y data in numpy array to pandas df
    (reshape and make into pandas DFs)
    :param data_array:[numpy array] array of x or y data [nbatch, seq_len,
    n_out]
    :param dates:[numpy array] array of dates [nbatch, seq_len, n_out]
    :param ids: [numpy array] array of seg_ids [nbatch, seq_len, n_out]
    :return:[pd dataframe] df with cols
    ['date', 'seg_id_nat', 'temp_c', 'discharge_cms]
    """
    data_array = np.reshape(data_array,
                            [data_array.shape[0]*data_array.shape[1],
                             data_array.shape[2]])

    dates = np.reshape(dates, [dates.shape[0]*dates.shape[1], dates.shape[2]])
    ids = np.reshape(ids, [ids.shape[0]*ids.shape[1], ids.shape[2]])
    df_preds = pd.DataFrame(data_array, columns=col_names)
    df_dates = pd.DataFrame(dates, columns=['date'])
    df_ids = pd.DataFrame(ids, columns=['seg_id_nat'])
    df = pd.concat([df_dates, df_ids, df_preds], axis=1)
    return df


def take_first_half(df):
    """
    filter out the second half of the dates in the predictions. this is to
    retain a "test" set of the i/o data for evaluation
    :param df:[pd dataframe] df of predictions or observations cols ['date',
    'seg_id_nat', 'temp_c', 'discharge_cms']
    :return: [pd dataframe] same cols as input, but only the first have of dates
    """
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    unique_dates = df.index.unique()
    halfway_date = unique_dates[int(len(unique_dates)/2)]
    df_first_half = df.loc[:halfway_date]
    df_first_half.reset_index(inplace=True)
    return df_first_half


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
        y_scl['seg_outflow'] = np.exp(y_scl['seg_outflow'])
    return y_scl


def rmse_masked(y_true, y_pred):
    """
    Compute cost as RMSE with masking (the tf.where call replaces pred_s-y_s
    with 0 when y_s is nan; num_y_s is a count of just those non-nan
    observations) so we're only looking at predictions with corresponding
    observations available
    (credit: @aappling-usgs)
    :param y_true: [array-like] observed y values
    :param y_pred: [array-like] predicted y values
    :return: rmse (one value for each training sample)
    """
    # count the number of non-nans
    num_y_true = np.sum(~np.isnan(y_true))
    zero_or_error = np.where(np.isnan(y_true),
                             0,
                             y_pred - y_true)
    sum_squared_errors = np.sum(zero_or_error ** 2)
    rmse_loss = np.sqrt(sum_squared_errors / num_y_true)
    return rmse_loss


def nse(y_true, y_pred):
    """
    compute the nash-sutcliffe model efficiency coefficient
    :param y_true: [array-like] observed y values
    :param y_pred: [array-like] predicted y values
    :return: [float] the nash-sutcliffe efficiency coefficient
    """
    q_mean = np.nanmean(y_true)
    numerator = np.nansum((y_true-y_pred)**2)
    denominator = np.nansum((y_true - q_mean)**2)
    return 1 - (numerator/denominator)


def filter_negative_preds(y_true, y_pred)
    # print a warning if there are a lot of negatives
    n_negative = len(y_pred[y_pred > 0])
    if n_negative/len(y_pred) > 0.05:
        raise Warning('More than 5% of predictions were negative')
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
    return rmse_masked(np.log(y_true), np.log(y_pred))


def nse_logged(y_true, y_pred):
    """
    compute the rmse of the logged data
    :param y_true: [array-like] observed y values
    :param y_pred: [array-like] predicted y values
    :return: [float] the rmse of the logged data
    """
    y_true, y_pred = filter_negative_preds(y_true, y_pred)
    return nse(np.log(y_true), np.log(y_pred))


def filter_by_percentile(y_true, y_pred, percentile, less_than=True):
    """
    filter an array by a percentile. The data less than (or greater than if
    `less_than=False`) will be changed to NaN
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


def percentile_metric(y_true, y_pred, metric, percentile, less_than=False):
    """
    compute the rmse of the top 10 percent of data 
    :param y_true: [array-like] observed y values
    :param y_pred: [array-like] predicted y values
    :param metric: [function] metric function
    :param percentile: [number] percentile number 0-100
    :param less_than: [bool] whether you want the data *less than* the
    percentile. If False, the data greater than the percentile will remain.
    """
    y_true_filt, y_pred_filt = filter_by_percentile(y_true, y_pred, percentile,
                                                    less_than)
    return metric(y_true_filt, y_pred_filt)


def predict_from_file(model_weights_dir, io_data, hidden_size, partition,
                      outfile, flow_in_temp=False, logged_q=False,
                      half_tst=False):
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
    :return:
    """
    io_data = get_data_if_file(io_data)
    model = RGCNModel(hidden_size, A=io_data['dist_matrix'],
                      flow_in_temp=flow_in_temp)

    model.load_weights(model_weights_dir)
    preds = predict(model, io_data, partition, outfile, logged_q=logged_q,
                    half_tst=half_tst)
    return preds


def predict(model, io_data, partition, outfile, logged_q=False, half_tst=False):
    """
    use trained model to make predictions and then evaluate those predictions.
    nothing is returned but three files are saved an rmse_flow, rmse_temp, and
    predictions feather file.
    :param model: the trained TF model
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
    dist_matrix = io_data['dist_matrix']

    # evaluate training
    if partition == 'trn' or partition == 'tst':
        pass
    else:
        raise ValueError('partition arg needs to be "trn" or "tst"')

    num_segs = dist_matrix.shape[0]
    y_pred = model.predict(io_data[f'x_{partition}'], batch_size=num_segs)
    y_pred_pp = prepped_array_to_df(y_pred, io_data[f'dates_{partition}'],
                                    io_data[f'ids_{partition}'],
                                    io_data['y_vars'])

    y_pred_pp = unscale_output(y_pred_pp, io_data['y_obs_trn_std'],
                               io_data['y_obs_trn_mean'], io_data['y_vars'],
                               logged_q)

    if half_tst and partition == 'tst':
        y_pred_pp = take_first_half(y_pred_pp)

    y_pred_pp.to_feather(outfile)
    return y_pred_pp


def get_var_names(variable):
    """
    get the long variable names from 'flow' or 'temp'
    :param variable: [str] either 'flow' or 'temp'
    :return: [str] long variable names
    """
    if variable == 'flow':
        obs_var = 'discharge_cms'
        seg_var = 'seg_outflow'
    elif variable == 'temp':
        obs_var = 'temp_c'
        seg_var = 'seg_tave_water'
    else:
        raise ValueError('variable param must be "flow" or "temp"')
    return obs_var, seg_var


def load_if_not_df(pred_data):
    print("pred_data:", pred_data)
    if isinstance(pred_data, str):
        return(pd.read_feather(pred_data))
    else:
        return pred_data


def fmt_preds_obs(pred_data, obs_file, variable):
    """
    combine predictions and observations in one dataframe
    :param pred_data:[str] filepath to the predictions file
    :param obs_file:[str] filepath to the observations file
    :param variable: [str] either 'flow' or 'temp'
    """
    obs_var, seg_var = get_var_names(variable)
    pred_data = load_if_not_df(pred_data)
    if {'date', 'seg_id_nat'}.issubset(pred_data.columns):
        pred_data.set_index(['date', 'seg_id_nat'], inplace=True)
    obs = xr.open_zarr(obs_file).to_dataframe()
    obs_cln = obs[[obs_var]]
    obs_cln.columns = ['obs']
    preds = pred_data[[seg_var]]
    preds.columns = ['pred']
    combined = preds.join(obs_cln)
    return combined


def calc_metrics(df):
    """
    calculate metrics (rmse and nse) on one reach
    :param df:[pd dataframe] dataframe of observations and predictions for
    one reach
    :return: [pd Series] the rmse and nse for that one reach
    """
    obs = df['obs'].values
    pred = df['pred'].values
    if len(obs) > 10:
        metrics = {
                'rmse': rmse_masked(obs, pred),
                'nse': nse(obs, pred),
                'rmse_top10': percentile_metric(obs, pred, rmse_masked, 90,
                                                less_than=False),
                'rmse_bot10': percentile_metric(obs, pred, rmse_masked, 10,
                                                less_than=True),
                'rmse_logged': rmse_logged(obs, pred),
                'nse_top10': percentile_metric(obs, pred, nse, 90,
                                               less_than=False),
                'nse_bot10': percentile_metric(obs, pred, nse, 10,
                                               less_than=True),
                'nse_logged': nse_logged(obs, pred)
                }

    else:
        metrics = {
                'rmse': np.nan,
                'nse': np.nan,
                'rmse_top10': np.nan,
                'rmse_bot10': np.nan,
                'rmse_logged': np.nan,
                'nse_top10': np.nan,
                'nse_bot10': np.nan,
                'nse_logged': np.nan
                }
    return pd.Series(metrics)


def overall_metrics(pred_data, obs_file, variable, partition, outfile=None):
    """
    calculate overall metrics
    :param pred_data: [str] path to predictions feather file
    :param obs_file: [str] path to observations csv file
    :param outfile: [str] file where the metrics should be written
    :param variable: [str] either 'flow' or 'temp'
    :param partition: [str] either 'trn' or 'temp'
    :return: [pd Series] the overall metrics
    """
    print("variable: ", variable)
    data = fmt_preds_obs(pred_data, obs_file, variable)
    metrics = calc_metrics(data)
    metrics['variable'] = variable
    metrics['partition'] = partition
    if outfile:
        metrics.to_csv(outfile, header=False)
    return metrics


def reach_specific_metrics(pred_data, obs_file, variable, partition,
                           outfile=None):
    """
    calculate reach-specific metrics
    :param pred_data: [str] path to predictions feather file
    :param obs_file: [str] path to observations csv file
    :param outfile: [str] file where the metrics should be written
    :param variable: [str] either 'flow' or 'temp'
    :return: [pd DataFrame] the reach-specific metrics
    """
    data = fmt_preds_obs(pred_data, obs_file, variable)
    reach_metrics = data.groupby('seg_id_nat').apply(
            calc_metrics).reset_index()
    reach_metrics['variable'] = variable
    reach_metrics['partition'] = partition
    if outfile:
        reach_metrics.to_feather(outfile)
    return reach_metrics


def combined_reach_specific(pred_trn, pred_tst, obs_temp, obs_flow,
                            outfile=None):
    trn_temp = reach_specific_metrics(pred_trn, obs_temp, 'temp', 'trn')
    trn_flow = reach_specific_metrics(pred_trn, obs_flow, 'flow', 'trn')
    tst_temp = reach_specific_metrics(pred_tst, obs_temp, 'temp', 'tst')
    tst_flow = reach_specific_metrics(pred_tst, obs_flow, 'flow', 'tst')
    df_all = [trn_temp, tst_temp, trn_flow, tst_flow]
    df_all = pd.concat(df_all, axis=0)
    if outfile:
        df_all.to_csv(outfile, index=False)
    return df_all


def combine_overall(pred_trn, pred_tst, obs_temp, obs_flow, outfile=None):
    trn_temp = overall_metrics(pred_trn, obs_temp, 'temp', 'trn')
    tst_temp = overall_metrics(pred_tst, obs_temp, 'temp', 'tst')
    trn_flow = overall_metrics(pred_trn, obs_flow, 'flow', 'trn')
    tst_flow = overall_metrics(pred_tst, obs_flow, 'flow', 'tst')
    df_all = [trn_temp, tst_temp, trn_flow, tst_flow]
    df_all = pd.concat(df_all, axis=1).T
    if outfile:
        df_all.to_csv(outfile)
    return df_all


def combine_csvs(csv_files, outfile):
    dfs = [pd.read_csv(f, index_col=0, header=None).T for f in csv_files]
    df_all = pd.concat(dfs, axis=0)
    df_all.to_csv(outfile, index=False)
    return df_all
