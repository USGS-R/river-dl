import os
import json
import pandas as pd
import numpy as np
from RGCN import RGCNModel
from train import get_data_if_file


def prepped_array_to_df(y_pred, dates, ids, col_names):
    """
    post process y data (reshape and make into pandas DFs)
    :param y_pred:[numpy array] array of predictions [nbatch, seq_len, n_out]
    :param dates:[numpy array] array of dates [nbatch, seq_len, n_out]
    :param ids: [numpy array] array of seg_ids [nbatch, seq_len, n_out]
    :return:[pd dataframe] df with cols
    ['date', 'seg_id_nat', 'temp_c', 'discharge_cms]
    """
    y_pred = np.reshape(y_pred, [y_pred.shape[0]*y_pred.shape[1],
                                 y_pred.shape[2]])

    dates = np.reshape(dates, [dates.shape[0]*dates.shape[1], dates.shape[2]])
    ids = np.reshape(ids, [ids.shape[0]*ids.shape[1], ids.shape[2]])
    df_preds = pd.DataFrame(y_pred, columns=col_names)
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
    :param y_true: [tensor] observed y values
    :param y_pred: [tensor] predicted y values
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
    :param y_true:
    :param y_pred:
    :return:
    """
    q_mean = np.nanmean(y_true)
    numerator = np.nansum((y_true-y_pred)**2)
    denominator = np.nansum((y_true - q_mean)**2)
    return 1 - (numerator/denominator)


def predict(model_weight_dir, io_data, hidden_size, partition, outfile,
        logged_q=False, half_tst=False):
    """
    use trained model to make predictions and then evaluate those predictions.
    nothing is returned but three files are saved an rmse_flow, rmse_temp, and
    predictions feather file.
    :param model_weight_dir:
    :param x_data: [dict] dictionary or .npz file with all the x_data
    :param y_data: [dict] dictionary or .npz file with all the y_data
    :param dist_matrix: [dict] dictionary or .npz file with all the dist_matrix
    :param hidden_size: [int] the number of hidden units in model
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

    out_size = len(io_data['y_vars'])
    model = RGCNModel(hidden_size, out_size, A=io_data['dist_matrix'])

    model.load_weights(os.path.join(model_weight_dir))

    # evaluate training
    if partition == 'trn' or partition == 'tst':
        pass
    else:
        raise ValueError('partition arg needs to be "trn" or "tst"')

    num_segs = io_data['dist_matrix'].shape[0]
    y_pred = model.predict(io_data[f'x_{partition}'], batch_size=num_segs)
    y_pred_pp = prepped_array_to_df(y_pred, io_data[f'dates_{partition}'],
                                    io_data[f'ids_{partition}'],
                                    io_data['y_vars'])

    y_pred_pp = unscale_output(y_pred_pp, io_data['y_trn_obs_std'],
                               io_data['y_trn_obs_mean'], io_data['y_vars'],
                               logged_q)

    if half_tst and partition == 'tst':
        y_pred_pp = take_first_half(y_pred_pp)

    y_pred_pp.to_feather(outfile)


def get_var_names(variable):
    """

    :param variable: [str] either 'flow' or 'temp'
    :return:
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


def fmt_preds_obs(pred_file, obs_file, variable):
    """
    combine predictions and observations in one dataframe
    :param pred_file:[str] filepath to the predictions file
    :param obs_file:[str] filepath to the observations file
    :param variable: [str] either 'flow' or 'temp'
    """
    obs_var, seg_var = get_var_names(variable)
    pred_data = pd.read_feather(pred_file)
    pred_data.set_index(['date', 'seg_id_nat'], inplace=True)
    obs = pd.read_csv(obs_file, parse_dates=['date'],
                      infer_datetime_format=True,
                      index_col=['date', 'seg_id_nat'])
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
    if df['obs'].count() > 10:
        reach_rmse = rmse_masked(df['obs'], df['pred'])
        reach_nse = nse(df['obs'].values, df['pred'].values)
        return pd.Series(dict(rmse=reach_rmse, nse=reach_nse))
    else:
        return pd.Series(dict(rmse=np.nan, nse=np.nan))



def overall_metrics(pred_file, obs_file, outfile, variable):
    """
    calculate overall metrics 
    :param pred_file: [str] path to predictions feather file
    :param obs_file: [str] path to observations csv file
    :param outfile: [str] file where the metrics should be written
    :param variable: [str] either 'flow' or 'temp'
    :return: [pd Series] the overall metrics
    """
    data = fmt_preds_obs(pred_file, obs_file, variable)
    metrics = calc_metrics(data)
    metrics.to_csv(outfile)
    return metrics


def reach_specific_metrics(pred_file, obs_file, outfile, variable):
    """
    calculate reach-specific metrics 
    :param pred_file: [str] path to predictions feather file
    :param obs_file: [str] path to observations csv file
    :param outfile: [str] file where the metrics should be written
    :param variable: [str] either 'flow' or 'temp'
    :return: [pd DataFrame] the reach-specific metrics
    """
    data = fmt_preds_obs(pred_file, obs_file, variable)
    reach_metrics = data.groupby('seg_id_nat').apply(
            calc_metrics).reset_index()
    reach_metrics.to_feather(outfile)
    return reach_metrics
