import json
import pandas as pd
import numpy as np
from RGCN import rmse_masked


def post_process(y_pred, dates, ids):
    """
    post process y data (reshape and make into pandas DFs)
    :param y_pred:[numpy array] array of predictions [nbatch, seq_len, n_out]
    :param dates:[numpy array] array of dates [nbatch, seq_len, n_out]
    :param ids: [numpy array] array of seg_ids [nbatch, seq_len, n_out]
    :return:[pd dataframe] df with cols
    ['date', 'seg_id_nat', 'temp_degC', 'discharge_cms]
    """
    y_pred = np.reshape(y_pred, [y_pred.shape[0]*y_pred.shape[1],
                                 y_pred.shape[2]])

    dates = np.reshape(dates, [dates.shape[0]*dates.shape[1], dates.shape[2]])
    ids = np.reshape(ids, [ids.shape[0]*ids.shape[1], ids.shape[2]])
    df_preds = pd.DataFrame(y_pred, columns=['temp_degC', 'discharge_cms'])
    df_dates = pd.DataFrame(dates, columns=['date'])
    df_ids = pd.DataFrame(ids, columns=['seg_id_nat'])
    df = pd.concat([df_dates, df_ids, df_preds], axis=1)
    return df


def take_first_half(df):
    """
    filter out the second half of the dates in the predictions. this is to
    retain a "test" set of the i/o data for evaluation
    :param df:[pd dataframe] df of predictions or observations cols ['date',
    'seg_id_nat', 'temp_degC', 'discharge_cms']
    :return: [pd dataframe] same cols as input, but only the first have of dates
    """
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    unique_dates = df.index.unique()
    halfway_date = unique_dates[int(len(unique_dates)/2)]
    df_first_half = df.loc[:halfway_date]
    df_first_half.reset_index(inplace=True)
    return df_first_half


def unscale_output(y_scl, y_std, y_mean):
    """
    unscale output data given a standard deviation and a mean value for the
    outputs
    :param y_scl: [pd dataframe] scaled output data (predicted or observed)
    :param y_std:[numpy array] array of standard deviation of variables [n_out]
    :param y_mean:[numpy array] array of variable means [n_out]
    :return:
    """
    data_cols = ['temp_degC', 'discharge_cms']
    yscl_data = y_scl[data_cols]
    y_unscaled_data = (yscl_data * y_std) + y_mean
    y_scl[data_cols] = y_unscaled_data
    return y_scl


def rmse_masked(y_true, y_pred):
    """
    Compute cost as RMSE with masking (the tf.where call replaces pred_s-y_s
    with 0 when y_s is nan; num_y_s is a count of just those non-nan
    observations) so we're only looking at predictions with corresponding
    observations available
    (credit: @aappling-usgs)
    :param data: [tensor] true (observed) y values. these may have nans and 
    sample weights
    :param y_pred: [tensor] predicted y values
    :return: rmse (one value for each training sample)
    """

    # count the number of non-nans
    num_y_true = np.sum(np.isnan(y_true))
    print (num_y_true)
    zero_or_error = np.where(np.isnan(y_true),
                             0,
                             y_pred - y_true)
    print(zero_or_error.shape)
    sum_squared_errors = np.sum(zero_or_error ** 2)
    print(sum_squared_errors.shape)
    rmse_loss = np.sqrt(sum_squared_errors / num_y_true)
    print(rmse_loss)
    return rmse_loss



def predict_evaluate(trained_model, io_data, tag, num_segs, run_tag, outdir):
    """
    use trained model to make predictions and then evaluate those predictions.
    nothing is returned but three files are saved an rmse_flow, rmse_temp, and
    predictions feather file.
    :param trained_model:[tf model] model with trained weights loaded
    :param io_data:[dict] dictionary with all the io data for x_trn, y_trn,
    y_tst, etc.
    :param tag: [str] must be 'trn' or 'tst'; whether you want to predict for
    the train or the dev period
    :param num_segs: [int] the number of segments in the data for prediction
    :return:[none]
    """
    # evaluate training
    if tag == 'trn' or tag == 'tst':
        pass
    else:
        raise ValueError('tag arg needs to be "trn" or "tst"')

    y_pred = trained_model.predict(io_data[f'x_{tag}'],
                                   batch_size=num_segs)
    y_pred_pp = post_process(y_pred, io_data[f'dates_{tag}'],
                             io_data[f'ids_{tag}'])

    print(y_pred_pp)
    y_pred_pp = unscale_output(y_pred_pp, io_data['y_trn_obs_std'],
                               io_data['y_trn_obs_mean'])
    print(y_pred_pp)

    y_obs_pp = post_process(io_data[f'y_obs_{tag}'],
                            io_data[f'dates_{tag}'],
                            io_data[f'ids_{tag}'])
    if tag == 'trn':
        y_obs_pp = unscale_output(y_obs_pp, io_data['y_trn_obs_std'],
                                  io_data['y_trn_obs_mean'])


    rmse_temp = rmse_masked(y_obs_pp['temp_degC'].values,
                            y_pred_pp['temp_degC'].values)
    rmse_flow = rmse_masked(y_obs_pp['discharge_cms'].values,
                            y_pred_pp['discharge_cms'].values)
    metrics_data = {f'rmse_temp_{tag}{run_tag}': str(rmse_temp),
                    f'rmse_flow_{tag}{run_tag}': str(rmse_flow)}

    # save files
    with open(f'{outdir}{tag}_metrics{run_tag}.json', 'w') as f:
        json.dump(metrics_data, f)
    y_pred_pp.to_feather(f'{outdir}{tag}_preds{run_tag}.feather')
