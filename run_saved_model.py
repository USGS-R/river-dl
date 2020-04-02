import tensorflow as tf
import pandas as pd
from RGCN_tf2 import RGCNModel, rmse_masked
from data_utils import read_process_data
import numpy as np


def post_process(y_pred, dates_ids):
    """
    post process y data (reshape and make into pandas DFs)
    :param y_pred:
    :param dates_ids:
    :param y_std:
    :param y_mean:
    :return:
    """
    y_pred = np.reshape(y_pred, [y_pred.shape[0]*y_pred.shape[1],
                                 y_pred.shape[2]])

    dates_ids = np.reshape(dates_ids, [dates_ids.shape[0]*dates_ids.shape[1],
                                       dates_ids.shape[2]])
    df_preds = pd.DataFrame(y_pred, columns=['temp_degC', 'discharge_cms'])
    df_dates = pd.DataFrame(dates_ids, columns=['date', 'seg_id_nat'])
    df = pd.concat([df_dates, df_preds], axis=1)
    return df


def take_first_half(df):
    """
    filter out the second half of the dates in the predictions
    :param df:
    :return:
    """
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    unique_dates = df.index.unique()
    halfway_date = unique_dates[int(len(unique_dates)/2)]
    df_first_half = df.loc[:halfway_date]
    df_first_half.reset_index(inplace=True)
    return df_first_half


def unscale_predictions(y_scl, y_std, y_mean):
    data_cols = ['temp_degC', 'discharge_cms']
    yscl_data = y_scl[data_cols]
    y_unscaled_data = (yscl_data * y_std) + y_mean
    y_scl[data_cols] = y_unscaled_data
    return y_scl


def predict_evaluate(trained_model, io_data, tag, num_segs):
    """

    :param trained_model:
    :param io_data:
    :param tag: [str] should be 'trn' or 'dev'
    :param num_segs:
    :return:
    """
    # evaluate training
    if tag == 'trn':
        data_tag = 'trn'
    elif tag == 'dev':
        data_tag = 'tst'
    else:
        raise ValueError('tag arg needs to be "trn" or "dev"')

    y_pred = trained_model.predict(io_data[f'x_{data_tag}'],
                                   batch_size=num_segs)
    y_pred_pp = post_process(y_pred, io_data['dates_ids_tst'])

    y_pred_pp = unscale_predictions(y_pred_pp, io_data['y_trn_obs_std'],
                                    io_data['y_trn_obs_mean'])

    y_obs_pp = post_process(io_data['y_tst_obs'], io_data['dates_ids_tst'])

    # only save the first half of the predictions to maintain a test holdout
    if tag == 'dev':
        y_pred_pp = take_first_half(y_pred_pp)
        y_obs_pp = take_first_half(y_obs_pp)

    rmse_temp = rmse_masked(y_obs_pp['temp_degC'], y_pred_pp['temp_degC'])
    rmse_flow = rmse_masked(y_obs_pp['discharge_cms'],
                            y_pred_pp['discharge_cms'])

    # save files
    with open(f'data/out/{tag}_rmse_flow.txt', 'w') as f:
        f.write(str(rmse_flow.numpy()))
    with open(f'data/out/{tag}_rmse_temp.txt', 'w') as f:
        f.write(str(rmse_temp.numpy()))
    y_pred_pp.to_feather(f'data/out/{tag}_preds.feather')


hidden_size = 20

data = read_process_data(subset=True, trn_ratio=0.67, batch_offset=1,
                         dist_type='upstream')
print('read in data')
num_segs = data['dist_matrix'].shape[0]
print(num_segs)
model = RGCNModel(hidden_size, 2, A=data['dist_matrix'])

model.load_weights('data/out/trained_weights/')

predict_evaluate(model, data, 'dev', num_segs)
