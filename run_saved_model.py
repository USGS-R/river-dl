import argparse
import json
import pandas as pd
from RGCN_tf2 import RGCNModel, rmse_masked
from data_utils import read_process_data
import numpy as np


def post_process(y_pred, dates_ids):
    """
    post process y data (reshape and make into pandas DFs)
    :param y_pred:[numpy array] array of predictions [nbatch, seq_len, n_out]
    :param dates_ids:[numpy array] array of dates and seg_id's
    [nbatch, seq_len, n_out]
    :return:[pd dataframe] df with cols
    ['date', 'seg_id_nat', 'temp_degC', 'discharge_cms]
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


def predict_evaluate(trained_model, io_data, tag, num_segs, network, run_tag,
                     outdir):
    """
    use trained model to make predictions and then evaluate those predictions.
    nothing is returned but three files are saved an rmse_flow, rmse_temp, and
    predictions feather file.
    :param trained_model:[tf model] model with trained weights loaded
    :param io_data:[dict] dictionary with all the io data for x_trn, y_trn,
    y_tst, etc.
    :param tag: [str] must be 'trn' or 'dev'; whether you want to predict for
    the train or the dev period
    :param num_segs: [int] the number of segments in the data for prediction
    :param network: [str] 'full' or 'subset'; whether you are making predictions
    on the full or the subset of the network. This is only used in file naming
    :return:[none]
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
    y_pred_pp = post_process(y_pred, io_data[f'dates_ids_{data_tag}'])

    y_pred_pp = unscale_output(y_pred_pp, io_data['y_trn_obs_std'],
                               io_data['y_trn_obs_mean'])

    y_obs_pp = post_process(io_data[f'y_{data_tag}_obs'],
                            io_data[f'dates_ids_{data_tag}'])
    if tag == 'trn':
        y_obs_pp = unscale_output(y_obs_pp, io_data['y_trn_obs_std'],
                                  io_data['y_trn_obs_mean'])

    # only save the first half of the predictions to maintain a test holdout
    if tag == 'dev':
        y_pred_pp = take_first_half(y_pred_pp)
        y_obs_pp = take_first_half(y_obs_pp)

    rmse_temp = rmse_masked(y_obs_pp['temp_degC'], y_pred_pp['temp_degC'])
    rmse_flow = rmse_masked(y_obs_pp['discharge_cms'],
                            y_pred_pp['discharge_cms'])
    metrics_data = {f'rmse_temp_{network}_{tag}{run_tag}':
                        str(rmse_temp.numpy()),
                    f'rmse_flow_{network}_{tag}{run_tag}':
                        str(rmse_flow.numpy())}

    # save files
    with open(f'{outdir}{network}/{tag}_metrics{run_tag}.json', 'w') as f:
        json.dump(metrics_data, f)
    y_pred_pp.to_feather(f'{outdir}preds/{network}/'
                         f'{network}_{tag}_preds{run_tag}.feather')


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--network", help='network - "full" or "subset"',
                    choices=['full', 'subset'])
parser.add_argument("-o", "--outdir", help='directory where the output should\
                    be written')
parser.add_argument("-t", "--tag", help='tag to append to end of output files',
                    default='')
parser.add_argument("-w", "--weights_dir", help='directory where\
                    trained_weights_{network}{tag}/ is')
args = parser.parse_args()

hidden_size = 20
network = args.network
outdir = args.outdir
dist_mat = args.dist_matrix
weights_dir = args.weights_dir
run_tag = args.tag
if run_tag != '':
    run_tag = f'_{run_tag}'


if network == "full":
    subset = False
elif network == "subset":
    subset = True

data = read_process_data(subset=subset, trn_ratio=0.67, batch_offset=1,
                         dist_type=dist_mat)
num_segs = data['dist_matrix'].shape[0]
model = RGCNModel(hidden_size, 2, A=data['dist_matrix'])

model.load_weights(f'{weights_dir}/trained_weights_{network}{run_tag}/')

predict_evaluate(model, data, 'dev', num_segs, network, run_tag, outdir)
predict_evaluate(model, data, 'trn', num_segs, network, run_tag, outdir)
