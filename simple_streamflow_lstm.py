import random
from random import seed
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers
import pandas as pd
from sklearn import preprocessing, metrics

def plot_pred(pred, true):
    df = pd.DataFrame([true, pred]).T
    df.columns=['true', 'predicted']
    ax = df.plot()
    ax.set_ylabel('flow [cfs]')
    ax.set_xlabel('time (daily)')
    plt.show()

def select_data(df):
    unwanted_cols = None
    # unwanted_cols = ['seg_id_nat', 'model_idx', 'date', 'seg_upstream_inflow', 'seginc_gwflow', 'seg_width'] 
    # wanted_cols=None
    wanted_cols = ['seg_tave_air', 'seg_rain']#, 'seg_upstream_inflow']
    target_col = 'seg_outflow'
    if unwanted_cols and not wanted_cols:
        predictor_cols = [c for c in df.columns if c not in unwanted_cols and
                          c != target_col]
    elif wanted_cols:
        predictor_cols = [c for c in df.columns if c in wanted_cols and
                          c != target_col]


    return df[predictor_cols], df[target_col]


def get_df_for_rand_seg(df):
    seed(34)
    seg_id_idx = random.randint(1, 42)
    seg_id = df['seg_id_nat'].iloc[seg_id_idx]
    df_seg = df[df['seg_id_nat'] == seg_id]
    print (seg_id)
    return seg_id, df_seg


def scale(df, std=None, mean=None):
    if not isinstance(std, pd.Series) and not isinstance(mean, pd.Series):
        std = df.std()
        mean = df.mean()
    scaled = (df - mean)/std
    scaled = scaled.fillna(0)
    return scaled, std, mean


def train_lstm(train_data, hidden_units=10, epochs=40):
    simple_lstm_model = models.Sequential([layers.LSTM(hidden_units),
                                           layers.Dense(1)])

    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    simple_lstm_model.compile(optimizer=opt, loss='mae')
    tensorboard_cbk = tf.keras.callbacks.TensorBoard(log_dir='output\\simple_lstm')
    tf.keras.utils.plot_model(simple_lstm_model,
                              'output/simple_lstm_model.png',
                              show_shapes=True)
    simple_lstm_model.fit(train_data, epochs=epochs,
                          callbacks=[tensorboard_cbk])
    return simple_lstm_model


def separate_trn_tst(df, train_per=0.8):
    nrows = df.shape[0]
    sep_idx = int(nrows * train_per)
    df_trn = df.iloc[:sep_idx, :]
    df_tst = df.iloc[sep_idx:, :]
    return df_trn, df_tst


def unscale_data(df, std, mean):
    return (df * std) + mean


def get_format_preds(model, x, unscale=False, y_std=None, y_mean=None):
    preds = model.predict(x) 
    preds = pd.Series([p[0] for p in preds])
    if unscale:
        return unscale_data(preds, y_std, y_mean)
    else:
        return preds


def read_process_data(trn_ratio=0.8):
    # read in data
    df = pd.read_feather('data/sntemp_input_output_subset.feather')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['seg_id_nat', 'date'])
    seg_id, df = get_df_for_rand_seg(df)

    # separate trn_tst
    df_trn, df_tst = separate_trn_tst(df)

    # separate the predictors from the targets
    predictors_trn, target_trn = select_data(df_trn)
    predictors_tst, target_tst = select_data(df_tst)

    # scale the data
    x_trn, x_trn_std, x_trn_mean = scale(predictors_trn)
    # add a dimension for the timesteps (just using 1)
    x_trn = np.expand_dims(x_trn, axis=1)
    y_trn, y_trn_std, y_trn_mean = scale(target_trn)

    trn = tf.data.Dataset.from_tensor_slices((x_trn, y_trn))
    trn = trn.batch(365).shuffle(365)

    x_tst = scale(predictors_tst, x_trn_std, x_trn_mean)[0]
    x_tst = np.expand_dims(x_tst, axis=1)
    y_tst = target_tst
    tst = tf.data.Dataset.from_tensor_slices((x_tst, y_tst))
    return trn, x_trn, y_trn, x_tst, y_tst, y_trn_std, y_trn_mean

train_data, x_trn, y_trn, x_test, y_test, y_tr_std, y_tr_mean = read_process_data()
model = train_lstm(train_data, hidden_units=20)

# test data
tst_preds = get_format_preds(model, x_test, True, y_tr_std, y_tr_mean)
plot_pred(tst_preds, y_test.reset_index(drop=True))
print('rmse test: ', metrics.mean_squared_error(y_test.values, tst_preds.values))

# train data
trn_preds = get_format_preds(model, x_trn, True, y_tr_std, y_tr_mean)
y_trn_unscaled = unscale_data(y_trn, y_tr_std, y_tr_mean)
plot_pred(trn_preds, y_trn_unscaled.reset_index(drop=True))
print('rmse train: ', metrics.mean_squared_error(y_trn_unscaled.values, trn_preds.values))
