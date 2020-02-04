import random
from random import seed
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers
import pandas as pd
from sklearn import preprocessing

def plot_pred(pred, true):
    df = pd.DataFrame([true, pred]).T
    df.columns=['true', 'predicted']
    df.plot()
    plt.show()

def select_data(df):
    unwanted_cols = ['seg_id_nat', 'model_idx', 'date'] 
    target_col = 'seg_outflow'
    predictor_cols = [c for c in df.columns if c not in unwanted_cols and
                      c != target_col]
    return df[predictor_cols], df[target_col]


def get_df_for_rand_seg(df):
    seg_id_idx = random.randint(1, 42)
    seg_id = df['seg_id_nat'].iloc[seg_id_idx]
    df_seg = df[df['seg_id_nat'] == seg_id]
    return seg_id, df_seg


def scale(df):
    std = df.std()
    mean = df.mean()
    scaled = preprocessing.scale(df)
    return scaled, std, mean


def train_lstm(train_data, hidden_units=10, epochs=40):
    simple_lstm_model = models.Sequential([layers.LSTM(hidden_units),
                                           layers.Dense(1)])

    simple_lstm_model.compile(optimizer='adam', loss='mae')
    simple_lstm_model.fit(train_data, epochs=epochs)
    return simple_lstm_model


def separate_trn_tst(df, train_per=0.8):
    nrows = df.shape[0]
    sep_idx = int(nrows * train_per)
    df_trn = df.iloc[:sep_idx, :]
    df_tst = df.iloc[sep_idx:, :]
    return df_trn, df_tst


def read_process_data(trn_ratio=0.8):
    # read in data
    df = pd.read_feather('data/sntemp_input_output_subset.feather')
    # do for just one reach
    seg_id, df_seg = get_df_for_rand_seg(df)

    # separate trn_tst
    df_trn, df_tst = separate_trn_tst(df_seg)

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

    x_tst = scale(predictors_tst)[0]
    x_tst = np.expand_dims(x_tst, axis=1)
    y_tst = scale(target_tst)[0]
    tst = tf.data.Dataset.from_tensor_slices((x_tst, y_tst))
    return trn, x_tst, y_tst, y_trn_std, y_trn_mean

train_data, x_test, y_test, y_tr_std, y_tr_mean = read_process_data()
model = train_lstm(train_data)
preds = model.predict(x_test) 
preds = [p[0] for p in preds]


plot_pred(preds, y_test)
# plot_pred(x_trn, y_trn)
