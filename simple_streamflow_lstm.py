import random
from random import seed
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing


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
    return df_seg


def scale(df):
    std = df.std()
    mean = df.mean()
    scaled = preprocessing.scale(df)
    return scaled, std, mean

df = pd.read_feather('data/sntemp_input_output_subset.feather')
# do for just one reach
df_seg = get_df_for_rand_seg(df)

# separate the testing from the training 80/20
nrows = df_seg.shape[0]
sep_idx = int(nrows * 0.8)
df_trn = df_seg.iloc[:sep_idx, :]
df_tst = df_seg.iloc[sep_idx:, :]

# separate the predictors from the targets
predictors_trn, target_trn = select_data(df_trn)
predictors_tst, target_tst = select_data(df_tst)

x_trn, x_trn_std, x_trn_mean = scale(predictors_trn)
# add a dimension for the timesteps (just using 1)
x_trn = np.expand_dims(x_trn, axis=1)
y_trn, y_trn_std, y_trn_mean = scale(target_trn)

trn = tf.data.Dataset.from_tensor_slices((x_trn, y_trn))
trn = trn.batch(365).shuffle(365)

x_tst = scale(predictors_tst)[0]
y_tst = scale(target_tst)[0]
tst = tf.data.Dataset.from_tensor_slices((x_tst, y_tst))

simple_lstm_model = tf.keras.models.Sequential([tf.keras.layers.LSTM(10),
                                                tf.keras.layers.Dense(1)])

simple_lstm_model.compile(optimizer='adam', loss='mae')
simple_lstm_model.fit(trn, epochs=10)
