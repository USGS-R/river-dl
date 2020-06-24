import os
from prefect import task, Flow
import tensorflow as tf
from tensorflow.keras import models, layers
from preproc_utils import prep_data
from postproc_utils import predict


@task()
def train_lstm(io_data):
    hidden_units = 20
    lstm_model = models.Sequential([layers.LSTM(hidden_units,
                                                return_sequences=True),
                                    layers.Dense(1)])

    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    lstm_model.compile(optimizer=opt, loss='mse')
    epochs = 200
    lstm_model.fit(x=io_data['x_trn'], y=io_data['y_pre_trn'], epochs=epochs)
    epochs = 100
    lstm_model.fit(x=io_data['x_trn'], y=io_data['y_obs_trn'], epochs=epochs)
    return lstm_model


with Flow('run_simple') as flow:
    out_dir = "../data/out/all_segs_lstm"
    yvars = ['seg_outflow']
    data = prep_data('../data/in/obs_temp_full', '../data/in/obs_flow_full',
                     '../data/in/uncal_sntemp_input_output',
                     ['seg_rain', 'seg_tave_air', 'seginc_swrad', 'seg_length',
                      'seginc_potet', 'seg_slope', 'seg_humid', 'seg_elev'],
                     yvars, yvars,
                     out_file=os.path.join(out_dir, 'prepped.npz'))
    model = train_lstm(data)
    preds_trn_file = os.path.join(out_dir, 'preds_trn.feather')
    preds_tst_file = os.path.join(out_dir, 'preds_tst.feather')
    preds_trn = predict(model, data, 'trn', preds_trn_file, num_segs=456)
    preds_tst = predict(model, data, 'tst', preds_tst_file, num_segs=456)

flow.run()
