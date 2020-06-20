import tensorflow as tf
from tensorflow.keras import models, layers
from preproc_utils import prep_data

yvars = ['seg_outflow']

data = prep_data('../data/in/obs_temp_full', '../data/in/obs_flow_full',
          '../data/in/uncal_sntemp_input_output', ['seg_rain', 'seg_tave_air'],
          yvars, yvars, segment_id=1659)


hidden_units = 20
simple_lstm_model = models.Sequential([layers.LSTM(hidden_units),
                                       layers.Dense(1)])

opt = tf.keras.optimizers.Adam(learning_rate=0.01)
simple_lstm_model.compile(optimizer=opt, loss='mae')
epochs = 1
simple_lstm_model.fit(x=data['x_trn'], y=data['y_pre_trn'], epochs=epochs)
epochs = 1
simple_lstm_model.fit(x=data['x_trn'], y=data['y_obs_trn'], epochs=epochs)
simple_lstm_model.save('simple')


print('x')
