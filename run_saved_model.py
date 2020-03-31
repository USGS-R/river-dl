import tensorflow as tf
from RGCN_tf2 import RGCNModel, rmse_masked
from data_utils import read_process_data, post_process


tf.random.set_seed(23)
learning_rate = 0.01
learning_rate_pre = 0.005
hidden_size = 20

data = read_process_data(subset=False, trn_ratio=0.67, batch_offset=1,
                         dist_type='upstream')
num_segs = data['dist_mat'].shape[0]
model = RGCNModel(hidden_size, 2, A=data['dist_mat'])
optimizer = tf.optimizers.Adam(learning_rate=learning_rate_pre)
model.compile(optimizer, loss=rmse_masked)

model.load_weights('data/out/trained_weights.h5')
y_pred = model.predict(data['x_tst'], batch_size=num_segs)
y_pred_pp = post_process(y_pred, data['dates_ids_tst'], data['y_trn_obs_std'],
                         data['y_trn_obs_mean'])
y_pred_pp.to_feather('data/out/y_tst_preds.feather')
