import datetime
import tensorflow as tf
from RGCN_tf2 import RGCNModel, rmse_masked
from data_utils import read_process_data, process_adj_matrix

start_time = datetime.datetime.now()

# Declare constants ######
tf.random.set_seed(23)
learning_rate = 0.01
learning_rate_pre = 0.005
epochs_finetune = 3
epochs_pre = 3
batch_offset = 0.5  # for the batches, offset half the year
hidden_size = 20
n_seg = 42

# set up model/read in data
data = read_process_data(subset=True, trn_ratio=0.67, batch_offset=1,
                         pretrain_out_vars='both', finetune_out_vars='temp',
                         dist_type='upstream')
model = RGCNModel(hidden_size, 2, A=data['dist_matrix'])
optimizer = tf.optimizers.Adam(learning_rate=learning_rate_pre)
model.compile(optimizer, loss=rmse_masked)

# pretrain
x_trn_pre = data['x_trn_pre']
y_trn_pre = data['y_trn_pre']
model.fit(x=x_trn_pre, y=y_trn_pre, epochs=epochs_pre, batch_size=n_seg)
pre_train_time = datetime.datetime.now()
print('elapsed time pretrain (includes building graph):',
      pre_train_time - start_time)

# finetune
x_trn_obs = data['x_trn']
y_trn_obs = data['y_trn_obs']
model.fit(x=x_trn_obs, y=y_trn_obs, epochs=epochs_finetune, batch_size=n_seg)
finetune_time = datetime.datetime.now()
print('elapsed time finetune:', finetune_time - pre_train_time)

model.save_weights('data/out/trained_weights.h5')

