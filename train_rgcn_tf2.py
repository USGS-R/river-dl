import datetime
import tensorflow as tf
from RGCN_tf2 import RGCNModel, rmse_masked
from data_utils import read_process_data, process_adj_matrix

start_time = datetime.datetime.now()

# Declare constants ######
tf.random.set_seed(23)
learning_rate_pre = 0.005
learning_rate_ft = 0.01
epochs_pre = 200
epochs_finetune = 100
batch_offset = 0.5  # for the batches, offset half the year
hidden_size = 20
out_dir = 'data/out/'
out_time_file = f'{out_dir}training_time.txt'

# set up model/read in data
data = read_process_data(subset=True, trn_ratio=0.67, batch_offset=1,
                         pretrain_out_vars='both', finetune_out_vars='temp',
                         dist_type='upstream')
n_seg = data['dist_matrix'].shape[0]
model = RGCNModel(hidden_size, 2, A=data['dist_matrix'])

# pretrain
optimizer_pre = tf.optimizers.Adam(learning_rate=learning_rate_pre)
model.compile(optimizer_pre, loss=rmse_masked)

csv_log_pre = tf.keras.callbacks.CSVLogger(f'{out_dir}pretrain_log.csv')

x_trn_pre = data['x_trn_pre']
y_trn_pre = data['y_trn_pre']

model.fit(x=x_trn_pre, y=y_trn_pre, epochs=epochs_pre, batch_size=n_seg,
          callbacks=[csv_log_pre])

pre_train_time = datetime.datetime.now()
pre_train_time_elapsed = pre_train_time - start_time
with open(out_time_file, 'w') as f:
    f.write(f'elapsed time pretrain (includes building graph):\
             {pre_train_time_elapsed} \n')

# finetune
optimizer_ft = tf.optimizers.Adam(learning_rate=learning_rate_ft)
model.compile(optimizer_ft, loss=rmse_masked)

csv_log_ft = tf.keras.callbacks.CSVLogger(f'{out_dir}finetune_log.csv')

x_trn_obs = data['x_trn']
y_trn_obs = data['y_trn_obs']

model.fit(x=x_trn_obs, y=y_trn_obs, epochs=epochs_finetune, batch_size=n_seg,
          callbacks=[csv_log_ft])

finetune_time = datetime.datetime.now()
finetune_time_elapsed = finetune_time - pre_train_time
with open(out_time_file, 'a') as f:
    f.write(f'elapsed time finetune:\
             {finetune_time_elapsed} \n')

model.save_weights(f'{out_dir}trained_weights/')

