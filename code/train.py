import argparse
import numpy as np
import datetime
import tensorflow as tf
from RGCN import RGCNModel, rmse_masked

start_time = datetime.datetime.now()

# Declare constants ######
learning_rate_pre = 0.005
learning_rate_ft = 0.01
hidden_size = 20


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--outdir", help='directory where the output should\
                    be written')
parser.add_argument('-i', "--input_data_file", help='data file [something].npz')
parser.add_argument("-t", "--tag", help='tag to append to end of file',
                    default='')
parser.add_argument("-f", "--finetune_epochs", help='num finetuning epochs',
                    default=100, type=int)
parser.add_argument("-p", "--pretrain_epochs", help='num pretraining epochs',
                    default=200, type=int)
parser.add_argument("-s", "--seed", help='random seed', type=int)

args = parser.parse_args()
epochs_pre = args.pretrain_epochs
epochs_finetune = args.finetune_epochs
if args.seed:
    seed = args.seed
    tf.random.set_seed(seed)
else:
    seed = None

in_data_file = args.input_data_file
out_dir = args.outdir
tag = args.tag
if tag != '':
    tag = f'_{tag}'

data = np.load(in_data_file)
n_seg = data['dist_matrix'].shape[0]
model = RGCNModel(hidden_size, 2, A=data['dist_matrix'], rand_seed=seed)

# pretrain
optimizer_pre = tf.optimizers.Adam(learning_rate=learning_rate_pre)
model.compile(optimizer_pre, loss=rmse_masked)

csv_log_pre = tf.keras.callbacks.CSVLogger(f'{out_dir}pretrain_log{tag}.csv')

x_trn_pre = data['x_trn_pre']
# combine with weights to pass to loss function
y_trn_pre = np.concatenate([data['y_trn_pre'], data['y_pre_wgts']], axis=2)

model.fit(x=x_trn_pre, y=y_trn_pre, epochs=epochs_pre, batch_size=n_seg,
          callbacks=[csv_log_pre])

pre_train_time = datetime.datetime.now()
pre_train_time_elapsed = pre_train_time - start_time
out_time_file = f'{out_dir}training_time{tag}.txt'
with open(out_time_file, 'w') as f:
    f.write(f'elapsed time pretrain (includes building graph):\
             {pre_train_time_elapsed} \n')

model.save_weights(f'{out_dir}pretrained_weights{tag}/')

# finetune
optimizer_ft = tf.optimizers.Adam(learning_rate=learning_rate_ft)
model.compile(optimizer_ft, loss=rmse_masked)

csv_log_ft = tf.keras.callbacks.CSVLogger(f'{out_dir}finetune_log{tag}.csv')

x_trn_obs = data['x_trn']
y_trn_obs = np.concatenate([data['y_obs_trn'], data['y_obs_wgts']], axis=2)

model.fit(x=x_trn_obs, y=y_trn_obs, epochs=epochs_finetune, batch_size=n_seg,
          callbacks=[csv_log_ft])

finetune_time = datetime.datetime.now()
finetune_time_elapsed = finetune_time - pre_train_time
with open(out_time_file, 'a') as f:
    f.write(f'elapsed time finetune:\
             {finetune_time_elapsed} \n')

model.save_weights(f'{out_dir}trained_weights{tag}/')

