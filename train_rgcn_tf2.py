import argparse
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
batch_offset = 1  # for the batches, offset half the year
hidden_size = 20

# read in arguments
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--network", help='network - "full" or "subset"',
                    choices=['full', 'subset'])
parser.add_argument("-o", "--outdir", help='directory where the output should\
                    be written')
parser.add_argument("-f", "--finetune_vars", help='whether to finetune on\
                    temp, flow, or both', choices=['temp', 'flow', 'both'])
parser.add_argument("-p", "--pretrain_vars", help='whether to pretrain on\
                    temp, flow, or both', choices=['temp', 'flow', 'both'])
parser.add_argument("-d", "--dist_matrix", help='which type of distance matrix\
                    to use', choices=['upstream', 'downstream', 'updown'],
                    default='upstream')
parser.add_argument("-t", "--tag", help='tag to append to end of file',
                    default='')
args = parser.parse_args()

network = args.network
out_dir = args.outdir
pt_vars = args.pretrain_vars
ft_vars = args.finetune_vars
dist_mat = args.dist_matrix
tag = args.tag
if tag != '':
    tag = f'_{tag}'


if network == "full":
    subset = False
elif network == "subset":
    subset = True

# set up model/read in data
data = read_process_data(subset=subset, trn_ratio=0.67,
                         batch_offset=batch_offset,
                         pretrain_out_vars=pt_vars, finetune_out_vars=ft_vars,
                         dist_type=dist_mat)
n_seg = data['dist_matrix'].shape[0]
model = RGCNModel(hidden_size, 2, A=data['dist_matrix'])

# pretrain
optimizer_pre = tf.optimizers.Adam(learning_rate=learning_rate_pre)
model.compile(optimizer_pre, loss=rmse_masked)

csv_log_pre = tf.keras.callbacks.CSVLogger(f'{out_dir}pretrain_log_{network}'
                                           f'{tag}.csv')

x_trn_pre = data['x_trn_pre']
y_trn_pre = data['y_trn_pre']

model.fit(x=x_trn_pre, y=y_trn_pre, epochs=epochs_pre, batch_size=n_seg,
          callbacks=[csv_log_pre])

pre_train_time = datetime.datetime.now()
pre_train_time_elapsed = pre_train_time - start_time
out_time_file = f'{out_dir}training_time_{network}.txt'
with open(out_time_file, 'w') as f:
    f.write(f'elapsed time pretrain (includes building graph):\
             {pre_train_time_elapsed} \n')

model.save_weights(f'{out_dir}pretrained_weights_{network}{tag}/')

# finetune
optimizer_ft = tf.optimizers.Adam(learning_rate=learning_rate_ft)
model.compile(optimizer_ft, loss=rmse_masked)

csv_log_ft = tf.keras.callbacks.CSVLogger(f'{out_dir}finetune_log_{network}'
                                          f'{tag}.csv')

x_trn_obs = data['x_trn']
y_trn_obs = data['y_trn_obs']

model.fit(x=x_trn_obs, y=y_trn_obs, epochs=epochs_finetune, batch_size=n_seg,
          callbacks=[csv_log_ft])

finetune_time = datetime.datetime.now()
finetune_time_elapsed = finetune_time - pre_train_time
with open(out_time_file, 'a') as f:
    f.write(f'elapsed time finetune:\
             {finetune_time_elapsed} \n')

model.save_weights(f'{out_dir}trained_weights_{network}{tag}/')

