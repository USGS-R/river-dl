import os
import numpy as np
from numpy.lib.npyio import NpzFile
import datetime
import tensorflow as tf
from river_dl.RGCN import RGCNModel, weighted_masked_rmse


def get_data_if_file(d):
    """
    rudimentary check if data .npz file is already loaded. if not, load it
    :param d:
    :return:
    """
    if isinstance(d, NpzFile):
        return d
    else:
        return np.load(d)


def train_model(io_data, pretrain_epochs, finetune_epochs,
                hidden_units, out_dir, flow_in_temp=False, seed=None,
                pretrain_temp_rmse_weight=0.5, finetune_temp_rmse_weight=0.5,
                learning_rate_pre=0.005, learning_rate_ft=0.01):
    """
    train the rgcn
    :param x_data: [dict or str] the data file or data dict of the x_data
    :param y_data: [dict or str] the data file or data dict of the y_data
    :param dist_matrix: [dict or str] data file or data dict of the dist_matrix
    :param pretrain_epochs: [int] number of pretrain epochs
    :param finetune_epochs: [int] number of finetune epochs
    :param hidden_units: [int] number of hidden layers
    :param out_dir: [str] directory where the output files should be written
    :param flow_in_temp: [bool] whether the flow predictions should feed
    into the temp predictions
    :param seed: [int] random seed
    :param pretrain_temp_rmse_weight: [float] weight between 0 and 1. How much
    to weight the rmse of temperature in pretraining compared to flow. The
    difference between one and the temperature_weight becomes the flow_weight.
    If you want to weight the rmse of temperature and the rmse of flow equally,
    the temperature_weight would be 0.5
    :param finetune_temp_rmse_weight: [float] weight between 0 and 1. How much
    to weight the rmse of temperature in finetuning compared to flow. The
    difference between one and the temperature_weight becomes the flow_weight.
    If you want to weight the rmse of temperature and the rmse of flow equally,
    the temperature_weight would be 0.5
    :param learning_rate_ft: [float] the finetune learning rate
    :return: [tf model]  finetuned model
    """
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")

    start_time = datetime.datetime.now()
    io_data = get_data_if_file(io_data)
    dist_matrix = io_data['dist_matrix']

    n_seg = dist_matrix.shape[0]
    out_size = len(io_data['y_vars'])
    model = RGCNModel(hidden_units, flow_in_temp=flow_in_temp,
                      A=dist_matrix, rand_seed=seed)

    # pretrain
    optimizer_pre = tf.optimizers.Adam(learning_rate=learning_rate_pre)
    model.compile(optimizer_pre,
                  loss=weighted_masked_rmse(pretrain_temp_rmse_weight))

    csv_log_pre = tf.keras.callbacks.CSVLogger(
        os.path.join(out_dir, f'pretrain_log.csv'))

    x_trn_pre = io_data['x_trn']
    # combine with weights to pass to loss function
    y_trn_pre = np.concatenate([io_data['y_pre_trn'], io_data['y_pre_wgts']],
                               axis=2)

    model.fit(x=x_trn_pre, y=y_trn_pre, epochs=pretrain_epochs,
              batch_size=n_seg, callbacks=[csv_log_pre])

    pre_train_time = datetime.datetime.now()
    pre_train_time_elapsed = pre_train_time - start_time
    out_time_file = os.path.join(out_dir, 'training_time.txt')
    with open(out_time_file, 'w') as f:
        f.write(f'elapsed time pretrain (includes building graph):\
                 {pre_train_time_elapsed} \n')

    model.save_weights(os.path.join(out_dir, 'pretrained_weights/'))

    # finetune
    optimizer_ft = tf.optimizers.Adam(learning_rate=learning_rate_ft)
    model.compile(optimizer_ft,
                  loss=weighted_masked_rmse(finetune_temp_rmse_weight))

    csv_log_ft = tf.keras.callbacks.CSVLogger(
        os.path.join(out_dir, 'finetune_log.csv'))

    x_trn_obs = io_data['x_trn']
    y_trn_obs = np.concatenate([io_data['y_obs_trn'], io_data['y_obs_wgts']],
                               axis=2)

    model.fit(x=x_trn_obs, y=y_trn_obs, epochs=finetune_epochs,
              batch_size=n_seg, callbacks=[csv_log_ft])

    finetune_time = datetime.datetime.now()
    finetune_time_elapsed = finetune_time - pre_train_time
    with open(out_time_file, 'a') as f:
        f.write(f'elapsed time finetune:\
                 {finetune_time_elapsed} \n')

    model.save_weights(os.path.join(out_dir, f'trained_weights/'))
    return model

