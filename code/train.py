import os
import numpy as np
import datetime
import tensorflow as tf
from RGCN import RGCNModel, rmse_masked


def get_data_if_file(d):
    """
    rudimentary check if data .npz file is already loaded. if not, load it
    :param d:
    :return:
    """
    if isinstance(d, dict):
        return d
    else:
        return np.load(d)


def train_model(x_data, y_data, dist_matrix, pretrain_epochs, finetune_epochs,
                hidden_units, out_dir, seed=None, learning_rate_pre=0.005,
                learning_rate_ft=0.01):
    """
    train the rgcn
    :param x_data: [dict or str] the data file or data dict of the x_data
    :param y_data: [dict or str] the data file or data dict of the y_data
    :param dist_matrix: [dict or str] data file or data dict of the dist_matrix
    :param pretrain_epochs: [int] number of pretrain epochs
    :param finetune_epochs: [int] number of finetune epochs
    :param hidden_units: [int] number of hidden layers
    :param out_dir: [str] directory where the output files should be written
    :param seed: [int] random seed
    :param learning_rate_pre: [float] the pretrain learning rate
    :param learning_rate_ft: [float] the finetune learning rate
    :return: [tf model]  finetuned model
    """

    start_time = datetime.datetime.now()
    x_data = get_data_if_file(x_data)
    y_data = get_data_if_file(y_data)

    n_seg = dist_matrix.shape[0]
    out_size = len(y_data['y_vars'])
    model = RGCNModel(hidden_units, out_size, A=dist_matrix, rand_seed=seed)

    # pretrain
    optimizer_pre = tf.optimizers.Adam(learning_rate=learning_rate_pre)
    model.compile(optimizer_pre, loss=rmse_masked)

    csv_log_pre = tf.keras.callbacks.CSVLogger(
        os.path.join(out_dir, f'pretrain_log.csv'))

    x_trn_pre = x_data['x_trn']
    # combine with weights to pass to loss function
    y_trn_pre = np.concatenate([y_data['y_trn_pre'], y_data['y_pre_wgts']],
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
    model.compile(optimizer_ft, loss=rmse_masked)

    csv_log_ft = tf.keras.callbacks.CSVLogger(
        os.path.join(out_dir, 'finetune_log.csv'))

    x_trn_obs = x_data['x_trn']
    y_trn_obs = np.concatenate([y_data['y_obs_trn'], y_data['y_obs_wgts']],
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

