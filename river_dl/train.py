import os
import random
import numpy as np
from numpy.lib.npyio import NpzFile
import datetime
import tensorflow as tf
from river_dl.RGCN import RGCNModel
from river_dl.loss_functions import weighted_masked_rmse, weighted_masked_rmse_gw
from river_dl.rnns import LSTMModel, GRUModel


def get_data_if_file(d):
    """
    rudimentary check if data .npz file is already loaded. if not, load it
    :param d:
    :return:
    """
    if isinstance(d, NpzFile) or isinstance(d, dict):
        return d
    else:
        return np.load(d, allow_pickle=True)


def train_model(
    io_data,
    pretrain_epochs,
    finetune_epochs,
    hidden_units,
    out_dir,
    flow_in_temp=False,
    model_type="rgcn",
    loss_type="GW",
    seed=None,
    dropout=0,
    lamb=1,
    lamb2=0,
    lamb3=0,
    learning_rate_pre=0.005,
    learning_rate_ft=0.01,
):
    """
    train the rgcn
    :param io_data: [dict or str] input and output data for model
    :param pretrain_epochs: [int] number of pretrain epochs
    :param finetune_epochs: [int] number of finetune epochs
    :param hidden_units: [int] number of hidden layers
    :param out_dir: [str] directory where the output files should be written
    :param flow_in_temp: [bool] whether the flow predictions should feed
    into the temp predictions
    :param model_type: [str] which model to use (either 'lstm', 'rgcn', or
    'lstm_grad_correction')
    :param seed: [int] random seed
    :param lamb: [float] (short for 'lambda') weight between 0 and 1. How much
    to weight the auxiliary rmse is weighted compared to the main rmse. The
    difference between one and lambda becomes the main rmse weight.
    :param learning_rate_pre: [float] the pretrain learning rate
    :param learning_rate_ft: [float] the finetune learning rate
    :return: [tf model]  finetuned model
    """
    if tf.test.gpu_device_name():
        print("Default GPU Device: {}".format(tf.test.gpu_device_name()))
    else:
        print("Not using GPU")

    start_time = datetime.datetime.now()
    io_data = get_data_if_file(io_data)
    dist_matrix = io_data["dist_matrix"]

    n_seg = len(np.unique(io_data["ids_trn"]))
    if n_seg > 1:
        batch_size = n_seg
    else:
        num_years = io_data["x_trn"].shape[0]
        batch_size = num_years

    if model_type == "lstm":
        model = LSTMModel(hidden_units, lamb=lamb)
    elif model_type == "rgcn":
        model = RGCNModel(
            hidden_units,
            flow_in_temp=flow_in_temp,
            A=dist_matrix,
            rand_seed=seed,
        )
    elif model_type == "lstm_grad_correction":
        grad_log_file = os.path.join(out_dir, "grad_correction.txt")
        model = LSTMModel(
            hidden_units,
            gradient_correction=True,
            lamb=lamb,
            dropout=dropout,
            grad_log_file=grad_log_file,
        )
    elif model_type == "gru":
        model = GRUModel(hidden_units, lamb=lamb)

    if seed:
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # pretrain
    if pretrain_epochs > 0:
        optimizer_pre = tf.optimizers.Adam(learning_rate=learning_rate_pre)

        # use built in 'fit' method unless model is grad correction
        x_trn_pre = io_data["x_trn"]
        # combine with weights to pass to loss function
        y_trn_pre = np.concatenate(
            [io_data["y_pre_trn"], io_data["y_pre_wgts"]], axis=2
        )

        if model_type == "rgcn":
            model.compile(optimizer_pre, loss=weighted_masked_rmse(lamb=lamb))
        else:
            model.compile(optimizer_pre)

        csv_log_pre = tf.keras.callbacks.CSVLogger(
            
            os.path.join(out_dir, f"pretrain_log.csv")
        )
        model.fit(
            x=x_trn_pre,
            y=y_trn_pre,
            epochs=pretrain_epochs,
            batch_size=batch_size,
            callbacks=[csv_log_pre],
        )

        model.save_weights(os.path.join(out_dir, "pretrained_weights/"))

    pre_train_time = datetime.datetime.now()
    pre_train_time_elapsed = pre_train_time - start_time
    out_time_file = os.path.join(out_dir, "training_time.txt")
    with open(out_time_file, "w") as f:
        f.write(
            f"elapsed time pretrain (includes building graph):\
                 {pre_train_time_elapsed} \n"
        )

    # finetune
    if finetune_epochs > 0:
        optimizer_ft = tf.optimizers.Adam(learning_rate=learning_rate_ft)
        temp_index = np.where(io_data['y_vars']=="seg_tave_water")[0]
        temp_mean = io_data['y_mean'][temp_index]
        temp_sd = io_data['y_std'][temp_index]
        
        if model_type == "rgcn" and loss_type.lower()=="gw":
            model.compile(optimizer_ft, loss=weighted_masked_rmse_gw(temp_index,temp_mean, temp_sd,lamb=lamb,lamb2=lamb2,lamb3=lamb3))
        elif model_type == "rgcn":
            model.compile(optimizer_ft, loss=weighted_masked_rmse(lamb=lamb))
        else:
            model.compile(optimizer_ft)

        csv_log_ft = tf.keras.callbacks.CSVLogger(
            os.path.join(out_dir, "finetune_log.csv")
        )

        x_trn_obs = io_data["x_trn"]
        if loss_type.lower()!="gw":
            y_trn_obs = np.concatenate(
                [io_data["y_obs_trn"], io_data["y_obs_wgts"]], axis=2
            )
        else:
            y_trn_obs = np.concatenate(
                [io_data["y_obs_trn"], io_data["GW_trn"],io_data["y_obs_wgts"]], axis=2
            )


        model.fit(
            x=x_trn_obs,
            y=y_trn_obs,
            epochs=finetune_epochs,
            batch_size=batch_size,
            callbacks=[csv_log_ft],
        )

        model.save_weights(os.path.join(out_dir, f"trained_weights/"))

    finetune_time = datetime.datetime.now()
    finetune_time_elapsed = finetune_time - pre_train_time
    with open(out_time_file, "a") as f:
        f.write(
            f"elapsed time finetune:\
                 {finetune_time_elapsed} \nloss type: {loss_type}\n"
        )

    return model
