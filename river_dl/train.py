import os
import random
import numpy as np
from numpy.lib.npyio import NpzFile
import datetime
import tensorflow as tf
from river_dl.RGCN import RGCNModel
from river_dl.loss_functions import weighted_masked_rmse_gw
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
    loss_func_ft,
    out_dir,
    loss_func_pre = None,
    model_type="rgcn",
    seed=None,
    dropout=0,
    recurrent_dropout=0,
    num_tasks=1,
    learning_rate_pre=0.005,
    learning_rate_ft=0.01,
    updated_io_data=None
):
    """
    train the rgcn
    :param io_data: [dict or str] input and output data for model
    :param pretrain_epochs: [int] number of pretrain epochs
    :param finetune_epochs: [int] number of finetune epochs
    :param hidden_units: [int] number of hidden layers
    :param loss_func_ft: [function] loss function that the model will be fit to
    :param out_dir: [str] directory where the output files should be written
    :param loss_func_pre: [function] optional 2nd loss function to use for the 
    pretrain epochs, if None, loss_func_ft will be used for both pretrain and 
    finetune
    :param model_type: [str] which model to use (either 'lstm', 'rgcn', or
    'gru')
    :param seed: [int] random seed
    :param recurrent_dropout: [float] value between 0 and 1 for the probability
    of a reccurent element to be zero
    :param dropout: [float] value between 0 and 1 for the probability of an
    input element to be zero
    :param num_tasks: [int] number of tasks (variables_to_log to be predicted)
    :param learning_rate_pre: [float] the pretrain learning rate
    :param learning_rate_ft: [float] the finetune learning rate
    :return: [tf model]  finetuned model
    """



    if tf.test.gpu_device_name():
        print("Default GPU Device: {}".format(tf.test.gpu_device_name()))
    else:
        print("Not using GPU")
    
    #use loss_func for both pretrain and finetune if loss_func_ft is not given
    if loss_func_pre is None:
        loss_func_pre = loss_func_ft

    start_time = datetime.datetime.now()
    io_data = get_data_if_file(io_data)

    n_seg = len(np.unique(io_data["ids_trn"]))
    if n_seg > 1:
        batch_size = n_seg
    else:
        num_years = io_data["x_trn"].shape[0]
        batch_size = num_years

    if model_type == "lstm":
        model = LSTMModel(
            hidden_units,
            num_tasks=num_tasks,
            recurrent_dropout=recurrent_dropout,
            dropout=dropout,
        )
    elif model_type == "rgcn":
        dist_matrix = io_data["dist_matrix"]
        model = RGCNModel(
            hidden_units,
            num_tasks=num_tasks,
            A=dist_matrix,
            rand_seed=seed,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
        )
    elif model_type == "gru":
        model = GRUModel(
            hidden_units,
            num_tasks=num_tasks,
            recurrent_dropout=recurrent_dropout,
            dropout=dropout,
        )
    else:
        raise ValueError(
            f"The 'model_type' provided ({model_type}) is not supported"
        )

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
        y_trn_pre = io_data["y_pre_trn"]

        model.compile(optimizer_pre, loss=loss_func_pre)

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

        model.compile(optimizer_ft, loss=loss_func_ft)


        csv_log_ft = tf.keras.callbacks.CSVLogger(
            os.path.join(out_dir, "finetune_log.csv")
        )

        x_trn_obs = io_data["x_trn"]


        if "GW_trn_reshape" in io_data.files:
            temp_air_index = np.where(io_data['x_vars']=='seg_tave_air')[0]
            air_unscaled = io_data['x_trn'][:,:,temp_air_index]*io_data['x_std'][temp_air_index] +io_data['x_mean'][temp_air_index]
            y_trn_obs = np.concatenate(
                [io_data["y_obs_trn"], io_data["GW_trn_reshape"], air_unscaled], axis=2
            )
            
            with tf.device('/CPU:0'):
                model.fit(
                    x=x_trn_obs,
                    y=y_trn_obs,
                    epochs=finetune_epochs,
                    batch_size=batch_size,
                    callbacks=[csv_log_ft],
                )
        else:
            if pretrain_epochs == 0:
                print('Using PB outputs as inputs')
                # Import the pretraining data and append it to the x vars
                y_trn_pre = io_data["y_pre_trn"]
                y_trn_pre = np.nan_to_num(y_trn_pre, nan = 0) # has some nans - not great solution
                x_trn_obs = np.concatenate([x_trn_obs, y_trn_pre], 2) 
                # Do the same for the validation and testing x vars
                y_val_pre = io_data["y_pre_val"]
                y_val_pre = np.nan_to_num(y_val_pre, nan = 0)
                x_val_obs = io_data["x_val"]
                x_val_obs = np.concatenate([x_val_obs, y_val_pre], 2)
                y_tst_pre = io_data["y_pre_tst"]
                y_tst_pre = np.nan_to_num(y_tst_pre, nan = 0)
                x_tst_obs = io_data["x_tst"]
                x_tst_obs = np.concatenate([x_tst_obs, y_tst_pre], 2)
                # Update the saved file (so that PB outputs are there for eval; generate same file for no PB outputs too)
                print("Saving the x data with associated pretraining output", x_trn_obs.shape, x_val_obs.shape, x_tst_obs.shape)
                np.savez_compressed(updated_io_data, x_trn = x_trn_obs, x_val = x_val_obs, x_tst = x_tst_obs,
                                    x_std = io_data['x_std'], x_mean = io_data['x_mean'], x_vars = io_data['x_vars'],
                                    ids_trn = io_data['ids_trn'], times_trn = io_data['times_trn'],
                                    ids_val = io_data['ids_val'], times_val = io_data['times_val'],
                                    ids_tst = io_data['ids_tst'], times_tst = io_data['times_tst'], dist_matrix = io_data['dist_matrix'],
                                    y_obs_trn = io_data['y_obs_trn'], y_obs_wgts = io_data['y_obs_wgts'],
                                    y_obs_val = io_data['y_obs_val'], y_obs_tst = io_data['y_obs_tst'],
                                    y_std = io_data['y_std'], y_mean = io_data['y_mean'], y_obs_vars = io_data['y_obs_vars'],
                                    y_pre_trn = io_data['y_pre_trn'], y_pre_wgts = io_data['y_pre_wgts'],
                                    y_pre_val = io_data['y_pre_val'], y_pre_tst = io_data['y_pre_tst'], y_pre_vars = io_data['y_pre_vars'])
            else:
                np.savez_compressed(updated_io_data, x_trn = io_data['x_trn'], x_val = io_data['x_val'], x_tst = io_data['x_tst'],
                                    x_std = io_data['x_std'], x_mean = io_data['x_mean'], x_vars = io_data['x_vars'],
                                    ids_trn = io_data['ids_trn'], times_trn = io_data['times_trn'],
                                    ids_val = io_data['ids_val'], times_val = io_data['times_val'],
                                    ids_tst = io_data['ids_tst'], times_tst = io_data['times_tst'], dist_matrix = io_data['dist_matrix'],
                                    y_obs_trn = io_data['y_obs_trn'], y_obs_wgts = io_data['y_obs_wgts'],
                                    y_obs_val = io_data['y_obs_val'], y_obs_tst = io_data['y_obs_tst'],
                                    y_std = io_data['y_std'], y_mean = io_data['y_mean'], y_obs_vars = io_data['y_obs_vars'],
                                    y_pre_trn = io_data['y_pre_trn'], y_pre_wgts = io_data['y_pre_wgts'],
                                    y_pre_val = io_data['y_pre_val'], y_pre_tst = io_data['y_pre_tst'], y_pre_vars = io_data['y_pre_vars'])
            y_trn_obs = io_data["y_obs_trn"]
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
                 {finetune_time_elapsed} \nloss type: gw\n"
        )

    return model
