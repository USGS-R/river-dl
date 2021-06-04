import os
import random
import numpy as np
from numpy.lib.npyio import NpzFile
import datetime
import tensorflow as tf
from river_dl.RGCN import RGCNModel
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
    loss_func,
    out_dir,
    model_type="rgcn",
    seed=None,
    dropout=0,
    recurrent_dropout=0,
    num_tasks=1,
    learning_rate_pre=0.005,
    learning_rate_ft=0.01,
):
    """
    train the rgcn
    :param io_data: [dict or str] input and output data for model
    :param pretrain_epochs: [int] number of pretrain epochs
    :param finetune_epochs: [int] number of finetune epochs
    :param hidden_units: [int] number of hidden layers
    :param loss_func: [function] loss function that the model will be fit to
    :param out_dir: [str] directory where the output files should be written
    :param model_type: [str] which model to use (either 'lstm', 'rgcn', or
    'gru')
    :param seed: [int] random seed
    :param recurrent_dropout: [float] value between 0 and 1 for the probability of a reccurent element to be zero
    :param dropout: [float] value between 0 and 1 for the probability of an input element to be zero
    :param num_tasks: [int] number of tasks (outputs to be predicted)
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
        model = LSTMModel(hidden_units, num_tasks=num_tasks, recurrent_dropout=recurrent_dropout, dropout=dropout)
    elif model_type == "rgcn":
        model = RGCNModel(
            hidden_units,
            num_tasks=num_tasks,
            A=dist_matrix,
            rand_seed=seed,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout
        )
    elif model_type == "gru":
        model = GRUModel(hidden_units, num_tasks=num_tasks, recurrent_dropout=recurrent_dropout, dropout=dropout)
    else:
        raise ValueError(f"The 'model_type' provided ({model_type}) is not supported")

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

        model.compile(optimizer_pre, loss=loss_func)

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

        model.compile(optimizer_ft, loss=loss_func)

        csv_log_ft = tf.keras.callbacks.CSVLogger(
            os.path.join(out_dir, "finetune_log.csv")
        )

        x_trn_obs = io_data["x_trn"]
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
                 {finetune_time_elapsed} \n"
        )

    return model
