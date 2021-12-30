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


def train(
        model,
        x_trn,
        y_trn,
        epochs,
        batch_size,
        x_val=None,
        y_val=None,
        weight_dir=None,
        best_val_weight_dir=None,
        log_file=None,
        time_file=None,
        seed=None,
        early_stop_patience=None,
        use_cpu=False
):
    """
    train a model

    :param model: [compiled TF model] a TF model compiled with a loss function
    and an optimizer
    :param x_trn: [array-like] input training data broken into sequences
    :param y_trn: [array-like] target training data broken into sequences
    :param epochs: [int] number of train epochs
    :param batch_size: [int] size of training batches
    :param x_val: [array-like] input validation data broken into sequences
    :param y_val: [array-like] target validation data broken into sequences
    :param weight_dir: [str] path to directory where trained weights will be
    saved from the last training epoch
    :param best_val_weight_dir: [str] path to directory where trained weights
    will be saved from the training epoch with the best validation performance 
    :param log_file: [str] path to file where training log will be saved
    :param time_file: [str] path to file where training time will be written
    :param seed: [int] random seed
    :param early_stop_patience [int] Number of epochs with no improvement after
    which training will be stopped. Default is none meaning that training will
    continue for all specified epochs
    :param use_cpu: [bool] If True, ensures that training happens on CPU. This
    can be desirable in some cases (e.g., when using the GW loss function)
    :return: [tf model] trained model
    """
    if tf.test.gpu_device_name():
        print("Default GPU Device: {}".format(tf.test.gpu_device_name()))
    else:
        print("Not using GPU")

    if seed:
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # Set up early stopping rounds if desired, setting this to the total number
    # of epochs is the same as not using it
    callbacks = []
    if early_stop_patience:
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=early_stop_patience,
            restore_best_weights=False,
            verbose=1)
        callbacks.append(early_stop)

    if log_file:
        csv_log = tf.keras.callbacks.CSVLogger(log_file)
        callbacks.append(csv_log)

    # Save alternate weight file that saves the best validation weights
    if best_val_weight_dir:
        best_val = tf.keras.callbacks.ModelCheckpoint(
            best_val_weight_dir,
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            save_weights_only=True,
            mode='min',
            save_freq='epoch')
        callbacks.append(best_val)

    if isinstance(x_val, np.ndarray) and isinstance(y_val, np.ndarray):
        validation_data = (x_val, y_val)
    else:
        validation_data = None

    # train the model
    start_time = datetime.datetime.now()
    if use_cpu:
        with tf.device('/CPU:0'):
            model.fit(
                x=x_trn,
                y=y_trn,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                validation_data=validation_data
            )
    else:
        model.fit(
            x=x_trn,
            y=y_trn,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            validation_data=validation_data
        )

    # write out time
    end_time = datetime.datetime.now()
    time_elapsed = end_time - start_time
    print(f"Training time: {time_elapsed}")
    if time_file:
        with open(time_file, "a") as f:
            f.write(f"elapsed training time: {time_elapsed}\n")

    # Save our trained weights
    if weight_dir:
        model.save_weights(weight_dir)

    return model


def train_model(
        io_data,
        epochs,
        hidden_units=None,
        loss_func=None,
        out_dir=".",
        model=None,
        model_type=None,
        seed=None,
        dropout=0,
        recurrent_dropout=0,
        num_tasks=1,
        learning_rate=0.01,
        train_type='pre',
        early_stop_patience=None,
        limit_pretrain=False,
):
    """
    Pretrain or finetune a model with hardcoded output file/dir names. You can
    pass either
    1) compiled model 
    2) "model_type" and associated parameters and the model will be compiled

    :param io_data: [dict or str] input and output data for model
    :param epochs: [int] number of train epochs
    :param model: [compiled TF model] a TF model compiled with a loss function 
    and an optimizer
    :param hidden_units: [int] number of hidden layers
    :param loss_func: [function] loss function that the model will be fit to
    :param out_dir: [str] directory where the output files should be written
    :param model_type: [str] which model to use (either 'lstm', 'rgcn', or
    'gru')
    :param seed: [int] random seed
    :param recurrent_dropout: [float] value between 0 and 1 for the probability
    of a reccurent element to be zero
    :param dropout: [float] value between 0 and 1 for the probability of an
    input element to be zero
    :param num_tasks: [int] number of tasks (variables_to_log to be predicted)
    :param learning_rate: [float] the learning rate
    :param train_type: [str] Either pretraining (pre) or finetuning (finetune)
    :param early_stop_patience [int]  Number of epochs with no improvement after
    which training will be stopped.
    :param limit_pretrain [bool] If true, limits pretraining to just the
    training partition.  If false (default), pretrains on all available data.
    :return: [tf model]  Model
    """
    if train_type not in ['pre', 'finetune']:
        raise ValueError("Specify train_type as either pre or finetune")

    if tf.test.gpu_device_name():
        print("Default GPU Device: {}".format(tf.test.gpu_device_name()))
    else:
        print("Not using GPU")

    io_data = get_data_if_file(io_data)

    n_seg = len(np.unique(io_data["ids_trn"]))

    if n_seg > 1:
        batch_size = n_seg
    else:
        num_years = io_data["x_trn"].shape[0]
        batch_size = num_years

    if not model and not model_type:
        raise ValueError("You must either pass a model or a model_type")

    if not model:
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

        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss_func)

    # pretrain
    if train_type == 'pre':

        # Create a dummy directory for the snakemake if you don't want
        # pre-training
        if epochs == 0:
            os.makedirs(os.path.join(out_dir, "pretrained_weights/"),
                        exist_ok=True)
            print("Dummy directory created without pretraining.  "
                  "Set epochs to >0 to pretrain")
        else:
            # Pull out variables from the IO data
            if limit_pretrain:
                x_trn_pre = io_data["x_trn"]
                y_trn_pre = io_data["y_pre_trn"]
            else:
                x_trn_pre = io_data["x_pre_full"]
                y_trn_pre = io_data["y_pre_full"]

            model = train(model=model,
                          x_trn=x_trn_pre,
                          y_trn=y_trn_pre,
                          epochs=epochs,
                          batch_size=batch_size,
                          weight_dir=os.path.join(out_dir,
                                                  "pretrained_weights/"),
                          log_file=os.path.join(out_dir, "pretrain_log.csv"),
                          time_file=os.path.join(out_dir, "pretrain_time.txt"),
                          seed=seed)

    # finetune
    if train_type == 'finetune':

        # Load pretrain weights if they exist
        if os.path.exists(os.path.join(out_dir,
                                       "pretrained_weights/checkpoint")):
            weights = os.path.join(out_dir, "pretrained_weights/")
            model.load_weights(weights)

        # Specify our variables
        y_trn_obs = io_data["y_obs_trn"]
        x_trn = io_data["x_trn"]
        y_val_obs = io_data['y_obs_val']
        x_val = io_data['x_val']

        if "GW_trn_reshape" in io_data.files:
            temp_air_index = np.where(io_data['x_vars'] == 'seg_tave_air')[0]
            air_unscaled = io_data['x_trn'][:, :, temp_air_index] * \
                           io_data['x_std'][temp_air_index] + \
                           io_data['x_mean'][temp_air_index]
            y_trn_obs = np.concatenate(
                [io_data["y_obs_trn"], io_data["GW_trn_reshape"], air_unscaled],
                axis=2
            )
            air_val = io_data['x_val'][:, :, temp_air_index] * \
                      io_data['x_std'][temp_air_index] + \
                      io_data['x_mean'][temp_air_index]
            y_val_obs = np.concatenate(
                [io_data["y_obs_val"], io_data["GW_val_reshape"], air_val],
                axis=2
            )
            # Run the finetuning within the training engine on CPU for the GW
            # loss function
            use_cpu = True
        else:
            use_cpu = False

        model = train(model=model,
                      x_trn=x_trn,
                      y_trn=y_trn_obs,
                      epochs=epochs,
                      batch_size=batch_size,
                      x_val=x_val,
                      y_val=y_val_obs,
                      weight_dir=os.path.join(out_dir, "trained_weights/"),
                      best_val_weight_dir=os.path.join(out_dir,
                                                       "best_val_weights/"),
                      log_file=os.path.join(out_dir, "finetune_log.csv"),
                      time_file=os.path.join(out_dir, "finetune_time.txt"),
                      early_stop_patience=early_stop_patience,
                      use_cpu=use_cpu)

    return model
