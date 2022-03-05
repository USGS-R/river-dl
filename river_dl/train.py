import os
import random
import numpy as np
import datetime
import tensorflow as tf


def train_model(
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
    early_stop_patience = None,
    limit_pretrain = False,
    keep_portion = None,
    use_cpu=False
):
    """
    train the model
    :param model: [compiled TF model] a TF model compiled with a loss function 
    and an optimizer
    :param epochs: [int] number of train epochs
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
    :param limit_pretrain [bool] If true, limits pretraining to just the training partition.  If false (default), pretrains on all available data.
    :param keep_portion [float] Mask out observed sequence leading up to the keep portion if specified.
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


    # If keep portion is specified, mask the sequence outside that portion to
    # force the loss function focus on only the keep portion.
    if keep_portion is not None:
        if keep_portion > 1:
            period = int(keep_portion)
        else:
            period = int(keep_portion * y_trn.shape[1])
        y_trn[:, :-period, ...] = np.nan
        y_val[:, :-period, ...] = np.nan
                
    # Set up early stopping rounds if desired, setting this to the total number
    # of epochs is the same as not using it
    callbacks = []
    if early_stop_patience:
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=early_stop_patience, restore_best_weights=False,
            verbose=1)
        callbacks.append(early_stop)

    if log_file:
        csv_log = tf.keras.callbacks.CSVLogger(log_file)
        callbacks.append(csv_log)

    # Save alternate weight file that saves the best validation weights
    if best_val_weight_dir and isinstance(x_val, np.ndarray):
        best_val = tf.keras.callbacks.ModelCheckpoint(
            best_val_weight_dir, monitor='val_loss', verbose=0, save_best_only=True,
            save_weights_only=True, mode='min', save_freq='epoch')
        callbacks.append(best_val)
    elif best_val_weight_dir and not isinstance(x_val, np.ndarray):
        raise ValueError("best_val_weight_dir requires validation data")


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
