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


# This is a training engine that initializes our model and contains routines for pretraining and finetuning
class trainer():
    def __init__(self, model, optimizer, loss_fn, weights=None):
        self.model = model
        self.model.compile(optimizer=optimizer, loss=loss_fn)
        if weights:
            self.model.load_weights(weights)

    def pre_train(self, x, y, epochs, batch_size, out_dir):

        ## Set up training log callback
        csv_log = tf.keras.callbacks.CSVLogger(
            os.path.join(out_dir, f"pretrain_log.csv")
        )

        # Use generic fit statement
        self.model.fit(
            x=x,
            y=y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[csv_log],
        )
        # Save the pretrained weights
        self.model.save_weights(os.path.join(out_dir, "pretrained_weights/"))
        return self.model

    def fine_tune(self, x, y, x_val, y_val, epochs, batch_size, out_dir, early_stop_patience=None, use_cpu = False):
        # Specify our training log
        csv_log = tf.keras.callbacks.CSVLogger(
            os.path.join(out_dir, "finetune_log.csv")
        )

        # Set up early stopping rounds if desired, setting this to the total number of epochs is the same as not using it
        if not early_stop_patience:
            early_stop_patience = epochs

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=early_stop_patience, restore_best_weights=False,
            verbose=1)

        # Save alternate weight file that saves the best validation weights
        best_val = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(out_dir, 'best_val_weights/'), monitor='val_loss', verbose=0, save_best_only=True,
            save_weights_only=True, mode='min', save_freq='epoch')

        # Ensure that training happens on CPU if using the GW loss function
        if use_cpu:
            with tf.device('/CPU:0'):
                self.model.fit(
                    x=x,
                    y=y,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[csv_log, early_stop, best_val],
                    validation_data=(x_val, y_val)
                )
        else:
           self.model.fit(
               x=x,
               y=y,
               epochs=epochs,
               batch_size=batch_size,
               callbacks=[csv_log, early_stop, best_val],
               validation_data=(x_val, y_val)
        )

        # Save our trained weights
        self.model.save_weights(os.path.join(out_dir, f"trained_weights/"))
        return self.model


def train_model(
    io_data,
    epochs,
    hidden_units,
    loss_func,
    out_dir,
    model_type="rgcn",
    seed=None,
    dropout=0,
    recurrent_dropout=0,
    num_tasks=1,
    learning_rate = 0.01,
    train_type = 'pre',
    early_stop_patience = None,
    limit_pretrain = False,
):
    """
    train the rgcn
    :param io_data: [dict or str] input and output data for model
    :param epochs: [int] number of train epochs
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
    :param early_stop_patience [int]  Number of epochs with no improvement after which training will be stopped.
    :param limit_pretrain [bool] If true, limits pretraining to just the training partition.  If false (default), pretrains on all available data.
    :return: [tf model]  Model
    """
    if train_type not in ['pre','finetune']:
        raise ValueError("Specify train_type as either pre or finetune")

    if tf.test.gpu_device_name():
        print("Default GPU Device: {}".format(tf.test.gpu_device_name()))
    else:
        print("Not using GPU")

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

    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    # pretrain
    if train_type == 'pre':

        ## Create a dummy directory for the snakemake if you don't want pre-training
        if epochs == 0:
            os.makedirs(os.path.join(out_dir, "pretrained_weights/"), exist_ok = True)
            print("Dummy directory created without pretraining.  Set epochs to >0 to pretrain")
        else:
            # Pull out variables from the IO data
            if limit_pretrain:
                x_trn_pre = io_data["x_trn"]
                y_trn_pre = io_data["y_pre_trn"]
            else:
                x_trn_pre = io_data["x_pre_full"]
                y_trn_pre = io_data["y_pre_full"]

            # Initialize our model within the training engine
            engine = trainer(model, optimizer, loss_func)

            # Call the pretraining routine from the training engine
            model = engine.pre_train(x_trn_pre,y_trn_pre,epochs, batch_size,out_dir)

            # Log our training times
            pre_train_time = datetime.datetime.now()
            pre_train_time_elapsed = pre_train_time - start_time
            print(f"Pretraining time: {pre_train_time_elapsed}")
            out_time_file = os.path.join(out_dir, "training_time.txt")

            with open(out_time_file, "w") as f:
                f.write(
                    f"elapsed time pretrain (includes building graph):\
                         {pre_train_time_elapsed} \n"
                )

    # finetune
    if train_type == 'finetune':

        # Load pretrain weights if they exist
        if os.path.exists(os.path.join(out_dir, "pretrained_weights/checkpoint")):
            weights = os.path.join(out_dir, "pretrained_weights/")
        else:
            weights = None
            #model.load_weights(os.path.join(out_dir, "pretrained_weights/"))

        # Initialize our model within the training engine
        engine = trainer(model, optimizer, loss_func, weights)

        # Specify our variables
        y_trn_obs = io_data["y_obs_trn"]
        x_trn = io_data["x_trn"]
        y_val_obs = io_data['y_obs_val']
        x_val = io_data['x_val']

        if "GW_trn_reshape" in io_data.files:
            temp_air_index = np.where(io_data['x_vars'] == 'seg_tave_air')[0]
            air_unscaled = io_data['x_trn'][:, :, temp_air_index] * io_data['x_std'][temp_air_index] + \
                           io_data['x_mean'][temp_air_index]
            y_trn_obs = np.concatenate(
                [io_data["y_obs_trn"], io_data["GW_trn_reshape"], air_unscaled], axis=2
            )
            air_val = io_data['x_val'][:, :, temp_air_index] * io_data['x_std'][temp_air_index] + io_data['x_mean'][
                temp_air_index]
            y_val_obs = np.concatenate(
                [io_data["y_obs_val"], io_data["GW_val_reshape"], air_val], axis=2
            )
            # Run the finetuning within the training engine on CPU for the GW loss function
            model = engine.fine_tune(x_trn, y_trn_obs, x_val, y_val_obs, epochs, batch_size, out_dir, early_stop_patience, use_cpu=True)

        else:
            # Run the finetuning within the training engine on default device
            model = engine.fine_tune(x_trn, y_trn_obs, x_val, y_val_obs, epochs, batch_size, out_dir, early_stop_patience)

        # Log our training time
        finetune_time = datetime.datetime.now()
        finetune_time_elapsed = finetune_time - start_time
        print(f"Finetuning time: {finetune_time_elapsed}")
        out_time_file = os.path.join(out_dir, "training_time.txt")
        with open(out_time_file, "a") as f:
            f.write(
                f"elapsed time finetune (includes building graph):\
                     {finetune_time_elapsed}\n"
            )

    return model
