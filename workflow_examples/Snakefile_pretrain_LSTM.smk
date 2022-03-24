import os
import tensorflow as tf
import numpy as np

code_dir = config['code_dir']
# if using river_dl installed with pip this is not needed
import sys
sys.path.insert(1, code_dir)

from river_dl.preproc_utils import asRunConfig
from river_dl.preproc_utils import prep_all_data
from river_dl.evaluate import combined_metrics
from river_dl.postproc_utils import plot_obs
from river_dl.predict import predict_from_io_data
from river_dl.train import train_model
from river_dl import loss_functions as lf
from river_dl.tf_models import LSTMModel

out_dir = config['out_dir']
loss_function = lf.multitask_rmse(config['lambdas'])
pred_weights = config['pred_weights']

rule all:
    input:
        expand("{outdir}/{metric_type}_metrics.csv",
                outdir=out_dir,
                metric_type=['overall', 'month', 'reach', 'month_reach'],
        ),
        expand("{outdir}/asRunConfig.yml",  outdir=out_dir)
        

rule as_run_config:
    output:
        "{outdir}/asRunConfig.yml"
    run:
        asRunConfig(config, code_dir, output[0])

rule prep_io_data:
    input:
         config['sntemp_file'],
         config['obs_file'],
         config['dist_matrix_file']
    output:
        "{outdir}/prepped.npz"
    run:
        prep_all_data(
                  x_data_file=input[0],
                  pretrain_file=input[0],
                  y_data_file=input[1],
                  distfile=input[2],
                  x_vars=config['x_vars'],
                  y_vars_pretrain=config['y_vars_pretrain'],
                  y_vars_finetune=config['y_vars_finetune'],
                  spatial_idx_name='segs_test',
                  time_idx_name='times_test',
                  catch_prop_file=None,
                  exclude_file=None,
                  train_start_date=config['train_start_date'],
                  train_end_date=config['train_end_date'],
                  val_start_date=config['val_start_date'],
                  val_end_date=config['val_end_date'],
                  test_start_date=config['test_start_date'],
                  test_end_date=config['test_end_date'],
                  segs=None,
                  out_file=output[0],
                  trn_offset = config['trn_offset'],
                  tst_val_offset = config['tst_val_offset'])


# Pretrain the model on process based model
rule pre_train:
    input:
        "{outdir}/prepped.npz"
    output:
        directory("{outdir}/pretrained_weights/"),
        "{outdir}/pretrain_log.csv",
        "{outdir}/pretrain_time.txt",
    params:
        # getting the base path to put the training outputs in
        # I omit the last slash (hence '[:-1]' so the split works properly
        weight_dir=lambda wildcards, output: os.path.split(output[0][:-1])[0],
    run:
        data = np.load(input[0])

        optimizer = tf.optimizers.Adam(learning_rate=config['pretrain_learning_rate']) 

        model = LSTMModel(
            config['hidden_size'],
            recurrent_dropout=config['recurrent_dropout'],
            dropout=config['dropout'],
            num_tasks=len(config['y_vars_pretrain']),
        )

        model.compile(optimizer=optimizer, loss=loss_function)
        train_model(model,
                    x_trn = data['x_pre_full'],
                    y_trn = data['y_pre_full'],
                    epochs = config['pt_epochs'],
                    batch_size = 2,
                    seed=config['seed'],
                    # I need to add a trailing slash here. Otherwise the wgts
                    # get saved in the "outdir"
                    weight_dir = output[0] + "/",
                    log_file = output[1],
                    time_file = output[2])


# Finetune/train the model on observations
rule finetune_train:
    input:
        "{outdir}/prepped.npz",
        "{outdir}/pretrained_weights/"
    output:
        directory("{outdir}/finetune_weights/"),
        directory("{outdir}/best_val_weights/"),
        "{outdir}/finetune_log.csv",
        "{outdir}/finetune_time.txt",
    run:
        data = np.load(input[0])
        optimizer = tf.optimizers.Adam(learning_rate=config['finetune_learning_rate']) 

        model = LSTMModel(
            config['hidden_size'],
            recurrent_dropout=config['recurrent_dropout'],
            dropout=config['dropout'],
            num_tasks=len(config['y_vars_pretrain']),
        )

        model.compile(optimizer=optimizer, loss=loss_function)
        model.load_weights(input[1] + "/")
        train_model(model,
                    x_trn = data['x_trn'],
                    y_trn = data['y_obs_trn'],
                    epochs = config['pt_epochs'],
                    batch_size = 2,
                    seed=config['seed'],
                    x_val = data['x_val'],
                    y_val = data['y_obs_val'],
                    # I need to add a trailing slash here. Otherwise the wgts
                    # get saved in the "outdir"
                    weight_dir = output[0] + "/",
                    best_val_weight_dir = output[1] + "/",
                    log_file = output[2],
                    time_file = output[3],
                    early_stop_patience=config['early_stopping'])


rule make_predictions:
    input:
        "{outdir}/"+pred_weights+'/',
        "{outdir}/prepped.npz"
    output:
        "{outdir}/{partition}_preds.feather",
    group: 'train_predict_evaluate'
    run:
        model = LSTMModel(
            config['hidden_size'],
            recurrent_dropout=config['recurrent_dropout'],
            dropout=config['dropout'],
            num_tasks=len(config['y_vars_pretrain']),
        )
        weight_dir = input[0] + '/'
        model.load_weights(weight_dir)
        predict_from_io_data(model=model, 
                             io_data=input[1],
                             partition=wildcards.partition,
                             outfile=output[0],
                             trn_offset = config['trn_offset'],
                             spatial_idx_name='segs_test',
                             time_idx_name='times_test',
                             tst_val_offset = config['tst_val_offset'])


def get_grp_arg(wildcards):
    if wildcards.metric_type == 'overall':
        return None
    elif wildcards.metric_type == 'month':
        return 'month'
    elif wildcards.metric_type == 'reach':
        return 'seg_id_nat'
    elif wildcards.metric_type == 'month_reach':
        return ['seg_id_nat', 'month']


rule combine_metrics:
    input:
         config['obs_file'],
         "{outdir}/trn_preds.feather",
         "{outdir}/val_preds.feather"
    output:
         "{outdir}/{metric_type}_metrics.csv"
    group: 'train_predict_evaluate'
    params:
        grp_arg = get_grp_arg
    run:
        combined_metrics(obs_file=input[0],
                         pred_trn=input[1],
                         pred_val=input[2],
                         spatial_idx_name='segs_test',
                         time_idx_name='times_test',
                         group=params.grp_arg,
                         outfile=output[0])


rule plot_prepped_data:
    input:
        "{outdir}/prepped.npz",
    output:
        "{outdir}/{variable}_{partition}.png",
    run:
        plot_obs(input[0], wildcards.variable, output[0],
                 partition=wildcards.partition)
