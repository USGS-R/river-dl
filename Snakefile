import os

from river_dl.preproc_utils import asRunConfig
from river_dl.preproc_utils import prep_all_data
from river_dl.evaluate import combined_metrics
from river_dl.postproc_utils import plot_obs
from river_dl.predict import predict_from_io_data
from river_dl.train import train_model
from river_dl import loss_functions as lf
import numpy as np

out_dir = config['out_dir']
code_dir = config['code_dir']
loss_function = lf.multitask_rmse(config['lambdas'])

rule all:
    input:
        expand("{outdir}/exp_{exp}/{metric_type}_metrics.csv",
                outdir=out_dir,
                metric_type=['overall', 'month', 'reach', 'month_reach'],
                exp = ['pretrain', 'PBIn']
        ),
        expand("{outdir}/asRunConfig.yml", outdir=out_dir)
        

rule as_run_config:
    output:
        "{outdir}/asRunConfig.yml"
    run:
        asRunConfig(config,output[0])


def combine_inputs_and_pretrain_outputs(input_data, outfile):
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

    np.savez_compressed()


rule prep_io_data_pb_in:
    input:
        "{outdir}/exp_pretrain/prepped.npz"
    output:
        "{outdir}/exp_PBIn/prepped.npz"
    run:
        combine_inputs_and_pretrain_outputs(input[0], output[0])


rule prep_io_data_pretrain:
    input:
         config['sntemp_file'],
         config['obs_file'],
         config['dist_matrix_file'],
    output:
        "{outdir}/exp_pretrain/prepped.npz"
    run:
        prep_all_data(
                  x_data_file=input[0],
                  pretrain_file=input[0],
                  y_data_file=input[1],
                  distfile=input[2],
                  x_vars=config['x_vars'],
                  y_vars_pretrain=config['y_vars_pretrain'],
                  y_vars_finetune=config['y_vars_finetune'],
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


# use "train" if wanting to use GPU on HPC
# rule train:
#    input:
#        "{outdir}/prepped.npz"
#    output:
#        directory("{outdir}/trained_model/"),
#        directory("{outdir}/pretrained_model/"),
#    params:
#        # getting the base path to put the training outputs in
#        # I omit the last slash (hence '[:-1]' so the split works properly
#        run_dir=lambda wildcards, output: os.path.split(output[0][:-1])[0],
#        pt_epochs=config['pt_epochs'],
#        ft_epochs=config['ft_epochs'],
#        lamb=config['lamb'],
#    shell:
#        """
#        module load analytics cuda10.1/toolkit/10.1.105 
#        run_training -e /home/jsadler/.conda/envs/rgcn --no-node-list "python {code_dir}/train_model_cli.py -o {params.run_dir} -i {input[0]} -p {params.pt_epochs} -f {params.ft_epochs} --lambdas {params.lamb} --loss_func multitask_rmse --model rgcn -s 135"
#        """


# use "train_model" if wanting to use CPU or local GPU
rule train_model_local_or_cpu:
    input:
        "{outdir}/exp_{exp}/prepped.npz"
    output:
        directory("{outdir}/exp_{exp}/trained_weights/"),
        #directory("{outdir}/pretrained_weights/"),
    params:
        # getting the base path to put the training outputs in
        # I omit the last slash (hence '[:-1]' so the split works properly
        run_dir=lambda wildcards, output: os.path.split(output[0][:-1])[0],
    run:
        train_model(input[0], config['pt_epochs'], config['ft_epochs'], config['hidden_size'],
                    loss_func_ft=loss_function, out_dir=params.run_dir, model_type='rgcn', num_tasks=len(config['y_vars_finetune']),
                    updated_io_data=output[1])

rule make_predictions:
    input:
        "{outdir}/exp_{exp}/trained_weights/",
        "{outdir}/exp_{exp}/prepped.npz"
    output:
        "{outdir}/exp_{exp}/{partition}_preds.feather",
    group: 'train_predict_evaluate'
    run:
        model_dir = input[0] + '/'
        predict_from_io_data(model_type='rgcn', model_weights_dir=model_dir,
                             hidden_size=config['hidden_size'], io_data=input[1],
                             partition=wildcards.partition, outfile=output[0],
                             num_tasks=len(config['y_vars_finetune']),
                             trn_offset = config['trn_offset'],
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
         "{outdir}/exp_{exp}/trn_preds.feather",
         "{outdir}/exp_{exp}/val_preds.feather"
    output:
         "{outdir}/exp_{exp}/{metric_type}_metrics.csv"
    group: 'train_predict_evaluate'
    params:
        grp_arg = get_grp_arg
    run:
        combined_metrics(obs_file=input[0],
                         pred_trn=input[1],
                         pred_val=input[2],
                         group=params.grp_arg,
                         outfile=output[0])


rule plot_prepped_data:
    input:
        "{outdir}/exp_{exp}/prepped.npz",
    output:
        "{outdir}/exp_{exp}/{variable}_{partition}.png",
    run:
        plot_obs(input[0], wildcards.variable, output[0],
                 partition=wildcards.partition)
