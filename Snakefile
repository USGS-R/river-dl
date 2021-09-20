import os

from river_dl.preproc_utils import prep_all_data
from river_dl.evaluate import combined_metrics
from river_dl.postproc_utils import plot_obs
from river_dl.predict import predict_from_io_data
from river_dl.train import train_model
from river_dl import loss_functions as lf

out_dir = config['out_dir']
code_dir = config['code_dir']
loss_function = lf.multitask_rmse(config['lambdas'])

rule all:
    input:
        expand("{outdir}/{metric_type}_metrics.csv",
                outdir=out_dir,
                metric_type=['overall', 'month', 'reach', 'month_reach'],
        ),
        expand("{outdir}/config.yml",
                outdir = out_dir,
        )


rule copy_config:
    output:
        "{outdir}/config.yml"
    shell:
        """
        scp config.yml {output[0]}
        """

rule prep_io_data:
    input:
        config['sntemp_file'],
        config['obs_file'],
        config['dist_matrix'],
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
                  catch_prop_file=None,
                  exclude_file=None,
                  train_start_date=config['train_start_date'],
                  train_end_date=config['train_end_date'],
                  val_start_date=config['val_start_date'],
                  val_end_date=config['val_end_date'],
                  test_start_date=config['test_start_date'],
                  test_end_date=config['test_end_date'],
                  segs=None,
                  out_file=output[0])


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
        "{outdir}/prepped.npz"
    output:
        directory("{outdir}/trained_weights/"),
        directory("{outdir}/pretrained_weights/"),
    params:
        # getting the base path to put the training outputs in
        # I omit the last slash (hence '[:-1]' so the split works properly
        run_dir=lambda wildcards, output: os.path.split(output[0][:-1])[0],
    run:
        train_model(input[0], config['pt_epochs'], config['ft_epochs'], config['hidden_size'],
                    loss_func_ft=loss_function, out_dir=params.run_dir, model_type='rgcn', num_tasks=len(config['y_vars_finetune']))

rule make_predictions:
    input:
        "{outdir}/trained_weights/",
        "{outdir}/prepped.npz"
    output:
        "{outdir}/{partition}_preds.feather",
    group: 'train_predict_evaluate'
    run:
        model_dir = input[0] + '/'
        predict_from_io_data(model_type='rgcn', model_weights_dir=model_dir,
                             hidden_size=config['hidden_size'], io_data=input[1],
                             partition=wildcards.partition, outfile=output[0],
                             num_tasks=len(config['y_vars_finetune']))


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
