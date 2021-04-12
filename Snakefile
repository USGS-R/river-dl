import os
import pandas as pd

# this is needed for running on HPC if using GPU
shell.prefix("module load analytics cuda10.0/toolkit/10.0.130 \n \
              run_training --no-node-list -e /home/jsadler/.conda/envs/myenv")

# add scripts dir to path

from river_dl.preproc_utils import prep_data
from river_dl.postproc_utils import predict, combined_metrics, plot_obs
from river_dl.train import train_model

out_dir = config['out_dir']
code_dir = config['code_dir']

rule all:
    input:
        expand("{outdir}/{metric_type}_metrics.csv",
                outdir=out_dir,
                metric_type=['overall', 'month', 'reach', 'month_reach'],
        ),
        expand( "{outdir}/{plt_variable}_{partition}.png",
                outdir=out_dir,
                plt_variable=['temp', 'flow'],
                partition=['trn', 'tst'],
        ),

rule prep_io_data:
    input:
         config['obs_temp'],
         config['obs_flow'],
         config['sntemp_file'],
         config['dist_matrix'],
    output:
        "{outdir}/prepped.npz"
    run:
        prep_data(input[0], input[1], input[2], input[3], config['x_vars'],
                  catch_prop_file=None,
                  exclude_file=None,
                  test_start_date=config['test_start_date'],
                  primary_variable=config['primary_variable'],
                  log_q=False, segs=None,
                  n_test_yr=config['n_test_yr'], out_file=output[0])


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
#    group: 'train_predict_evaluate'
#    shell:
#        """
#        "python {code_dir}/train_model.py -o {params.run_dir} -i {input[0]} -m {output[0]} -p {params.pt_epochs} -f {params.ft_epochs} --lamb {params.lamb} --model rgcn -s 135"
#        """

# use "train_model" if wanting to use CPU or local GPU
rule train_model:
    input:
        "{outdir}/prepped.npz"
    output:
        directory("{outdir}/trained_model/"),
        directory("{outdir}/pretrained_model/"),
    params:
        # getting the base path to put the training outputs in
        # I omit the last slash (hence '[:-1]' so the split works properly
        run_dir=lambda wildcards, output: os.path.split(output[0][:-1])[0],
    run:
        train_model(input[0], config['pt_epochs'], config['ft_epochs'], 20, params.run_dir,
                    model_type='rgcn', lamb=config['lamb'])

rule make_predictions:
    input:
        "{outdir}/trained_model/",
        "{outdir}/prepped.npz"
    output:
        "{outdir}/{partition}_preds.feather",
    group: 'train_predict_evaluate'
    run:
        model_dir = input[0] + '/'
        predict(model_dir, input[1], partition=wildcards.partition,
                outfile=output[0], half_tst=config['half_test'],
                logged_q=False)


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
         config['obs_temp'],
         config['obs_flow'],
         "{outdir}/trn_preds.feather",
         "{outdir}/tst_preds.feather"
    output:
         "{outdir}/{metric_type}_metrics.csv"
    group: 'train_predict_evaluate'
    params:
        grp_arg = get_grp_arg
    run:
        combined_metrics(obs_temp=input[0],
                         obs_flow=input[1],
                         pred_trn=input[2],
                         pred_tst=input[3],
                         grp=params.grp_arg,
                         outfile=output[0])


rule plot_prepped_data:
    input:
        "{outdir}/prepped.npz",
    output:
        "{outdir}/{variable}_{partition}.png",
    run:
        plot_obs(input[0], wildcards.variable, output[0],
                 partition=wildcards.partition)
