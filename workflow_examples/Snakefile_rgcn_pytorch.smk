import os
import numpy as np
import torch
import torch.optim as optim

code_dir = config['code_dir']
# if using river_dl installed with pip this is not needed
import sys
sys.path.insert(1, code_dir)

from river_dl.preproc_utils import asRunConfig
from river_dl.preproc_utils import prep_all_data
from river_dl.torch_utils import train_torch
from river_dl.torch_utils import rmse_masked
from river_dl.evaluate import combined_metrics
from river_dl.torch_models import RGCN_v1
from river_dl.predict import predict_from_io_data

out_dir = config['out_dir']

rule all:
    input:
        f"{out_dir}/finetuned_weights.pth",
        f"{out_dir}/finetune_log.csv",
        expand("{outdir}/{metric_type}_metrics.csv",
                outdir=out_dir,
                metric_type=['overall', 'month', 'reach', 'month_reach'],
        ),
        expand("{outdir}/asRunConfig.yml",  outdir=out_dir),
        expand("{outdir}/Snakefile", outdir=out_dir),        

rule as_run_config:
    output:
        "{outdir}/asRunConfig.yml"
    run:
        asRunConfig(config, code_dir, output[0])

rule copy_snakefile:
    output:
        "{outdir}/Snakefile"
    #group: "prep"
    shell:
        """
        scp Snakefile_rgcn_pytorch.smk {output[0]}
        """

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
        "{outdir}/pretrained_weights.pth",
        "{outdir}/pretrain_log.csv",
    params:
        # getting the base path to put the training outputs in
        # I omit the last slash (hence '[:-1]' so the split works properly
        weight_dir=lambda wildcards, output: os.path.split(output[0][:-1])[0],
    run:
        data = np.load(input[0])
        num_segs = len(np.unique(data['ids_trn']))
        adj_mx = data['dist_matrix']
        in_dim = len(data['x_vars'])
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = RGCN_v1(in_dim, config['hidden_size'], adj_mx,device=device, seed=config['seed'])
        opt = optim.Adam(model.parameters(),lr=config['pretrain_learning_rate'])
        train_torch(model,
                    loss_function = rmse_masked,
                    optimizer= opt,
                    x_train= data['x_trn'],
                    y_train = data['y_pre_trn'],
                    max_epochs = config['pt_epochs'],
                    early_stopping_patience=config['early_stopping'],
                    batch_size = num_segs,
                    weights_file = output[0],
                    log_file = output[1],
                    device=device)


# Finetune/train the model on observations
rule finetune_train:
    input:
        "{outdir}/prepped.npz",
        "{outdir}/pretrained_weights.pth",
        "{outdir}/pretrain_log.csv",
    output:
        "{outdir}/finetuned_weights.pth",
        "{outdir}/finetune_log.csv",
    run:
        data = np.load(input[0])
        num_segs = len(np.unique(data['ids_trn']))
        adj_mx = data['dist_matrix']
        in_dim = len(data['x_vars'])
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = RGCN_v1(in_dim,config['hidden_size'],adj_mx,device=device, seed=config['seed'])
        opt = optim.Adam(model.parameters(),lr=config['finetune_learning_rate'])
        scheduler = optim.lr_scheduler.LambdaLR(opt,lr_lambda=lambda epoch: 0.97 ** epoch)
        model.load_state_dict(torch.load(input[1]))
        train_torch(model,
            loss_function=rmse_masked,
            optimizer=opt,
            x_train=data['x_trn'],
            y_train=data['y_obs_trn'],
            x_val=data['x_val'],
            y_val=data['y_obs_val'],
            max_epochs=config['ft_epochs'],
            early_stopping_patience=config['early_stopping'],
            batch_size = num_segs,
            weights_file=output[0],
            log_file=output[1],
            device=device)



rule make_predictions:
    input:
        "{outdir}/finetuned_weights.pth",
        "{outdir}/prepped.npz"
    output:
        "{outdir}/{partition}_preds.feather",
    group: 'train_predict_evaluate'
    run:
        data = np.load(input[1])
        adj_mx = data['dist_matrix']
        in_dim = len(data['x_vars'])
        model = RGCN_v1(in_dim,config['hidden_size'],adj_mx)
        opt = optim.Adam(model.parameters(),lr=config['finetune_learning_rate'])
        model.load_state_dict(torch.load(input[0]))
        predict_from_io_data(model=model, 
                             io_data=input[1],
                             partition=wildcards.partition,
                             outfile=output[0],
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


