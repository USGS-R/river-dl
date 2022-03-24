import os
from river_dl.preproc_utils import asRunConfig
from river_dl.preproc_utils import prep_all_data
from river_dl.torch_utils import train_torch
from river_dl.torch_utils import rmse_masked
from river_dl.evaluate import combined_metrics
import numpy as np
import torch
import torch.optim as optim

from river_dl.torch_models import RGCN_v1
from river_dl.predict import predict_from_io_data

out_dir = config['out_dir']
code_dir = ''

### Set up your hyperparameters as wild cards
offsets = [15,.25, .5, 1]
sequence_length = [60,180,365]
run_id = ['RGCN_Hypertune']

rule all:
    input:
        expand("{outdir}/{offset}_{seq_length}/{run_id}/{metric_type}_metrics.csv",
            outdir=out_dir,
            offset=offsets,
            seq_length=sequence_length,
            metric_type=['overall', 'month', 'reach'],
            run_id=run_id
        ),
        expand("{outdir}/asRunConfig_{run_id}.yml", outdir=out_dir, run_id = run_id),
        expand("{outdir}/Snakefile_{run_id}", outdir=out_dir, run_id = run_id),

rule as_run_config:
    output:
        "{outdir}/asRunConfig_{run_id}.yml"
    group: "prep"
    run:
        asRunConfig(config,code_dir,output[0])

rule copy_snakefile:
    output:
        "{outdir}/Snakefile_{run_id}"
    group: "prep"
    shell:
        """
        scp Snakefile {output[0]}
        """


rule prep_io_data:
    input:
        config['sntemp_file'],
        config['obs_file'],
        config['dist_matrix_file'],
    output:
        "{outdir}/{offset}_{seq_length}/prepped.npz"
    threads: 2
    group: "prep"
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
            trn_offset= float(wildcards.offset),
            tst_val_offset= float(wildcards.offset),
            seq_len=int(wildcards.seq_length),
        )

# Pretrain the model on process based model
rule pre_train:
    input:
        "{outdir}/{offset}_{seq_length}/prepped.npz"
    output:
        "{outdir}/{offset}_{seq_length}/{run_id}/pretrained_weights.pth",
        "{outdir}/{offset}_{seq_length}/{run_id}/pretrain_log.csv",
    threads: 4
    group: 'train'
    run:
        os.system("module load analytics cuda11.3/toolkit/11.3.0")
        os.system("export LD_LIBRARY_PATH=/cm/shared/apps/nvidia/TensorRT-6.0.1.5/lib:/cm/shared/apps/nvidia/cudnn_8.0.5/lib64:$LD_LIBRARY_PATH")

        data = np.load(input[0])
        num_segs = len(np.unique(data['ids_trn']))
        adj_mx = data['dist_matrix']
        in_dim = len(data['x_vars'])
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = RGCN_v1(in_dim,config['hidden_size'],adj_mx,device=device)
        opt = optim.Adam(model.parameters(),lr=config['pretrain_learning_rate'])

        train_torch(model,
            loss_function=rmse_masked,
            optimizer=opt,
            x_train=data['x_pre_full'],
            y_train=data['y_pre_full'],
            max_epochs=config['pt_epochs'],
            batch_size=num_segs,
            weights_file=output[0],
            log_file=output[1],
            device=device,
            keep_portion=float(wildcards.offset))



# Finetune/train the model on observations
rule finetune_train:
    input:
        "{outdir}/{offset}_{seq_length}/prepped.npz",
        "{outdir}/{offset}_{seq_length}/{run_id}/pretrained_weights.pth",
        "{outdir}/{offset}_{seq_length}/{run_id}/pretrain_log.csv",
    output:
        "{outdir}/{offset}_{seq_length}/{run_id}/finetuned_weights.pth",
        "{outdir}/{offset}_{seq_length}/{run_id}/finetune_log.csv",
    threads: 4
    group: 'train'
    run:
        os.system("module load analytics cuda11.3/toolkit/11.3.0")
        os.system("export LD_LIBRARY_PATH=/cm/shared/apps/nvidia/TensorRT-6.0.1.5/lib:/cm/shared/apps/nvidia/cudnn_8.0.5/lib64:$LD_LIBRARY_PATH")

        data = np.load(input[0])
        num_segs = len(np.unique(data['ids_trn']))
        adj_mx = data['dist_matrix']
        in_dim = len(data['x_vars'])
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = RGCN_v1(in_dim,config['hidden_size'],adj_mx,device=device)
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
            batch_size=num_segs,
            weights_file=output[0],
            log_file=output[1],
            device=device,
            keep_portion=float(wildcards.offset))


rule make_predictions:
    input:
        "{outdir}/{offset}_{seq_length}/{run_id}/finetuned_weights.pth",
        "{outdir}/{offset}_{seq_length}/prepped.npz"
    output:
        "{outdir}/{offset}_{seq_length}/{run_id}/{partition}_preds.feather",
    group: 'train'
    threads: 3
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
            trn_offset=float(wildcards.offset),
            tst_val_offset=float(wildcards.offset))


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
        "{outdir}/{offset}_{seq_length}/{run_id}/trn_preds.feather",
        "{outdir}/{offset}_{seq_length}/{run_id}/val_preds.feather",
        "{outdir}/{offset}_{seq_length}/{run_id}/tst_preds.feather"
    output:
        "{outdir}/{offset}_{seq_length}/{run_id}/{metric_type}_metrics.csv"
    group: 'predict_eval'
    threads: 3
    params:
        grp_arg=get_grp_arg
    run:
        combined_metrics(obs_file=input[0],
            pred_trn=input[1],
            pred_val=input[2],
            pred_tst=input[3],
            group=params.grp_arg,
            outfile=output[0])
