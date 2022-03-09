import os
import numpy as np
import torch
import torch.optim as optim

code_dir = config['code_dir']
# if using river_dl installed with pip this is not needed
import sys
sys.path.insert(1, code_dir)

from river_dl.preproc_utils import prep_all_data
from river_dl.evaluate import combined_metrics
from river_dl.postproc_utils import plot_obs
from river_dl.predict import predict_from_io_data
from river_dl.torch_utils import train_torch
from river_dl.torch_utils import rmse_masked, rmse_masked_gw
from river_dl.gw_utils import prep_annual_signal_data, calc_pred_ann_temp,calc_gw_metrics
from river_dl.torch_models import RGCN_v1

out_dir = config['out_dir'] 
pred_weights = config['pred_weights']


module base_workflow:
    snakefile: "Snakefile_rgcn_pytorch.smk"
    config: config

use rule * from base_workflow as base_*

#this allows us to import all the rules from Snakefile but write a custom train_model_local_or_cpu rule
use rule finetune_train from base_workflow as base_finetune_train with:
    output:
        ""

#modify rule all to include the additional gw output files        
use rule all from base_workflow as base_all with:
    input:
        expand("{outdir}/{metric_type}_metrics.csv",
                outdir=out_dir,
                metric_type=['overall', 'month', 'reach', 'month_reach'],
        ),
        expand("{outdir}/GW_stats_{partition}.csv",
                outdir=out_dir,
                partition=['trn', 'tst','val']
        ),
        expand("{outdir}/GW_summary.csv", outdir=out_dir
        ),
        expand("{outdir}/asRunConfig.yml",outdir=out_dir)

        
 
rule prep_ann_temp:
    input:
         config['obs_file'],
         config['sntemp_file'],
         "{outdir}/prepped.npz",
    output:
        "{outdir}/prepped_withGW.npz",
    run:
        prep_annual_signal_data(input[0], input[1], input[2],
                  train_start_date=config['train_start_date'],
                  train_end_date=config['train_end_date'],
                  val_start_date=config['val_start_date'],
                  val_end_date=config['val_end_date'],
                  test_start_date=config['test_start_date'],
                  test_end_date=config['test_end_date'], 
                  out_file=output[0],
                  extraResSegments = config['extraResSegments'],
                  reach_file= config['reach_attr_file'],
                  gw_loss_type=config['gw_loss_type'],
                  trn_offset = config['trn_offset'],
                  tst_val_offset = config['tst_val_offset'])


#get the GW loss parameters
def get_gw_loss(input_data, temp_var="temp_c"):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    io_data=np.load(input_data)
    temp_index = np.where(io_data['y_obs_vars']==temp_var)[0]
    temp_mean = torch.from_numpy(io_data['y_mean'][temp_index]).to(device)
    temp_sd = torch.from_numpy(io_data['y_std'][temp_index]).to(device)
    gw_mean = io_data['GW_mean']
    gw_std = io_data['GW_std']
    return rmse_masked_gw(rmse_masked,temp_index,temp_mean, temp_sd,gw_mean=gw_mean, gw_std = gw_std,lambda_Ar=config['lambdas_gw'][0],lambda_delPhi=config['lambdas_gw'][1],lambda_Tmean=config['lambdas_gw'][2], num_task=len(io_data['y_obs_vars']), gw_type=config['gw_loss_type'])


# Finetune/train the model on observations
rule finetune_train:
    input:
        "{outdir}/prepped_withGW.npz",
        "{outdir}/pretrained_weights.pth",
        "{outdir}/pretrain_log.csv",
    output:
        "{outdir}/finetuned_weights.pth",
        "{outdir}/finetune_log.csv",
    run:
        data = np.load(input[0])
        temp_air_index = np.where(data['x_vars'] == 'seg_tave_air')[0]
        air_unscaled = data['x_trn'][:, :, temp_air_index] * data['x_std'][temp_air_index] + \
                       data['x_mean'][temp_air_index]
        y_trn_obs = np.concatenate(
            [data["y_obs_trn"], data["GW_trn_reshape"], air_unscaled], axis=2
        )
        air_val = data['x_val'][:, :, temp_air_index] * data['x_std'][temp_air_index] + data['x_mean'][
            temp_air_index]
        y_val_obs = np.concatenate(
            [data["y_obs_val"], data["GW_val_reshape"], air_val], axis=2
        )
        num_segs = len(np.unique(data['ids_trn']))
        adj_mx = data['dist_matrix']
        in_dim = len(data['x_vars'])
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = RGCN_v1(in_dim,config['hidden_size'],adj_mx,device=device, seed=config['seed'])
        opt = optim.Adam(model.parameters(),lr=config['finetune_learning_rate'])
        model.load_state_dict(torch.load(input[1]))
        train_torch(model,
            loss_function=get_gw_loss(input[0]),
            optimizer=opt,
            x_train=data['x_trn'],
            y_train=y_trn_obs,
            x_val=data['x_val'],
            y_val=y_val_obs,
            max_epochs=config['ft_epochs'],
            early_stopping_patience=config['early_stopping'],
            batch_size = num_segs,
            weights_file=output[0],
            log_file=output[1],
            device=device)


                 
rule compile_pred_GW_stats:
    input:
        "{outdir}/prepped_withGW.npz",
        "{outdir}/trn_preds.feather",
        "{outdir}/tst_preds.feather",
        "{outdir}/val_preds.feather"
    output:
        "{outdir}/GW_stats_trn.csv",
        "{outdir}/GW_stats_tst.csv",
        "{outdir}/GW_stats_val.csv",
    run: 
        calc_pred_ann_temp(input[0],input[1],input[2], input[3], output[0], output[1], output[2])
        
rule calc_gw_summary_metrics:
    input:
        "{outdir}/GW_stats_trn.csv",
        "{outdir}/GW_stats_tst.csv",
        "{outdir}/GW_stats_val.csv",
    output:
        "{outdir}/GW_summary.csv",
        "{outdir}/GW_scatter.png",
        "{outdir}/GW_boxplot.png",
    run:
        calc_gw_metrics(input[0],input[1],input[2],output[0], output[1], output[2])
 
