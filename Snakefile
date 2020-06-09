import os

# add scripts dir to path
import sys
code_dir = "code/"
scripts_path =  os.path.abspath(code_dir)
sys.path.insert(0, scripts_path)

from preproc_utils import prep_x, prep_y, prep_adj_matrix
from train import train_model
from postproc_utils import predict, overall_metrics, reach_specific_metrics


rule all:
    input:
         expand("data/out/test_new/{partition}_{variable}_metrics.json",
                partition=['trn', 'tst'], variable=['flow', 'temp']),
         expand("data/out/test_new/{partition}_{variable}_reach_metrics.feather",
                partition=['trn', 'tst'], variable=['flow', 'temp'])

rule prep_x_data:
    input:
        "data/in/uncal_sntemp_input_output",
    output:
        "{outdir}/x_prepped.npz"
    params:
        x_vars=['seg_rain', 'seg_tave_air'],
        test_start_date='2004-09-30',
        n_test_yr=12
    run:
        prep_x(input[0], params.x_vars, params.test_start_date,
               params.n_test_yr, output[0])


rule prep_y_data:
    input:
        "data/in/obs_temp_full.csv",
        "data/in/obs_flow_full.csv",
        "data/in/uncal_sntemp_input_output",
        "{outdir}/x_prepped.npz"
    output:
        "{outdir}/y_prepped.npz"
    params:
        pt_vars=['seg_tave_water', 'seg_outflow'],
        ft_vars=['seg_tave_water', 'seg_outflow'],
    run:
        prep_y(input[0], input[1], input[2], input[3],
               pretrain_vars=params.pt_vars, finetune_vars=params.ft_vars,
               out_file=output[0])

rule prep_adj_matrix_data:
    input:
        "data/in/distance_matrix.npz"
    output:
        "{outdir}/dist_matrix.npz"
    params:
        dist_type='upstream',
    run:
        prep_adj_matrix(input[0], params.dist_type, out_file=output[0])


rule train:
    input:
        "{outdir}/x_prepped.npz",
        "{outdir}/y_prepped.npz",
        "{outdir}/dist_matrix.npz",
    output:
        directory("{outdir}/trained_weights/"),
        directory("{outdir}/pretrained_weights/")
    params:
        n_hidden=20,
        pt_epochs=2,
        ft_epochs=2,
    run:
        train_model(input[0], input[1], input[2], params.pt_epochs,
                    params.ft_epochs, params.n_hidden, wildcards.outdir)


rule make_predictions:
    input: 
        "{outdir}/trained_weights/",
        "{outdir}/x_prepped.npz",
        "{outdir}/y_prepped.npz",
        "{outdir}/dist_matrix.npz"
    params:
        hidden_size=20,
        half_tst=True,
    output:
        "{outdir}/{partition}_preds.feather",
    run:
        predict(input[0], input[1], input[2], input[3], params.hidden_size,
                wildcards.partition, output[0], half_tst=params.half_tst)


rule calc_overall_metrics:
    input:
         "{outdir}/{partition}_preds.feather",
         "data/in/obs_{variable}_full.csv",
    output:
         "{outdir}/{partition}_{variable}_metrics.json"
    run:
         overall_metrics(input[0], input[1], output[0], wildcards.variable)


rule calc_reach_specific_metrics:
    input: 
        "{outdir}/{partition}_preds.feather",
         "data/in/obs_{variable}_full.csv",
    output:
        "{outdir}/{partition}_{variable}_reach_metrics.feather",
    run:
        reach_specific_metrics(input[0], input[1], output[0],
                               wildcards.variable)
