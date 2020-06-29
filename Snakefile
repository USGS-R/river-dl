import os
import pandas as pd

# add scripts dir to path
import sys
code_dir = "../../drb-dl-model/code/"
scripts_path =  os.path.abspath(code_dir)
sys.path.insert(0, scripts_path)
shell.prefix("module load analytics cuda100/toolkit/10.0.130 \n \
              run_training -e /home/jsadler/.conda/envs/rgcn --no-node-list ")

from preproc_utils import prep_data
from postproc_utils import predict_from_file, overall_metrics, reach_specific_metrics, combine_csvs
data_dir = "/home/jsadler/drb-dl-model/data/in"
out_dir = "/caldera/projects/usgs/water/iidd/datasci/drb_ml_model/experiments"
exper = "snk_nw"
num_replicates = 5


rule all:
    input:
        f"{out_dir}/ex_{exper}/overall_metrics.nc"


rule prep_io_data:
    input:
         f"{data_dir}/obs_temp_full",
         f"{data_dir}/obs_flow_full",
         f"{data_dir}/uncal_sntemp_input_output",
         f"{data_dir}/distance_matrix.npz",
         f"{data_dir}/seg_attr_drb.feather",
    output:
        "{outdir}/prepped.npz"
    params:
        pt_vars='both',
        ft_vars='both',
        x_vars =['seg_rain', 'seg_tave_air', 'seginc_swrad', 'seg_length',
                 'seginc_potet', 'seg_slope', 'seg_humid', 'seg_elev'],
        test_start_date='2004-09-30',
        n_test_yr=12
    run:
        prep_data(input[0], input[1], input[2], input[3], params.x_vars,
                  input[4], pretrain_vars=params.pt_vars,
                  finetune_vars=params.ft_vars,
                  test_start_date=params.test_start_date,
                  n_test_yr=params.n_test_yr, out_file=output[0])


# this must be a shell command so that the correct GPU libraries are loaded and used
rule train:
    input:
         "{outdir}/ex_{experiment}/prepped.npz"
    output:
        directory("{outdir}/ex_{experiment}/{run_id}/trained_weights/"),
        directory("{outdir}/ex_{experiment}/{run_id}/pretrained_weights/")
    params:
        run_dir=lambda wildcards, output: os.path.split(output[0])[0],
        pt_epochs=200,
        ft_epochs=100
    shell:
         """
         "python {code_dir}/train_model.py -o {params.run_dir} -i {input[0]} -m {output[0]} -p {params.pt_epochs} -f {params.ft_epochs}" 
         """

rule make_predictions:
    input: 
        "{outdir}/ex_{experiment}/{run_id}/trained_weights/",
        "{outdir}/ex_{experiment}/prepped.npz",
    params:
        hidden_size=20,
        half_tst=True,
    output:
        "{outdir}/ex_{experiment}/{run_id}/{partition}_preds.feather",
    run:
        predict_from_file(input[0], input[1], params.hidden_size,
                          wildcards.partition, output[0], half_tst=params.half_tst)


rule calc_overall_metrics:
    input:
         "{outdir}/ex_{experiment}/{run_id}/{partition}_preds.feather",
         expand("{data_dir}/obs_{variable}_full", data_dir=data_dir, allow_missing=True)
    output:
         "{outdir}/ex_{experiment}/{run_id}/{partition}_{variable}_metrics.csv"
    run:
         overall_metrics(input[0], input[1], wildcards.variable, wildcards.partition, output[0])


rule combined_metrics:
    input:
        expand("{outdir}/ex_{experiment}/{run_id}/{partition}_{variable}_metrics.csv", partition=['trn', 'tst'],
               variable=['temp', 'flow'], allow_missing=True)
    output:
         "{outdir}/ex_{experiment}/{run_id}/combined_metrics.csv"
    run:
        combine_csvs(input, output[0])


rule combine_replicates:
    input:
         expand("{outdir}/ex_{experiment}/{run_id}/combined_metrics.csv",
                run_id=list(range(num_replicates)), allow_missing=True)
    output:
         "{outdir}/ex_{experiment}/overall_metrics.nc"
    run:
        df_list = []
        for i, metric_file in enumerate(input):
            df = pd.read_csv(metric_file)
            df['run_id'] = i
            df_list.append(df)
        df = pd.concat(df_list, axis=0)
        ds = df.set_index(['run_id', 'variable', 'partition']).to_xarray()
        ds.to_netcdf(output[0])


rule calc_reach_specific_metrics:
    input: 
        "{outdir}/ex_{experiment}/{run_id}/{partition}_preds.feather",
        expand("{data_dir}/obs_{variable}_full.csv", data_dir=data_dir, allow_missing=True)
    output:
        "{outdir}/ex_{experiment}/{run_id}/{partition}_{variable}_reach_metrics.feather",
    run:
        reach_specific_metrics(input[0], input[1], output[0],
                               wildcards.variable)
