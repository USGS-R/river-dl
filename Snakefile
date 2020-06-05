import os

# add scripts dir to path
import sys
code_dir = "code/"
scripts_path =  os.path.abspath(code_dir)
sys.path.insert(0, scripts_path)

from preproc_utils import prep_x, prep_y, prep_adj_matrix
from train import train_model
from postproc_utils import predict

outdir = ''

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
        pt_vars=['seg_tave_water', 'seg_tave_rain'],
        ft_vars=['seg_tave_water', 'seg_tave_rain'],
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
        directory("{outdir}/trained_weights"),
        directory("{outdir}/pretrained_weights")
    params:
        n_hidden=20,
        pt_epochs=200,
        ft_epochs=100,
    run:
        train_model(input[0], input[1], input[2], params.pt_epochs,
                    params.ft_epochs, params.n_hidden, outdir)


rule make_predictions:
    input: 
        "{outdir}/trained_weights",
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
                params.half_tst, wildcards.partition, output[0])


rule Exp_A_evaluate:
    input: 
        "{outdir}/{partition}_preds.feather",
         "data/in/obs_temp_full.csv",
         "data/in/obs_flow_full.csv",
    output:
        "{outdir}/{partition}_metrics.json",
        "{outdir}/{partition}_temp_reach_metrics.feather",
        "{outdir}/{partition}_flow_reach_metrics.feather"
    shell:
        "python  {code_dir}\\eval.py -o experiments\\A\\{wildcards.a_vers}\\{wildcards.model_id}\\ -p {input[0]} -T {input[1]} -Q {input[2]} -s {wildcards.partition}"


rule Exp_A_results:
    input:
        expand("experiments\\A\\{a_vers}\\{model_id}\\tst_metrics.json",
                model_id=utils.get_model_ids('A'), allow_missing=True)
    output:
        "experiments\\A\\{a_vers}\\summary.csv"
    run:
        utils.summarize_results(input, output[0])


rule Exp_A_meta_results:
    input:
        expand("experiments\\A\\{a_vers}\\summary.csv", a_vers=a_versions)
    output:
        "experiments\\A\\meta_summary.csv"
    run:
        utils.meta_results(input, output[0])

