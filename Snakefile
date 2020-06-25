import os
import pandas as pd

num_replicates = 5

out_dir = "../data/out"
exper = "snk_nw"

rule all:
    input:
        f"{out_dir}/ex_{exper}/overall_metrics.nc"

rule prep_data:
    output:
        "{outdir}/ex_{experiment}/prepped.npz"
    params:
        pt_vars='both',
        ft_vars='both',
    shell:
         """
         python prep_data.py -o {output[0]} -p {params.pt_vars} -f {params.ft_vars}
         """

rule train_predict_eval:
    input:
         "{outdir}/ex_{experiment}/prepped.npz"
    output:
         "{outdir}/ex_{experiment}/{run_id}/metrics.csv"
    params:
        run_dir=lambda wildcards, output: os.path.split(output[0])[0]
    shell:
         """
         python trn_pred_eval.py -o {params.run_dir} -i {input[0]} -m {output[0]}
         """

rule combine_replicates:
    input:
         expand("{outdir}/ex_{experiment}/{run_id}/metrics.csv",
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
