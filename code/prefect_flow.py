import os
import pandas as pd
from prefect.engine.results import LocalResult
from prefect import task, Flow, Parameter
from preproc_utils import prep_data, prep_adj_matrix
from train import train_model
from postproc_utils import all_overall, reach_specific_metrics, predict_from_file


@task
def assign_run_id(df, run_id):
    df['run_id'] = i
    return df

@task
def concat_and_write(dfs, outfile):
    df_summary = pd.concat(dfs, axis=0)
    df_summary.to_csv(outfile)
    ds = df_summary.set_index(['run_id', 'variable', 'partition']).to_xarray()
    ds.to_netcdf(outfile.replace('.csv', '.nc'))



with Flow("run_model") as flow:
    in_dir = '../data/in'
    out_dir = '/caldera/projects/usgs/water/iidd/datasci/drb_ml_model/experiments/montague'
    obs_temp_file = os.path.join(in_dir, 'obs_temp_full')
    obs_flow_file = os.path.join(in_dir, 'obs_flow_full')
    sntemp = os.path.join(in_dir, 'uncal_sntemp_input_output')
    dist_file = os.path.join(in_dir, 'distance_matrix.npz')
    x_vars = Parameter('x_vars', default=['seg_rain', 'seg_tave_air'])
    y_vars_default = ['seg_tave_water', 'seg_outflow']
    pt_vars = Parameter('pt_vars', default=y_vars_default)
    ft_vars = Parameter('ft_vars', default=y_vars_default)
    tst_st = Parameter('tst_st', default='2004-09-30')
    tst_y = Parameter('tst_y', default=12)
    exclude = Parameter('exclude', default=None)
    log_q = Parameter('log_q', default=False)
    dist_type = Parameter('dist_type', default="updown")
    pt_epochs = 2
    ft_epochs = 2
    h_units = 20

    res_dir = LocalResult(dir=out_dir)
    prepped_data = prep_data(obs_temp_file, obs_flow_file, sntemp, x_vars,
                             pt_vars, ft_vars, tst_st, tst_y, exclude, log_q,
                             task_args=dict(result=res_dir))
    dist_matrix = prep_adj_matrix(dist_file, dist_type)

    overall_metrics = []
    for i in range(1):
        model_weights = train_model(prepped_data, dist_matrix, pt_epochs,
                                    ft_epochs, h_units, out_dir=out_dir, 
                                    task_args=dict(target=f"model{i}",
                                                   result=res_dir))
        preds_trn_file = os.path.join(out_dir, 'preds_trn.feather') 
        preds_tst_file = os.path.join(out_dir, 'preds_tst.feather')
        preds_trn = predict_from_file(model_weights, prepped_data, dist_matrix,
                                      h_units, 'trn', preds_trn_file,
                                      task_args=dict(target=f"pred_trn{i}",
                                                     result=res_dir))
        preds_tst = predict_from_file(model_weights, prepped_data, dist_matrix,
                                      h_units, 'tst', preds_tst_file,
                                      task_args=dict(target=f"pred_tst{i}",
                                                     result=res_dir))
        df_overall = all_overall(preds_trn, preds_tst, obs_temp_file,
                                 obs_flow_file)
        df_overall = assign_run_id(df_overall, i)
        overall_metrics.append(df_overall)

    concat_and_write(overall_metrics,
                     os.path.join(out_dir, 'overall_summary.csv'))

flow.run(exclude='../../dl_experiments/montague/include_mont.yml')
