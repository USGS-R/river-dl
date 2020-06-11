import os
from prefect import task, Flow, Parameter
from preproc_utils import prep_data, prep_adj_matrix
from train import train_model
from postproc_utils import overall_metrics, reach_specific_metrics, predict

with Flow("run_model") as flow:
    in_dir = '../data/in'
    out_dir = '../data/out'
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
    pt_epochs = 1
    ft_epochs = 1
    h_units = 20

    prepped_data = prep_data(obs_temp_file, obs_flow_file, sntemp, x_vars,
                             pt_vars, ft_vars, tst_st, tst_y, exclude, log_q)

    dist_matrix = prep_adj_matrix(dist_file, dist_type)

    for i in range(1):
        model = train_model(prepped_data, dist_matrix, pt_epochs, ft_epochs,
                            h_units, out_dir=out_dir)
        preds_trn_file = os.path.join(out_dir, 'preds_trn.feather')
        preds_tst_file = os.path.join(out_dir, 'preds_tst.feather')
        preds_trn = predict(model, prepped_data, 'trn', preds_trn_file,
                            num_segs=456)
        preds_tst = predict(model, prepped_data, 'tst', preds_tst_file,
                            num_segs=456)
        overall_trn_temp = overall_metrics(preds_trn, obs_temp_file,
                                           'temp_trn_metrics.json', 'temp')
        overall_tst_temp = overall_metrics(preds_tst, obs_temp_file,
                                           'temp_tst_metrics.json', 'temp')
        overall_trn_flow = overall_metrics(preds_trn, obs_flow_file,
                                           'flow_trn_metrics.json', 'flow')
        overall_tst_flow = overall_metrics(preds_tst, obs_flow_file,
                                           'flow_tst_metrics.json', 'flow')
        reach_trn_temp = reach_specific_metrics(preds_trn, obs_temp_file,
                                                'temp_trn_metrics.json', 'temp')
        reach_tst_temp = reach_specific_metrics(preds_tst, obs_temp_file,
                                                'temp_tst_metrics.json', 'temp')
        reach_trn_flow = reach_specific_metrics(preds_trn, obs_flow_file,
                                                'flow_trn_metrics.json', 'flow')
        reach_tst_flow = reach_specific_metrics(preds_tst, obs_flow_file,
                                                'flow_tst_metrics.json', 'flow')

flow.run(exclude=None)
