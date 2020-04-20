import preproc_utils


def test_all_reals():
    data = preproc_utils.read_process_data('test_data/', subset=False,
                                           finetune_out_vars='both',
                                           n_test_yr=1,
                                           test_start_date='2005-10-01')
    assert data['y_trn_obs_std'].all()
    assert data['y_trn_obs_mean'].all()
