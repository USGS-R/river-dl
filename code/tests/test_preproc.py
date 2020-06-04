import pandas as pd
import numpy as np
import xarray as xr
import preproc_utils
import postproc_utils

# segments in test dataset
segments = ['2007', '2012']
# date range in test dataset
min_date = '2004-09-15'
max_date = '2006-10-15'

obs_flow = 'test_data/obs_flow.csv'
obs_temp = 'test_data/obs_temp.csv'
sntemp = 'test_data/test_data'


def test_weight_creation():
    n_segs = 4
    n_dates = 5
    num_pretrain_vars = 4
    dims = (n_segs, n_dates)
    pre_das = [xr.DataArray(np.random.rand(*dims), dims=['seg_id_nat', 'date'])
               for i in range(num_pretrain_vars)]
    ds_pre = xr.Dataset({'a': pre_das[0], 'b': pre_das[1], 'c': pre_das[2],
                         'd': pre_das[3]})

    num_finetune_vars = 4
    ft_das = [xr.DataArray(np.random.rand(*dims), dims=['seg_id_nat', 'date'])
              for i in range(num_finetune_vars)]

    ds_ft = xr.Dataset({'a': ft_das[0], 'b': ft_das[1]})

    ds_ft['a'][2, 3] = np.nan
    ds_ft['a'][0, 0] = np.nan
    ds_ft['b'][1, 1:3] = np.nan

    ft_wgts, ft_data = preproc_utils.mask_ft_wgts_data(ds_pre, ds_ft)

    assert ft_wgts['a'].sum() == 18
    assert ft_wgts['b'].sum() == 18
    assert ft_wgts['c'].sum() == 0
    assert ft_wgts['d'].sum() == 0

    assert ft_data['a'][2, 3] == ds_pre['a'][2, 3]
    assert ft_data['a'][0, 0] == ds_pre['a'][0, 0]
    assert ft_data['b'][1, 2] == ds_pre['b'][1, 2]
    assert ft_data['a'][1, 3] == ds_ft['a'][1, 3]


def test_read_exclude():
    exclude_file = 'test_data/exclude.yml'
    ex0 = preproc_utils.read_exclude_segs_file(exclude_file)
    assert ex0 == [{'seg_id_nats': [2007]}]
    exclude_file = 'test_data/exclude1.yml'
    ex1 = preproc_utils.read_exclude_segs_file(exclude_file)
    assert ex1 == [{'seg_id_nats': [2007], 'start_date': '2005-09-15'},
                   {'seg_id_nats': [2012], 'end_date': '2005-09-15'}]


def test_prep():
    """
    testing whether I can reconstruct the original data after processing
    :return:
    """
    x_data_file = 'test_data/x_data.npz'
    x_data = preproc_utils.prep_x(sntemp, ['seg_tave_air', 'seg_rain'],
                                  test_start_date='2005-09-15', n_test_yr=1,
                                  out_file=x_data_file)
    ft_vars = ['seg_tave_water', 'seg_outflow']
    pt_vars = ft_vars
    y_data = preproc_utils.prep_y(obs_temp, obs_flow, sntemp, x_data_file,
                                  pt_vars, ft_vars, "test_data/exclude.yml")

    sample_x = postproc_utils.prepped_array_to_df(x_data['x_trn'], x_data['dates_trn'],
                                                  x_data['ids_trn'],
                                                  x_data['x_cols']).set_index(
        ['seg_id_nat', 'date']).to_xarray()
    sample_y = postproc_utils.prepped_array_to_df(y_data['y_obs_trn'],
                                                  x_data['dates_trn'],
                                                  x_data['ids_trn'],
                                                  y_data['y_vars_ft']).set_index(
        ['seg_id_nat', 'date']).to_xarray()

    # read in unprocessed observations/inputs
    obs_y_flow = pd.read_csv(obs_flow, parse_dates=['date']).set_index(
        ['seg_id_nat', 'date']).to_xarray()
    obs_y_temp = pd.read_csv(obs_temp, parse_dates=['date']).set_index(
        ['seg_id_nat', 'date']).to_xarray()
    sntemp_x = xr.open_zarr(sntemp)

    # make sure they are the same
    # air temp
    sntemp_air_t = sntemp_x['seg_tave_air'].loc[:, sample_x.date].values
    processed_air_t = sample_x['seg_tave_air'].loc[:, sample_x.date].values
    processed_air_t = processed_air_t * x_data['x_std'][0] + x_data['x_mean'][0]
    assert np.allclose(processed_air_t, sntemp_air_t)

    # rain
    sntemp_r = sntemp_x['seg_rain'].loc[:, sample_x.date].values
    processed = sample_x['seg_rain'].loc[:, sample_x.date].values
    processed = processed * x_data['x_std'][1] + x_data['x_mean'][1]
    assert np.allclose(processed, sntemp_r)

    # temp
    obs = obs_y_temp['temp_c'].loc[:, sample_y.date].values
    processed = sample_y['seg_tave_water'].loc[:, sample_y.date].values
    processed = processed * y_data['y_obs_trn_std'][0] + \
                y_data['y_obs_trn_mean'][0]
    mask = ~(np.isnan(obs))
    assert np.allclose(processed[mask], obs[mask])

    # flow
    obs = obs_y_flow['discharge_cms'].loc[:, sample_y.date].values
    processed = sample_y['seg_outflow'].loc[:, sample_y.date].values
    processed = processed * y_data['y_obs_trn_std'][1] + \
                y_data['y_obs_trn_mean'][1]
    mask = ~(np.isnan(obs))
    assert np.allclose(processed[mask], obs[mask])

