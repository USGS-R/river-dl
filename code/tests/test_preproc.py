import numpy as np
import xarray as xr
import preproc_utils

# segments in test dataset
segments = ['2007', '2012']
# date range in test dataset
min_date = '2004-09-15'
max_date = '2006-10-15'


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
    assert ex1 == [{'seg_id_nats': [2007], 'start_date':'2005-09-15'},
                   {'seg_id_nats': [2012], 'end_date':'2005-09-15'}]



