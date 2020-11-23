import pandas as pd
import numpy as np
import xarray as xr
import preproc_utils
import postproc_utils

# segments in test dataset
segments = ["2007", "2012"]
# date range in test dataset
min_date = "2004-09-15"
max_date = "2006-10-15"

obs_flow = "test_data/obs_flow.csv"
obs_temp = "test_data/obs_temp.csv"
sntemp = "test_data/test_data"


def test_weight_creation():
    n_segs = 4
    n_dates = 5
    num_pretrain_vars = 4
    dims = (n_segs, n_dates)
    pre_das = [
        xr.DataArray(np.random.rand(*dims), dims=["seg_id_nat", "date"])
        for i in range(num_pretrain_vars)
    ]
    ds_pre = xr.Dataset(
        {"a": pre_das[0], "b": pre_das[1], "c": pre_das[2], "d": pre_das[3]}
    )

    num_finetune_vars = 4
    ft_das = [
        xr.DataArray(np.random.rand(*dims), dims=["seg_id_nat", "date"])
        for i in range(num_finetune_vars)
    ]

    ds_ft = xr.Dataset({"a": ft_das[0], "b": ft_das[1]})

    ds_ft["a"][2, 3] = np.nan
    ds_ft["a"][0, 0] = np.nan
    ds_ft["b"][1, 1:3] = np.nan

    ft_wgts, ft_data = preproc_utils.mask_ft_wgts_data(ds_pre, ds_ft)

    assert ft_wgts["a"].sum() == 18
    assert ft_wgts["b"].sum() == 18
    assert ft_wgts["c"].sum() == 0
    assert ft_wgts["d"].sum() == 0

    assert ft_data["a"][2, 3] == ds_pre["a"][2, 3]
    assert ft_data["a"][0, 0] == ds_pre["a"][0, 0]
    assert ft_data["b"][1, 2] == ds_pre["b"][1, 2]
    assert ft_data["a"][1, 3] == ds_ft["a"][1, 3]


def test_read_exclude():
    exclude_file = "test_data/exclude_2007.yml"
    ex0 = preproc_utils.read_exclude_segs_file(exclude_file)
    assert ex0 == [{"seg_id_nats_ex": [2007]}]
    exclude_file = "test_data/exclude1.yml"
    ex1 = preproc_utils.read_exclude_segs_file(exclude_file)
    assert ex1 == [
        {"seg_id_nats_ex": [2007], "start_date": "2005-03-15"},
        {"seg_id_nats_ex": [2012], "end_date": "2005-03-15"},
    ]


class PreppedData:
    def __init__(
        self,
        ft_vars=["seg_tave_water", "seg_outflow"],
        pt_vars=["seg_tave_water", "seg_outflow"],
        x_cols=["seg_tave_air", "seg_rain"],
        test_st="2005-09-15",
        n_test_yr=1,
        exclude_file=None,
    ):
        self.x_data_file = "test_data/x_data.npz"
        self.x_data = preproc_utils.prep_x(
            sntemp,
            x_cols,
            test_start_date=test_st,
            n_test_yr=n_test_yr,
            out_file=self.x_data_file,
        )
        self.y_data = preproc_utils.prep_y(
            obs_temp,
            obs_flow,
            sntemp,
            self.x_data_file,
            pt_vars,
            ft_vars,
            exclude_file=exclude_file,
        )

        self.sample_x = (
            postproc_utils.prepped_array_to_df(
                self.x_data["x_trn"],
                self.x_data["dates_trn"],
                self.x_data["ids_trn"],
                self.x_data["x_cols"],
            )
            .set_index(["seg_id_nat", "date"])
            .to_xarray()
        )

        self.sample_y_obs_trn = (
            postproc_utils.prepped_array_to_df(
                self.y_data["y_obs_trn"],
                self.x_data["dates_trn"],
                self.x_data["ids_trn"],
                self.y_data["y_vars"],
            )
            .set_index(["seg_id_nat", "date"])
            .to_xarray()
        )

        self.sample_y_pre_trn = (
            postproc_utils.prepped_array_to_df(
                self.y_data["y_pre_trn"],
                self.x_data["dates_trn"],
                self.x_data["ids_trn"],
                self.y_data["y_vars"],
            )
            .set_index(["seg_id_nat", "date"])
            .to_xarray()
        )

        self.sample_y_obs_wgts = (
            postproc_utils.prepped_array_to_df(
                self.y_data["y_obs_wgts"],
                self.x_data["dates_trn"],
                self.x_data["ids_trn"],
                self.y_data["y_vars"],
            )
            .set_index(["seg_id_nat", "date"])
            .to_xarray()
        )

        self.sample_y_pre_wgts = (
            postproc_utils.prepped_array_to_df(
                self.y_data["y_pre_wgts"],
                self.x_data["dates_trn"],
                self.x_data["ids_trn"],
                self.y_data["y_vars"],
            )
            .set_index(["seg_id_nat", "date"])
            .to_xarray()
        )

        # read in unprocessed observations/inputs
        self.obs_y_flow = (
            pd.read_csv(obs_flow, parse_dates=["date"])
            .set_index(["seg_id_nat", "date"])
            .to_xarray()
        )
        self.obs_y_temp = (
            pd.read_csv(obs_temp, parse_dates=["date"])
            .set_index(["seg_id_nat", "date"])
            .to_xarray()
        )
        self.sntemp_x = xr.open_zarr(sntemp)


def test_prep():
    """
    testing whether I can reconstruct the original data after processing
    :return:
    """
    prepped = PreppedData()

    # make sure they are the same
    # air temp
    sntemp_air_t = (
        prepped.sntemp_x["seg_tave_air"].loc[:, prepped.sample_x.date].values
    )
    processed_air_t = (
        prepped.sample_x["seg_tave_air"].loc[:, prepped.sample_x.date].values
    )
    processed_air_t = (
        processed_air_t * prepped.x_data["x_std"][0]
        + prepped.x_data["x_mean"][0]
    )
    assert np.allclose(processed_air_t, sntemp_air_t)

    # rain
    sntemp_r = prepped.sntemp_x["seg_rain"].loc[:, prepped.sample_x.date].values
    processed = (
        prepped.sample_x["seg_rain"].loc[:, prepped.sample_x.date].values
    )
    processed = (
        processed * prepped.x_data["x_std"][1] + prepped.x_data["x_mean"][1]
    )
    assert np.allclose(processed, sntemp_r)

    # temp
    obs = (
        prepped.obs_y_temp["temp_c"]
        .loc[:, prepped.sample_y_obs_trn.date]
        .values
    )
    processed = (
        prepped.sample_y_obs_trn["seg_tave_water"]
        .loc[:, prepped.sample_y_obs_trn.date]
        .values
    )
    processed = (
        processed * prepped.y_data["y_obs_trn_std"][0]
        + prepped.y_data["y_obs_trn_mean"][0]
    )
    mask = ~(np.isnan(obs))
    assert np.allclose(processed[mask], obs[mask])

    # flow
    obs = (
        prepped.obs_y_flow["discharge_cms"]
        .loc[:, prepped.sample_y_obs_trn.date]
        .values
    )
    processed = (
        prepped.sample_y_obs_trn["seg_outflow"]
        .loc[:, prepped.sample_y_obs_trn.date]
        .values
    )
    processed = (
        processed * prepped.y_data["y_obs_trn_std"][1]
        + prepped.y_data["y_obs_trn_mean"][1]
    )
    mask = ~(np.isnan(obs))
    assert np.allclose(processed[mask], obs[mask])


def test_no_exclude():
    # test no exclude
    prepped = PreppedData()
    wgts = prepped.sample_y_obs_wgts.to_array()
    seg_2007_sum = wgts.sel(seg_id_nat=2007).sum().values
    seg_2012_sum = wgts.sel(seg_id_nat=2012).sum().values
    assert seg_2007_sum > 0
    assert seg_2012_sum > 0


def test_exclude_in():
    # try just *including* 2007 (exclude_in.yml)
    prepped = PreppedData(exclude_file="test_data/exclude_in.yml")
    wgts = prepped.sample_y_obs_wgts.to_array()
    seg_2007_sum = wgts.sel(seg_id_nat=2007).sum().values
    seg_2012_sum = wgts.sel(seg_id_nat=2012).sum().values
    assert seg_2007_sum > 0
    assert seg_2012_sum == 0


def test_exclude_2007():
    # try just excluding 2007 (exclude_2007.yml)
    prepped = PreppedData(exclude_file="test_data/exclude_2007.yml")
    wgts = prepped.sample_y_obs_wgts.to_array()
    seg_2007_sum = wgts.sel(seg_id_nat=2007).sum().values
    seg_2012_sum = wgts.sel(seg_id_nat=2012).sum().values
    assert seg_2007_sum == 0
    assert seg_2012_sum > 0


def test_exclude_periods():
    # excluding 2007 for second half, 2012 first half(exclude1.yml)
    prepped = PreppedData(exclude_file="test_data/exclude1.yml")
    wgts = prepped.sample_y_obs_wgts.to_array()
    start_date = wgts.date.min()
    mid_date = "2005-03-15"
    end_date = wgts.date.max()
    seg_2012_sum_1st = (
        wgts.sel(seg_id_nat=2012, date=slice(start_date, mid_date)).sum().values
    )
    seg_2012_sum_2nd = (
        wgts.sel(seg_id_nat=2012, date=slice(mid_date, end_date)).sum().values
    )
    seg_2007_sum_1st = (
        wgts.sel(seg_id_nat=2007, date=slice(start_date, mid_date)).sum().values
    )
    seg_2007_sum_2nd = (
        wgts.sel(seg_id_nat=2007, date=slice(mid_date, end_date)).sum().values
    )
    assert seg_2007_sum_1st > 0
    assert seg_2007_sum_2nd == 0
    assert seg_2012_sum_1st == 0
    assert seg_2012_sum_2nd > 0


def test_var_weights():
    # try with both variables
    prepped = PreppedData()
    wgts = prepped.sample_y_obs_wgts.to_array()
    flow_sum = wgts.sel(variable="seg_outflow").sum().values
    temp_sum = wgts.sel(variable="seg_tave_water").sum().values
    assert flow_sum > 0
    assert temp_sum > 0


def test_var_weights_temp_ft():
    # try with just temp in finetuning
    prepped = PreppedData(ft_vars=["seg_tave_water"])
    wgts_obs = prepped.sample_y_obs_wgts.to_array()
    wgts_pre = prepped.sample_y_pre_wgts.to_array()
    flow_sum_pre = wgts_pre.sel(variable="seg_outflow").sum().values
    temp_sum_pre = wgts_pre.sel(variable="seg_tave_water").sum().values
    flow_sum = wgts_obs.sel(variable="seg_outflow").sum().values
    temp_sum = wgts_obs.sel(variable="seg_tave_water").sum().values
    assert flow_sum_pre > 0
    assert temp_sum_pre > 0
    assert flow_sum == 0
    assert temp_sum > 0


def test_var_weights_flow_ft():
    # try with just flow in finetuning
    prepped = PreppedData(ft_vars=["seg_outflow"])
    wgts_obs = prepped.sample_y_obs_wgts.to_array()
    wgts_pre = prepped.sample_y_pre_wgts.to_array()
    flow_sum_pre = wgts_pre.sel(variable="seg_outflow").sum().values
    temp_sum_pre = wgts_pre.sel(variable="seg_tave_water").sum().values
    flow_sum = wgts_obs.sel(variable="seg_outflow").sum().values
    temp_sum = wgts_obs.sel(variable="seg_tave_water").sum().values
    assert flow_sum_pre > 0
    assert temp_sum_pre > 0
    assert flow_sum > 0
    assert temp_sum == 0


def test_var_weights_temp_ft_pt():
    # try with just temp in finetuning and pretraining
    prepped = PreppedData(
        ft_vars=["seg_tave_water"], pt_vars=["seg_tave_water"]
    )
    wgts_obs = prepped.sample_y_obs_wgts.to_array()
    wgts_pre = prepped.sample_y_pre_wgts.to_array()
    temp_sum_pre = wgts_pre.sel(variable="seg_tave_water").sum().values
    temp_sum = wgts_obs.sel(variable="seg_tave_water").sum().values
    assert len(wgts_obs.variable) == 1
    assert len(wgts_pre.variable) == 1
    assert temp_sum_pre > 0
    assert temp_sum > 0


def test_var_weights_flow_ft_pt():
    # try with just flow in finetuning and pretraining
    prepped = PreppedData(ft_vars=["seg_outflow"], pt_vars=["seg_outflow"])
    wgts_obs = prepped.sample_y_obs_wgts.to_array()
    wgts_pre = prepped.sample_y_pre_wgts.to_array()
    flow_sum_pre = wgts_pre.sel(variable="seg_outflow").sum().values
    flow_sum = wgts_obs.sel(variable="seg_outflow").sum().values
    assert len(wgts_obs.variable) == 1
    assert len(wgts_pre.variable) == 1
    assert flow_sum_pre > 0
    assert flow_sum > 0
