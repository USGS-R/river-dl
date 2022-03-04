import numpy as np
import pytest
import pandas as pd
import xarray as xr
from river_dl import evaluate as ev


def get_partition_nse(df, partition):
    return df.query(f"partition == '{partition}'")['nse'].values[0]

def test_pred_data():
    obs_file = "test_data/obs_temp_flow"
    obs_ds = xr.open_zarr(obs_file)
    obs_df = obs_ds.to_dataframe()
    metrics = ev.combined_metrics(obs_file = obs_file,
                                  pred_data = {"trn": obs_df,
                                               "val": obs_df,
                                               "tst": obs_df},
                                  spatial_idx_name="segs_test",
                                  time_idx_name="times_test")
    assert get_partition_nse(metrics, 'trn') == 1
    assert get_partition_nse(metrics, 'val') == 1
    assert get_partition_nse(metrics, 'tst') == 1


def test_pred_data_no_val():
    obs_file = "test_data/obs_temp_flow"
    obs_ds = xr.open_zarr(obs_file)
    obs_df = obs_ds.to_dataframe()
    metrics = ev.combined_metrics(obs_file = obs_file,
                                  pred_data = {"trn": obs_df,
                                               "tst": obs_df},
                                  spatial_idx_name="segs_test",
                                  time_idx_name="times_test")
    assert get_partition_nse(metrics, 'trn') == 1
    assert get_partition_nse(metrics, 'tst') == 1

    with pytest.raises(IndexError):
        get_partition_nse(metrics, 'val') == 1


def test_pred_data_prd_val():
    obs_file = "test_data/obs_temp_flow"
    obs_ds = xr.open_zarr(obs_file)
    obs_df = obs_ds.to_dataframe()
    metrics = ev.combined_metrics(obs_file = obs_file,
                                  pred_data = {"trn": obs_df,
                                               "tst": obs_df},
                                  pred_val = obs_df,
                                  spatial_idx_name="segs_test",
                                  time_idx_name="times_test")
    assert get_partition_nse(metrics, 'trn') == 1
    assert get_partition_nse(metrics, 'tst') == 1

    with pytest.raises(IndexError):
        get_partition_nse(metrics, 'val') == 1


def test_id_dict():
    obs_file = "test_data/obs_temp_flow"
    obs_ds = xr.open_zarr(obs_file)
    obs_df = obs_ds.to_dataframe()
    metrics = ev.combined_metrics(obs_file = obs_file,
                                  pred_data = {"trn": obs_df,
                                               "tst": obs_df},
                                  spatial_idx_name="segs_test",
                                  time_idx_name="times_test",
                                  id_dict={"run_id":4,
                                            "exp_id": "giz"}
                                  )
    assert "run_id" in metrics.columns
    assert "exp_id" in metrics.columns

    assert np.sum(metrics['run_id'].values != 4) == 0
    assert np.sum(metrics['exp_id'].values != "giz") == 0
