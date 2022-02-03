"""
This script takes the original test data and changes the index names to 
`seg_id_nat` for the spatial index and `date` for the time index
"""
import xarray as xr
import pandas as pd

def rename_dims_and_clear_encoding(ds):
    ds = ds.rename({"segs_test": "seg_id_nat", "times_test": "date"})
    to_store = ds.copy()
    for var in to_store.variables:
        to_store[var].encoding.clear()
    return to_store

data_dir = "../../../drb-dl-model/data/in/"

obs_ds = rename_dims_and_clear_encoding(xr.open_zarr("test_data/obs_temp_flow"))
obs_ds.to_zarr("test_data/obs_temp_flow_seg_id_nat", mode="w")

sntemp_ds = rename_dims_and_clear_encoding(xr.open_zarr("test_data/test_data"))
sntemp_ds.to_zarr("test_data/test_data_seg_id_nat", mode="w")

seg_attr = pd.read_feather("test_data/test_seg_attr.feather")
seg_attr.rename(columns = {"segs_test": "seg_id_nat"})
seg_attr.to_feather("test_data/test_seg_attr_seg_id_nat.feather")
seg_attr.to_csv("test_data/test_seg_attr_seg_id_nat.csv", index=False)
