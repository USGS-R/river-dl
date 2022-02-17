"""
This script creates data to test river-dl functions with. The script subsets
delaware river basin data that has 456 segments and ~40 years of data to just 
two segments and ~ 3years. Also there are far fewer variables that are used.
This subset is small enough that it can live with the repo so that the repo can
be self-contained (i.e., the code and the data used to test the code are both
part of the repo)
The following datasets are subset:
    - sntemp input and output (used for input data to test the models)
    - temperature and flow observations
    - segment attributes
    - distance matrix
"""
import pandas as pd
import numpy as np
import os


def select_data(df, col, selection):
    df = df.set_index(col)
    df = df.loc[selection]
    df.reset_index(inplace=True)
    return df


def sel_date_segs(df, segs, start_date, end_date):
    df = df[df["seg_id_nat"].notna()]
    df["seg_id_nat"] = df.seg_id_nat.astype(int)
    df = select_data(df, "date", slice(start_date, end_date))
    df = select_data(df, "seg_id_nat", segs)
    df = df.rename(columns={"seg_id_nat": "segs_test", "date": "times_test"})
    df.set_index(["segs_test", "times_test"], inplace=True)
    return df


# need to subset this data so it's just two years and two sites. I think
# such a dataset should be representative enough to run tests against
data_dir = "../../../drb-dl-model/data/in/"
start_date = "2003-09-15"
end_date = "2006-10-15"
segs = [2012, 2007, 2014, 2037]

# Subset sntemp data
dfs = pd.read_feather(
    os.path.join(data_dir, "uncal_sntemp_input_output_subset.feather")
)
test_variables = ['seg_rain', 'seg_tave_air', 'seg_slope', 'seg_humid',
                  'seg_tave_water', 'seg_outflow']
test_variables.extend(['date', 'seg_id_nat'])
dfs["date"] = pd.to_datetime(dfs["date"])
dfs = dfs[test_variables]
dfs = sel_date_segs(dfs, segs, start_date, end_date)
dfs.to_xarray().to_zarr("test_data/test_data", mode="w")

# Subset temperature and flow observations
dft = pd.read_csv(
    os.path.join(data_dir, "obs_temp_full.csv"),
    parse_dates=["date"],
    infer_datetime_format=True,
)
dfq = pd.read_csv(
    os.path.join(data_dir, "obs_flow_full.csv"),
    parse_dates=["date"],
    infer_datetime_format=True,
)
dft = sel_date_segs(dft, segs, start_date, end_date)
dfq = sel_date_segs(dfq, segs, start_date, end_date)

dft = dft[["temp_c"]]
dfq = dfq[["discharge_cms"]]

df_combined = dft.join(dfq)
df_combined.to_xarray().to_zarr("test_data/obs_temp_flow", mode="w")

# Subset segment attributes
df_seg_attr = pd.read_feather(os.path.join(data_dir, "seg_attr_drb.feather"))
df_seg_attr = df_seg_attr[df_seg_attr['seg_id_nat'].isin(segs)]
df_seg_attr.rename(columns = {'seg_id_nat': 'segs_test'})
df_seg_attr.reset_index().to_feather("test_data/test_seg_attr.feather")
df_seg_attr.reset_index().to_csv("test_data/test_seg_attr.csv", index=False)

# Subset distance matrix
dist_matrix_sample = np.random.randint(1, 100, size=(2, 2))
# distance between a point and itself should be zero
np.fill_diagonal(dist_matrix_sample, 0)
dist_matrix_data = {
    'updown': dist_matrix_sample,
    'downstream': dist_matrix_sample,
    'complete': dist_matrix_sample,
    'upstream': dist_matrix_sample,
    'rowcolnames': np.array(segs),
}
np.savez_compressed("test_data/test_dist_matrix.npz", **dist_matrix_data)


