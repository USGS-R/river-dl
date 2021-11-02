import pandas as pd
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
dfs = pd.read_feather(
    os.path.join(data_dir, "uncal_sntemp_input_output_subset.feather")
)
dfs["date"] = pd.to_datetime(dfs["date"])
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

start_date = "2003-09-15"
end_date = "2006-10-15"
segs = [2012, 2007]

dft = sel_date_segs(dft, segs, start_date, end_date)
dfq = sel_date_segs(dfq, segs, start_date, end_date)
dfs = sel_date_segs(dfs, segs, start_date, end_date)

dft = dft[["temp_c"]]
dfq = dfq[["discharge_cms"]]

df_combined = dft.join(dfq)

df_combined.to_xarray().to_zarr("test_data/obs_temp_flow", mode="w")
dfs.to_xarray().to_zarr("test_data/test_data", mode="w")
