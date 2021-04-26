import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt


def get_var_names(variable):
    """
    get the long variable names from 'flow' or 'temp'
    :param variable: [str] either 'flow' or 'temp'
    :return: [str] long variable names
    """
    if variable == "flow":
        obs_var = "discharge_cms"
        seg_var = "seg_outflow"
    elif variable == "temp":
        obs_var = "temp_c"
        seg_var = "seg_tave_water"
    else:
        raise ValueError('variable param must be "flow" or "temp"')
    return obs_var, seg_var


def load_if_not_df(pred_data):
    if isinstance(pred_data, str):
        return pd.read_feather(pred_data)
    else:
        return pred_data


def trim_obs(obs, preds):
    obs_trim = obs.reset_index()
    trim_preds = preds.reset_index()
    obs_trim = obs_trim[
        (obs_trim.date >= trim_preds.date.min())
        & (obs_trim.date <= trim_preds.date.max())
        & (obs_trim.seg_id_nat.isin(trim_preds.seg_id_nat.unique()))
    ]
    return obs_trim.set_index(["date", "seg_id_nat"])


def fmt_preds_obs(pred_data, obs_file, variable):
    """
    combine predictions and observations in one dataframe
    :param pred_data:[str] filepath to the predictions file
    :param obs_file:[str] filepath to the observations file
    :param variable: [str] either 'flow' or 'temp'
    """
    obs_var, seg_var = get_var_names(variable)
    pred_data = load_if_not_df(pred_data)
    # pred_data.loc[:, "seg_id_nat"] = pred_data["seg_id_nat"].astype(int)
    if {"date", "seg_id_nat"}.issubset(pred_data.columns):
        pred_data.set_index(["date", "seg_id_nat"], inplace=True)
    obs = xr.open_zarr(obs_file).to_dataframe()
    obs_cln = obs[[obs_var]]
    obs_cln.columns = ["obs"]
    preds = pred_data[[seg_var]]
    preds.columns = ["pred"]
    obs_cln_trim = trim_obs(obs_cln, preds)
    combined = preds.join(obs_cln_trim)
    return combined


def plot_obs(prepped_data, variable, outfile, partition="trn"):
    """
    plot training observations
    :param prepped_data: [str] path to npz file of prepped data
    :param variable: [str] which variable to plot, 'flow' or 'temp'
    :param outfile: [str] where to store the resulting file
    :return: None
    """
    data = np.load(prepped_data, allow_pickle=True)
    df = prepped_array_to_df(
        data[f"y_obs_{partition}"],
        data[f"dates_{partition}"],
        data[f"ids_{partition}"],
        data["y_vars"],
    )
    _, seg_var = get_var_names(variable)
    df_piv = df.pivot(index="date", columns="seg_id_nat", values=seg_var)
    df_piv.dropna(axis=1, how="all", inplace=True)
    try:
        df_piv.plot(subplots=True, figsize=(8, 12))
    except TypeError:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "NO DATA")
    plt.tight_layout()
    plt.savefig(outfile)


def plot_ts(pred_file, obs_file, variable, out_file):
    combined = fmt_preds_obs(pred_file, obs_file, variable)
    combined = combined.droplevel("seg_id_nat")
    ax = combined.plot(alpha=0.65)
    plt.tight_layout()
    plt.savefig(out_file)


def prepped_array_to_df(data_array, dates, ids, col_names):
    """
    convert prepped x or y data in numpy array to pandas df
    (reshape and make into pandas DFs)
    :param data_array:[numpy array] array of x or y data [nbatch, seq_len,
    n_out]
    :param dates:[numpy array] array of dates [nbatch, seq_len, n_out]
    :param ids: [numpy array] array of seg_ids [nbatch, seq_len, n_out]
    :return:[pd dataframe] df with cols
    ['date', 'seg_id_nat', 'temp_c', 'discharge_cms]
    """
    data_array = np.reshape(
        data_array,
        [data_array.shape[0] * data_array.shape[1], data_array.shape[2]],
    )

    dates = np.reshape(dates, [dates.shape[0] * dates.shape[1], dates.shape[2]])
    ids = np.reshape(ids, [ids.shape[0] * ids.shape[1], ids.shape[2]])
    df_preds = pd.DataFrame(data_array, columns=col_names)
    df_dates = pd.DataFrame(dates, columns=["date"])
    df_ids = pd.DataFrame(ids, columns=["seg_id_nat"])
    df = pd.concat([df_dates, df_ids, df_preds], axis=1)
    return df
