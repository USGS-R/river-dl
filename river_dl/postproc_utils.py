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

def trim_obs(obs, preds, spatial_idx_name="seg_id_nat", time_idx_name="date"):
    obs_trim = obs.reset_index()
    trim_preds = preds.reset_index()
    obs_trim = obs_trim[
        (obs_trim[time_idx_name] >= trim_preds[time_idx_name].min())
        & (obs_trim[time_idx_name] <= trim_preds[time_idx_name].max())
        & (obs_trim[spatial_idx_name].isin(trim_preds[spatial_idx_name].unique()))
    ]
    return obs_trim.set_index([time_idx_name, spatial_idx_name])


def fmt_preds_obs(pred_data,
                  obs_file,
                  spatial_idx_name="seg_id_nat",
                  time_idx_name="date"):
    """
    combine predictions and observations in one dataframe
    :param pred_data:[str] filepath to the predictions file
    :param obs_file:[str] filepath to the observations file
    :param spatial_idx_name: [str] name of column that is used for spatial
        index (e.g., 'seg_id_nat')
    :param time_idx_name: [str] name of column that is used for temporal index
        (usually 'time')
    """
    pred_data = load_if_not_df(pred_data)

    if {time_idx_name, spatial_idx_name}.issubset(pred_data.columns):
        pred_data.set_index([time_idx_name, spatial_idx_name], inplace=True)
    obs = xr.open_zarr(obs_file,consolidated=False).to_dataframe()
    variables_data = {}

    for var_name in pred_data.columns:
        obs_var = obs.copy()
        obs_var = obs_var[[var_name]]
        obs_var.columns = ["obs"]
        preds_var = pred_data[[var_name]]
        preds_var.columns = ["pred"]
        # trimming obs to preds speeds up following join greatly
        obs_var = trim_obs(obs_var, preds_var, spatial_idx_name, time_idx_name)
        combined = preds_var.join(obs_var)
        variables_data[var_name] = combined
    return variables_data


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

def plot_ts_obs_preds(pred_file, obs_file, index_start = 0, index_end=3, outfile=None):
    """
    Plots the observations vs predictions for a subset of reaches with the most observations
    @param pred_file: [str] path to feather file with predictions
    @param obs_file: [str] path to observation file
    @param index_start: [int] start reach index
    @param index_end: [int] end reach index
    """
    combined = fmt_preds_obs(pred_file, obs_file)['temp_c']
    counts = combined.groupby(combined.index.get_level_values(1)).count().sort_values("obs",ascending=False)
    combined = combined[combined.index.get_level_values(1).isin(counts.index[index_start:index_end])].reset_index(level='seg_id_nat')
    segs = np.unique(combined.seg_id_nat)
    num_plots = len(segs)
    fig, axes = plt.subplots(nrows=num_plots)
    for seg, ax in zip(segs, axes.flat):
        df = combined.loc[combined.seg_id_nat == seg]#.melt(id_vars=['seg_id_nat','date'])
        ax.plot("pred", data = df, label="pred",alpha=0.5)
        ax.plot("obs", data=df, label = 'obs',alpha=0.5)
        ax.legend()
        ax.set_title(seg)
    plt.tight_layout()
    if out_file:
        plt.savefig(outfile)
    else:
        plt.show()


def prepped_array_to_df(data_array, dates, ids, col_names, spatial_idx_name='seg_id_nat', time_idx_name='date'):
    """
    convert prepped x or y_dataset data in numpy array to pandas df
    (reshape and make into pandas DFs)
    :param data_array:[numpy array] array of x or y_dataset data [nbatch, seq_len,
    n_out]
    :param dates:[numpy array] array of dates [nbatch, seq_len, n_out]
    :param ids: [numpy array] array of seg_ids [nbatch, seq_len, n_out]
    :return:[pd dataframe] df with cols
    ['date', 'seg_id_nat', 'temp_c', 'discharge_cms]
    """
    num_out_vars = data_array.shape[2]
    flat_data = [data_array[:, :, i].flatten() for i in range(num_out_vars)]
    flat_data_combined = np.stack(flat_data, axis=1)

    dates = dates.flatten()
    ids = ids.flatten()
    df_preds = pd.DataFrame(flat_data_combined , columns=col_names)
    df_preds = pd.DataFrame(data_array, columns=col_names)
    df_dates = pd.DataFrame(dates, columns=[time_idx_name])
    df_ids = pd.DataFrame(ids, columns=[spatial_idx_name])
    df = pd.concat([df_dates, df_ids, df_preds], axis=1)
    return df
