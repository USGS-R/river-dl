import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from copy import deepcopy


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


def plot_obs(prepped_data, variable, outfile, spatial_idx_name="seg_id_nat", time_idx_name="date",  partition="trn"):
    """
    plot training observations
    :param prepped_data: [str] path to npz file of prepped data
    :param variable: [str] which variable to plot, 'flow' or 'temp'
    :param outfile: [str] where to store the resulting file
    :param spatial_idx_name: [str] name of column that is used for spatial
        index (e.g., 'seg_id_nat')
    :param time_idx_name: [str] name of column that is used for temporal index
        (usually 'time')
    :return: None
    """
    data = np.load(prepped_data, allow_pickle=True)
    df = prepped_array_to_df(
        data[f"y_obs_{partition}"],
        data[f"times_{partition}"],
        data[f"ids_{partition}"],
        data["y_obs_vars"],
        spatial_idx_name = spatial_idx_name,
        time_idx_name = time_idx_name
    )
    df_piv = df.pivot(index=time_idx_name, columns=spatial_idx_name)
    df_piv.dropna(axis=1, how="all", inplace=True)
    try:
        df_piv.plot(subplots=True, figsize=(8, 12))
    except TypeError:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "NO DATA")
    plt.tight_layout()
    plt.savefig(outfile)


def plot_ts(pred_file, obs_file, spatial_idx_name, variable, out_file):
    combined = fmt_preds_obs(pred_file, obs_file, variable)
    combined = combined.droplevel(spatial_idx_name)
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
    if outfile:
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
    :param spatial_idx_name: [str] name of column that is used for spatial
        index (e.g., 'seg_id_nat')
    :param time_idx_name: [str] name of column that is used for temporal index
        (usually 'time')
    :return:[pd dataframe] df with cols
    """
    num_out_vars = data_array.shape[-1]
    flat_data = [data_array[..., i].flatten() for i in range(num_out_vars)]
    flat_data_combined = np.stack(flat_data, axis=1)

    dates = dates.flatten()
    ids = ids.flatten()
    df_preds = pd.DataFrame(flat_data_combined , columns=col_names)
    df_dates = pd.DataFrame(dates, columns=[time_idx_name])
    df_ids = pd.DataFrame(ids, columns=[spatial_idx_name])
    df = pd.concat([df_dates, df_ids, df_preds], axis=1)
    return df


def combine_preds(fileList,weights=None,pred_vars=None, outFile = "composite.feather", spatial_idx_name="seg_id_nat", time_idx_name="date"):
    """
    combine multiple model outputs into 1 composite file
    :param fileList: [str] list of model prediction files
    :param weights: [list] list model weights corresponding to the list of model prediction files. This could be a list of 
dataframes with spatial_idx_name and / or time_idx_name columns and a modelWeight column or it could be a single value for 
each model (range of 0 - 1). If None, the models are weighted equally
    :param pred_vars: [str] list of predicted variables
    :param outFile: [str] feather file where the composite predictions should be written
    """
    idx_cols = [spatial_idx_name, time_idx_name]

    for i in range(len(fileList)):
        thisFile = fileList[i]
        tempDF = pd.read_feather(thisFile)
        if not pred_vars:
            pred_vars = [x for x in tempDF.columns if x not in idx_cols]
        if weights:
            thisWeight = weights[i]
            if type(thisWeight)==pd.DataFrame:
                tempDF=tempDF.merge(thisWeight)
            else:
                tempDF['modelWeight']=float(thisWeight)
        else:
            tempDF['modelWeight']=1.0/len(fileList)
        
        #make the composite dataframe
        if thisFile==fileList[0]:
            compositeDF = tempDF.iloc[:,:-1]
            for thisVar in pred_vars:
                compositeDF[thisVar]=compositeDF[thisVar].values*tempDF.modelWeight.values
            #save the weights for this model to ensure they are 1 across all models    
            weightCheckDF = deepcopy(tempDF[[spatial_idx_name, time_idx_name,'modelWeight']])
        else:
            for thisVar in pred_vars:
                compositeDF[thisVar]=compositeDF[thisVar].values+tempDF[thisVar]*tempDF.modelWeight.values
            weightCheckDF['modelWeight']=weightCheckDF['modelWeight']+tempDF['modelWeight']
            
            
    #check that all cummulative weights are less than 1.01
    np.testing.assert_allclose(weightCheckDF.modelWeight, 1, rtol=1e-02, atol=1e-02, equal_nan=True, err_msg='Model weights did not sum to 1', verbose=True)

    #drop predicted variables that weren't merged
    colsToDrop = [x for x in compositeDF.columns if x not in pred_vars and x not in idx_cols]
    if len(colsToDrop)>0:
        compositeDF.drop(columns=colsToDrop,inplace=True)    
    #save the output
    compositeDF.to_feather(outFile)
