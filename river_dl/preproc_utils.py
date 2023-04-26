import pandas as pd
import numpy as np
import yaml, time, os
import xarray as xr
import datetime
import subprocess
import shutil
import sys


def saveRunLog(config,code_dir,outFile):
    """
    function to add the run specs to a csv log file
    :param config: [dict] the current config dictionary
    :param code_dir: [str] path to river-dl directory
    :param outFile: [str] the filename for the output
    """

    #check if the log already exists
    newLog = not os.path.exists(outFile)

    #add a default run description if needed
    if not "runDescription" in config.keys():
        config['runDescription']=""

    #replace commas with semi-colons for csv readability
    config['runDescription']=config['runDescription'].replace(",",";")

    with open(outFile,"a+") as f:
        if newLog:
            f.write("'Date','Directory','Description'\n")
        f.write("%s,'%s','%s'\n"%(datetime.date.today().strftime("%m/%d/%y"),os.path.join(os.path.relpath(os.getcwd(),code_dir),config['out_dir']),config['runDescription']))

def asRunConfig(config, code_dir, outFile):
    """
    function to save the as-run config settings to a text file
    :param config: [dict] the current config dictionary
    :param code_dir: [str] path to river-dl directory
    :param outFile: [str] the filename for the output
    """
    #store some run parameters
    config['runDate']=datetime.date.today().strftime("%m/%d/%y")
    with open(os.path.join(code_dir,".git/HEAD"),'r') as head:
        ref = head.readline().split(' ')[-1].strip()
        branch = ref.split("/")[-1]
    with open(os.path.join(code_dir,'.git/', ref),'r') as git_hash:
        commit = git_hash.readline().strip()
    status = str(subprocess.Popen(['git status'], cwd = None if code_dir=="" else code_dir,shell=True,stdout=subprocess.PIPE).communicate()[0]).split("\\n")
    modifiedFiles = [x.split()[1].strip() for x in status if "modified" in x]
    newFiles = [x.split()[1].strip() for x in status if "new file" in x]
    config['gitStatus']= 'unknown' if len(status)==1 else 'dirty' if len(modifiedFiles)>0 or len(newFiles)>0 else 'clean'
    #if the repo is dirty, make a zipped archive of the code directory
    if config['gitStatus']=='dirty':
        shutil.make_archive(os.path.join(os.path.dirname(outFile),"river_dl"),"zip",os.path.join(code_dir,"river_dl"))
    config['gitModified']=modifiedFiles
    config['gitNew']=newFiles
    config['gitBranch']=branch
    config['gitCommit'] = commit
    #and the file info for the input files
    config['input_file_info']={config[x]:{'file_size':os.stat(config[x]).st_size,'file_date':time.strftime("%m/%d/%Y %I:%M:%S %p",time.localtime(os.stat(config[x]).st_ctime))} for x in config.keys() if "file" in x and x!="input_file_info"}
    with open(outFile,'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    #add the log entry
    saveRunLog(config,code_dir,os.path.join(code_dir,"runLog.csv"))

def scale(dataset, std=None, mean=None):
    """
    scale the data so it has a standard deviation of 1 and a mean of zero
    :param dataset: [xr dataset] input or output data
    :param std: [xr dataset] standard deviation if scaling test data with dims
    :param mean: [xr dataset] mean if scaling test data with dims
    :return: scaled data with original dims
    """
    if not isinstance(std, xr.Dataset) or not isinstance(mean, xr.Dataset):
        std = dataset.std(skipna=True)
        mean = dataset.mean(skipna=True)
    # adding small number in case there is a std of zero
    scaled = (dataset - mean) / (std + 1e-10)
    check_if_finite(std)
    check_if_finite(mean)
    return scaled, std, mean


def sel_partition_data(dataset, time_idx_name, start_dates, end_dates):
    """
    select the data from a date range or a set of date ranges
    :param dataset: [xr dataset] input or output data with date dimension
    :param time_idx_name: [str] name of column that is used for temporal index
        (usually 'time')
    :param start_dates: [str or list] fmt: "YYYY-MM-DD"; date(s) to start period
    (can have multiple discontinuos periods)
    :param end_dates: [str or list] fmt: "YYYY-MM-DD"; date(s) to end period
    (can have multiple discontinuos periods)
    :return: dataset of just those dates
    """
   # if it just one date range
    if isinstance(start_dates, str):
        if isinstance(end_dates, str):
            return dataset.sel({time_idx_name: slice(start_dates, end_dates)})
        else:
            raise ValueError("start_dates is str but not end_date")
    # if it's a list of date ranges
    elif isinstance(start_dates, list) or isinstance(start_dates, tuple):
        if len(start_dates) == len(end_dates):
            data_list = []
            for i in range(len(start_dates)):
                date_slice = slice(start_dates[i], end_dates[i])
                data_list.append(dataset.sel({time_idx_name: date_slice}))
            return xr.concat(data_list, dim=time_idx_name)
        else:
            raise ValueError("start_dates and end_dates must have same length")
    else:
        raise ValueError("start_dates must be either str, list, or tuple")


def separate_trn_tst(
    dataset,
    time_idx_name,
    train_start_date,
    train_end_date,
    val_start_date=None,
    val_end_date=None,
    test_start_date=None,
    test_end_date=None,
):
    """
    separate the train data from the test data according to the start and end
    dates. This assumes your training data is in one continuous block. Be aware,
    if your train/test/val partitions are discontinuous (composed of multiple
    periods), depending on your sequence length and how the data line up, you
    could end up with sequences starting in one period and ending in another.
    The breaking up of sequences would happen in the `convert_batch_reshape`
    function
    :param dataset: [xr dataset] input or output data with dims
    :param time_idx_name: [str] name of column that is used for temporal index
        (usually 'time')
    :param train_start_date: [str or list] fmt: "YYYY-MM-DD"; date(s) to start
    train period (can have multiple discontinuous periods)
    :param train_end_date: [str or list] fmt: "YYYY-MM-DD"; date(s) to end train
     period (can have multiple discontinuous periods)
    :param val_start_date: [str or list] fmt: "YYYY-MM-DD"; date(s) to start
     validation period (can have multiple discontinuous periods)
    :param val_end_date: [str or list] fmt: "YYYY-MM-DD"; date(s) to end
    validation period (can have multiple discontinuous periods)
    :param test_start_date: [str or list] fmt: "YYYY-MM-DD"; date(s) to start
    test period (can have multiple discontinuous periods)
    :param test_end_date: [str or list] fmt: "YYYY-MM-DD"; date(s) to end test
    period (can have multiple discontinuous periods)
    :return: [tuple] separated data
    """
    train = sel_partition_data(
        dataset, time_idx_name, train_start_date, train_end_date
    )


    if val_start_date and val_end_date:
        val = sel_partition_data(
            dataset, time_idx_name, val_start_date, val_end_date
        )
        
    elif val_start_date and not val_end_date:
        raise ValueError("With a val_start_date a val_end_date must be given")
    elif val_end_date and not val_start_date:
        raise ValueError("With a val_end_date a val_start_date must be given")
    else:
        val = None

    if test_start_date and test_end_date:
        test = sel_partition_data(
            dataset, time_idx_name, test_start_date, test_end_date
        )
    elif test_start_date and not test_end_date:
        raise ValueError("With a test_start_date a test_end_date must be given")
    elif test_end_date and not test_start_date:
        raise ValueError("With a test_end_date a test_start_date must be given")
    else:
        test = None

    return train, val, test


def split_into_batches(data_array, seq_len=365, offset=1.0,
                       fill_batch=True, fill_nan=False, fill_time=False):
    """
    split training data into batches with size of seq_len
    :param data_array: [numpy array] array of training data with dims [nseg,
    ndates, nfeat]
    :param seq_len: [int] length of sequences (e.g., 365)
    :param offset: [float] How to offset the batches. Values < 1 are taken as fractions, (e.g., 0.5 means that
    the first batch will be 0-365 and the second will be 182-547), values > 1 are used as a constant number of
    observations to offset by.
    :param fill_batch: [bool] when True, batches are filled to match the seq_len.
    This ensures that data are not dropped when the data_array length is not
    a multiple of the seq_len. Data are added to the end of the sequence.
    When False, data may be dropped.
    :param fill_nan: [bool] When True, filled in data are np.nan (e.g., because 
    data_array is observation data that should not contribute to the loss).
    When False, filled in data are replicates of the previous timesteps.
    :param fill_time: [bool] When True, filled in data are time indices that
    follow in sequence from the previous timesteps. When False, filled in data 
    are replicates of the previous timesteps.
    :return: [numpy array] batched data with dims [nbatches, nseg, seq_len
    (batch_size), nfeat]
    """
    if offset>1:
        period = int(offset)
    else:
        period = int(offset*seq_len)
    
    nsteps = data_array.shape[1]
    num_batches = nsteps//period
    if fill_batch:
        #Check if nsteps is an exact multiple of the period.
        exact_batches = nsteps/period
        if (num_batches != exact_batches):
            #append timesteps to the data_array
            #exact_batches will always be greater than num_batches.
            #Determine how many timesteps to replicate to get a full batch
            num_rep_steps = int(period - np.floor((exact_batches - num_batches) * seq_len))
            if fill_nan:
                #fill in with nan values 
                nan_array = np.empty((data_array.shape[0], 
                                      num_rep_steps, 
                                      data_array.shape[2]))
                nan_array.fill(np.nan)
                data_array = np.concatenate((data_array, 
                                             nan_array), 
                                            axis = 1)
            else:
                #fill in by replicating the previous timesteps in the data_array
                if fill_time:
                    #data are an np.datetime64 object. These must be unique, so
                    #cannot be replicated. Add timesteps sequentially
                    fill_dates_array = data_array[:,(nsteps-num_rep_steps):nsteps,:].copy()
                    #add num_rep_steps to each index. 
                    # Sending the smallest possible sample of data_array to the function
                    time_unit = get_time_unit(data_array[0:2,0:2,0:2])
                    fill_dates_array = fill_dates_array[:,:,:] + np.timedelta64(num_rep_steps, time_unit)
                    
                    data_array = np.concatenate((data_array, 
                                                 fill_dates_array),
                                                axis = 1)
                else:
                    data_array = np.concatenate((data_array, 
                                                 data_array[:,(nsteps-num_rep_steps):nsteps,:]),
                                                axis = 1)
            
            num_batches = num_batches+1
                
    combined=[]
    for i in range(num_batches+1):
        idx = int(period*i)
        batch = data_array[:,idx:idx+seq_len,...]            
        combined.append(batch)
    combined = [b for b in combined if b.shape[1]==seq_len]
    combined = np.asarray(combined)
    return combined


def read_obs(obs_file, y_vars, x_data):
    """
    read and format multiple observation files. we read in the pretrain data to
    make sure we have the same indexing.
    :param x_data: [xr.Dataset] xarray dataset used to match spatial and
    temporal domain
    :param y_vars: [list of str] which variables_to_log to prepare data for
    :param obs_file: [list] filenames of observation file
    :return: [xr dataset] the observations in the same time
    """
    ds = xr.open_zarr(obs_file, consolidated=False)
    obs = xr.merge([x_data, ds], join="left")
    obs = obs[y_vars]
    return obs


def join_catch_properties(x_data_ts, catch_props):
    """
    append the catchment properties to the x time series data
    :param x_data_ts: [xr dataset] timeseries x-data
    :param catch_props: [xr dataset] catchment properties data
    :return: [xr dataset] the merged datasets
    """
    # broadcast the catchment properties on the ts data so that there is a value
    # for each date
    _, ds_catch = xr.broadcast(x_data_ts, catch_props)
    return xr.merge([x_data_ts, ds_catch], join="left")


def prep_catch_props(x_data_ts, catch_prop_file, catch_prop_vars, spatial_idx_name, replace_nan_with_mean=True):
    """
    read catch property file and join with ts data
    :param x_data_ts: [xr dataset] timeseries x-data
    :param catch_prop_file: [str] the feather file of catchment attributes
    :param catch_prop_vars: [list of str] the catchment attributes to use, if None, all attributes will be kept
    :param spatial_idx_name: [str] name of column that is used for spatial
        index (e.g., 'seg_id_nat')
    :param replace_nan_with_mean: [bool] if true, any nan will be replaced with
    the mean of that variable
    :return: [xr dataset] merged datasets
    """
    df_catch_props = pd.read_feather(catch_prop_file)
    
    #keep only the requested variables
    if catch_prop_vars:
        catch_prop_vars.append(spatial_idx_name) 
        df_catch_props = df_catch_props[catch_prop_vars]

    # replace nans with column means
    if replace_nan_with_mean:
        df_catch_props = df_catch_props.apply(
            lambda x: x.fillna(x.mean()), axis=0
        )
    ds_catch_props = df_catch_props.set_index(spatial_idx_name).to_xarray()

    return join_catch_properties(x_data_ts, ds_catch_props)


def reshape_for_training(data):
    """
    reshape the data for training
    :param data: training data (either x or y_dataset or mask) dims: [nbatch, nseg,
    len_seq, nfeat/nout]
    :return: reshaped data [nbatch * nseg, len_seq, nfeat/nout]
    """
    n_batch, n_seg, seq_len, n_feat = data.shape
    return np.reshape(data, [n_batch * n_seg, seq_len, n_feat])


def get_exclude_start_end(exclude_grp):
    """
    get the start and end dates for the exclude group
    :param exclude_grp: [dict] dictionary representing the exclude group from
    the exclude yml file
    :return: [tuple of datetime objects] start date, end date
    """
    start = exclude_grp.get("start_date")
    if start:
        start = datetime.datetime.strptime(start, "%Y-%m-%d")

    end = exclude_grp.get("end_date")
    if end:
        end = datetime.datetime.strptime(end, "%Y-%m-%d")
    return start, end


def get_exclude_vars(exclude_grp):
    """
    get the variables_to_log to exclude for the exclude group
    :param exclude_grp: [dict] dictionary representing the exclude group from
    the exclude yml file
    :return: [list] variables_to_log to exclude
    """
    variable = exclude_grp.get("variable")
    if not variable or variable == "both":
        return ["seg_tave_water", "seg_outflow"]
    elif variable == "temp":
        return ["seg_tave_water"]
    elif variable == "flow":
        return ["seg_outflow"]
    else:
        raise ValueError("exclude variable must be flow, temp, or both")


def get_exclude_seg_ids(exclude_grp, all_segs):
    """
    get the segments to exclude
    :param exclude_grp: [dict] dictionary representing the exclude group from
    the exclude yml file
    :param all_segs: [array] all of the segments. this is needed if we are doing
    a reverse exclusion
    :return: [list like] the segments to exclude
    """
    # ex_segs are the sites to exclude
    if "seg_id_nats_ex" in exclude_grp.keys():
        ex_segs = exclude_grp["seg_id_nats_ex"]
    # exclude all *but* the "seg_id_nats_in"
    elif "seg_id_nats_in" in exclude_grp.keys():
        ex_mask = ~all_segs.isin(exclude_grp["seg_id_nats_in"])
        ex_segs = all_segs[ex_mask]
    else:
        ex_segs = all_segs
    return ex_segs


def exclude_segments(y_data, exclude_segs):
    """
    exclude segments from being trained on by setting their weights as zero
    :param y_data:[xr dataset] y_dataset data. this is used to get the dimensions
    :param exclude_segs: [list] list of segments to exclude in the loss
    calculation
    :return:
    """
    weights = initialize_weights(y_data, 1)
    for seg_grp in exclude_segs:
        # get the start and end dates is present
        start, end = get_exclude_start_end(seg_grp)
        exclude_vars = get_exclude_vars(seg_grp)
        segs_to_exclude = get_exclude_seg_ids(seg_grp, weights.seg_id_nat)

        # loop through the data_vars
        for v in exclude_vars:
            # set those weights to zero
            weights[v].load()
            weights[v].loc[
                dict(date=slice(start, end), seg_id_nat=segs_to_exclude)
            ] = 0
    return weights


def initialize_weights(y_data, initial_val=1):
    """
    initialize all weights with a value.
    :param y_data:[xr dataset] y_dataset data. this is used to get the dimensions
    :param initial_val: [num] a number to initialize the weights with. should
    be between 0 and 1 (inclusive)
    :return: [xr dataset] dataset weights initialized with a uniform value
    """
    weights = y_data.copy(deep=True)
    for v in y_data.data_vars:
        weights[v].load()
        weights[v].loc[:, :] = initial_val
    return weights


def reduce_training_data_random(
    data_file,
    train_start_date="1980-10-01",
    train_end_date="2004-09-30",
    reduce_amount=0,
    out_file=None,
    segs=None,
):
    """
    artificially reduce the amount of training data in the training dataset
    :param train_start_date: [str] date (fmt YYYY-MM-DD) for when training data
    starts
    :param train_end_date: [str] date (fmt YYYY-MM-DD) for when training data
    ends
    :param data_file: [str] path to the observations data file
    :param reduce_amount: [float] fraction to reduce the training data by.
    For example, if 0.9, a random 90% of the training data will be set to nan
    :param out_file: [str] file to which the reduced dataset will be written
    :param segs: [array-like] segments to reduce data of
    :return: [xarray dataset] updated weights (nan where reduced)
    """
    # read in an convert to dataframe
    ds = xr.open_zarr(data_file)
    df = ds.to_dataframe()
    idx = pd.IndexSlice
    df_trn = df.loc[idx[train_start_date:train_end_date, :], :]
    if segs:
        df_trn = df_trn.loc[idx[:, segs], :]
    non_null = df_trn.dropna()
    reduce_idx = non_null.sample(frac=reduce_amount).index
    df.loc[reduce_idx] = np.nan
    reduced_ds = df.to_xarray()
    if out_file:
        reduced_ds.to_zarr(out_file)
    return reduced_ds


def filter_reduce_dates(df, start_date, end_date, reduce_between=False):
    df_filt = df.copy()
    df_filt = df_filt.reset_index()
    if reduce_between:
        df_filt = df_filt[
            (df_filt["date"] > start_date) & (df_filt["date"] < end_date)
        ]
    else:
        df_filt = df_filt[
            (df_filt["date"] < start_date) | (df_filt["date"] > end_date)
        ]
    return df_filt.set_index(df.index.names)


def reduce_training_data_continuous(
    data_file,
    reduce_start="1980-10-01",
    reduce_end="2004-09-30",
    train_start=None,
    train_end=None,
    segs=None,
    reduce_between=True,
    out_file=None,
):
    """
    reduce the amount of data in the training dataset by replacing a section
    or the inverse of that section with nan
    :param data_file: [str] path to the observations data file
    :param reduce_start: [str] date (fmt YYYY-MM-DD) for when reduction data
    starts
    :param reduce_end: [str] date (fmt YYYY-MM-DD) for when reduction data
    ends
    :param train_start: [str] data (fmt YYYY-MM-DD) the start of the training
    data. If left None (default) the reduction will be done for all of the
    dates. This is only relevant if reduce_between == False, because in that
    case, the inverse of the range (reduce_start, reduce_end) is used and it may
    be necessary to limit making the nans to the training time period
    :param train_end: [str] data (fmt YYYY-MM-DD) the end of the training
    data. If left None (default) the reduction will be done for all of the
    dates. This is only relevant if reduce_between == False, because in that
    case, the inverse of the range (reduce_start, reduce_end) is used and it may
    be necessary to limit making the nans to the training time period
    :param segs: [list-like] segment id's for which the data should be reduced
    :param reduce_between: [bool] if True the data *in* the range (reduce_start,
    reduce_end) will be made nan. if False, the data *outside* of that range
    will be made nan
    :param out_file: [str] file to which the reduced dataset will be written
    :return: [xarray dataset] updated weights (nan where reduced)
    """
    # read in an convert to dataframe
    ds = xr.open_zarr(data_file)
    df = ds.to_dataframe()
    idx = pd.IndexSlice
    df_red = df.copy()
    df_red = df_red.loc[idx[train_start:train_end, :]]
    if segs:
        df_red = df_red.loc[idx[:, segs], :]
    df_red = filter_reduce_dates(
        df_red, reduce_start, reduce_end, reduce_between
    )
    df.loc[df_red.index] = np.nan
    reduced_ds = df.to_xarray()
    if out_file:
        reduced_ds.to_zarr(out_file)
    return reduced_ds


def convert_batch_reshape(
    dataset,
    spatial_idx_name="seg_id_nat",
    time_idx_name="date",
    seq_len=365,
    offset=1.0,
    fill_batch=True, 
    fill_nan=False,
    fill_time=False
):
    """
    convert xarray dataset into numpy array, swap the axes, batch the array and
    reshape for training
    :param dataset: [xr dataset] data to be batched
    :param spatial_idx_name: [str] name of column that is used for spatial
        index (e.g., 'seg_id_nat')
    :param time_idx_name: [str] name of column that is used for temporal index
        (usually 'time')
    :param seq_len: [int] length of sequences (e.g., 365)
    :param offset: [float] 0-1, how to offset the batches (e.g., 0.5 means that
    the first batch will be 0-365 and the second will be 182-547)
    :param fill_batch: [bool] when True, batches are filled to match the seq_len.
    This ensures that data are not dropped when the data_array length is not
    a multiple of the seq_len. Data are added to the end of the sequence.
    When False, data may be dropped.
    :param fill_nan: [bool] When True, filled in data are np.nan (e.g., because 
    data_array is observation data that should not contribute to the loss).
    :param fill_time: [bool] When True, filled in data are time indices that
    follow in sequence from the previous timesteps. When False, filled in data 
    are replicates of the previous timesteps.
    :return: [numpy array] batched and reshaped dataset
    """
    # If there is no dataset (like if a test or validation set is not supplied)
    # just return None
    if not dataset:
        return None

    # convert xr.dataset to numpy array
    dataset = dataset.transpose(spatial_idx_name, time_idx_name)

    arr = dataset.to_array().values

    # if the dataset is empty, just return it as is
    if dataset[time_idx_name].size == 0:
        return arr

    # before [nfeat, nseg, ndates]; after [nseg, ndates, nfeat]
    # this is the order that the split into batches expects
    arr = np.moveaxis(arr, 0, -1)

    # batch the data
    # after [nbatch, nseg, seq_len, nfeat]
    batched = split_into_batches(arr, seq_len=seq_len, offset=offset,
                                 fill_batch=fill_batch, fill_nan=fill_nan,
                                 fill_time=fill_time)

    # reshape data
    # after [nbatch * nseg, seq_len, nfeat]
    reshaped = reshape_for_training(batched)
    return reshaped


def coord_as_reshaped_array(
    dataset,
    coord_name,
    spatial_idx_name="seg_id_nat",
    time_idx_name="date",
    seq_len=365,
    offset=1.0,
    fill_batch=True, 
    fill_nan=False,
    fill_time=False
):
    """
    convert an xarray coordinate to an xarray data array and reshape that array
    :param dataset:
    :param coord_name: [str] the name of the coordinate to convert/reshape
    :param spatial_idx_name: [str] name of column that is used for spatial
        index (e.g., 'seg_id_nat')
    :param time_idx_name: [str] name of column that is used for temporal index
        (usually 'time')
    :param seq_len: [int] length of sequences (e.g., 365)
    :param offset: [float] 0-1, how to offset the batches (e.g., 0.5 means that
    the first batch will be 0-365 and the second will be 182-547)
    :param fill_batch: [bool] when True, batches are filled to match the seq_len.
    This ensures that data are not dropped when the data_array length is not
    a multiple of the seq_len. Data are added to the end of the sequence.
    When False, data may be dropped.
    :param fill_nan: [bool] When True, filled in data are np.nan (e.g., because 
    data_array is observation data that should not contribute to the loss).
    When False, filled in data are replicates of the previous timesteps.
    :param fill_time: [bool] When True, filled in data are time indices that
    follow in sequence from the previous timesteps. When False, filled in data 
    are replicates of the previous timesteps.
    :return:
    """
    # If there is no dataset (like if a test or validation set is not supplied)
    # just return None
    if not dataset:
        return None

    # I need one variable name. It can be any in the dataset, but I'll use the
    # first
    first_var = next(iter(dataset.data_vars.keys()))
    coord_array = xr.broadcast(dataset[coord_name], dataset[first_var])[0]
    new_var_name = coord_name + "1"
    dataset[new_var_name] = coord_array
    reshaped_np_arr = convert_batch_reshape(
        dataset[[new_var_name]],
        spatial_idx_name,
        time_idx_name,
        seq_len=seq_len,
        offset=offset,
        fill_batch=fill_batch, 
        fill_nan=fill_nan,
        fill_time=fill_time
    )
    return reshaped_np_arr


def check_if_finite(xarr):
    assert np.isfinite(xarr.to_array().values).all()


def log_variables(y_dataset, variables_to_log):
    """
    take the log of given variables
    :param variables_to_log: [list of str] variables to take the log of
    :param y_dataset: [xr dataset] the y data
    :return: [xr dataset] the data logged
    """
    for v in variables_to_log:
        y_dataset[v].load()
        y_dataset[v].loc[:, :] = y_dataset[v] + 1e-6
        y_dataset[v].loc[:, :] = xr.ufuncs.log(y_dataset[v])
    return y_dataset


def prep_y_data(
    y_data_file,
    y_vars,
    x_data,
    train_start_date,
    train_end_date,
    val_start_date=None,
    val_end_date=None,
    test_start_date=None,
    test_end_date=None,
    train_sites=None,
    val_sites=None,
    test_sites=None,
    explicit_spatial_partition=True,
    spatial_idx_name="seg_id_nat",
    time_idx_name="date",
    seq_len=365,
    log_vars=None,
    exclude_file=None,
    normalize_y=True,
    y_type="obs",
    y_std=None,
    y_mean=None,
    trn_offset = 1.0,
    tst_val_offset = 1.0,
    fill_batch=True, 
    fill_nan=True
):
    """
    prepare y_dataset data

    :param y_data_file: [str] temperature observations file
    :param y_vars: [str or list of str] target variable(s)
    :param x_data: [xr.Dataset] xarray dataset used to match spatial and
    temporal domain
    :param spatial_idx_name: [str] name of column that is used for spatial
        index (e.g., 'seg_id_nat')
    :param time_idx_name: [str] name of column that is used for temporal index
        (usually 'time')
    :param train_start_date: [str or list] fmt: "YYYY-MM-DD"; date(s) to start
    train period (can have multiple discontinuous periods)
    :param train_end_date: [str or list] fmt: "YYYY-MM-DD"; date(s) to end train
    period (can have multiple discontinuous periods)
    :param val_start_date: [str or list] fmt: "YYYY-MM-DD"; date(s) to start
    validation period (can have multiple discontinuous periods)
    :param val_end_date: [str or list] fmt: "YYYY-MM-DD"; date(s) to end
    validation period (can have multiple discontinuous periods)
    :param test_start_date: [str or list] fmt: "YYYY-MM-DD"; date(s) to start
    test period (can have multiple discontinuous periods)
    :param test_end_date: [str or list] fmt: "YYYY-MM-DD"; date(s) to end test
    period (can have multiple discontinuous periods)
    :param train_sites: [list of site_ids] all sites for the training partition.
    :param val_sites: [list of site_ids] all sites for the validation partition.
    :param test_sites: [list of site_ids] all sites for the testing partition.
    :param explicit_spatial_partition: [bool] when True and train_sites 
    (val_sites, test_sites) is specified, the train_sites (val_sites, tst_sites)
    are removed from the other partitions, unless sites are provided for those 
    partitions. When False and train_sites (val_sites, test_sites) is specified, 
    the train_sites (val_sites, tst_sites) may appear in other partitions.
    :param seq_len: [int] length of sequences (e.g., 365)
    :param log_vars: [list-like] which variables_to_log (if any) to take log of
    :param exclude_file: [str] path to exclude file
    :param normalize_y: [bool] whether or not to normalize the y_dataset values
    :param y_type: [str] "obs" if observations or "pre" if pretraining
    :param y_std: [array-like] standard deviations of y_dataset variables_to_log
    :param y_mean: [array-like] means of y_dataset variables_to_log
    :param trn_offset: [str] value for the training offset
    :param tst_val_offset: [str] value for the testing and validation offset
    :param fill_batch: [bool] when True, batches are filled to match the seq_len.
    This ensures that data are not dropped when the data_array length is not
    a multiple of the seq_len. Data are added to the end of the sequence.
    When False, data may be dropped.
    :param fill_nan: [bool] When True, filled in data are np.nan (e.g., because 
    data_array is observation data that should not contribute to the loss).
    When False, filled in data are replicates of the previous timesteps.
    :returns: training and testing data along with the means and standard
    deviations of the training input and output data
    """
    # I assume that if `y_vars` is a string only one variable has been passed
    # so I put that in a list which is what the rest of the functions expect
    if isinstance(y_vars, str):
        y_vars = [y_vars]

    # when specifying mean and std, they get passed as an np.ndarray where we need xr.Datasets
    if isinstance(y_mean, np.ndarray):
        y_mean = xr.Dataset(dict(zip(y_vars,y_mean)))
        y_std = xr.Dataset(dict(zip(y_vars,y_std)))

    y_data = read_obs(y_data_file, y_vars, x_data)

    y_trn, y_val, y_tst = separate_trn_tst(
        y_data,
        time_idx_name,
        train_start_date,
        train_end_date,
        val_start_date,
        val_end_date,
        test_start_date,
        test_end_date,
    )


    # replace trn, val and tst sites' data with np.nan
    if train_sites:
        y_trn = y_trn.where(y_trn[spatial_idx_name].isin(train_sites))
        if explicit_spatial_partition:
            #remove training sites from validation and testing, unless
            # sites are provided for those partitions
            if not val_sites:
                y_val = y_val.where(~y_val[spatial_idx_name].isin(train_sites))
            if not test_sites:
                y_tst = y_tst.where(~y_tst[spatial_idx_name].isin(train_sites))
    
    if val_sites:
        y_val = y_val.where(y_val[spatial_idx_name].isin(val_sites))
        if explicit_spatial_partition:
            #remove validation sites from training and testing, unless
            # sites are provided for those partitions
            if not train_sites:
                y_trn = y_trn.where(~y_trn[spatial_idx_name].isin(val_sites))
            if not test_sites:
                y_tst = y_tst.where(~y_tst[spatial_idx_name].isin(val_sites))
    
    if test_sites:
        y_tst = y_tst.where(y_tst[spatial_idx_name].isin(test_sites))
        if explicit_spatial_partition:
            #remove test sites from training and validation, unless
            # sites are provided for those partitions
            if not train_sites:
                y_trn = y_trn.where(~y_trn[spatial_idx_name].isin(test_sites))
            if not val_sites:
                y_val = y_val.where(~y_val[spatial_idx_name].isin(test_sites))

    if log_vars:
        y_trn = log_variables(y_trn, log_vars)

    # filter pretrain/finetune y_dataset
    if exclude_file:
        exclude_segs = read_exclude_segs_file(exclude_file)
        y_wgts = exclude_segments(y_trn, exclude_segs=exclude_segs)
    else:
        y_wgts = initialize_weights(y_trn)
    # scale y_dataset training data and get the mean and std
    # scale the validation partition to benchmark epoch performance
    if normalize_y:
    # check if mean and std are already calculated/exist
        if not isinstance(y_std, xr.Dataset) or not isinstance(y_mean, xr.Dataset):
            y_trn, y_std, y_mean = scale(y_trn)
        else:
            y_trn, _, _ = scale(y_trn,y_std,y_mean)
            
        if y_val:
            y_val, _, _ = scale(y_val, y_std, y_mean)

        if y_tst:
            y_tst, _, _ = scale(y_tst, y_std, y_mean)
    else:
        _, y_std, y_mean = scale(y_trn)
        y_std = y_std/y_std
        y_mean = y_mean * 0 

    if y_type == 'obs':
        data = {
            "y_obs_trn": convert_batch_reshape(
                y_trn, spatial_idx_name, time_idx_name, offset=trn_offset, seq_len=seq_len,
                fill_batch=fill_batch, fill_nan=fill_nan, fill_time=False
            ),
            "y_obs_wgts": convert_batch_reshape(
                y_wgts, spatial_idx_name, time_idx_name, offset=trn_offset, seq_len=seq_len,
                fill_batch=fill_batch, fill_nan=fill_nan, fill_time=False
            ),
            "y_obs_val": convert_batch_reshape(
                y_val, spatial_idx_name, time_idx_name, offset=tst_val_offset, seq_len=seq_len,
                fill_batch=fill_batch, fill_nan=fill_nan, fill_time=False
            ),
            "y_obs_tst": convert_batch_reshape(
                y_tst, spatial_idx_name, time_idx_name, offset=tst_val_offset, seq_len=seq_len,
                fill_batch=fill_batch, fill_nan=fill_nan, fill_time=False
            ),
            "y_std": y_std.to_array().values,
            "y_mean": y_mean.to_array().values,
            "y_obs_vars": y_vars,
        }
    elif y_type == 'pre':
        if normalize_y:
            if not isinstance(y_std, xr.Dataset) or not isinstance(y_mean, xr.Dataset):
                y_trn, y_std, y_mean = scale(y_trn)
            else:
                y_data, _, _ = scale(y_data, y_std, y_mean)

        data = {
            "y_pre_full": convert_batch_reshape(
                y_data, spatial_idx_name, time_idx_name, offset=trn_offset, seq_len=seq_len,
                fill_batch=fill_batch, fill_nan=fill_nan, fill_time=False
            ),
            "y_pre_trn": convert_batch_reshape(
                y_trn, spatial_idx_name, time_idx_name, offset=trn_offset, seq_len=seq_len,
                fill_batch=fill_batch, fill_nan=fill_nan, fill_time=False
            ),
            "y_pre_vars": y_vars,
        }
    return data


def prep_all_data(
    x_data_file,
    y_data_file,
    x_vars,
    train_start_date,
    train_end_date,
    val_start_date=None,
    val_end_date=None,
    test_start_date=None,
    test_end_date=None,
    train_sites=None,
    val_sites=None,
    test_sites=None,
    explicit_spatial_partition=True,
    y_vars_finetune=None,
    y_vars_pretrain=None,
    spatial_idx_name="seg_id_nat",
    time_idx_name="date",
    seq_len=365,
    pretrain_file=None,
    distfile=None,
    dist_idx_name="rowcolnames",
    dist_type="updown",
    catch_prop_file=None,
    catch_prop_vars=None,
    exclude_file=None,
    log_y_vars=False,
    out_file=None,
    segs=None,
    earliest_time=None,
    latest_time=None,
    normalize_y=True,
    trn_offset = 1.0,
    tst_val_offset = 1.0,
    check_pre_partitions=True,
    fill_batch=True
):
    """
    prepare input and output data for DL model training read in and process
    data into training and testing datasets. the training and testing data are
    scaled to have a std of 1 and a mean of zero
    :param x_data_file: [str] path to Zarr file with x data. Data should have
    a spatial coordinate and a time coordinate that are specified in the
    `spatial_idx_name` and `time_idx_name` arguments. Assumes that all spaces will be used,
    unless segs is specified. Assumes all times will be used,
    unless an earliest_time or latest_time is specified.
    :param y_data_file: [str] observations Zarr file. Data should have a spatial
    coordinate and a time coordinate that are specified in the
    spatial_idx_name` and `time_idx_name` arguments
    :param train_start_date: [str or list] fmt: "YYYY-MM-DD"; date(s) to start
    train period (can have multiple discontinuous periods)
    :param train_end_date: [str or list] fmt: "YYYY-MM-DD"; date(s) to end train
    period (can have multiple discontinuous periods)
    :param val_start_date: [str or list] fmt: "YYYY-MM-DD"; date(s) to start
    validation period (can have multiple discontinuous periods)
    :param val_end_date: [str or list] fmt: "YYYY-MM-DD"; date(s) to end
    validation period (can have multiple discontinuous periods)
    :param test_start_date: [str or list] fmt: "YYYY-MM-DD"; date(s) to start
    test period (can have multiple discontinuous periods)
    :param test_end_date: [str or list] fmt: "YYYY-MM-DD"; date(s) to end test
    period (can have multiple discontinuous periods)
    :param train_sites: [list of site_ids] all sites for the training partition.
    :param val_sites: [list of site_ids] all sites for the validation partition.
    :param test_sites: [list of site_ids] all sites for the testing partition.
    :param explicit_spatial_partition: [bool] when True and train_sites 
    (val_sites, test_sites) is specified, the train_sites (val_sites, tst_sites)
    are removed from the other partitions. When False and train_sites 
    (val_sites, test_sites) is specified, the train_sites (val_sites, tst_sites)
    may appear in other partitions.
    :param spatial_idx_name: [str] name of column that is used for spatial
    index (e.g., 'seg_id_nat')
    :param time_idx_name: [str] name of column that is used for temporal index
    (usually 'time')
    :param x_vars: [list] variables_to_log that should be used as input. If None, all
    of the variables_to_log will be used
    :param y_vars_finetune: [str or list of str] finetune target variable(s)
    :param y_vars_pretrain: [str or list of str] pretrain target variable(s)
    :param seq_len: [int] length of sequences (e.g., 365)
    :param pretrain_file: [str] Zarr file with the pretraining data. Should have
    a spatial coordinate and a time coordinate that are specified in the
    `spatial_idx_name` and `time_idx_name` arguments
    :param distfile: [str] path to the distance matrix .npz file
    :param dist_idx_name: [str] name of index to sort dist_matrix by. This is
    the name of an array in the distance matrix .npz file
    :param dist_type: [str] type of distance matrix ("upstream", "downstream" or
    "updown")
    :param catch_prop_file: [str] the path to the catchment properties file. If
    left unfilled, the catchment properties will not be included as predictors
    :param catch_prop_vars: [list of str] list of catchment properties to use. If
    left unfilled and a catchment property file is supplied all variables will be used.
    :param exclude_file: [str] path to exclude file
    :param log_y_vars: [bool] whether or not to take the log of discharge in
    training
    :param segs: [list-like] which segments to prepare the data for
    :param earliest_time: [str] when specified, filters the x_data to remove earlier times
    :param latest_time: [str] when specified, filters the x_data to remove later times
    :param normalize_y: [bool] whether or not to normalize the y_dataset values
    :param out_file: [str] file to where the values will be written
    :param trn_offset: [str] value for the training offset
    :param tst_val_offset: [str] value for the testing and validation offset
    :param check_pre_partitions [bool] when True, pretarining partitions are
    checked for unique data in each partition.
    :param fill_batch: [bool] when True, batches are filled to match the seq_len.
    This ensures that data are not dropped when the data_array length is not
    a multiple of the seq_len. Data are added to the end of the sequence.
    When False, data may be dropped.
    :returns: training and testing data along with the means and standard
    deviations of the training input and output data
            "x_trn": x training data
            "x_val": x validation data
            "x_tst": x test data
            "x_std": x standard deviations
            "x_mean": x means
            "x_cols": x column names
            "ids_trn": segment ids of the training data
            "times_trn": dates of the training data
            "ids_val": segment ids of the validation data
            "times_val": dates of the validation data
            "ids_tst": segment ids of the test data
            "times_tst": dates of the test data
            'y_pre_trn': y_dataset pretrain data for train set
            'y_obs_trn': y_dataset observations for train set
            "y_pre_wgts": y_dataset weights for pretrain data
            "y_obs_wgts": weights for y_dataset observations
            "y_obs_val": y_dataset observations for validation set
            "y_obs_tst": y_dataset observations for train set
            "y_std": standard deviations of y_dataset data
            "y_mean": means of y_dataset data
            "y_obs_vars": y_dataset observation variable names
            'y_pre_val': y_dataset pretrain data for validation data
            'y_pre_tst': y_dataset pretrain data for test data
            'y_pre_vars':  y_dataset pretrain data variable names
            "dist_matrix": prepared adjacency matrix
    """
    if pretrain_file and not y_vars_pretrain:
        raise ValueError("included pretrain file but no pretrain vars")

    x_data = xr.open_zarr(x_data_file,consolidated=False)
    x_data = x_data.sortby([spatial_idx_name, time_idx_name])

    if segs:
        x_data = x_data.sel({spatial_idx_name: segs})
    
    if earliest_time:
        mask_etime = (x_data[time_idx_name] >= np.datetime64(earliest_time))
        x_data = x_data.where(mask_etime, drop=True)
    
    if latest_time:
        mask_ltime = (x_data[time_idx_name] <= np.datetime64(latest_time))
        x_data = x_data.where(mask_ltime, drop=True)

    x_data = x_data[x_vars]

    if catch_prop_file:
        x_data = prep_catch_props(x_data, catch_prop_file, catch_prop_vars, spatial_idx_name)
        #update the list of x_vars
        x_vars = list(x_data.data_vars)
    
    # make sure we don't have any weird or missing input values
    check_if_finite(x_data)
    
    x_trn, x_val, x_tst = separate_trn_tst(
        x_data,
        time_idx_name,
        train_start_date,
        train_end_date,
        val_start_date,
        val_end_date,
        test_start_date,
        test_end_date,
    )
    
    x_trn_scl, x_std, x_mean = scale(x_trn)

    x_scl, _, _ = scale(x_data,std=x_std,mean=x_mean)

    if x_val:
        x_val_scl, _, _ = scale(x_val, std=x_std, mean=x_mean)
    else:
        x_val_scl = None

    if x_tst:
        x_tst_scl, _, _ = scale(x_tst, std=x_std, mean=x_mean)
    else:
        x_tst_scl = None

    # read, filter observations for finetuning

    x_data_dict = {
        "x_pre_full": convert_batch_reshape(
            x_scl,spatial_idx_name, time_idx_name, seq_len=seq_len,
            offset=trn_offset, fill_batch=fill_batch, fill_nan=False, fill_time=False
        ),
        "x_trn": convert_batch_reshape(
            x_trn_scl, spatial_idx_name, time_idx_name, seq_len=seq_len,
            offset=trn_offset, fill_batch=fill_batch, fill_nan=False, fill_time=False
        ),
        "x_val": convert_batch_reshape(
            x_val_scl,
            spatial_idx_name,
            time_idx_name,
            offset=tst_val_offset,
            seq_len=seq_len, 
            fill_batch=fill_batch, 
            fill_nan=False, 
            fill_time=False
        ),
        "x_tst": convert_batch_reshape(
            x_tst_scl,
            spatial_idx_name,
            time_idx_name,
            offset=tst_val_offset,
            seq_len=seq_len, 
            fill_batch=fill_batch, 
            fill_nan=False, 
            fill_time=False
        ),
        "x_std": x_std.to_array().values,
        "x_mean": x_mean.to_array().values,
        "x_vars": np.array(x_vars),
        "ids_trn": coord_as_reshaped_array(
            x_trn,
            spatial_idx_name,
            spatial_idx_name,
            time_idx_name,
            offset=trn_offset,
            seq_len=seq_len, 
            fill_batch=fill_batch, 
            fill_nan=False, 
            fill_time=False
        ),
        "times_trn": coord_as_reshaped_array(
            x_trn,
            time_idx_name,
            spatial_idx_name,
            time_idx_name,
            offset=trn_offset,
            seq_len=seq_len, 
            fill_batch=fill_batch, 
            fill_nan=False, 
            fill_time=True
        ),
        "ids_val": coord_as_reshaped_array(
            x_val,
            spatial_idx_name,
            spatial_idx_name,
            time_idx_name,
            offset=tst_val_offset,
            seq_len=seq_len, 
            fill_batch=fill_batch, 
            fill_nan=False, 
            fill_time=False
        ),
        "times_val": coord_as_reshaped_array(
            x_val,
            time_idx_name,
            spatial_idx_name,
            time_idx_name,
            offset=tst_val_offset,
            seq_len=seq_len, 
            fill_batch=fill_batch, 
            fill_nan=False, 
            fill_time=True
        ),
        "ids_tst": coord_as_reshaped_array(
            x_tst,
            spatial_idx_name,
            spatial_idx_name,
            time_idx_name,
            offset=tst_val_offset,
            seq_len=seq_len, 
            fill_batch=fill_batch, 
            fill_nan=False, 
            fill_time=False
        ),
        "times_tst": coord_as_reshaped_array(
            x_tst,
            time_idx_name,
            spatial_idx_name,
            time_idx_name,
            offset=tst_val_offset,
            seq_len=seq_len,
            fill_batch=fill_batch, 
            fill_nan=False, 
            fill_time=True
        ),
    }
    if distfile:
        x_data_dict["dist_matrix"] = prep_adj_matrix(
            infile=distfile,
            dist_type=dist_type,
            dist_idx_name=dist_idx_name,
            segs=segs,
        )

    y_obs_data = {}
    y_pre_data = {}
    if y_data_file:
        y_obs_data = prep_y_data(
            y_data_file=y_data_file,
            y_vars=y_vars_finetune,
            x_data=x_data,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            val_start_date=val_start_date,
            val_end_date=val_end_date,
            test_start_date=test_start_date,
            test_end_date=test_end_date,
            train_sites=train_sites,
            val_sites=val_sites,
            test_sites=test_sites,
            explicit_spatial_partition=explicit_spatial_partition,
            spatial_idx_name=spatial_idx_name,
            time_idx_name=time_idx_name,
            seq_len=seq_len,
            log_vars=log_y_vars,
            exclude_file=exclude_file,
            normalize_y=normalize_y,
            y_type="obs",
            trn_offset = trn_offset,
            tst_val_offset = tst_val_offset,
            fill_batch=fill_batch, 
            fill_nan=True
        )
        
        #check that the trn, val, and tst partitions have unique data
        check_partitions(x_data_dict, y_obs_data)
        
        # if there is a y_data_file and a pretrain file, use the observation
        # mean and standard deviation to do the scaling/centering
        if pretrain_file:
            y_pre_data = prep_y_data(
                y_data_file=pretrain_file,
                y_vars=y_vars_pretrain,
                x_data=x_data,
                train_start_date=train_start_date,
                train_end_date=train_end_date,
                val_start_date=val_start_date,
                val_end_date=val_end_date,
                test_start_date=test_start_date,
                test_end_date=test_end_date,
                spatial_idx_name=spatial_idx_name,
                time_idx_name=time_idx_name,
                seq_len=seq_len,
                log_vars=log_y_vars,
                exclude_file=exclude_file,
                normalize_y=normalize_y,
                y_type="pre",
                y_std=y_obs_data["y_std"],
                y_mean=y_obs_data["y_mean"],
                trn_offset = trn_offset,
                tst_val_offset = tst_val_offset,
                fill_batch=fill_batch, 
                fill_nan=True
            )
            if check_pre_partitions:
                #check that the trn, val, and tst partitions have unique data
                check_partitions(x_data_dict, y_pre_data, pre = True)
        
    # if there is no observation file, use the pretrain mean and standard dev
    # to do the scaling/centering
    elif pretrain_file and not y_obs_data:
        y_pre_data = prep_y_data(
            y_data_file=pretrain_file,
            y_vars=y_vars_pretrain,
            x_data=x_data,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            val_start_date=val_start_date,
            val_end_date=val_end_date,
            test_start_date=test_start_date,
            test_end_date=test_end_date,
            spatial_idx_name=spatial_idx_name,
            time_idx_name=time_idx_name,
            seq_len=seq_len,
            log_vars=log_y_vars,
            exclude_file=exclude_file,
            normalize_y=normalize_y,
            y_type="pre",
            trn_offset = trn_offset,
            tst_val_offset = tst_val_offset,
            fill_batch=fill_batch, 
            fill_nan=True
        )
        if check_pre_partitions:
            #check that the trn, val, and tst partitions have unique data
            check_partitions(x_data_dict, y_pre_data, pre = True)
    else:
        raise Warning("No y_dataset data was provided")
    
    all_data = {**x_data_dict, **y_obs_data, **y_pre_data}
    if out_file:
        np.savez_compressed(out_file, **all_data)
    return all_data


def sort_dist_matrix(mat, row_col_names, segs=None):
    """
    sort the distance matrix by id
    :return:
    """
    if segs is not None:
        row_col_names = row_col_names.astype(type(segs[0]))
    df = pd.DataFrame(mat, columns=row_col_names, index=row_col_names)
    if segs:
        df = df[segs]
        df = df.loc[segs]
    df = df.sort_index(axis=0)
    df = df.sort_index(axis=1)
    return df


def prep_adj_matrix(infile, dist_type, dist_idx_name, segs=None, out_file=None):
    """
    process adj matrix.
    **The resulting matrix is sorted by id **
    :param infile: [str] path to the distance matrix .npz file
    :param dist_type: [str] type of distance matrix ("upstream", "downstream" or
    "updown")
    :param dist_idx_name: [str] name of index to sort dist_matrix by. This is
    the name of an array in the distance matrix .npz file
    :param segs: [list-like] which segments to prepare the data for
    :param out_file: [str] path to save the .npz file to
    :return: [numpy array] processed adjacency matrix
    """
    adj_matrices = np.load(infile)
    adj = adj_matrices[dist_type]
    adj = sort_dist_matrix(adj, adj_matrices[dist_idx_name], segs=segs)
    adj = np.where(np.isinf(adj), 0, adj)
    adj = -adj
    mean_adj = np.mean(adj[adj != 0])
    std_adj = np.std(adj[adj != 0])
    adj[adj != 0] = adj[adj != 0] - mean_adj
    adj[adj != 0] = adj[adj != 0] / std_adj
    adj[adj != 0] = 1 / (1 + np.exp(-adj[adj != 0]))

    I = np.eye(adj.shape[0])
    A_hat = adj.copy() + I
    D = np.sum(A_hat, axis=1)
    D_inv = D ** -1.0
    D_inv = np.diag(D_inv)
    A_hat = np.matmul(D_inv, A_hat)
    if out_file:
        np.savez_compressed(out_file, dist_matrix=A_hat)
    return A_hat


def read_exclude_segs_file(exclude_file):
    """
    read the exclude segs file. should be a yml file with start_date and list of
    segments to exclude
    --
    example exclude file:

    group_after_2017:
        start_date: "2017-10-01"
        variable: "temp"
        seg_id_nats_ex:
            - 1556
            - 1569
    group_2018_water_year:
        start_date: "2017-10-01"
        end_date: "2018-10-01"
        seg_id_nats_ex:
            - 1653
    group_all_time:
        seg_id_nats_in:
            - 1806
            - 2030

    --
    :param exclude_file: [str] exclude segs file
    :return: [list] list of dictionaries of segments to exclude. dict keys must
    have 'seg_id_nats' and may also have 'start_date' and 'end_date'
    """
    with open(exclude_file, "r") as s:
        d = yaml.safe_load(s)
    return [val for key, val in d.items()]

def check_partitions(data, y, pre=False):
    '''
    Function to check that trn, val, and tst partitions have unique observation
    times and site ids
    
    :param data: [dict] data dictionary with keys 'ids_<partition>' and 
    'times_<partition>', where partition = trn, val, and tst.
    :param y: [dict] data dictionary with keys 'y_<phase>_<partition>', 
    where phase is obs or pre, and partition = trn, val, and tst.
    :param pre: [bool] when True, phase is pre for pretraining. when False,
    phase is obs.
    
    returns 0 when all data have unique times and sites. Otherwise sys.exit
    is called.
    '''
    if pre:
        phase = 'pre'
    else:
        phase = 'obs'
    
    #Get site ids and times for train, val, test data into a matrix
    df_trn = pd.DataFrame({'ids': np.reshape(data['ids_trn'], (-1)),
                           'times': np.reshape(data['times_trn'], (-1)),
                           'obs': np.reshape(y['y_'+phase+'_trn'], (-1))})
    df_val = pd.DataFrame({'ids': np.reshape(data['ids_val'], (-1)),
                           'times': np.reshape(data['times_val'], (-1)),
                           'obs': np.reshape(y['y_'+phase+'_val'], (-1))})
    df_tst = pd.DataFrame({'ids': np.reshape(data['ids_tst'], (-1)),
                           'times': np.reshape(data['times_tst'], (-1)),
                           'obs': np.reshape(y['y_'+phase+'_tst'], (-1))})
    #remove rows with nan observations
    df_trn.dropna(inplace=True)
    df_val.dropna(inplace=True)
    df_tst.dropna(inplace=True)
    
    #When the data are aggregated into a single dataframe
    # there should be no duplicated rows
    df = pd.concat([df_trn, df_val, df_tst])
    duplicate_rows = df.duplicated(keep=False)
    if any(duplicate_rows == True):
        print(df.loc[duplicate_rows])
        sys.exit('There are observations within multiple data partitions')
    else:
        return(0)

def get_time_unit(data_array):
    '''
    Function to get the timestep unit from a numpy.datetime64 array column
    
    :param data_array: [np 3D array] the array's second axis must be time
    in np.datetime64 format with nanoseconds specified (default).
    
    returns the unit of the timestep (day, month, etc.)
    '''
    time_delta = data_array[0,1,0] - data_array[0,0,0]
    time_unit = np.datetime_data(time_delta)[0]
    if time_unit != 'ns':
        sys.exit('time unit must be provided as YYYY-MM-DDT:HH:MM:SS.000000000 nanoseconds')
    
    if time_delta == 86400000000000:
        time_unit = 'D'
    elif time_delta == 24000000000:
        time_unit = 'h'
    else:
        sys.exit('time_delta does not correspond to 1 h or D')
    
    return(time_unit)