import pandas as pd
import numpy as np
import yaml
import xarray as xr
import datetime
from dateutil.relativedelta import relativedelta


def get_non_varying_cols(df, std_thresh=1e-5):
    """
    get columns that don't vary at all
    :param df: [dataframe] dataframe of input/output data
    :param std_thresh: [float] when a variable (column) has a std deviation
    below the std_thresh it will be counted as "non-varying" 
    :return: index of columns (like a list of column names) that have a std
    below the threshold
    """
    st_dev = df.std()
    sml_std = st_dev[st_dev < std_thresh]
    print(f"the {sml_std.index} columns were removed b/c they had small std\
            deviations")
    return sml_std.index.to_list()


def get_unwanted_cols(df):
    """
    get the columns that should be removed in the input data including
    1) columns that don't vary at all, 2) columns that are for the model but
    aren't actually predictors like seg_id_nat and 3) columns that are too
    strong of predictors?
    """
    # non_varying_cols = get_non_varying_cols(df)
    # sntemp_cols = ['model_idx', 'date']
    # unwanted_cols = ['seg_upstream_inflow', 'seginc_gwflow', 'seg_width']
    # first lets just try taking the flow and temp out since that is what
    # xiaowei did 
    # unwanted_cols = []
    # non_varying_cols = []
    # unwanted_cols.extend(non_varying_cols)
    # unwanted_cols.extend(sntemp_cols)
    return ['model_idx', 'date']


def convert_to_np_arr(df):
    """
    convert dataframe to numpy arrays with dimensions [nseg, ndate, nfeat]
    :param df: [dataframe] input or output data
    :return: numpy array
    """
    df = df.reset_index()
    seg_id_groups = df.groupby('seg_id_nat')
    # this should work, but it raises an error
    # data_by_seg_id = seg_id_groups.apply(pd.DataFrame.to_numpy)
    seg_id_arrays = []
    for seg_id, seg_id_df in seg_id_groups:
        del seg_id_df['seg_id_nat']
        seg_id_arrays.append(seg_id_df.to_numpy())
    array_for_all_seg_ids = np.array(seg_id_arrays)
    return array_for_all_seg_ids


def filter_unwanted_cols(df):
    """
    filter out unwanted columns
    :param df: [dataframe] unfiltered data
    :return: [dataframe] filtered data
    """
    unwanted_cols = get_unwanted_cols(df)
    wanted_cols = [c for c in df.columns if c not in unwanted_cols]
    return df[wanted_cols]


def sep_x_y(ds, predictor_vars=None):
    """
    separate into input and output
    :param ds: [xr dataset] the raw input and output data
    :param predictor_vars: [list] list of predictor column names
    :return: [tuple] df of predictors (x), df of targets (y)
    """
    target_vars = ['seg_tave_water', 'seg_outflow']
    if not predictor_vars:
        predictor_vars = [v for v in ds.data_vars if v not in target_vars]
    return ds[predictor_vars], ds[target_vars]


def get_df_for_rand_seg(df, seg_id):
    df_seg = df[df['seg_id_nat'] == seg_id]
    return df_seg


def scale(data_arr, std=None, mean=None):
    """
    scale the data so it has a standard deviation of 1 and a mean of zero
    :param data_arr: [numpy array] input or output data with dims
    [nseg, ndates, nfeats]
    :param std: [numpy array] standard deviation if scaling test data with dims
    [nfeats]
    :param mean: [numpy array] mean if scaling test data with dims [nfeats]
    :return: scaled data with original dims
    """
    if not isinstance(std, xr.Dataset) or not isinstance(mean, xr.Dataset):
        std = data_arr.std(skipna=True)
        mean = data_arr.mean(skipna=True)
    # adding small number in case there is a std of zero
    scaled = (data_arr - mean)/(std + 1e-10)
    return scaled, std, mean


def get_format_preds(model, x, unscale=False, y_std=None, y_mean=None):
    preds = model.predict(x)
    preds = pd.Series([p[0] for p in preds])
    if unscale:
        return unscale_data(preds, y_std, y_mean)
    else:
        return preds


def unscale_data(df, std, mean):
    return (df * std) + mean


def separate_trn_tst(dataset, test_start, n_test_years):
    """
    separate the train data from the test data according to the start and end
    dates. This assumes your training data is in one continuous block and all
    the dates that are not in the training are in the testing.
    :param dataset: [xr dataset] input or output data with dims
    :param test_start: [str] date where training data should start
    :param test_end: [str] date where training data should end
    """
    start_date = datetime.datetime.strptime(test_start, '%Y-%m-%d')
    test_end = start_date + relativedelta(years=n_test_years)
    tst = dataset.sel(date=slice(test_start, test_end))
    # take all the rest
    trn_dates = dataset.date[~dataset.date.isin(tst.date)]
    trn = dataset.sel(date=trn_dates)
    return trn, tst


def split_into_batches(data_array, seq_len=365, offset=1):
    """
    split training data into batches with size of batch_size
    :param data_array: [numpy array] array of training data with dims [nseg,
    ndates, nfeat]
    :param seq_len: [int] length of sequences (i.e., 365)
    :param offset: [float] 0-1, how to offset the batches (e.g., 0.5 means that
    the first batch will be 0-365 and the second will be 182-547)
    :return: [numpy array] batched data with dims [nbatches, nseg, seq_len
    (batch_size), nfeat]
    """
    combined = []
    for i in range(int(1/offset)):
        start = int(i * offset * seq_len)
        idx = np.arange(start=start, stop=data_array.shape[1],
                        step=seq_len)
        split = np.split(data_array, indices_or_sections=idx, axis=1)
        # add all but the first and last batch since they will be smaller
        combined.extend(split[1:-1])
    combined = np.asarray(combined)
    return combined


def read_format_data(filename):
    """
    read in data into a dataframe, then format the dates and sort
    :param filename: [str] filename that has the data
    :return: [dataframe] formatted/sorted data
    """
    # read in x, y_pretrain data
    if filename.endswith('csv'):
        df = pd.read_csv(filename)
    elif filename.endswith('feather'):
        df = pd.read_feather(filename)
        del df['model_idx']
    else:
        raise ValueError('file_format should be "feather" or "csv"')
    df['date'] = pd.to_datetime(df['date'])
    df['seg_id_nat'] = pd.to_numeric(df['seg_id_nat'])
    df = df.sort_values(['date', 'seg_id_nat'])
    df = df.set_index(['seg_id_nat', 'date'])
    ds = df.to_xarray()
    return ds


def read_multiple_obs(obs_files, x_data):
    """
    read and format multiple observation files
    :param obs_files: [list] list of filenames of observation files
    :param x_data:
    :return:
    """
    obs = [x_data]
    for filename in obs_files:
        ds = read_format_data(filename)
        obs.append(ds)
    obs = xr.merge(obs, join='left')
    obs = obs[['temp_C', 'discharge_cms']]
    obs = obs.rename({'temp_C': 'seg_tave_water',
                      'discharge_cms': 'seg_outflow'})
    return obs


def reshape_for_training(data):
    """
    reshape the data for training
    :param data: training data (either x or y or mask) dims: [nbatch, nseg,
    len_seq, nfeat/nout]
    :return: reshaped data [nbatch * nseg, len_seq, nfeat/nout]
    """
    n_batch, n_seg, seq_len, n_feat = data.shape
    return np.reshape(data, [n_batch*n_seg, seq_len, n_feat])


def exclude_segments(weights, exclude_segs):
    """
    exclude segments from being trained on by setting their weights as zero
    :param weights: [xr dataset] dataset of weights
    :param exclude_segs: [list] list of segments to exclude in the loss
    calculation
    :return:
    """
    for seg_grp in exclude_segs:
        start = seg_grp.get('start_date')
        if start:
            start = datetime.datetime.strptime(start[0], '%Y-%m-%d')

        end = seg_grp.get('end_date')
        if end:
            end = datetime.datetime.strptime(end[0], '%Y-%m-%d')

        weights.seg_tave_water.loc[seg_grp['seg_id_nats'], start:end] = 0
        weights.seg_outflow.loc[seg_grp['seg_id_nats'], start:end] = 0
    return weights


def create_weight_vectors(y_data, out_cols, exclude_segs):
    """
    filter out either flow, temperature, or neither in the pre-training and 
    finetune y data
    :param y_data: [xr dataset]
    :param out_cols: [str] which columns to have count in the loss function;
    either 'temp', 'flow', or 'both'
    :param exclude_segs: [list] list of segment ids to exclude from the loss
    function
    :return: [xr dataset] dataset of weights between one and zero
    """
    weights = y_data.copy(deep=True)
    # assume all weights will be one (fully counted)
    weights.seg_tave_water.loc[:, :] = 1
    weights.seg_outflow.loc[:, :] = 1

    if out_cols == "both":
        pass
    elif out_cols == 'temp':
        weights.seg_outflow.loc[:, :] = 0
    elif out_cols == 'flow':
        weights.seg_tave_water.loc[:, :] = 0
    else:
        raise ValueError('out_cols needs to be "flow", "temp", or "both"')

    if exclude_segs:
        weights = exclude_segments(weights, exclude_segs)
    return weights


def convert_batch_reshape(dataset):
    """
    convert xarray dataset into numpy array, swap the axes, batch the array and
    reshape for training
    :param dataset: [xr dataset] x or y data
    :return: [numpy array] batched and reshaped dataset
    """
    # convert xr.dataset to numpy array
    dataset = dataset.transpose('seg_id_nat', 'date')
    arr = dataset.to_array().values

    # before [nfeat, nseg, ndates]; after [nseg, ndates, nfeat]
    # this is the order that the split into batches expects
    arr = np.moveaxis(arr, 0, -1)

    # batch the data
    # after [nbatch, nseg, seq_len, nfeat]
    batched = split_into_batches(arr)

    # reshape data
    # after [nbatch * nseg, seq_len, nfeat]
    reshaped = reshape_for_training(batched)
    return reshaped


def coord_as_reshaped_array(dataset, coord_name):
    coord_array = xr.broadcast(dataset[coord_name], dataset['seg_tave_air'])[0]
    new_var_name = coord_name + '1'
    dataset[new_var_name] = coord_array
    reshaped_np_arr = convert_batch_reshape(dataset[[new_var_name]])
    return reshaped_np_arr


def read_process_data(data_dir='data/in/', subset=True,
                      pretrain_out_vars="both", finetune_out_vars="both",
                      dist_type='upstream', test_start_date='2004-09-30',
                      n_test_yr=12, exclude_segs=None):
    """
    read in and process data into training and testing datasets. the training 
    and testing data are scaled to have a std of 1 and a mean of zero
    :param data_dir:
    :param subset: [bool] whether you want data for the subsection (True) or
    for the entire DRB (false)
    :param trn_ratio: [float] ratio of training data. as pecentage (i.e., 0.8 )
    would mean that 80% of the data would be for training and the rest for test
    :param batch_offset:
    observations. if False, the mask for all discharge values will be False
    :param pretrain_out_vars: [str] which parameters to fine tune on should be
    "temp", "flow" or "both"
    :param finetune_out_vars: [str] which parameters to fine tune on should be
    "temp", "flow" or "both"
    :param dist_type: [str] type of distance matrix ("upstream", "downstream" or
    "updown")
    :param test_start_date: the date to start for the test period
    :param n_test_yr: number of years to take for the test period
    :param exclude_segs: [dict] which (if any) segments to exclude from loss
    calculation and the start (and optionally end date) to exclude
    :returns: training and testing data along with the means and standard
    deviations of the training input and output data
            'x_trn': batched, input data for the training period scaled and
                     centred using the std and mean from entire period of record
                     [n_samples, seq_len, n_feat]
            'x_tst': un-batched input data for the test period scaled and
                     centered using the std and mean from entire period of
                     record
            'x_trn_pre': batched, scaled, centered input data for entire period
                         of record of SNTemp
            'y_trn_pre': batched, scaled, and centered output data for entire
                         period of record of SNTemp [n_samples, seq_len, n_out]
            'y_obs_trn': batched, scaled, and centered output observation data
                         for the training period
            'y_trn_obs_std': standard deviation of the y observations training
                             data [n_out]
            'y_trn_obs_mean': mean of the observation training data [n_out]
            'y_obs_tst': un-batched, unscaled, uncentered observation data for
                         the test period [n_yrs, n_seg, len_seq, n_out]
            'dates_ids_trn: batched dates and national seg ids for training data
                            [n_samples, seq_len, 2]
            'dates_ids_tst: un-batched dates and national seg ids for testing
                            data [n_yrs, n_seg, len_seq, 2]
    """
    if subset:
        pretrain_file = f'{data_dir}uncal_sntemp_input_output_subset.feather'
        obs_files = [f'{data_dir}obs_temp_subset.csv',
                     f'{data_dir}obs_flow_subset.csv']
    else:
        pretrain_file = f'{data_dir}uncal_sntemp_input_output.feather'
        obs_files = [f'{data_dir}obs_temp_full.csv',
                     f'{data_dir}obs_flow_full.csv']

    # read, y_pretrain
    ds_pre = read_format_data(pretrain_file)
    # set seg_shade to all zeros b/c there are some nan in full sntemp io
    # note: in the sntemp_input_output data, seg_shade is always zero except
    # when it's 'nan'
    ds_pre.seg_shade.loc[:, :] = 0

    # read, filter observations for finetuning
    ds_y_obs = read_multiple_obs(obs_files, ds_pre)

    # separate trn_tst for fine-tuning;
    pt_train, pt_test = separate_trn_tst(ds_pre, test_start_date, n_test_yr)
    y_obs_train, y_obs_test = separate_trn_tst(ds_y_obs, test_start_date,
                                               n_test_yr)
    # dates_ids_trn, dates_ids_tst = separate_trn_tst(dates_ids, trn_ratio)

    # separate x, y
    x, y_pre = sep_x_y(ds_pre)
    x_trn, _ = sep_x_y(pt_train)
    x_tst, _ = sep_x_y(pt_test)

    # filter pretrain/finetune y
    y_pre_weights = create_weight_vectors(y_pre, pretrain_out_vars,
                                          exclude_segs)
    y_obs_weights = create_weight_vectors(y_obs_train, finetune_out_vars,
                                          exclude_segs)

    # scale on all x data
    x_scl, x_std, x_mean = scale(x)
    x_trn_scl, _, _ = scale(x_trn, std=x_std, mean=x_mean)
    x_tst_scl, _, _ = scale(x_tst, std=x_std, mean=x_mean)

    # scale y training data and get the mean and std
    print(y_obs_train)
    y_trn_obs_scl, y_trn_obs_std, y_trn_obs_mean = scale(y_obs_train)
    print(y_trn_obs_std, y_trn_obs_mean)
    # for pre-training, keep everything together
    y_trn_pre_scl, _, _ = scale(y_pre)

    data = {'x_trn': convert_batch_reshape(x_trn_scl),
            'x_tst': convert_batch_reshape(x_tst_scl),
            'x_trn_pre': convert_batch_reshape(x_scl),
            'y_trn_pre': convert_batch_reshape(y_trn_pre_scl),
            'y_obs_trn': convert_batch_reshape(y_trn_obs_scl),
            'y_trn_obs_std': y_trn_obs_std.to_array().values,
            'y_trn_obs_mean': y_trn_obs_mean.to_array().values,
            'y_pre_wgts': convert_batch_reshape(y_pre_weights),
            'y_obs_wgts': convert_batch_reshape(y_obs_weights),
            'y_obs_tst': convert_batch_reshape(y_obs_test),
            'ids_trn': coord_as_reshaped_array(x_trn, 'seg_id_nat'),
            'dates_trn': coord_as_reshaped_array(x_trn, 'date'),
            'ids_tst': coord_as_reshaped_array(x_tst, 'seg_id_nat'),
            'dates_tst': coord_as_reshaped_array(x_tst, 'date'),
            'dist_matrix': process_adj_matrix(data_dir, dist_type, subset)
            }
    return data


def sort_dist_matrix(mat, row_col_names):
    """
    sort the distance matrix by seg_id_nat
    :return:
    """
    df = pd.DataFrame(mat, columns=row_col_names, index=row_col_names)
    df = df.sort_index(axis=0)
    df = df.sort_index(axis=1)
    return df


def process_adj_matrix(data_dir, dist_type, subset=True):
    """
    process adj matrix.
    **The matrix is sorted by seg_id_nat **
    :param dist_type: [str] type of distance matrix ("upstream", "downstream" or
    "updown")
    :param subset: [bool] whether you want data for the subsection (True) or
    for the entire DRB (False)
    :return: [numpy array] processed adjacency matrix
    """
    if subset:
        data_file = f'{data_dir}distance_matrix_subset.npz'
    else:
        data_file = f'{data_dir}distance_matrix.npz'
    adj_matrices = np.load(data_file)
    adj = adj_matrices[dist_type]
    adj = sort_dist_matrix(adj, adj_matrices['rowcolnames'])
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
    return A_hat


def read_exclude_segs_file(exclude_file):
    """
    read the exclude segs file. should be a yml file with start_date and list of
    segments to exclude
    :param exclude_file: [str] exclude segs file
    :return: [list] list of segments to exclude
    """
    with open('data/in/exclude.yml', 'r') as s:
        d = yaml.safe_load(s)
    return [val for key, val in d.items()]
