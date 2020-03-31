import pandas as pd
import numpy as np


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
    non_varying_cols = get_non_varying_cols(df)
    sntemp_cols = ['model_idx', 'date']
    # unwanted_cols = ['seg_upstream_inflow', 'seginc_gwflow', 'seg_width']
    # first lets just try taking the flow and temp out since that is what
    # xiaowei did 
    unwanted_cols = []
    non_varying_cols = []
    unwanted_cols.extend(non_varying_cols)
    unwanted_cols.extend(sntemp_cols)
    return unwanted_cols


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


def sep_x_y(df):
    """
    separate into input and output
    :param df: [dataframe] the raw input and output data
    :return: [tuple] df of predictors (x), df of targets (y)
    """
    target_cols = ['seg_outflow', 'seg_tave_water']
    predictor_cols = [c for c in df.columns if c not in target_cols]
    return df[predictor_cols], df[target_cols]


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
    nseg, ndates, nfeats = data_arr.shape
    all_segs = np.reshape(data_arr, [nseg*ndates, nfeats])
    if not isinstance(std, np.ndarray) or not isinstance(mean, np.ndarray):
        std = np.nanstd(all_segs, axis=0)
        mean = np.nanmean(all_segs, axis=0)
    # adding small number in case there is a std of zero
    scaled = (all_segs - mean)/(std + 1e-10)
    scaled = np.reshape(scaled, [nseg, ndates, nfeats])
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


def separate_trn_tst(data_arr, trn_ratio=0.8):
    """
    separate the train data from the test data according to the trn_ratio along
    the date axis
    :param data_arr: [numpy array] input or output data with dims
    [nseg, ndates, nfeat]
    :param trn_ratio: the amount of data to take for training. it will be taken
    from the beginning of the dataset
    """
    sep_idx = int(data_arr.shape[1] * trn_ratio)
    trn = data_arr[:, :sep_idx, :]
    tst = data_arr[:, sep_idx:, :]
    return trn, tst


def split_into_batches(data_array, batch_size=365, offset=0.5):
    """
    split training data into batches with size of batch_size
    :param data_array: [numpy array] array of training data with dims [nseg,
    ndates, nfeat]
    :param batch_size: [int] number of batches
    :param offset: [float] 0-1, how to offset the batches (e.g., 0.5 means that
    the first batch will be 0-365 and the second will be 182-547)
    :return: [numpy array] batched data with dims [nbatches, nseg, ndates
    (batch_size), nfeat]
    """
    combined = []
    for i in range(int(1/offset)):
        start = int(i * offset * batch_size)
        idx = np.arange(start=start, stop=data_array.shape[1],
                        step=batch_size)
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
    else:
        raise ValueError('file_format should be "feather" or "csv"')
    df['date'] = pd.to_datetime(df['date'])
    df['seg_id_nat'] = pd.to_numeric(df['seg_id_nat'])
    df = df.sort_values(['date', 'seg_id_nat'])
    df = df.set_index('seg_id_nat')
    return df


def read_multiple_obs(obs_files, x_data):
    """
    read and format multiple observation files
    :param obs_files: [list] list of filenames of observation files
    :param x_data:
    :return:
    """
    obs = []
    for filename in obs_files:
        df = read_format_obs(filename, x_data)
        df.set_index([df.index, 'date'], inplace=True)
        obs.append(df)
    obs = pd.concat(obs, axis=1)
    obs.reset_index(level=1, inplace=True)
    return obs


def read_format_obs(obs_file: str, x_data: pd.DataFrame) -> pd.DataFrame:
    """
    format obs data so it has the same dimensions as input data
    :type obs_file: str
    :param obs_file: file from which to read the observations
    :type x_data: pandas DataFrame
    :param x_data: input data
    :rtype: pd.DataFrame
    :return: formatted observation data
    """
    df_y_obs = read_format_data(obs_file)
    df_y_obs = df_y_obs.reset_index()
    x_data = x_data.reset_index()
    x_data = x_data[['seg_id_nat', 'date']]
    merged = pd.merge(x_data, df_y_obs, how='outer', on=['date', 'seg_id_nat'])
    merged_filt = merged[(merged['date'] >= x_data.date.min()) &
                         (merged['date'] <= x_data.date.max())]
    merged_filt = merged_filt.set_index('seg_id_nat')
    return merged_filt


def reshape_for_training(data):
    """
    reshape the data for training
    :param data: training data (either x or y or mask) dims: [nbatch, nseg,
    len_seq, nfeat/nout]
    :return: reshaped data [nbatch * nseg, len_seq, nfeat/nout]
    """
    n_batch, n_seg, seq_len, n_feat = data.shape
    return np.reshape(data, [n_batch*n_seg, seq_len, n_feat])


def filter_output_var(y_data, out_cols):
    """
    filter out either flow, temperature, or neither in the pre-training and 
    finetune y data
    """
    if out_cols == "both":
        pass
    elif out_cols == 'temp':
        y_data[:, :, 1] = np.nan
    elif out_cols == 'flow':
        y_data[:, :, 0] = np.nan
    else:
        raise ValueError('out_cols needs to be "flow", "temp", or "both"')
    return y_data


def read_process_data(subset=True, trn_ratio=0.8, batch_offset=0.5,
                      pretrain_out_vars="both", finetune_out_vars="both",
                      dist_type='upstream'):
    """
    read in and process data into training and testing datasets. the training 
    and testing data are scaled to have a std of 1 and a mean of zero
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
            'y_trn_obs': batched, scaled, and centered output observation data
                         for the training period
            'y_trn_obs_std': standard deviation of the y observations training
                             data [n_out]
            'y_trn_obs_mean': mean of the observation training data [n_out]
            'y_tst_obs': un-batched, unscaled, uncentered observation data for
                         the test period [n_yrs, n_seg, len_seq, n_out]
            'dates_ids_trn: batched dates and national seg ids for training data
                            [n_samples, seq_len, 2]
            'dates_ids_tst: un-batched dates and national seg ids for testing
                            data [n_yrs, n_seg, len_seq, 2]
    """
    data_dir = 'data/in/'
    if subset:
        pretrain_file = f'{data_dir}uncal_sntemp_input_output_subset.feather'
        obs_files = [f'{data_dir}obs_temp_subset.csv',
                     f'{data_dir}obs_flow_subset.csv']
    else:
        pretrain_file = f'{data_dir}uncal_sntemp_input_output.feather'
        obs_files = [f'{data_dir}obs_temp_full.csv',
                     f'{data_dir}obs_flow_full.csv']

    # read, filter, separate x, y_pretrain
    df_pre = read_format_data(pretrain_file)
    df_pre_filt = filter_unwanted_cols(df_pre)
    x, y_pre = sep_x_y(df_pre_filt)
    df_dates_ids = df_pre[['date']]
    # have to have a seg_id column since it gets squashed in the processing code
    # have to name it something other than seg_id_nat to avoid duplicate col ...
    # names
    df_dates_ids['seg_id_nat1'] = df_dates_ids.index

    # read, filter y for finetuning
    df_y_obs = read_multiple_obs(obs_files, df_pre)
    df_y_obs_filt = filter_unwanted_cols(df_y_obs)

    # convert to numpy arrays
    x = convert_to_np_arr(x)
    y_pre = convert_to_np_arr(y_pre)
    y_obs = convert_to_np_arr(df_y_obs_filt)
    dates_ids = convert_to_np_arr(df_dates_ids)

    # separate trn_tst for fine-tuning;
    x_trn, x_tst = separate_trn_tst(x, trn_ratio)
    y_trn_obs, y_tst_obs = separate_trn_tst(y_obs, trn_ratio)
    dates_ids_trn, dates_ids_tst = separate_trn_tst(dates_ids, trn_ratio)

    # filter pretrain/finetune y
    y_pre = filter_output_var(y_pre, pretrain_out_vars)
    y_trn_obs = filter_output_var(y_trn_obs, finetune_out_vars)

    # scale on all x data
    x_scl, x_std, x_mean = scale(x)
    x_trn_scl, _, _ = scale(x_trn, std=x_std, mean=x_mean)
    x_tst_scl, _, _ = scale(x_tst, std=x_std, mean=x_mean)
    # for pre-training, keep everything together
    x_trn_pre_scl = x_scl

    # scale y training data and get the mean and std
    y_trn_obs_scl, y_trn_obs_std, y_trn_obs_mean = scale(y_trn_obs)
    # for pre-training, keep everything together
    y_trn_pre_scl, _, _ = scale(y_pre)

    # batch the data
    x_trn_batch = split_into_batches(x_trn_scl, offset=batch_offset)
    x_tst_batch = split_into_batches(x_tst_scl, offset=1)
    x_trn_pre_batch = split_into_batches(x_trn_pre_scl, offset=batch_offset)
    y_trn_pre_batch = split_into_batches(y_trn_pre_scl, offset=batch_offset)
    y_trn_obs_batch = split_into_batches(y_trn_obs_scl, offset=batch_offset)
    y_tst_batch = split_into_batches(y_tst_obs, offset=1)
    dates_ids_trn_batch = split_into_batches(dates_ids_trn, offset=batch_offset)
    dates_ids_tst_batch = split_into_batches(dates_ids_tst, offset=1)

    # reshape data
    x_trn_batch = reshape_for_training(x_trn_batch)
    x_tst_batch = reshape_for_training(x_tst_batch)
    x_trn_pre_batch = reshape_for_training(x_trn_pre_batch)
    y_trn_pre_batch = reshape_for_training(y_trn_pre_batch)
    y_trn_obs_batch = reshape_for_training(y_trn_obs_batch)
    y_tst_batch = reshape_for_training(y_tst_batch)
    dates_ids_trn_batch = reshape_for_training(dates_ids_trn_batch)
    dates_ids_tst_batch = reshape_for_training(dates_ids_tst_batch)

    data = {'x_trn': x_trn_batch,
            'x_tst': x_tst_batch,
            'x_trn_pre': x_trn_pre_batch,
            'y_trn_pre': y_trn_pre_batch,
            'y_trn_obs': y_trn_obs_batch,
            'y_trn_obs_std': y_trn_obs_std,
            'y_trn_obs_mean': y_trn_obs_mean,
            'y_tst_obs': y_tst_batch,
            'dates_ids_trn': dates_ids_trn_batch,
            'dates_ids_tst': dates_ids_tst_batch,
            'dist_matrix': process_adj_matrix(dist_type, subset)
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


def process_adj_matrix(dist_type, subset=True):
    """
    process adj matrix.
    **The matrix is sorted by seg_id_nat **
    :param dist_type: [str] type of distance matrix ("upstream", "downstream" or
    "updown")
    :param subset: [bool] whether you want data for the subsection (True) or
    for the entire DRB (False)
    :return: [numpy array] processed adjacency matrix
    """
    data_dir = "data/in/"
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
    adj[adj != 0] = 1 / (1 + np.exp(adj[adj != 0]))

    I = np.eye(adj.shape[0])
    A_hat = adj.copy() + I
    D = np.sum(A_hat, axis=1)
    D_inv = D ** -1.0
    D_inv = np.diag(D_inv)
    A_hat = np.matmul(D_inv, A_hat)
    return A_hat


def post_process(y_pred, dates_ids, y_std, y_mean):
    y_pred = np.reshape(y_pred, [y_pred.shape[0]*y_pred.shape[1],
                                 y_pred.shape[2]])
    # unscale
    y_pred = (y_pred * y_std) + y_mean

    dates_ids = np.reshape(dates_ids, [dates_ids.shape[0]*dates_ids.shape[1],
                                       dates_ids.shape[2]])
    df_preds = pd.DataFrame(y_pred, columns=['temperature', 'flow'])
    df_dates = pd.DataFrame(dates_ids, columns=['date', 'seg_id_nat'])
    df = pd.concat([df_dates, df_preds], axis=1)
    return df


# read_process_data()
# process_adj_matrix('downstream', True)