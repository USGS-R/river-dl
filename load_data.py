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
    nseg = data_arr.shape[0]
    ndates = data_arr.shape[1]
    nfeats = data_arr.shape[2]
    all_segs = np.reshape(data_arr, [nseg*ndates, nfeats])
    if not isinstance(std, np.ndarray) or not isinstance(mean, np.ndarray):
        std = np.std(all_segs, axis=0)
        mean = np.mean(all_segs, axis=0)
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
    df = df.sort_values(['date'])
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


def read_process_data(trn_ratio=0.8, batch_offset=0.5, incl_discharge=True):
    """
    read in and process data into training and testing datasets. the training 
    and testing data are scaled to have a std of 1 and a mean of zero
    :param trn_ratio: [float] ratio of training data. as pecentage (i.e., 0.8 )
    would mean that 80% of the data would be for training and the rest for test
    :param batch_offset:
    :param incl_discharge: [bool] whether or not to include discharge
    observations. if False, the mask for all discharge values will be False
    :returns: training and testing data along with the means and standard
    deviations of the training input and output data
    """
    # read, filter, separate x, y_pretrain
    df_pre = read_format_data('data/sntemp_input_output_subset.feather')
    df_pre_filt = filter_unwanted_cols(df_pre)
    x, y_pre = sep_x_y(df_pre_filt)

    # read, filter y for fine tuning
    df_y_obs = read_multiple_obs(['data/obs_temp_subset.csv',
                                   'data/obs_flow_subset.csv'], df_pre)
    df_y_obs_filt = filter_unwanted_cols(df_y_obs)
    obs_mask = df_y_obs_filt.notna().astype(int)

    # convert to numpy arrays
    x = convert_to_np_arr(x)
    y_pre = convert_to_np_arr(y_pre)
    y_obs = convert_to_np_arr(df_y_obs_filt)
    obs_mask = convert_to_np_arr(obs_mask)
    if not incl_discharge:
        obs_mask[:, :, 1] = 0

    # separate trn_tst
    x_trn, x_tst = separate_trn_tst(x, trn_ratio)
    y_trn_pre, y_tst_pre = separate_trn_tst(y_pre, trn_ratio)
    y_trn_obs, y_tst_obs = separate_trn_tst(y_obs, trn_ratio)
    msk_trn, msk_tst = separate_trn_tst(obs_mask, trn_ratio)

    # scale the data
    x_trn_scl, x_trn_std, x_trn_mean = scale(x_trn)
    x_tst_scl = scale(x_tst, x_trn_std, x_trn_mean)[0]
    y_trn_pre_scl, y_trn_pre_std, y_trn_pre_mean = scale(y_trn_pre)
    y_trn_obs_scl, y_trn_obs_std, y_trn_obs_mean = scale(y_trn_obs)

    # batch the training data
    x_trn_batch = split_into_batches(x_trn_scl, offset=batch_offset)
    y_trn_pre_batch = split_into_batches(y_trn_pre_scl, offset=batch_offset)
    y_trn_obs_batch = split_into_batches(y_trn_obs_scl, offset=batch_offset)
    msk_batch = split_into_batches(msk_trn, offset=batch_offset)

    # reshape data
    x_trn_batch = reshape_for_training(x_trn_batch)
    y_trn_pre_batch = reshape_for_training(y_trn_pre_batch)
    y_trn_obs_batch = reshape_for_training(y_trn_obs_batch)
    msk_batch = reshape_for_training(msk_batch)

    data = {'x_trn': x_trn_batch,
            'x_tst': x_tst_scl,
            'y_trn_pre': y_trn_pre_batch,
            'y_trn_pre_std': y_trn_pre_std,
            'y_trn_pre_mean': y_trn_pre_mean,
            'y_tst_pre': y_tst_pre,
            'y_trn_obs': y_trn_obs_batch,
            'y_trn_obs_std': y_trn_obs_std,
            'y_trn_obs_mean': y_trn_obs_mean,
            'y_trn_msk': msk_batch,
            'y_tst_obs': y_tst_obs,
            'y_tst_msk': msk_tst
            }
    return data


def process_adj_matrix():
    adj_up = np.load('data/up_full.npy')
    adj_dn = np.load('data/dn_full.npy')
    adj = adj_up  # +adj_dn#adj_up #adj_up+adj_dn
    # adj/=5000
    # adj[adj!=0] = 1/adj[adj!=0]
    adj = -adj
    mean_adj = np.mean(adj[adj != 0])
    std_adj = np.std(adj[adj != 0])
    adj[adj != 0] = adj[adj != 0] - mean_adj
    adj[adj != 0] = adj[adj != 0] / std_adj
    adj[adj != 0] = 1 / (1 + np.exp(adj[adj != 0]))

    I = np.eye(adj.shape[0])
    A_hat = adj.copy() + I
    # D = np.sum(A_hat, axis=0)
    # D_inv = D**-0.5
    # D_inv = np.diag(D_inv)
    # A_hat = np.matmul(np.matmul(D_inv,A_hat), D_inv)
    D = np.sum(A_hat, axis=1)
    D_inv = D ** -1.0
    D_inv = np.diag(D_inv)
    A_hat = np.matmul(D_inv, A_hat)
    return A_hat

''' Load data '''
# d = read_process_data(trn_ratio=0.667, batch_offset=1)
# feat = np.load('processed_features.npy')
# label = np.load('sim_temp.npy')  # np.load('obs_temp.npy')
# obs = np.load('obs_temp.npy')  # np.load('obs_temp.npy')
# mask = (label != -11).astype(int)
# maso = (obs != -11).astype(int)
#
# flow = np.load('sim_flow.npy')
# phy = np.concatenate([np.expand_dims(label, 2), np.expand_dims(flow, 2)],
#                      axis=2)
# phy = np.reshape(phy, [-1, rgcn_tf2.n_phys_vars])
#
#
# phy = preprocessing.scale(phy)
# phy = np.reshape(phy, [rgcn_tf2.n_seg, -1, rgcn_tf2.n_phys_vars])
#
# # delete the features we don't need in training (temp and flow)
# feat = np.delete(feat, [9, 10], 2)

