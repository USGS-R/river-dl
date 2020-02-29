import pandas as pd
import numpy as np
import RGCN_tf2 as rgcn_tf2
from sklearn import preprocessing


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
    sntemp_cols = ['model_idx', 'date', 'seg_id_nat']
    unwanted_cols = ['seg_upstream_inflow', 'seginc_gwflow', 'seg_width'] 
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
    seg_id_groups = df.groupby('seg_id_nat')
    # this should work, but it raises an error
    # data_by_seg_id = seg_id_groups.apply(pd.DataFrame.to_numpy)
    seg_id_arrays = []
    for seg_id, seg_id_df in seg_id_groups:
        del seg_id_df['seg_id_nat']
        seg_id_arrays.append(seg_id_df.to_numpy())
    array_for_all_seg_ids = np.array(seg_id_arrays)
    return array_for_all_seg_ids


def sep_x_y(df):
    """
    filter data and separate into input and output, then convert to numpy arrays
    divided by seg_id
    :param df: [dataframe] the raw input and output data
    """
    unwanted_cols = get_unwanted_cols(df)
    target_cols = ['seg_outflow', 'seg_tave_water']
    predictor_cols = [c for c in df.columns if c not in unwanted_cols and
                      c not in target_cols]
    # add seg_id_nat here so we can df them into numpy arrays by segment
    target_cols.append('seg_id_nat')
    predictor_cols.append('seg_id_nat')
    pred_arr = convert_to_np_arr(df[predictor_cols])
    target_arr = convert_to_np_arr(df[target_cols])
    return pred_arr, target_arr


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
    if not std or not mean:
        std = np.std(all_segs, axis=0)
        mean = np.mean(all_segs, axis=0)
    # adding small number in case there is a std of zero
    scaled = (all_segs - mean)/(std + 1e-10)
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


def read_process_data(trn_ratio=0.8, seg_id=None):
    """
    read in and process data into training and testing datasets. the training 
    and testing data are scaled to have a std of 1 and a mean of zero
    :param trn_ratio: [float] ratio of training data. as pecentage (i.e., 0.8 )
    would mean that 80% of the data would be for training and the rest for test
    :param seg_id: [int] if you want just data for one segment id, this
    argument is that national segment id. Default is that all are given
    :returns: training and testing data along with the means and standard
    deviations of the training input and output data
    """
    # read in data
    df = pd.read_feather('data/sntemp_input_output_subset.feather')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date'])

    x, y = sep_x_y(df)

    # separate trn_tst
    x_trn, x_tst = separate_trn_tst(x, trn_ratio)
    y_trn, y_tst = separate_trn_tst(y, trn_ratio)

    # scale the data
    x_trn_scl, x_trn_std, x_trn_mean = scale(x_trn)
    # add a dimension for the timesteps (just using 1)
    y_trn_scl, y_trn_std, y_trn_mean = scale(y_trn)

    x_tst_scl = scale(x_tst, x_trn_std, x_trn_mean)[0]
    data = {'x_trn': x_trn_scl,
            'x_std': x_trn_std,
            'x_mean': x_trn_mean,
            'x_tst': x_trn_scl,
            'y_trn': y_trn_scl,
            'y_std': y_trn_std,
            'y_mean': y_trn_mean,
            'y_tst': y_tst,
            }
    return data



def process_adj_matrix():
    adj_up = np.load('up_full.npy')
    adj_dn = np.load('dn_full.npy')
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
read_process_data()
feat = np.load('processed_features.npy')
label = np.load('sim_temp.npy')  # np.load('obs_temp.npy')
obs = np.load('obs_temp.npy')  # np.load('obs_temp.npy')
mask = (label != -11).astype(int)
maso = (obs != -11).astype(int)

flow = np.load('sim_flow.npy')
phy = np.concatenate([np.expand_dims(label, 2), np.expand_dims(flow, 2)],
                     axis=2)
phy = np.reshape(phy, [-1, rgcn_tf2.n_phys_vars])


phy = preprocessing.scale(phy)
phy = np.reshape(phy, [rgcn_tf2.n_seg, -1, rgcn_tf2.n_phys_vars])

# delete the features we don't need in training (temp and flow)
feat = np.delete(feat, [9, 10], 2)

