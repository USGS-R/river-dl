import pandas as pd
import numpy as np
import RGCN_tf2 as rgcn_tf2
from sklearn import preprocessing


def select_data(df):
    unwanted_cols = None
    # unwanted_cols = ['seg_id_nat', 'model_idx', 'date', 'seg_upstream_inflow', 'seginc_gwflow', 'seg_width'] 
    # wanted_cols=None
    wanted_cols = ['seg_tave_air', 'seg_rain']#, 'seg_upstream_inflow']
    target_col = 'seg_outflow'
    if unwanted_cols and not wanted_cols:
        predictor_cols = [c for c in df.columns if c not in unwanted_cols and
                          c != target_col]
    elif wanted_cols:
        predictor_cols = [c for c in df.columns if c in wanted_cols and
                          c != target_col]


    return df[predictor_cols], df[target_col]


def scale(df, std=None, mean=None):
    if not isinstance(std, pd.Series) and not isinstance(mean, pd.Series):
        std = df.std()
        mean = df.mean()
    scaled = (df - mean)/std
    scaled = scaled.fillna(0)
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


def separate_trn_tst(df, train_per=0.8):
    nrows = df.shape[0]
    sep_idx = int(nrows * train_per)
    df_trn = df.iloc[:sep_idx, :]
    df_tst = df.iloc[sep_idx:, :]
    return df_trn, df_tst

def read_process_data(trn_ratio=0.8):
    # read in data
    df = pd.read_feather('data/sntemp_input_output_subset.feather')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['seg_id_nat', 'date'])
    seg_id, df = get_df_for_rand_seg(df)

    # separate trn_tst
    df_trn, df_tst = separate_trn_tst(df)

    # separate the predictors from the targets
    predictors_trn, target_trn = select_data(df_trn)
    predictors_tst, target_tst = select_data(df_tst)

    # scale the data
    x_trn, x_trn_std, x_trn_mean = scale(predictors_trn)
    # add a dimension for the timesteps (just using 1)
    x_trn = np.expand_dims(x_trn, axis=1)
    y_trn, y_trn_std, y_trn_mean = scale(target_trn)

    trn = tf.data.Dataset.from_tensor_slices((x_trn, y_trn))
    trn = trn.batch(365).shuffle(365)

    x_tst = scale(predictors_tst, x_trn_std, x_trn_mean)[0]
    x_tst = np.expand_dims(x_tst, axis=1)
    y_tst = target_tst
    tst = tf.data.Dataset.from_tensor_slices((x_tst, y_tst))
    return trn, x_trn, y_trn, x_tst, y_tst, y_trn_std, y_trn_mean



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

