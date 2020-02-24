import numpy as np

phy_size = 2
# the number of river segments
N_seg = 42
n_total_sample = 13149
# the number of sections to divide the total samples in
cv_divisions = 3
# the cross-validation index
cv_index = 2
# number of samples in the test set
n_samp_tst = n_total_sample/cv_divisions
# number of time steps per batch or sequence that the model will be trained on
n_step = 365


''' Load data '''
feat = np.load('processed_features.npy')
label = np.load('sim_temp.npy')  # np.load('obs_temp.npy')
obs = np.load('obs_temp.npy')  # np.load('obs_temp.npy')
mask = (label != -11).astype(int)
maso = (obs != -11).astype(int)

# seg_test = np.load('sel_test_id.npy')

# seg_test = np.load('sel_seg_hard.npy')
flow = np.load('sim_flow.npy')
phy = np.concatenate([np.expand_dims(label, 2), np.expand_dims(flow, 2)],
                     axis=2)
phy = np.reshape(phy, [-1, phy_size])

from sklearn import preprocessing

phy = preprocessing.scale(phy)
phy = np.reshape(phy, [N_seg, -1, phy_size])

# phy = feat[:,:,[9,10]] # 42-13149-2

# phy = np.reshape(phy,[-1,phy_size])
# for i in range(phy_size):
#    phy[:,i]-=np.min(phy[:,i])
#    phy[:,i]/=np.max(phy[:,i])
# phy = np.reshape(phy,[N_seg,-1,phy_size])


feat = np.delete(feat, [9, 10], 2)

# adj = np.load('up_dist.npy') #
adj_up = np.load('up_full.npy')
adj_dn = np.load('dn_full.npy')
# adj_up = np.load('up_full.npy')
# adj_dn = np.load('dn_full.npy')
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

x_te = feat[:, cv_idx * 4383:(cv_idx + 1) * 4383, :]
y_te = label[:, cv_idx * 4383:(cv_idx + 1) * 4383]
o_te = obs[:, cv_idx * 4383:(cv_idx + 1) * 4383]
m_te = mask[:, cv_idx * 4383:(cv_idx + 1) * 4383]
mo_te = maso[:, cv_idx * 4383:(cv_idx + 1) * 4383]
p_te = phy[:, cv_idx * 4383:(cv_idx + 1) * 4383, :]

# m_te[seg_test,:]=0.0
# m_te_seg = m_te.copy()
# m_te_seg[:,:]=0
# m_te_seg[seg_test,:]=1.0

# mo_te[seg_test,:]=0.0
# mo_te_seg = m_te.copy()
# mo_te_seg[:,:]=0
# mo_te_seg[seg_test,:]=1.0


if cv_idx == 1:
    x_tr_1 = feat[:, :4383, :]
    y_tr_1 = label[:, :4383]
    o_tr_1 = obs[:, :4383]
    m_tr_1 = mask[:, :4383]
    mo_tr_1 = maso[:, :4383]
    p_tr_1 = phy[:, :4383, :]

    x_tr_2 = feat[:, 2 * 4383:3 * 4383, :]
    y_tr_2 = label[:, 2 * 4383:3 * 4383:]
    o_tr_2 = obs[:, 2 * 4383:3 * 4383:]
    m_tr_2 = mask[:, 2 * 4383:3 * 4383:]
    mo_tr_2 = maso[:, 2 * 4383:3 * 4383:]
    p_tr_2 = phy[:, 2 * 4383:3 * 4383, :]

if cv_idx == 2:
    x_tr_1 = feat[:, :4383, :]
    y_tr_1 = label[:, :4383]
    o_tr_1 = obs[:, :4383]
    m_tr_1 = mask[:, :4383]
    mo_tr_1 = maso[:, :4383]
    p_tr_1 = phy[:, :4383, :]

    x_tr_2 = feat[:, 4383:2 * 4383, :]
    y_tr_2 = label[:, 4383:2 * 4383:]
    o_tr_2 = obs[:, 4383:2 * 4383:]
    m_tr_2 = mask[:, 4383:2 * 4383:]
    mo_tr_2 = maso[:, 4383:2 * 4383:]
    p_tr_2 = phy[:, 4383:2 * 4383, :]

# s_perc = 0.02 #0.2#0.4 #0.6 #0.8
##ts = np.sum(m_tr_1[seg_test,:])+np.sum(m_tr_2[seg_test,:])
#
# indices = np.random.choice(np.arange(4383), replace=False, size=int(4383 * (1-s_perc)))
# m_tr_1[:,indices] = 0.0
# indices = np.random.choice(np.arange(4383), replace=False, size=int(4383 * (1-s_perc)))
# m_tr_2[:,indices] = 0.0
#
# ts = np.sum(m_tr_1)+np.sum(m_tr_2)
# print(ts)

# m_tr_1[seg_test,:]=0
# m_tr_2[seg_test,:]=0
# mo_tr_1[seg_test,:]=0
# mo_tr_2[seg_test,:]=0

x_train_1 = np.zeros([N_seg * N_sec, n_steps, input_size])
y_train_1 = np.zeros([N_seg * N_sec, n_steps])
o_train_1 = np.zeros([N_seg * N_sec, n_steps])
m_train_1 = np.zeros([N_seg * N_sec, n_steps])
mo_train_1 = np.zeros([N_seg * N_sec, n_steps])
p_train_1 = np.zeros([N_seg * N_sec, n_steps, phy_size])

x_train_2 = np.zeros([N_seg * N_sec, n_steps, input_size])
y_train_2 = np.zeros([N_seg * N_sec, n_steps])
o_train_2 = np.zeros([N_seg * N_sec, n_steps])
m_train_2 = np.zeros([N_seg * N_sec, n_steps])
mo_train_2 = np.zeros([N_seg * N_sec, n_steps])
p_train_2 = np.zeros([N_seg * N_sec, n_steps, phy_size])

x_test = np.zeros([N_seg * N_sec, n_steps, input_size])
y_test = np.zeros([N_seg * N_sec, n_steps])
o_test = np.zeros([N_seg * N_sec, n_steps])
m_test = np.zeros([N_seg * N_sec, n_steps])
mo_test = np.zeros([N_seg * N_sec, n_steps])
# mo_test_seg = np.zeros([N_seg*N_sec,n_steps])
p_test = np.zeros([N_seg * N_sec, n_steps, phy_size])

for i in range(1, N_sec + 1):
    x_train_1[(i - 1) * N_seg:i * N_seg, :, :] = x_tr_1[:,
                                                 int((i - 1) * n_steps / 2):int(
                                                     (i + 1) * n_steps / 2), :]
    y_train_1[(i - 1) * N_seg:i * N_seg, :] = y_tr_1[:,
                                              int((i - 1) * n_steps / 2):int(
                                                  (i + 1) * n_steps / 2)]
    o_train_1[(i - 1) * N_seg:i * N_seg, :] = o_tr_1[:,
                                              int((i - 1) * n_steps / 2):int(
                                                  (i + 1) * n_steps / 2)]
    m_train_1[(i - 1) * N_seg:i * N_seg, :] = m_tr_1[:,
                                              int((i - 1) * n_steps / 2):int(
                                                  (i + 1) * n_steps / 2)]
    mo_train_1[(i - 1) * N_seg:i * N_seg, :] = mo_tr_1[:,
                                               int((i - 1) * n_steps / 2):int(
                                                   (i + 1) * n_steps / 2)]
    p_train_1[(i - 1) * N_seg:i * N_seg, :, :] = p_tr_1[:,
                                                 int((i - 1) * n_steps / 2):int(
                                                     (i + 1) * n_steps / 2), :]
    x_train_2[(i - 1) * N_seg:i * N_seg, :, :] = x_tr_2[:,
                                                 int((i - 1) * n_steps / 2):int(
                                                     (i + 1) * n_steps / 2), :]
    y_train_2[(i - 1) * N_seg:i * N_seg, :] = y_tr_2[:,
                                              int((i - 1) * n_steps / 2):int(
                                                  (i + 1) * n_steps / 2)]
    o_train_2[(i - 1) * N_seg:i * N_seg, :] = o_tr_2[:,
                                              int((i - 1) * n_steps / 2):int(
                                                  (i + 1) * n_steps / 2)]
    m_train_2[(i - 1) * N_seg:i * N_seg, :] = m_tr_2[:,
                                              int((i - 1) * n_steps / 2):int(
                                                  (i + 1) * n_steps / 2)]
    mo_train_2[(i - 1) * N_seg:i * N_seg, :] = mo_tr_2[:,
                                               int((i - 1) * n_steps / 2):int(
                                                   (i + 1) * n_steps / 2)]
    p_train_2[(i - 1) * N_seg:i * N_seg, :, :] = p_tr_2[:,
                                                 int((i - 1) * n_steps / 2):int(
                                                     (i + 1) * n_steps / 2), :]
    x_test[(i - 1) * N_seg:i * N_seg, :, :] = x_te[:,
                                              int((i - 1) * n_steps / 2):int(
                                                  (i + 1) * n_steps / 2), :]
    y_test[(i - 1) * N_seg:i * N_seg, :] = y_te[:,
                                           int((i - 1) * n_steps / 2):int(
                                               (i + 1) * n_steps / 2)]
    o_test[(i - 1) * N_seg:i * N_seg, :] = o_te[:,
                                           int((i - 1) * n_steps / 2):int(
                                               (i + 1) * n_steps / 2)]
    m_test[(i - 1) * N_seg:i * N_seg, :] = m_te[:,
                                           int((i - 1) * n_steps / 2):int(
                                               (i + 1) * n_steps / 2)]
    mo_test[(i - 1) * N_seg:i * N_seg, :] = mo_te[:,
                                            int((i - 1) * n_steps / 2):int(
                                                (i + 1) * n_steps / 2)]
    #    mo_test_seg[(i-1)*N_seg:i*N_seg,:]=mo_te_seg[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    p_test[(i - 1) * N_seg:i * N_seg, :, :] = p_te[:,
                                              int((i - 1) * n_steps / 2):int(
                                                  (i + 1) * n_steps / 2), :]
