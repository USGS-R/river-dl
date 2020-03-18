# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:38:50 2019

@author: xiaoweijia
"""

from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import random
import datetime
start_time = datetime.datetime.now()

''' Declare constants '''
learning_rate = 0.01
learning_rate_pre = 0.005
epochs = 100
epochs_pre = 3#70
batch_size = 2000
hidden_size = 20 
input_size = 20-2
phy_size = 2
T = 13149
cv_idx = 2 # 0,1,2 for cross validation, 4383 lenth for each one
npic = 12
n_steps = int(4383/npic) # cut it to 16 pieces #43 #12 #46 
n_classes = 1 
N_sec = (npic-1)*2+1
N_seg = 42
kb=1.0

''' Build Graph '''
tf.reset_default_graph()
random.seed(9001)
# Graph input/output
x = tf.placeholder("float", [None, n_steps,input_size]) #tf.float32
p = tf.placeholder("float", [None, n_steps,phy_size])
y = tf.placeholder("float", [None, n_steps]) #tf.int32
m = tf.placeholder("float", [None, n_steps])
A = tf.placeholder("float", [N_seg,N_seg])

keep_prob = tf.placeholder(tf.float32)


W2 = tf.get_variable('W_2',[hidden_size, n_classes], tf.float32,
                                 tf.random_normal_initializer(stddev=0.02))
b2 = tf.get_variable('b_2',[n_classes],  tf.float32,
                                 initializer=tf.constant_initializer(0.0))

Wg1 = tf.get_variable('W_g1',[hidden_size, hidden_size], tf.float32,
                                 tf.random_normal_initializer(stddev=0.02))
bg1 = tf.get_variable('b_g1',[hidden_size],  tf.float32,
                                 initializer=tf.constant_initializer(0.0))
Wg2 = tf.get_variable('W_g2',[hidden_size, hidden_size], tf.float32,
                                 tf.random_normal_initializer(stddev=0.02))
bg2 = tf.get_variable('b_g2',[hidden_size],  tf.float32,
                                 initializer=tf.constant_initializer(0.0))


Wa1 = tf.get_variable('W_a1',[hidden_size, hidden_size], tf.float32,
                                 tf.random_normal_initializer(stddev=0.02))
Wa2 = tf.get_variable('W_a2',[hidden_size, hidden_size], tf.float32,
                                 tf.random_normal_initializer(stddev=0.02))
ba = tf.get_variable('b_a',[hidden_size],  tf.float32,
                                 initializer=tf.constant_initializer(0.0))

Wc1 = tf.get_variable('W_c1',[hidden_size, hidden_size], tf.float32,
                                 tf.random_normal_initializer(stddev=0.02))
Wc2 = tf.get_variable('W_c2',[hidden_size, hidden_size], tf.float32,
                                 tf.random_normal_initializer(stddev=0.02))
bc = tf.get_variable('b_c',[hidden_size],  tf.float32,
                                 initializer=tf.constant_initializer(0.0))


Wp = tf.get_variable('W_p',[hidden_size, phy_size], tf.float32,
                                 tf.random_normal_initializer(stddev=0.02))
bp = tf.get_variable('b_p',[phy_size],  tf.float32,
                                 initializer=tf.constant_initializer(0.0))



lstm_cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=1.0) 
#tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0) 
#lstm_cell = [tf.compat.v1.nn.rnn_cell.LSTMCell(size) for size in [128, 256]]


o_sr = []
#o_hr = []
#o_gr = []
o_phy = []

#output_series, current_state = tf.nn.dynamic_rnn(lstm_cell, tf.expand_dims(x[:,0,:],axis=1), dtype=tf.float32) #46*(500*7)
#output_series, current_state = lstm_cell(x[:,0,:], lstm_cell.get_initial_state(inputs=x[:,0,:])) #46*(500*7)
output_series, current_state = lstm_cell(x[:,0,:], lstm_cell.zero_state(N_seg, dtype=tf.float32)) #46*(500*7)



c_pre,h_pre = current_state
state_graph = tf.contrib.rnn.LSTMStateTuple(c_pre,h_pre)
#h_gr = tf.matmul(A,tf.matmul(h,Wg))
## whether transform the state as well
#c_gr = tf.matmul(A,tf.matmul(c,Wg))
#state_graph = current_state #tf.contrib.rnn.LSTMStateTuple(c_gr,h_gr)
o_sr.append(h_pre)
#o_hr.append(h_pre)
#o_gr.append(h_pre)
#o_sr.append(h_gr)
o_phy.append(tf.matmul(h_pre,Wp)+bp)
        
#with tf.variable_scope("rnn", reuse=tf.AUTO_REUSE):
for t in range(1,n_steps):
#    output_series, current_state = tf.nn.dynamic_rnn(lstm_cell, tf.expand_dims(x[:,t,:],axis=1), initial_state=state_graph, dtype=tf.float32) #46*(500*7)
    output_series, current_state = lstm_cell(x[:,t,:], state_graph) #46*(500*7)
  
    c,h = current_state
    h_gr = tf.nn.tanh(tf.matmul(A,tf.matmul(h_pre,Wg1)+bg1))
    # whether transform the state as well
    c_gr = tf.nn.tanh(tf.matmul(A,tf.matmul(c_pre,Wg2)+bg2))
    
    
    h_pre = tf.nn.sigmoid(tf.matmul(h,Wa1)+tf.matmul(h_gr,Wa2)+ba)
    c_pre = tf.nn.sigmoid(tf.matmul(c,Wc1)+tf.matmul(c_gr,Wc2)+bc)
    state_graph = tf.contrib.rnn.LSTMStateTuple(c_pre,h_pre)
    o_sr.append(h_pre)
#    o_hr.append(h)
#    o_gr.append(h_gr)
    o_phy.append(tf.matmul(h_pre,Wp)+bp)

o_sr = tf.stack(o_sr,axis=1) # N_seg - T - hidden_size
oh = tf.reshape(o_sr,[-1,hidden_size])
o_phy = tf.stack(o_phy,axis=1)
o_phy = o_phy[:,:,:phy_size]  # I don't think I need this line. It doesn't change anything
#o_phy = tf.reshape(o_phy,[-1,phy_size])
o_phy = tf.reshape(o_phy,[-1])


#cost_phy = tf.sqrt(tf.reduce_sum(tf.square(tf.multiply((o_phy-tf.reshape(p,[-1])),tf.reshape(m,[-1]))))/(tf.reduce_sum(tf.reshape(m,[-1]))+1))
cost_phy = tf.sqrt(tf.reduce_sum(tf.square(o_phy-tf.reshape(p,[-1])))/(phy_size*N_seg*n_steps))



oh = tf.nn.dropout(oh,keep_prob)

y_prd = tf.matmul(oh,W2)+b2

#  cost function
y_prd_fin = tf.reshape(y_prd,[-1,n_steps])
y_prd = tf.reshape(y_prd,[-1])
cost_sup = tf.sqrt(tf.reduce_sum(tf.square(tf.multiply((y_prd-tf.reshape(y,[-1])),tf.reshape(m,[-1]))))/(tf.reduce_sum(tf.reshape(m,[-1]))+1))

#cost = cost_sup #+ 1.0*cost_phy

tvars = tf.trainable_variables()
for i in tvars:
    print(i)
saver = tf.train.Saver(max_to_keep=3)

optimizer_pre = tf.train.AdamOptimizer(learning_rate_pre)
#grads = tf.gradients(cost, tvars)
gvs_pre = optimizer_pre.compute_gradients(cost_phy)
#capped_gvs_pre = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs_pre]
train_op_pre = optimizer_pre.apply_gradients(gvs_pre)

optimizer = tf.train.AdamOptimizer(learning_rate)
gvs = optimizer.compute_gradients(cost_sup)
#capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
#optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.apply_gradients(gvs)
#train_op = optimizer.apply_gradients(zip(grads, tvars))


''' Load data '''
feat = np.load('data/processed_features.npy')
label = np.load('data/sim_temp.npy') #np.load('obs_temp.npy')
obs = np.load('data/obs_temp.npy') #np.load('obs_temp.npy')
mask = (label!=-11).astype(int)
maso = (obs!=-11).astype(int)

#seg_test = np.load('sel_test_id.npy')

# seg_test = np.load('sel_seg_hard.npy')
flow = np.load('data/sim_flow.npy')
phy = np.concatenate([np.expand_dims(label,2),np.expand_dims(flow,2)],axis=2)
phy = np.reshape(phy,[-1,phy_size])

from sklearn import preprocessing
phy = preprocessing.scale(phy)
phy = np.reshape(phy,[N_seg,-1,phy_size])

#phy = feat[:,:,[9,10]] # 42-13149-2

#phy = np.reshape(phy,[-1,phy_size])
#for i in range(phy_size):
#    phy[:,i]-=np.min(phy[:,i])
#    phy[:,i]/=np.max(phy[:,i])
#phy = np.reshape(phy,[N_seg,-1,phy_size])


feat = np.delete(feat,[9,10],2)

#adj = np.load('up_dist.npy') #
adj_up = np.load('data/up_full.npy')
adj_dn = np.load('data/dn_full.npy')
#adj_up = np.load('up_full.npy') 
#adj_dn = np.load('dn_full.npy')
adj = adj_up#+adj_dn#adj_up #adj_up+adj_dn
#adj/=5000
#adj[adj!=0] = 1/adj[adj!=0]
adj = -adj
mean_adj = np.mean(adj[adj!=0])
std_adj = np.std(adj[adj!=0])
adj[adj!=0] = adj[adj!=0]-mean_adj
adj[adj!=0] = adj[adj!=0]/std_adj
adj[adj!=0] = 1/(1+np.exp(adj[adj!=0]))

I = np.eye(adj.shape[0])
A_hat = adj.copy() + I
#D = np.sum(A_hat, axis=0)
#D_inv = D**-0.5
#D_inv = np.diag(D_inv)
#A_hat = np.matmul(np.matmul(D_inv,A_hat), D_inv)
D = np.sum(A_hat, axis=1)
D_inv = D**-1.0
D_inv = np.diag(D_inv)
A_hat = np.matmul(D_inv,A_hat)

x_te = feat[:,cv_idx*4383:(cv_idx+1)*4383,:]
y_te = label[:,cv_idx*4383:(cv_idx+1)*4383]
o_te = obs[:,cv_idx*4383:(cv_idx+1)*4383]
m_te = mask[:,cv_idx*4383:(cv_idx+1)*4383]
mo_te = maso[:,cv_idx*4383:(cv_idx+1)*4383]
p_te = phy[:,cv_idx*4383:(cv_idx+1)*4383,:]

#m_te[seg_test,:]=0.0
#m_te_seg = m_te.copy()
#m_te_seg[:,:]=0
#m_te_seg[seg_test,:]=1.0

#mo_te[seg_test,:]=0.0
#mo_te_seg = m_te.copy()
#mo_te_seg[:,:]=0
#mo_te_seg[seg_test,:]=1.0


if cv_idx==1:
    x_tr_1 = feat[:,:4383,:]
    y_tr_1 = label[:,:4383]
    o_tr_1 = obs[:,:4383]
    m_tr_1 = mask[:,:4383]
    mo_tr_1 = maso[:,:4383]
    p_tr_1 = phy[:,:4383,:]
    
    x_tr_2 = feat[:,2*4383:3*4383,:]
    y_tr_2 = label[:,2*4383:3*4383:]
    o_tr_2 = obs[:,2*4383:3*4383:]
    m_tr_2 = mask[:,2*4383:3*4383:]
    mo_tr_2 = maso[:,2*4383:3*4383:] 
    p_tr_2 = phy[:,2*4383:3*4383,:]

if cv_idx==2:
    x_tr_1 = feat[:,:4383,:]
    y_tr_1 = label[:,:4383]
    o_tr_1 = obs[:,:4383]
    m_tr_1 = mask[:,:4383]
    mo_tr_1 = maso[:,:4383]
    p_tr_1 = phy[:,:4383,:]
    
    x_tr_2 = feat[:,4383:2*4383,:]
    y_tr_2 = label[:,4383:2*4383:]
    o_tr_2 = obs[:,4383:2*4383:]
    m_tr_2 = mask[:,4383:2*4383:]
    mo_tr_2 = maso[:,4383:2*4383:]
    p_tr_2 = phy[:,4383:2*4383,:]


#s_perc = 0.02 #0.2#0.4 #0.6 #0.8
##ts = np.sum(m_tr_1[seg_test,:])+np.sum(m_tr_2[seg_test,:])
#
#indices = np.random.choice(np.arange(4383), replace=False, size=int(4383 * (1-s_perc)))
#m_tr_1[:,indices] = 0.0
#indices = np.random.choice(np.arange(4383), replace=False, size=int(4383 * (1-s_perc)))
#m_tr_2[:,indices] = 0.0
#
#ts = np.sum(m_tr_1)+np.sum(m_tr_2)
#print(ts)
    
#m_tr_1[seg_test,:]=0
#m_tr_2[seg_test,:]=0
#mo_tr_1[seg_test,:]=0
#mo_tr_2[seg_test,:]=0

x_train_1 = np.zeros([N_seg*N_sec,n_steps,input_size])
y_train_1 = np.zeros([N_seg*N_sec,n_steps])
o_train_1 = np.zeros([N_seg*N_sec,n_steps])
m_train_1 = np.zeros([N_seg*N_sec,n_steps])
mo_train_1 = np.zeros([N_seg*N_sec,n_steps])
p_train_1 = np.zeros([N_seg*N_sec,n_steps,phy_size])

x_train_2 = np.zeros([N_seg*N_sec,n_steps,input_size])
y_train_2 = np.zeros([N_seg*N_sec,n_steps])
o_train_2 = np.zeros([N_seg*N_sec,n_steps])
m_train_2 = np.zeros([N_seg*N_sec,n_steps])
mo_train_2 = np.zeros([N_seg*N_sec,n_steps])
p_train_2 = np.zeros([N_seg*N_sec,n_steps,phy_size])

x_test = np.zeros([N_seg*N_sec,n_steps,input_size])
y_test = np.zeros([N_seg*N_sec,n_steps])
o_test = np.zeros([N_seg*N_sec,n_steps])
m_test = np.zeros([N_seg*N_sec,n_steps])
mo_test = np.zeros([N_seg*N_sec,n_steps])
#mo_test_seg = np.zeros([N_seg*N_sec,n_steps])
p_test = np.zeros([N_seg*N_sec,n_steps,phy_size])

for i in range(1,N_sec+1):
    x_train_1[(i-1)*N_seg:i*N_seg,:,:]=x_tr_1[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]
    y_train_1[(i-1)*N_seg:i*N_seg,:]=y_tr_1[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    o_train_1[(i-1)*N_seg:i*N_seg,:]=o_tr_1[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    m_train_1[(i-1)*N_seg:i*N_seg,:]=m_tr_1[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    mo_train_1[(i-1)*N_seg:i*N_seg,:]=mo_tr_1[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    p_train_1[(i-1)*N_seg:i*N_seg,:,:]=p_tr_1[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]
    x_train_2[(i-1)*N_seg:i*N_seg,:,:]=x_tr_2[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]
    y_train_2[(i-1)*N_seg:i*N_seg,:]=y_tr_2[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    o_train_2[(i-1)*N_seg:i*N_seg,:]=o_tr_2[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    m_train_2[(i-1)*N_seg:i*N_seg,:]=m_tr_2[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    mo_train_2[(i-1)*N_seg:i*N_seg,:]=mo_tr_2[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    p_train_2[(i-1)*N_seg:i*N_seg,:,:]=p_tr_2[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]
    x_test[(i-1)*N_seg:i*N_seg,:,:]=x_te[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]
    y_test[(i-1)*N_seg:i*N_seg,:]=y_te[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    o_test[(i-1)*N_seg:i*N_seg,:]=o_te[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    m_test[(i-1)*N_seg:i*N_seg,:]=m_te[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    mo_test[(i-1)*N_seg:i*N_seg,:]=mo_te[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
#    mo_test_seg[(i-1)*N_seg:i*N_seg,:]=mo_te_seg[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    p_test[(i-1)*N_seg:i*N_seg,:,:]=p_te[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]
    
''' Session starts '''
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# pretraining
print('Pretraining starts')
print('==================================')
for epoch in range(epochs_pre): #range(epochs):

    alos = 0
    alos_s = 0
    alos_p = 0
    
    idx = range(N_sec)
    idx = random.sample(idx,N_sec)

    alos_s = 0
    alos_p = 0
    for i in range(N_sec): # better code?
        index = range(idx[i]*N_seg, (idx[i]+1)*N_seg)
        
        batch_x = x_train_1[index,:,:]
        batch_y = y_train_1[index,:]
        batch_m = m_train_1[index,:]
        batch_p = p_train_1[index,:,:]
        
#        
        _,los_s,los_p = sess.run(
            [train_op_pre, cost_sup,cost_phy],
            feed_dict = {
                    x: batch_x,
                    y: batch_y,
                    m: batch_m,
                    keep_prob: kb,
                    A: A_hat,
                    p:batch_p
        })
        alos_s += los_s
        alos_p += los_p
    print('Epoch '+str(epoch)+': loss '+"{:.4f}".format(alos/N_sec) \
          +': loss_s '+"{:.4f}".format(alos_s/N_sec)\
          +': loss_p '+"{:.4f}".format(alos_p/N_sec) )
    

    alos_s = 0
    alos_p = 0
    for i in range(N_sec): # better code?
        index = range(idx[i]*N_seg, (idx[i]+1)*N_seg)
        
        batch_x = x_train_2[index,:,:]
        batch_y = y_train_2[index,:]
        batch_m = m_train_2[index,:]
        batch_p = p_train_2[index,:,:]
        
        _,los_s,los_p = sess.run(
            [train_op_pre, cost_sup, cost_phy],
            feed_dict = {
                    x: batch_x,
                    y: batch_y,
                    m: batch_m,
                    keep_prob: kb,
                    A: A_hat,
                    p:batch_p
        })
        alos_s += los_s
        alos_p += los_p
    print('Epoch '+str(epoch)+': loss '+"{:.4f}".format(alos/N_sec) \
          +': loss_s '+"{:.4f}".format(alos_s/N_sec)\
          +': loss_p '+"{:.4f}".format(alos_p/N_sec) )
    
    alos_s = 0
    alos_p = 0
    for i in range(N_sec): # better code?
        index = range(idx[i]*N_seg, (idx[i]+1)*N_seg)
        
        batch_x = x_test[index,:,:]
        batch_y = y_test[index,:]
        batch_m = m_test[index,:]
        batch_p = p_test[index,:,:]
        
        _,los_s,los_p = sess.run(
            [train_op_pre, cost_sup, cost_phy],
            feed_dict = {
                    x: batch_x,
                    y: batch_y,
                    m: batch_m,
                    keep_prob: kb,
                    A: A_hat,
                    p:batch_p
        })
        alos_s += los_s
        alos_p += los_p
    print('Epoch '+str(epoch)+': loss '+"{:.4f}".format(alos/N_sec) \
          +': loss_s '+"{:.4f}".format(alos_s/N_sec)\
          +': loss_p '+"{:.4f}".format(alos_p/N_sec) )

end_time = datetime.datetime.now()
print('elapsed time: ', end_time - start_time)
raise Exception('stope')

print('Fine-tuning starts')
print('==================================')
#total_batch = int(np.floor(N_tr/batch_size))
los = 0
mre = 10
pred = np.zeros(y_test.shape)
for epoch in range(epochs): #range(epochs):
    if np.isnan(los):
            break
    alos = 0
    alos_s = 0
    alos_p = 0
    
    idx = range(N_sec)
    idx = random.sample(idx,N_sec)
    
    for i in range(N_sec): # better code?
        index = range(idx[i]*N_seg, (idx[i]+1)*N_seg)
        
        batch_x = x_train_1[index,:,:]
        batch_y = o_train_1[index,:]
        batch_m = mo_train_1[index,:]
        batch_p = p_train_1[index,:,:]
        
#        # debug
#        prd,cc,hh,cg,hg,oo,oh,og = sess.run(
#            [y_prd,c,h,c_gr,h_gr,o_sr,o_hr,o_gr],
#            feed_dict = {
#                    x: batch_x,
#                    y: batch_y,
#                    m: batch_m,
#                    keep_prob: kb,
#                    A: A_hat
#        })
#        
        _,los_s,los_p = sess.run(
            [train_op, cost_sup,cost_phy],
            feed_dict = {
                    x: batch_x,
                    y: batch_y,
                    m: batch_m,
                    keep_prob: kb,
                    A: A_hat,
                    p:batch_p
        })
        alos += los
        alos_s += los_s
        alos_p += los_p
        if np.isnan(los):
            break
    print('Epoch '+str(epoch)+': loss '+"{:.4f}".format(alos/N_sec) \
          +': loss_s '+"{:.4f}".format(alos_s/N_sec)\
          +': loss_p '+"{:.4f}".format(alos_p/N_sec) )
    
    if np.isnan(los):
        break
        
    alos = 0
    alos_s = 0
    alos_p = 0
    for i in range(N_sec): # better code?
        index = range(idx[i]*N_seg, (idx[i]+1)*N_seg)
        
        batch_x = x_train_2[index,:,:]
        batch_y = o_train_2[index,:]
        batch_m = mo_train_2[index,:]
        batch_p = p_train_2[index,:,:]
#        # debug
#        prd,cc,hh,cg,hg,oo,oh,og = sess.run(
#            [y_prd,c,h,c_gr,h_gr,o_sr,o_hr,o_gr],
#            feed_dict = {
#                    x: batch_x,
#                    y: batch_y,
#                    m: batch_m,
#                    keep_prob: kb,
#                    A: A_hat
#        })
        _,los_s,los_p = sess.run(
            [train_op, cost_sup, cost_phy],
            feed_dict = {
                    x: batch_x,
                    y: batch_y,
                    m: batch_m,
                    keep_prob: kb,
                    A: A_hat,
                    p:batch_p
        })
        alos += los
        alos_s += los_s
        alos_p += los_p
        if np.isnan(los):
            break
    print('Epoch '+str(epoch)+': loss '+"{:.4f}".format(alos/N_sec) \
          +': loss_s '+"{:.4f}".format(alos_s/N_sec)\
          +': loss_p '+"{:.4f}".format(alos_p/N_sec) )

# test on segments with training samples
for i in range(N_sec): # better code?
    index = range(i*N_seg, (i+1)*N_seg)
    
    batch_x = x_test[index,:,:]
    batch_y = o_test[index,:]
    batch_m = mo_test[index,:]
    
    batch_prd = sess.run(
        [y_prd_fin],
        feed_dict = {
                x: batch_x,
                y: batch_y,
                m: batch_m,
                keep_prob: 1.0,
                A: A_hat
    })
    pred[index,:]=batch_prd
    
prd_o = np.zeros([N_seg,4383])
prd_o[:,:365] = pred[0:N_seg,:]

for j in range(N_sec-1):   # 18*125    +250 = 2500
    st_idx = 365-(int((j+1)*365/2)-int(j*365/2))
    prd_o[:, 365+int(j*365/2):365+int((j+1)*365/2)] = pred[(j+1)*N_seg:(j+2)*N_seg,st_idx:]


po = np.reshape(prd_o,[-1])
ye = np.reshape(o_te,[-1])
me = np.reshape(mo_te,[-1])
rmse = np.sqrt(np.sum(np.square((po-ye)*me))/np.sum(me))
print( 'Seg Test RMSE: '+"{:.4f}".format(rmse) )


#    # test on segments without training samples
#    me = np.reshape(mo_te_seg,[-1])
#    rmse = np.sqrt(np.sum(np.square((po-ye)*me))/np.sum(me))
#    print( 'Segwo Test RMSE: '+"{:.4f}".format(rmse) )


#print('saving...')
#np.save('./results/prd_RGCN_full_obstemp_cv'+str(cv_idx)+'.npy',prd_o)


    
    
    
    
    
    









