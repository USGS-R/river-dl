import numpy as np
import torch
from river_dl.GraphWaveNet import gwnet
import torch.utils.data
import torch.optim as optim
import pandas as pd
import time
import os
from tqdm import tqdm


def reshape_for_gwn(cat_data, keep_portion=None):
    if isinstance(cat_data, str):
        cat_data = np.load(cat_data)
    n_segs = len(np.unique(cat_data['ids_trn']))

    files = ['x_pre_full',
             'x_trn',
             'x_val',
             'x_tst',
             'ids_trn',
             'times_trn',
             'ids_val',
             'times_val',
             'ids_tst',
             'times_tst',
             'y_obs_trn',
             'y_obs_wgts',
             'y_obs_val',
             'y_obs_tst',
             'y_pre_full',
             'y_pre_trn']

    cat_reshaped = {}
    for i in files:
        shapes_rdl = cat_data[i].shape
        reshaped = cat_data[i].reshape(int(shapes_rdl[0] / n_segs), n_segs, shapes_rdl[1], shapes_rdl[2])
        reshaped = np.moveaxis(reshaped,3,1)
        #reshaped = np.transpose(reshaped,(0,3,2,1))
        cat_reshaped[i] = reshaped

    for file in set(cat_data.files) - set(files):
        cat_reshaped[file] = cat_data[file]

    files_y = list(filter(lambda x: 'y_' in x, files))
    if keep_portion:
        for i in files_y:
            cat_reshaped[i] = cat_reshaped[i][:, -keep_portion:, ...]
    return cat_reshaped

## Generic PyTorch Training Routine
def train_loop(epoch_index, dataloader, model, loss_function, optimizer, device = 'cpu'):
    train_loss=[]
    with tqdm(dataloader, ncols=100, desc= f"Epoch {epoch_index+1}", unit="batch") as tepoch:
        for x, y in tepoch: #enumerate(dataloader):
            trainx = x.to(device)
            trainy = y.to(device)
            optimizer.zero_grad()
            output = model(trainx)
            loss = loss_function(trainy, output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            train_loss.append(loss.item())
            tepoch.set_postfix(loss=loss.item())
    mean_loss = np.mean(train_loss)
    return mean_loss

def val_loop(dataloader, model, loss_function, device = 'cpu'):
    val_loss = []
    for iter, (x, y) in enumerate(dataloader):
        testx = x.to(device)
        testy = y.to(device)
        output = model(testx)
        loss = loss_function(testy, output)
        val_loss.append(loss.item())
    mval_loss = np.mean(val_loss)
    print(f"Valid loss: {mval_loss:.2f}")
    return mval_loss


def train_torch(model,
                loss_function,
                optimizer,
                x_train,
                y_train,
                batch_size,
                max_epochs,
                early_stopping_patience,
                x_val = None,
                y_val = None,
                shuffle = False,
                weights_file = None,
                log_file= None):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f"Training on {device}")
    print("start training...",flush=True)

    epochs_since_best = 0
    best_loss = 1000 # Will get overwritten

    # Put together dataloaders
    train_data = []
    for i in range(len(x_train)):
        train_data.append([torch.from_numpy(x_train[i]).float(),
                           torch.from_numpy(y_train[i]).float()])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

    if x_val is not None:
        val_data = []
        for i in range(len(x_val)):
            val_data.append([torch.from_numpy(x_val[i]).float(),
                            torch.from_numpy(y_val[i]).float()])

        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

    val_time = []
    train_time = []

    model.to(device)
    ### Run training loop
    train_log = pd.DataFrame(columns=['split', 'epoch', 'loss', 'time'])
    for i in range(max_epochs):

        #Train
        t1 = time.time()
        #print(f"Epoch: {i + 1}")

        model.train()
        epoch_loss = train_loop(i, train_loader, model, loss_function, optimizer, device)
        train_time.append(time.time() - t1)
        train_log.append({'split':'train','epoch':i,'loss':epoch_loss,'time':time.time()-t1}, ignore_index=True)

        #Val
        if x_val is not None:
            s1 = time.time()
            model.eval()
            epoch_val_loss = val_loop(val_loader, model, loss_function, device)

            if epoch_val_loss < best_loss:
                torch.save(model.state_dict(), weights_file)
                best_loss = epoch_val_loss
                epochs_since_best = 0
            else:
                epochs_since_best += 1
            if epochs_since_best > early_stopping_patience:
                print(f"Early Stopping at Epoch {i}")
                break
            train_log.append({'split': 'val', 'epoch': i, 'loss': epoch_val_loss, 'time': time.time() - s1},
                             ignore_index=True)
            val_time.append(time.time()-s1)

    train_log.to_csv(log_file, index=False)
    if x_val is None:
        torch.save(model.state_dict(), weights_file)
        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    else:
        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        print("Average Validation (Inference) Time: {:.4f} secs/epoch".format(np.mean(val_time)))
    return model


def rmse_masked(y_true, y_pred):
    num_y_true = torch.count_nonzero(~torch.isnan(y_true))
    if num_y_true > 0:
        zero_or_error = torch.where(
            torch.isnan(y_true), torch.zeros_like(y_true), y_pred - y_true
        )
        sum_squared_errors = torch.sum(torch.square(zero_or_error))
        rmse_loss = torch.sqrt(sum_squared_errors / num_y_true)
    else:
        rmse_loss = 0.0
    return rmse_loss


def predict_torch(x_data, model, batch_size, device='cpu'):
    data = []
    for i in range(len(x_data)):
        data.append(torch.from_numpy(x_data[i]).float())

    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, pin_memory=True)
    model.to(device)
    model.eval()
    predicted = []
    for iter, x in enumerate(dataloader):
        trainx = x.to(device)
        with torch.no_grad():
            output = model(trainx)
        predicted.append(output)
    predicted = torch.cat(predicted, dim=0)
    return predicted



'''

device = 'cpu'
data = np.load('../river-dl/output_test/0.75_180/prepped.npz')
data = reshape_for_gwn(data)

adj_mx = data['dist_matrix']

supports = [torch.tensor(adj_mx).to(device).float()]
in_dim = len(data['x_vars'])
out_dim = data['y_obs_trn'].shape[3]
num_nodes = adj_mx.shape[0]
lrate = 0.0001
wdecay = 0.0001

model = gwnet('cpu',num_nodes, supports=supports, aptinit=supports[0], in_dim=in_dim, out_dim = out_dim, layers=5,kernel_size=7,blocks=1)
opt = optim.Adam(model.parameters(), lr=lrate, weight_decay=wdecay)
scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda epoch: 0.97 ** epoch)
lfunc = rmse


trained = train_torch(model,
                      lfunc,
                      opt,
                      data['x_trn'][:10,...],
                      data['y_obs_trn'][:10,...],
                      5,
                      2,
                      20,
                      data['x_val'][:10,...],
                      data['y_obs_val'][:10,...])

train_data = []
for i in range(len(data['x_trn'])):
    train_data.append([torch.from_numpy(data['x_trn'][i]).float(),
                       torch.from_numpy(data['y_obs_trn'][i]).float()])





### Cyclic learning rate:
## https://arxiv.org/abs/1506.01186


model = gwnet('cpu',num_nodes, supports=supports, aptinit=supports[0], in_dim=in_dim, out_dim = out_dim, layers=5,kernel_size=7,blocks=1)
opt = optim.Adam(model.parameters(), lr=lrate, weight_decay=0)
scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda epoch: 0.97 ** epoch)
lfunc = rmse

train_loader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=True, pin_memory=True)

def train_loop_lr(dataloader, model, loss_function, optimizer, device = 'cpu'):
    train_loss=[]
    lr = 0.000001
    lr_log = []
    #with tqdm(dataloader, unit="batch") as tepoch:
    for iter, (x, y) in enumerate(dataloader):
        print(f"lr: {lr:.5f}")
        trainx = x.to(device)
        trainy = y.to(device)
        optimizer.zero_grad()
        output = model(trainx)
        loss = loss_function(output, trainy)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
        optimizer.step()
        train_loss.append(loss.item())
        for g in optimizer.param_groups:
            g['lr'] = lr*1.5
        lr *= 1.5
        lr_log.append(lr)
        print(f"loss = {loss.item():.4f}")
        if loss.item() > 50:
            break
    return lr_log, train_loss


lr_log, loss_log = train_loop_lr(train_loader,model, lfunc, opt)
iter = np.arange(len(lr_log))
diff = np.gradient(loss_log)
import matplotlib.pyplot as plt

plt.plot(np.log10(lr_log),loss_log)
plt.ylim(0,2)
plt.show()

plt.plot(lr_log, diff)
plt.ylim(-.2,.2)
plt.xscale('log')
plt.show()

lr_log[np.argmin(diff)]
'''