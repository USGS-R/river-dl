import numpy as np
import torch
import torch.utils.data
import pandas as pd
import time
from tqdm import tqdm


def reshape_for_gwn(cat_data, keep_portion=None):
    """
    Helper function to reshape input data for GraphWaveNet Model
    @param cat_data: dictionary or path to .npz file
    @param keep_portion: [float]  If < 1, fraction of prediction sequence to keep, if >1, absolute length to keep
    @return: [dict] reformatted data
    """
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
        if keep_portion > 1:
            period = int(keep_portion)
        else:
            seq_len = cat_reshaped['y_obs_trn'].shape[1]
            period = int(keep_portion * seq_len)
        for i in files_y:
            cat_reshaped[i] = cat_reshaped[i][:, -period:, ...]
    return cat_reshaped

## Generic PyTorch Training Routine
def train_loop(epoch_index, dataloader, model, loss_function, optimizer, device = 'cpu'):
    """
    @param epoch_index: [int] Epoch number
    @param dataloader: [object] torch dataloader with train and val data
    @param model: [object] initialized torch model
    @param loss_function: loss function
    @param optimizer: [object] Chosen optimizer
    @param device: [str] cpu or gpu
    @return: [float] epoch loss
    """
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
    """
    @param dataloader: [object] torch dataloader with train and val data
    @param model: [object] initialized torch model
    @param loss_function: loss function
    @param device: [str] cpu or gpu
    @return: [float] epoch validation loss
    """
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
                log_file= None,
                device = 'cpu'):
    """
    @param model: [objetct] initialized torch model
    @param loss_function: loss function
    @param optimizer: [object] chosen optimizer
    @param batch_size: [int]
    @param max_epochs: [maximum number of epochs to run for]
    @param early_stopping_patience: [int] number of epochs without improvement in validation loss to run before stopping training
    @param shuffle: [bool] Shuffle training batches
    @param weights_file: [str] path save trained model weights
    @param log_file: [str] path to save training log to
    @return: [object] trained model
    """

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
    log_cols = ['epoch', 'loss', 'time']
    train_log = pd.DataFrame(columns=log_cols)
    for i in range(max_epochs):

        #Train
        t1 = time.time()
        #print(f"Epoch: {i + 1}")

        model.train()
        epoch_loss = train_loop(i, train_loader, model, loss_function, optimizer, device)
        train_time.append(time.time() - t1)
        train_log = pd.concat([train_log,pd.DataFrame([[i, epoch_loss, time.time()-t1]],columns=log_cols,index=['train'])])

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
            train_log = pd.concat([train_log,pd.DataFrame([[i, epoch_val_loss, time.time()-s1]],columns=log_cols,index=['val'])])
            val_time.append(time.time()-s1)

    train_log.to_csv(log_file)
    #print(train_log)
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


def predict_torch(x_data, model, batch_size):
    """
    @param model: [object] initialized torch model
    @param batch_size: [int]
    @param device: [str] cuda or cpu
    @return: [tensor] predicted values
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
            output = model(trainx.to(device)).cpu()
        predicted.append(output)
    predicted = torch.cat(predicted, dim=0)
    return predicted
