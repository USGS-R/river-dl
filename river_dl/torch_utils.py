import numpy as np
import torch
import torch.utils.data
import pandas as pd
import time
from tqdm import tqdm
import math as m

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
        ## shape in n_batch*nseg,seq_len,n_var
        ## shape out batches, n_var, n_seg, n_seq_len

    for file in set(cat_data.files) - set(files):
        cat_reshaped[file] = cat_data[file]

    files_y = list(filter(lambda x: 'y_' in x, files))
    if keep_portion is not None:
        if keep_portion > 1:
            period = int(keep_portion)
        else:
            seq_len = cat_reshaped['y_obs_trn'].shape[3]
            period = int(keep_portion * seq_len)
        for i in files_y:
            cat_reshaped[i] = cat_reshaped[i][:,:,:,-period:]
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
                early_stopping_patience=False,
                x_val = None,
                y_val = None,
                shuffle = False,
                weights_file = None,
                log_file= None,
                device = 'cpu',
                keep_portion = None):
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
    
    if not early_stopping_patience:
        early_stopping_patience = max_epochs

    epochs_since_best = 0
    best_loss = 1000 # Will get overwritten

    if keep_portion is not None:
        if keep_portion > 1:
            period = int(keep_portion)
        else:
            period = int(keep_portion * y_train.shape[1])
        y_train[:, :-period, ...] = np.nan
        if y_val is not None:
            y_val[:, :-period, ...] = np.nan


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
    log_cols = ['epoch', 'loss', 'val_loss','time','val_time']
    train_log = pd.DataFrame(columns=log_cols) 

    for i in range(max_epochs):

        #Train
        t1 = time.time()
        #print(f"Epoch: {i + 1}")

        model.train()
        epoch_loss = train_loop(i, train_loader, model, loss_function, optimizer, device)
        train_time.append(time.time() - t1)
        train_log = pd.concat([train_log,pd.DataFrame([[i, epoch_loss, np.nan,time.time()-t1,np.nan]],columns=log_cols,index=[i])])

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
            train_log.loc[train_log.epoch==i,"val_loss"]=epoch_val_loss
            train_log.loc[train_log.epoch==i,"val_time"]=time.time() - s1
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    data = []
    for i in range(len(x_data)):
        data.append(torch.from_numpy(x_data[i]).float())

    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, pin_memory=True)
    model.eval()
    predicted = []
    for iter, x in enumerate(dataloader):
        trainx = x.to(device)
        with torch.no_grad():
            output = model(trainx).detach().cpu()
        predicted.append(output)
    predicted = torch.cat(predicted, dim=0)
    return predicted


def rmse_masked_gw(loss_function_main, temp_index,temp_mean, temp_sd,gw_mean, gw_std, lambda_Ar=0, lambda_delPhi=0, lambda_Tmean = 0, num_task=2, gw_type='fft'):
    """
    calculate a weighted, masked rmse that includes the groundwater terms
    :param lamb, lamb2, lamb3: [float] (short for lambda). The factor that the auxiliary loss, Ar loss, and deltaPhi loss
    will be multiplied by before added to the main loss.
    :param temp_index: [int]. Index of temperature column (0 if temp is the primary prediction, 1 if it is the auxiliary).
    :param temp_mean: [float]. Mean temperature for unscaling the predicted temperatures.
    :param temp_sd: [float]. Standard deviation of temperature for unscalind the predicted temperatures.
    :param gw_type: [str]. Type of gw loss, either 'fft' (fourier fast transform) or 'linalg' (linear algebra)
    """

    def rmse_masked_combined_gw(data, y_pred):
        
        Ar_obs, Ar_pred, delPhi_obs, delPhi_pred,Tmean_obs,Tmean_pred = GW_loss_prep(temp_index,data, y_pred, temp_mean, temp_sd,gw_mean, gw_std, num_task, type=gw_type)
        rmse_Ar = rmse_masked(Ar_obs.float(),Ar_pred.float())
        rmse_delPhi = rmse_masked(delPhi_obs.float(),delPhi_pred.float())
        rmse_Tmean = rmse_masked(Tmean_obs.float(),Tmean_pred.float())

        
        rmse_loss = loss_function_main(data[:,:,:num_task],y_pred) + lambda_Ar*rmse_Ar +lambda_delPhi*rmse_delPhi+lambda_Tmean*rmse_Tmean

        return rmse_loss
    return rmse_masked_combined_gw

def GW_loss_prep(temp_index, data, y_pred, temp_mean, temp_sd, gw_mean, gw_std, num_task, type='fft'):
    # assumes that axis 0 of data and y_pred are the reaches and axis 1 are daily values
    # assumes the first two columns of data are the observed flow and temperature, and the remaining
    # ones (extracted here) are the data for gw analysis


    assert type=='fft', "the groundwater loss calculation method must be fft"

    y_true = data[:, :, num_task:]
    y_true_temp = data[:, :, int(temp_index):(int(temp_index) + 1)] 

    y_pred_temp = y_pred[:, :, int(temp_index):(int(temp_index) + 1)]  # extract just the predicted temperature
    # unscale the predicted temps prior to calculating the amplitude and phase
    y_pred_temp = y_pred_temp * temp_sd + temp_mean
    y_true_temp = y_true_temp * temp_sd + temp_mean

    #set temps < 1 to 1
    y_pred_temp[y_pred_temp<1]=1
    y_true_temp[y_true_temp<1]=1
    
    Ar_obs = y_true[:, 0, 0]
    delPhi_obs = y_true[:, 0, 1]
    Tmean_obs = y_true[:, 0, 2]
    if type=='fft':
        y_pred_temp = torch.squeeze(y_pred_temp)
        y_pred_mean = torch.mean(y_pred_temp, 1, keepdims=True)
        temp_demean = y_pred_temp - y_pred_mean
        fft_torch = torch.fft.rfft(temp_demean)
        Phiw = torch.angle(fft_torch)
        phiIndex = torch.argmax(torch.abs(fft_torch), 1)
        Phiw_out=Phiw[:,1]

        Aw = torch.max(torch.abs(fft_torch), 1).values / fft_torch.shape[1]  # tf.shape(fft_tf, out_type=tf.dtypes.float32)[1]

        #get the air signal properties
        y_true_air = y_true[:, :, -1]
        y_true_air_mean = torch.mean(y_true_air, 1, keepdims=True)
        air_demean = y_true_air - y_true_air_mean
        fft_torch_air = torch.fft.rfft(air_demean)
        Phia = torch.angle(fft_torch_air)

        phiIndex_air = torch.argmax(torch.abs(fft_torch_air), 1)
        Phia_out=Phia[:,1]

        Aa = torch.max(torch.abs(fft_torch_air), 1).values / fft_torch.shape[1]  # tf.shape(fft_tf_air, out_type=tf.dtypes.float32)[1]
        
        # calculate and scale predicted values
        # delPhi_pred = the difference in phase between the water temp and air temp sinusoids, in days
        delPhi_pred = (Phia_out-Phiw_out)
        delPhi_pred = (delPhi_pred * 365 / (2 * m.pi) - gw_mean[1]) / gw_std[1]
        
        # Ar_pred = the ratio of the water temp and air temp amplitudes
        Ar_pred = (Aw / Aa - gw_mean[0]) / gw_std[0]
        
    elif type=="linalg":
        x_lm = y_true[:,:,-3:-1] #extract the sin(wt) and cos(wt)

        #a tensor of the sin(wt) and cos(wt) for each reach x day, the 1's are for the intercept of the linear regression
        # T(t) = T_mean + a*sin(wt) + b*cos(wt)
        # Johnson, Z.C., Johnson, B.G., Briggs, M.A., Snyder, C.D., Hitt, N.P., and Devine, W.D., 2021, Heed the data gap: Guidelines for 
        #using incomplete datasets in annual stream temperature analyses: Ecological Indicators, v. 122, p. 107229, 
        #http://www.sciencedirect.com/science/article/pii/S1470160X20311687.

        X_mat=torch.stack((torch.ones(y_pred_temp.shape[0:2]).to(device), x_lm[:,:,0],x_lm[:,:,1]),axis=1)
        #getting the coefficients using a 3-d version of the normal equation:
        #https://cmdlinetips.com/2020/03/linear-regression-using-matrix-multiplication-in-python-using-numpy/
        #http://mlwiki.org/index.php/Normal_Equation
        X_mat_T = torch.permute(X_mat,dims=(0,2,1))
        X_mat_T_dot = torch.einsum('bij,bjk->bik',X_mat_T,X_mat)#eigensums are used instead of dot products because we want the dot products of axis 1 and 2, not 0
        X_mat_inv = torch.linalg.pinv(X_mat_T_dot)
        X_mat_inv_dot = torch.einsum('bij,bjk->bik',X_mat_inv,X_mat_T)#eigensums are used instead of dot products because we want the dot products of axis 1 and 2, not 0
        a_b = torch.einsum('bij,bik->bjk',X_mat_inv_dot.float(),y_pred_temp.float())#eigensums are used instead of dot products because we want the dot products of axis 1 and 2, not 0
        #the tensor a_b has the coefficients from the regression (reach x [[intercept],[a],[b]])
        #Aw = amplitude of the water temp sinusoid (deg C)
        #A = sqrt (a^2 + b^2)
        Aw = torch.sqrt(a_b[:,1,0]**2+a_b[:,2,0]**2)
        #Phiw = phase of the water temp sinusoid (radians)
        #Phi = atan (b/a) - in radians
        Phiw = torch.atan(a_b[:,2,0]/a_b[:,1,0])

        #calculate the air properties
        y_true_air = y_true[:, :, -1:]
        a_b_air = torch.einsum('bij,bik->bjk',X_mat_inv_dot,y_true_air)
        A_air = torch.sqrt(a_b_air[:,1,0]**2+a_b_air[:,2,0]**2)
        Phi_air = torch.atan(a_b_air[:,2,0]/a_b_air[:,1,0])

        #calculate and scale predicted values
        #delPhi_pred = the difference in phase between the water temp and air temp sinusoids, in days
        delPhi_pred = Phi_air-Phiw
        delPhi_pred = (delPhi_pred * 365 / (2 * m.pi) - gw_mean[1]) / gw_std[1]

        #Ar_pred = the ratio of the water temp and air temp amplitudes
        Ar_pred = (Aw/A_air-gw_mean[0])/gw_std[0]
        y_pred_temp = torch.squeeze(y_pred_temp)
        y_pred_mean = torch.mean(y_pred_temp, 1, keepdims=True)

    #scale the predicted mean temp
    Tmean_pred = torch.squeeze((y_pred_mean-gw_mean[2])/gw_std[2])

    return Ar_obs, Ar_pred, delPhi_obs, delPhi_pred, Tmean_obs, Tmean_pred
