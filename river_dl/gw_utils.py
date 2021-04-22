import pandas as pd
import numpy as np
import xarray as xr
from sklearn.linear_model import LinearRegression
from datetime import datetime
import math
from itertools import compress
import matplotlib.pyplot as plt

from river_dl.preproc_utils import separate_trn_tst, read_multiple_obs
from river_dl.postproc_utils import calc_metrics

def amp_phi (Date, temp):
    """
    calculate the annual signal properties (phase and amplitude) for a temperature times series
    :param Date: vector of dates
    :param temp: vector of temperatures
    :returns: amplitude and phase
    """
    #convert the date to decimal data
    
    date_decimal = [float(x)/365 for x in ((Date-np.datetime64('1980-10-01'))/np.timedelta64(1, 'D'))]
    
    x = [[math.sin(2*math.pi*j),math.cos(2*math.pi*j)] for j in date_decimal]
    model = LinearRegression().fit(list(compress(x, np.isfinite(temp))),list(compress(temp, np.isfinite(temp))))
    amp = math.sqrt(model.coef_[0]**2+model.coef_[1]**2)
    phi = math.asin(model.coef_[1]/amp)
    
    return amp, phi

def annTempStats(thisData):
    """
    calculate the annual signal properties (phase and amplitude) for temperature times series
    :param thisData: [xr dataset] with time series data of air and water temp for each segment
    :returns: data frame with phase and amplitude of air and observed water temp, along with the
    phase shift and amplitude ratio for each segment
    """

    print("annTempStats started")
    
    air_amp=[]
    air_phi=[]
    water_amp_obs = []
    water_phi_obs = []
    
    #get the phase and amplitude for air and water temps for each segment
    for i in range(len(thisData['seg_id_nat'])):
        thisSeg = thisData['seg_id_nat'][i].data
        

        #get the air temp properties
        amp, phi = amp_phi(thisData['date'].values,thisData['seg_tave_air'][:,i].values)
        air_amp.append(amp)
        air_phi.append(phi)
        
        #get the water temp properties
        if np.sum(np.isfinite(thisData['seg_tave_water'][:,i].values))>365: #this requires at least 1 year of data, need to add other data requirements here
            amp, phi = amp_phi(thisData['date'].values,thisData['seg_tave_water'][:,i].values)
        else:
            amp=np.nan
            phi=np.nan
        water_amp_obs.append(amp)
        water_phi_obs.append(phi)

    Ar_obs = [water_amp_obs[x]/air_amp[x] for x in range(len(water_amp_obs))]
    delPhi_obs = [(water_phi_obs[x]-air_phi[x])*365/(2*math.pi) for x in range(len(water_amp_obs))]
    
    tempDF = pd.DataFrame({'seg_id_nat':thisData['seg_id_nat'], 'air_amp':air_amp,'air_phi':air_phi,'water_amp_obs':water_amp_obs,'water_phi_obs':water_phi_obs,'Ar_obs':Ar_obs,'delPhi_obs':delPhi_obs})
    
    return tempDF

def prep_annual_signal_data(
    obs_temper_file,
    pretrain_file,
    test_start_date="2004-09-30",
    n_test_yr=12,
    out_file=None
):
    """
    add annual air and water temp signal properties (phase and amplitude to
    the prepped dataset
    :param obs_temper_file: [str] temperature observations file (csv)
    :param pretrain_file: [str] the file with the pretraining data (SNTemp data)
    :param catch_prop_file: [str] the path to the catchment properties file. If
    left unfilled, the catchment properties will not be included as predictors
    :param test_start_date: [str] the date to start for the test period
    :param n_test_yr: [int] number of years to take for the test period
    :param exclude_file: [str] path to exclude file
    :param out_file: [str] file to where the values will be written
    :returns: phase and amplitude of air and observed water temp, along with the
    phase shift and amplitude ratio
    """
    
    
    #read in the SNTemp data
    ds_pre = xr.open_zarr(pretrain_file)
    
    #read in the observed temperature data and join to the SNTemp data
    obs = [ds_pre.sortby(["seg_id_nat","date"])]
    obs.append(xr.open_zarr(obs_temper_file).transpose())
    obs=xr.merge(obs,join="left")
    obs=obs[["seg_tave_air","temp_c"]]
    obs = obs.rename({"temp_c": "seg_tave_water"})
    
    #split into testing and training
    obs_trn, obs_tst = separate_trn_tst(obs, test_start_date, n_test_yr)
    
    print("test and train datasets created")
    
    #get the annual signal properties for the training and testing data
    GW_trn = annTempStats(obs_trn)
    GW_tst = annTempStats(obs_tst)
    
    #save the GW data
    data = {}
    data['GW_tst']=GW_tst
    data['GW_trn']=GW_trn
    data['GW_cols']=GW_trn.columns.values.astype('str')
    np.savez_compressed(out_file, **data)

def calc_amp_phi(thisData):
    segList = np.unique(thisData['seg_id_nat'])
    water_amp_preds = []
    water_phi_preds = []
    for thisSeg in segList:
        amp, phi = amp_phi(thisData.loc[thisData.seg_id_nat==thisSeg,"date"].values,thisData.loc[thisData.seg_id_nat==thisSeg,"seg_tave_water"])
        water_amp_preds.append(amp)
        water_phi_preds.append(phi)
    return pd.DataFrame({'seg_id_nat':segList,'water_amp_pred':water_amp_preds,'water_phi_pred':water_phi_preds})
    
def merge_pred_obs(gw_obs,obs_col,pred):
    obsDF = pd.DataFrame(gw_obs[obs_col],columns=gw_obs['GW_cols'])
    obsDF = obsDF.merge(pred)
    obsDF['Ar_pred']=obsDF['water_amp_pred']/obsDF['air_amp']
    obsDF['delPhi_pred'] = (obsDF['water_phi_pred']-obsDF['air_phi'])*365/(2*math.pi)
    return obsDF
    
def calc_pred_ann_temp(GW_data,trn_data,tst_data, trn_output, tst_output):
    gw_obs = np.load(GW_data)
    
    trn_preds = pd.read_feather(trn_data)
    tst_preds = pd.read_feather(tst_data)
    
    gw_trn = calc_amp_phi(trn_preds)
    gw_tst = calc_amp_phi(tst_preds)
    
    gw_stats_trn = merge_pred_obs(gw_obs,'GW_trn',gw_trn)
    gw_stats_tst = merge_pred_obs(gw_obs,'GW_tst',gw_tst)
                       
    gw_stats_trn.to_csv(trn_output)
    gw_stats_tst.to_csv(tst_output)
    
def calc_gw_metrics(trnFile,tstFile,outFile,figFile):
    trnDF = pd.read_csv(trnFile)
    tstDF = pd.read_csv(tstFile)
    
    resultsDF = 0
    for i in range(2):
        if i==0:
            thisData=trnDF
            partition="trn"
        elif i==1:
            thisData = tstDF
            partition="tst"
        for thisVar in ['Ar','delPhi']:
            print(thisVar)
            tempDF = pd.DataFrame(calc_metrics(thisData[["{}_obs".format(thisVar),"{}_pred".format(thisVar)]].rename(columns={"{}_obs".format(thisVar):"obs","{}_pred".format(thisVar):"pred"}))).T
            tempDF['variable']=thisVar
            tempDF['partition']=partition

            if type(resultsDF)==int:
                resultsDF = tempDF
            else:
                resultsDF = resultsDF.append(tempDF,ignore_index=True)
                
    resultsDF.to_csv(outFile,header=True, index=False)
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(2, 2, 1, aspect='equal')
    ax.set_title('Ar, Training')
    ax.axline((np.nanmean(trnDF.Ar_pred),np.nanmean(trnDF.Ar_pred)), slope=1.0,linewidth=1, color='r', label="1 to 1 line")
    ax.scatter(x=trnDF.Ar_obs,y=trnDF.Ar_pred)
    ax.legend()
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")

    ax = fig.add_subplot(2, 2, 2, aspect='equal')
    ax.set_title('delta Phi, Training')
    ax.axline((np.nanmean(trnDF.delPhi_pred),np.nanmean(trnDF.delPhi_pred)), slope=1.0,linewidth=1, color='r')
    ax.scatter(x=trnDF.delPhi_obs,y=trnDF.delPhi_pred)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")

    ax = fig.add_subplot(2, 2, 3, aspect='equal')
    ax.set_title('Ar, Testing')
    ax.axline((np.nanmean(tstDF.Ar_pred),np.nanmean(tstDF.Ar_pred)), slope=1.0,linewidth=1, color='r', label="1 to 1 line")
    ax.scatter(x=tstDF.Ar_obs,y=tstDF.Ar_pred)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")

    ax = fig.add_subplot(2, 2, 4, aspect='equal')
    ax.set_title('delta Phi, Testing')
    ax.axline((np.nanmean(tstDF.delPhi_pred),np.nanmean(tstDF.delPhi_pred)), slope=1.0,linewidth=1, color='r')
    ax.scatter(x=tstDF.delPhi_obs,y=tstDF.delPhi_pred)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")

    plt.savefig(figFile)