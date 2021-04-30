import pandas as pd
import numpy as np
import xarray as xr
from sklearn.linear_model import LinearRegression
from datetime import datetime
import math
from itertools import compress
import matplotlib.pyplot as plt
import seaborn as sns

from river_dl.preproc_utils import separate_trn_tst, read_multiple_obs
from river_dl.postproc_utils import calc_metrics

def amp_phi (Date, temp, isWater=False):
    """
    calculate the annual signal properties (phase and amplitude) for a temperature times series
    :param Date: vector of dates
    :param temp: vector of temperatures
    :param isWater: boolean indicator if the temp data is water temps (versus air)
    :returns: amplitude and phase
    """
    #convert the date to decimal data
    date_decimal = [float(x)/365 for x in ((Date-np.datetime64('1980-10-01'))/np.timedelta64(1, 'D'))]
    
    #remove water temps below 1C or above 60C
    #if isWater:
    #    print(temp)
    #    temp = [x if x >=1 and x<=60 else np.nan for x in temp]
    
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

    
    air_amp=[]
    air_phi=[]
    water_amp_obs = []
    water_phi_obs = []
    water_amp_sntemp = []
    water_phi_sntemp = []
    
    #get the phase and amplitude for air and water temps for each segment
    for i in range(len(thisData['seg_id_nat'])):
        thisSeg = thisData['seg_id_nat'][i].data
        #get the air temp properties
        amp, phi = amp_phi(thisData['date'].values,thisData['seg_tave_air'][:,i].values,isWater=False)
        air_amp.append(amp)
        air_phi.append(phi)

        #get the sntemp water temp properties
        amp, phi = amp_phi(thisData['date'].values,thisData['seg_tave_water_sntemp'][:,i].values,isWater=True)
        water_amp_sntemp.append(amp)
        water_phi_sntemp.append(phi)
        
        #get the water temp properties
        #ensure sufficient data
        if np.sum(np.isfinite(thisData['seg_tave_water'][:,i].values))>(365*2): #this requires at least 2 years of data, other requirements added below
            waterDF = pd.DataFrame({'date':thisData['date'].values,'seg_tave_water':thisData['seg_tave_water'][:,i].values})
            #require temps > 1 and <60 C for signal analysis
            waterDF.loc[(waterDF.seg_tave_water<1)|(waterDF.seg_tave_water>60),"seg_tave_water"]=np.nan
            waterDF.dropna(inplace=True)
            
            if waterDF.shape[0]<(365*2):
                amp = np.nan
                phi = np.nan
            else:
            
                #get the longest set of temp records with no gaps >30 days
                dateDiff = [0]
                dateDiff.extend([int((waterDF.date.iloc[x]-waterDF.date.iloc[x-1])/np.timedelta64(1, 'D')) for x in range(1,waterDF.shape[0])])
                waterDF['dateDiff']=dateDiff
                if max(dateDiff)>31:
                    waterDF['bin']=pd.cut(waterDF.date,bins=waterDF.date.loc[(waterDF.dateDiff>31) | (waterDF.dateDiff==0)].values, include_lowest=True, labels=False)
                    waterSum = waterDF[['date','bin']].groupby('bin',as_index=False).count()
                    #keep the longest series
                    maxBin = waterSum.bin[waterSum.date==np.max(waterSum.date)].values[0]
                    waterDF = waterDF.loc[waterDF.bin==maxBin]
                
                if waterDF.shape[0]>=(365*2):
                    amp, phi = amp_phi(waterDF.date.values,waterDF.seg_tave_water.values,isWater=True)
                else:
                    amp = np.nan
                    phi = np.nan
            
        else:
            amp=np.nan
            phi=np.nan
        water_amp_obs.append(amp)
        water_phi_obs.append(phi)

    Ar_obs = [water_amp_obs[x]/air_amp[x] for x in range(len(water_amp_obs))]
    delPhi_obs = [(water_phi_obs[x]-air_phi[x])*365/(2*math.pi) for x in range(len(water_amp_obs))]
    
    #remove Ar >1.1
    delPhi_obs=[delPhi_obs[i] if Ar_obs[i] <=1.1 else np.nan for i in range(len(delPhi_obs))]
    Ar_obs = [x if x <= 1.1 else np.nan for x in Ar_obs]
    
    #remove delPhi <-10
    Ar_obs = [Ar_obs[i] if delPhi_obs[i] >=-10 else np.nan for i in range(len(Ar_obs))]
    delPhi_obs = [x if x >=-10 else np.nan for x in delPhi_obs]
    
    #reset delPhi -10 to 0
    delPhi_obs = [x if x > 0 else 0 if np.isfinite(x) else np.nan for x in delPhi_obs]
    
    
    
    Ar_sntemp = [water_amp_sntemp[x]/air_amp[x] for x in range(len(water_amp_sntemp))]
    delPhi_sntemp = [(water_phi_sntemp[x]-air_phi[x])*365/(2*math.pi) for x in range(len(water_amp_sntemp))]
    
    tempDF = pd.DataFrame({'seg_id_nat':thisData['seg_id_nat'], 'air_amp':air_amp,'air_phi':air_phi,'water_amp_obs':water_amp_obs,'water_phi_obs':water_phi_obs,'Ar_obs':Ar_obs,'delPhi_obs':delPhi_obs,'water_amp_sntemp':water_amp_sntemp,'water_phi_sntemp':water_phi_sntemp,'Ar_sntemp':Ar_sntemp,'delPhi_sntemp':delPhi_sntemp})
    
    return tempDF

def prep_annual_signal_data(
    obs_temper_file,
    pretrain_file,
    train_start_date,
    train_end_date,
    val_start_date,
    val_end_date,
    test_start_date,
    test_end_date,
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
    obs=obs[["seg_tave_air","seg_tave_water","temp_c"]]
    obs = obs.rename({"seg_tave_water": "seg_tave_water_sntemp"})
    obs = obs.rename({"temp_c": "seg_tave_water"})
    
    #split into testing and training
    obs_trn, obs_val, obs_tst = separate_trn_tst(obs, train_start_date,
        train_end_date,
        val_start_date,
        val_end_date,
        test_start_date,
        test_end_date)
    
    
    #get the annual signal properties for the training and testing data
    GW_trn = annTempStats(obs_trn)
    GW_tst = annTempStats(obs_tst)
    GW_val = annTempStats(obs_val)
    
    #save the GW data
    data = {}
    data['GW_tst']=GW_tst
    data['GW_trn']=GW_trn
    data['GW_val']=GW_val
    data['GW_cols']=GW_trn.columns.values.astype('str')
    np.savez_compressed(out_file, **data)

def calc_amp_phi(thisData):
    segList = np.unique(thisData['seg_id_nat'])
    water_amp_preds = []
    water_phi_preds = []
    for thisSeg in segList:
        amp, phi = amp_phi(thisData.loc[thisData.seg_id_nat==thisSeg,"date"].values,thisData.loc[thisData.seg_id_nat==thisSeg,"seg_tave_water"],isWater=True)
        water_amp_preds.append(amp)
        water_phi_preds.append(phi)
    return pd.DataFrame({'seg_id_nat':segList,'water_amp_pred':water_amp_preds,'water_phi_pred':water_phi_preds})
    
def merge_pred_obs(gw_obs,obs_col,pred):
    obsDF = pd.DataFrame(gw_obs[obs_col],columns=gw_obs['GW_cols'])
    obsDF = obsDF.merge(pred)
    obsDF['Ar_pred']=obsDF['water_amp_pred']/obsDF['air_amp']
    obsDF['delPhi_pred'] = (obsDF['water_phi_pred']-obsDF['air_phi'])*365/(2*math.pi)
    return obsDF
    
def calc_pred_ann_temp(GW_data,trn_data,tst_data, val_data,trn_output, tst_output,val_output):
    gw_obs = np.load(GW_data)
    
    trn_preds = pd.read_feather(trn_data)
    tst_preds = pd.read_feather(tst_data)
    val_preds = pd.read_feather(val_data)
    
    gw_trn = calc_amp_phi(trn_preds)
    gw_tst = calc_amp_phi(tst_preds)
    gw_val = calc_amp_phi(val_preds)
    
    gw_stats_trn = merge_pred_obs(gw_obs,'GW_trn',gw_trn)
    gw_stats_tst = merge_pred_obs(gw_obs,'GW_tst',gw_tst)
    gw_stats_val = merge_pred_obs(gw_obs,'GW_val',gw_val)
                       
    gw_stats_trn.to_csv(trn_output)
    gw_stats_tst.to_csv(tst_output)
    gw_stats_val.to_csv(val_output)
    
def calc_gw_metrics(trnFile,tstFile,valFile,outFile,figFile1, figFile2):
    trnDF = pd.read_csv(trnFile)
    tstDF = pd.read_csv(tstFile)
    valDF = pd.read_csv(valFile)
    
    resultsDF = 0
    for i in range(3):
        if i==0:
            thisData=trnDF
            partition="trn"
        elif i==1:
            thisData = tstDF
            partition="tst"
        elif i==2:
            thisData = valDF
            partition="val"
        for thisVar in ['Ar','delPhi']:
            print(thisVar)
            tempDF = pd.DataFrame(calc_metrics(thisData[["{}_obs".format(thisVar),"{}_pred".format(thisVar)]].rename(columns={"{}_obs".format(thisVar):"obs","{}_pred".format(thisVar):"pred"}))).T
            tempDF['variable']=thisVar
            tempDF['partition']=partition
            tempDF['model']='RGCN'

            if type(resultsDF)==int:
                resultsDF = tempDF
            else:
                resultsDF = resultsDF.append(tempDF,ignore_index=True)
                
            tempDF = pd.DataFrame(calc_metrics(thisData[["{}_obs".format(thisVar),"{}_sntemp".format(thisVar)]].rename(columns={"{}_obs".format(thisVar):"obs","{}_sntemp".format(thisVar):"pred"}))).T
            tempDF['variable']=thisVar
            tempDF['partition']=partition
            tempDF['model']='SNTemp'
            resultsDF = resultsDF.append(tempDF,ignore_index=True)
                
    resultsDF.to_csv(outFile,header=True, index=False)
    
    fig = plt.figure(figsize=(15, 15))
    partDict = {'Training':trnDF,'Testing':tstDF,'Validation':valDF}
    metricLst = ['Ar','delPhi']
    thisFig = 0
    for thisPart in partDict.keys():
            thisData = partDict[thisPart]
            for thisMetric in metricLst:
                thisFig = thisFig + 1
                ax = fig.add_subplot(len(partDict), len(metricLst), thisFig, aspect='equal')
                ax.set_title('{}, {}'.format(thisMetric, thisPart))
                ax.axline((np.nanmean(thisData['{}_pred'.format(thisMetric)]),np.nanmean(thisData['{}_pred'.format(thisMetric)])), slope=1.0,linewidth=1, color='r', label="1 to 1 line")
                ax.scatter(x=thisData['{}_obs'.format(thisMetric)],y=thisData['{}_pred'.format(thisMetric)],label="RGCN",color="blue")
                ax.scatter(x=thisData['{}_obs'.format(thisMetric)],y=thisData['{}_sntemp'.format(thisMetric)],label="SNTEMP",color="red")
                for i, label in enumerate(thisData.seg_id_nat):
                    ax.annotate(int(label), (thisData['{}_obs'.format(thisMetric)][i],thisData['{}_pred'.format(thisMetric)][i]))
                if thisFig==1:
                          ax.legend()
                ax.set_xlabel("Observed")
                ax.set_ylabel("Predicted")

    plt.savefig(figFile1)
    
    fig = plt.figure(figsize=(15, 15))
    partDict = {'Training':trnDF,'Testing':tstDF,'Validation':valDF}
    metricLst = ['Ar','delPhi']
    thisFig = 0
    for thisPart in partDict.keys():
            thisData = partDict[thisPart]
            for thisMetric in metricLst:
                thisFig = thisFig + 1
                colsToPlot = ['{}_obs'.format(thisMetric),'{}_sntemp'.format(thisMetric),'{}_pred'.format(thisMetric)]
                nObs =["n: " + str(np.sum(np.isfinite(thisData[thisCol].values))) for thisCol in colsToPlot]
                ax = fig.add_subplot(len(partDict), len(metricLst), thisFig)
                ax.set_title('{}, {}'.format(thisMetric, thisPart))
                ax=sns.boxplot(data=thisData[colsToPlot])
                # Add it to the plot
                pos = range(len(nObs))
                for tick,label in zip(pos,ax.get_xticklabels()):
                    ax.text(pos[tick],
                            np.nanmin(thisData[colsToPlot].values)-0.1*(np.nanmax(thisData[colsToPlot].values)-np.nanmin(thisData[colsToPlot].values)),
                            nObs[tick],
                            horizontalalignment='center',
                            weight='semibold')
                ax.set_ylim(np.nanmin(thisData[colsToPlot].values)-0.2*(np.nanmax(thisData[colsToPlot].values)-np.nanmin(thisData[colsToPlot].values)),np.nanmax(thisData[colsToPlot].values))

    plt.savefig(figFile2)
    
