import pandas as pd
import numpy as np
import xarray as xr
from sklearn.linear_model import LinearRegression
from datetime import datetime
import math
from itertools import compress

from river_dl.preproc_utils import separate_trn_tst, read_multiple_obs

def annTempStats(thisData):
    """
    calculate the annual signal properties (phase and amplitude) for temperature times series
    :param thisData: [xr dataset] with time series data of air and water temp for each segment
    :returns: data frame with phase and amplitude of air and observed water temp, along with the
    phase shift and amplitude ratio for each segment
    """

    #convert the date to decimal dates
    date_decimal = [float(x)/365 for x in ((thisData['date']-np.datetime64('1980-10-01'))/86400000000000.0)]
    
    air_amp=[]
    air_phi=[]
    water_amp_obs = []
    water_phi_obs = []
    nTemps = []
    
    #get the phase and amplitude for air and water temps for each segment
    for i in range(len(thisData['seg_id_nat'])):
        thisSeg = thisData['seg_id_nat'][i]
        tmean_air = thisData['seg_tave_air'][:,i]
        tmean_water = thisData['seg_tave_water'][i,:]
        x = [[math.sin(2*math.pi*j),math.cos(2*math.pi*j)] for j in date_decimal]
        model = LinearRegression().fit(x,tmean_air)
        amp = math.sqrt(model.coef_[0]**2+model.coef_[1]**2)
        phi = math.asin(model.coef_[1]/amp)
        air_amp.append(amp)
        air_phi.append(phi)
        if np.sum(np.isfinite(tmean_water).values)>365: #this requires at least 1 year of data, need to add other data requirements here
            model = LinearRegression().fit(list(compress(x, np.isfinite(tmean_water).values)),list(compress(tmean_water, np.isfinite(tmean_water).values)))
            amp = math.sqrt(model.coef_[0]**2+model.coef_[1]**2)
            phi = math.asin(model.coef_[1]/amp)
        else:
            amp=np.nan
            phi=np.nan
        water_amp_obs.append(amp)
        water_phi_obs.append(phi)
        nTemps.append(np.sum(np.isfinite(tmean_water).values))
    Ar_obs = [water_amp_obs[x]/air_amp[x] for x in range(len(water_amp_obs))]
    delPhi_obs = [(water_phi_obs[x]-air_phi[x])*365/(2*math.pi) for x in range(len(water_amp_obs))]
    
    tempDF = pd.DataFrame({'seg_id_nat':thisData['seg_id_nat'], 'air_amp':air_amp,'air_phi':air_phi,'water_amp_obs':water_amp_obs,'water_phi_obs':water_phi_obs,'Ar_obs':Ar_obs,'delPhi_obs':delPhi_obs})
    
    return tempDF

def prep_annual_signal_data(
    obs_temper_file,
    pretrain_file,
    catch_prop_file=None,
    test_start_date="2004-09-30",
    n_test_yr=12,
    exclude_file=None,
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
    obs.append(xr.open_zarr(obs_temper_file))
    obs=xr.merge(obs,join="left")
    obs=obs[["seg_tave_air","temp_c"]]
    obs = obs.rename({"temp_c": "seg_tave_water"})
    
    #split into testing and training
    obs_trn, obs_tst = separate_trn_tst(obs, test_start_date, n_test_yr)
    
    #get the annual signal properties for the training and testing data
    GW_trn = annTempStats(obs_trn)
    GW_tst = annTempStats(obs_tst)
    
    #add the GW data to the existing preppedData file
    preppedData = np.load(out_file)
    data = {k:v for  k, v in preppedData.items()
    data['GW_tst']=GW_tst
    data['GW_trn']=GW_trn
    data['GW_cols']=GW_trn.columns.values.astype('str')}
    np.savez_compressed(out_file, **data)
    
    
    