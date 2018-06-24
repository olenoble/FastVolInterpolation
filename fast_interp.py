#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 15:25:17 2018

@author: olivier
"""

import pandas as pd
import numpy as np
from dateutil.parser import parse
from time import time
from datetime import timedelta
from scipy.stats import norm


############################################################################
### Params
tenor_target = 3
strike_target = 1.0

############################################################################
### Various Functions
def getInterpolatedFwd(inputs, datelist, daterefs, term):
    fwd_vals = np.empty(len(datelist))
    for i, d in enumerate(datelist):
        pos_vec = daterefs[d]
        fwd_vals[i] = np.interp(term,
                                inputs[pos_vec[0]:pos_vec[1]][:,0],
                                inputs[pos_vec[0]:pos_vec[1]][:,1])

    return pd.DataFrame({'Date': datelist, 'F': fwd_vals})

def generate_dates(option_ref, all_dates):
    all_volfwd = np.empty((0,4))
    
    start_dates = option_ref['Date'].values
    end_dates = option_ref['TargetExpiry'].values
    strike_list = option_ref['strike'].values
    ids = option_ref['id'].values
    for i, k in enumerate(ids):
        end_dt = end_dates[i]
        start_dt = start_dates[i]
        K = strike_list[i]
        
        unique_dt = all_dates[(all_dates >= start_dt) & (all_dates <= end_dt)]
        all_out_temp = np.empty((len(unique_dt), 4))
        all_out_temp[:, 0] = k
        all_out_temp[:, 1] = unique_dt
        all_out_temp[:, 2] = K
        all_out_temp[:, 3] = end_dt
        all_volfwd = np.vstack((all_volfwd, all_out_temp))

    output = pd.DataFrame(all_volfwd, columns=['id','Date','strike','Expiry'])
    output['Date'] = np.array(output['Date'], dtype='M8[ns]')
    output['Expiry'] = np.array(output['Expiry'], dtype='M8[ns]')
    return output

def generate_all_fwd(option_ref):
    option_ref = option_ref.sort_values(['Date', 'Expiry'])
    all_dt = option_ref['Date'].values
    unique_dt = np.unique(all_dt)
    alldtpos = np.searchsorted(all_dt, unique_dt, side='left')
    alldtpos = np.append(alldtpos, len(option_ref)-1)
    dt_ref = {x: [alldtpos[i], alldtpos[i+1]] for i,x in enumerate(unique_dt)}
    
    expiry_ts = (option_ref['Expiry'].values- all_dt) / np.timedelta64(30,'D')
    all_fwd = np.empty(len(option_ref))
    
    for i, td in enumerate(unique_dt):
        pos_i = dt_ref[td]
        exp_list = expiry_ts[pos_i[0]:pos_i[1]]
        
        fwd_pos = fwd_date_ref[td]
        fwdtemp = fwd_vals_arr[fwd_pos[0]:fwd_pos[1]]
        interpfwd = np.interp(exp_list, fwdtemp[:,0], fwdtemp[:,1])
        all_fwd[pos_i[0]:pos_i[1]] = interpfwd

    return option_ref.assign(F=all_fwd)


def CalculateVols(option_ref):
    option_ref = option_ref.sort_values(['Date', 'Expiry'])
    all_dt = option_ref['Date'].values
    unique_dt = np.unique(all_dt)
    alldtpos = np.searchsorted(all_dt, unique_dt, side='left')
    alldtpos = np.append(alldtpos, len(option_ref)-1)
    dt_ref = {x: [alldtpos[i], alldtpos[i+1]] for i,x in enumerate(unique_dt)}
    
    strike_ts = option_ref['strike'].values
    expiry_ts = (option_ref['Expiry'].values- all_dt) / np.timedelta64(30,'D')
    all_vols = np.empty(len(option_ref))
    for i, td in enumerate(unique_dt):
        pos_i = dt_ref[td]
        k_list = strike_ts[pos_i[0]:pos_i[1]]
        exp_list = expiry_ts[pos_i[0]:pos_i[1]]
        
        volpos = vol_date_ref[td]
        vol_temp = allvol_arr[volpos[0]:volpos[1]]
        
        fwd_pos = fwd_date_ref[td]
        fwdtemp = fwd_vals_arr[fwd_pos[0]:fwd_pos[1]]
        exp_max_ref = np.searchsorted(fwdtemp[:,0], exp_list[-1],side='left')
        exp_max_ref = min(len(fwdtemp)-1, exp_max_ref)
        max_exp = fwdtemp[exp_max_ref,0]
        vol_temp = vol_temp[vol_temp[:,0] <= max_exp]
        
        num_opt = len(k_list)
        strike_vols = np.empty((num_opt, exp_max_ref + 1))
        for j in range(exp_max_ref+1):
            vol_temp_exp = vol_temp[vol_temp[:, 0]==fwdtemp[j,0]]
            vol_exp_interp = np.interp(k_list, vol_temp_exp[:,1], vol_temp_exp[:,2])
            strike_vols[:, j] = vol_exp_interp
        
        for j in range(num_opt):
            pos_vol = pos_i[0] + j
            all_vols[pos_vol] = np.interp(exp_list[j], fwdtemp[0:exp_max_ref+1,0], strike_vols[j,:])
        
    return option_ref.assign(vol=all_vols)

def BSPrice(option_data):
    print(option_data.head())
    F = option_data['F'].values
    K = option_data['strike'].values
    iv = option_data['vol'].values
    
    vttx_sqrt = np.sqrt((option_data['Expiry'].values - option_data['Date'].values) / np.timedelta64(365,'D'))
    varsqr = iv * vttx_sqrt
    d1 = (np.log(F/K) + 0.5 * varsqr * varsqr)/varsqr
    d2 = d1 - varsqr
    return norm.cdf(d1) * F- norm.cdf(d2) * K

############################################################################
### get data
if False:
    vol = pd.read_csv('vol_test.csv')
    vol['Date'] = [parse(x) for x in vol['Date'].values]
    vol['Expiry'] = [parse(x) for x in vol['Expiry'].values]

time_elapsed = []
time_elapsed += [time()]
vol = vol.sort_values(['Date', 'Expiry', 'Strike'])

############################################################################
### Build Data Objects
time_elapsed += [time()]
alldatevec = vol['Date'].values
allvol_arr = vol[['Month', 'Strike', 'Vol']].values
alldatelist = np.unique(alldatevec)
alldatepos = np.searchsorted(alldatevec, alldatelist, side='left')
alldatepos = np.append(alldatepos, len(vol)-1)
vol_date_ref = {x: [alldatepos[i], alldatepos[i+1]] for i,x in enumerate(alldatelist)}

time_elapsed += [time()]
fwd_data = vol[['Date', 'Month', 'F']].drop_duplicates().sort_values(['Date', 'Month'])  ##4ms
fwddatevec = fwd_data['Date'].values
fwd_vals_arr = fwd_data[['Month', 'F']].values
fwddatelist = np.unique(fwddatevec)
fwddatepos = np.searchsorted(fwddatevec, fwddatelist, side='left')
fwddatepos = np.append(fwddatepos, len(fwd_data)-1)
fwd_date_ref = {x: [fwddatepos[i], fwddatepos[i+1]] for i,x in enumerate(fwddatelist)}


############################################################################
### First get strikes
time_elapsed += [time()]

strikes = getInterpolatedFwd(fwd_vals_arr, alldatelist, fwd_date_ref, tenor_target)
strikes = strikes.assign(strike= strike_target * strikes['F'].values)

time_elapsed += [time()]

############################################################################
### Get Data For Options
opt_info = strikes.assign(TargetExpiry=strikes['Date'] + timedelta(tenor_target * 30))
opt_info = opt_info.assign(id=range(len(opt_info)))

option_info_all = generate_dates(opt_info,alldatelist)
time_elapsed += [time()]

opt_info2 = generate_all_fwd(option_info_all)
time_elapsed += [time()]

opt_info3 = CalculateVols(opt_info2)
time_elapsed += [time()]


############################################################################
### Calculate BS
opt_prm = BSPrice(opt_info3)
opt_info3 = opt_info3.assign(premium=opt_prm)
time_elapsed += [time()]


time_ref = ['Sorting', 'VolHash', 'FwdHash', 'StrikeCalc', 'DateGeneration', 'FwdCalc', 'VolCalc', 'BlackScholes']
time_check = pd.Series(np.round(np.diff(time_elapsed) * 1000,0), time_ref)
print(time_check)
print('Total Time (s): ' + str(time_check.sum() / 1000))

opt_info_final = opt_info3.sort_values(['id', 'Date'])
opt_info_final.to_csv('final_20180624.csv')
