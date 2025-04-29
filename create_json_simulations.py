import os
import numpy as np
import pandas as pd
import json
from parameters import par

# Simulation and global parameters
nTrials_per_stim = par.nTrials_per_stim
n_subjects = par.n_subjects                
nShuff = par.nShuff                 
w_sig = par.w_sig      
w_noise = par.w_noise    
stdX_noise = par.stdX_noise                   
stdY = par.stdY                           
ratio = par.ratio                         
stdX_sig = ratio * stdX_noise       
simLen = par.simLen                             
stimWin = par.stimWin                     
delays = par.delays
n_binsS = par.n_binsS                             
n_binsX = par.n_binsX
n_binsY = par.n_binsY
eps = par.eps                             
nTrials = nTrials_per_stim * n_binsS

# create the main folder "Simulations" if it does not exist
base_dir = "Simulations"
os.makedirs(base_dir, exist_ok=True)

# draw a random delay for each subject
reps_delays = np.random.choice(delays, n_subjects, replace=True)

# subject loop
for simIdx in range(n_subjects):
    print(f"Simulation number: {simIdx}")
    d = reps_delays[simIdx]  
    t_start = stimWin[0] 
    t_del = t_start + d

    subj_file = f"subject{simIdx:02d}.json"
    filepath = os.path.join(base_dir, subj_file)

    # dictionary to save the time series for each subject
    data_json = {}
    
    # weight loop
    for sigIdx in range(len(w_sig)):
        for noiseIdx in range(len(w_noise)):

            weight_index = sigIdx * len(w_noise) + noiseIdx     # create an unique index for the weights
            
            # generate stimulus for each trial (from 1 to 4)
            S = np.random.randint(1, n_binsS + 1, size=nTrials)
            
            # simulate time series for X_noise and X_signal
            X_noise = np.random.normal(0, stdX_noise, size=(simLen, nTrials))
            X_signal = eps * np.random.normal(0, stdX_noise, size=(simLen, nTrials))

            # insert the stimulus in the stimulus window
            X_signal[stimWin[0]:stimWin[1], :] = np.tile(S, (stimWin[1] - stimWin[0], 1))

            # add multiplicative noise
            X_signal = X_signal * (1 + np.random.normal(0, stdX_sig, size=(simLen, nTrials)))

            # compute the contribution to Y from signal and noise with delay
            X2Ysig = w_sig[sigIdx] * np.vstack((
                eps * np.random.normal(0, stdX_noise, size=(d, nTrials)),
                X_signal[0:simLen-d, :]
            ))
            X2Ynoise = w_noise[noiseIdx] * np.vstack((
                eps * np.random.normal(0, stdX_noise, size=(d, nTrials)),
                X_noise[0:simLen-d, :]
            ))
            Y = X2Ysig + X2Ynoise + np.random.normal(0, stdY, size=(simLen, nTrials))

            # save results into the dictionary
            data_json[f'{weight_index}'] = {
                'S': S.tolist(),
                't_start': int(t_start),
                'd': int(d),
                'X_noise': X_noise.tolist(),
                'X_signal': X_signal.tolist(),
                'Y': Y.tolist()
            }

    with open(filepath, 'w') as f:
        json.dump(data_json, f)
    
    print(f"file creat: {filepath}")

