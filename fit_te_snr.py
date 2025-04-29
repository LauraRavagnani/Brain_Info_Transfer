import os
import numpy as np
import pandas as pd
import json
from parameters import par
import info_utils as iu

# Simulation and global parameters
nTrials_per_stim = par.nTrials_per_stim
n_subjects = par.n_subjects                
nShuff = par.nShuff                 
w_sig = par.w_sig[5]            # signal weight for the computation of Y
w_noise = par.w_noise[10]       # noise weight for the computation of Y
ratio = par.ratio               # ratio between stdX_sig and stdX_noise

simLen = par.simLen                             
stimWin = par.stimWin                      
delays = par.delays
n_binsS = par.n_binsS                           
n_binsX = par.n_binsX
n_binsY = par.n_binsY
eps = par.eps                            
n_bins = par.n_binsX
n_trials = nTrials_per_stim * n_binsS
snr = par.snr

# file with new time series 2E
file_time_series_2E = 'time_series_2E.npz'

# file to store the computation results
file_fit_te_direct = 'fit_te_direct.npz'
file_fit_te_direct_r = 'fit_te_direct_r.npz'

# draw a random delay for each subject
reps_delays = np.random.choice(delays, n_subjects, replace=True)

# matrices to store the simulated time series for each subject and value of snr
matrix_S = np.zeros((n_subjects, len(snr), n_trials))
matrix_X_noise = np.zeros((n_subjects, len(snr), n_trials))
matrix_X_signal = np.zeros((n_subjects, len(snr), n_trials))
matrix_Y_del = np.zeros((n_subjects, len(snr), n_trials))
matrix_Y = np.zeros((n_subjects, len(snr), n_trials))

# matrices to store the fit and te mean over the subjects 
all_fit_E = np.zeros((n_subjects, len(snr)))
all_te_E = np.zeros((n_subjects, len(snr)))



# -------------------------- routine to create the time series with different SNR -------------------------- #

def fig2E_routine():   
    print('creating files')
    
    for simIdx in range(n_subjects):
        print(f'subject {simIdx}')

        d = reps_delays[simIdx]  
        t_start = stimWin[0] 

        for i, val in enumerate(snr):
                stdX_noise = 1 / val    # snr = delta/sigma, with delta = 1 for all simulations
                stdY = 1 / val
                stdX_sig = ratio * stdX_noise

                # generate stimulus for each trial (from 1 to 4)
                S = np.random.randint(1, n_binsS + 1, size=n_trials)

                # simulate time series for X_noise and X_signal
                X_noise = np.random.normal(0, stdX_noise, size=(simLen, n_trials))
                X_signal = eps * np.random.normal(0, stdX_noise, size=(simLen, n_trials))
                
                # insert the stimulus in the stimulus window
                X_signal[stimWin[0]:stimWin[1], :] = np.tile(S, (stimWin[1] - stimWin[0], 1))

                # add multiplicative noise
                X_signal = X_signal * (1 + np.random.normal(0, stdX_sig, size=(simLen, n_trials)))

                # compute the contribution to Y from signal and noise with delay
                X2Ysig = w_sig * np.vstack((
                    eps * np.random.normal(0, stdX_noise, size=(d, n_trials)),
                    X_signal[0:simLen-d, :]
                ))
                X2Ynoise = w_noise * np.vstack((
                    eps * np.random.normal(0, stdX_noise, size=(d, n_trials)),
                    X_noise[0:simLen-d, :]
                ))
                Y = X2Ysig + X2Ynoise + np.random.normal(0, stdY, size=(simLen, n_trials))

                # save the results in the matrices (only at the time step at which the binning is computed)
                matrix_S[simIdx][i][:] = S
                matrix_X_noise[simIdx][i][:] = X_noise[t_start,:]
                matrix_X_signal[simIdx][i][:] = X_signal[t_start,:]
                matrix_Y_del[simIdx][i][:] = Y[t_start + d,:]
                matrix_Y[simIdx][i][:] = Y[t_start,:]

    return matrix_S, matrix_X_noise, matrix_X_signal, matrix_Y_del, matrix_Y


# if the file where the results are stored does not exist, run the simulation and create the file with stored results
if not os.path.exists(file_time_series_2E):
    matrix_S, matrix_X_noise, matrix_X_signal, matrix_Y_del, matrix_Y = fig2E_routine()
    np.savez(file_time_series_2E, matrix_S=matrix_S, matrix_X_noise=matrix_X_noise, matrix_X_signal=matrix_X_signal, matrix_Y_del=matrix_Y_del, matrix_Y=matrix_Y)



# -------------------------- routine to create the FIT and TE wrt SNR -------------------------- #

def compute_FIT_TE_2E(n_bins):
    print('computing fit te')

    # load simulated time series and store the quantities into arrays
    if os.path.exists(file_time_series_2E):
        loadnpz = np.load(file_time_series_2E)

        matrix_S = loadnpz['matrix_S']
        matrix_X_noise = loadnpz['matrix_X_noise']
        matrix_X_signal = loadnpz['matrix_X_signal']
        matrix_Y_del = loadnpz['matrix_Y_del']
        matrix_Y = loadnpz['matrix_Y']

    # discretize neural activity with equipopolated bins and compute FIT and TE
    for subject_index in range(n_subjects):
        print(f'subject {subject_index}')

        # loop on different snr
        for snr_index in range(len(snr)):
            S = matrix_S[subject_index][snr_index][:].astype(int)

            bX_noise = pd.qcut(matrix_X_noise[subject_index][snr_index][:], n_bins, labels=range(1, n_bins + 1)).astype(int)

            bX_sig = pd.qcut(matrix_X_signal[subject_index][snr_index][:], n_bins, labels=range(1, n_bins + 1)).astype(int)

            bY_del = pd.qcut(matrix_Y_del[subject_index][snr_index][:], n_bins, labels=range(1, n_bins + 1)).astype(int)

            bY = pd.qcut(matrix_Y[subject_index][snr_index][:], n_bins, labels=range(1, n_bins + 1)).astype(int)

            bX = (bX_sig - 1) * n_bins + bX_noise

            te, fit, _, _, _, _ = iu.compute_FIT_TE(S, bX, bY_del, bY)

            all_fit_E[subject_index][snr_index] = fit
            all_te_E[subject_index][snr_index] = te
            
            
    # mean and std (sem) over subjects
    fit_E_mean = np.mean(all_fit_E, axis=0)
    fit_E_std = np.std(all_fit_E, axis=0) / np.sqrt(n_subjects)
    te_E_mean = np.mean(all_te_E, axis=0)
    te_E_std = np.std(all_te_E, axis=0) / np.sqrt(n_subjects)
    return fit_E_mean, fit_E_std, te_E_mean, te_E_std 


# if the file where the results are stored does not exist, run the computation and create the file with stored results
if not os.path.exists(file_fit_te_direct):
    fit_E_mean, fit_E_std, te_E_mean, te_E_std = compute_FIT_TE_2E(n_bins)
    np.savez(file_fit_te_direct, fit_E_mean=fit_E_mean, fit_E_std=fit_E_std, te_E_mean=te_E_mean, te_E_std=te_E_std)
  



# -------------------------- routine to create the FIT and TE wrt SNR with alternative binning -------------------------- #

def compute_FIT_TE_2E_r(n_bins):
    print('computing fit te right binning')

    # load simulated time series and store the quantities into arrays
    if os.path.exists(file_time_series_2E):
        loadnpz = np.load(file_time_series_2E)

        matrix_S = loadnpz['matrix_S']
        matrix_X_noise = loadnpz['matrix_X_noise']
        matrix_X_signal = loadnpz['matrix_X_signal']
        matrix_Y_del = loadnpz['matrix_Y_del']
        matrix_Y = loadnpz['matrix_Y']

        matrix_X = matrix_X_signal + matrix_X_noise     # create stimulus X by adding signal and noise

    for subject_index in range(n_subjects):
        print(f'subject {subject_index}')

         # loop on different snr
        for snr_index in range(len(snr)):
            S = matrix_S[subject_index][snr_index][:].astype(int)

            bX = pd.qcut(matrix_X[subject_index][snr_index][:], n_bins, labels=range(1, n_bins + 1)).astype(int)

            bY_del = pd.qcut(matrix_Y_del[subject_index][snr_index][:], n_bins, labels=range(1, n_bins + 1)).astype(int)

            bY = pd.qcut(matrix_Y[subject_index][snr_index][:], n_bins, labels=range(1, n_bins + 1)).astype(int)

            te, fit, _, _, _, _ = iu.compute_FIT_TE(S, bX, bY_del, bY)

            all_fit_E[subject_index][snr_index] = fit
            all_te_E[subject_index][snr_index] = te
            
            
    # mean and std (sem) over subjects
    fit_E_mean_r = np.mean(all_fit_E, axis=0)
    fit_E_std_r = np.std(all_fit_E, axis=0) / np.sqrt(n_subjects)
    te_E_mean_r = np.mean(all_te_E, axis=0)
    te_E_std_r = np.std(all_te_E, axis=0) / np.sqrt(n_subjects)
    return fit_E_mean_r, fit_E_std_r, te_E_mean_r, te_E_std_r 


# if the file where the results are stored does not exist, run the computation and create the file with stored results
if not os.path.exists(file_fit_te_direct_r):
    fit_E_mean_r, fit_E_std_r, te_E_mean_r, te_E_std_r = compute_FIT_TE_2E_r(n_bins)
    np.savez(file_fit_te_direct_r, fit_E_mean_r=fit_E_mean_r, fit_E_std_r=fit_E_std_r, te_E_mean_r=te_E_mean_r, te_E_std_r=te_E_std_r)
  
