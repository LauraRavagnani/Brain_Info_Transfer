import info_utils as iu
import os
import numpy as np
import pandas as pd
from parameters import par

# import parameters
n_subjects = par.n_subjects
n_weights = len(par.w_noise) * len(par.w_sig)
n_binsS = par.n_binsS
n_bins = par.n_binsX
delay_max = par.delay_max
simLen = par.simLen
nShuff = par.nShuff
n_trials = par.nTrials_per_stim * par.n_binsS
n_test = par.n_test
w_sig = par.w_sig
w_noise = par.w_noise
eps = par.eps
stdX_noise = par.stdX_noise 
stdY = par.stdY

# files to store the results
file_heatmap = 'heatmap_3b_w.npz'   
file_heatmap_r = 'heatmap_3b_r.npz'   
file_heatmap_r_5 = 'heatmap_5b_r.npz'  

# matrices to store the fit and te mean over the subjects                                        
all_fit = np.zeros((n_subjects, n_weights))
all_te = np.zeros((n_subjects, n_weights))


# -------------------------- routine to create the heatmaps -------------------------- #

def fig2A_routine(n_bins):    
    for subject_index in range(n_subjects):
        print(f'subject {subject_index}')

        # load simulated time series and store the quantities into arrays
        subject_file = os.path.join("Simulations", f"subject{subject_index:02d}.json")

        df = pd.read_json(subject_file, orient='index')

        S = df['S']
        d = df['d'].values[0]
        t_start = df['t_start'].values[0]
        X_noise = df['X_noise']
        X_signal = df['X_signal']

        Y = df['Y']

        # discretize neural activity with equipopolated bins and compute FIT and TE
        for w in range(n_weights):
            bX_noise = pd.qcut(X_noise[w][t_start][:], n_bins, labels=range(1, n_bins + 1)).astype(int)

            bX_sig = pd.qcut(X_signal[w][t_start][:], n_bins, labels=range(1, n_bins + 1)).astype(int)

            bY_del = pd.qcut(Y[w][t_start + d][:], n_bins, labels=range(1, n_bins + 1)).astype(int)

            bY = pd.qcut(Y[w][t_start][:], n_bins, labels=range(1, n_bins + 1)).astype(int)

            bX = (bX_sig - 1) * n_bins + bX_noise

            te, fit, _, _, _, _ = iu.compute_FIT_TE(S[w], bX, bY_del, bY)

            all_fit[subject_index][w] = fit
            all_te[subject_index][w] = te
            
    # mean over subjects
    fit_heatmap = np.mean(all_fit, axis=0).reshape(len(par.w_noise), len(par.w_sig))
    te_heatmap = np.mean(all_te, axis=0).reshape(len(par.w_noise), len(par.w_sig))

    return fit_heatmap, te_heatmap


# if the file where the results are stored does not exist, run the computation and create the file with stored results
if not os.path.exists(file_heatmap):

    fit_heatmap, te_heatmap = fig2A_routine(n_bins=n_bins)
    np.savez(file_heatmap, fit_heatmap=fit_heatmap, te_heatmap=te_heatmap)


# -------------------------- routine to create the heatmaps with alternative binning -------------------------- #

def fig2A_routine_r(n_bins):
    print('right binning')    
    for subject_index in range(n_subjects):
        print(f'subject {subject_index}')

        # load simulated time series and store the quantities into arrays
        subject_file = os.path.join("Simulations", f"subject{subject_index:02d}.json")

        df = pd.read_json(subject_file, orient='index')

        S = df['S']
        d = df['d'].values[0]
        t_start = df['t_start'].values[0]
        X_noise = df['X_noise']
        X_signal = df['X_signal']
        Y = df['Y']

        # discretize neural activity with equipopolated bins and compute FIT and TE
        for w in range(n_weights):

            X_noise_t = np.array(X_noise[w][t_start])
            X_signal_t = np.array(X_signal[w][t_start])

            X_t = X_noise_t + X_signal_t       # create stimulus X by adding signal and noise

            bX = pd.qcut(X_t, n_bins, labels=range(1, n_bins + 1)).astype(int)

            bY_del = pd.qcut(Y[w][t_start + d][:], n_bins, labels=range(1, n_bins + 1)).astype(int)

            bY = pd.qcut(Y[w][t_start][:], n_bins, labels=range(1, n_bins + 1)).astype(int)

            te, fit, _, _, _, _ = iu.compute_FIT_TE(S[w], bX, bY_del, bY)

            all_fit[subject_index][w] = fit
            all_te[subject_index][w] = te
            
    # mean over subjects
    fit_heatmap = np.mean(all_fit, axis=0).reshape(len(par.w_noise), len(par.w_sig))
    te_heatmap = np.mean(all_te, axis=0).reshape(len(par.w_noise), len(par.w_sig))

    return fit_heatmap, te_heatmap

# if the file where the results are stored does not exist, run the computation and create the file with stored results
if not os.path.exists(file_heatmap_r):

    fit_heatmap, te_heatmap = fig2A_routine_r(n_bins=n_bins)
    np.savez(file_heatmap_r, fit_heatmap=fit_heatmap, te_heatmap=te_heatmap)


# -------------------------- routine to create the heatmaps with alternative binning + 5 bins -------------------------- #

# if the file where the results are stored does not exist, run the computation and create the file with stored results
if not os.path.exists(file_heatmap_r_5):
    print('5 bins')
    fit_heatmap_5, te_heatmap_5 = fig2A_routine_r(n_bins=5)
    np.savez(file_heatmap_r_5, fit_heatmap_5=fit_heatmap_5, te_heatmap_5=te_heatmap_5)



