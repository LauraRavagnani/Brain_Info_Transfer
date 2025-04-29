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

# files to store the results
file_tempor_loc = 'tempor_loc.npz'
file_tempor_loc_r = 'tempor_loc_r.npz'

# matrices to store the fit and te mean over subjects and delays
all_fit_B = np.zeros((n_subjects, simLen, delay_max))
all_te_B = np.zeros((n_subjects, simLen, delay_max))


# -------------------------- routine to create the temporal location -------------------------- #

def fig2B_routine(w_idx, simLen, delay_max, n_bins):
    print('wrong binning')
    for subject_index in range(n_subjects):
        print(f'subject {subject_index}')

        # load simulated time series and store the quantities into arrays
        subject_file = os.path.join("Simulations", f"subject{subject_index:02d}.json")

        df = pd.read_json(subject_file, orient='index')

        S = df['S']
        X_noise = df['X_noise']
        X_signal = df['X_signal']
        Y = df['Y']

        # Loop over time and delays
        for t in range(simLen):
            for d in range(delay_max):

                # discretize neural activity with equipopolated bins and compute FIT and TE
                if (t+d) < simLen:
                    bX_noise = pd.qcut(X_noise[w_idx][t][:], n_bins, labels=range(1, n_bins + 1)).astype(int)

                    bX_sig = pd.qcut(X_signal[w_idx][t][:], n_bins, labels=range(1, n_bins + 1)).astype(int)

                    bY = pd.qcut(Y[w_idx][t][:], n_bins, labels=range(1, n_bins + 1)).astype(int)

                    bY_del = pd.qcut(Y[w_idx][t+d][:], n_bins, labels=range(1, n_bins + 1)).astype(int)

                else: # take the noise from the beginning if t+d > simLen
                    bX_noise = pd.qcut(X_noise[w_idx][t][:], n_bins, labels=range(1, n_bins + 1)).astype(int)

                    bX_sig = pd.qcut(X_signal[w_idx][t][:], n_bins, labels=range(1, n_bins + 1)).astype(int)

                    bY = pd.qcut(Y[w_idx][t][:], n_bins, labels=range(1, n_bins + 1)).astype(int)

                    bY_del = pd.qcut(Y[w_idx][t+d-simLen][:], n_bins, labels=range(1, n_bins + 1)).astype(int)

                bX = (bX_sig - 1) * n_bins + bX_noise

                te, fit, _, _, _, _ = iu.compute_FIT_TE(S[w_idx], bX, bY_del, bY)
                
                all_fit_B[subject_index][t][d] = fit
                all_te_B[subject_index][t][d] = te

    # mean over subjects and delays
    fit_B = np.mean(all_fit_B, axis=(0, 2))
    fit_B_std = np.std(all_fit_B, axis=(0, 2))

    te_B = np.mean(all_te_B, axis=(0, 2))
    te_B_std = np.std(all_te_B, axis=(0, 2))

    return fit_B, fit_B_std, te_B, te_B_std


# if the file where the results are stored does not exist, run the computation and create the file with stored results
if not os.path.exists(file_tempor_loc):
    fit_B, fit_B_std, te_B, te_B_std = fig2B_routine(w_idx=65, simLen=simLen, delay_max=delay_max, n_bins=n_bins) # w_idx=65 correspond to: w_sig=0.5, w_noise=1
    fit_B_std /= np.sqrt(n_subjects * delay_max)    # standard error of the mean
    te_B_std /= np.sqrt(n_subjects * delay_max)     # standard error of the mean
    np.savez(file_tempor_loc, fit_B=fit_B, fit_B_std=fit_B_std, te_B=te_B, te_B_std=te_B_std)



# -------------------------- routine to create the temporal location with alternative binning -------------------------- #

def fig2B_routine_r(w_idx, simLen, delay_max, n_bins):
    print('right binning')
    for subject_index in range(n_subjects):
        print(f'soggetto {subject_index}')

        # load simulated time series and store the quantities into arrays
        subject_file = os.path.join("Simulations", f"subject{subject_index:02d}.json")

        df = pd.read_json(subject_file, orient='index')

        S = df['S']
        X_noise = df['X_noise']
        X_signal = df['X_signal']
        Y = df['Y']

        # create stimulus X by adding signal and noise

        # Loop over time and delays
        for t in range(simLen):
            for d in range(delay_max):
                
                X_noise_t = np.array(X_noise[w_idx][t])
                X_signal_t = np.array(X_signal[w_idx][t])

                X_t = X_noise_t + X_signal_t       # create stimulus X by adding signal and noise

                # discretize neural activity with equipopolated bins and compute FIT and TE
                if (t+d) < simLen:
                    bX = pd.qcut(X_t, n_bins, labels=range(1, n_bins + 1)).astype(int)

                    bY = pd.qcut(Y[w_idx][t][:], n_bins, labels=range(1, n_bins + 1)).astype(int)

                    bY_del = pd.qcut(Y[w_idx][t+d][:], n_bins, labels=range(1, n_bins + 1)).astype(int)

                else:
                    bX = pd.qcut(X_t, n_bins, labels=range(1, n_bins + 1)).astype(int)

                    bY = pd.qcut(Y[w_idx][t][:], n_bins, labels=range(1, n_bins + 1)).astype(int)

                    bY_del = pd.qcut(Y[w_idx][t+d-simLen][:], n_bins, labels=range(1, n_bins + 1)).astype(int)

                te, fit, _, _, _, _ = iu.compute_FIT_TE(S[w_idx], bX, bY_del, bY)
                
                all_fit_B[subject_index][t][d] = fit
                all_te_B[subject_index][t][d] = te

    # mean over subjects and delays
    fit_B = np.mean(all_fit_B, axis=(0, 2))
    fit_B_std = np.std(all_fit_B, axis=(0, 2))

    te_B = np.mean(all_te_B, axis=(0, 2))
    te_B_std = np.std(all_te_B, axis=(0, 2))

    return fit_B, fit_B_std, te_B, te_B_std


# if the file where the results are stored does not exist, run the computation and create the file with stored results
if not os.path.exists(file_tempor_loc_r):
    fit_B, fit_B_std, te_B, te_B_std = fig2B_routine_r(w_idx=65, simLen=simLen, delay_max=delay_max, n_bins=n_bins) # w_idx=65 correspond to: w_sig=0.5, w_noise=1
    fit_B_std /= np.sqrt(n_subjects * delay_max)    # standard error of the mean
    te_B_std /= np.sqrt(n_subjects * delay_max)     # standard error of the mean
    np.savez(file_tempor_loc_r, fit_B=fit_B, fit_B_std=fit_B_std, te_B=te_B, te_B_std=te_B_std)
