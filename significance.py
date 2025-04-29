import info_utils as iu
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# files to store the results
file_sig_Sshuff = 'sig_w_Sshuff.npz' # "w" stands for "wrong" way of binning
file_sig_Sfix = 'sig_w_Sfix.npz'
file_sig_Sshuff_r = 'sig_r_Sshuff.npz' # "r" stands for "right" way of binning
file_sig_Sfix_r = 'sig_r_Sfix.npz'

# matrices to store the fit and te mean over the subjects   
all_fit_sig_Sshuff = np.zeros((n_test, n_subjects, nShuff))
all_fit_sig_Sfix = np.zeros((n_test, n_subjects, nShuff))
all_te_sig_Sfix = np.zeros((n_test, n_subjects, nShuff))


# -------------------------- routine to shuffle S across trials -------------------------- #

def significance_routine_Sshuff(w_idx, n_bins, n_shuff):
    for subject_index in range(n_subjects):
        print(f'subject {subject_index}')

        # load simulated time series and store the quantities into arrays
        subject_file = os.path.join("Simulations", f"subject{subject_index:02d}.json")

        df = pd.read_json(subject_file, orient='index')

        for i, w in enumerate(w_idx):
            print(f'\tweight index: {w}')

            S = np.array(df['S'][w])
            d = df['d'][w]
            t_start = df['t_start'][w]
            X_noise = df['X_noise'][w]
            X_signal = df['X_signal'][w]
            Y = df['Y'][w]

            # discretize neural activity with equipopolated bins and compute FIT
            for shuff in range(n_shuff):
                # shuffling
                np.random.shuffle(S)

                bX_noise = pd.qcut(X_noise[t_start][:], n_bins, labels=range(1, n_bins + 1)).astype(int)

                bX_sig = pd.qcut(X_signal[t_start][:], n_bins, labels=range(1, n_bins + 1)).astype(int)

                bY = pd.qcut(Y[t_start][:], n_bins, labels=range(1, n_bins + 1)).astype(int)

                bY_del = pd.qcut(Y[t_start+d][:], n_bins, labels=range(1, n_bins + 1)).astype(int)

                bX = (bX_sig - 1) * n_bins + bX_noise

                _, fit, _, _, _, _ = iu.compute_FIT_TE(S, bX, bY_del, bY)

                all_fit_sig_Sshuff[i][subject_index][shuff] = fit

        # mean over subjects
        fit_sig_Sshuff = np.mean(all_fit_sig_Sshuff, axis=1)

    return fit_sig_Sshuff


# if the file where the results are stored does not exist, run the computation and create the file with stored results
if not os.path.exists(file_sig_Sshuff):

    fit_sig_Sshuff = significance_routine_Sshuff(w_idx=[6, 59, 97, 101], n_bins=n_bins, n_shuff=nShuff)
    np.savez(file_sig_Sshuff, fit_sig_Sshuff=fit_sig_Sshuff)


# -------------------------- routine to shuffle X across trials with the same S -------------------------- #

def significance_routine_Sfix(w_idx, n_bins, n_shuff):
    print('S fix')
    for subject_index in range(n_subjects):
        print(f'\tsubject {subject_index}')

        # load simulated time series and store the quantities into arrays
        subject_file = os.path.join("Simulations", f"subject{subject_index:02d}.json")

        df = pd.read_json(subject_file, orient='index')

        for i, w in enumerate(w_idx):
            print(f'weight index: {w}')

            S = np.array(df['S'][w])
            d = df['d'][w]
            t_start = df['t_start'][w]
            X_noise = np.array(df['X_noise'][w])
            X_signal = np.array(df['X_signal'][w])
            Y = np.array(df['Y'][w])

            # retrieve the indexes basing on the value of the stimulus S and sort the quantities
            index_sorted = np.concatenate([np.where(S == i)[0] for i in [1,2,3,4]])

            S = S[index_sorted]
            Y = Y[:, index_sorted]
            X_noise = X_noise[:, index_sorted]
            X_signal = X_signal[:, index_sorted]

            idx1 = np.where(S == 1)[0]
            idx2 = np.where(S == 2)[0]
            idx3 = np.where(S == 3)[0]
            idx4 = np.where(S == 4)[0]

            for shuff in range(n_shuff):

                # shuffle the indexes divided by stimulus value
                np.random.shuffle(idx1)
                np.random.shuffle(idx2)
                np.random.shuffle(idx3)
                np.random.shuffle(idx4)

                shuff_idx = np.concatenate((idx1, idx2, idx3, idx4), dtype=int)

                # discretize neural activity with equipopolated bins and compute FIT and TE
                bX_noise = pd.qcut(X_noise[t_start, shuff_idx], n_bins, labels=range(1, n_bins + 1)).astype(int)

                bX_sig = pd.qcut(X_signal[t_start, shuff_idx], n_bins, labels=range(1, n_bins + 1)).astype(int)

                bY = pd.qcut(Y[t_start, :], n_bins, labels=range(1, n_bins + 1)).astype(int)

                bY_del = pd.qcut(Y[t_start+d, :], n_bins, labels=range(1, n_bins + 1)).astype(int)

                bX = (bX_sig - 1) * n_bins + bX_noise

                te, fit, _, _, _, _ = iu.compute_FIT_TE(S, bX, bY_del, bY)

                all_fit_sig_Sfix[i][subject_index][shuff] = fit
                all_te_sig_Sfix[i][subject_index][shuff] = te

        # mean over subjects
        fit_sig = np.mean(all_fit_sig_Sfix, axis=1)
        te_sig = np.mean(all_te_sig_Sfix, axis=1)

    return fit_sig, te_sig


# if the file where the results are stored does not exist, run the computation and create the file with stored results
if not os.path.exists(file_sig_Sfix):

    fit_sig_Sfix, te_sig_Sfix = significance_routine_Sfix(w_idx=[6, 59, 97, 101], n_bins=n_bins, n_shuff=nShuff)
    np.savez(file_sig_Sfix, fit_sig_Sfix=fit_sig_Sfix, te_sig_Sfix=te_sig_Sfix)



# -------------------------- routine to shuffle S across trials with alternative binning -------------------------- #

def significance_routine_Sshuff_r(w_idx, n_bins, n_shuff):
    print('s shuff right binning')
    for subject_index in range(n_subjects):
        print(f'subject {subject_index}')

        # load simulated time series and store the quantities into arrays
        subject_file = os.path.join("Simulations", f"subject{subject_index:02d}.json")

        df = pd.read_json(subject_file, orient='index')

        for i, w in enumerate(w_idx):
            print(f'\tweight index: {w}')

            S = np.array(df['S'][w])
            d = df['d'][w]
            t_start = df['t_start'][w]
            X_noise = df['X_noise'][w]
            X_signal = df['X_signal'][w]
            Y = df['Y'][w]

            X_noise_t = np.array(X_noise[t_start])
            X_signal_t = np.array(X_signal[t_start])

            X_t = X_noise_t + X_signal_t       # create stimulus X by adding signal and noise

            # discretize neural activity with equipopolated bins and compute FIT
            for shuff in range(n_shuff):

                # shuffling
                np.random.shuffle(S)

                bX = pd.qcut(X_t, n_bins, labels=range(1, n_bins + 1)).astype(int)
                
                bY = pd.qcut(Y[t_start][:], n_bins, labels=range(1, n_bins + 1)).astype(int)

                bY_del = pd.qcut(Y[t_start+d][:], n_bins, labels=range(1, n_bins + 1)).astype(int)

                _, fit, _, _, _, _ = iu.compute_FIT_TE(S, bX, bY_del, bY)

                all_fit_sig_Sshuff[i][subject_index][shuff] = fit

        # mean over subjects
        fit_sig_Sshuff_r = np.mean(all_fit_sig_Sshuff, axis=1)

    return fit_sig_Sshuff_r


# if the file where the results are stored does not exist, run the computation and create the file with stored results
if not os.path.exists(file_sig_Sshuff_r):

    fit_sig_Sshuff_r = significance_routine_Sshuff_r(w_idx=[6, 59, 97, 101], n_bins=n_bins, n_shuff=nShuff)
    np.savez(file_sig_Sshuff_r, fit_sig_Sshuff_r=fit_sig_Sshuff_r)



# -------------------------- routine to shuffle X across trials with the same S -------------------------- #

def significance_routine_Sfix_r(w_idx, n_bins, n_shuff):
    print('S fix right binning')
    for subject_index in range(n_subjects):
        print(f'\tsubject {subject_index}')

        # load simulated time series and store the quantities into arrays
        subject_file = os.path.join("Simulations", f"subject{subject_index:02d}.json")

        df = pd.read_json(subject_file, orient='index')

        for i, w in enumerate(w_idx):
            print(f'weight index: {w}')

            S = np.array(df['S'][w])
            d = df['d'][w]
            t_start = df['t_start'][w]
            X_noise = np.array(df['X_noise'][w])
            X_signal = np.array(df['X_signal'][w])
            Y = np.array(df['Y'][w])

            X = X_noise + X_signal  # create stimulus X by adding signal and noise

            # retrieve the indexes basing on the value of the stimulus S and sort the quantities
            index_sorted = np.concatenate([np.where(S == i)[0] for i in [1,2,3,4]])

            S = S[index_sorted]
            Y = Y[:, index_sorted]
            X = X[:, index_sorted]

            idx1 = np.where(S == 1)[0]
            idx2 = np.where(S == 2)[0]
            idx3 = np.where(S == 3)[0]
            idx4 = np.where(S == 4)[0]


            for shuff in range(n_shuff):

                # shuffle the indexes divided by stimulus value
                np.random.shuffle(idx1)
                np.random.shuffle(idx2)
                np.random.shuffle(idx3)
                np.random.shuffle(idx4)

                shuff_idx = np.concatenate((idx1, idx2, idx3, idx4), dtype=int)

                # discretize neural activity with equipopolated bins and compute FIT and TE
                bX = pd.qcut(X[t_start, shuff_idx], n_bins, labels=range(1, n_bins + 1)).astype(int)
            
                bY = pd.qcut(Y[t_start, :], n_bins, labels=range(1, n_bins + 1)).astype(int)

                bY_del = pd.qcut(Y[t_start+d, :], n_bins, labels=range(1, n_bins + 1)).astype(int)

                te, fit, _, _, _, _ = iu.compute_FIT_TE(S, bX, bY_del, bY)

                all_fit_sig_Sfix[i][subject_index][shuff] = fit
                all_te_sig_Sfix[i][subject_index][shuff] = te

         # mean over subjects
        fit_sig_r = np.mean(all_fit_sig_Sfix, axis=1)
        te_sig_r = np.mean(all_te_sig_Sfix, axis=1)


    return fit_sig_r, te_sig_r


# if the file where the results are stored does not exist, run the computation and create the file with stored results
if not os.path.exists(file_sig_Sfix_r):

    fit_sig_Sfix_r, te_sig_Sfix_r = significance_routine_Sfix_r(w_idx=[6, 59, 97, 101], n_bins=n_bins, n_shuff=nShuff)
    np.savez(file_sig_Sfix_r, fit_sig_Sfix_r=fit_sig_Sfix_r, te_sig_Sfix_r=te_sig_Sfix_r)