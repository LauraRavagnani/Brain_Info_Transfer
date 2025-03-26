import numpy as np
import numpy.random as npr
import pandas as pd

from compute_FIT_TE_DFI import compute_FIT_TE_DFI

npr.seed(2503)

def sim2ABroutine():

    # simulation parameters

    nTrials_per_stim = 500      # number of trials for each simulation
    simReps = 50                # number of simulations
    nShuff = 10                 # number of permutations

    w_sig = np.linspace(0, 1, num=11)      # signal weights for Y computation
    w_noise = np.linspace(0, 1, num=11)    # noise weights for Y computation
    stdX_noise = 2                      # std of gaussian noise in X_noise
    stdY = 2                            # std of gaussian noise in Y
    ratio = 0.2                         # ratio between stdX_sig and stdX_noise
    stdX_sig = ratio * stdX_noise       # std of gaussian noise in X_signal

    # global parameters

    simLen = 60                             # simulation length in units of 10 ms
    stimWin = [30, 35]                      # X stimulus encoding window in units of 10 ms
    delays = np.linspace(4, 6, num=3, dtype=int)
    delay_max = 10                          # Maximum Computed Delay
    n_binsS = 4                             # number of stimulus values
    n_binsX = 3
    n_binsY = 3
    eps = 1e-52                             # infinitesimal value

    nTrials = nTrials_per_stim * n_binsS

    # Draw random delay for each simulation repetition
    reps_delays = npr.choice(delays, simReps, replace=True)

    # structures

    # A
    fit_A = np.full(shape=(simReps, len(w_sig), len(w_noise)), fill_value=np.nan)
    te_A = fit_A.copy()
    dfi_A = fit_A.copy()
    fitsh = np.full((simReps, len(w_sig), len(w_noise), nShuff), np.nan)
    dish = fitsh.copy()
    dfish = fitsh.copy()
    fitsh_cond = np.full((simReps, len(w_sig), len(w_noise), nShuff), np.nan)
    dish_cond = fitsh_cond.copy()
    dfish_cond = fitsh_cond.copy()

    # B
    S_B = np.full(shape=(simReps, nTrials), fill_value=-1, dtype=int) # I changed this
    X_noise_B = np.full(shape=(simReps, simLen, nTrials), fill_value=np.nan) # I changed this
    X_signal_B = X_noise_B.copy()
    Y_B = X_noise_B.copy()
    fit_B = np.full(shape=(simReps, simLen, delay_max), fill_value=np.nan)
    te_B = fit_B.copy()
    dfi_B = fit_B.copy()

    print('\n Starting part A \n')

    # simulations
    for simIdx in range(simReps):
        print('Simulation number: ', simIdx, '\n')
        for sigIdx in range(len(w_sig)):
            for noiseIdx in range(len(w_noise)):
                # draw the stimulus value for each trial
                S = npr.randint(1, n_binsS + 1, size=nTrials)

                # simulate neural activity
                X_noise = npr.normal(0, stdX_noise, size=(simLen, nTrials)) # Noise time series

                X_signal = eps * npr.normal(0, stdX_noise, size=(simLen, nTrials)) # Infinitesimal signal to avoid binning 

                X_signal[stimWin[0]:stimWin[1],:] = np.tile(S, (stimWin[1]-stimWin[0], 1)) # Assigning Stimulus to Window

                # Adding multiplicative noise (we changed the dimension from nTrials to (simLen, nTrials) to have a different error for each time step)
                X_signal = X_signal * (1 + npr.normal(0, stdX_sig, size=(simLen, nTrials))) 

                # Time lagged single-trial input from the 2 dimensions of X and Y (we multpily everything by the weights, they mutiply only the stim/noise)
                X2Ysig = w_sig[sigIdx] * np.vstack((eps * npr.normal(0, stdX_noise, size=(reps_delays[simIdx], nTrials)),\
                            X_signal[0:len(X_signal)-reps_delays[simIdx],:]))
                X2Ynoise = w_noise[noiseIdx] * np.vstack((eps * npr.normal(0, stdX_noise, size=(reps_delays[simIdx], nTrials)),\
                            X_noise[0:len(X_signal)-reps_delays[simIdx],:]))

                # Computing Y + gaussian noise
                Y = X2Ysig + X2Ynoise + npr.normal(0,stdY,size=(simLen, nTrials))

                # Save for 2B
                if (sigIdx == 10) and (noiseIdx == 5):
                    S_B[simIdx] = S
                    X_noise_B[simIdx]= X_noise
                    X_signal_B[simIdx]= X_signal
                    Y_B[simIdx]= Y

                # First time point at which Y receives stim info from X
                t = stimWin[0] + reps_delays[simIdx]
                d = reps_delays[simIdx]

                # Discretize Neural Activity
                _, bin_edges = pd.cut(X_noise[t-d,:], n_binsX, retbins=True)
                bX_noise = np.digitize(X_noise[t-d, :], bins=bin_edges, right=True)

                _, bin_edges = pd.cut(X_signal[t-d,:], n_binsX, retbins=True)
                bX_sig = np.digitize(X_signal[t-d,:], bins=bin_edges, right=True)

                _, bin_edges = pd.cut(Y[t,:], n_binsY, retbins=True)
                bYt = np.digitize(Y[t,:], bins=bin_edges, right=True)

                _, bin_edges = pd.cut(Y[t-d,:], n_binsY, retbins=True)
                bYpast = np.digitize(Y[t-d,:], bins=bin_edges, right=True)

                bX = (bX_sig - 1) * n_binsX + bX_noise

                te_A[simIdx][sigIdx][noiseIdx], dfi_A[simIdx][sigIdx][noiseIdx], fit_A[simIdx][sigIdx][noiseIdx] = compute_FIT_TE_DFI(S, bX, bYt, bYpast)

                ######## SHUFFLING ########

                #   XSh = np.empty_like(bX)

                #   for shIdx in range(nShuff):
                #       for Ss in np.unique(S):
                #           idx = (S == Ss)
                #           tmpX = bX[idx]
                #           rIdx = npr.choice(np.sum(idx), np.sum(idx), replace=False)
                #           XSh[idx] = tmpX[rIdx]
                #   
                #   dish_cond, dfish_cond, fitsh_cond = compute_FIT_TE(S, XSh, bYt, bYpast)

                #   idx = npr.choice(nTrials, nTrials, replace=False)
                #   Ssh = S[idx]
                #   XSh = bX[idx]

                #   _, dfish, fitsh = compute_FIT_TE(Ssh, bX, bYt, bYpast)
                #   dish = DI_infToolBox(XSh, bYt, bYpast, 'naive', 0)

    print('\n Part A finished \n')

    print('\n Starting part B \n')

    # only 2B part
    for simIdx in range(simReps):
        print('Simulation number: ', simIdx, '\n')

        # Loop over time and delays
        for t in range(simLen):
            for d in range(delay_max):

                # Discretize Neural Activity
                _, bin_edges = pd.cut(X_noise_B[simIdx][t-d,:], n_binsX, retbins=True)
                bX_noise = np.digitize(X_noise_B[simIdx][t-d, :], bins=bin_edges, right=True)

                _, bin_edges = pd.cut(X_signal_B[simIdx][t-d,:], n_binsX, retbins=True)
                bX_sig = np.digitize(X_signal_B[simIdx][t-d,:], bins=bin_edges, right=True)

                _, bin_edges = pd.cut(Y_B[simIdx][t,:], n_binsY, retbins=True)
                bYt = np.digitize(Y_B[simIdx][t,:], bins=bin_edges, right=True)

                _, bin_edges = pd.cut(Y_B[simIdx][t-d,:], n_binsY, retbins=True)
                bYpast = np.digitize(Y_B[simIdx][t-d,:], bins=bin_edges, right=True)

                bX = (bX_sig - 1) * n_binsX + bX_noise

                te_B[simIdx][t][d], dfi_B[simIdx][t][d], fit_B[simIdx][t][d] = compute_FIT_TE_DFI(S_B[simIdx], bX, bYt, bYpast)

    return fit_A, te_A, dfi_A, fit_B, te_B, dfi_B