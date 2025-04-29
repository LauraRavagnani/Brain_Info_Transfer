import numpy as np

class params():
    def __init__(self, n_subjects, nTrials_per_stim, nShuff, n_test, w_sig, w_noise, stdX_noise, stdY, ratio, simLen, stimWin, delays, delay_max, n_binsS, n_binsX, n_binsY, eps, snr, max_delay_eeg):
        # number of subjects
        self.n_subjects = n_subjects

        # number of trials for each stimulus value
        self.nTrials_per_stim = nTrials_per_stim

        # number of shuffling for significance test
        self.nShuff = nShuff

        # number of weights to test the significance
        self.n_test = n_test

        # signal and noise weights
        self.w_sig = w_sig
        self.w_noise = w_noise

        # std for X_noise, Y, X_sig gaussian noise
        self.stdX_noise = stdX_noise
        self.stdY = stdY
        self.ratio = ratio

        # simulation length in unit of 10 ms
        self.simLen = simLen

        # stimulus window in unit of 10 ms
        self.stimWin = stimWin

        # delays values for simulations
        self.delays = delays

        # maximum delay for fig2B in unit of 10 ms
        self.delay_max = delay_max

        # number of bins for X, Y, S
        self.n_binsS = n_binsS
        self.n_binsX = n_binsX
        self.n_binsY = n_binsY

        # infinitesimal constant
        self.eps = eps

        # signal to noise ratio for fig 2E
        self.snr = snr

        # maximum delay for eeg analysis
        self.max_delay_eeg = max_delay_eeg


par = params(
    n_subjects = 50,
    nTrials_per_stim = 500,
    nShuff = 100,
    n_test = 4,
    w_sig = np.linspace(0, 1, num=11),
    w_noise = np.linspace(0, 1, num=11),
    stdX_noise = 2,
    stdY = 2, 
    ratio = 0.2, 
    simLen = 50,
    stimWin = [20, 25],
    delays = np.linspace(4, 6, num=3, dtype=int),
    delay_max = 10,
    n_binsS = 4,
    n_binsX = 3,
    n_binsY = 3,
    eps = 1e-52,  
    snr = np.arange(0.05, 1.05, 0.05),
    max_delay_eeg = 60
)
