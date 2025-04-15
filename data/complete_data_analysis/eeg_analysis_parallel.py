import pandas as pd
import h5py as h5
import numpy as np
from joblib import Parallel, delayed
from itertools import product
import pickle
from info_utils import compute_FIT_TE

# ----------------Data-Analysis-of-first-EEG-Dataset---------------------

# Function that helps parallelizing
def inner_cycle (temp, d):

    t = temp + max_delay + 1

    # Discretizing Neural Signals

    leeg_d = pd.qcut(leeg[t,:], bins, labels=range(1,bins+1)).astype(int)
    leegp = pd.qcut(leeg[t-d,:], bins, labels=range(1,bins+1)).astype(int)

    reeg_d = pd.qcut(reeg[t,:], bins, labels=range(1,bins+1)).astype(int)
    reegp = pd.qcut(reeg[t-d,:], bins, labels=range(1,bins+1)).astype(int)

    # Left eye visibility
    pastX = reegp
    Y = leeg_d
    pastY = leegp

    # LOT to ROT transfer
    left_vals_LR = compute_FIT_TE(LES, pastXR, YR, pastYR)

    # ROT to LOT transfer
    left_vals_RL = compute_FIT_TE(LES, pastX, Y, pastY)

    # Right Eye visibility
    pastXR = leegp
    YR = reeg_d
    pastYR = reegp
                 
    # LOT to ROT transfer
    right_vals_LR = compute_FIT_TE(RES, pastXR, YR, pastYR)

    # ROT to LOT transfer
    right_vals_RL = compute_FIT_TE(RES, pastX, Y, pastY)

    return temp, d, right_vals_LR, right_vals_RL, left_vals_LR, left_vals_RL

# -------- MAIN ANALYSIS ROUTINE ---------

# global parameters
max_delay = 60
bins = 3
time = 300 # timesteps to compute fit

# metadata
meta_path = '/mnt/FIT_project/EEGdata/metadata.mat'
metadata = h5.File(meta_path,'r')

timesteps = len(metadata['time']) # Number of timesteps
num_files = int(np.array(metadata['Ns']).item()) # Number of files

# structures to save results

right_LR_te = np.full((num_files, timesteps, max_delay), np.nan)
right_LR_fit = right_LR_te.copy()
right_LR_TEQe = right_LR_te.copy()
right_LR_TELe = right_LR_te.copy()
right_LR_FITQe = right_LR_te.copy()
right_LR_FITLe = right_LR_te.copy()

right_RL_te = right_LR_te.copy()
right_RL_fit = right_LR_te.copy()
right_RL_TEQe = right_LR_te.copy()
right_RL_TELe = right_LR_te.copy()
right_RL_FITQe = right_LR_te.copy()
right_RL_FITLe = right_LR_te.copy()

left_LR_te = right_LR_te.copy() 
left_LR_fit = right_LR_te.copy()
left_LR_TEQe = right_LR_te.copy()
left_LR_TELe = right_LR_te.copy()
left_LR_FITQe = right_LR_te.copy()
left_LR_FITLe = right_LR_te.copy()

left_RL_te = right_LR_te.copy()
left_RL_fit = right_LR_te.copy()
left_RL_TEQe = right_LR_te.copy()
left_RL_TELe = right_LR_te.copy()
left_RL_FITQe = right_LR_te.copy()
left_RL_FITLe = right_LR_te.copy()

for file in range(num_files):
    print('File ',file+1)
    file_path = '/mnt/FIT_project/EEGdata/data_s{0}.mat'.format(file+1)
    data = h5.File(file_path,'r')

    eeg_prel = np.array(data['feeg']) # eeg data
    eye_visib_prel = np.array(data['eyebubs'])
    ntrials = np.shape(eeg_prel)[2]

    if ((np.shape(eeg_prel)[2] - 1) %4==0):
        ntrials = np.shape(eeg_prel)[2] - 1
    elif ((np.shape(eeg_prel)[2] - 2) %4==0):
        ntrials = np.shape(eeg_prel)[2] - 2
    elif ((np.shape(eeg_prel)[2] - 3) %4==0):
        ntrials = np.shape(eeg_prel)[2] - 3

    eeg = eeg_prel[:,:,:ntrials]
    eye_visib = eye_visib_prel[:,:ntrials] # Eye region visibility, related to stimulus

    left_electrode = np.array(data['LEmaxMI']).item() # Left electrode with max MI
    right_electrode = np.array(data['REmaxMI']).item() # Right electrode with max MI
    leeg = eeg[int(left_electrode-1),:,:]
    reeg = eeg[int(right_electrode-1),:,:]

    left_eye_v = eye_visib[0,:]
    right_eye_v = eye_visib[1,:]

    # Discretizing stimulus
    _, bin_edges = pd.qcut(left_eye_v, bins, retbins=True)
    LES = np.digitize(left_eye_v, bins=bin_edges, right=False)
    _, bin_edges = pd.qcut(right_eye_v, bins, retbins=True)
    RES = np.digitize(right_eye_v, bins=bin_edges, right=False)

    # Parallel computation on times and delays
    index_iter = product(range(time), range(max_delay))
    results = Parallel(n_jobs=-1,verbose=10)(
    delayed(inner_cycle)(*pair) for pair in index_iter
    )
    
    # Saving results of parallel computation
    for res in results:
        t, d, right_vals_LR, right_vals_RL, left_vals_LR, left_vals_RL = res

        # te, dfi, fit, TEQe, TELe, FITQe, FITLe
        right_LR_te[file, t, d] = right_vals_LR[0]
        right_LR_fit[file, t, d] = right_vals_LR[1]
        right_LR_TEQe[file, t, d] = right_vals_LR[2]
        right_LR_TELe[file, t, d] = right_vals_LR[3]
        right_LR_FITQe[file, t, d] = right_vals_LR[4]
        right_LR_FITLe[file, t, d] = right_vals_LR[5]

        right_RL_te[file, t, d] = right_vals_RL[0]
        right_RL_fit[file, t, d] = right_vals_RL[1]
        right_RL_TEQe[file, t, d] = right_vals_RL[2]
        right_RL_TELe[file, t, d] = right_vals_RL[3]
        right_RL_FITQe[file, t, d] = right_vals_RL[4]
        right_RL_FITLe[file, t, d] = right_vals_RL[5]

        left_LR_te[file, t, d] = left_vals_LR[0]
        left_LR_fit[file, t, d] = left_vals_LR[1]
        left_LR_TEQe[file, t, d] = left_vals_LR[2]
        left_LR_TELe[file, t, d] = left_vals_LR[3]
        left_LR_FITQe[file, t, d] = left_vals_LR[4]
        left_LR_FITLe[file, t, d] = left_vals_LR[5]

        left_RL_te[file, t, d] = left_vals_RL[0]
        left_RL_fit[file, t, d] = left_vals_RL[1]
        left_RL_TEQe[file, t, d] = left_vals_RL[2]
        left_RL_TELe[file, t, d] = left_vals_RL[3]
        left_RL_FITQe[file, t, d] = left_vals_RL[4]
        left_RL_FITLe[file, t, d] = left_vals_RL[5]

left_LR_data = [left_LR_te, left_LR_fit, left_LR_TEQe, left_LR_TELe, left_LR_FITQe, left_LR_FITLe]
left_RL_data = [left_RL_te, left_RL_fit, left_RL_TEQe, left_RL_TELe, left_RL_FITQe, left_RL_FITLe]
right_LR_data = [right_LR_te, right_LR_fit, right_LR_TEQe, right_LR_TELe, right_LR_FITQe, right_LR_FITLe]
right_RL_data = [right_RL_te, right_RL_fit, right_RL_TEQe, right_RL_TELe, right_RL_FITQe, right_RL_FITLe]

# save left and right eye values in both directions
left_LR_file = 'eeg_left_LR_values.pkl'
right_LR_file = 'eeg_right_LR_values.pkl'
left_RL_file = 'eeg_left_RL_values.pkl'
right_RL_file = 'eeg_right_RL_values.pkl'

with open(left_LR_file, 'wb') as file:
    pickle.dump(left_LR_data, file)

with open(right_LR_file, 'wb') as file:
    pickle.dump(right_LR_data, file)

with open(left_RL_file, 'wb') as file:
    pickle.dump(left_RL_data, file)

with open(right_RL_file, 'wb') as file:
    pickle.dump(right_RL_data, file)
