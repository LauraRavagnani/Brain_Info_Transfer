import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from itertools import product
import scipy.io as sio
import pickle as pkl
from info_utils import compute_FIT_TE
from parameters import par

# ----------------Data-Analysis-of-first-EEG-Dataset---------------------

# Loading Data
with open('data/new_data.pkl', 'rb') as f:
    data_raw = pkl.load(f) 
    data = np.transpose(data_raw, (1,2,0))

# Global Parameters
max_delay = par.max_delay_eeg
bins = par.n_binsX
timesteps = np.shape(data)[1]

# Channels to consider (Electrodes)
left_visual = 96
right_visual = 148 
left_motor = 184  
right_motor = 51

# Selecting channels
data_vis_left = data[left_visual,:,:]
data_vis_right = data[right_visual,:,:]
data_mot_left = data[left_motor,:,:]
data_mot_right = data[right_motor,:,:]

# Loading Stimulus Data
df = pd.read_excel("P1.xlsx", sheet_name=0)
exc_f = df.to_numpy()
S_original = exc_f[:,2]
unique_words, S_raw = np.unique(S_original, return_inverse=True)
S = S_raw + 1

# Structure to save FITs, TEs and their corrections
lr_fit_vis = np.full((timesteps-max_delay, max_delay), np.nan)
lrl_fit_vis = lr_fit_vis.copy()
lrq_fit_vis = lr_fit_vis.copy()
lr_te_vis = lr_fit_vis.copy()
lrl_te_vis = lr_fit_vis.copy()
lrq_te_vis = lr_fit_vis.copy()
rl_fit_vis = lr_fit_vis.copy()
rll_fit_vis = lr_fit_vis.copy()
rlq_fit_vis = lr_fit_vis.copy()
rl_te_vis = lr_fit_vis.copy()
rll_te_vis = lr_fit_vis.copy()
rlq_te_vis = lr_fit_vis.copy()
lr_fit_mot = lr_fit_vis.copy()
lrl_fit_mot = lr_fit_vis.copy()
lrq_fit_mot = lr_fit_vis.copy()
lr_te_mot = lr_fit_vis.copy()
lrl_te_mot = lr_fit_vis.copy()
lrq_te_mot = lr_fit_vis.copy()
rl_fit_mot = lr_fit_vis.copy()
rll_fit_mot = lr_fit_vis.copy()
rlq_fit_mot = lr_fit_vis.copy()
rl_te_mot = lr_fit_vis.copy()
rll_te_mot = lr_fit_vis.copy()
rlq_te_mot = lr_fit_vis.copy()

# Function to use in each task of the parallelization
def inner_cycle (temp, d):

    t = temp + max_delay + 1

    # Discretizing Neural Signals
    # Visual

    L_sig_vis = pd.qcut(data_vis_left[t,:], bins, labels=range(1,bins+1)).astype(int)
    L_sig_vis_p = pd.qcut(data_vis_left[t-d,:], bins, labels=range(1,bins+1)).astype(int)
    R_sig_vis = pd.qcut(data_vis_right[t,:], bins, labels=range(1,bins+1)).astype(int)
    R_sig_vis_p = pd.qcut(data_vis_right[t-d,:], bins, labels=range(1,bins+1)).astype(int)

    # Motor
    L_sig_mot = pd.qcut(data_mot_left[t,:], bins, labels=range(1,bins+1)).astype(int)
    L_sig_mot_p = pd.qcut(data_mot_left[t-d,:], bins, labels=range(1,bins+1)).astype(int)
    R_sig_mot = pd.qcut(data_mot_right[t,:], bins, labels=range(1,bins+1)).astype(int)
    R_sig_mot_p = pd.qcut(data_mot_right[t-d,:], bins, labels=range(1,bins+1)).astype(int)

    # Left to right Visual

    lr_te_vis, lr_fit_vis, lrq_te_vis, lrl_te_vis, lrq_fit_vis, lrl_fit_vis = compute_FIT_TE(S, L_sig_vis_p, R_sig_vis, R_sig_vis_p)    

    # Right to left Visual

    rl_te_vis, rl_fit_vis, rlq_te_vis, rll_te_vis, rlq_fit_vis, rll_fit_vis = compute_FIT_TE(S, R_sig_vis_p, L_sig_vis, L_sig_vis_p)    

    # Left visual to right motor

    lr_te_mot, lr_fit_mot, lrq_te_mot, lrl_te_mot, lrq_fit_mot, lrl_fit_mot = compute_FIT_TE(S, L_sig_vis_p, R_sig_mot, R_sig_mot_p) 

    # Right visual to left motor

    rl_te_mot, rl_fit_mot, rlq_te_mot, rll_te_mot, rlq_fit_mot, rll_fit_mot = compute_FIT_TE(S, R_sig_vis_p, L_sig_mot, L_sig_mot_p)    
    
    return temp, d, lr_te_vis, lr_fit_vis, lrq_te_vis, lrl_te_vis, lrq_fit_vis, lrl_fit_vis, rl_te_vis, rl_fit_vis, rlq_te_vis, rll_te_vis, rlq_fit_vis, rll_fit_vis, lr_te_mot, lr_fit_mot, lrq_te_mot, lrl_te_mot, lrq_fit_mot, lrl_fit_mot, rl_te_mot, rl_fit_mot, rlq_te_mot, rll_te_mot, rlq_fit_mot, rll_fit_mot
                
# Iterable to use in parallelization
index_iter = product(range(timesteps-max_delay-1), range(max_delay))

# Computing the parallelized for loops
results = Parallel(n_jobs=-1,verbose=10)(
delayed(inner_cycle)(*pair) for pair in index_iter
)
    
# Unloading the results
for res in results:
    t, d, lrtevis, lrfitvis, lrqtevis, lrltevis, lrqfitvis, lrlfitvis, rltevis, rlfitvis, rlqtevis, rlltevis, rlqfitvis, rllfitvis, lrtemot, lrfitmot, lrqtemot, lrltemot, lrqfitmot, lrlfitmot, rltemot, rlfitmot, rlqtemot, rlltemot, rlqfitmot, rllfitmot = res
    lr_te_vis[t, d] = lrtevis
    lr_fit_vis[t, d] = lrfitvis
    lrq_te_vis[t, d] = lrqtevis
    lrl_te_vis[t, d] = lrltevis
    lrq_fit_vis[t, d] = lrqfitvis
    lrl_fit_vis[t, d] = lrlfitvis
    rl_te_vis[t, d] = rltevis
    rl_fit_vis[t, d] = rlfitvis
    rlq_te_vis[t, d] = rlqtevis
    rll_te_vis[t, d] = rlltevis
    rlq_fit_vis[t, d] = rlqfitvis
    rll_fit_vis[t, d] = rllfitvis
    lr_te_mot[t, d] = lrtemot
    lr_fit_mot[t, d] = lrfitmot
    lrq_te_mot[t, d] = lrqtemot
    lrl_te_mot[t, d] = lrltemot
    lrq_fit_mot[t, d] = lrqfitmot
    lrl_fit_mot[t, d] = lrlfitmot
    rl_te_mot[t, d] = rltemot
    rl_fit_mot[t, d] = rlfitmot
    rlq_te_mot[t, d] = rlqtemot
    rll_te_mot[t, d] = rlltemot
    rlq_fit_mot[t, d] = rlqfitmot
    rll_fit_mot[t, d] = rllfitmot

# Saving results
with open('results_correct_800_all.pkl', 'wb') as f:
    pkl.dump([lr_te_vis, lr_fit_vis, lrq_te_vis, lrl_te_vis, lrq_fit_vis, lrl_fit_vis, 
              rl_te_vis, rl_fit_vis, rlq_te_vis, rll_te_vis, rlq_fit_vis, rll_fit_vis, 
              lr_te_mot, lr_fit_mot, lrq_te_mot, lrl_te_mot, lrq_fit_mot, lrl_fit_mot, 
              rl_te_mot, rl_fit_mot, rlq_te_mot, rll_te_mot, rlq_fit_mot, rll_fit_mot], f)

         
