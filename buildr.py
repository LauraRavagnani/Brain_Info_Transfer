import numpy as np
import findtrial

# function to build response matrix
def build_r(s, RMAT):
    
    # Check if s is a vector
    if s.ndim != 1:
        raise ValueError('Stimulus array must be a vector.')
    
    # Check if length of s matches the number of trials in RMAT
    if len(s) != RMAT.shape[1]:
        raise ValueError('Each response-array must have the same length as the stimulus array.')

    # Get unique stimulus values
    edg_vec = np.unique(s)
    n_stm = len(edg_vec)
    
    # Count trials per stimulus
    nt = np.histogram(s, bins=np.append(edg_vec, edg_vec[-1] + 1))[0]
    
    # Check if there are stimuli with no corresponding response
    if np.min(nt) == 0:
        raise ValueError('One or more stimuli with no corresponding response.')
    
    # Find trials for each stimulus
    max_nt = np.max(nt)
    trl_lin_ind = findtrial(nt, max_nt, n_stm)
    
    # Sort the bin indices
    _, stm_linear_ind = np.sort(np.digitize(s, edg_vec, right=True) - 1)
    
    # Prepare the response matrix
    L = RMAT.shape[0]
    R = np.zeros((L, max_nt, n_stm))
    
    # Assign the sorted response data to the response matrix
    R[:, trl_lin_ind] = RMAT[:, stm_linear_ind]
    
    return R, nt