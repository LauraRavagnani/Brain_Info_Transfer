import numpy as np

def find_trial(Nt, maxNt, Ns):
    # Generate array from 1 to maxNt
    one2maxNt = np.arange(1, maxNt + 1)
    
    # Create a boolean matrix where each column corresponds to whether the trial index
    # is less than or equal to the corresponding value in Nt
    trial_mask = (one2maxNt[None, :] <= Nt[:, None])
    
    # Find the indices where the condition is True
    ind = np.nonzero(trial_mask.flatten())[0] + 1  # +1 to match MATLAB's 1-indexing
    
    return ind