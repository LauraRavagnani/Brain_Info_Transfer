import numpy as np

# function to compute probability distribution
def get_joint_prob_distr(target, source_var1, source_var2, source_var3):

    assert np.min(source_var1) > 0, "Invalid values in source variable 1"
    assert np.min(source_var2) > 0, "Invalid values in source variable 2"
    assert np.min(source_var3) > 0, "Invalid values in source variable 3"
    assert np.min(target) > 0, "Invalid values in target"
    
    count = len(source_var1)
    
    print('aaa')

    # compute probabilities from (multi-dim) histogram frequencies
    result, _ = np.histogramdd(
        np.vstack([source_var1, source_var2, source_var3, target]).T, 
        bins=[np.max(source_var1), np.max(source_var2), np.max(source_var3), np.max(target)]
    )
    
    return result / count