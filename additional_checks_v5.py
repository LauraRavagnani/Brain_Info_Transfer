import numpy as np

def additional_checks_v5(R, Par, opts):
    print('===== OPTIONS SUMMARY =========================================')
    print(f'  - method: {Par["methodName"]}')
    print(f'  - bias correction: {opts["bias"]}')
    print(f'  - btsp repetitions: {Par["btsp"]}')
    print('===== ADDITIONAL WARNINGS =====================================')

    warning_flag = False
    
    # Output options check
    NmatchingOutputOpts = sum([
        Par.get(key, 0) for key in ['doHR', 'doHRS', 'doHlR', 'doHlRS', 'doHiR', 'doHiRS', 'doChiR', 'doHshR']
    ])
    
    if Par["Noutput"] > NmatchingOutputOpts:
        warning_flag = True
        print("Warning: Unknown selection in the output options list.")
    
    # Unused fields check
    if len(opts) > Par["numberOfSpecifiedOptions"] + 4:
        warning_flag = True
        print("Warning: Number of fields in the options structure exceeds specified options.")
    
    # Binning checks (for 'dr' method)
    if Par["methodName"] == 'dr':
        mask = np.ones(Par["Nt"])  # Adjust mask as needed
        
        # Check if response values are integers
        if np.any(np.abs(R[:, mask] - np.floor(np.abs(R[:, mask]))) != 0):
            raise ValueError(f"Method '{Par['methodName']}' requires responses to be discretized into non-negative integers.")
        
        # Check if all cells are binned the same number of times
        Nb = np.max(R[:, mask], axis=1)
        if len(np.unique(Nb)) != 1:
            warning_flag = True
            print("Warning: The number of bins for discretization differs across responses.")
        
        # Check if all bin values from 0 to Nb are included
        for c in range(Par["Nc"]):
            unique_R = np.unique(R[c, mask])
            for bin_val in range(1, Nb[c] + 1):
                if bin_val not in unique_R:
                    warning_flag = True
                    print(f"Warning: Trials do not include all values from 0 to {Nb[c]} for response {c}.")
    
    # Non-trial consistency check
    if np.any(~Par["trials_indxes"]):
        if len(np.unique(R[:, ~Par["trials_indxes"]])) != 1:
            print("Warning: Non-trials have non-unique values.")
    
    # Bias correction checks
    if Par["biasCorrNum"] == 2 and Par["methodName"].lower() != 'dr':
        warning_flag = True
        print("Warning: Bias correction 'pt' can only be used with method 'dr'. Returning naive estimates.")
    
    if Par["biasCorrNum"] == -1 and not Par["testMode"] and Par["methodName"].lower() != 'dr':
        raise ValueError("User-defined bias corrections can only be used with method 'dr'. Returning naive estimates.")
    
    if Par["biasCorrNum"] == 3 and not Par["testMode"] and Par["methodName"].lower() != 'gs':
        raise ValueError("Bias correction 'gsb' can only be used with method 'gs'.")
    
    # Check output options compatibility with method
    if Par["doHiR"]:
        warning_flag = H_ind_checks('HiR', Par, warning_flag)
    if Par["doHiRS"]:
        warning_flag = H_ind_checks('HiRS', Par, warning_flag)
    if Par["doChiR"]:
        warning_flag = H_ind_checks('ChiR', Par, warning_flag)
    
    # Custom bias correction function check
    if Par["biasCorrNum"] == -1:
        check_biasCorr_func(Par)

    if not warning_flag:
        print("No errors or warnings.")
    
    print("===============================================================")

def H_ind_checks(output_opt, Par, warning_flag):
    if Par["methodName"].lower() != 'dr':
        warning_flag = True
        print(f"Warning: Only method 'dr' can be used with output option '{output_opt}'. Returning NaN.")
    
    if Par["biasCorrNum"] not in [0, 1]:
        warning_flag = True
        print(f"Warning: Only bias correction 'qe' can be used with output option '{output_opt}'. Returning naive estimate.")
    
    return warning_flag

def check_biasCorr_func(Par):
    bias_corr_func = globals().get(Par["biasCorrFuncName"])
    
    x = np.random.rand(100) * 10
    bias = bias_corr_func(x)
    
    if not np.isscalar(bias):
        raise ValueError(f"Non-scalar bias estimate returned by function '{Par['biasCorrFuncName']}'")
    if bias < 0:
        raise ValueError(f"Negative bias estimate returned by function '{Par['biasCorrFuncName']}'")
    if bias == 0:
        print(f"Warning: Null bias estimate returned by function '{Par['biasCorrFuncName']}'")
