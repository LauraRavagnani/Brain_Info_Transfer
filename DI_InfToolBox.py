import numpy as np
from scipy.stats import entropy
import compute_entropy
import buildr

# function to compute mutual information
def information(R, opts, *output_list):

    L, T, S = R.shape
    nt = opts.get("nt", T if isinstance(opts.get("nt"), int) else np.array(opts.get("nt")))
    method = opts.get("method", "dr")
    bias = opts.get("bias", "naive")
    btsp = opts.get("btsp", 0)
    xtrp = opts.get("xtrp", 0)
    verbose = opts.get("verbose", False)

    if verbose:
        print(f"Method: {method}, Bias: {bias}, Bootstrap: {btsp}, Extrapolation: {xtrp}")

    # Compute entropies
    HR = compute_entropy(R, nt, method, bias)
    HRS = compute_conditional_entropy(R, nt, method, bias)
    
    # Compute requested information measures
    results = []
    for measure in output_list:
        if measure.lower() == "i":
            results.append(HR - HRS)
        elif measure.lower() == "ish":
            HiRS, HshRS = shuffle_entropy(R, nt, method, bias)
            results.append(HR - HiRS + HshRS - HRS)
        elif measure.lower() == "ix":
            HlR = compute_entropy_limited(R, nt, method, bias)
            results.append(HlR - HR)
        elif measure.lower() == "ilin":
            HiRS = compute_hiRS(R, nt, method, bias)
            results.append(HlR - HiRS)
        elif measure.lower() == "syn":
            HiRS = compute_hiRS(R, nt, method, bias)
            HlR = compute_entropy_limited(R, nt, method, bias)
            results.append(HR - HRS - HlR + HiRS)
        elif measure.lower() == "synsh":
            HshRS = shuffle_entropy(R, nt, method, bias)[1]
            HlR = compute_entropy_limited(R, nt, method, bias)
            results.append(HR + HshRS - HRS - HlR)
        else:
            raise ValueError(f"Unknown output option: {measure}")

    return tuple(results)

def compute_conditional_entropy(R, nt, method, bias):
    """ Compute conditional entropy H(R|S). """
    return entropy(R.flatten()) - np.mean([entropy(R[:, :, s].flatten()) for s in range(R.shape[2])])

def shuffle_entropy(R, nt, method, bias):
    """ Compute shuffled entropy for bias correction. """
    shuffled_R = np.copy(R)
    np.random.shuffle(shuffled_R)
    return entropy(shuffled_R.flatten()), compute_conditional_entropy(shuffled_R, nt, method, bias)

def compute_entropy_limited(R, nt, method, bias):
    """ Compute entropy with limited trials. """
    return entropy(R[:, :nt.min(), :].flatten())

def compute_hiRS(R, nt, method, bias):
    """ Compute HiRS for bias correction. """
    return entropy(R[:, :nt.min(), :].flatten())

# Example usage
R = np.random.rand(2, 10, 3)  # Example response matrix
opts = {"nt": [10, 10, 7], "method": "dr", "bias": "naive", "verbose": True}
I, Ish = information(R, opts, "I", "Ish")
print("I:", I, "Ish:", Ish)


# function to compute differential information as
# DI(X,Y) = I(hY,X ; Y) - I(hY, Y)
def di_inf_toolbox(X, Y, hY, bias, compute_shuffled, information):

    # compute I(hY,X ; Y)
    R, nt = buildr(Y, np.concatenate([hY, X]))
    di_sh = None
    
    opt = {'nt': nt, 'method': 'dr', 'bias': bias, 'trperm': 0}
    
    if compute_shuffled:
        IhyXY, IhyXYsh = information(R, opt, 'I', 'Ish')
        
        # compute I(hY ; Y)
        R, nt = buildr(Y, hY)
        opt['nt'] = nt
        IhyY, IhyYsh = information(R, opt, 'I', 'Ish')
        
        di = IhyXY - IhyY
        di_sh = IhyXYsh - IhyYsh
    else:
        IhyXY = information(R, opt, 'I')
        
        # compute I(hY ; Y)
        R, nt = buildr(Y, hY)
        opt['nt'] = nt
        IhyY = information(R, opt, 'I')
        
        di = IhyXY - IhyY
    
    return di, di_sh
