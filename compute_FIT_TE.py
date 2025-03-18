# This function computes the Feature-specific Information Transfer (FIT)
# Transfer Entropy (TE) and directed feature information (DFI) 
# Quadratic extrapolation bias correction is available for TE and FI

# input: (N = number of experimental trials)
# feature = discrete feature value (1 x N)
# X = discrete past activity of the sender X_past (1 x N)
# Y = discrete present activity of the receiver Y_pres (1 x N)
# hY = discrete past activity of the receiver Y_past (1 x N)
# doQe = set to 1 if you want to use the Quadratic Extrapolation technique to correct for the limited sampling bias (by default is 0)
# xtrap = number of data resamplings when estimating FIT and TE on datasets with N/2 and N/4 trials (if no value is specified, the default is xtrap = 20# 
# output:
# te = transfer entropy value (from X to Y; see Schreiber T. (2000) Phys Rev Letters)
# dfi = DFI value (see Ince R., et al. (2015) Sci Reports)
# FIT = Feature-specific Information Transfer value (from X to Y about S)
# TEQe = bias-corrected TE value using quadratic extrapolation
# TELe = bias-corrected TE value using linear extrapolation 
# FITQe = bias-corrected FIT value using quadratic extrapolation
# FITLe = bias-corrected FIT value using linear extrapolation

# Build the two four-variables probability distributions needed to compute FIT

import probability_dist
import compute_SUI
import TE
import DFI

import numpy as np
import numpy.random as npr

def compute_FIT_TE(feature, X, Y, hY, xtrap=20):
    # Build the two four-variables probability distributions needed to compute FIT
    pXYhYS = probability_dist(X, Y, hY, feature)    # probability distribution for the PID with (Xp, Yp, Yt) as sources and S as target
    pXShYY = probability_dist(X, feature, hY, Y)    # probability distribution for the PID with (Xp, Yp, S) as sources and Yt as target

    # Compute the two FIT atoms and FIT
    sui_S = compute_SUI(pXYhYS)
    sui_Y = compute_SUI(pXShYY)

    fit = np.min(sui_S,sui_Y)

    # Compute TE
    te = TE(pXYhYS)

    # Compute DFI
    dfi = DFI(pXYhYS)

    # Compute quadratic extrapolation bias correction for FIT and TE
    fit_all = fit
    te_all = te

    FIT2 = np.zeros(xtrap)
    FIT4 = np.zeros(xtrap)
    TE2 = np.zeros(xtrap)
    TE4 = np.zeros(xtrap)

    for xIdx in xtrap:
        # Partition trials into subsamples of N/2 and N/4
        numberOfTrials = len(X)

        rIdx = npr.choice(numberOfTrials, numberOfTrials, replace=False)

        idx21 = rIdx[ : int(numberOfTrials/2)]
        idx22 = rIdx[int(numberOfTrials/2) : ]

        idx41 = rIdx[ : int(numberOfTrials/4)]
        idx42 = rIdx[int(numberOfTrials/4) : int(numberOfTrials/2)]
        idx43 = rIdx[int(numberOfTrials/2) : int(3 * numberOfTrials/4)]
        idx44 = rIdx[int(3 * numberOfTrials/4) : ]

        X21 = X[idx21]
        X22 = X[idx22]

        Y21 = Y[idx21]
        Y22 = Y[idx22]

        hY21 = hY[idx21]
        hY22 = hY[idx22]

        feat21 = feature[idx21]
        feat22 = feature[idx22]

        X41 = X[idx41]
        X42 = X[idx42]
        X43 = X[idx43]
        X44 = X[idx44]

        Y41 = Y[idx41]
        Y42 = Y[idx42]
        Y43 = Y[idx43]
        Y44 = Y[idx44]

        hY41 = hY[idx41]
        hY42 = hY[idx42]
        hY43 = hY[idx43]
        hY44 = hY[idx44]

        feat41 = feature[idx41]
        feat42 = feature[idx42]
        feat43 = feature[idx43]
        feat44 = feature[idx44]

        pXYhYS21 = probability_dist(X21, Y21, hY21, feat21)
        pXShYY21 = probability_dist(X21, feat21, hY21, Y21)
        SUI21S = compute_SUI(pXYhYS21)
        SUI21Y = compute_SUI(pXShYY21)
        FIT21 = min(SUI21S, SUI21Y)
        TE21 = TE(pXYhYS21)
        
        pXYhYS22 = probability_dist(X22, Y22, hY22, feat22)
        pXShYY22 = probability_dist(X22, feat22, hY22, Y22)
        SUI22S = compute_SUI(pXYhYS22)
        SUI22Y = compute_SUI(pXShYY22)
        FIT22 = min(SUI22S, SUI22Y)
        TE22 = TE(pXYhYS22)

        pXYhYS41 = probability_dist(X41, Y41, hY41, feat41)
        pXShYY41 = probability_dist(X41, feat41, hY41, Y41)
        SUI41S = compute_SUI(pXYhYS41)
        SUI41Y = compute_SUI(pXShYY41)
        FIT41 = min(SUI41S, SUI41Y)
        TE41 = TE(pXYhYS41)

        pXYhYS42 = probability_dist(X42, Y42, hY42, feat42)
        pXShYY42 = probability_dist(X42, feat42, hY42, Y42)
        SUI42S = compute_SUI(pXYhYS42)
        SUI42Y = compute_SUI(pXShYY42)
        FIT42 = min(SUI42S, SUI42Y)
        TE42 = TE(pXYhYS42)

        pXYhYS43 = probability_dist(X43, Y43, hY43, feat43)
        pXShYY43 = probability_dist(X43, feat43, hY43, Y43)
        SUI43S = compute_SUI(pXYhYS43)
        SUI43Y = compute_SUI(pXShYY43)
        FIT43 = min(SUI43S, SUI43Y)
        TE43 = TE(pXYhYS43)

        pXYhYS44 = probability_dist(X44, Y44, hY44, feat44)
        pXShYY44 = probability_dist(X44, feat44, hY44, Y44)
        SUI44S = compute_SUI(pXYhYS44)
        SUI44Y = compute_SUI(pXShYY44)
        FIT44 = min(SUI44S, SUI44Y)
        TE44 = TE(pXYhYS44)

        FIT2[xIdx] = (FIT21 + FIT22) / 2
        FIT4(xIdx) = (FIT41 + FIT42 + FIT43 + FIT44) / 4
        TE2(xIdx) = (TE21 + TE22) / 2
        TE4(xIdx) = (TE41 + TE42 + TE43 + TE44) / 4

        #   x = [1/length(X41) 1/length(X21) 1/length(X)];
        #   y = [mean(FIT4) mean(FIT2) FITAll];
    #   
        #   p2 = polyfit(x, y, 2); 
        #   p1 = polyfit(x, y, 1);  
        #   FITQe = p2(3);
        #   FITLe = p1(2);
        #   
        #   y = [mean(TE4) mean(TE2) TEAll];
    #   
        #   p2 = polyfit(x, y, 2); 
        #   p1 = polyfit(x, y, 1);  
        #   TEQe = p2(3);
        #   TELe = p1(2);