import quadratic_extrapolation_v2

def xtrploop(R, pars):
    """
    Loops over iterations of the extrapolation procedure.
    
    Parameters:
        R: Input data (type depends on your application)
        pars: An object (or dict-like) with at least the attribute 'xtrp'
        
    Returns:
        A tuple of 9 elements: (HR, HRS, HlR, HlRS, HiR, HiRS, ChiR, HshR, HshRS)
        
    Note:
        The function quadratic_extrapolation_v2(R, pars) must be defined elsewhere.
    """
    # If there is only one extrapolation, call the function directly.
    if pars.xtrp == 1:
        return quadratic_extrapolation_v2(R, pars)
    
    # Otherwise, initialize accumulators for each output.
    HR = HRS = HlR = HlRS = HiR = HiRS = ChiR = HshR = HshRS = 0
    
    # Run quadratic_extrapolation_v2 pars.xtrp times and average the results.
    for _ in range(pars.xtrp):
        HRtmp, HRStmp, HlRtmp, HlRStmp, HiRtmp, HiRStmp, ChiRtmp, HshRtmp, HshRStmp = quadratic_extrapolation_v2(R, pars)
        HR    += HRtmp    / pars.xtrp
        HRS   += HRStmp   / pars.xtrp
        HlR   += HlRtmp   / pars.xtrp
        HlRS  += HlRStmp  / pars.xtrp
        HiR   += HiRtmp   / pars.xtrp
        HiRS  += HiRStmp  / pars.xtrp
        ChiR  += ChiRtmp  / pars.xtrp
        HshR  += HshRtmp  / pars.xtrp
        HshRS += HshRStmp / pars.xtrp

    return HR, HRS, HlR, HlRS, HiR, HiRS, ChiR, HshR, HshRS
