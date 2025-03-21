import numpy as np
import findtrial
import direct_method_v6a
import gaussian_method_v7_1_0
import additional_checks_v5

def build_parameters_structure_v6(R, opts, response_matrix_name, *varargin):
    # Initialization of the response matrix options
    opts['responseMatrixName'] = response_matrix_name

    if len(varargin) == 0:
        raise ValueError("No output option specified")

    # R -----------------------------------------------------------------------
    # Ensure that R is a NumPy array of type double (float64)
    if not isinstance(R, np.ndarray) or R.dtype != np.float64:
        raise TypeError('Responses must be of type double.')

    if R.ndim < 4:
        pars = {'Nc': R.shape[0], 'Ns': R.shape[2], 'size2ofR': R.shape[1]}
    else:
        raise ValueError("Response matrix can be at most a 3-D matrix.")

    # NT ----------------------------------------------------------------------
    if not isinstance(opts['nt'], np.float64):
        raise TypeError('opts.nt must be of type double.')

    # Handling nt as scalar or array
    if np.isscalar(opts['nt']):
        pars['Nt'] = np.ones(pars['Ns']) * opts['nt']
        pars['maxNt'] = opts['nt']
        pars['totNt'] = opts['nt'] * pars['Ns']
        pars['trials_indxes'] = np.arange(pars['totNt'])
    else:
        if len(opts['nt']) != pars['Ns']:
            raise ValueError(f"size({response_matrix_name},3) must match length(opts.nt). Try transposing nt.")
        pars['Nt'] = np.array(opts['nt'])
        pars['maxNt'] = np.max(pars['Nt'])
        pars['totNt'] = np.sum(pars['Nt'])
        pars['trials_indxes'] = findtrial(pars['Nt'], pars['maxNt'], pars['Ns'])

    # R and NT compatibility
    if pars['maxNt'] != pars['size2ofR']:
        raise ValueError(f"max(opts.nt) must be equal to size({response_matrix_name},2).")

    # TEST MODE (optional) ----------------------------------------------------
    pars['numberOfSpecifiedOptions'] = 0
    pars['testMode'] = False
    if 'testMode' in opts and opts['testMode']:
        pars['numberOfSpecifiedOptions'] += 1
        pars['testMode'] = True

    # METHOD ------------------------------------------------------------------
    pars['methodName'] = opts['method']
    method = opts['method'].lower()
    if method == 'dr':
        pars['methodFunc'] = direct_method_v6a
        pars['methodNum'] = 1
    elif method == 'gs':
        pars['methodFunc'] = gaussian_method_v7_1_0
        pars['methodNum'] = 2
    else:
        if pars['testMode']:
            pars['methodFunc'] = eval(opts['method'])
            pars['methodNum'] = -1
        else:
            raise ValueError(f"Method option {opts['method']} not found.")

    # BIAS --------------------------------------------------------------------
    bias_map = {
        'naive': 0,
        'qe': 1,
        'pt': 2,
        'gsb': 3
    }
    pars['biasCorrNum'] = bias_map.get(opts['bias'], -1)
    if pars['biasCorrNum'] == -1:
        pars['biasCorrFuncName'] = opts['bias']

    # BTSP (optional) ---------------------------------------------------------
    if 'btsp' in opts:
        pars['numberOfSpecifiedOptions'] += 1
        if not float(opts['btsp']).is_integer():
            raise ValueError('opts.btsp must be an integer.')
        pars['btsp'] = int(opts['btsp'])
    else:
        pars['btsp'] = 0

    # XTRP (optional) ---------------------------------------------------------
    if 'xtrp' in opts:
        pars['numberOfSpecifiedOptions'] += 1
        if not float(opts['xtrp']).is_integer() or opts['xtrp'] <= 0:
            raise ValueError('opts.xtrp must be a positive integer.')
        pars['xtrp'] = int(opts['xtrp'])
    else:
        pars['xtrp'] = 1

    # OUTPUT LIST -------------------------------------------------------------
    pars['Noutput'] = len(varargin)

    # Checking for specific output options
    where_options = {
        'hr': False, 'hrs': False, 'hlr': False, 'hirs': False,
        'hir': False, 'chir': False, 'hshr': False, 'hshrs': False,
        'hirsdef': False
    }
    
    for idx, option in enumerate(varargin):
        option_lower = option.lower()
        if option_lower in where_options:
            where_options[option_lower] = True

    # Populate the parsed output flags based on options
    pars.update(where_options)

    # Setting specific conditions based on Nc
    if pars['Nc'] == 1:
        pars['doHR'] = any([pars['hr'], pars['hlr'], pars['hir'], pars['chir'], pars['hshr']])
        pars['doHRS'] = any([pars['hrs'], pars['hirs'], pars['hiRS'], pars['hshrs']])
        pars['doHlR'] = False
        pars['doHlRS'] = False
        pars['doHiR'] = False
        pars['doHiRS'] = False
        pars['doChiR'] = False
        pars['doHshR'] = False
        pars['doHshRS'] = False
    else:
        pars['doHR'] = any([pars['hr']])
        pars['doHRS'] = any([pars['hrs']])
        pars['doHlR'] = any([pars['hlr']])
        pars['doHlRS'] = any([pars['hirs']])
        pars['doHiR'] = any([pars['hir']])
        pars['doHiRS'] = any([pars['hirsdef']])

    # ADDCHECKS (optional) ---------------------------------------------------
    if 'verbose' in opts and opts['verbose']:
        pars['numberOfSpecifiedOptions'] += 1
        pars['addChecks'] = True
    else:
        pars['addChecks'] = False

    if pars.get('addChecks', False):
        additional_checks_v5(R, pars, opts)

    return pars
