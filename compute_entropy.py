import numpy as np
import xtrploop
import build_parameters_structure_v6

def compute_entropy(R, Opt, *args):
    if len(args) < 1:
        raise ValueError("Insufficient arguments provided.")
    
    response_matrix_name = "R"  # Python lacks 'inputname' equivalent, so assigning manually
    
    pars = Opt.get('pars') if 'pars' in Opt else build_parameters_structure_v6(R, Opt, response_matrix_name, *args)
    
    HRS = np.zeros((pars['Ns'], (pars['btsp'] * pars['doHRSbs']) + 1))
    HlRS = np.zeros((pars['Ns'], (pars['btsp'] * pars['doHlRSbs']) + 1))
    HiR = np.zeros((1, (pars['btsp'] * pars['doHiRbs']) + 1))
    ChiR = np.zeros((1, (pars['btsp'] * pars['doChiRbs']) + 1))
    HshRS = np.zeros((pars['Ns'], (pars['btsp'] * pars['doHshRSbs']) + 1))
    
    pars['isBtsp'] = 0
    if pars['biasCorrNum'] == 1:
        HR, HRS[:, 0], HlR, HlRS[:, 0], HiR[0], HiRS, ChiR[0], HshR, HshRS[:, 0] = xtrploop(R, pars)
    else:
        HR, HRS[:, 0], HlR, HlRS[:, 0], HiR[0], HiRS, ChiR[0], HshR, HshRS[:, 0] = pars['methodFunc'](R, pars)
    
    Ps = pars['Nt'] / pars['totNt']
    
    if pars['btsp'] > 0:
        pars.update({
            'doHR': False, 'doHRS': pars['doHRSbs'], 'doHlR': False, 'doHlRS': pars['doHlRSbs'],
            'doHiR': pars['doHiRbs'], 'doHiRS': False, 'doChiR': pars['doChiRbs'],
            'doHshR': False, 'doHshRS': pars['doHshRSbs'], 'isBtsp': 1
        })
        
        for k in range(1, pars['btsp'] + 1):
            if pars['methodNum'] != 3:
                rand_indexes = np.random.permutation(pars['totNt'])
                R[:, pars['trials_indxes'][rand_indexes]] = R[:, pars['trials_indxes']]
            
            if pars['biasCorrNum'] == 1:
                _, HRS[:, k * pars['doHRSbs']], _, HlRS[:, k * pars['doHlRSbs']], HiR[:, k * pars['doHiRbs']], _, ChiR[:, k * pars['doChiRbs']], _, HshRS[:, k * pars['doHshRSbs']] = xtrploop(R, pars)
            else:
                _, HRS[:, k * pars['doHRSbs']], _, HlRS[:, k * pars['doHlRSbs']], HiR[:, k * pars['doHiRbs']], _, ChiR[:, k * pars['doChiRbs']], _, HshRS[:, k * pars['doHshRSbs']] = pars['methodFunc'](R, pars)
                
        Ps_bootstrap = np.tile(Ps, (pars['btsp'] + 1, 1)).T
    
    outputs = [None] * pars['Noutput']
    outputs[pars['whereHR']] = HR
    
    if pars['doHRSbs'] and pars['btsp'] > 0:
        outputs[pars['whereHRS']] = np.sum(HRS * Ps_bootstrap, axis=0)
    else:
        outputs[pars['whereHRS']] = np.sum(HRS[:, 0] * Ps, axis=0)
    
    if pars['Nc'] == 1:
        outputs[pars['whereHlR']] = outputs[pars['whereHshR']] = HR
        outputs[pars['whereHlRS']] = np.sum(HRS * Ps_bootstrap, axis=0) if pars['doHlRSbs'] and pars['btsp'] > 0 else np.sum(HRS[:, 0] * Ps, axis=0)
        outputs[pars['whereHiRS']] = np.sum(HRS[:, 0] * Ps, axis=0)
        outputs[pars['whereHshRS']] = np.sum(HRS * Ps_bootstrap, axis=0) if pars['doHshRSbs'] and pars['btsp'] > 0 else np.sum(HRS[:, 0] * Ps, axis=0)
        outputs[pars['whereHiR']], outputs[pars['whereChiR']] = (HR, HR) if pars['methodNum'] == 1 else (np.nan, np.nan)
    else:
        outputs[pars['whereHlR']] = HlR
        outputs[pars['whereHlRS']] = np.sum(HlRS * Ps_bootstrap, axis=0) if pars['doHlRSbs'] and pars['btsp'] > 0 else np.sum(HlRS * np.tile(Ps, ((pars['btsp'] * pars['doHlRSbs']) + 1, 1)).T, axis=0)
        outputs[pars['whereHiR']] = HiR
        outputs[pars['whereHiRS']] = np.sum(HiRS * Ps, axis=0)
        outputs[pars['whereChiR']] = ChiR
        outputs[pars['whereHshR']] = HshR
        outputs[pars['whereHshRS']] = np.sum(HshRS * Ps_bootstrap, axis=0) if pars['doHshRSbs'] and pars['btsp'] > 0 else np.sum(HshRS * Ps, axis=0)
    
    return outputs
