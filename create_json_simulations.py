import os
import numpy as np
import pandas as pd
import json

# Parametri di simulazione
nTrials_per_stim = 50    # numero di prove per simulazione
simReps = 2                # numero di simulazioni (soggetti)
nShuff = 10                 # numero di permutazioni (non utilizzato in questo script)

w_sig = np.linspace(0, 1, num=11)      # pesi del segnale per il calcolo di Y
w_noise = np.linspace(0, 1, num=11)    # pesi del rumore per il calcolo di Y
stdX_noise = 2                      # deviazione standard del rumore in X_noise
stdY = 2                            # deviazione standard del rumore in Y
ratio = 0.2                         # rapporto tra stdX_sig e stdX_noise
stdX_sig = ratio * stdX_noise       # deviazione standard del segnale in X_signal

# Parametri globali
simLen = 50                             # lunghezza della simulazione (unità di 10 ms)
stimWin = [20, 25]                      # finestra temporale di codifica dello stimolo in X_signal
delays = np.linspace(4, 6, num=3, dtype=int)
n_binsS = 4                             # numero di valori dello stimolo
n_binsX = 3
n_binsY = 3
eps = 1e-52                             # valore infinitesimo per evitare errori di binning

nTrials = nTrials_per_stim * n_binsS

# create the main folder "Simulations" if it does not exist
base_dir = "Simulations"
os.makedirs(base_dir, exist_ok=True)

# Estrai un ritardo casuale per ciascuna simulazione (soggetto)
reps_delays = np.random.choice(delays, simReps, replace=True)

# Loop sulle simulazioni (subject)
for simIdx in range(simReps):
    print(f"Simulazione numero: {simIdx}")
    d = reps_delays[simIdx]  # ritardo per questa simulazione
    t_start = stimWin[0] 
    t_del = t_start + d

    subj_file = f"subject{simIdx:02d}.json"
    filepath = os.path.join(base_dir, subj_file)

    data_json = {}
    
    # Loop sui pesi: combinazione tra w_sig e w_noise (totale 10x10 = 100 combinazioni)
    for sigIdx in range(len(w_sig)):
        for noiseIdx in range(len(w_noise)):

            weight_index = sigIdx * len(w_noise) + noiseIdx
            
            # Genera lo stimolo per ogni trial (valori interi da 1 a n_binsS)
            S = np.random.randint(1, n_binsS + 1, size=nTrials)
            
            # Simula le timeseries per X_noise e X_signal (dimensione: simLen x nTrials)
            X_noise = np.random.normal(0, stdX_noise, size=(simLen, nTrials))
            
            X_signal = eps * np.random.normal(0, stdX_noise, size=(simLen, nTrials))
            # Inserisci lo stimolo nella finestra definita
            X_signal[stimWin[0]:stimWin[1], :] = np.tile(S, (stimWin[1] - stimWin[0], 1))
            # Applica rumore moltiplicativo
            X_signal = X_signal * (1 + np.random.normal(0, stdX_sig, size=(simLen, nTrials)))

            # Calcola il contributo a Y dal segnale e dal rumore con ritardo
            X2Ysig = w_sig[sigIdx] * np.vstack((
                eps * np.random.normal(0, stdX_noise, size=(d, nTrials)),
                X_signal[0:simLen-d, :]
            ))
            X2Ynoise = w_noise[noiseIdx] * np.vstack((
                eps * np.random.normal(0, stdX_noise, size=(d, nTrials)),
                X_noise[0:simLen-d, :]
            ))
            Y = X2Ysig + X2Ynoise + np.random.normal(0, stdY, size=(simLen, nTrials))
            
            
            # Per la discretizzazione si considerano istanti specifici (al tempo t-d, t, t-d)
#            noise_vals_d = X_noise[t-d, :]
#            sig_vals_d = X_signal[t-d, :]
            
#            _, bin_edges_noise = pd.cut(noise_vals_d, n_binsX, retbins=True)
#            bX_noise = np.digitize(noise_vals_d, bins=bin_edges_noise, right=True)
#            
#            _, bin_edges_sig = pd.cut(sig_vals_d, n_binsX, retbins=True)
#            bX_signal = np.digitize(sig_vals_d, bins=bin_edges_sig, right=True)
#            
#            _, bin_edges_Yt = pd.cut(Y[t, :], n_binsY, retbins=True)
#            bYt = np.digitize(Y[t, :], bins=bin_edges_Yt, right=True)
#            
#            _, bin_edges_Ypast = pd.cut(Y[t-d, :], n_binsY, retbins=True)
#            bYpast = np.digitize(Y[t-d, :], bins=bin_edges_Ypast, right=True)
            
            # Per ogni trial, convertiamo le timeseries in stringhe, separando i valori con il carattere ","
#            x_noise_t = [",".join(map(str, X_noise[t_start, i])) for i in range(nTrials)]
#            x_signal_t = [",".join(map(str, X_signal[t_start, i])) for i in range(nTrials)]
#            Y_t = [",".join(map(str, Y[t_start, i])) for i in range(nTrials)]
#            Y_tdel = [",".join(map(str, Y[t_del, i])) for i in range(nTrials)]

            #   x_noise_t = [X_noise[t_start, i] for i in range(nTrials)]
            #   x_signal_t = [X_signal[t_start, i] for i in range(nTrials)]
            #   Y_t = [Y[t_start, i] for i in range(nTrials)]
            #   Y_tdel = [Y[t_del, i] for i in range(nTrials)]

            data_json[f'{weight_index}'] = {
                'S': S.tolist(),
                't_start': int(t_start),
                'd': int(d),
                'X_noise': X_noise.tolist(),
                'X_signal': X_signal.tolist(),
                'Y': Y.tolist()
            }

            with open(filepath, 'w') as f:
                json.dump(data_json, f)
            
            print(f"Salvato: {filepath}")

