import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from parameters import par

# import global parameters and simulation parameters
n_subjects = par.n_subjects
n_weights = len(par.w_noise) * len(par.w_sig)
n_binsS = par.n_binsS
n_bins = par.n_binsX
delay_max = par.delay_max
simLen = par.simLen
nShuff = par.nShuff
n_trials = par.nTrials_per_stim * par.n_binsS
n_test = par.n_test
w_sig = par.w_sig
w_noise = par.w_noise
snr = par.snr


#-------------------------------figure 4B-C-------------------------------------------
def plot_map4(l_array, r_array, title, min_lim, max_lim, type, upper, upper_col):
    fig, ax = plt.subplots(nrows = 2, ncols =2, figsize=(16,10))

    if type=="Paper":
        v_max = max(np.max(np.mean(l_array[:,0:upper], axis=0).T),np.max(np.mean(r_array[:,0:upper], axis=0).T))*upper_col
        v_min = min(np.min(np.mean(l_array[:,0:upper], axis=0).T),np.min(np.mean(r_array[:,0:upper], axis=0).T))*0.6

        l = ax[0][0].imshow(np.mean(l_array, axis=0).T, origin='lower', cmap='plasma', vmin=v_min, vmax=v_max)
        r = ax[0][1].imshow(np.mean(r_array, axis=0).T, origin='lower', cmap='cividis', vmin=v_min, vmax=v_max)

        l_m = np.mean(l_array, axis = (0,2))
        r_m = np.mean(r_array, axis = (0,2))

        l_s = np.std(l_array, axis=(0,2))/np.sqrt(15+120)
        r_s = np.std(r_array, axis=(0,2))/np.sqrt(15+120)

    elif type=="New":

        v_max = max(np.max(l_array[:,0:upper]),np.max(r_array[:,0:upper]))*0.6
        v_min = min(np.min(l_array[:,0:upper]),np.min(r_array[:,0:upper]))*0.6

        l = ax[0][0].imshow(l_array, origin='lower', cmap='plasma',vmin=v_min, vmax=v_max)
        r = ax[0][1].imshow(r_array, origin='lower', cmap='cividis',vmin=v_min, vmax=v_max)

        l_m = np.mean(l_array, axis = 0)
        r_m = np.mean(r_array, axis = 0)
      
        l_s = np.std(l_array, axis=0)/np.sqrt(15+120)
        r_s = np.std(r_array, axis=0)/np.sqrt(15+120)


    tmax = l_m.shape[0]

    t = np.arange(tmax)

    v_min_2 = min(np.min(l_m[0:upper]),np.min(r_m[0:upper]))*min_lim
    v_max_2 = max(np.max(l_m[0:upper]),np.max(r_m[0:upper]))*max_lim

    ax[1][0].grid(alpha=0.3)
    ax[1][0].plot(l_m, label='mean', color='C1')
    ax[1][0].set_title('LOT to ROT ' + title + ' (averaged over delays)')
    ax[1][0].fill_between(t, l_m-l_s, l_m+l_s, alpha=0.4, label = 'mean error', color='C1')
    ax[1][0].legend()
    ax[1][0].set_xlabel('peri-stimulus time (ms)')
    ax[1][0].set_ylabel('bits')
    ax[1][0].set_ylim((v_min_2,v_max_2))

    ax[1][1].grid(alpha=0.3)
    ax[1][1].plot(r_m, label='mean')
    ax[1][1].set_title('ROT to LOT ' + title + ' (averaged over delays)')
    ax[1][1].fill_between(t, r_m-r_s, r_m+r_s, alpha=0.4, label = 'mean error', color='C0')
    ax[1][1].legend()
    ax[1][1].set_xlabel('peri-stimulus time (ms)')
    ax[1][1].set_ylabel('bits')
    ax[1][1].set_ylim((v_min_2,v_max_2))

    l_cbar = ax[0][0].figure.colorbar(l, label='bits', location = 'bottom', pad = 0.15)
    r_cbar = ax[0][1].figure.colorbar(r, label='bits', location = 'bottom', pad = 0.15)

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))  

    l_cbar.ax.xaxis.set_major_formatter(formatter)
    r_cbar.ax.xaxis.set_major_formatter(formatter)   

    ax[1][0].yaxis.set_major_formatter(formatter)
    ax[1][1].yaxis.set_major_formatter(formatter)

    ax[0][0].set_xlim(0,upper)
    ax[0][1].set_xlim(0,upper)

    ax[0][0].set_xlabel('peri-stimulus time (ms)')
    ax[0][1].set_xlabel('peri-stimulus time (ms)')

    ax[0][0].set_ylabel('delay (ms)')
    ax[0][1].set_ylabel('delay (ms)')

    ax[0][0].set_title('LOT to ROT ' + title)
    ax[0][1].set_title('ROT to LOT ' + title)

    ax[0][0].set_yticks(list(np.arange(0,80,20)))
    ax[0][0].set_yticklabels(list(np.arange(0,80,20)))
    ax[0][1].set_yticks(list(np.arange(0,80,20)))
    ax[0][1].set_yticklabels(list(np.arange(0,80,20)))

    plt.show()

#-------------------------- figure 2A -------------------------------#

def plot_2A(fit_heatmap, te_heatmap, title):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 8), squeeze=False)

    fit_ax = ax[0][0]
    te_ax = ax[0][1]

    im_fit = fit_ax.imshow(fit_heatmap, origin='lower', cmap='YlGn')
    im_te = te_ax.imshow(te_heatmap, origin='lower', cmap='RdPu')

    fit_ax.set_title('FIT', fontsize=22)
    te_ax.set_title('TE', fontsize=22)

    cbar_fit = fit_ax.figure.colorbar(im_fit, ax=fit_ax)
    cbar_te = te_ax.figure.colorbar(im_te, ax=te_ax)

    cbar_fit.set_label(label='bits', fontsize=20)
    cbar_fit.ax.tick_params(labelsize=18)

    cbar_te.set_label(label='bits', fontsize=20)
    cbar_te.ax.tick_params(labelsize=18)

    fit_ax.set_xlabel("$w_{noise}$", fontsize=20)
    fit_ax.set_ylabel("$w_{stim}$", fontsize=20)

    te_ax.set_xlabel("$w_{noise}$", fontsize=20)
    te_ax.set_ylabel("$w_{stim}$", fontsize=20)

    fit_ax.set_xticks(np.arange(11))
    fit_ax.set_xticklabels(w_noise.round(1))
    fit_ax.set_yticks(np.arange(11))
    fit_ax.set_yticklabels(w_sig.round(1))

    te_ax.set_xticks(np.arange(11))
    te_ax.set_xticklabels(w_noise.round(1))
    te_ax.set_yticks(np.arange(11))
    te_ax.set_yticklabels(w_sig.round(1))

    fit_ax.tick_params(axis='both', which='major', labelsize=16)
    te_ax.tick_params(axis='both', which='major', labelsize=16)

    fig.suptitle(title, fontsize=24)
    plt.show()


#-------------------------- figure 2B -------------------------------#

def plot_2B(fit_B, fit_B_std, te_B, te_B_std, title):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 10), squeeze=False)

    fit_ax = ax[0][0]
    te_ax = ax[1][0]

    fit_ax.plot(np.linspace(0, 10*simLen, simLen), fit_B, label='mean FIT', color='green', linestyle='-', linewidth=1.5, marker='o', ms=8, mfc='yellow')
    fit_ax.fill_between(np.linspace(0, 10*simLen, simLen), fit_B+fit_B_std, fit_B-fit_B_std, label='SEM FIT', color='lightgreen', alpha=0.8)

    te_ax.plot(np.linspace(0, 10*simLen, simLen), te_B, label='mean TE',  color='magenta', linestyle='-', linewidth=1.5, marker='o', ms=8, mfc='yellow')
    te_ax.fill_between(np.linspace(0, 10*simLen, simLen), te_B+te_B_std, te_B-te_B_std, label='SEM TE', color='hotpink', alpha=0.4)

    te_ax.set_ylim(0, 0.08)

    fit_ax.axvspan(10*par.stimWin[0], 10*par.stimWin[1], color='grey', alpha=0.15)
    te_ax.axvspan(10*par.stimWin[0], 10*par.stimWin[1], color='grey', alpha=0.15, label='stimulus window')

    fit_ax.set_ylabel("bits", fontsize=13)

    te_ax.set_xlabel("time (ms)", fontsize=13)
    te_ax.set_ylabel("bits", fontsize=13)

    fit_ax.grid(alpha=0.3, linestyle='--')
    te_ax.grid(alpha=0.3, linestyle='--')

    fig.legend(loc=(0.82, 0.75), fontsize=10)

    fig.suptitle(title, fontsize=18)
    plt.show()
    fig.savefig('fig2B_r.png',dpi=600)


#-------------------------- significance TE -------------------------------#

def plot_significance_te(te_sig_Sshuff, te_val, bins, title):

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

    fit_dist = np.zeros((n_test, bins))
    fit_edges = np.zeros((n_test, bins+1))
    fit_centers = np.zeros((n_test, bins))
    width = np.zeros(n_test)

    for i in range(n_test):
        fit_dist[i], fit_edges[i] = np.histogram(te_sig_Sshuff[i], bins=bins, density=False)
        width[i] = fit_edges[i][1]-fit_edges[i][0]
        fit_centers[i] = (fit_edges[i][:-1] + fit_edges[i][1:]) / 2


    ax[0][0].bar(fit_centers[0], fit_dist[0], align='center', width=width[0], color='hotpink', edgecolor='hotpink', alpha=0.5, label='S-shuff null')
    ax[0][1].bar(fit_centers[1], fit_dist[1], align='center', width=width[1], color='hotpink', edgecolor='hotpink', alpha=0.5)
    ax[1][0].bar(fit_centers[2], fit_dist[2], align='center', width=width[2], color='hotpink', edgecolor='hotpink', alpha=0.5)
    ax[1][1].bar(fit_centers[3], fit_dist[3], align='center', width=width[3], color='hotpink', edgecolor='hotpink', alpha=0.5)

    ax[0][0].vlines(te_val[0], ymin=0, ymax=1.1*np.max(fit_dist[0]), color='darkred', linestyle='dotted', linewidth=2, label='Measured')
    ax[0][1].vlines(te_val[1], ymin=0, ymax=1.1*np.max(fit_dist[1]), color='darkred', linestyle='dotted', linewidth=2)
    ax[1][0].vlines(te_val[2], ymin=0, ymax=1.1*np.max(fit_dist[2]), color='darkred', linestyle='dotted', linewidth=2)
    ax[1][1].vlines(te_val[3], ymin=0, ymax=1.1*np.max(fit_dist[3]), color='darkred', linestyle='dotted', linewidth=2)

    ax[0][0].set_xlabel('TE value (bits)', fontsize=12)
    ax[0][1].set_xlabel('TE value (bits)', fontsize=12)
    ax[1][0].set_xlabel('TE value (bits)', fontsize=12)
    ax[1][1].set_xlabel('TE value (bits)', fontsize=12)

    ax[0][0].set_ylabel('Frequences', fontsize=12)
    ax[0][1].set_ylabel('Frequences', fontsize=12)
    ax[1][0].set_ylabel('Frequences', fontsize=12)
    ax[1][1].set_ylabel('Frequences', fontsize=12)

    ax[0][0].set_title('$w_{sig}$ = '+f'{w_sig[0]}, ' + '$w_{noise}$ = '+f'{w_noise[6]:.1f}', fontsize=16)
    ax[0][1].set_title('$w_{sig}$ = '+f'{w_sig[5]}, ' + '$w_{noise}$ = '+f'{w_noise[4]}', fontsize=16)
    ax[1][0].set_title('$w_{sig}$ = '+f'{w_sig[8]}, ' + '$w_{noise}$ = '+f'{w_noise[9]}', fontsize=16)
    ax[1][1].set_title('$w_{sig}$ = '+f'{w_sig[9]}, ' + '$w_{noise}$ = '+f'{w_noise[2]}', fontsize=16)

    ax[0][0].grid(color='grey', linestyle='--', alpha=.2)
    ax[0][1].grid(color='grey', linestyle='--', alpha=.2)
    ax[1][0].grid(color='grey', linestyle='--', alpha=.2)
    ax[1][1].grid(color='grey', linestyle='--', alpha=.2)

    fig.suptitle(title, fontsize=20)
    fig.legend(fontsize=16)
    plt.show()


#-------------------------- significance FIT -------------------------------#

def plot_significance_fit(fit_sig_Sshuff, fit_sig_Sfix, fit_val, bins_s, bins_f, factor, title):

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

    fit_dist = np.zeros((n_test, bins_s))
    fit_edges = np.zeros((n_test, bins_s+1))
    fit_centers = np.zeros((n_test, bins_s))
    fit_qunatiles_s = np.zeros((n_test, 2))
    width = np.zeros(n_test)

    fit_dist_fix = np.zeros((n_test, bins_f))
    fit_edges_fix = np.zeros((n_test, bins_f+1))
    fit_centers_fix = np.zeros((n_test, bins_f))
    fit_qunatiles_fix = np.zeros((n_test, 2))
    
    width_fix = np.zeros(n_test)

    for i in range(n_test):
        fit_dist[i], fit_edges[i] = np.histogram(fit_sig_Sshuff[i], bins=bins_s, density=False)
        width[i] = fit_edges[i][1]-fit_edges[i][0]
        fit_centers[i] = (fit_edges[i][:-1] + fit_edges[i][1:]) / 2
        fit_qunatiles_s[i] = np.quantile(fit_sig_Sshuff[i], (0.025, 0.975))

        fit_dist_fix[i], fit_edges_fix[i] = np.histogram(fit_sig_Sfix[i], bins=bins_f, density=False)
        width_fix[i] = (fit_edges_fix[i][1]-fit_edges_fix[i][0])
        fit_centers_fix[i] = (fit_edges_fix[i][:-1] + fit_edges_fix[i][1:]) / 2
        fit_qunatiles_fix[i] = np.quantile(fit_sig_Sfix[i], (0.025, 0.975))
    
    ## HISTOGRAMS DISTRIBUTIONS
    ax[0][0].bar(fit_centers[0], fit_dist[0], align='center', width=width[0], color='gray', edgecolor='gray', alpha=0.5, label='S-shuff null')
    ax[0][1].bar(fit_centers[1], fit_dist[1], align='center', width=width[1], color='gray', edgecolor='gray', alpha=0.5)
    ax[1][0].bar(fit_centers[2], fit_dist[2], align='center', width=width[2], color='gray', edgecolor='gray', alpha=0.5)
    ax[1][1].bar(fit_centers[3], fit_dist[3], align='center', width=width[3], color='gray', edgecolor='gray', alpha=0.5)

    ax[0][0].bar(fit_centers_fix[0], fit_dist_fix[0], align='center', width=width[0]*factor, color='green', edgecolor='darkgreen', alpha=0.5, label='S-fix null')
    ax[0][1].bar(fit_centers_fix[1], fit_dist_fix[1], align='center', width=width[1]*factor, color='green', edgecolor='darkgreen', alpha=0.5)
    ax[1][0].bar(fit_centers_fix[2], fit_dist_fix[2], align='center', width=width[2]*factor, color='green', edgecolor='darkgreen', alpha=0.5)
    ax[1][1].bar(fit_centers_fix[3], fit_dist_fix[3], align='center', width=width[3]*factor, color='green', edgecolor='darkgreen', alpha=0.5)


    ## MEASURE (HEATMAP VAL)
    ax[0][0].vlines(fit_val[0], ymin=0, ymax=1.1*np.max(fit_dist_fix[0]), color='darkred', linestyle='dotted', linewidth=1.5, label='Measured')
    ax[0][1].vlines(fit_val[1], ymin=0, ymax=1.1*np.max(fit_dist_fix[1]), color='darkred', linestyle='dotted', linewidth=1.5)
    ax[1][0].vlines(fit_val[2], ymin=0, ymax=1.1*np.max(fit_dist_fix[2]), color='darkred', linestyle='dotted', linewidth=1.5)
    ax[1][1].vlines(fit_val[3], ymin=0, ymax=1.1*np.max(fit_dist_fix[3]), color='darkred', linestyle='dotted', linewidth=1.5)


    ## P-VAL QUANTILE 
    ax[0][0].vlines((fit_qunatiles_s[0,0], fit_qunatiles_s[0,1]), ymin=0, ymax=1.1*np.max(fit_dist_fix[0]), color='black', linestyle='-', linewidth=1, label='0.05 p-val')


    ## LABELS
    ax[0][0].set_xlabel('FIT value (bits)', fontsize=12)
    ax[0][1].set_xlabel('FIT value (bits)', fontsize=12)
    ax[1][0].set_xlabel('FIT value (bits)', fontsize=12)
    ax[1][1].set_xlabel('FIT value (bits)', fontsize=12)

    ax[0][0].set_ylabel('Frequences', fontsize=12)
    ax[0][1].set_ylabel('Frequences', fontsize=12)
    ax[1][0].set_ylabel('Frequences', fontsize=12)
    ax[1][1].set_ylabel('Frequences', fontsize=12)

    ax[0][0].set_title('$w_{sig}$ = '+f'{w_sig[0]}, ' + '$w_{noise}$ = '+f'{w_noise[6]:.1f}', fontsize=14)
    ax[0][1].set_title('$w_{sig}$ = '+f'{w_sig[5]}, ' + '$w_{noise}$ = '+f'{w_noise[4]}', fontsize=14)
    ax[1][0].set_title('$w_{sig}$ = '+f'{w_sig[8]}, ' + '$w_{noise}$ = '+f'{w_noise[9]}', fontsize=14)
    ax[1][1].set_title('$w_{sig}$ = '+f'{w_sig[9]}, ' + '$w_{noise}$ = '+f'{w_noise[2]}', fontsize=14)


    ## FANCY STUFF
    ax[0][0].grid(color='grey', linestyle='--', alpha=.2)
    ax[0][1].grid(color='grey', linestyle='--', alpha=.2)
    ax[1][0].grid(color='grey', linestyle='--', alpha=.2)
    ax[1][1].grid(color='grey', linestyle='--', alpha=.2)

    fig.suptitle(title, fontsize=20)
    fig.legend(fontsize=14)
    plt.show()


#-------------------------- figure 2E -------------------------------#

def plot_fig2E(fit_E, fit_E_std, te_E, te_E_std, snr, title):

    fig, ax = plt.subplots(1,2, figsize=(15,6))

    ax[0].plot( snr, fit_E, linewidth=1.5, color='green', marker='o', ms=8, mfc='yellow', label='mean FIT' )
    ax[0].fill_between(snr, fit_E+fit_E_std, fit_E-fit_E_std, label='SEM FIT', color='lightgreen', alpha=0.8)
    ax[0].set_xlabel(r'SNR(${\delta/\sigma}$)')
    ax[0].set_ylabel('FIT (bits)')
    ax[0].set_title('FIT')
    ax[0].grid(alpha = 0.3, linestyle ='--')
    ax[0].legend()

    ax[1].plot( snr, te_E, linewidth=1.5, color='magenta', marker='o', ms=8, mfc='yellow', label='mean TE' )
    ax[1].fill_between(snr, te_E+te_E_std, te_E-te_E_std, label='SEM TE', color='hotpink', alpha=0.4)
    ax[1].set_xlabel(r'SNR(${\delta/\sigma}$)')
    ax[1].set_ylabel('TE (bits)')
    ax[1].set_title('TE')
    ax[1].grid(alpha = 0.3, linestyle ='--')
    ax[1].legend()    

    fig.suptitle(title, fontsize=20)
    plt.show()