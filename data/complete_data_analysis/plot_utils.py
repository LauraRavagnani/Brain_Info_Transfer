import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


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