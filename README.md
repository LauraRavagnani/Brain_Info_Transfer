# Brain_Info_Transfer
Test of a new information theoretic measure termed Feature-specific Information Transfer (FIT), quantifying how much information about a specific feature flows between two regions. 

<p align="center">
<img width="341" alt="flow_lr" src="https://github.com/user-attachments/assets/29d4486b-b6c7-40a3-a039-2e9c4ed21610" />
</p>


## Data
The data used in this work can be retrieved from this publicy avaiable [EEG dataset](https://datadryad.org/dataset/doi:10.5061/dryad.8m2g3).

## Requisites

Libraries:
* ```numpy```
* ```pandas```
* ```matplotlib```

## Scripts
```final_notebook.ipynb``` is the notebook containing all the results computed with the following scripts:

* ```info_utils.py```: contains functions to compute information theoretic measures
* ```plot_utils.py```: contains functions to plot the results
* ```parameters.py```: contains all the global parameters used in the computations
* ```visualize_graphs.py```: display graphs of DM halos from the simulations

* ```create_json_simulations.py```: create the simulated time series and save in json files
* ```heatmap.py```: create the heatmaps
* ```tempor_loc.py```: compute the temporal location of the neural activity
* ```significance.py```: perform the significance tests
* `fit_te_snr.py`: run the behaviour of FIT and TE wrt SNR

* `eeg_analysis_parallel.py`: to parallely compute the analysis on the first eeg dataset
* `eeg_new_analysis_parallel.py`: to parallely compute the analysis on the second eeg dataset

## Authors and Acknowledgments
### Authors
* **Lorenzo Cavezza** - [Lorycav](https://github.com/Lorycav)
* **Giulia Doda** - [giuliadoda](https://github.com/giuliadoda)
* **Giacomo Longaroni** - [GiacomoLongaroni](https://github.com/GiacomoLongaroni)
* **Laura Ravagnani** - [LauraRavagnani](https://github.com/LauraRavagnani)

### Acknowledgments
This work is based on
<a id="1">[1]</a> 
Marco Celotto et al. "An information-theoretic quantification of the content of communication between brain regions." 37th Conference on Neural Information Processing System (NeurIPS 2023).
