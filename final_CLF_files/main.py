# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 12:41:23 2022

@author: Vid
"""

"""
Created on Sun Jan 30 20:14:53 2022

@author: Vid

Extreme Gradient Boosting Classifier.

The class contains methods for:
    - training/testing (simultaneously)
    - predicting unknown samples to obtain
      probability arrays for each event.
      This file also outputs provisional signals in a .pkl file
      given some dataset of unknown events
      
All datasets are to be situated in data folder in the same directory as this
.py file.

"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from data_wrangling import *
from XGBoostClassifier import *

#%% Figures sytle
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
#Options
params = {'text.usetex' : True,
          'font.size' : 5,
          'font.family' : 'lmodern',
          #'text.latex.unicode': True,
          'axes.labelsize':15,
          'legend.fontsize': 15,
          'xtick.labelsize': 10,
          'ytick.labelsize': 15,
          #'figure.figsize': [7.5,7.5/1.2],
                     }
plt.rcParams.update(params)

#%% Loading files - filenames.
"""
To read the files without modification of the code, they must be all stored
in a folder "data" where data and this python file must be located in the same 
directory.
"""

### Unkown data
# Total dataset - unknown data to analyse and extract signal out of
total_dataset = pd.read_pickle('data/total_dataset.pkl')

### Known - labeled data
# The signal decay, simulated as per the Standard Model
sig = pd.read_pickle('data/signal.pkl')
# B0 ---> signal with kaon reconstructed as pion and pion as kaon
k_pi_swap = pd.read_pickle("data/k_pi_swap.pkl")

# B0 ---> J/psi K^*0,     j/psi --> mu mu
jpsi = pd.read_pickle('data/jpsi.pkl')
# B0 ---> psi(2S) K^*0,   psi(2S) --> mu mu
psi2S = pd.read_pickle("data/psi2S.pkl")
# B0 ---> J/psi K^+0 with muon reconstructed as kaon and kaon as muon
jpsi_mu_k_swap = pd.read_pickle("data/jpsi_mu_k_swap.pkl")
# B0 ---> J/psi K^+0 with muon reconstructed as pion and pion as muon
jpsi_mu_pi_swap = pd.read_pickle("data/jpsi_mu_pi_swap.pkl")
# B_S^0 --->        \phi mu mu,      \phi --> KK and 1 K reconstructed as pion
phimumu = pd.read_pickle("data/phimumu.pkl")
# Lambda_b^0 --->   pK\mu\mu        with p reconstructed as K and K as pi 
pKmumu_piTok_kTop = pd.read_pickle("data/pKmumu_piTok_kTop.pkl")
# Lambda_b^0 --->   pK\mu\mu        with p reconstructed as pion
pKmumu_piTop = pd.read_pickle("data/pKmumu_piTop.pkl")

# Thrash data
b0_mm_trash0 = pd.read_pickle("data/b0_mm_trash0.pkl")



# Simulation which is flat in three angular variables and q^2
acceptance = pd.read_pickle("data/acceptance_mc.pkl")

# Produced files from hard-coding
cut_full= pd.read_pickle('output/filtered_full.pkl')
cut_peaking = pd.read_pickle("output/filtered_peaks.pkl")
filtered_wo_peaking = pd.read_pickle("output/filtered_wo_peaking.pkl")

#%% Datasets
datasets = [sig,k_pi_swap,
            #jpsi,jpsi_mu_k_swap, jpsi_mu_pi_swap,
            #psi2S, 
            phimumu,
            pKmumu_piTok_kTop,pKmumu_piTop,
            b0_mm_trash0] 

keys = total_dataset.keys()

dataset_names = ["sig", r"sig; $K \leftrightarrow \pi$",
                 #r"$J/\psi$", r"$J/\psi; \mu \leftrightarrow K$",
                 #r"$J/\psi; \mu \leftrightarrow \pi$",
                 #r"$\psi$(2S)",
                 r"$\phi \mu \mu$",
                 r"$p K \mu \mu K \leftrightarrow \pi$",
                 r"$p K \mu \mu; p \leftrightarrow \pi$",
                 r"bg"]

dataset_labels = [i for i in range(len(datasets))]
for i in range(len(datasets)):
    try:
        datasets[i] = datasets[i].drop(columns=["year","B0_ID"])
    except:
        pass
    datasets[i]["file"] = dataset_labels[i]

keys = datasets[0].keys()

#%% Histogram of events      

# =============================================================================
# plt.figure("len(classes)")
# plt.bar(np.arange(len(datasets)), [len(dat) for dat in datasets])
# plt.xticks(np.arange(len(datasets)), dataset_names)
# plt.grid()
# plt.ylabel("N")
# plt.show()
# 
# =============================================================================

#%% DataLoading

size_class = 85_000_0000
N_classes = len(datasets)
X,y,scaler = load_data(size_class=size_class, datasets = datasets)
try:
    if len(keysX)>79:
        keysX = keys[:-1]
except:
    pass

#%% Classifying

model = XGBoostCLF(X,y)
model.train() 
model.test()

# =============================================================================
# 
# model = XGBoost_sklearn_CLF(X,y)
# model.train_test()
# 
# =============================================================================

# List of all files for easier accessibility

# =============================================================================
# datasets_ALL = [sig,k_pi_swap,
#                 jpsi,jpsi_mu_k_swap, jpsi_mu_pi_swap,
#                 psi2S, 
#                 phimumu,
#                 pKmumu_piTok_kTop,pKmumu_piTop,
#                 b0_mm_trash0,
#                 cut_full,cut_peaking,filtered_wo_peaking,
#                 total_dataset] 
# =============================================================================
      
# Composition of unknown data
file = cut_full   #cut_ful::: ~2500 events obtained after all the hard-cutting

try:
    file= file.drop(columns=["year","B0_ID"])
except:
    pass
try:
    file = file.drop(columns=["file"])
except:
    pass

X_unknown_unscaled = file
X_unknown = scaler.transform(X_unknown_unscaled)

unknown_composition = model.get_composition_of_classes(X_unknown)
print("Composition\n",unknown_composition)

plt.figure("Composition")
plt.bar(np.arange(len(datasets)), unknown_composition)
plt.xticks(np.arange(len(datasets)), dataset_names)
plt.ylabel("Percentage")
plt.show()

#nonsigs_provis = model.get_provisional_nonsignals(X_unknown_unscaled,
#                                             N_classes,size_class)

# =============================================================================
# 
# sigs_provis = model.get_provisional_signals(X_unknown_unscaled,
#                                       N_classes,size_class)
# 
# nonsignalindices = model.get_provisional_indices(X_unknown_unscaled,
#                                   N_classes,size_class,signal=False)
# 
# =============================================================================

# Get indices for events classified as signal with probability>50%.
signalindices,prob_arr = model.get_provisional_indices(X_unknown,
                                  N_classes,size_class,decision_boundary=0.5,
                                  signal=True,save=False)


# Store final signals
final_signals=cut_full[0:0]
for i,ind in enumerate(signalindices):
   sig = cut_full.iloc[ind]
   final_signals = final_signals.append(sig)

final_signals.to_pickle('output/final_signals.pkl')

# =============================================================================
# 
# # Plot the distribution of probability values for the classified signals

# plt.figure("signal_probability_distribution")
# plt.hist(prob_arr[signalindices,0],bins=30,histtype="step")
# plt.ylabel("N")
# plt.xlabel("Probability of being a signal")
# plt.legend()
# plt.show()
# 
# =============================================================================

 
