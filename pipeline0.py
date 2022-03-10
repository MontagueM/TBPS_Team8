#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

from criteria import *


'''
pipeline for using all criteria functions in a certain order

input are dataframes, we should keep all data in a /data folder
by convention, which is one level beneath the pipeline directory

ie workfile = mainfolder/pipeline0.py
data = mainfolder/data/data1.csv for example

by convention, apply function that restrict the DOMAIN first before
applying filters that look at data from a given domain


'''

'''

for simple reference here are the 81 data columns present in the dataframes

'mu_plus_MC15TuneV1_ProbNNk', 'mu_plus_MC15TuneV1_ProbNNpi',
       'mu_plus_MC15TuneV1_ProbNNmu', 'mu_plus_MC15TuneV1_ProbNNe',
       'mu_plus_MC15TuneV1_ProbNNp', 'mu_plus_P', 'mu_plus_PT', 'mu_plus_ETA',
       'mu_plus_PHI', 'mu_plus_PE', 'mu_plus_PX', 'mu_plus_PY', 'mu_plus_PZ',
       'mu_plus_IPCHI2_OWNPV', 'mu_minus_MC15TuneV1_ProbNNk',
       'mu_minus_MC15TuneV1_ProbNNpi', 'mu_minus_MC15TuneV1_ProbNNmu',
       'mu_minus_MC15TuneV1_ProbNNe', 'mu_minus_MC15TuneV1_ProbNNp',
       'mu_minus_P', 'mu_minus_PT', 'mu_minus_ETA', 'mu_minus_PHI',
       'mu_minus_PE', 'mu_minus_PX', 'mu_minus_PY', 'mu_minus_PZ',
       'mu_minus_IPCHI2_OWNPV', 'K_MC15TuneV1_ProbNNk',
       'K_MC15TuneV1_ProbNNpi', 'K_MC15TuneV1_ProbNNmu',
       'K_MC15TuneV1_ProbNNe', 'K_MC15TuneV1_ProbNNp', 'K_P', 'K_PT', 'K_ETA',
       'K_PHI', 'K_PE', 'K_PX', 'K_PY', 'K_PZ', 'K_IPCHI2_OWNPV',
       'Pi_MC15TuneV1_ProbNNk', 'Pi_MC15TuneV1_ProbNNpi',
       'Pi_MC15TuneV1_ProbNNmu', 'Pi_MC15TuneV1_ProbNNe',
       'Pi_MC15TuneV1_ProbNNp', 'Pi_P', 'Pi_PT', 'Pi_ETA', 'Pi_PHI', 'Pi_PE',
       'Pi_PX', 'Pi_PY', 'Pi_PZ', 'Pi_IPCHI2_OWNPV', 'B0_MM',
       'B0_ENDVERTEX_CHI2', 'B0_ENDVERTEX_NDOF', 'B0_FDCHI2_OWNPV', 'Kstar_MM',
       'Kstar_ENDVERTEX_CHI2', 'Kstar_ENDVERTEX_NDOF', 'Kstar_FDCHI2_OWNPV',
       'J_psi_MM', 'J_psi_ENDVERTEX_CHI2', 'J_psi_ENDVERTEX_NDOF',
        'J_psi_FDCHI2_OWNPV', 'B0_IPCHI2_OWNPV', 'B0_DIRA_OWNPV', 'B0_OWNPV_X',
       'B0_OWNPV_Y', 'B0_OWNPV_Z', 'B0_FD_OWNPV', 'B0_ID', 'q2', 'phi',
       'costhetal', 'costhetak', 'polarity', 'year'],
      dtype='object'
'''


df_real = pd.read_pickle('data/total_dataset.pkl')
df_signal = pd.read_pickle('data/signal.pkl')

df_jpsi = pd.read_pickle('data/jpsi.pkl')
df_psi2s = pd.read_pickle('data/psi2S.pkl')

plt.hist(df_signal['B0_MM'], bins = 500, density = True, histtype = 'step', label = 'peak filtered')
plt.hist(df_real['B0_MM'], bins = 500, density = True, histtype = 'step', label = 'peak filtered')

plt.show()
threshold0 = 0.998
threshold1 = 0.998

print(len(df_real))
df_after0 = peaking_selection_psi2s(df_real, df_psi2s, threshold0)
print(len(df_after0))
df_after1 = peaking_selection_jpsi(df_after0, df_jpsi, threshold1)

print(len(df_after1))
plt.hist(df_after1['q2'], bins = 500, histtype = 'step', label = 'peaks filtered')
# plt.hist(df_real['q2'],  bins = 500, histtype = 'step', label = 'raw')
plt.xlabel('Invariant Mass of products (MeV/C^2)')
plt.ylabel('Counts')

plt.legend()
plt.show()

df_after1.to_pickle('output/peak_filtered.pkl')

functions = [ b0_endvertex_chi2, ipchi2_selection,  b0_ipchi2, kstar_endvertex_chi2, pion_pt_selection, kaon_pt_selection, kstar_fdchi2, b0_fdchi2, kstar_consistent, hypotheses_compound]


thresholds =   [0.9]*6 + [0.99] + [0.995] + [2.0] + [[0.5, 0.3]]

df_old = pd.read_pickle('output/peak_filtered.pkl')
for index, function in enumerate(functions):
    print('Current data length = ', len(df_old))
    print('Function and threshold = ', function, thresholds[index])

    df_new = function(df_old, df_signal, thresholds[index])
    df_old = df_new

df_new.to_pickle('output/final_filtered.pkl')


plt.hist(df_new['B0_MM'], bins = 500, density = True, histtype = 'step', label = 'cut data')
plt.hist(df_real['B0_MM'], bins = 500, density = True, histtype = 'step', label = 'data')
plt.hist(df_signal['B0_MM'], bins = 500, density = True, histtype = 'step', label = 'signal')
plt.legend()
plt.show()
