#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

from criteria import *


'''
pipeline for using all criteria functions in a certain order
pipeline2 gets a rough gauge of the false NEGATIVE rate from
the perfect simulated signal.pkl data

ideally we get 0%

input are dataframes, we should keep all data in a /data folder
by convention, which is one level beneath the pipeline directory

ie workfile = mainfolder/pipeline0.py
data = mainfolder/data/data1.csv for example

by convention, apply function that restrict the DOMAIN first before
applying filters that look at data from a given domain


'''


df_real = pd.read_pickle('data/total_dataset.pkl')
df_signal = pd.read_pickle('data/signal.pkl')

df_jpsi = pd.read_pickle('data/jpsi.pkl')
df_psi2s = pd.read_pickle('data/psi2S.pkl')


threshold0 = 0.998
threshold1 = 0.998

df_after0 = peaking_selection_psi2s(df_real, df_psi2s, threshold0)
# df_after0.to_pickle('output')


plt.hist(df_after0['q2'], bins = 500, histtype = 'step', label = 'peak filtered')
# plt.hist(df_real['q2'],  bins = 500, histtype = 'step', label = 'raw')
plt.xlabel('Invariant Mass of products (MeV/C^2)')
plt.ylabel('Counts')

plt.legend()
plt.show()

#
#
df_after1 = peaking_selection_jpsi(df_after0, df_jpsi, threshold1)

print(len(df_after1))
plt.hist(df_after1['q2'], bins = 500, histtype = 'step', label = 'peaks filtered')
# plt.hist(df_real['q2'],  bins = 500, histtype = 'step', label = 'raw')
plt.xlabel('Invariant Mass of products (MeV/C^2)')
plt.ylabel('Counts')

plt.legend()
plt.show()

functions = [b0_endvertex_chi2, b0_ipchi2, ipchi2_selection, kstar_consistent, kstar_endvertex_chi2, pion_pt_selection, kaon_pt_selection, hypotheses_compound]


thresholds = [0.9]*7 + [[0.5, 0.3]]


df_old = df_after1
for index, function in enumerate(functions):

    print(function)
    df_new = function(df_old, df_signal, thresholds[index])
    #
    # plt.hist(df_old['q2'], bins = 500, histtype = 'step', label = 'old')
    # plt.hist(df_new['q2'],  bins = 500, histtype = 'step', label = 'new')
    # plt.xlabel('Invariant Mass of products (MeV/C^2)')
    # plt.ylabel('Counts')
    #
    # plt.legend()
    # plt.show()
    df_old = df_new
    print('HIIIIIII', len(df_old))

plt.hist(df_new['B0_MM'], bins = 500, density = True, histtype = 'step', label = 'peak filtered')
plt.hist(df_real['B0_MM'], bins = 500, density = True, histtype = 'step', label = 'peak filtered')

plt.show()
