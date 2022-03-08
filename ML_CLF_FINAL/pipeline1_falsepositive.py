#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

from criteria import *
import time

'''
pipeline for using all criteria functions in a certain order
pipeline1 is to get a rough gauge of our false POSITIVE rate
using the trash data from b0 mass hard cut

ideally should be 0%

input are dataframes, we should keep all data in a /data folder
by convention, which is one level beneath the pipeline directory

ie workfile = mainfolder/pipeline0.py
data = mainfolder/data/data1.csv for example

by convention, apply function that restrict the DOMAIN first before
applying filters that look at data from a given domain


'''


### we use the trash b0_mm cut
df_peaked = pd.read_pickle('output/peak_filtered.pkl')
df_signal = pd.read_pickle('data/signal.pkl')

functions = [b0_endvertex_chi2, b0_ipchi2, ipchi2_selection, kstar_consistent, kstar_endvertex_chi2, pion_pt_selection, kaon_pt_selection, hypotheses_compound]

thresholds = [0.9]*7 + [[0.5, 0.3]]

time1 = time.time()

df_old = df_peaked
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

time_taken =  time.time() - time1
print(time_taken, 'hihihihihihi')
plt.hist(df_new['B0_MM'], bins = 500, density = True, histtype = 'step', label = 'peak filtered')
plt.hist(df_peaked['B0_MM'], bins = 500, density = True, histtype = 'step', label = 'peak filtered')

plt.show()
