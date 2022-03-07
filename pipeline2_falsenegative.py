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
df_signal = pd.read_pickle('data/acceptance_mc.pkl')

df_jpsi = pd.read_pickle('data/jpsi.pkl')
df_psi2s = pd.read_pickle('data/psi2S.pkl')

df_swap = pd.read_pickle('data/k_pi_swap.pkl')
plt.hist(df_swap['K_MC15TuneV1_ProbNNpi'], bins = 500, density = True, histtype = 'step', label = 'nnpi')

plt.hist(df_swap['K_MC15TuneV1_ProbNNk'], bins = 500, density = True, histtype = 'step', label = 'nnk')
plt.legend()
plt.show()
