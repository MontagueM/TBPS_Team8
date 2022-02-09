#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

from criteria import *


'''

input are dataframes, we should keep all data in a /data folder
by convention, which is one level beneath the pipeline directory

This file obtains cuts for the B0 invariant mass that clearly not part of
the resonant B0 peak = very confident that they are part of combinatorial
background and NOT B0
'''


df_real = pd.read_pickle('data/total_dataset.pkl')

def b0mm_cut(dataframe, threshold):

    df_cut = df_real[df_real['B0_MM'] < threshold]
    trash = df_real[df_real['B0_MM'] > threshold]

    return df_cut, trash

df_cut, trash = b0mm_cut(df_real, 5380)

print('\n\nPLEASE MAKE A FOLDER CALLED OUTPUT FIRSTTTTT\n\n')
trash.to_pickle('output/b0_mm_trash0.pkl')


plt.hist(df_cut['B0_MM'], bins = 500, density = True, histtype = 'step', label = 'peak filtered')
plt.hist(df_real['B0_MM'], bins = 500, density = True, histtype = 'step', label = 'peak filtered')

plt.show()
