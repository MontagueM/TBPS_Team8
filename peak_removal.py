#!/usr/bin/env python3

import pandas as pd
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt

from math_machinery import *



def make_hist(dataframe, bins):

    counts, edges = np.histogram(dataframe, bins = bins)

    midpoints = edges[1:] - (edges[1] - edges[0])/2

    return counts, midpoints


def fit_lorentzian(dataframe):
    pass






df_signal = pd.read_pickle('data/jpsi.pkl')
df_signal2 = pd.read_pickle('data/total_dataset.pkl')



# plt.hist(df_signal['q2'],bins=500,density=True)
plt.hist(df_signal2['q2'],bins=500,density=True, histtype = 'step')

plt.show()
