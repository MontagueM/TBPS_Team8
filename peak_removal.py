#!/usr/bin/env python3

import pandas as pd
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from math_machinery import *



def make_hist(dataframe,bins,column='q2',ranges = (0,20)):

    counts, edges = np.histogram(dataframe[column], bins = bins,range = ranges)

    midpoints = edges[1:] - (edges[1] - edges[0])/2

    return counts, midpoints


def fit_curve(dataframe,equation,rangelist=[(0,20)]):
    popt_list = []
    pcov_list= []
    
    for ranges in rangelist:
        counts,midpoints = make_hist(dataframe,bins = 1000,ranges = ranges)
        popt,pcov = opt.curve_fit(equation,midpoints,counts,p0=(11,1,150000))
        popt_list.append(popt)
        pcov_list.append(pcov)
    return popt_list,pcov_list






df_signal = pd.read_pickle('data/jpsi.pkl')
df_signal2 = pd.read_pickle('data/total_dataset.pkl')



# plt.hist(df_signal['q2'],bins=500,density=True)
plt.hist(df_signal2['q2'],bins=500,density=True, histtype = 'step')
ct,ed = make_hist(df_signal2,1000)
plt.plot(ed,ct)
plt.show()
#%%
ppt,pcv = fit_curve(df_signal2,lorenzian,[(9,11),(11,20)])
plt.plot(ed,gaussian(ed,ppt[0][0],ppt[0][1],15000))
plt.plot(ed,gaussian(ed,*ppt[1]))
plt.plot(ed,ct)
#%%
np.histogram(df_signal2['q2'],bins=500,range=(0.,20.))