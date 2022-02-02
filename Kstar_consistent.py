import pandas as df
import scipy.optimize as opt
import numpy as np
from math_machinery import *

"""
This selection criteria tests the consistency of the Kstar mass
"""

def Kstar_consistent(Dataframe_real, Dataframe_Signal, threshold):

    #Both input should have type of pandas.Dataframe

    Kstar_MM_Signal = Dataframe_Signal['Kstar_MM'].to_numpy()
    Kstar_MM_Data = Dataframe_Signal['Kstar_MM'].to_numpy()
    height, nbins = np.histogram(Kstar_MM_Signal, bins = 50)
    mid_bin = []

    for i in range (0,len(height)):
        mid_bin.append((nbins[i]+nbins[i+1])/2)
    mid_bin = np.array(mid_bin)

    fits, cov = opt.curve_fit(lorenzian, mid_bin, height)
    x0 = fits[0]
    gamma = fits[1]

    df_after = Dataframe_real[x0 - threshold*gamma < Dataframe_real['Kstar_MM']]
    df_after = df_after[ df_after['Kstar_MM'] < x0 + threshold*gamma]


    return df_after