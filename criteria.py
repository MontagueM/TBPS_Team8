#!/usr/bin/env python3

import pandas as df
import scipy.optimize as opt
import numpy as np

from math_machinery import *

mu_plus_probs =('mu_plus_MC15TuneV1_ProbNNmu', 'mu_plus_MC15TuneV1_ProbNNk', 'mu_plus_MC15TuneV1_ProbNNpi','mu_plus_MC15TuneV1_ProbNNe', 'mu_plus_MC15TuneV1_ProbNNp')

mu_minus_probs =('mu_minus_MC15TuneV1_ProbNNmu','mu_minus_MC15TuneV1_ProbNNk',
'mu_minus_MC15TuneV1_ProbNNpi', 'mu_minus_MC15TuneV1_ProbNNe', 'mu_minus_MC15TuneV1_ProbNNp')

k_probs =('K_MC15TuneV1_ProbNNk','K_MC15TuneV1_ProbNNpi',
'K_MC15TuneV1_ProbNNmu','K_MC15TuneV1_ProbNNe',
'K_MC15TuneV1_ProbNNp')

pi_probs =('Pi_MC15TuneV1_ProbNNpi','Pi_MC15TuneV1_ProbNNk',
'Pi_MC15TuneV1_ProbNNmu', 'Pi_MC15TuneV1_ProbNNe',
'Pi_MC15TuneV1_ProbNNp')

species = (mu_plus_probs, mu_minus_probs, k_probs, pi_probs)

'''
format: dataframe in, truncated dataframe out


'''
def hypotheses_compound(dataframe, base_threshold, other_threshold):

    print(dataframe.shape)
    for species_probs in species:

        ### standard pandas dataframe row selection with comparator
        ### this selects rows where BASE threshold is satisfied
        tempframe = dataframe[dataframe[species_probs[0]] > base_threshold]

        ### filters the frame further with remaining OTHER thresholds
        for i in range(1, 5):
            tempframe = tempframe[dataframe[species_probs[i]] < other_threshold]

    print(tempframe.shape)
    return tempframe






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
