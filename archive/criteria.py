#!/usr/bin/env python3

import pandas as pd
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


ipchi2_sets = (('mu_plus', 'mu_minus'), ('K', 'Pi'))

def IPCHI2_Selection(Dataframe_real, Dataframe_signal, threshold = 0.9): #This IP should be large, threshold shouble be given in [0-1] to say how many particle we want to keep

    for i in range(2):

        first_col = ipchi2_sets[i][0] + '_IPCHI2_OWNPV'
        second_col = ipchi2_sets[i][1] + '_IPCHI2_OWNPV'

        first_IPCHI2_signal = Dataframe_signal[first_col].to_numpy()
        height, bins = np.histogram(first_IPCHI2_signal, bins = 1000, range = (0,1000))
        N_tot = len(first_IPCHI2_signal)
        SUM = 0
        limit = 0

        for i in range (len(height)):
            SUM+= height[i]

            if SUM/N_tot > 1-threshold:
                limit = bins[i+1]
            break

        #print(limit)
        df_after = Dataframe_real[Dataframe_real[first_col]>limit]
        df_after = df_after[df_after[second_col]>limit]

    return df_after



def Kstar_ENDVERTEX_CHI2(Dataframe_real, Dataframe_Signal, threshold):
    Kstar_ENDVERTEX_CHI2_signal = Dataframe_Signal['Kstar_ENDVERTEX_CHI2'].to_numpy()

    height, bins = np.histogram(Kstar_ENDVERTEX_CHI2_signal, bins = 50)
    N_tot = len(Kstar_ENDVERTEX_CHI2_signal)
    SUM = 0
    limit = 0
    for i in range (len(height)-1,-1,-1):
        SUM+= height[i]
        if SUM/N_tot > 1-threshold:
            limit = bins[i-1]
            break
    print(limit)
    # Remove data below a certain threshold

    df_after = Dataframe_real[Dataframe_real['Kstar_ENDVERTEX_CHI2'] < limit]
    return df_after
