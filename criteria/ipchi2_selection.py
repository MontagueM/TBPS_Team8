import pandas as df
import scipy.optimize as opt
import numpy as np

"""
This selection criteria tests the impact paramter chi squared for all daughter particles, which we want to be large
"""

ipchi2_sets = (('mu_plus', 'mu_minus'), ('K', 'Pi'))

def ipchi2_selection(Dataframe_real, Dataframe_signal, threshold = 0.9): #This IP should be large, threshold shouble be given in [0-1] to say how many particle we want to keep

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
