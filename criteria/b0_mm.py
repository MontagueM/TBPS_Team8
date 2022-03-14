#!/usr/bin/env python3



"""
This selection criteria tests the consistency of the Kstar mass
"""

def b0_mm_consistent(Dataframe_real, threshold):

    #Both input should have type of pandas.Dataframe
    df_after = Dataframe_real[Dataframe_real['B0_MM'] < threshold]

    return df_after
