import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def peaking_selection_psi2s(Dataframe_real, Dataframe_psi2S, threshold = 0.994):

    q2_psi2S = Dataframe_psi2S['q2'].to_numpy()
    r = (1-threshold)/2
    height, bins = np.histogram(q2_psi2S, bins = 500, range = (12,15))

    N_tot = len(q2_psi2S)
    SUM1 = 0
    limit1 = 0

    for i in range (len(height)):
        SUM1+= height[i]
        #print(SUM)
        if SUM1/N_tot > r:
            limit1 = bins[i+1]
            #print(limit)
            break
    SUM2 = 0
    limit2 = 0
    for i in range (len(height)-1,-1,-1):
        SUM2+= height[i]
        if SUM2/N_tot > r:
            limit2 = bins[i-1]
            break

    print('Upper cut, lower cut, threshold is:', limit1, limit2, threshold)
    df_after = Dataframe_real[Dataframe_real['q2']>limit2]
    #print(len(df_after))
    df_after2 = Dataframe_real[Dataframe_real['q2']<limit1]
    df_final = pd.merge(df_after,df_after2,"outer")

    return df_final


def peaking_selection_jpsi(Dataframe_real, Dataframe_jpsi, threshold = 0.996):

    q2_Jpsi = Dataframe_jpsi['q2'].to_numpy()
    r = (1-threshold)/2
    height, bins = np.histogram(q2_Jpsi, bins = 500, range = (7,11))

    N_tot = len(q2_Jpsi)
    SUM1 = 0
    limit1 = 0

    for i in range (len(height)):

        SUM1+= height[i]
        #print(SUM)
        if SUM1/N_tot > r:
            limit1 = bins[i+1]
            #print(limit)
            break
    SUM2 = 0
    limit2 = 0

    for i in range (len(height)-1,-1,-1):
        SUM2+= height[i]
        if SUM2/N_tot > r:
            limit2 = bins[i-1]
            break

    print('Upper cut, lower cut, threshold is:', limit1, limit2, threshold)
    df_after = Dataframe_real[Dataframe_real['q2']>limit2]
    #print(len(df_after))
    df_after2 = Dataframe_real[Dataframe_real['q2']<limit1]
    df_final = pd.merge(df_after,df_after2,"outer")

    return df_final
