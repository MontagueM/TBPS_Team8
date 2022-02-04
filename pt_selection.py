import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def Pion_PT_Selection(Dataframe_real, Dataframe_signal, threshold = 0.9): #This IP should be large, threshold shouble be given in [0-1] to say how many particle we want to keep

    Pi_PT_Signal = Dataframe_signal['Pi_PT'].to_numpy()
    height, bins = np.histogram(Pi_PT_Signal, bins = 500, range = (0,6000))
    N_tot = len(Pi_PT_Signal)
    SUM = 0
    limit = 0

    for i in range (len(height)):
        SUM+= height[i]
        #print(SUM)
        if SUM/N_tot > 1-threshold:
        limit = bins[i+1]
        #print(limit)
        break

    #print(limit)
    df_after = Dataframe_real[Dataframe_real['Pi_PT']>limit]

    return df_after

def Kaon_PT_Selection(Dataframe_real, Dataframe_signal, threshold = 0.9): #This IP should be large, threshold shouble be given in [0-1] to say how many particle we want to keep

    K_PT_Signal = Dataframe_signal['K_PT'].to_numpy()
    height, bins = np.histogram(K_PT_Signal, bins = 500, range = (0,8000))
    N_tot = len(K_PT_Signal)
    SUM = 0
    limit = 0

    for i in range (len(height)):

        SUM+= height[i]
        #print(SUM)
        if SUM/N_tot > 1-threshold:
        limit = bins[i+1]
        print(limit)
        break

    #print(limit)
    df_after = Dataframe_real[Dataframe_real['K_PT']>limit]

    return df_after
