import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
The impact parameter chi squared for the B0 particle should be small as we want them to all come from the same primary vertex
"""

def b0_ipchi2(Dataframe_real, Dataframe_Signal, threshold):
    B0_IPCHI2_OWNPV_signal = Dataframe_Signal['B0_IPCHI2_OWNPV'].to_numpy()

    # Create a threshold as a percentage of the particles
    height, bins = np.histogram(B0_IPCHI2_OWNPV_signal, bins = 50)
    N_tot = len(B0_IPCHI2_OWNPV_signal)
    SUM = 0
    limit = 0
    for i in range (len(height)-1,-1,-1):
        SUM+= height[i]
        if SUM/N_tot > 1-threshold:
            limit = bins[i-1]
            break
    #print(limit)

    # Remove data below a certain threshold
    df_after = Dataframe_real[Dataframe_real['B0_IPCHI2_OWNPV'] < limit]
    print('b0 ipchi2 limit:', limit)

    return df_after
