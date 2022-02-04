import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
The impact parameter chi squared for the B0 particle should be small as we want them to all come from the same primary vertex
"""

def b0_vertex_ip_chi2(Dataframe_real, Dataframe_Signal, threshold):

    target_cols = ('B0_IPCHI2_OWNPV', 'B0_ENDVERTEX_CHI2')

    for target in target_cols:
        signal = Dataframe_Signal[target].to_numpy()

    # Create a threshold as a percentage of the particles
        height, bins = np.histogram(signal, bins = 50)
        N_tot = len(B0_IPCHI2_OWNPV_signal)
        SUM = 0
        limit = 0
        for i in range (len(height)-1,-1,-1):
            SUM+= height[i]
            if SUM/N_tot > 1-threshold:
                limit = bins[i-1]
                break

        # Remove data below a certain threshold
        df_after = Dataframe_real[Dataframe_real[target] < limit]

    return df_after
