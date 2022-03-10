import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read in the total data and signal dataframes
# df = pd.read_pickle('data/total_dataset.pkl')
# df_signal = pd.read_pickle('data/signal.pkl')

def kstar_endvertex_chi2(Dataframe_real, Dataframe_Signal, threshold):
    Kstar_ENDVERTEX_CHI2_signal = Dataframe_Signal['Kstar_ENDVERTEX_CHI2'].to_numpy()

    # Create a threshold as a percentage of the particles
    height, bins = np.histogram(Kstar_ENDVERTEX_CHI2_signal, bins = 50)
    N_tot = len(Kstar_ENDVERTEX_CHI2_signal)
    SUM = 0
    limit = 0
    for i in range (len(height)-1,-1,-1):
        SUM+= height[i]
        if SUM/N_tot > 1-threshold:
            limit = bins[i-1]
            break
    # print(limit)
    # Remove data below a certain threshold

    print('kstar_endvertex chi2 limit:', limit)
    df_after = Dataframe_real[Dataframe_real['Kstar_ENDVERTEX_CHI2'] < limit]
    return df_after
