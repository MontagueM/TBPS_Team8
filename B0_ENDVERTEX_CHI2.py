import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_pickle('data/total_dataset.pkl')
df_signal = pd.read_pickle('data/signal.pkl')

# Read in the total data and signal dataframes
def B0_ENDVERTEX_CHI2(Dataframe_real, Dataframe_Signal, threshold):
    B0_ENDVERTEX_CHI2_signal = Dataframe_Signal['B0_ENDVERTEX_CHI2'].to_numpy()
    
    # Create a threshold as a percentage of the particles
    height, bins = np.histogram(B0_ENDVERTEX_CHI2_signal, bins = 50)
    N_tot = len(B0_ENDVERTEX_CHI2_signal)
    SUM = 0
    limit = 0
    for i in range (len(height)-1,-1,-1):
        SUM+= height[i]
        if SUM/N_tot > 1-threshold:
            limit = bins[i-1]
            break
    print(limit)
    # Remove data below a certain threshold

    df_after = Dataframe_real[Dataframe_real['B0_ENDVERTEX_CHI2'] < limit]
    return df_after

# df_after = B0_ENDVERTEX_CHI2(df, df_signal, 0.80)

# plt.hist(df['B0_ENDVERTEX_CHI2'], histtype = 'step', label = 'Total dataset',range=(0,10),bins=50, density = True)
# plt.hist(df_signal['B0_ENDVERTEX_CHI2'], histtype = 'step', label = 'Simulated dataset',range=(0,10),bins=50, density = True)
# plt.hist(df_after['B0_ENDVERTEX_CHI2'], histtype = 'step', label = 'Post selection',range=(0,10),bins=50, density = True)
# plt.legend()
# plt.show()
