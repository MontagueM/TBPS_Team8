import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read in the total data and signal dataframes
df = pd.read_pickle('data/total_dataset.pkl')
df_signal = pd.read_pickle('data/signal.pkl')

def Kstar_ENDVERTEX_CHI2(Dataframe_real, Dataframe_Signal, threshold):
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
    print(limit)
    # Remove data below a certain threshold

    df_after = Dataframe_real[Dataframe_real['Kstar_ENDVERTEX_CHI2'] < limit]
    return df_after



# df_after = Kstar_ENDVERTEX_CHI2(df, df_signal, 0.95)

# plt.hist(df['Kstar_ENDVERTEX_CHI2'], histtype = 'step', label = 'Total dataset',range=(0,10),bins=50)#, density = True)
# plt.hist(df_signal['Kstar_ENDVERTEX_CHI2'], histtype = 'step', label = 'Simulated dataset',range=(0,10),bins=50)#, density = True)
# plt.hist(df_after['Kstar_ENDVERTEX_CHI2'], histtype = 'step', label = 'Post selection',range=(0,10),bins=50)#, density = True)
# plt.legend()
# plt.show()