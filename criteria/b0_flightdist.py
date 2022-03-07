
import numpy as np

def b0_fdchi2(Dataframe_real, Dataframe_signal, threshold = 0.995): #This IP should be large, threshold shouble be given in [0-1] to say how many particle we want to keep

    K_PT_Signal = Dataframe_signal['B0_FDCHI2_OWNPV'].to_numpy()
    height, bins = np.histogram(K_PT_Signal, bins = 500, range = (0,2000))
    N_tot = len(K_PT_Signal)
    SUM = 0
    limit = 0

    for i in range (len(height)):

        SUM+= height[i]
        #print(SUM)
        if SUM/N_tot > 1-threshold:
            limit = bins[i+1]
            break

    #print(limit)
    df_after = Dataframe_real[Dataframe_real['B0_FDCHI2_OWNPV']>limit]
    print('b0 flight distance limit', limit)
    return df_after



def kstar_fdchi2(Dataframe_real, Dataframe_signal, threshold = 0.99): #This IP should be large, threshold shouble be given in [0-1] to say how many particle we want to keep

    K_PT_Signal = Dataframe_signal['Kstar_FDCHI2_OWNPV'].to_numpy()
    height, bins = np.histogram(K_PT_Signal, bins = 1000, range = (0,2000))
    N_tot = len(K_PT_Signal)
    SUM = 0
    limit = 0

    for i in range (len(height)):

        SUM+= height[i]
        #print(SUM)
        if SUM/N_tot > 1-threshold:
            limit = bins[i+1]
            break

    df_after = Dataframe_real[Dataframe_real['Kstar_FDCHI2_OWNPV']>limit]
    print('kstar flight distance limit', limit)

    return df_after