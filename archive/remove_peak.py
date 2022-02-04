import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def peak_removal(dataframe,bin_i,bin_f,ratio):

    '''
    Function that removes an unwanted peak (peaking background)
    inputs are:
        dataframe
        bin start & bin end (for q2)
            (make sure the peak is more or less constant throuout the bin)
            (if peak is not constant it will look wonky)
        ratio of height of height that the peak should have over height that it does have
            (must be less than 1)
    
    function randomly selects N readings in the bin which are removed from the dataframe.
    
    function returns the dataframe unchanged expect for values in the bin.
    
    function can be iterated over multiple bins, each with their own ratio
    
    '''
    #remove everything outside thge bin
    df_inter = dataframe[bin_i < dataframe['q2']]
    df_inter = df_inter[bin_f >= df_inter['q2']] 
    
    #calculate number of data to be removed
    N= int((1-ratio)*len(df_inter))
    
    #fixes indices so that they are (1,2,3,...) 
    df_inter.index = range(len(df_inter))
    
    #creates an array of all rows to be removed
    remove_index = random.sample(range(0,len(df_inter)),N)
    
    #drops all rows of index in remove_index
    df_inter = df_inter.drop(index = remove_index)

    #puts back all values that are outside the bin
    df_after = pd.concat([dataframe[bin_i > dataframe['q2']], df_inter, dataframe[bin_f < dataframe['q2']]])
     
    return df_after
    




def whole_peak_removal(dataframe,value_range,bin_num):
    '''
    function that splits the peak into bins and appplies peak_removal to all the bins
    
    inputs are:
        dataframe (totaldataset)
        value_range: array of start and end of the q2 range of the peak
        bin_num: number of bins the peak is split into (takes about a minute to run for 50 bins)
    
    '''
    
    #isolate the value range
    df_peak = dataframe[value_range[0] < dataframe['q2']]
    df_peak = df_peak[value_range[1] >= df_peak['q2']] 
    
    #splits the dataframe into bins with height as the nuber of values in each bin
    height, nbins = np.histogram(df_peak['q2'].to_numpy(), bins = bin_num)
    
    #wanted height is set to the smallest height in the range
    wanted_height = min(height)
    
    #applies peak_removal for all bins
    for i in range(len(height)):
        bin_start = nbins[i]
        bin_end = nbins[i+1]
        bin_height = height[i]
        ratio = wanted_height/bin_height
        dataframe = peak_removal(dataframe,bin_start,bin_end,ratio)
    
    return(dataframe)


'''
#CODE TO RUN IN PIPELINE

import pandas as pd
from remove_peak import *

df = pd.read_pickle('data/total_dataset.pkl')

plt.hist(df['q2'], bins = 500)

df_intermediate = whole_peak_removal(df,[7.5,10.5],50)  
df_final = whole_peak_removal(df_intermediate,[12.5,14.5],50)  

plt.hist(df_final['q2'], bins = 500,histtype = 'step')
'''