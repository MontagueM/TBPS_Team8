import pandas as df
import scipy.optimize as opt
import numpy as np

mu_plus_probs =('mu_plus_MC15TuneV1_ProbNNmu', 'mu_plus_MC15TuneV1_ProbNNk', 'mu_plus_MC15TuneV1_ProbNNpi','mu_plus_MC15TuneV1_ProbNNe', 'mu_plus_MC15TuneV1_ProbNNp')

mu_minus_probs =('mu_minus_MC15TuneV1_ProbNNmu','mu_minus_MC15TuneV1_ProbNNk',
'mu_minus_MC15TuneV1_ProbNNpi', 'mu_minus_MC15TuneV1_ProbNNe', 'mu_minus_MC15TuneV1_ProbNNp')

k_probs =('K_MC15TuneV1_ProbNNk','K_MC15TuneV1_ProbNNpi',
'K_MC15TuneV1_ProbNNmu','K_MC15TuneV1_ProbNNe',
'K_MC15TuneV1_ProbNNp')

pi_probs =('Pi_MC15TuneV1_ProbNNpi','Pi_MC15TuneV1_ProbNNk',
'Pi_MC15TuneV1_ProbNNmu', 'Pi_MC15TuneV1_ProbNNe',
'Pi_MC15TuneV1_ProbNNp')

species = (mu_plus_probs, mu_minus_probs, k_probs, pi_probs)

'''
format: dataframe in, truncated dataframe out

'''
def hypotheses_compound(dataframe, dataframe_dummy, thresholds):

    base_threshold, other_threshold = thresholds

    print(dataframe.shape)
    for species_probs in species:

        ### standard pandas dataframe row selection with comparator
        ### this selects rows where BASE threshold is satisfied
        tempframe = dataframe[dataframe[species_probs[0]] > base_threshold]

        ### filters the frame further with remaining OTHER thresholds
        for i in range(1, 5):
            tempframe = tempframe[dataframe[species_probs[i]] < other_threshold]

    print(tempframe.shape)
    return tempframe
