#!/usr/bin/env python3

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

first_stage = (mu_plus_probs, mu_minus_probs)
second_stage = (k_probs, pi_probs)
# first_stage = (mu_plus_probs, mu_minus_probs, k_probs, pi_probs)
# second_stage = ()
'''
format: dataframe in, truncated dataframe out

'''
def hypotheses_compound(dataframe, dummy_frame, thresholds):

    base_threshold, other_threshold = thresholds

    for species_probs in first_stage:

        ### standard pandas dataframe row selection with comparator
        ### this selects rows where BASE threshold is satisfied
        tempframe = dataframe[dataframe[species_probs[0]] > base_threshold]

        ### filters the frame further with remaining OTHER thresholds
        for prob in species_probs[1:]:
            tempframe = tempframe[dataframe[prob] < other_threshold]

    for species_probs in second_stage:

        tempframe = dataframe[(dataframe[species_probs[0]] > base_threshold) | (dataframe[species_probs[1]] > base_threshold)]

        ### filters the frame further with remaining OTHER thresholds
        for prob in species_probs[2:]:
            tempframe = tempframe[dataframe[prob] < other_threshold]

    return tempframe
