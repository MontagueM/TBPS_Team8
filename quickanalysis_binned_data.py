# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 11:21:51 2022

@author: willi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
import json

#%% define variables

# binning regime for the b0 candidates to conform to that in SM predictions - in q^2 ranges
bins_q2 = [[0.1,0.98],[1.1,2.5],[2.5,4],[4,6],[6,8],[15,17],[17,19],[11,12.5],[1,6],[15,17.9]]

#number of bins of costhetal for analysis
num_bins = 20

# and load in prelim data from first Machine learning iteration - saved in file called binned_data
def load_data(i):
    name = 'binned_data/bin' + str(i)+ '.pkl'
    b = pd.read_pickle(name)
    return b
bins = []
for i in range(10):
    bins.append(load_data(i))

# Load in standard model predictions - copied from skeleton
si_preds = {}
pi_preds = {}
with open("data/std_predictions_si.json","r") as _f:
    si_preds = json.load(_f)
with open("data/std_predictions_pi.json","r") as _f:
    pi_preds = json.load(_f)

si_frames = []# Will start with si_frames for analysis 
pi_frames = []
for _binNo in si_preds.keys():
    si_frames.append(pd.DataFrame(si_preds[_binNo]).transpose())
    pi_frames.append(pd.DataFrame(pi_preds[_binNo]).transpose())

# for bin0 (q^2 between 0.1 and 0.98) to print predictions do
#print(si_frames[0])


#%% Display as histogram - copied from skeleton
#first analysis is in costhetal so set as default
def plot_binned_hist(bin_no,angle = 'costhetal'):
    plt.figure(bin_no)
    plt.hist(bins[bin_no][angle], bins=num_bins,range=[-1,1], density=True)
    title = 'q2 Bin: '+str(bins_q2[bin_no][0])+' to ' + str(bins_q2[bin_no][1])
    plt.title(title)
    plt.xlabel(r'$cos(\theta_l)$')
    plt.ylabel(r'Number of candidates')
    plt.grid()
    plt.show()

#plot_binned_hist(0)#test

#%% Fucntions for analysis -  from skeleton code

# Function to fit to histogram - copied from skeleton code
# NOTE: still using flat acceptance
def d2gamma_p_d2q2_dcostheta(fl, afb, cos_theta_l):
    """
    Returns the pdf for angular decay rate for costhetal
    :param fl: f_l observable
    :param afb: a_fb observable
    :param cos_theta_l: cos(theta_l)
    :return:
    """
    ctl = cos_theta_l
    c2tl = 2 * ctl ** 2 - 1
    acceptance = 0.5  # acceptance "function"
    scalar_array = 3/8 * (3/2 - 1/2 * fl + 1/2 * c2tl * (1 - 3 * fl) + 8/3 * afb * ctl) * acceptance
    normalised_scalar_array = scalar_array * 2  # normalising scalar array to account for the non-unity acceptance function
    return normalised_scalar_array

# Log likelihood for costhetal with two variables only. By minimising this (using minuit) we should obtain
# the optimum fit for the data
# could do with better understanding of full angular decay rate in other angles also
def log_likelihood(fl, afb, _bin):
    """
    Returns the negative log-likelihood of the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param _bin: number of the bin to fit
    :return:
    """
    _bin = bins[int(_bin)]
    ctl = _bin['costhetal']
    normalised_scalar_array = d2gamma_p_d2q2_dcostheta(fl=fl, afb=afb, cos_theta_l=ctl)
    return - np.sum(np.log(normalised_scalar_array))

#%% minimising function - again mostly uses skeleton code
bin_number_to_check = 0  # bin that we want to check in more details in the next cell
bin_results_to_check = None

log_likelihood.errordef = Minuit.LIKELIHOOD # think this defines how error deffined
def minimise_fit(bin_no,profile =False):
    """
    Returns and prints position of minimum and hence optimum fit parameters
    :param bin_no: bin to minimise
    :param profile: True or False to produce profile of minimum

    """
    decimal_places = 5
    starting_point = [0.2,0.2]
    #note input is function we aim to minimize and starting values for search
    m = Minuit(log_likelihood, fl=starting_point[0], afb=starting_point[1], _bin=bin_no)
    m.fixed['_bin'] = True  # fixing the bin number as we don't want to optimize it
    m.limits=((-1, 1), (-1, 1), None) # specifiying where to search for minimum
    m.migrad()#specifying to use gradient descent for minimum search
    m.hesse()#returning error
    if profile == True: # can plot profile of minimum, should be parabolic ideally
        bin_results_to_check = m
        plt.figure(figsize=(8, 5))
        plt.subplot(221)
        bin_results_to_check.draw_mnprofile('afb', bound=3)
        plt.subplot(222)
        bin_results_to_check.draw_mnprofile('fl', bound=3)
        plt.tight_layout()
        plt.show()
    fl = m.values[0] # returning found minimum values
    afb = m.values[1]
    fl_err = m.errors[0]
    afb_err = m.errors[1]
    print(f"\nBin {bin_no}:\nFL {np.round(fl, decimal_places)} pm {np.round(fl_err, decimal_places)},", 
          f"\nAFB {np.round(afb, decimal_places)} pm {np.round(afb_err, decimal_places)}.",
          f"\nFunction minimum considered valid: {m.fmin.is_valid}")
    return fl,fl_err,afb,afb_err





#%% minimiing and plotting code - mostly uses skelton code


def minimise_fit_plot(bin_no, profile=False,save=False):
    """
    Returns and prints position of minimum and hence optimum fit parameters
    as well as plotting the histogram for the bin with the fit on as well as the
    fit predicted by the standard model
    :param bin_no: bin to minimise
    :param profile: True or False to produce profile of minimum
    :param save: True or False to save figures produced

    """
    plt.figure(bin_no)
    decimal_places = 5
    SM_afb = si_frames[bin_no]['val']['AFB']#extracting standard model fit parameters
    SM_afb_err = si_frames[bin_no]['err']['AFB']
    SM_fl = si_frames[bin_no]['val']['FL']
    SM_fl_err = si_frames[bin_no]['err']['FL']
    cos_theta_l_bin = bins[bin_no]['costhetal']
    hist, _bins, _ = plt.hist(cos_theta_l_bin, bins=num_bins,range=[-1,1]) # plot hist
    x = np.linspace(-1, 1, num_bins)
    pdf_multiplier = np.sum(hist) * (np.max(cos_theta_l_bin) - np.min(cos_theta_l_bin)) / num_bins
    fl,fl_err,afb,afb_err = minimise_fit(bin_no,profile)
    print(f"SM FL {np.round(SM_fl, decimal_places)} pm {np.round(SM_fl_err, decimal_places)},")
    print(f"SM AFB {np.round(SM_afb, decimal_places)} pm {np.round(SM_afb_err, decimal_places)},")
    y_fit = d2gamma_p_d2q2_dcostheta(fl, afb, cos_theta_l=x) * pdf_multiplier
    y_actual = d2gamma_p_d2q2_dcostheta(fl=SM_fl, afb=SM_afb, cos_theta_l=x) * pdf_multiplier
    title = r'For range of $q^{2}$: '+str(bins_q2[bin_no][0])+' to ' + str(bins_q2[bin_no][1])
    plt.suptitle(title,y=0.95,fontsize=18)#adding titles
    plt.title(f"FL -Theory: {np.round(SM_fl,decimal_places)}, Fit: {np.round(fl,decimal_places)}" +
                 f"\nAFB -Theory: {np.round(SM_afb,decimal_places)}, Fit: {np.round(afb,decimal_places)}",
                 fontsize=10)
    plt.plot(x, y_fit, label=f'Fit for bin {bin_no}')
    plt.plot(x, y_actual, label=f'SM predcition for bin {bin_no}')
    plt.ylim(bottom = 0)
    plt.xlabel(r'$cos(\theta_l)$')
    plt.ylabel(r'Number of candidates')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if save ==True:
        plt.savefig('plots/Bin'+str(bin_no),dpi=300)
    plt.show()
    return fl,fl_err,afb,afb_err

#%%block to display and optionally save all plots for all 10 bins

fls = []
fl_errs = []
afbs = []
afb_errs = []

for i in range(10):#can select which to plot
    f,fe,a,ae = minimise_fit_plot(i,profile=False,save=False)
    fls.append(f)
    fl_errs.append(fe)
    afbs.append(a)
    afb_errs.append(ae)








