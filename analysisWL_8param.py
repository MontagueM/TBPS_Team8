# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 19:34:21 2022

@author: willi
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from iminuit import Minuit
import json
from scipy.optimize import NonlinearConstraint
import scipy.optimize as so
import random
from scipy import integrate




bins_q2 = [[0.1,0.98],[1.1,2.5],[2.5,4],[4,6],[6,8],[15,17],[17,19],[11,12.5],[1,6],[15,17.9]]
num_bins = 30

#function to filter and bin data file
def binq2_filter(data):
    data = data[['q2','phi','costhetal','costhetak']]
    bins = []
    for i in range(len(bins_q2)):
        b = data[data['q2']>bins_q2[i][0]]
        b = b[b['q2']<bins_q2[i][1]]
        bins.append(b)
    return bins

# Load in standard model predictions - copied from skeleton
si_preds = {}
pi_preds = {}
with open("predictions/std_predictions_si.json","r") as _f:
    si_preds = json.load(_f)
with open("predictions/std_predictions_pi.json","r") as _f:
    pi_preds = json.load(_f)

si_frames = []# Will start with si_frames for analysis 
pi_frames = []
for _binNo in si_preds.keys():
    si_frames.append(pd.DataFrame(si_preds[_binNo]).transpose())
    pi_frames.append(pd.DataFrame(pi_preds[_binNo]).transpose())

# for bin0 (q^2 between 0.1 and 0.98) to print predictions do
#print(si_frames[0])


# prepare results dataframe separate from main frame to display results
results = si_frames.copy()
res = [results[i].rename(columns={"val":" SM val","err":" SM err"}) for i in range(len(bins_q2))] 


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

# Function to fit to data - 3 angles and 8 observables
# NOTE: still using flat acceptance
def d2gamma_p_d2q2_d_all_angles(cos_theta_l,cos_theta_k,
                                phi,fl,afb,S3,S4,S5,S7,S8,S9):
    """
    Returns the pdf for angular decay rate for costhetal
    :param fl: f_l observable
    :param afb: a_fb observable
    :param cos_theta_l: cos(theta_l)
    :return:
    """
    # get different cos, sin and phi for final function
    ctl = cos_theta_l
    ctl_2 = ctl**2
    c2tl = 2 * ctl_2 - 1
    stl_2 = 1-ctl_2
    stl = np.sqrt(stl_2)
    s2tl = 2 * ctl * stl
    
    ctk = cos_theta_k
    ctk_2 = ctk**2
    #c2tk = 2 * ctk_2 - 1
    stk_2 = 1-ctk_2
    stk = np.sqrt(stk_2)
    s2tk = 2 * ctk * stk
    
    cp = np.cos(phi)
    cp_2 = cp**2
    c2p = 2*cp_2 -1
    sp = np.sin(phi)
    #sp_2 = sp**2
    s2p = 2 * sp * cp
    
    acceptance = 0.5  # acceptance "function" --> needs changing
    # equation to fit for
    const = 9/(32*np.pi)
    scalar_array = const * ( 0.75*(1-fl)*stk_2 + fl*ctk_2
                            +0.25*(1-fl)*stk_2*c2tl
                            -fl*ctk_2*c2tl + S3*stk_2*stl_2*c2p
                            +S4*s2tk*s2tl*cp +S5*s2tk*stl*cp
                            +(4/3)*afb*stk_2*ctl + S7*s2tk*stl*sp
                            +S8*s2tk*s2tl*sp + S9*stk_2*stl_2*s2p)* acceptance
    
    normalised_scalar_array = scalar_array * 2  # normalising scalar array to account for the 
                                                # non-unity acceptance function
    return normalised_scalar_array


# test integration for normalisation --turns out that angular distribution normalised regardless of 
# values of observables -- maybe useful to keep though in diagnostic of acceptance function when written

def integrate_pdf(cos_theta_l,cos_theta_k,
                  phi,fl, afb,S3,S4,S5,S7,S8,S9):
    I,Ierr = integrate.tplquad(d2gamma_p_d2q2_d_all_angles,-np.pi,np.pi,lambda phi: -1,
                                     lambda phi: 1, lambda phi, cos_theta_k:-1,lambda phi,cos_theta_k:1,
                                     args = (fl, afb,S3,S4,S5,S7,S8,S9))
    return I

# Log likelihood for costhetal with 8 parameters. By minimising this (using minuit) we should obtain
# the optimum fit for the data

def log_likelihood(fl,afb,S3,S4,S5,S7,S8,S9):
    """
    Returns the negative log-likelihood of the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param _bin: number of the bin to fit
    :return:
    """
    currbin = bins[int(_bin)]
    ctl = currbin['costhetal']
    ctk = currbin['costhetak']
    phi = currbin['phi']
    
    normalised_scalar_array = d2gamma_p_d2q2_d_all_angles(ctl,ctk,phi,fl,afb,
                                                          S3,S4,S5,S7,S8,S9)
    return - np.sum(np.log(normalised_scalar_array))

#%% minimising function - again mostly uses skeleton code
bin_number_to_check = 0  # bin that we want to check in more details in the next cell
bin_results_to_check = None

log_likelihood.errordef = Minuit.LIKELIHOOD # think this defines how error deffined
def minimise_fit(bin_no):
    """
    Returns and prints position of minimum and hence optimum fit parameters
    :param bin_no: bin to minimise
    :param profile: True or False to produce profile of minimum

    """
    biny = bins[bin_no]
    cos_theta_l = biny['costhetal']
    cos_theta_k = biny['costhetak']
    phi = biny['phi']
    stp = si_frames[bin_no]['val'].tolist()# Load SM predictions
    #stp2= [stp[i]*random.uniform(0.7,0.9) for i in range(len(stp))]#randomising starting points
    #                                      --> not used currently as still testing minimiser
    ers = si_frames[bin_no]['err'].tolist()
    #note input is function we aim to minimize and starting values for search
    m = Minuit(log_likelihood, fl=stp[0], afb=stp[1],S3=stp[2],S4=stp[3],S5=stp[4],
               S7=stp[5],S8=stp[6],S9=stp[7])
    m.limits=((-1, 1), (-1, 1),(-0.5,0.5),(-0.5,0.5),(-0.5,0.5),(-0.2,0.2),(-0.2,0.2),(-0.2,0.2)) 
    # specifiying where to search for minimum
    # strategy specifies more careful minimisation
    m.strategy=2
    # adding constraint of positive pdf for all angles
    con = lambda fl,afb,S3,S4,S5,S7,S8,S9: d2gamma_p_d2q2_d_all_angles(cos_theta_l,cos_theta_k,
                                                                       phi,fl,afb,S3,S4,S5,S7,S8,S9)
    # use scipy integration with minuit to add constraint
    m.scipy(constraints=NonlinearConstraint(con,0,np.inf))    
    m.hesse()#returning error
    # returning results and errors
    obs = [m.values[i]for i in range (8)]
    obs_errs = [m.errors[i] for i in range(8)]
    res[bin_no]["Fit minuit"] = obs
    res[bin_no]["Fit err"]=obs_errs
    #printing results
    print('\n-------------------------------------------------------')
    print('\nbin number: ',bin_no)
    print("\nFor q^2 range ",bins_q2[bin_no][0],' to ',bins_q2[bin_no][1])
    print(res[bin_no])
    print(f"\nFunction minimum considered valid: {m.fmin.is_valid}")
    print(' ')
    
    return obs,obs_errs,stp,ers

#%%block to display and optionally save all plots for all 10 bins

names = [['provisional_signals/','XGBClf_signals_9_classes_50000_class_size'],
         ['provisional_signals/','XGBClf_signals_10_classes_50000_class_size'],
         ['provisional_signals/','XGBsignals_9classes_20k_class_size'],
         ['provisional_signals/','XGBsignals_10classes_20k_class_size'],
         ['data/','signal']]

def min_plot_all_bins(bins,folder,fname):
    # mostly setting up for the patches in graph -- was slightly pointless
    colors = ['orchid'] # colour of patch -- could add more to vary
    patchesFull = []
    halfwidths = []
    midpoints = []
    obs_full = []
    obs_err_full = []
    obs_names = si_frames[0].index.values.tolist()
    for i in range(len(bins)):#can select which to plot
        # must be a better way of specifying bin without using global
        global _bin
        _bin = i
        # initialise minimisation for each bin of q^2
        obs,obs_err,sm,ers = minimise_fit(i)
        obs_full.append(obs)
        obs_err_full.append(obs_err)
        lowerx = bins_q2[i][0]
        width = bins_q2[i][1]-bins_q2[i][0]
        midpoints.append((bins_q2[i][1]+bins_q2[i][0])/2)
        halfwidths.append(width*0.5)
        patchesROW_q2 = []
        # getting patches for SM prediction and errors
        for j in range(len(sm)):
            loweryObs = sm[j] - ers[j]    
            patchesROW_q2.append(pat.Rectangle((lowerx,loweryObs),width,ers[j]*2,
                                              alpha = 0.3,color = colors[0] ))
        patchesFull.append(patchesROW_q2)
    obs_All = list(zip(*obs_full))
    obs_err_All = list(zip(*obs_err_full))
    patchesAll = list(zip(*patchesFull))
    
    fig, ax = plt.subplots(2,4)
    fig.suptitle('Observables with q2 for '+fname+'.pkl')
    ax = ax.ravel() 
    for x in range(len(obs_All)): # plot graph for each observable              
        #fig, ax = plt.subplots(1,1) # for individual plots
        #fig.suptitle(obs_names[x]+' with q2 for '+fname)
        os = list(obs_All[x])
        os_err = list(obs_err_All[x])
        ax[x].errorbar(midpoints,os,yerr =os_err,xerr=halfwidths,capsize = 4,fmt = 'o',label ='Fit')
        handles,labels = ax[x].get_legend_handles_labels()
        ax[x].set_xlabel('q2')
        ax[x].set_ylabel(obs_names[x])
        # custom handles for patches of SM predictions
        patchy = pat.Patch(color = colors[0] ,alpha = 0.5,label = 'SM prediction')
        handles.append(patchy)
        labels.append('SM prediction')
        ax[x].legend(handles,labels)
        # silly patches thing for SM predictions
        for y in range(len(patchesAll[x])):
            ax[x].add_patch(patchesAll[x][y])
    plt.plot()


#%% add files to process here with folder path and file name seperate

names = [['provisional_signals/','XGBClf_signals_9_classes_50000_class_size'],
         ['provisional_signals/','XGBClf_signals_10_classes_50000_class_size'],
         ['provisional_signals/','XGBsignals_9classes_20k_class_size'],
         ['provisional_signals/','XGBsignals_10classes_20k_class_size'],
         ['provisional_signals/','XGBClf_signals_9_classes_400000_class_size'],
         ['data/','signal']]


# choose files to process
for i in [5]:
    print('\nFor '+names[i][0]+names[i][1])
    file = pd.read_pickle(names[i][0]+names[i][1]+'.pkl')
    bins = binq2_filter(file)
    min_plot_all_bins(bins,names[i][0],names[i][1])













