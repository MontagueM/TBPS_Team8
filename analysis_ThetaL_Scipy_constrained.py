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

# setting up datafram with required observables (AFB and FL) to append fitted values to
results = si_frames.copy()
res = [results[i].rename(columns={"val":" SM val","err":" SM err"}) for i in range(len(bins_q2))] 

res = [res[i].iloc[[0,1]] for i in range(len(bins_q2))]

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
def d2gamma_p_d2q2_dcostheta(cos_theta_l,x):
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
    fl,afb = [x[i] for i in range(len(x))]
    scalar_array = 3/8 * (3/2 - 1/2 * fl + 1/2 * c2tl * (1 - 3 * fl) + 8/3 * afb * ctl) * acceptance
    normalised_scalar_array = scalar_array * 2  # normalising scalar array to account for the non-unity acceptance function
    return normalised_scalar_array

# Log likelihood for costhetal with two variables only. By minimising this (using minuit) we should obtain
# the optimum fit for the data
# could do with better understanding of full angular decay rate in other angles also
def log_likelihood(x):
    """
    Returns the negative log-likelihood of the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param _bin: number of the bin to fit
    :return:
    """
    currbin = bins[int(_bin)]
    ctl = currbin['costhetal']
    normalised_scalar_array = d2gamma_p_d2q2_dcostheta(ctl,x)
    return - np.sum(np.log(normalised_scalar_array))

#%% minimising function - again mostly uses skeleton code
bin_number_to_check = 0  # bin that we want to check in more details in the next cell
bin_results_to_check = None

log_likelihood.errordef = Minuit.LIKELIHOOD # think this defines how error deffined
def minimise_fit(bin_no,afb_s,fl_s):
    """
    Returns and prints position of minimum and hence optimum fit parameters
    :param bin_no: bin to minimise
    :param profile: True or False to produce profile of minimum
    """
    biny = bins[bin_no]
    cos_theta_l = biny['costhetal']
    starting_point = [fl_s,afb_s] #  use SM predictions as starting point
    # setting a constraint of positive pdf for all angles
    con = lambda x: d2gamma_p_d2q2_dcostheta(cos_theta_l,x)
    # setting bounds of AFB and FL to find optimum in
    bound = ((-1,1), (-1, 1))
    ftol = 1e-9
    print('\n----------------------------------------------')
    print('bin number: ',bin_no)
    print("For q^2 range ",bins_q2[bin_no][0],' to ',bins_q2[bin_no][1])
    print('\n')
    # begininng minimisation with scipy and SLSQP method
    resMin = so.minimize(log_likelihood,starting_point,method = 'SLSQP', bounds=bound, 
                         constraints=NonlinearConstraint(con,0,np.inf),tol = ftol,options = {'disp':True})
    obs = [resMin.x[i] for i in range(2)]
    obs_errs = [resMin.x[i]*0.05 for i in range(2)]
    res[bin_no]["Fit scipy"]=obs
    # print result
    print('\n')
    print(res[bin_no])
    return obs[0],obs_errs[0],obs[1],obs_errs[1]





#%% minimiing and plotting code - mostly uses skelton code


def minimise_fit_plot(bin_no,FolderName = None,save=False,show =False):
    """
    Returns and prints position of minimum and hence optimum fit parameters
    as well as plotting the histogram for single bin with the fit on as well as the
    fit predicted by the standard model
    :param bin_no: bin to minimise
    :param profile: True or False to produce profile of minimum
    :param save: True or False to save figures produced
    """
    decimal_places = 5
    #obtain starting guess --> use sm prediction at this stage
    SM_afb = si_frames[bin_no]['val']['AFB']#extracting standard model fit parameters
    SM_fl = si_frames[bin_no]['val']['FL']
    cos_theta_l_bin = bins[bin_no]['costhetal']

    fl,fl_err,afb,afb_err = minimise_fit(bin_no,SM_afb,SM_fl)
    if save == True or show ==True:
        plt.figure(FolderName +' bin ' + str(bin_no))
        hist, _bins, _ = plt.hist(cos_theta_l_bin, bins=num_bins,range=[-1,1],histtype='step') # plot hist
        # obtaining fit lines from histogram for SM and result of minimser
        c = np.linspace(-1, 1, num_bins)
        pdf_multiplier = np.sum(hist) * (np.max(cos_theta_l_bin) - np.min(cos_theta_l_bin)) / num_bins
        y_fit = d2gamma_p_d2q2_dcostheta(cos_theta_l=c,x=[fl,afb]) * pdf_multiplier
        y_actual = d2gamma_p_d2q2_dcostheta(cos_theta_l=c,x=[SM_fl,SM_afb]) * pdf_multiplier
        title = FolderName+' for range of $q^{2}$: '+str(bins_q2[bin_no][0])+' to ' + str(bins_q2[bin_no][1])
        plt.suptitle(title,y=0.95,fontsize=12)#adding titles
        plt.title(f"FL -Theory: {np.round(SM_fl,decimal_places)}, Fit: {np.round(fl,decimal_places)}" +
                  f"\nAFB -Theory: {np.round(SM_afb,decimal_places)}, Fit: {np.round(afb,decimal_places)}",
                 fontsize=10)
        plt.plot(c, y_fit, label=f'Minuit NLL fit for bin {bin_no}') #  minimser fit line
        plt.plot(c, y_actual, label=f'SM prediction for bin {bin_no}') #  SM fit line
        plt.ylim(bottom = 0)
        plt.xlabel(r'$cos(\theta_l)$')
        plt.ylabel(r'Number of candidates')
        plt.legend()
        plt.grid()
        plt.tight_layout() 
        if save == True:
            plt.savefig(FolderName+'/Bin'+str(bin_no),dpi=300)
        if show == True:
            plt.show()
        if show == False:
            plt.close()# closing by default as to not spam user with graphs
    return fl,fl_err,afb,afb_err

#%%block to display and optionally save all plots for all 10 bins

# Function to process all bins of a file and then give full observables with q^2 plot
def min_plot_all_bins(bins,folder,fname,show=False,save=False):# can plot histograms if desired, and save if needed
    # saving requires folder set up with same name as data without .pkl
    patchesAFB = []
    patchesFL = []
    halfwidths = []
    midpoints = []
    fls = []
    fl_errs = []
    afbs = []
    afb_errs = []
    for i in range(len(bins)):#can select which to plot
        # definitely a better way to set _bin without doing global but oh well
        global _bin
        _bin = i
        f,fe,a,ae = minimise_fit_plot(i,fname,save=save,show =show) #running plot regime
        fls.append(f)
        fl_errs.append(fe)
        afbs.append(a)
        afb_errs.append(ae)
        # mostly pointless code to get patches as SM prediction
        SM_afb = si_frames[i]['val']['AFB']#extracting standard model fit parameters
        SM_afb_err = si_frames[i]['err']['AFB']
        SM_fl = si_frames[i]['val']['FL']
        SM_fl_err = si_frames[i]['err']['FL']
        lowerx = bins_q2[i][0]
        width = bins_q2[i][1]-bins_q2[i][0]
        midpoints.append((bins_q2[i][1]+bins_q2[i][0])/2)
        halfwidths.append(width*0.5)
        loweryAFB = SM_afb - SM_afb_err
        loweryFL = SM_fl - SM_fl_err
        patchesAFB.append(pat.Rectangle((lowerx,loweryAFB),width,SM_afb_err*2,alpha = 0.5,color ='cornflowerblue' ))
        patchesFL.append(pat.Rectangle((lowerx,loweryFL),width,SM_fl_err*2,alpha = 0.5,color = 'orchid'))
        
    # code to plot the two subplots
    fig , (ax1,ax2) = plt.subplots(2,1)
    fig.suptitle('Observables with q2 for '+fname)
    ax1.errorbar(midpoints,afbs,yerr =afb_errs,xerr=halfwidths,fmt = 'o',capsize = 4,label = 'Fit result')
    handles,labels = ax1.get_legend_handles_labels()
    bluepatch = pat.Patch(color = 'cornflowerblue' ,alpha = 0.5,label = 'SM prediction')
    handles.append(bluepatch)
    labels.append('SM prediction')
    ax1.legend(handles,labels)
    ax1.set_xlabel('q2')
    ax1.set_ylabel('Afb')
    for i in range(len(patchesAFB)):
        ax1.add_patch(patchesAFB[i])
    ax2.errorbar(midpoints,fls,yerr =fl_errs,xerr=halfwidths,fmt = 'o',capsize = 4,label = 'Fit Result')
    handles1,labels1 = ax1.get_legend_handles_labels()
    orchidpatch = pat.Patch(color = 'orchid' ,alpha = 0.5,label = 'SM prediction')
    handles1.append(orchidpatch)
    labels1.append('SM prediction')
    ax2.legend(handles1,labels1)
    ax2.set_xlabel('q2')
    ax2.set_ylabel('Fl')
    for i in range(len(patchesFL)):
        ax2.add_patch(patchesFL[i])
    fig.tight_layout()
    fig.savefig('AFB_FL_withQ2/Plot'+fname,dpi=400)
    plt.show()


#%% 
# add folder path and file name for those to be analysed here - .pkl NOT needed
# First four found on teams, last one is from supplied dataset from Mitesh
names = [['provisional_signals/','XGBClf_signals_9_classes_50000_class_size'],
         ['provisional_signals/','XGBClf_signals_10_classes_50000_class_size'],
         ['provisional_signals/','XGBsignals_9classes_20k_class_size'],
         ['provisional_signals/','XGBsignals_10classes_20k_class_size'],
         ['provisional_signals/','XGBClf_signals_9_classes_400000_class_size'],
         ['data/','signal']]



for i in [4,5]: # running code for best dataset as yet and the signal dataset
    # -- amend 'names' for where you have them stored
    print('\nFor'+names[i][0]+names[i][1])
    file = pd.read_pickle(names[i][0]+names[i][1]+'.pkl')
    bins = binq2_filter(file)
    min_plot_all_bins(bins,names[i][0],names[i][1],show=False) # set show to true to view histograms












