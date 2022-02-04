# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 12:41:23 2022

@author: Vid
"""

"""
Created on Sun Jan 30 20:14:53 2022

@author: Vid

Extreme Gradient Boosting Classifier.

The class contains methods for:
    - training/testing (simultaneously)
    - predicting unknown samples to obtain
      probability arrays for each event.
      This file also outputs provisional signals in a .pkl file
      given some dataset of unknown events
      
All datasets are to be situated in data folder in the same directory as this
.py file.

"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from copy import * 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics 
import xgboost as xgb
from tqdm import tqdm
from scipy.stats import uniform, randint
from sklearn.metrics import *#auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
import pickle
import time 
import graphviz

#%% Figures sytle
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
#Options
params = {'text.usetex' : True,
          'font.size' : 5,
          'font.family' : 'lmodern',
          #'text.latex.unicode': True,
          'axes.labelsize':18,
          'legend.fontsize': 15,
          'xtick.labelsize': 12,
          'ytick.labelsize': 5,
          #'figure.figsize': [7.5,7.5/1.2],
                     }
plt.rcParams.update(params)

#%% Loading files - filenames.
"""
To read the files without modification of the code, they must be all stored
in a folder "data" where data and this python file must be located in the same 
directory.
"""

### Unkown data
# Total dataset - unknown data to analyse and extract signal out of
total_dataset = pd.read_pickle('data/total_dataset.pkl')

### Known - labeled data
# The signal decay, simulated as per the Standard Model
sig = pd.read_pickle('data/signal.pkl')

# B0 ---> J/psi K^*0,     j/psi --> mu mu
jpsi = pd.read_pickle('data/jpsi.pkl')
# B0 ---> psi(2S) K^*0,   psi(2S) --> mu mu
psi2S = pd.read_pickle("data/psi2S.pkl")
# B0 ---> J/psi K^+0 with muon reconstructed as kaon and kaon as muon
jpsi_mu_k_swap = pd.read_pickle("data/jpsi_mu_k_swap.pkl")
# B0 ---> J/psi K^+0 with muon reconstructed as pion and pion as muon
jpsi_mu_pi_swap = pd.read_pickle("data/jpsi_mu_pi_swap.pkl")
# B0 ---> J/psi K^+0 with kaon reconstructed as pion and pion as kaon
k_pi_swap = pd.read_pickle("data/k_pi_swap.pkl")

# B_S^0 --->        \phi mu mu,      \phi --> KK and 1 K reconstructed as pion
phimumu = pd.read_pickle("data/phimumu.pkl")
# Lambda_b^0 --->   pK\mu\mu        with p reconstructed as K and K as pi 
pKmumu_piTok_kTop = pd.read_pickle("data/pKmumu_piTok_kTop.pkl")
# Lambda_b^0 --->   pK\mu\mu        with p reconstructed as pion
pKmumu_piTop = pd.read_pickle("data/pKmumu_piTop.pkl")





# Simulation which is flat in three angular variables and q^2
acceptance = pd.read_pickle("data/acceptance_mc.pkl")

# All files in a list for easier accessibility
datasets = [total_dataset,  #0
            sig,jpsi,psi2S, #1,2,3
            jpsi_mu_k_swap,jpsi_mu_pi_swap,k_pi_swap, #4,5,6
            phimumu,pKmumu_piTok_kTop,pKmumu_piTop, #7,8,9
            acceptance] #10



#%% Data clean-up - always use files in list "datasets"

keys = total_dataset.keys()
datasets = [sig,#0
            jpsi,psi2S, #1,2
            jpsi_mu_k_swap,jpsi_mu_pi_swap,k_pi_swap, #3,4,5
            phimumu,pKmumu_piTok_kTop,pKmumu_piTop, #6,7,8
            acceptance,total_dataset] #9,10


dataset_names = ["Signal",
                 r"J/$\psi$",r"$\psi$(2S)",
                 r"J/$\psi$; $\mu$, $K$ swapped",
                 r"J/$\psi$; $\mu$, $\pi$ swapped",
                 r"J/$\psi$; $K$, $\pi$ swapped",
                 r"$\phi \mu \mu$",
                 r"$p K \mu \mu$; K, $\pi$ swapped",
                 r"$p K \mu \mu$; p, $\pi$ swapped",
                 "flat acceptance",
                 "Total dataset"]

dataset_labels = [i for i in range(len(datasets))]
#dataset_labels = [0,1,1,1,1,1,1,1,1,1,1]
for i in range(len(datasets)):
    datasets[i] = datasets[i].drop(columns=["year","B0_ID"])
    datasets[i]["file"] = dataset_labels[i]

keys = datasets[i].keys()

#%% DataLoading

def load_data(size_class,normalise=True):
    """
    Inputs:
        - size_class: number of events per class (maximum)
        - N_classes: number of classes (files) from "datasets"
        - normalise: scale/normalise features or not
    Given data from different classes - produces shuffled X and y.
        """

    merged = datasets[0].iloc[0:1]
    for i in range(9):
        if i==0:
            data = datasets[i]
            file = data.iloc[0:size_class]
            merged = merged.append(file)
        else:
            data = datasets[i]
            if len(data)<size_class:
                file = data
                merged= merged.append(file)
            else:
                file = data.iloc[0:size_class]
                merged = merged.append(file)
    
    print("Number of entries:",len(merged))
    
    # Randomly shuffle the data
    merged = merged.sample(frac=1)
    
    X = merged[keys[:-1]]
    y = merged[keys[-1]]
    
    if normalise:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
    return X,y


def batch_data(X,y,batch_size=None,N_batches=None):
    """
    Separate datasets X and y into batches.
    Either:
        - of size batch_size
        - number of batches N_batches
    By default, choose the size of batches.
    """
    N=len(y)
    # Specifying number of batches
    if N_batches != None:
        batch_size = int(N/N_batches)
        X_batches = []
        y_batches = []
        i=0
        while i<N:
            Xbatch = X[i:i+batch_size]
            ybatch = y[i:i+batch_size]
            X_batches.append(Xbatch)
            y_batches.append(ybatch)
            i += batch_size
        return X_batches, y_batches
    
    # Specifying the size of batches
    else:
        X_batches = []
        y_batches = []
        i=0
        while i<N:
            Xbatch = X[i:i+batch_size]
            ybatch = y[i:i+batch_size]
            X_batches.append(Xbatch)
            y_batches.append(ybatch)
            i += batch_size
        return X_batches, y_batches

def get_batched_traintest_data(X,y,test_size,batch_size=None,N_batches=None,
                               one_testset=False):
    """
    Separate dataset into training and testing data.
    Create equal number of batches for both.
    """
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=test_size)
    # Batch train data
    X_train_batches,y_train_batches = batch_data(X_train,y_train,
                                                 batch_size,N_batches)
    N_btchs = len(X_train_batches)
    
    if one_testset:
        return X_train_batches, X_test, y_train_batches, y_test
    else:
        # Make equal number of batches from testing data
        X_test_batches,y_test_batches = batch_data(X_test,y_test,N_batches=N_btchs)
        
        return X_train_batches, X_test_batches, y_train_batches, y_test_batches

size_class = 50000
N_classes = 9
X,y = load_data(size_class=size_class)
keysX = keys[:-1]
#%% Class-function defn's: Extreme Gradient Boosting


class XGBoost_sklearn_CLF():
    """
    XGBoost Extreme Gradient booster classifier
    Inputs:
        - X: features
        - y: corresponding labels
        - test_size: fraction of data which will be used for testing
    """
    def __init__(self,X,y,test_size):
        
        self.test_size = test_size
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(X, y,
                                                    test_size=test_size)
        self.X_test,self.X_val,self.y_test,self.y_val = train_test_split(
                self.X_test,self.y_test,test_size=0.4)
        
        self.dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        self.dtest = xgb.DMatrix(self.X_test, label=self.y_test)
        self.dval = xgb.DMatrix(self.X_val,label=self.y_val)
        
        self.N_classes = len(set(self.dtest.get_label()))
        print("NUMBER OF CLASSES:",self.N_classes)
        
    def train_test(self):
        """
        Trains and tests the model.
        Produces confusion matrix, overall accuracy and
        obtained signal accuracy.
        """
        print("\nExtreme Gradient Boosting Classifier (XGBoost)")
        start = time.time()
        params = {'max_depth': 6, 'eta': 1}
        params['objective'] = 'multi:softprob'
        params['eval_metric'] = 'mlogloss'
        params['num_class'] = self.N_classes         
            
        evallist = [(self.dval, 'eval'), (self.dtest, 'train')]

        self.clf = xgb.train(params=params,
                             dtrain=self.dtrain,
                             #early_stopping_rounds=10,
                             evals = evallist)

        # dump model with feature map
        self.clf.dump_model('dump.raw.txt')
        end = time.time()
        print("Trained for:",end-start,"seconds")
        
        prob_arr = self.clf.predict(self.dtest)
        prob_arr_best = self.clf.predict(self.dtest,
                            iteration_range=(0, self.clf.best_iteration))

        y_pred = self.get_labels(prob_arr)
        acc = self.get_accuracy(y_pred,self.y_test)
        conf_matrix = confusion_matrix(self.y_test,y_pred)        

        print("\nTesting data of size:",len(self.y_test))
        print("Accuracy:",acc)
        print(conf_matrix)
    
        #xgb.plot_tree(self.clf)
        #xgb.plot_importance(self.clf)
        
        return y_pred

    
    ###################################################################
                        ### Other methods ### 
    ###################################################################
    
    def get_labels(self,prob_arr):
        """
        Given an array of probability for each instance,
        find the label - index where prob==maximum
        """
        labels = []
        for i,arr in enumerate(prob_arr):
                label = np.argmax(arr)
                labels.append(label)
        return labels
    
    def get_accuracy(self,predictions,labels):
        acc = 0
        for p,l in zip(predictions,labels):
            if p==l:
                acc +=1                
        acc = acc/len(predictions)
        return acc
    
    def get_provisional_signals(self,X_unknown,
                        N_classes,size_class,plot_probability_distrib=False):
        """
        Stores the provisional signals in a .pkl file
        and return array of provisional signals
        """
        
        # Get probability arrays
        X_unknown = xgb.DMatrix(X_unknown)
        probability_arrays = self.clf.predict_proba(X_unknown)
        
        if plot_probability_distrib:
            # Plot the distribution of probability values for the classified signals
            plt.figure("signal_probability_distribution")
            plt.hist(probability_arrays,bins=30,histtype="step")
            plt.ylabel("N")
            plt.xlabel("Probability of being a signal")
            plt.show()
    
        # Get (provisional) signals from total_dataset.pkl
        sigs_provis=X_unknown_unscaled[0:0]
        sigcounter=0
        for i in tqdm(range(len(probability_arrays))):
            if np.argmax(probability_arrays[i])==0:
                sig = X_unknown_unscaled.iloc[i]
                sigs_provis = sigs_provis.append(sig)
                sigcounter += 1
                
        print(f"{sigcounter} signals detected in total_dataset.pkl")
        print(f"{sigcounter/len(X_unknown_unscaled)*100}% of unknown data classified as signal")
        
        filename="provisional_signals/XGBClf_signals_"+f"{N_classes}_classes_"+ \
                f"{size_class}_class_size"+".pkl"
                
        sigs_provis.to_pickle(filename)
        return sigs_provis
    
    def get_provisional_nonsignals(self,X_unknown,
                                   N_classes,size_class):
        """
        Get events from unknown data which weren't
        classified as signals. Stores them in .pkl file.
        """
        # Get probability arrays
        X_unknown = xgb.DMatrix(X_unknown)
        probability_arrays = self.clf.predict(X_unknown)
        # Get (provisional) signals from total_dataset.pkl
        
        nonsigs_provis=X_unknown_unscaled[0:0]
        nonsigcounter=0
        for i in tqdm(range(len(probability_arrays))):
            if np.argmax(probability_arrays[i])!=0:
                nonsig = X_unknown_unscaled.iloc[i]
                nonsigs_provis = nonsigs_provis.append(nonsig)
                nonsigcounter += 1
                
        print(f"{nonsigcounter} non-signals detected in total_dataset.pkl")
        print(f"{nonsigcounter/len(X_unknown_unscaled)*100}% of unknown data classified not a signal")
        
        filename="provisional_signals/XGBClf_nonsignals_"+f"{N_classes}_classes_"+ \
                f"{size_class}_class_size"+".pkl"
                
        nonsigs_provis.to_pickle(filename)
        return nonsigs_provis
    
    def get_provisional_indices(self,X_unknown,
                                N_classes,size_class,signal=False):
        """
        Get events from unknown data which weren't
        classified as signals. Stores them in .pkl file.
        """
        # Get probability arrays
        X_unknown = xgb.DMatrix(X_unknown)
        probability_arrays = self.clf.predict(X_unknown)
        
        if signal:
            indices = []
            for i in tqdm(range(len(probability_arrays))):
                if np.argmax(probability_arrays[i])==0:
                    indices.append(i)
                
                
            print(f"{len(indices)} signals detected in total_dataset.pkl")
            print(f"{len(indices)/len(X_unknown_unscaled)*100}% of unknown data classified as signal")
            
            filename="provisional_signals/XGBClf_signal_indices_"+f"{N_classes}_classes_"+ \
                    f"{size_class}_class_size"+".txt"
            
            with open(filename, "w") as file:
                file.write(str(indices))
            return indices, probability_arrays
        else:
            indices = []
            for i in tqdm(range(len(probability_arrays))):
                if np.argmax(probability_arrays[i])!=0:
                    indices.append(i)
                    
                
            print(f"{len(indices)} non-signals detected in total_dataset.pkl")
            print(f"{len(indices)/len(X_unknown_unscaled)*100}% of unknown data classified not a signal")
            
            filename="provisional_signals/XGBClf_nonsignal_indices_"+f"{N_classes}_classes_"+ \
                    f"{size_class}_class_size"+".txt"
            
            with open(filename, "w") as file:
                file.write(str(indices))
            return indices, probability_arrays
        
    
##%% Code execution

plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
#Options
params = {'text.usetex' : True,
          'font.size' : 5,
          'font.family' : 'lmodern',
          #'text.latex.unicode': True,
          'axes.labelsize':18,
          'legend.fontsize': 15,
          'xtick.labelsize': 12,
          'ytick.labelsize': 5,
          #'figure.figsize': [7.5,7.5/1.2],
                     }
plt.rcParams.update(params)

model = XGBoost_sklearn_CLF(X,y,test_size=0.25)

# Predicted labels
ypred = model.train_test() 

##%%

# predict labels on unknown data
data_unknown = datasets[-1]
X_unknown_unscaled = data_unknown[keys[:-1]]

scaler = StandardScaler()
scaler.fit(X_unknown_unscaled)
X_unknown = scaler.transform(X_unknown_unscaled)


#nonsigs_provis = model.get_provisional_nonsignals(X_unknown_unscaled,
#                                             N_classes,size_class)

# =============================================================================
# 
# sigs_provis = model.get_provisional_signals(X_unknown_unscaled,
#                                       N_classes,size_class)
# 
# nonsignalindices = model.get_provisional_indices(X_unknown_unscaled,
#                                   N_classes,size_class,signal=False)
# 
# =============================================================================
signalindices,prob_arr = model.get_provisional_indices(X_unknown,
                                  N_classes,size_class,signal=True)







