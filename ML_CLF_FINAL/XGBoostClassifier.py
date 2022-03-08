# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 11:09:41 2022

@author: Vid
"""

import numpy as np 
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import *#auc, accuracy_score, confusion_matrix, mean_squared_error
import pickle
import time 
import graphviz
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

class XGBoostCLF():
    def __init__(self,X,y,test_size=0.3):
        """
        XGBoost Extreme Gradient booster classifier
        Inputs:
            - X: features
            - y: corresponding labels
            - test_size: fraction of data which will be used for testing
        """
        self.test_size = test_size
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(X, y,
                                                    test_size=test_size)
        self.X_test,self.X_val,self.y_test,self.y_val = train_test_split(
                self.X_test,self.y_test,test_size=0.3333333333)
        
        self.dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        self.dtest = xgb.DMatrix(self.X_test, label=self.y_test)
        self.dval = xgb.DMatrix(self.X_val,label=self.y_val)
        
        self.N_classes = len(set(self.dtest.get_label()))
        print("\nNUMBER OF CLASSES:",self.N_classes)
        print("Training dataset:")
        
    def train(self):
        """
        Trains and tests the model.
        Produces confusion matrix, overall accuracy and
        obtained signal accuracy.
        Returns array of predicted labels
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
    
    def test(self):
        prob_arr = self.clf.predict(self.dtest)
        #prob_arr_best = self.clf.predict(self.dtest,
        #                   iteration_range=(0, self.clf.best_iteration))
        y_pred = self.get_labels(prob_arr)
        acc = self.get_accuracy(y_pred,self.y_test)
        conf_matrix = confusion_matrix(self.y_test,y_pred)        

        print("\nTesting data of size:",len(self.y_test))
        print("Accuracy:",acc)
        print(conf_matrix)
    
        # Plotting decision tree and feature importance with graphviz ##
        # xgb.plot_tree(self.clf)
        xgb.plot_importance(self.clf)
       
    def train_test(self):
        """
        Trains and tests the model
        """
        self.train()
        self.test()
    
    def get_labels(self,prob_arr):
        """
        Given an array of probability for each instance,
        find the label - index where prob==maximum
        """
        labels = []
        for i, arr in enumerate(prob_arr):   
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
    
    def get_composition_of_classes(self,X_unknown):
        """
        Input data needs to be normalised
        """
        N = len(X_unknown)
        X_unknown = xgb.DMatrix(X_unknown)
        probability_arrays = self.clf.predict(X_unknown)
        y_pred = self.get_labels(probability_arrays)        
        y_predS = pd.Series(y_pred)
        counts = y_predS.value_counts()
        indices = counts.index
        
        composition = np.zeros(self.N_classes)
        for ind,nb in zip(indices,counts):
            composition[ind] = round(nb/sum(counts),3)
    
        return composition
    
    def get_provisional_signals(self,X_unknown,
                        N_classes,size_class,plot_probability_distrib=False):
        """
        Input: scaled unknown data,N_classes, size_class
        Stores the provisional signals in a .pkl file
        and return array of provisional signals
        """
        
        # Get probability arrays
        N = len(X_unknown)
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
        sigs_provis=X_unknown[0:0]
        sigcounter=0
        for i in tqdm(range(len(probability_arrays))):
            if np.argmax(probability_arrays[i])==0:
                sig = X_unknown.iloc[i]
                sigs_provis = sigs_provis.append(sig)
                sigcounter += 1
                
        print(f"{sigcounter} signals detected in total_dataset.pkl")
        print(f"{sigcounter/N*100}% of unknown data classified as signal")
        
        #filename="provisional_signals/XGBClf_signals_"+f"{N_classes}_classes_"+ \
        #        f"{size_class}_class_size"+".pkl"
                
        #sigs_provis.to_pickle(filename)
        return sigs_provis
    
    def get_provisional_nonsignals(self,X_unknown,
                                   N_classes,size_class):
        """
        Get events from unknown data which weren't
        classified as signals. Stores them in .pkl file.
        """
        # Get probability arrays
        N = len(X_unknown)
        X_unknown = xgb.DMatrix(X_unknown)
        probability_arrays = self.clf.predict(X_unknown)
        # Get (provisional) signals from total_dataset.pkl
        
        nonsigs_provis=X_unknown[0:0]
        nonsigcounter=0
        for i in tqdm(range(len(probability_arrays))):
            if np.argmax(probability_arrays[i])!=0:
                nonsig = X_unknown.iloc[i]
                nonsigs_provis = nonsigs_provis.append(nonsig)
                nonsigcounter += 1
                
        print(f"\n{nonsigcounter} non-signals detected in unknown data")
        print(f"{nonsigcounter/N*100}% of unknown data classified not a signal")
        
        #filename="provisional_signals/XGBClf_nonsignals_"+f"{N_classes}_classes_"+ \
        #        f"{size_class}_class_size"+".pkl"
                
        #nonsigs_provis.to_pickle(filename)
        return nonsigs_provis
            
    def get_provisional_indices(self,X_unknown,
                                N_classes,size_class,decision_boundary=0.5,
                                signal=False,save=False):
        """
        Get events from unknown data which were/weren't
        classified as signals (signal=True/False).
        Stores indices of these events in .pkl file.
        """
        # Get probability arrays
        N = len(X_unknown)
        X_unknown = xgb.DMatrix(X_unknown)
        probability_arrays = self.clf.predict(X_unknown)
        y_pred = self.get_labels(probability_arrays)
        
        
        if signal:
            indices = []
            for i,arr in enumerate(probability_arrays):
                if np.argmax(arr)==0:
                    max_prob = probability_arrays[i][0]
                    if max_prob>decision_boundary:
                        indices.append(i)
                if np.argmax(arr)==1:
                    max_prob = probability_arrays[i][1]
                    if max_prob>decision_boundary:
                        indices.append(i)
                
                
            print(f"{len(indices)} signals detected")
            print(f"{len(indices)/N*100}% of unknown data classified as signal")
            
            filename="provisional_signals/XGBClf_signal_indices_"+f"{N_classes}_classes_"+ \
                    f"{size_class}_class_size"+".txt"
            
            if save:
                with open(filename, "w") as file:
                    file.write(str(indices))
            return indices, probability_arrays
        else:
            indices = []
            for i in tqdm(range(len(probability_arrays))):
                if np.argmax(probability_arrays[i])!=0:
                    indices.append(i)
                    
                
            print(f"{len(indices)} non-signals detected in total_dataset.pkl")
            print(f"{len(indices)/N*100}% of unknown data classified not a signal")
            
            filename="provisional_signals/XGBClf_nonsignal_indices_"+f"{N_classes}_classes_"+ \
                    f"{size_class}_class_size"+".txt"
            if save:
                with open(filename, "w") as file:
                    file.write(str(indices))
                
            # Returns indices of signals/non-signals from X_unknown and
            # probability arrays for each event
            return indices, probability_arrays
      