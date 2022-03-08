# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 11:02:01 2022

@author: Vid
"""
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split



def load_data(size_class,datasets,normalise=True):
    """
    Inputs:
        - size_class: number of events per class (maximum)
        - datasets: list of pandas DataFrames
        - N_classes: number of classes (files) from "datasets"
        - normalise: scale/normalise features or not
    Given data from different classes - produces shuffled X and y.
        """
    
    merged = datasets[0].iloc[0:1]
    keys = merged.keys()
    for i in range(len(datasets)):
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
    
    print("Total number of entries:",len(merged))
    print("Number of classes:",len(datasets))
    
    # Randomly shuffle the data
    merged = merged.sample(frac=1)
    
    X = merged[keys[:-1]]
    y = merged[keys[-1]]
    
    if normalise:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
    y = y.to_numpy()
    
    # Returns arrays of X==float64 and y==int64
    if normalise:
        return X,y,scaler
    else:
        return X.to_numpy(),y


def batch_data(X,y,batch_size=None,N_batches=None):
    """
    Separate datasets X and y into linked batches.
    Either:
        - of size batch_size
        - number of batches N_batches
    By default, the size of batches is chosen.
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
    
    if N_batches==None:
        return X_train,X_test,y_train,y_test
    
    else:
        if one_testset:
            return X_train_batches, X_test, y_train_batches, y_test
        else:
            # Make equal number of batches from testing data
            X_test_batches,y_test_batches = batch_data(X_test,y_test,N_batches=N_btchs)
            
            return X_train_batches, X_test_batches, y_train_batches, y_test_batches

