# -*- coding: utf-8 -*-
"""
Spyder Editor

Author: Benjamin Terrill
Date: 15/11/2019
Student ID: 15622143
Machine Learning Assignment 1
"""

import numpy as np
import numpy.linalg as lin

import pandas as pd
import matplotlib.pyplot as plt
from operator import attrgetter

print ("Hello World")

"""
Stuff for later

        df = pd.DataFrame(np.random.randn(100, 2))
        msk = np.random.rand(len(df)) < 0.7
        train = df[msk]
        test = df[~msk]
        print (len(test))
        print (len(train))
        
        train_x1 = train['x'].as_matrix()
        train_x1.sort(axis = 0)
        train_y1 = train['y'].as_matrix()
        train_y1.sort (axis = 0)
        
        test_x1 = test['x'].as_matrix()
        test_x1.sort (axis = 0)
        test_y1 = test['y'].as_matrix()
        test_y1.sort (axis = 0)
        
"""
"""
Task 1
"""


def main():
    
    dataset_train = pd.read_csv('task1_dataset.csv')
    dataset_test = pd.read_csv('task1_dataset.csv')
    print(dataset_train)
    """
    plt.figure()
    plt.scatter(dataset_train['x'], dataset_train['y'])
    
    degree = 3
    X = np.ones(dataset_train['x'].shape)
    for i in range(1,degree + 1):
        X = np.column_stack((X, dataset_test['x'] ** i))
       
    XX = X.transpose().dot(X)
    
    w1 = np.linalg.solve(XX, X.transpose().dot(dataset_train['y']))
    X2 = np.ones(dataset_test['x'].shape)
    for i in range(1,degree+1):
        X2 = np.column_stack((X2, dataset_test['x']**i))
        
    ytest1 = X2.dot(w1)
    print("X")
    print(X2)
    print ("y")
    print(ytest1)
    #plt.scatter(X2,ytest1, 'orange')
    """
   
    
    plt.figure()
    plt.scatter(dataset_train['x'], dataset_train['y'])
    
    train_x = dataset_train['x'].as_matrix()
    train_x.sort (axis = 0)
    train_y = dataset_train['y'].as_matrix()
    train_y.sort (axis = 0)
    
    test_x = dataset_test['x'].as_matrix()
    test_x.sort (axis = 0)
    test_y = dataset_test['y'].as_matrix()
    test_y.sort (axis = 0)
    
    print ("matrix")
    print(train_x)
    
    
    degree = 3
    X = np.ones(train_x.shape)
    for i in range(1,degree + 1):
        X = np.column_stack((X, train_x ** i))
       
    XX = X.transpose().dot(X)
    
    w1 = np.linalg.solve(XX, X.transpose().dot(train_y))
    X2 = np.ones(train_x.shape)
    for i in range(1,degree+1):
        X2 = np.column_stack((X2, test_x**i))
        
    ytest1 = X2.dot(w1)
    print("X")
    print(X2)
    print ("y")
    print(ytest1)
    plt.scatter(X2,ytest1, 'orange')

main()
        
        
        
        
        
        
        
        
        