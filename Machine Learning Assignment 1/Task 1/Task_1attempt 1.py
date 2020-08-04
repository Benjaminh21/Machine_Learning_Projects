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
Task 1
"""


def main():
    
    dataset_train = pd.read_csv('task1_dataset.csv')
    dataset_test = pd.read_csv('task1_dataset.csv')
    print(dataset_train)
    
    """
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
    
    
    train_x = dataset_train['x'].as_matrix()
    train_x.sort (axis = 0)
    train_y = dataset_train['y'].as_matrix()
    train_y.sort (axis = 0)
    
    test_x = dataset_test['x'].as_matrix()
    test_x.sort (axis = 0)
    test_y = dataset_test['y'].as_matrix()
    test_y.sort (axis = 0)
    
    
    X = np.column_stack((np.ones(train_x.shape), train_x)) 
    XX = X.transpose().dot(X)
    w = np.linalg.solve(XX, X.transpose().dot(train_y))
    Xtest = np.column_stack((np.ones(test_x.shape), test_x))
    ytest_predicted = Xtest.dot(w)   
    
    print(w)
    plt.figure()
    #plt.plot(test_x,test_y, 'green')
    #plt.plot(test_x, ytest_predicted, 'red')
    plt.scatter(dataset_train['x'],dataset_train['y'], 'bo')
    plt.legend(('training points', 'prediction', 'Data set'), loc = 'lower right')
    
    """
    w2 = getWeightsForPolynomialFit(train_x1,train_y1,2)
    Xtest2 = getPolynomialDataMatrix(test_x1, 2)
    ytest2 = Xtest2.dot(w2)
    plt.plot(test_x1, ytest2, 'purple')
    """
    
    w1 = getWeightsForPolynomialFit(train_x,train_y,1)
    Xtest1 = getPolynomialDataMatrix(test_x, 1)
    ytest1 = Xtest1.dot(w1)
    plt.plot(test_x, ytest1, 'orange')
    
    w2 = getWeightsForPolynomialFit(train_x,train_y,2)
    Xtest2 = getPolynomialDataMatrix(test_x, 2)
    ytest2 = Xtest2.dot(w2)
    plt.plot(test_x, ytest2, 'purple')
    
    w3 = getWeightsForPolynomialFit(train_x,train_y,3)
    Xtest3 = getPolynomialDataMatrix(test_x, 3)
    ytest3 = Xtest3.dot(w3)
    plt.plot(test_x, ytest3, 'blue')
    
    w5 = getWeightsForPolynomialFit(train_x,train_y,5)
    Xtest5 = getPolynomialDataMatrix(test_x, 5)
    ytest5 = Xtest5.dot(w5)
    plt.plot(test_x, ytest5, 'green')
    
    plt.ylim((-200, 50))
    plt.xlim((-5, 5))
    




def pol_regression(features_train, y_train, degree):
    
    return w

def getPolynomialDataMatrix(x, degree):
    X = np.ones(x.shape)
    for i in range(1,degree + 1):
        X = np.column_stack((X, x ** i))
    return X
    
def getWeightsForPolynomialFit(x,y,degree):
    X = getPolynomialDataMatrix(x, degree)

    XX = X.transpose().dot(X)
    w = np.linalg.solve(XX, X.transpose().dot(y))

    return w

main()














