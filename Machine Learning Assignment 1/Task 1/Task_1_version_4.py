# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 14:08:55 2019

@author: Ben Terrill
"""

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
    #dataset_test = pd.read_csv('tast1_dataset.csv')
    #print(dataset_train)
    #print(dataset_test)
    
    dataset_train_copy = dataset_train.copy()
    
    dataset_train2 = dataset_train_copy.sample(frac = 0.70, random_state = 0)
    print("test train", dataset_train2)
    dataset_test2 = dataset_train_copy.drop(dataset_train2.index)
    print("test test", dataset_test2)
    
    plt.clf()
    plt.scatter(dataset_train2['x'], dataset_train2['y'])
    
    train_x2 = dataset_train2['x'].as_matrix()
    train_x2.sort (axis = 0)
    train_y2 = dataset_train2['y'].as_matrix()
    train_y2.sort (axis = 0)
    
    test_x2 = dataset_test2['x'].as_matrix()
    test_x2.sort (axis = 0)
    test_y2 = dataset_test2['y'].as_matrix()
    test_y2.sort (axis = 0)

    
    plt.clf()
    plt.scatter(dataset_train['x'], dataset_train['y'])
    
    train_x = dataset_train['x'].as_matrix()
    train_x.sort (axis = 0)
    train_y = dataset_train['y'].as_matrix()
    train_y.sort (axis = 0)
    
    test_x = dataset_train['x'].as_matrix()
    test_x.sort (axis = 0)
    test_y = dataset_train['y'].as_matrix()
    test_y.sort (axis = 0)
    
   
    x = np.arange(-5, 5, 0.1) 
    print("Matrix: " , x)
    print("Matrix-x: " , test_x)
    
    avg = np.average(train_y) # Get average value of y
    print("Average: ", avg)
    y = []
    for i in range(len(x)):
        y.append(avg) # creating a y value for degree 0  
    plt.plot(x, y, 'blue')
    
        
  
    Ytest = pol_regression(train_x, train_y, x, test_y, 1)  
    #print("Degree 1", Degree_1)
    plt.plot(x, Ytest)
    
    Ytest = pol_regression(train_x, train_y, x, test_y, 3)  
    plt.plot(x, Ytest)

    Ytest = pol_regression(train_x, train_y, x, test_y, 5)   
    plt.plot(x, Ytest)
    
    Ytest = pol_regression(train_x, train_y, x, test_y, 10)   
    plt.plot(x, Ytest)

    plt.ylim((-200, 50))
    plt.xlim((-5, 5))
    plt.legend((' $x^{0}$', ' $x^{1}$', ' $x^{3}$', ' $x^{5}$',' $x^{10}$', 'Data points'), loc = 'lower right')
    
    for i in range (1, 10):
        rmse = eval_pol_regression(Ytest, train_x, train_y, i)
        print("rmse: ", rmse)
    
    
def pol_regression(train_x, train_y, test_x, test_y, degree):
    #print("train_x pre steps:", train_x)
    X = np.ones(train_x.shape)
    #print("np ones of train_x: ", X)
    for i in range(1, degree + 1):
        X = np.column_stack((X, train_x ** i))
    #print("np column stack: ", X)
    
    XX = X.transpose().dot(X)
    #print("XX", XX)
    
    w = np.linalg.solve(XX, X.transpose().dot(train_y))
    #print("w: ", w )
    
    Xtest1 = np.ones(test_x.shape)
    for i in range(1, degree+1):
        Xtest1 = np.column_stack((Xtest1, test_x ** i))
    #print("Xtest1: ", Xtest1)
    
    Ytest1 = Xtest1.dot(w)
    #print("Ytest1: ". Ytest1)
    return Ytest1

def eval_pol_regression(paremeters, x, y, degree):
    rmse = np.square(np.subtract(x, y)).mean()
    #rmse = 
    return rmse

main()