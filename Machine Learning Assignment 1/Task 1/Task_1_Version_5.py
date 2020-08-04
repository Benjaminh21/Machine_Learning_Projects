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
import math
import pandas as pd
import matplotlib.pyplot as plt
from operator import attrgetter

print ("Hello World")


"""
Task 1
x = numpy.random.rand(100, 5)
numpy.random.shuffle(x)
training, test = x[:80,:], x[80:,:]
"""
def main():  
    
    dataset_train = pd.read_csv('task1_dataset.csv')
    #dataset_test = pd.read_csv('tast1_dataset.csv')
    #print(dataset_train)
    #print(dataset_test)
    
    dataset_train_copy = dataset_train.copy()
    
    dataset_train2 = dataset_train_copy.sample(frac = 0.70, random_state = 0) 
    # train dataset gets 70% of the data.
    print("test train", dataset_train2)
    dataset_test2 = dataset_train_copy.drop(dataset_train2.index)
    # test dataset will get the remaining 30%.
    print("test test", dataset_test2)
    
    plt.clf()
    plt.scatter(dataset_train2['x'], dataset_train2['y'])
    # Plot data on graph
    
    train_x = dataset_train2['x'].as_matrix()
    train_x.sort (axis = 0)
    train_y = dataset_train2['y'].as_matrix()
    train_y.sort (axis = 0)
    #assign variable for each x and y
    test_x = dataset_test2['x'].as_matrix()
    test_x.sort (axis = 0)
    test_y = dataset_test2['y'].as_matrix()
    test_y.sort (axis = 0)

 
   
    x = np.arange(-5, 5, 0.1) 
    print("Matrix: " , x)
    print("Matrix-x: " , test_x)
    #This is used to plot along the entire x axis.
    #Stops the graph from plotting only at training X asix points.
    
    avg = np.average(train_y) # Get average value of y
    print("Average: ", avg)
    y = []
    for i in range(len(x)):
        y.append(avg) # creating a y value for degree 0  
    plt.plot(x, y, 'blue')
    

    y_value1= pol_regression(train_x, train_y, x, test_y, 1)  
    plt.plot(x, y_value1)
    #plot the line for degree 1 

    y_value3= pol_regression(train_x, train_y, x, test_y, 3)  
    plt.plot(x, y_value3)

    y_value5= pol_regression(train_x, train_y, x, test_y, 5)   
    plt.plot(x, y_value5)
    
    y_value10 = pol_regression(train_x, train_y, x, test_y, 10)   
    plt.plot(x, y_value10)

    plt.ylim((-200, 50))
    # limit the graph to -200 and 50 along y axis
    plt.xlim((-5, 5))
    # limit graph to -5,5 along the x axis
    plt.legend((' $x^{0}$', ' $x^{1}$', ' $x^{3}$', ' $x^{5}$',' $x^{10}$', 'Data points'), loc = 'lower right')
    
    rmsetrain = np.zeros((14,1))
    rmsetest = np.zeros((14,1))

    for i in range(1, 14):
            rmsetrain[i - 1], rmsetest[i - 1] = eval_pol_regression(y_value1, train_x, train_y,test_x, test_y, i)
            
    print("rmsetrain: ")
    print(rmsetrain)
    print("rmsetest: ")
    print(rmsetest)     
    
    plt.figure();
    plt.semilogy(range(-5,9), rmsetrain)
    plt.semilogy(range(-5,9), rmsetest)
    plt.legend(('RMSE Train', 'RMSE Test'), loc = 'lower right')
        
    
    
    
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
    # Get the weights for polynomial fit. 
    #print("w: ", w )
    
    Xtest1 = np.ones(test_x.shape)
    for i in range(1, degree+1):
        Xtest1 = np.column_stack((Xtest1, test_x ** i))
    #print("Xtest1: ", Xtest1)
    
    parameters = Xtest1.dot(w)
    #print("Ytest1: ". Ytest1)
    return parameters


def eval_pol_regression(parameters, trainx, trainy, testx, testy, degree):
    #SSEtrain = np.zeros((11,1))
    #SSEtest = np.zeros((11,1))
    
    trainX = np.ones(trainx.shape)
    for i in range(1, degree + 1):
        trainX = np.column_stack((trainX, trainx ** i))
    
    testX = np.ones(testy.shape)
    for i in range(1, degree + 1):
        testX = np.column_stack((testX, testx ** i))
    

    X = np.ones(trainx.shape)
    for i in range(1, degree + 1):
        X = np.column_stack((X, trainx ** i))
    
    XX = X.transpose().dot(X)
    w = np.linalg.solve(XX, X.transpose().dot(trainy))
    #for i in range (1, 10):   
    SSEtrain = np.mean((trainX.dot(w) - trainy)**2)
    SSEtest = np.mean((testX.dot(w) - testy)**2)
    rmseTrain = math.sqrt(SSEtrain)
    rmseTest = math.sqrt(SSEtest)
    return rmseTrain, rmseTest


main()