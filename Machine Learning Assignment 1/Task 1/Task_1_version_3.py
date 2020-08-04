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
x = numpy.random.rand(100, 5)
numpy.random.shuffle(x)
training, test = x[:80,:], x[80:,:]
"""
def main():  
    
    dataset_train = pd.read_csv('task1_dataset.csv')
    dataset_test = pd.read_csv('task1_dataset.csv')
    #print(dataset_train)
    #print(dataset_test)
    #dataset = pd.read_csv('task1_dataset.csv')
    
    dataset_train_copy = dataset_train.copy()
    
    dataset_train2 = dataset_train_copy.sample(frac = 0.50, random_state = 0)
    #print("test train", dataset_train2)
    dataset_test2 = dataset_train_copy.drop(dataset_train2.index)
    #pprint("test test", dataset_test2)
    
    
    plt.clf()
    plt.scatter(dataset_train['x'], dataset_train['y'])
    
    train_x = dataset_train2['x'].as_matrix()
    train_x.sort (axis = 0)
    train_y = dataset_train2['y'].as_matrix()
    train_y.sort (axis = 0)
    
    test_x = dataset_test2['x'].as_matrix()
    test_x.sort (axis = 0)
    test_y = dataset_test2['y'].as_matrix()
    test_y.sort (axis = 0)

    #print("x train", train_x)
    #print("y train", train_y)
    
    #print("x test", test_x)
    #print("y test", test_y)

    
    X = np.column_stack((np.ones(train_x.shape), train_x))

    XX = X.transpose().dot(X)
    
    w = np.linalg.solve(XX, X.transpose().dot(train_y))

    Xtest = np.column_stack((np.ones(test_x.shape), test_x))

    ytest_predicted = Xtest.dot(w)
    print("x", test_x)
    print("y", ytest_predicted)
    plt.plot(test_x, ytest_predicted, 'g')
    #plt.plot(test_x, ytest_predicted, 'g')
    
    #degree = 5
    
    
    
    
    """
    t = np.polyfit(train_x, train_y, 3)
    print("t", t)
    f = np.poly1d(t)
    print("f", f)
    x_test = np.linspace(train_x[0], train_x[-1], 100)
    y_test = f(x_test)
    print("y_test", y_test)
    print("x_test", x_test)
    #plt.plot(x_test, y_test, 'orange')
    #plt.xlim([train_x[0]-1, train_x[-1] + 1])
    
    Degree_2 = pol_regression(train_x, train_y, test_x, test_y, 3) 
    f = np.poly1d(Degree_2)
    print("test", Degree_2)
    print("x", train_x)
    x_new = np.linspace(train_x[0], train_x[-1], 100)
    y_new = f(x_new)
    print("x_new", x_new)
    print("y_new", y_new)
    plt.plot(x_new, y_new, 'purple')
    plt.plot(x_test, y_test, 'orange')
    plt.xlim([train_x[0]-1, train_x[-1] + 1])
    """
    
    #Degree_0 = pol_regression(train_x, train_y, test_x, test_y, 0)   
    #plt.plot(test_x, Degree_0, 'orange')
    #print("Degree 0: ")
    #print(Degree_0)
    """
    Degree_1 = pol_regression(train_x, train_y, test_x, test_y, 1)  
    print("Degree 1", Degree_1)
    plt.plot(test_x, Degree_1, 'green')
    
    #Degree_2 = pol_regression(train_x, train_y, test_x, test_y, 2)  
    #plt.plot(test_x, Degree_2, 'orange')
    #print("test x", test_x)
    #print("Degree", Degree_2)
    
    Degree_3 = pol_regression(train_x, train_y, test_x, test_y, 3)  
    plt.plot(test_x, Degree_3, 'red')
    
    Degree_4 = pol_regression(train_x, train_y, test_x, test_y, 4)   
    plt.plot(test_x, Degree_4, 'blue')
    """
    """
    Degree_5 = pol_regression(train_x, train_y, test_x, test_y, 5)   
    plt.plot(test_x, Degree_5, 'purple')
    """
    #Degree_10 = pol_regression(train_x, train_y, test_x, test_y, 10)   
    #plt.plot(test_x, Degree_10, 'black')
    
    plt.ylim((-200, 50))
    plt.xlim((-5, 5))
    plt.legend(('Degree 1', 'Degree 2', 'Degree 3', 'Degree 5', 'Data points'), loc = 'lower right')
    
    
def pol_regression(train_x, train_y, test_x, test_y, degree):
    X = np.ones(train_x.shape)
    print("X")
    print(X)
    for i in range(1, degree + 1):
        X = np.column_stack((X, train_x ** i))
    print("X")
    print(X)
    
    XX = X.transpose().dot(X)
    print("XX")
    print(XX)
    
    w = np.linalg.solve(XX, X.transpose().dot(train_y))
    print("w")
    print(w)
    
    Xtest1 = np.ones(test_x.shape)
    for i in range(1, degree+1):
        Xtest1 = np.column_stack((Xtest1, test_x ** i))
    print("Xtest1: ")
    print(Xtest1)
    
    Ytest1 = Xtest1.dot(w)
    print("Ytest1")
    print(Ytest1)
    print (train_y)
    return Ytest1

main()

















