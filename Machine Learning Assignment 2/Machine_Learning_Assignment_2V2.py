# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 17:00:40 2019

@author: Ben Terrill
Date: 15/11/2019
Student ID: 15622143
Machine Learning Assignment 1

"""

#
import numpy as np
import numpy.linalg as lin

import pandas as pd
import matplotlib.pyplot as plt
#
#
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
#


def main():  
    
    df = pd.read_csv('NuclearPowerDataset.csv')
    #print('dataset', df)
    
    
    """Task 1"""
    #data_summary(df)
    
    """Task 3"""
    
    df_copy = df.copy()

    
    df_train = df_copy.sample(frac = 0.90) 
    #train dataset gets 90% of the data.

    df_test = df_copy.drop(df_train.index)
    # test dataset will get the remaining 30%.

    df.iloc[:, [0,9]]
    x = df_train.iloc[:, [5,6,7,8]]
    y = df_train['Status']
    
    x2 = df_test.iloc[:, [5,6,7,8]]
    y2 = df_test['Status']
    #print("Y: ", x)
    
    trainX = x
    trainY = y
    testX = x2
    testY = y2
    numberOfNeurons = 500
    numberOfTrees = 500
    epochs = 400
    
    ANN_Accuracy, Forest_Accuracy = ANN(df, trainX, trainY, testX, testY, numberOfNeurons, numberOfTrees, epochs)
    #print("Ann Accuracy: ", ANN_Accuracy)
    #print("Forest Accuracy: ", Forest_Accuracy)

    """Bonus Task 1"""
    """
    ANN_Epoch_Accuracy = []
    epochs = 1

    for i in range(0,9):
        epochs = i*100
        AANN_Accuracy, Forest_Accuracy = ANN(df, trainX, trainY, testX, testY, 500, 50, epochs)
        ANN_Epoch_Accuracy.append(ANN_Accuracy)
            
    print("Epoch:", ANN_Epoch_Accuracy)
    """
    
    """Bonus Task 2"""
    """
    Forest_Accuracy_list = []
    for i in range(1,15):
        numberOfTrees = i*100
        print(numberOfTrees)
        ANN_Accuracy, Forest_Accuracy = ANN(df, trainX, trainY, testX, testY, numberOfNeurons, numberOfTrees, epochs)
        Forest_Accuracy_list.append(Forest_Accuracy)
        
    print("Tree Accuracy", Forest_Accuracy_list)
    """
    """Task 4"""
    
    AccuracyANNSet1 = []
    AccuracyRFCSet1 = []
    AccuracyANNSet2 = []
    AccuracyRFCSet2 = []
    AccuracyANNSet3 = []
    AccuracyRFCSet3 = []
    
    count = 0
    
    df_copy_2 = df.copy()
    
    X = df_copy_2.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,12]]
    y = df_copy_2.iloc[:, [0]]

    
    kf = KFold(n_splits=10,shuffle=True, random_state = 1)
    kf.get_n_splits(X)
    
    epochs = 400
    
    for train_index, test_index in kf.split(X):
        trainX, testX = X.iloc[train_index], X.iloc[test_index]
        trainY, testY = y.iloc[train_index], y.iloc[test_index]
        #trainY=trainY.squeeze()
        #testY=testY.squeeze()
        
        
        ANNAccuracySet1, ForestAccuracySet1 = ANN(df_copy_2, trainX, trainY, testX, testY, 50, 20, epochs)
        AccuracyANNSet1.append(ANNAccuracySet1)
        AccuracyRFCSet1.append(ForestAccuracySet1)
        
        ANNAccuracySet2, ForestAccuracySet2 = ANN(df_copy_2, trainX, trainY, testX, testY, 500, 500, epochs)
        AccuracyANNSet2.append(ANNAccuracySet2)
        AccuracyRFCSet2.append(ForestAccuracySet2)
        
        ANNAccuracySet3, ForestAccuracySet3 = ANN(df_copy_2, trainX, trainY, testX, testY, 1000, 10000, epochs)
        AccuracyANNSet3.append(ANNAccuracySet3)
        AccuracyRFCSet3.append(ForestAccuracySet3)
        
        count = count+1
        print("Count: ", count)
        

    
    #Get the mean of each set

    #del AccuracyANNSet1[0]
    annMeanSet1 = np.mean(AccuracyANNSet1)
    forestMeanSet1 = np.mean(AccuracyRFCSet1)
    print(annMeanSet1)
    print("ANN Mean Set1", annMeanSet1)
    print("Forest Mean Set1", forestMeanSet1)
    
    annMeanSet2 = np.mean(AccuracyANNSet2)
    forestMeanSet2 = np.mean(AccuracyRFCSet2)

    print("ANN Mean Set2", annMeanSet2)
    print("Forest Mean Set2", forestMeanSet2)
    
    annMeanSet3 = np.mean(AccuracyANNSet3)
    forestMeanSet3 = np.mean(AccuracyRFCSet3)
    
    print("ANN Mean Set3", annMeanSet3)
    print("Forest Mean Set3", forestMeanSet3)
    

    AAcur = [annMeanSet1, annMeanSet2, annMeanSet3]
    FAcur = [forestMeanSet1, forestMeanSet2, forestMeanSet3]
    index = ['Set 1', 'Set 2', 'Set 3']
    
    df2 = pd.DataFrame({'ANN':AAcur, 'Forest':FAcur}, index=index)
    ax = df2.plot.bar(rot=0)

  


def ANN(df, trainX, trainY, testX, testY, no_neurons, no_trees, epochs):
    scaler = StandardScaler()
    scaler.fit(trainX)
    
    trainX = scaler.transform(trainX)
    testX = scaler.transform(testX)
    
    mlp = MLPClassifier(hidden_layer_sizes=(no_neurons, no_neurons), max_iter=400, activation='logistic')
    ANN = mlp.fit(trainX, trainY)   
    #print("mlp", ANN)
    
    
    predictions = ANN.predict(testX)
    #print("Confusion Matrix: ")
    #print(confusion_matrix(testY,predictions))
    #print(" ")
    #print("Classification Report: ")
    #print(classification_report(testY,predictions))
    #print(accuracy_score(testY, predictions))
    ANNAccuracy = accuracy_score(testY, predictions)
    
    """Task 3.5 Random Forests"""
    
    Classifier = RandomForestClassifier(n_estimators=no_trees, random_state=0, min_samples_leaf = 5)
    Classifier.fit(trainX, trainY)
    y_pred = Classifier.predict(testX)
    #print(y_pred)
    #print("Confusion Matrix Random Forest: ")
    #print(confusion_matrix(testY, y_pred))
    #print(" ")
    #print("Classification Report Random Forest: ")
    #print(classification_report(testY, y_pred))
    #print(accuracy_score(testY, y_pred))
    ForestAccuracy = accuracy_score(testY, y_pred)
    
    
    return ANNAccuracy, ForestAccuracy

def data_summary(df):
    #Task 1 - Data Summary#
    #Mean of each feature
    Means = df.mean()
    print("Mean of each feauture: ")
    print(Means)
    print(" ")
    #Sum of each feature
    Sums = df.sum()
    print("Sum of each feature: ")
    print(Sums)
    print(" ")
    #Min of each feature
    Mins = df.min()
    print("Min of each feature: ")
    print(Mins)
    print(" ")
    #Max of each feature
    Max = df.max()
    print("Max of each feature: ")
    print(Max)
    print(" ")
    #Median of each feature
    Medians = df.median()
    print("Median of each feature: ")
    print(Medians)
    print(" ")
    #Standard deviation of each feature
    std = df.std()
    print("Standard Deviation of each feature: ")
    print(std)
    print(" ")
    #Variance of each feature
    var = df.var()
    print("Variance of each feature: ")
    print(var)
    print(" ")
    #Count of each feature
    Count = df.count()
    print("Count: ")
    print(Count)
    print(" ")
    
    #Check for missing data
    df.isnull()
    #NO MISSING VALUES FOUND#
    #Find number of features
    print("Number of features: ", len(df.columns))
    #There are 13 features#
    
    #Total data in dataset
    #TotalCount = len(df.columns) * len(df.rows)
    TotalCount = df.shape
    print("Total size of dataset", TotalCount)
    
    #Plots
    STA1 = df.iloc[:, [0,9]]
    
    #Box Plot
    bp = STA1.boxplot(column='Vibration_sensor_1', by='Status')
    plt.show()

    STA2 = df.groupby("Status")["Vibration_sensor_2"].plot.kde()

    
    return 0

    
main()  
    
    
    
    
    
    
    
    
    
    
    
    