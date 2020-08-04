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
import math
import pandas as pd
import matplotlib.pyplot as plt
from operator import attrgetter
import seaborn as sns
import statistics
#


def main():  
    
    df = pd.read_csv('NuclearPowerDataset.csv')
    #print('dataset', df)
    data_summary(df)

    
    

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
    STA2 = df.iloc[0:498, [10]]
    STA3 = df.iloc[499:995, [10]]
    
    #Box Plot
    bp = STA1.boxplot(column='Vibration_sensor_1', by='Status')
    plt.show()
    #Density Plot
    #ax = STA2.plot.kde()
    #ax = STA3.plot.kde()
    STA4 = df.groupby("Status")["Vibration_sensor_2"].plot.kde()
    
    return 0

    
main()  
    
    
    
    
    
    
    
    
    
    
    
    