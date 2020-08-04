# -*- coding: utf-8 -*-
"""
Spyder Editor

Author: Benjamin Terrill
Date: 15/11/2019
Student ID: 15622143
Machine Learning Assignment 1
"""

import numpy as np
import copy 
import pandas as pd
import matplotlib.pyplot as plt
"""
from operator import attrgetter
"""
print ("Hello World")


"""
Task 2
"""

def compute_euclidean_distance(centroids, df, colmap):
    #distance = np.sqrt((vec_1 - vec_2) **2 + (vec_1 - vec_2)  **2)
    for i in centroids.keys():
        df['distance_{}'.format(i)] = (
            np.sqrt(
                (df['x'] - centroids[i][0]) ** 2
                + (df['y'] - centroids[i][1]) ** 2
            )
        )
    print (df)
    
    #for i in centroids.keys():
    centroids_distance = ['distance_{}'.format(i) for i in centroids.keys()] 
    df ['closest'] = df.loc[:, centroids_distance].idxmin(axis = 1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_')))
    #df['colour'] = df['closest'].map(lambda x: colmap[x])
    print (df)
    return df
    
    
    
def initialise_centroids(x,y,k):
    #minX = min(dataset,key=attrgetter('x')).x  
    #minY = min(dataset,key=attrgetter('y')).y
    #maxX = max(dataset,key=attrgetter('x')).x
    #maxY = max(dataset,key=attrgetter('y')).y
    centroids = {}
    for i in range(k):
        centroids[str(i + 1)] = [np.random.uniform(x.min(), x.max()), np.random.uniform(y.min(), y.max())]
    return centroids   

"""
def kmeans(df, k, centroids):
    if k<1:
       return 
    changed = True
    while(changed):
        changed = False
        for i in centroids.keys():
            centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
            centroids[i][1] = np.mean(df[df['closest'] == i]['y'])  
            
    return centroids
"""     
           
def kmeans(df, centroids, k):
    for i in centroids.keys():
        #avg = np.mean(df[df['closest'] == i]['x'])
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])  
        #print("centroids", centroids)
    return centroids
    
    
def main():
    
    dataset = pd.read_csv('task2_dataset.csv')
    
    
    x = np.array(dataset.height.values)
    y = np.array(dataset.tail_length.values)
    
    df = pd.DataFrame({
            'x':(dataset.height.values),
            'y':(dataset.tail_length.values)
    })
    
    #print (df)
    
    k = 2
    
    centroids = initialise_centroids(x,y,k)
    print("Centroids: ")
    print(centroids)
    
    plt.scatter(dataset.height, dataset.tail_length, color = 'red', s = 4)
    plt.scatter(dataset.height, dataset.leg_length, color = 'black', s = 4)
    plt.xlabel('Height')
    plt.ylabel('Tail Length')
    
    colmap = {1: 'blue',2: 'orange', 3:'green'}
    
    for i in centroids.keys():
        plt.scatter(*centroids[i], s = 50)
    
        
        
    df = compute_euclidean_distance(centroids, df, colmap)
    print (df)
    plt.scatter(df['x'],df['y'],color = df['colour'], s = 4)
    
    #centroids_old = copy.deepcopy(centroids)
    #centroids_new = {} 
    #centroids = kmeans(df, centroids, k)
    
    #print ('test')
    #print (centroids_new)
    

main()
    















