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
import operator
import pandas as pd
import matplotlib.pyplot as plt
"""
from operator import attrgetter
"""
print ("Hello World")


"""
Task 2
"""

class Point:
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cid = -1
    def dist(self, point2):
        return np.sqrt(np.power(self.x-point2.x,2)+np.power(self.y-point2.y,2))
    def compute_euclidean_distance(vec_1, vec_2):
        distance = np.sqrt((vec_1 - vec_2) **2 + (vec_1 - vec_2)  **2)
        return distance
    def __repr__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ")"
    def __str__(self):
        return "(" + self.x + ", " + self.y + ")"


def compute_euclidean_distance(p1,p2):
    return np.sqrt(np.power(p1.x-p2.x,2)+np.power(p1.y-p2.y,2))
    

def initialise_centroids(pointset, k):
    #minX = min(pointset,key=operator.attrgetter('x')).x  
    #minY = min(pointset,key=operator.attrgetter('y')).y
    #maxX = max(pointset,key=operator.attrgetter('x')).x
    #maxY = max(pointset,key=operator.attrgetter('y')).y
      
    centroids = []
    for i in range(k):
        centroids.append(Point(np.random.uniform(4.5, 8), np.random.uniform(1, 7)))
    return centroids

def getClusterAssignments(pointset, centroids):
    k = len(centroids)

    """make an empty array of arrays (a list of lists of points, each list of points are the points for a certain cluster)"""
    cluster_assigned = []
    for i in range(k):
        cluster_assigned.append([]);

    """for every point find its nearest centroid, then add to that cluster list"""
    for p in pointset:
        best = p.dist(centroids[0])
        #best = compute_euclidean_distance(p, centroids[i])
        bestId = 0
        for i in range(k):
            currentDistance = compute_euclidean_distance(p, centroids[i])
            if currentDistance <= best:
                best = currentDistance
                bestId = i
        cluster_assigned[bestId].append(p)
    return cluster_assigned


def kmeans(pointset, k):
    if k<1:
        return 
    changed = True
    centroids = initialise_centroids(pointset, k)
    print("Centroids: ",centroids)
    while(changed):
        changed = False
        """cluster assignment"""
        cluster_assigned = getClusterAssignments(pointset, centroids)
        
        for i in range(k):
            avg = Point(0,0)
            count = 1
            for p in cluster_assigned[i]:
                avg.x += p.x
                avg.y += p.y
                count+=1
            avg.x/=count
            avg.y/=count
            if compute_euclidean_distance(avg, centroids[i])>0:
                changed=True
            centroids[i]=avg
    return centroids, cluster_assigned

def main():
    dataset = pd.read_csv('task2_dataset.csv')
    k = 3
    pointset2 = [Point(2,3),Point(5,6),Point(2,9)]
    pointset = []
    for i in range (len(dataset.height-1)):
        #print("i", i)
        pointset.append(Point(dataset.height[i], dataset.tail_length[i]))
        
    #print("P2", pointset2)
    #print("p1", pointset)
    centroids, cluster_assigned = kmeans(pointset, k)
    plt.scatter(dataset.height, dataset.tail_length, color = 'orange', s = 4)
    plt.scatter(dataset.height, dataset.leg_length, color = 'blue', s = 4)
    plt.xlabel('Height')
    plt.ylabel('Tail Length')
    
    #print("centroids", centroids)
    #for i in range (k):
    #    plt.scatter(centroids[i],centroids[i] , s = 50)
    
    
    
    #print("Pointset: ", pointset)
    print("Final Centroids: " ,centroids)
    #print("Clustered Points: " , cluster_assigned)
    

if __name__ == "__main__":
    main()
